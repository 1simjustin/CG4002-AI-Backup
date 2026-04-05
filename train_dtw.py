"""
Plan B: DTW + Classical ML for Action Quality Assessment.

Standalone script — no imports from the PyTorch codebase.
Computes per-limb Dynamic Time Warping distances between shifu (expert)
and student IMU recordings, then trains scikit-learn regressors to map
the 4 DTW distances to a 0.0–1.0 quality score.

Usage:
    python train_dtw.py
    python train_dtw.py --negative_pair_prob 0.15 --val_split 0.2
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ── Column definitions (12 channels per limb, 48 total) ─────────────────

LIMB_COLUMNS = {
    "left_arm": [
        "left_arm_upper_arm_ax", "left_arm_upper_arm_ay", "left_arm_upper_arm_az",
        "left_arm_upper_arm_gx", "left_arm_upper_arm_gy", "left_arm_upper_arm_gz",
        "left_arm_forearm_ax", "left_arm_forearm_ay", "left_arm_forearm_az",
        "left_arm_forearm_gx", "left_arm_forearm_gy", "left_arm_forearm_gz",
    ],
    "right_arm": [
        "right_arm_upper_arm_ax", "right_arm_upper_arm_ay", "right_arm_upper_arm_az",
        "right_arm_upper_arm_gx", "right_arm_upper_arm_gy", "right_arm_upper_arm_gz",
        "right_arm_forearm_ax", "right_arm_forearm_ay", "right_arm_forearm_az",
        "right_arm_forearm_gx", "right_arm_forearm_gy", "right_arm_forearm_gz",
    ],
    "left_leg": [
        "left_leg_thigh_ax", "left_leg_thigh_ay", "left_leg_thigh_az",
        "left_leg_thigh_gx", "left_leg_thigh_gy", "left_leg_thigh_gz",
        "left_leg_shin_ax", "left_leg_shin_ay", "left_leg_shin_az",
        "left_leg_shin_gx", "left_leg_shin_gy", "left_leg_shin_gz",
    ],
    "right_leg": [
        "right_leg_thigh_ax", "right_leg_thigh_ay", "right_leg_thigh_az",
        "right_leg_thigh_gx", "right_leg_thigh_gy", "right_leg_thigh_gz",
        "right_leg_shin_ax", "right_leg_shin_ay", "right_leg_shin_az",
        "right_leg_shin_gx", "right_leg_shin_gy", "right_leg_shin_gz",
    ],
}

LIMB_NAMES = list(LIMB_COLUMNS.keys())

ALL_COLUMNS: List[str] = []
for _cols in LIMB_COLUMNS.values():
    ALL_COLUMNS.extend(_cols)

LIMB_SLICES = {
    "left_arm":  slice(0, 12),
    "right_arm": slice(12, 24),
    "left_leg":  slice(24, 36),
    "right_leg": slice(36, 48),
}


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DTW + ML for AQA (Plan B)")
    p.add_argument("--recordings_dir", type=str,
                   default=os.path.join(os.path.dirname(__file__), "recordings"))
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--negative_pair_prob", type=float, default=0.15)
    p.add_argument("--pseudo_shifu_ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(os.path.dirname(__file__), "checkpoints"))
    p.add_argument("--fastdtw_radius", type=int, default=1)
    return p.parse_args()


# ── Data loading & normalisation ─────────────────────────────────────────

def load_labels(recordings_dir: str) -> dict:
    path = os.path.join(recordings_dir, "labels.json")
    with open(path) as f:
        return json.load(f)


def compute_global_stats(
    recordings_dir: str, labels: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean & std (after per-sample mean subtraction)."""
    running_sum = np.zeros(48, dtype=np.float64)
    running_sq = np.zeros(48, dtype=np.float64)
    n_total = 0

    for fname in labels:
        df = pd.read_csv(os.path.join(recordings_dir, fname))
        vals = df[ALL_COLUMNS].values.astype(np.float64)
        vals = vals - vals.mean(axis=0, keepdims=True)  # stage 1
        running_sum += vals.sum(axis=0)
        running_sq += (vals ** 2).sum(axis=0)
        n_total += vals.shape[0]

    mean = running_sum / n_total
    std = np.sqrt(running_sq / n_total - mean ** 2)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def load_csv_normalised(
    filepath: str,
    global_mean: np.ndarray,
    global_std: np.ndarray,
) -> np.ndarray:
    """Load CSV → (T, 48) numpy array with 2-stage normalisation."""
    df = pd.read_csv(filepath)
    vals = df[ALL_COLUMNS].values.astype(np.float32)
    vals = vals - vals.mean(axis=0, keepdims=True)  # stage 1
    vals = (vals - global_mean) / global_std           # stage 2
    return vals


# ── Pairing logic ────────────────────────────────────────────────────────

def build_pairs(
    labels: dict,
    pseudo_shifu_ratio: float = 0.5,
    seed: int = 42,
) -> List[Tuple[str, str, float, str]]:
    """Pair every reference with every student of the same movement."""
    rng = random.Random(seed)

    by_movement: Dict[str, Dict] = defaultdict(lambda: {"refs": [], "students": []})
    perfect_students: Dict[str, List[str]] = defaultdict(list)

    for fname, meta in labels.items():
        mov = meta["movement"]
        if meta["role"] == "shifu":
            by_movement[mov]["refs"].append(fname)
        if meta["role"] == "student":
            by_movement[mov]["students"].append((fname, meta["score"]))
            if meta["score"] == 1.0:
                perfect_students[mov].append(fname)

    for mov, candidates in perfect_students.items():
        k = max(1, round(len(candidates) * pseudo_shifu_ratio))
        promoted = rng.sample(candidates, k)
        by_movement[mov]["refs"].extend(promoted)

    pairs = []
    for movement, group in by_movement.items():
        for ref_file in group["refs"]:
            for student_file, score in group["students"]:
                if ref_file == student_file:
                    continue
                pairs.append((ref_file, student_file, score, movement))
    return pairs


def build_ref_index(
    labels: dict,
    pseudo_shifu_ratio: float = 0.5,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """movement -> [reference filenames] for negative sampling."""
    rng = random.Random(seed)

    index: Dict[str, List[str]] = defaultdict(list)
    perfect_students: Dict[str, List[str]] = defaultdict(list)

    for fname, meta in labels.items():
        if meta["role"] == "shifu":
            index[meta["movement"]].append(fname)
        elif meta["role"] == "student" and meta["score"] == 1.0:
            perfect_students[meta["movement"]].append(fname)

    for mov, candidates in perfect_students.items():
        k = max(1, round(len(candidates) * pseudo_shifu_ratio))
        promoted = rng.sample(candidates, k)
        index[mov].extend(promoted)

    return dict(index)


def inject_negatives(
    pairs: List[Tuple[str, str, float, str]],
    ref_index: Dict[str, List[str]],
    negative_pair_prob: float,
    seed: int,
) -> List[Tuple[str, str, float, str]]:
    """Deterministically inject cross-movement mismatches (score=0.0)."""
    rng = random.Random(seed + 1)  # offset seed to avoid correlation
    movements = list(ref_index.keys())
    result = []

    for ref_file, stu_file, score, movement in pairs:
        if rng.random() < negative_pair_prob:
            other = [m for m in movements if m != movement]
            neg_mov = rng.choice(other)
            neg_ref = rng.choice(ref_index[neg_mov])
            result.append((neg_ref, stu_file, 0.0, f"{neg_mov}_vs_{movement}"))
        else:
            result.append((ref_file, stu_file, score, movement))

    return result


# ── DTW feature extraction ───────────────────────────────────────────────

def compute_limb_dtw(
    ref_data: np.ndarray,
    student_data: np.ndarray,
    radius: int = 1,
) -> List[float]:
    """Compute DTW distance for each of the 4 limbs. Returns [d1, d2, d3, d4]."""
    distances = []
    for limb in LIMB_NAMES:
        sl = LIMB_SLICES[limb]
        ref_limb = ref_data[:, sl]      # (T_ref, 12)
        stu_limb = student_data[:, sl]   # (T_stu, 12)
        dist, _ = fastdtw(ref_limb, stu_limb, radius=radius, dist=euclidean)
        distances.append(float(dist))
    return distances


def extract_features(
    pairs: List[Tuple[str, str, float, str]],
    recordings_dir: str,
    global_mean: np.ndarray,
    global_std: np.ndarray,
    radius: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute DTW features for all pairs.
    Returns X (N, 4), y (N,), movements (N,).
    """
    csv_cache: Dict[str, np.ndarray] = {}

    def get_csv(fname: str) -> np.ndarray:
        if fname not in csv_cache:
            path = os.path.join(recordings_dir, fname)
            csv_cache[fname] = load_csv_normalised(path, global_mean, global_std)
        return csv_cache[fname]

    X_list, y_list, mov_list = [], [], []
    n = len(pairs)
    t0 = time.time()

    for i, (ref_f, stu_f, score, mov) in enumerate(pairs):
        ref_data = get_csv(ref_f)
        stu_data = get_csv(stu_f)
        dtw_dists = compute_limb_dtw(ref_data, stu_data, radius=radius)
        X_list.append(dtw_dists)
        y_list.append(score)
        mov_list.append(mov)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            print(f"  [{i+1:4d}/{n}] {elapsed:.1f}s elapsed")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), mov_list


# ── Training & evaluation ────────────────────────────────────────────────

def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    movements: List[str],
    args: argparse.Namespace,
) -> None:
    # Split
    (X_train, X_val, y_train, y_val,
     mov_train, mov_val) = train_test_split(
        X, y, movements,
        test_size=args.val_split,
        random_state=args.seed,
    )

    print(f"\nDataset: {len(X_train)} train / {len(X_val)} val")
    print(f"Score distribution (train): "
          f"0.0={np.sum(y_train==0.0)}, 0.1={np.sum(y_train==0.1)}, "
          f"0.5={np.sum(y_train==0.5)}, 1.0={np.sum(y_train==1.0)}")

    # ── Train models ─────────────────────────────────────────────────
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=args.seed,
        ),
        "SVR (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
        ]),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds_train = np.clip(model.predict(X_train), 0.0, 1.0)
        preds_val = np.clip(model.predict(X_val), 0.0, 1.0)
        mae_train = np.mean(np.abs(preds_train - y_train))
        mae_val = np.mean(np.abs(preds_val - y_val))
        results[name] = {
            "model": model,
            "preds_val": preds_val,
            "mae_train": mae_train,
            "mae_val": mae_val,
        }

    # ── Overall results ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"{'Model':<20} {'Train MAE':>10} {'Val MAE':>10}")
    print("-" * 50)
    for name, r in results.items():
        print(f"{name:<20} {r['mae_train']:>10.4f} {r['mae_val']:>10.4f}")
    print("=" * 50)

    # ── Per-movement breakdown ───────────────────────────────────────
    mov_val_arr = np.array(mov_val)
    unique_movs = sorted(set(mov_val))

    print(f"\n{'Movement':<25}", end="")
    for name in results:
        print(f" {name:>15}", end="")
    print(f" {'Count':>6}")
    print("-" * 70)

    for mov in unique_movs:
        mask = mov_val_arr == mov
        count = mask.sum()
        print(f"{mov:<25}", end="")
        for name, r in results.items():
            mae = np.mean(np.abs(r["preds_val"][mask] - y_val[mask]))
            print(f" {mae:>15.4f}", end="")
        print(f" {count:>6}")

    # ── Per-score-bucket breakdown ───────────────────────────────────
    buckets = sorted(set(y_val))
    print(f"\n{'GT Score':<12}", end="")
    for name in results:
        print(f" {name:>15}", end="")
    print(f" {'Count':>6}")
    print("-" * 55)

    for bucket in buckets:
        mask = y_val == bucket
        count = mask.sum()
        print(f"{bucket:<12.1f}", end="")
        for name, r in results.items():
            mae = np.mean(np.abs(r["preds_val"][mask] - y_val[mask]))
            print(f" {mae:>15.4f}", end="")
        print(f" {count:>6}")

    # ── Feature importances (RandomForest) ───────────────────────────
    rf_model = results["RandomForest"]["model"]
    importances = rf_model.feature_importances_
    print("\nRandomForest feature importances:")
    for limb, imp in zip(LIMB_NAMES, importances):
        bar = "#" * int(imp * 50)
        print(f"  {limb:<12} {imp:.4f}  {bar}")

    # ── Export best model ────────────────────────────────────────────
    best_name = min(results, key=lambda n: results[n]["mae_val"])
    best = results[best_name]

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "best_dtw_model.joblib")
    joblib.dump({
        "model": best["model"],
        "model_name": best_name,
        "val_mae": float(best["mae_val"]),
        "feature_names": [f"dtw_{limb}" for limb in LIMB_NAMES],
        "global_mean": global_mean,
        "global_std": global_std,
    }, out_path)
    print(f"\nSaved best model ({best_name}, val MAE={best['mae_val']:.4f}) -> {out_path}")


# ── Shared state for export (set in main) ────────────────────────────────
global_mean: Optional[np.ndarray] = None
global_std: Optional[np.ndarray] = None


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    global global_mean, global_std

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 50)
    print("  DTW + Classical ML for AQA (Plan B)")
    print("=" * 50)

    # Load labels
    labels = load_labels(args.recordings_dir)
    print(f"Loaded {len(labels)} recordings from {args.recordings_dir}")

    # Global normalisation stats
    print("Computing global normalisation stats...")
    global_mean, global_std = compute_global_stats(args.recordings_dir, labels)

    # Build pairs
    pairs = build_pairs(labels, args.pseudo_shifu_ratio, args.seed)
    ref_index = build_ref_index(labels, args.pseudo_shifu_ratio, args.seed)
    print(f"Built {len(pairs)} matched pairs")

    # Inject negatives
    pairs = inject_negatives(pairs, ref_index, args.negative_pair_prob, args.seed)
    n_neg = sum(1 for _, _, s, _ in pairs if s == 0.0)
    print(f"After negative injection: {len(pairs)} pairs ({n_neg} negatives)")

    # Extract DTW features
    print(f"\nExtracting DTW features (radius={args.fastdtw_radius})...")
    t0 = time.time()
    X, y, movements = extract_features(
        pairs, args.recordings_dir, global_mean, global_std,
        radius=args.fastdtw_radius,
    )
    print(f"Feature extraction done in {time.time()-t0:.1f}s")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Train & evaluate
    train_and_evaluate(X, y, movements, args)


if __name__ == "__main__":
    main()
