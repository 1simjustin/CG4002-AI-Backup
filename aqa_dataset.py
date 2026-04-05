"""
AQA Dataset & DataLoader for open-set action quality assessment.

Each sample pairs a student IMU recording with a shifu (expert) recording
of the same movement. The student has a ground-truth quality score in
{0.1, 0.5, 1.0} indicating how well they performed relative to the expert.

Pairing strategy:
    - References = shifu recordings + a fraction of score=1.0 students
      (pseudo-shifu, controlled by pseudo_shifu_ratio). Pseudo-shifu
      creates cross-session pairs that break session memorisation.
    - Each reference is paired with every student of the same movement.

Synthetic mismatches:
    ~35% of training pairs are replaced with cross-movement mismatches
    (shifu of one movement paired with student of a different movement,
    score forced to 0.0). This teaches the model a strong open-set
    boundary -- completely different movements should score near zero.

Input normalisation (two stages):
    1. Per-sample mean subtraction: removes session-specific gravity/
       orientation offset so the model sees motion deltas, not absolute
       sensor orientation.
    2. Global z-score: zero-mean, unit-variance per channel computed from
       all training CSVs, so accel and gyro channels are on the same scale.

Data augmentation (training only, applied independently per limb):
    1. Jitter: additive Gaussian noise (std=0.02), simulates sensor noise.
    2. Scaling: uniform multiplier in [0.9, 1.1], amplitude invariance.
    3. Random rotation: small SO(3) rotation (up to 10 deg) on accel/gyro
       xyz triples, simulates slight sensor misplacement between sessions.
    4. Channel dropout: zeroes out individual channels with 15% per-channel
       probability, simulates partial sensor noise/failure.
    5. Temporal cutout: zeroes out a contiguous time window (up to 15% of
       sequence), forces the model to not rely on any single phase.
"""

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# ── Column definitions (12 channels per limb) ──────────────────────────

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


# ── Helpers ─────────────────────────────────────────────────────────────

def _extract_timestamp(filename: str) -> str:
    """Sortable YYYYMMDD_HHMMSS from filename."""
    parts = filename.replace(".csv", "").split("_")
    return parts[-2] + "_" + parts[-1]


def _build_pairs(
    labels: dict,
    pseudo_shifu_ratio: float = 0.5,
    seed: int = 42,
) -> List[Tuple[str, str, float, str]]:
    """
    Pair every reference recording with every student of the same movement.

    References = all shifu files + a fraction of score=1.0 student files
    (controlled by pseudo_shifu_ratio). This creates cross-session pairs
    that break session memorisation.

    Args:
        labels: The labels dict from labels.json.
        pseudo_shifu_ratio: Fraction of score=1.0 students to promote to
            pseudo-shifu references. 0.0 = shifu only (original behaviour),
            1.0 = all perfect students become references.
        seed: Random seed for reproducible pseudo-shifu selection.

    Returns (reference_file, student_file, score, movement).
    """
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

    # Promote a random subset of perfect students to pseudo-shifu
    for mov, candidates in perfect_students.items():
        k = max(1, round(len(candidates) * pseudo_shifu_ratio))
        promoted = rng.sample(candidates, k)
        by_movement[mov]["refs"].extend(promoted)

    pairs = []
    for movement, group in by_movement.items():
        for ref_file in group["refs"]:
            for student_file, score in group["students"]:
                if ref_file == student_file:
                    continue  # skip self-pair
                pairs.append((ref_file, student_file, score, movement))
    return pairs


def _build_ref_index(
    labels: dict,
    pseudo_shifu_ratio: float = 0.5,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """movement -> [reference_filenames] for negative sampling.

    References include shifu files and a fraction of score=1.0 students.
    """
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


def _compute_global_stats(recordings_dir: str, labels: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std from ALL CSVs referenced in labels.

    Returns (mean_48, std_48) each of shape (48,).
    These are computed AFTER per-sample mean subtraction, so the global
    mean should be near zero -- the std captures the dynamic range of
    each channel type (accel vs gyro).
    """
    all_cols = []
    for limb_cols in LIMB_COLUMNS.values():
        all_cols.extend(limb_cols)

    running_sum = np.zeros(48, dtype=np.float64)
    running_sq = np.zeros(48, dtype=np.float64)
    n_total = 0

    for fname in labels:
        df = pd.read_csv(os.path.join(recordings_dir, fname))
        vals = df[all_cols].values.astype(np.float64)  # (T, 48)
        # Per-sample mean subtraction first
        vals = vals - vals.mean(axis=0, keepdims=True)
        running_sum += vals.sum(axis=0)
        running_sq += (vals ** 2).sum(axis=0)
        n_total += vals.shape[0]

    mean = running_sum / n_total
    std = np.sqrt(running_sq / n_total - mean ** 2)
    std[std < 1e-8] = 1.0  # avoid division by zero for constant channels
    return mean.astype(np.float32), std.astype(np.float32)


# Column order matching the 48-channel layout
_ALL_COLUMNS = []
for _limb_cols in LIMB_COLUMNS.values():
    _ALL_COLUMNS.extend(_limb_cols)


def _load_limbs(
    filepath: str,
    global_mean: Optional[np.ndarray] = None,
    global_std: Optional[np.ndarray] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load CSV -> dict of limb_name: (seq_len, 12) tensors.

    Normalisation (when global_mean/std provided):
        1. Per-sample mean subtraction (removes gravity/orientation offset)
        2. Global z-score (puts all channels on same scale)
    """
    df = pd.read_csv(filepath)
    vals = df[_ALL_COLUMNS].values.astype(np.float32)  # (T, 48)

    if global_mean is not None:
        # Stage 1: per-sample mean subtraction
        vals = vals - vals.mean(axis=0, keepdims=True)
        # Stage 2: global z-score
        vals = (vals - global_mean) / global_std

    limbs = {}
    offset = 0
    for limb in LIMB_COLUMNS:
        limbs[limb] = torch.tensor(vals[:, offset:offset + 12], dtype=torch.float32)
        offset += 12
    return limbs


def _truncate_or_pad(tensor: torch.Tensor, max_len: int) -> torch.Tensor:
    """Truncate to max_len or right-zero-pad.  Input: (seq, C)."""
    if tensor.size(0) >= max_len:
        return tensor[:max_len]
    pad = torch.zeros(max_len - tensor.size(0), tensor.size(1))
    return torch.cat([tensor, pad], dim=0)


def _augment(
    tensor: torch.Tensor,
    jitter_std: float,
    scale_range: Tuple[float, float],
    rotation_deg: float = 10.0,
    channel_drop_prob: float = 0.15,
    cutout_prob: float = 0.3,
    cutout_max_frac: float = 0.15,
) -> torch.Tensor:
    """
    Apply augmentations independently, each with its own probability.

    Args:
        tensor: (seq_len, 12) single-limb tensor.
        jitter_std: Gaussian noise std.
        scale_range: (low, high) uniform multiplier.
        rotation_deg: Max random rotation (degrees) applied to accel/gyro
            xyz triples, simulating slight sensor misplacement.
        channel_drop_prob: Probability of zeroing out each individual
            channel, simulating partial sensor noise/failure.
        cutout_prob: Probability of applying temporal cutout.
        cutout_max_frac: Max fraction of sequence to mask (e.g. 0.15 = 15%).
    """
    # 1. Jitter: additive Gaussian noise
    if random.random() < 0.5:
        tensor = tensor + torch.randn_like(tensor) * jitter_std

    # 2. Scaling: uniform multiplier
    if random.random() < 0.5:
        tensor = tensor * random.uniform(*scale_range)

    # 3. Small random rotation on accel/gyro xyz triples
    #    Each limb has 2 segments x (ax,ay,az,gx,gy,gz) = 12 channels.
    #    We apply the same rotation to both accel and gyro of each segment
    #    to maintain physical consistency.
    if random.random() < 0.4:
        angle = random.uniform(-rotation_deg, rotation_deg) * (3.14159265 / 180.0)
        # Random rotation axis (simplified: rotate around a random principal axis)
        axis = random.randint(0, 2)  # 0=x, 1=y, 2=z
        c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        R = torch.eye(3, dtype=tensor.dtype)
        # Rotation around chosen axis
        axes = [i for i in range(3) if i != axis]
        R[axes[0], axes[0]] = c
        R[axes[0], axes[1]] = -s
        R[axes[1], axes[0]] = s
        R[axes[1], axes[1]] = c

        rotated = tensor.clone()
        # Apply to each (accel_xyz, gyro_xyz) triple in each segment
        for seg_offset in (0, 6):  # 2 segments per limb
            for modal_offset in (0, 3):  # accel, then gyro
                start = seg_offset + modal_offset
                xyz = tensor[:, start:start + 3]  # (seq, 3)
                rotated[:, start:start + 3] = xyz @ R.T
        tensor = rotated

    # 4. Channel dropout: zero out individual channels
    if random.random() < 0.3:
        mask = torch.rand(tensor.size(1)) > channel_drop_prob  # True = keep
        tensor = tensor * mask.unsqueeze(0).float()

    # 5. Temporal cutout: zero out a contiguous time window
    if random.random() < cutout_prob:
        seq_len = tensor.size(0)
        max_cut = max(1, int(seq_len * cutout_max_frac))
        cut_len = random.randint(1, max_cut)
        start = random.randint(0, seq_len - cut_len)
        tensor = tensor.clone()
        tensor[start:start + cut_len] = 0.0

    return tensor


# ── Dataset ─────────────────────────────────────────────────────────────

class AQADataset(Dataset):
    """
    Pairs student IMU recordings with shifu (expert) recordings for
    quality scoring.

    Each sample returns 8 limb tensors (4 shifu + 4 student), a
    ground-truth score, and a mismatch flag. During training, a fraction
    of pairs are replaced with synthetic mismatches (cross-movement pairs
    with score=0.0) to provide negative signal for open-set learning.

    Args:
        recordings_dir: Folder containing CSVs and labels.json.
        max_seq_len: Fixed sequence length (truncate or right-zero-pad).
        augment: Enable data augmentation (training only). Applies jitter,
            scaling, random rotation, channel dropout, and temporal cutout
            independently per limb and per role (shifu/student).
        jitter_std: Gaussian noise std for jitter augmentation.
        scale_range: (low, high) uniform multiplier for scaling augmentation.
        negative_pair_prob: Probability of replacing the shifu with a
            reference from a different movement (synthetic mismatch).
            Default 0.0 (disabled); training default is 0.35.
        pseudo_shifu_ratio: Fraction of score=1.0 students promoted to
            pseudo-shifu references. Creates cross-session pairs.
        global_mean: Pre-computed (48,) channel means. If None, computed
            from all CSVs in labels.json.
        global_std: Pre-computed (48,) channel stds.
        seed: Random seed for reproducible pairing and pseudo-shifu selection.
    """

    def __init__(
        self,
        recordings_dir: str,
        max_seq_len: int = 128,
        augment: bool = False,
        jitter_std: float = 0.02,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        negative_pair_prob: float = 0.0,
        pseudo_shifu_ratio: float = 0.5,
        global_mean: Optional[np.ndarray] = None,
        global_std: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.recordings_dir = recordings_dir
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.negative_pair_prob = negative_pair_prob

        labels_path = os.path.join(recordings_dir, "labels.json")
        with open(labels_path) as f:
            self.labels = json.load(f)

        self.pairs = _build_pairs(self.labels, pseudo_shifu_ratio, seed)
        self._ref_index = _build_ref_index(self.labels, pseudo_shifu_ratio, seed)
        self._movements = list(self._ref_index.keys())

        # Normalisation stats: compute once if not provided
        if global_mean is None or global_std is None:
            self.global_mean, self.global_std = _compute_global_stats(
                recordings_dir, self.labels
            )
        else:
            self.global_mean = global_mean
            self.global_std = global_std

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        shifu_file, student_file, score, movement = self.pairs[idx]
        is_mismatch = False

        # Synthetic mismatch: swap reference to a different movement
        if self.negative_pair_prob > 0 and random.random() < self.negative_pair_prob:
            other_movements = [m for m in self._movements if m != movement]
            neg_movement = random.choice(other_movements)
            shifu_file = random.choice(self._ref_index[neg_movement])
            score = 0.0
            is_mismatch = True

        # Load and split into limb tensors (with normalisation)
        shifu_limbs = _load_limbs(
            os.path.join(self.recordings_dir, shifu_file),
            self.global_mean, self.global_std,
        )
        student_limbs = _load_limbs(
            os.path.join(self.recordings_dir, student_file),
            self.global_mean, self.global_std,
        )

        # Truncate / pad to fixed length
        for limb in LIMB_NAMES:
            shifu_limbs[limb] = _truncate_or_pad(shifu_limbs[limb], self.max_seq_len)
            student_limbs[limb] = _truncate_or_pad(student_limbs[limb], self.max_seq_len)

        # Augmentation (applied independently per role and limb)
        if self.augment:
            for limb in LIMB_NAMES:
                shifu_limbs[limb] = _augment(shifu_limbs[limb], self.jitter_std, self.scale_range)
                student_limbs[limb] = _augment(student_limbs[limb], self.jitter_std, self.scale_range)

        return {
            "shifu_left_arm": shifu_limbs["left_arm"],       # (max_seq_len, 12)
            "shifu_right_arm": shifu_limbs["right_arm"],
            "shifu_left_leg": shifu_limbs["left_leg"],
            "shifu_right_leg": shifu_limbs["right_leg"],
            "student_left_arm": student_limbs["left_arm"],
            "student_right_arm": student_limbs["right_arm"],
            "student_left_leg": student_limbs["left_leg"],
            "student_right_leg": student_limbs["right_leg"],
            "score": torch.tensor(score, dtype=torch.float32),
            "is_mismatch": torch.tensor(is_mismatch, dtype=torch.bool),
            "movement": movement,
        }


# ── Collate & DataLoader ───────────────────────────────────────────────

_LIMB_KEYS = [f"{role}_{limb}" for role in ("shifu", "student") for limb in LIMB_NAMES]


def aqa_collate_fn(batch: list) -> dict:
    """Stack fixed-length limb tensors into batched tensors."""
    collated = {}
    for key in _LIMB_KEYS:
        collated[key] = torch.stack([s[key] for s in batch])  # (B, seq, 12)
    collated["score"] = torch.stack([s["score"] for s in batch])
    collated["is_mismatch"] = torch.stack([s["is_mismatch"] for s in batch])
    collated["movement"] = [s["movement"] for s in batch]
    return collated


def get_aqa_dataloader(
    recordings_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    max_seq_len: int = 128,
    augment: bool = False,
    negative_pair_prob: float = 0.0,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    dataset = AQADataset(
        recordings_dir=recordings_dir,
        max_seq_len=max_seq_len,
        augment=augment,
        negative_pair_prob=negative_pair_prob,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=aqa_collate_fn,
        num_workers=num_workers,
        **kwargs,
    )


# ── Smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    RECORDINGS = os.path.join(os.path.dirname(__file__), "recordings")

    # Test with negative pairs enabled
    ds = AQADataset(RECORDINGS, max_seq_len=128, augment=True, negative_pair_prob=0.35)
    print(f"Total samples: {len(ds)}")

    sample = ds[0]
    print(f"\nSample keys: {list(sample.keys())}")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")

    # Test batch
    loader = get_aqa_dataloader(RECORDINGS, batch_size=4, max_seq_len=128,
                                augment=True, negative_pair_prob=0.35)
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {v}")

    # Count mismatches over full dataset
    mismatch_count = sum(1 for i in range(len(ds)) if ds[i]["is_mismatch"].item())
    print(f"\nMismatches: {mismatch_count}/{len(ds)} ({100*mismatch_count/len(ds):.1f}%)")
