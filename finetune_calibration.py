"""
Score-only fine-tuning for the AQA Siamese model.

Recalibrates the learned temperatures (tau_arm, tau_leg) and the global
normalisation statistics to a new IMU data distribution, WITHOUT
retraining the feature extractors. Useful when:

    - IMU post-processing changes (e.g. offset calibration -> AHRS)
    - Sensor hardware changes slightly
    - You have only a small calibration set (10-30 pairs)

The feature extractors (arm/leg Conv + LSTM, ~35K params) are frozen.
Only 2 scalar temperatures are optimised, plus the 96 values in
global_mean/global_std are recomputed (not learned) from the new data.

The fine-tuning runs in inference mode (model.eval()) so the extractors
use the same RWAP pooling as deployment -- this is important because the
temperature is being calibrated to match inference behaviour.

Usage:
    python finetune_calibration.py \
        --checkpoint checkpoints/best_model.pt \
        --new_recordings_dir recordings_ahrs \
        --save_path checkpoints/best_model_ahrs.pt

    # With contrastive loss (if you have mismatch pairs via multiple movements)
    python finetune_calibration.py \
        --checkpoint checkpoints/best_model.pt \
        --new_recordings_dir recordings_ahrs \
        --save_path checkpoints/best_model_ahrs.pt \
        --negative_pair_prob 0.3 \
        --contrastive_weight 0.2
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from aqa_dataset import AQADataset, aqa_collate_fn, _compute_global_stats
from aqa_model import LIMB_NAMES
from infer import load_model
from train import contrastive_loss, extract_model_inputs, score_mse_loss


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def summarise_labels(recordings_dir: str) -> dict:
    """Print breakdown of the calibration set and warn if too small."""
    labels_path = os.path.join(recordings_dir, "labels.json")
    with open(labels_path) as f:
        labels = json.load(f)

    by_movement: dict = defaultdict(
        lambda: {"shifu": 0, "students": defaultdict(int)}
    )
    for _, meta in labels.items():
        mov = meta["movement"]
        if meta["role"] == "shifu":
            by_movement[mov]["shifu"] += 1
        else:
            by_movement[mov]["students"][meta["score"]] += 1

    print(f"\nCalibration data:")
    print(f"  {'Movement':<15s}  {'Shifu':>6s}  {'s=1.0':>6s}  {'s=0.5':>6s}  {'s=0.1':>6s}")
    total_students = 0
    score_coverage: set = set()
    for mov in sorted(by_movement):
        counts = by_movement[mov]
        s = counts["students"]
        total_students += sum(s.values())
        score_coverage.update(s.keys())
        print(f"  {mov:<15s}  {counts['shifu']:>6d}  "
              f"{s.get(1.0, 0):>6d}  {s.get(0.5, 0):>6d}  {s.get(0.1, 0):>6d}")
    print(f"  Movements: {len(by_movement)}  |  Total students: {total_students}  |  "
          f"Score levels: {sorted(score_coverage)}")

    if total_students < 10:
        print("  WARNING: Fewer than 10 student recordings. Results may be unreliable.")
    if len(score_coverage) < 2:
        print("  WARNING: Only one score level present. Temperature cannot be "
              "uniquely identified from a single score.")

    return labels


@torch.no_grad()
def collect_distance_stats(model, loader, device) -> dict:
    """Per-limb embedding distances grouped by ground-truth score."""
    model.eval()
    by_key: dict = defaultdict(list)

    for batch in loader:
        inputs = extract_model_inputs(batch, device)
        targets = batch["score"].to(device)
        is_mismatch = batch["is_mismatch"].to(device)
        output = model(**inputs)

        for i in range(targets.size(0)):
            # Tag mismatches as "MISS" so they're visible separately
            key_score = "MISS" if is_mismatch[i].item() else round(float(targets[i].item()), 2)
            for limb in LIMB_NAMES:
                shifu_emb = output["embeddings"][f"shifu_{limb}"][i]
                student_emb = output["embeddings"][f"student_{limb}"][i]
                dist = torch.norm(shifu_emb - student_emb).item()
                by_key[(key_score, limb)].append(dist)

    return by_key


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Mean per-sample predicted score, grouped by ground-truth score."""
    model.eval()
    by_score: dict = defaultdict(list)

    for batch in loader:
        inputs = extract_model_inputs(batch, device)
        targets = batch["score"].to(device)
        is_mismatch = batch["is_mismatch"].to(device)
        output = model(**inputs)
        mean_scores = output["limb_scores"].mean(dim=1)  # (B,)
        for i in range(targets.size(0)):
            key = "MISS" if is_mismatch[i].item() else round(float(targets[i].item()), 2)
            by_score[key].append(mean_scores[i].item())

    return by_score


def _print_distance_table(dist_stats: dict) -> None:
    score_keys = sorted(
        {k[0] for k in dist_stats},
        key=lambda x: (isinstance(x, str), x),  # numeric first, then 'MISS'
    )
    header = f"  {'Score':<6s}  " + "  ".join(f"{l:>14s}" for l in LIMB_NAMES)
    print(header)
    for score in score_keys:
        row = f"  {str(score):<6s}"
        for limb in LIMB_NAMES:
            vals = dist_stats.get((score, limb), [])
            if vals:
                row += f"  {np.mean(vals):>6.3f}±{np.std(vals):<5.3f} "
            else:
                row += f"  {'--':>14s}"
        print(row)


def _print_eval_table(eval_stats: dict) -> None:
    keys = sorted(eval_stats, key=lambda x: (isinstance(x, str), x))
    for gt in keys:
        preds = eval_stats[gt]
        print(f"  GT={str(gt):<5s}  n={len(preds):>3d}  "
              f"mean={np.mean(preds):.3f}  std={np.std(preds):.3f}  "
              f"range=[{np.min(preds):.3f}, {np.max(preds):.3f}]")


def sanity_checks(eval_stats: dict) -> None:
    """Post-fit sanity checks on score ordering and mismatch separation."""
    numeric = sorted(k for k in eval_stats if not isinstance(k, str))
    means = [np.mean(eval_stats[k]) for k in numeric]

    print("\n--- Sanity checks ---")
    if len(numeric) >= 2:
        ordered = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
        status = "OK (monotonic)" if ordered else "FAILED"
        print(f"  Score ordering: {status}")
        for k, m in zip(numeric, means):
            print(f"    GT={k:.2f} -> predicted mean={m:.3f}")
        if not ordered:
            print("    FIX: feature extractors likely need retraining, or collect more "
                  "data covering the full score range.")
    else:
        print("  Score ordering: SKIPPED (need >=2 score levels)")

    if "MISS" in eval_stats:
        miss_mean = np.mean(eval_stats["MISS"])
        ok = miss_mean < 0.2
        print(f"  Mismatch separation: {'OK' if ok else 'DEGRADED'} (mean={miss_mean:.3f})")
        if not ok:
            print("    FIX: open-set boundary may have collapsed. Try increasing "
                  "--contrastive_weight or reducing --lr.")

    if numeric and means:
        spread = means[-1] - means[0]
        print(f"  Score range spread: {spread:.3f} "
              f"({'OK' if spread > 0.3 else 'COMPRESSED — limited discriminative power'})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Score-only fine-tuning for AQA model")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to existing trained checkpoint")
    p.add_argument("--new_recordings_dir", type=str, required=True,
                   help="Directory with new-format recordings + labels.json")
    p.add_argument("--save_path", type=str, required=True,
                   help="Where to save the re-calibrated checkpoint")

    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--steps", type=int, default=200,
                   help="Number of optimisation steps")
    p.add_argument("--lr", type=float, default=0.05,
                   help="Learning rate (can be high — only 2 params)")

    p.add_argument("--contrastive_weight", type=float, default=0.0,
                   help="Weight for contrastive loss. >0 requires mismatch pairs.")
    p.add_argument("--contrastive_margin", type=float, default=1.5)
    p.add_argument("--contrastive_floor", type=float, default=0.5)
    p.add_argument("--negative_pair_prob", type=float, default=0.0,
                   help="Generate synthetic mismatches (requires >=2 movements)")

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Error: checkpoint not found: {args.checkpoint}")
    if not os.path.isdir(args.new_recordings_dir):
        sys.exit(f"Error: recordings dir not found: {args.new_recordings_dir}")

    # -- Data summary --
    labels = summarise_labels(args.new_recordings_dir)

    # -- Load old checkpoint --
    print(f"\nLoading old checkpoint: {args.checkpoint}")
    model, checkpoint, old_mean, old_std = load_model(args.checkpoint, device)
    old_tau_arm = model.temperature_arm.item()
    old_tau_leg = model.temperature_leg.item()
    print(f"  Old temperatures: tau_arm={old_tau_arm:.4f}  tau_leg={old_tau_leg:.4f}")

    # -- Recompute normalisation stats from new data --
    print(f"\nRecomputing normalisation stats from new recordings...")
    new_mean, new_std = _compute_global_stats(args.new_recordings_dir, labels)

    if old_mean is not None:
        mean_shift = float(np.abs(new_mean - old_mean).mean())
        std_ratio = float((new_std / np.clip(old_std, 1e-8, None)).mean())
        print(f"  Mean |shift|: {mean_shift:.4f}   Std ratio (new/old, mean): {std_ratio:.4f}")
        if std_ratio < 0.5 or std_ratio > 2.0:
            print("  WARNING: large std shift -- the feature extractor was trained on "
                  "a meaningfully different distribution. Temperature-only fine-tuning "
                  "may not be sufficient.")
    else:
        print("  (Old checkpoint has no normalisation stats — using new stats fresh.)")

    # -- Build dataset with new stats --
    if args.negative_pair_prob > 0:
        n_movements = len({meta["movement"] for meta in labels.values()})
        if n_movements < 2:
            sys.exit("Error: --negative_pair_prob requires >= 2 movements in the dataset.")

    dataset = AQADataset(
        recordings_dir=args.new_recordings_dir,
        max_seq_len=args.max_seq_len,
        augment=False,
        negative_pair_prob=args.negative_pair_prob,
        pseudo_shifu_ratio=0.0,  # Don't expand the calibration set artificially
        global_mean=new_mean,
        global_std=new_std,
        seed=args.seed,
    )
    if len(dataset) < 5:
        sys.exit(f"Error: only {len(dataset)} pairs — need at least 5.")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=aqa_collate_fn, num_workers=0,
    )
    print(f"  Dataset: {len(dataset)} pairs")

    # -- Freeze everything except temperatures --
    for p in model.parameters():
        p.requires_grad = False
    model.log_temperature_arm.requires_grad = True
    model.log_temperature_leg.requires_grad = True
    trainable = [model.log_temperature_arm, model.log_temperature_leg]
    print(f"  Trainable params: {sum(p.numel() for p in trainable)} (temperatures only)")

    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    # IMPORTANT: eval() — calibrate against the same pooling (RWAP) used at
    # inference. BN uses running stats; dropout is disabled. Gradients still
    # flow through the temperature parameters.
    model.eval()

    # -- Pre-fit diagnostics --
    print(f"\nPre-fit embedding distances:")
    _print_distance_table(collect_distance_stats(model, loader, device))
    print(f"\nPre-fit predicted scores:")
    _print_eval_table(evaluate(model, loader, device))

    # -- Fine-tune loop --
    print(f"\nFitting temperatures ({args.steps} steps, lr={args.lr})")
    hdr = (f"{'Step':>5s}  {'Loss':>8s}  {'ScoreMSE':>9s}  "
           f"{'Contrast':>9s}  {'tau_arm':>8s}  {'tau_leg':>8s}")
    print(hdr)
    print("-" * len(hdr))

    step = 0
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            inputs = extract_model_inputs(batch, device)
            targets = batch["score"].to(device)
            is_mismatch = batch["is_mismatch"].to(device)

            output = model(**inputs)

            l_score = score_mse_loss(output["limb_scores"], targets, is_mismatch)

            if args.contrastive_weight > 0:
                l_cont = contrastive_loss(
                    output["embeddings"], targets, is_mismatch,
                    margin=args.contrastive_margin,
                    floor=args.contrastive_floor,
                )
            else:
                l_cont = torch.tensor(0.0, device=device)

            loss = l_score + args.contrastive_weight * l_cont

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(f"{step:>5d}  {loss.item():>8.4f}  {l_score.item():>9.4f}  "
                      f"{l_cont.item():>9.4f}  "
                      f"{model.temperature_arm.item():>8.4f}  "
                      f"{model.temperature_leg.item():>8.4f}")

    # -- Post-fit diagnostics --
    print(f"\nPost-fit embedding distances:")
    _print_distance_table(collect_distance_stats(model, loader, device))
    print(f"\nPost-fit predicted scores:")
    post_eval = evaluate(model, loader, device)
    _print_eval_table(post_eval)

    sanity_checks(post_eval)

    # -- Save new checkpoint --
    new_checkpoint = dict(checkpoint)
    new_checkpoint["model_state_dict"] = model.state_dict()
    new_checkpoint["global_mean"] = new_mean
    new_checkpoint["global_std"] = new_std
    new_checkpoint["finetune"] = {
        "source_checkpoint": args.checkpoint,
        "recordings_dir": args.new_recordings_dir,
        "steps": args.steps,
        "lr": args.lr,
        "contrastive_weight": args.contrastive_weight,
        "old_tau_arm": old_tau_arm,
        "old_tau_leg": old_tau_leg,
        "new_tau_arm": model.temperature_arm.item(),
        "new_tau_leg": model.temperature_leg.item(),
    }

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(new_checkpoint, args.save_path)

    print(f"\nSaved: {args.save_path}")
    print(f"  tau_arm: {old_tau_arm:.4f} -> {model.temperature_arm.item():.4f}")
    print(f"  tau_leg: {old_tau_leg:.4f} -> {model.temperature_leg.item():.4f}")


if __name__ == "__main__":
    main()
