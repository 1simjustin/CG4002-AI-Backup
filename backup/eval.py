"""
Batch evaluation script for the AQA Siamese Network.

Compares each shifu recording against student recordings of the same
movement (matched) and optionally against students of other movements
(mismatched). Outputs a formatted table.

Usage:
    python eval.py
    python eval.py --checkpoint checkpoints/best_model.pt
    python eval.py --mismatches          # include cross-movement comparisons
    python eval.py --csv results.csv     # also save to CSV
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from aqa_model import AQAModel, LIMB_NAMES, compute_overall_score
from infer import load_sensor_columns, SlidingWindowBuffer, run_inference


def build_test_pairs(labels_path, include_mismatches=False):
    """
    Build a list of (label, shifu_csv, student_csv, gt_score) tuples.

    For each movement, pairs the first shifu with every student of
    that movement (matched). If include_mismatches is True, also pairs
    each shifu with the first gt=1.0 student of every OTHER movement.
    """
    with open(labels_path) as f:
        labels = json.load(f)

    # Group by movement
    movements = {}
    for fname, meta in sorted(labels.items()):
        mov = meta["movement"]
        if mov not in movements:
            movements[mov] = {"shifu": [], "students": {}}
        if meta["role"] == "shifu":
            movements[mov]["shifu"].append(fname)
        else:
            movements[mov]["students"].setdefault(meta["score"], []).append(fname)

    pairs = []

    # Matched pairs: first shifu vs all students of same movement
    for mov in sorted(movements):
        shifu_csv = movements[mov]["shifu"][0]
        for score in sorted(movements[mov]["students"].keys(), reverse=True):
            for student_csv in movements[mov]["students"][score]:
                label = f"{mov} {score}"
                pairs.append((label, shifu_csv, student_csv, score))

    # Mismatched pairs: each shifu vs first gt=1.0 student of other movements
    if include_mismatches:
        for shifu_mov in sorted(movements):
            shifu_csv = movements[shifu_mov]["shifu"][0]
            for student_mov in sorted(movements):
                if student_mov == shifu_mov:
                    continue
                student_csv = movements[student_mov]["students"][1.0][0]
                label = f"{shifu_mov} vs {student_mov}"
                pairs.append((label, shifu_csv, student_csv, 0.0))

    return pairs


def run_single(model, shifu_path, student_path, device,
               global_mean=None, global_std=None):
    """Load CSVs, fill buffers, run inference, return scores dict."""
    shifu_data = load_sensor_columns(shifu_path)
    student_data = load_sensor_columns(student_path)

    shifu_buf = SlidingWindowBuffer()
    student_buf = SlidingWindowBuffer()
    for i in range(max(len(shifu_data), len(student_data))):
        if i < len(shifu_data):
            shifu_buf.push(shifu_data[i])
        if i < len(student_data):
            student_buf.push(student_data[i])

    return run_inference(model, shifu_buf, student_buf, device,
                         global_mean, global_std)


def main():
    p = argparse.ArgumentParser(description="Batch AQA evaluation")
    p.add_argument("--recordings_dir", type=str,
                   default=os.path.join(os.path.dirname(__file__), "recordings"))
    p.add_argument("--checkpoint", type=str,
                   default=os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pt"))
    p.add_argument("--mismatches", action="store_true",
                   help="Include cross-movement mismatch comparisons")
    p.add_argument("--csv", type=str, default=None,
                   help="Optional path to save results as CSV")
    args = p.parse_args()

    labels_path = os.path.join(args.recordings_dir, "labels.json")
    if not os.path.isfile(labels_path):
        sys.exit(f"Error: labels.json not found at {labels_path}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Error: checkpoint not found: {args.checkpoint}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = AQAModel()
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False,
    )
    if missing:
        print(f"Note: Using defaults for missing keys: {missing}")
    model.to(device)
    model.eval()

    val_metric = checkpoint.get("val_loss", checkpoint.get("val_mae", None))
    metric_name = "val_loss" if "val_loss" in checkpoint else "val_mae"
    print(f"Model: epoch {checkpoint['epoch']}, {metric_name} {val_metric:.4f}")
    print(f"Checkpoint: {args.checkpoint}")

    global_mean = checkpoint.get("global_mean", None)
    global_std = checkpoint.get("global_std", None)
    if global_mean is not None:
        print("Normalisation stats loaded from checkpoint\n")
    else:
        print("WARNING: No normalisation stats in checkpoint (old model?)\n")

    # Build pairs and run
    pairs = build_test_pairs(labels_path, include_mismatches=args.mismatches)

    header = f"{'Test':<30s} {'GT':>4s}  {'Over':>5s}  {'LA':>5s}  {'RA':>5s}  {'LL':>5s}  {'RL':>5s}  {'Student CSV':<s}"
    print(header)
    print("-" * len(header))

    results = []
    prev_group = None
    for label, shifu_csv, student_csv, gt in pairs:
        group = label.split()[0]
        if prev_group and group != prev_group:
            print()
        prev_group = group

        shifu_path = os.path.join(args.recordings_dir, shifu_csv)
        student_path = os.path.join(args.recordings_dir, student_csv)
        scores = run_single(model, shifu_path, student_path, device,
                            global_mean, global_std)

        print(f"{label:<30s} {gt:4.1f}  {scores['overall']:5.3f}  "
              f"{scores['left_arm']:5.3f}  {scores['right_arm']:5.3f}  "
              f"{scores['left_leg']:5.3f}  {scores['right_leg']:5.3f}  "
              f"{student_csv}")

        results.append({
            "label": label,
            "shifu_csv": shifu_csv,
            "student_csv": student_csv,
            "gt_score": gt,
            "overall": scores["overall"],
            "left_arm": scores["left_arm"],
            "right_arm": scores["right_arm"],
            "left_leg": scores["left_leg"],
            "right_leg": scores["right_leg"],
        })

    print("-" * len(header))
    print(f"Total pairs evaluated: {len(results)}")

    # Optional CSV export
    if args.csv:
        import csv
        fieldnames = ["label", "shifu_csv", "student_csv", "gt_score",
                      "overall", "left_arm", "right_arm", "left_leg", "right_leg"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {args.csv}")


if __name__ == "__main__":
    main()
