"""
Training script for the open-set AQA Siamese Network.

Loss:
    L_total = L_contrastive

The contrastive loss is the sole training signal. Per-limb scores
emerge from embedding distances (exp(-dist^2)) without being forced
toward a broadcast scalar target.

The overall score is NOT learned -- it is computed post-hoc via
compute_overall_score() at inference time.

Usage:
    python train.py
    python train.py --epochs 200 --batch_size 16 --lr 1e-3
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from aqa_dataset import AQADataset, aqa_collate_fn
from aqa_model import AQAModel, LIMB_NAMES

LIMB_KEYS = [
    "shifu_left_arm", "shifu_right_arm", "shifu_left_leg", "shifu_right_leg",
    "student_left_arm", "student_right_arm", "student_left_leg", "student_right_leg",
]


# -- Contrastive loss ------------------------------------------------------

def contrastive_loss(
    embeddings: dict,
    scores: torch.Tensor,
    is_mismatch: torch.Tensor,
    margin: float = 2.0,
) -> torch.Tensor:
    """
    Per-limb target-distance contrastive loss on raw embeddings.

    For each limb pair (shifu_limb, student_limb):
        - Matched pairs (is_mismatch=False):
              target_dist = margin * (1 - score)
              loss = (D - target_dist)^2
          score=1.0 -> target=0 (pull together)
          score=0.5 -> target=margin/2 (moderate distance)
          score=0.1 -> target=0.9*margin (push far apart)
          Every score level gets gradient signal.

        - Mismatched pairs (is_mismatch=True, score=0.0):
              loss = max(0, margin - D)^2
          Pushes embeddings apart until distance >= margin.

    Where D = Euclidean distance between raw embeddings.

    Args:
        embeddings: dict of 8 tensors, each (B, embed_dim).
        scores: (B,) ground-truth scores.
        is_mismatch: (B,) bool flags.
        margin: max target distance for worst matched pairs & mismatch push distance.

    Returns:
        Scalar contrastive loss averaged over batch and limbs.
    """
    B = scores.size(0)
    device = scores.device
    total = torch.tensor(0.0, device=device)

    for limb in LIMB_NAMES:
        shifu_emb = embeddings[f"shifu_{limb}"]
        student_emb = embeddings[f"student_{limb}"]

        # Euclidean distance between raw embeddings
        dist = torch.norm(shifu_emb - student_emb, dim=-1)  # (B,)

        # Matched: target distance proportional to (1 - score)
        target_dist = margin * (1.0 - scores)  # (B,)
        matched_loss = (~is_mismatch).float() * (dist - target_dist).pow(2)

        # Mismatched: push apart beyond margin
        mismatched_loss = is_mismatch.float() * F.relu(margin - dist).pow(2)

        total = total + (matched_loss + mismatched_loss).mean()

    return total / len(LIMB_NAMES)


# -- Helpers ---------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train open-set AQA model")
    p.add_argument("--recordings_dir", type=str,
                   default=os.path.join(os.path.dirname(__file__), "recordings"))
    p.add_argument("--save_dir", type=str,
                   default=os.path.join(os.path.dirname(__file__), "checkpoints"))

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Loss
    p.add_argument("--contrastive_margin", type=float, default=1.5,
                   help="Max target distance for matched pairs & mismatch push distance")
    p.add_argument("--score_loss_weight", type=float, default=0.3,
                   help="Weight for supervised score MSE loss (0.0 to disable)")

    # Data
    p.add_argument("--negative_pair_prob", type=float, default=0.15)
    p.add_argument("--pseudo_shifu_ratio", type=float, default=0.5,
                   help="Fraction of score=1.0 students to use as pseudo-shifu (0.0-1.0)")
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def extract_model_inputs(batch: dict, device: torch.device) -> dict:
    """Pull the 8 limb tensors and move to device."""
    return {k: batch[k].to(device) for k in LIMB_KEYS}


# -- Train / Validate -----------------------------------------------------

def score_mse_loss(
    limb_scores: torch.Tensor,
    targets: torch.Tensor,
    is_mismatch: torch.Tensor,
) -> torch.Tensor:
    """
    Supervised MSE on per-limb scores for MATCHED pairs only.

    Each limb score is independently pushed toward the ground-truth target.
    Mismatched pairs are EXCLUDED -- the contrastive loss already pushes
    their embeddings apart, and including them here creates a massive bias
    toward low scores (since mismatches dominate the 0.0 target bucket).

    Args:
        limb_scores: (B, 4) per-limb scores from the model.
        targets: (B,) ground-truth scores.
        is_mismatch: (B,) bool flags.

    Returns:
        Scalar MSE loss (only over matched pairs). Returns 0.0 if no
        matched pairs exist in the batch.
    """
    matched = ~is_mismatch  # (B,)
    if not matched.any():
        return torch.tensor(0.0, device=limb_scores.device)

    matched_scores = limb_scores[matched]           # (M, 4)
    matched_targets = targets[matched]               # (M,)
    target_expanded = matched_targets.unsqueeze(1).expand_as(matched_scores)
    return F.mse_loss(matched_scores, target_expanded)


def train_one_epoch(model, loader, optimizer, args, device):
    model.train()
    accum = {"loss": 0, "cont_loss": 0, "score_loss": 0, "n": 0}

    for batch in loader:
        inputs = extract_model_inputs(batch, device)
        targets = batch["score"].to(device)              # (B,)
        is_mismatch = batch["is_mismatch"].to(device)    # (B,)
        B = targets.size(0)

        output = model(**inputs)

        l_cont = contrastive_loss(
            output["embeddings"], targets, is_mismatch, args.contrastive_margin,
        )

        # Supervised score calibration loss
        l_score = score_mse_loss(output["limb_scores"], targets, is_mismatch)

        loss = l_cont + args.score_loss_weight * l_score

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        accum["loss"] += loss.item() * B
        accum["cont_loss"] += l_cont.item() * B
        accum["score_loss"] += l_score.item() * B
        accum["n"] += B

    n = accum["n"]
    return {
        "loss": accum["loss"] / n,
        "cont_loss": accum["cont_loss"] / n,
        "score_loss": accum["score_loss"] / n,
    }


@torch.no_grad()
def validate(model, loader, args, device):
    model.eval()
    accum = {"loss": 0, "cont_loss": 0, "score_loss": 0, "n": 0}

    for batch in loader:
        inputs = extract_model_inputs(batch, device)
        targets = batch["score"].to(device)
        is_mismatch = batch["is_mismatch"].to(device)
        B = targets.size(0)

        output = model(**inputs)

        l_cont = contrastive_loss(
            output["embeddings"], targets, is_mismatch, args.contrastive_margin,
        )
        l_score = score_mse_loss(output["limb_scores"], targets, is_mismatch)
        loss = l_cont + args.score_loss_weight * l_score

        accum["loss"] += loss.item() * B
        accum["cont_loss"] += l_cont.item() * B
        accum["score_loss"] += l_score.item() * B
        accum["n"] += B

    n = accum["n"]
    return {
        "loss": accum["loss"] / n,
        "cont_loss": accum["cont_loss"] / n,
        "score_loss": accum["score_loss"] / n,
    }


# -- Main ------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Data splits (separate dataset instances for augment flag) --
    # Compute normalisation stats once and share between train/val
    train_ds = AQADataset(
        args.recordings_dir, max_seq_len=args.max_seq_len,
        augment=True, negative_pair_prob=args.negative_pair_prob,
        pseudo_shifu_ratio=args.pseudo_shifu_ratio, seed=args.seed,
    )
    val_ds = AQADataset(
        args.recordings_dir, max_seq_len=args.max_seq_len,
        augment=False, negative_pair_prob=0.0,
        pseudo_shifu_ratio=args.pseudo_shifu_ratio, seed=args.seed,
        global_mean=train_ds.global_mean, global_std=train_ds.global_std,
    )

    n_total = len(train_ds)
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed)).tolist()
    n_val = int(n_total * args.val_split)
    train_loader = DataLoader(
        Subset(train_ds, indices[n_val:]), batch_size=args.batch_size,
        shuffle=True, collate_fn=aqa_collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        Subset(val_ds, indices[:n_val]), batch_size=args.batch_size,
        shuffle=False, collate_fn=aqa_collate_fn, num_workers=0,
    )
    print(f"Dataset: {n_total} total | {n_total - n_val} train | {n_val} val")

    # -- Model / Optimizer / Scheduler --
    model = AQAModel().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience,
    )

    # -- Training loop --
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = -1

    header = (f"{'Ep':>4}  {'TrLoss':>8}  {'TrCont':>8}  {'TrScr':>8}  "
              f"{'VlLoss':>8}  {'VlCont':>8}  {'VlScr':>8}  {'Tau':>6}  {'LR':>10}")
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        tm = train_one_epoch(model, train_loader, optimizer, args, device)
        vm = validate(model, val_loader, args, device)

        scheduler.step(vm["loss"])
        lr = optimizer.param_groups[0]["lr"]
        tau = model.temperature.item()

        is_best = vm["loss"] < best_val_loss
        if is_best:
            best_val_loss = vm["loss"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "args": vars(args),
                "global_mean": train_ds.global_mean,
                "global_std": train_ds.global_std,
            }, os.path.join(args.save_dir, "best_model.pt"))

        mark = " *" if is_best else ""
        print(f"{epoch:4d}  {tm['loss']:8.4f}  {tm['cont_loss']:8.4f}  {tm['score_loss']:8.4f}  "
              f"{vm['loss']:8.4f}  {vm['cont_loss']:8.4f}  {vm['score_loss']:8.4f}  "
              f"{tau:6.3f}  {lr:10.2e}{mark}")

    print("-" * len(header))
    print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Final temperature: Tau = {model.temperature.item():.4f}")
    print(f"Saved: {os.path.join(args.save_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
