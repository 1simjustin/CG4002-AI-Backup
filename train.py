"""
Training script for the open-set AQA Siamese Network.

The model uses separate arm/leg feature extractors with per-limb-group
learned temperatures. Training shapes the embedding space so that
similar movements cluster together and dissimilar movements separate,
enabling open-set generalisation to unseen movement types.

Loss (multi-task):
    L_total = L_contrastive + w_score * L_score_mse + w_bank * L_bank

    L_contrastive (primary signal):
        Per-limb target-distance contrastive loss on raw embeddings.
        - Matched pairs: target_dist = floor + (margin - floor) * (1 - score)
          Pulls embeddings to a distance proportional to quality gap.
          score=1.0 -> target=floor (0.5), score=0.5 -> midpoint, score=0.1 -> near margin
        - Mismatched pairs (synthetic cross-movement, score=0.0):
          Pushes embeddings apart until distance >= margin.

    L_score_mse (supervised calibration, weight=1.0):
        MSE between model's per-limb scores and ground-truth, applied
        only to matched pairs. Prevents score drift from contrastive
        loss alone. Mismatched pairs are excluded to avoid biasing
        toward low scores (since mismatches all have target 0.0).

    L_bank (memory bank cross-contrastive, weight=0.1):
        Computes pairwise L2 distances between current batch student
        embeddings and cached shifu embeddings from recent batches.
        All cross-batch pairs are treated as negatives (push-apart).
        This provides additional negative signal without increasing
        batch size -- important since contrastive learning benefits
        from seeing many negatives per step.

Memory bank:
    A FIFO buffer (default capacity 256) of detached embeddings from
    recent batches. Updated after each backward pass. The bank loss
    gives the model ~256 additional negative comparisons per step
    beyond the ~8 in-batch pairs.

Optimiser: AdamW with weight decay 1e-4 and gradient clipping (norm <= 1.0).
Scheduler: ReduceLROnPlateau (halves LR if val loss stalls for 15 epochs).

The overall score is NOT learned -- it is computed post-hoc via
compute_overall_score() at inference time.

Usage:
    python train.py
    python train.py --epochs 200 --batch_size 16 --lr 1e-3
    python train.py --negative_pair_prob 0.4 --bank_capacity 512
"""

import argparse
import os
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from aqa_dataset import AQADataset, aqa_collate_fn
from aqa_model import AQAModel, LIMB_NAMES, MODEL_CONFIGS

LIMB_KEYS = [
    "shifu_left_arm", "shifu_right_arm", "shifu_left_leg", "shifu_right_leg",
    "student_left_arm", "student_right_arm", "student_left_leg", "student_right_leg",
]


# -- Memory Bank -----------------------------------------------------------

class EmbeddingMemoryBank:
    """
    Fixed-capacity FIFO memory bank of recent embeddings for contrastive
    learning. Stores detached embeddings from past batches so the current
    batch can compute contrastive loss against a larger pool of negatives
    without increasing batch size.

    The bank stores per-limb shifu embeddings paired with their scores and
    mismatch flags. At each step, the current batch's student embeddings
    are compared against banked shifu embeddings (and vice versa) to
    generate additional contrastive pairs.

    Args:
        capacity: Maximum number of samples stored (oldest evicted first).
    """

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        # Per-limb deques of (embedding, score, is_mismatch) tuples
        self._shifu: dict = {limb: deque(maxlen=capacity) for limb in LIMB_NAMES}
        self._student: dict = {limb: deque(maxlen=capacity) for limb in LIMB_NAMES}
        self._scores: deque = deque(maxlen=capacity)
        self._is_mismatch: deque = deque(maxlen=capacity)

    @property
    def size(self) -> int:
        return len(self._scores)

    @torch.no_grad()
    def push(self, embeddings: dict, scores: torch.Tensor, is_mismatch: torch.Tensor):
        """Store detached embeddings from the current batch."""
        B = scores.size(0)
        for i in range(B):
            for limb in LIMB_NAMES:
                self._shifu[limb].append(embeddings[f"shifu_{limb}"][i].detach().clone())
                self._student[limb].append(embeddings[f"student_{limb}"][i].detach().clone())
            self._scores.append(scores[i].detach().clone())
            self._is_mismatch.append(is_mismatch[i].detach().clone())

    def get_bank_tensors(self, device: torch.device) -> dict:
        """
        Return banked embeddings as stacked tensors.

        Returns dict with:
            'shifu_{limb}': (N, embed_dim) for each limb
            'student_{limb}': (N, embed_dim) for each limb
            'scores': (N,)
            'is_mismatch': (N,)
        """
        if self.size == 0:
            return None
        result = {}
        for limb in LIMB_NAMES:
            result[f"shifu_{limb}"] = torch.stack(list(self._shifu[limb])).to(device)
            result[f"student_{limb}"] = torch.stack(list(self._student[limb])).to(device)
        result["scores"] = torch.stack(list(self._scores)).to(device)
        result["is_mismatch"] = torch.stack(list(self._is_mismatch)).to(device)
        return result


# -- Contrastive loss ------------------------------------------------------

def contrastive_loss(
    embeddings: dict,
    scores: torch.Tensor,
    is_mismatch: torch.Tensor,
    margin: float = 2.0,
    floor: float = 0.2,
) -> torch.Tensor:
    """
    Per-limb target-distance contrastive loss on raw embeddings.

    For each limb pair (shifu_limb, student_limb):
        - Matched pairs (is_mismatch=False):
              target_dist = floor + (margin - floor) * (1 - score)
              loss = (D - target_dist)^2
          score=1.0 -> target=floor (close but not forced to zero)
          score=0.5 -> target=midpoint between floor and margin
          score=0.1 -> target=near margin (push far apart)
          Every score level gets gradient signal.

          The floor prevents the loss from demanding zero distance for
          perfect students, which is unachievable since shifu and student
          are different recordings with natural variation. Without it,
          the model cannot satisfy dist=0, leading to an undertrained
          temperature and compressed scores.

        - Mismatched pairs (is_mismatch=True, score=0.0):
              loss = max(0, margin - D)^2
          Pushes embeddings apart until distance >= margin.

    Where D = Euclidean distance between raw embeddings.

    Args:
        embeddings: dict of 8 tensors, each (B, embed_dim).
        scores: (B,) ground-truth scores.
        is_mismatch: (B,) bool flags.
        margin: max target distance for worst matched pairs & mismatch push distance.
        floor: minimum target distance for score=1.0 pairs. Acknowledges
            that even perfect matches have irreducible natural variation.

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

        # Matched: target distance with floor for score=1.0 pairs
        # score=1.0 -> floor, score=0.0 -> margin
        target_dist = floor + (margin - floor) * (1.0 - scores)  # (B,)
        matched_loss = (~is_mismatch).float() * (dist - target_dist).pow(2)

        # Mismatched: push apart beyond margin
        mismatched_loss = is_mismatch.float() * F.relu(margin - dist).pow(2)

        total = total + (matched_loss + mismatched_loss).mean()

    return total / len(LIMB_NAMES)


def memory_bank_loss(
    current_embeddings: dict,
    bank: EmbeddingMemoryBank,
    margin: float = 2.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Cross-contrastive loss between current batch embeddings and the memory
    bank. For each limb, we compute distances between:
        - Current student embeddings vs banked shifu embeddings (treat as
          mismatch negatives -- different samples are assumed to be
          different movements for the purpose of pushing apart).

    This provides additional negative signal without requiring larger
    batch sizes. Only push-apart (mismatch) loss is applied since we
    don't know the ground-truth score between cross-batch pairs.

    Args:
        current_embeddings: dict of 8 tensors, each (B, embed_dim).
        bank: The memory bank with stored embeddings.
        margin: Push-apart distance for negatives.
        device: Torch device.

    Returns:
        Scalar loss averaged over limbs, or 0.0 if bank is empty.
    """
    bank_data = bank.get_bank_tensors(device)
    if bank_data is None:
        return torch.tensor(0.0, device=device)

    total = torch.tensor(0.0, device=device)

    for limb in LIMB_NAMES:
        # Current student vs banked shifu: (B, D) vs (N, D) -> (B, N)
        curr_student = current_embeddings[f"student_{limb}"]     # (B, D)
        banked_shifu = bank_data[f"shifu_{limb}"]                # (N, D)

        # Pairwise distances: (B, N)
        dists = torch.cdist(curr_student, banked_shifu, p=2)

        # Push-apart loss: all cross-batch pairs treated as negatives
        push_loss = F.relu(margin - dists).pow(2)
        total = total + push_loss.mean()

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
    p.add_argument("--contrastive_floor", type=float, default=0.5,
                   help="Min target distance for score=1.0 pairs (natural variation floor)")
    p.add_argument("--score_loss_weight", type=float, default=1.0,
                   help="Weight for supervised score MSE loss (0.0 to disable)")
    p.add_argument("--bank_loss_weight", type=float, default=0.1,
                   help="Weight for memory bank cross-contrastive loss (0.0 to disable)")
    p.add_argument("--bank_capacity", type=int, default=256,
                   help="Memory bank capacity (number of cached samples)")

    # Data
    p.add_argument("--negative_pair_prob", type=float, default=0.35)
    p.add_argument("--pseudo_shifu_ratio", type=float, default=0.5,
                   help="Fraction of score=1.0 students to use as pseudo-shifu (0.0-1.0)")
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)

    # Architecture
    p.add_argument("--model_type", type=str, default="original",
                   choices=list(MODEL_CONFIGS),
                   help="Architecture variant to train. "
                        f"Choices: {list(MODEL_CONFIGS)}. Default: original.")
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


def train_one_epoch(model, loader, optimizer, args, device, memory_bank):
    model.train()
    accum = {"loss": 0, "cont_loss": 0, "score_loss": 0, "bank_loss": 0, "n": 0}

    for batch in loader:
        inputs = extract_model_inputs(batch, device)
        targets = batch["score"].to(device)              # (B,)
        is_mismatch = batch["is_mismatch"].to(device)    # (B,)
        B = targets.size(0)

        output = model(**inputs)

        l_cont = contrastive_loss(
            output["embeddings"], targets, is_mismatch,
            margin=args.contrastive_margin, floor=args.contrastive_floor,
        )

        # Supervised score calibration loss
        l_score = score_mse_loss(output["limb_scores"], targets, is_mismatch)

        # Memory bank cross-contrastive loss
        l_bank = memory_bank_loss(
            output["embeddings"], memory_bank,
            margin=args.contrastive_margin, device=device,
        )

        loss = (l_cont
                + args.score_loss_weight * l_score
                + args.bank_loss_weight * l_bank)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Update memory bank AFTER backward (uses detached embeddings)
        memory_bank.push(output["embeddings"], targets, is_mismatch)

        accum["loss"] += loss.item() * B
        accum["cont_loss"] += l_cont.item() * B
        accum["score_loss"] += l_score.item() * B
        accum["bank_loss"] += l_bank.item() * B
        accum["n"] += B

    n = accum["n"]
    return {
        "loss": accum["loss"] / n,
        "cont_loss": accum["cont_loss"] / n,
        "score_loss": accum["score_loss"] / n,
        "bank_loss": accum["bank_loss"] / n,
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
            output["embeddings"], targets, is_mismatch,
            margin=args.contrastive_margin, floor=args.contrastive_floor,
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

    # -- Data splits --
    # Two dataset instances: train (augment=True, mismatches enabled) and
    # val (augment=False, no mismatches). Normalisation stats are computed
    # once from the training set and shared with the val set.
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
    model = AQAModel(**MODEL_CONFIGS[args.model_type]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    arm_params = sum(p.numel() for p in model.arm_extractor.parameters())
    leg_params = sum(p.numel() for p in model.leg_extractor.parameters())
    print(f"Architecture: {args.model_type}")
    print(f"Parameters: {total_params:,} (arm: {arm_params:,}, leg: {leg_params:,})")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience,
    )

    # -- Memory bank --
    memory_bank = EmbeddingMemoryBank(capacity=args.bank_capacity)
    print(f"Memory bank capacity: {args.bank_capacity}")

    # -- Training loop --
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = -1

    header = (f"{'Ep':>4}  {'TrLoss':>8}  {'TrCont':>8}  {'TrScr':>8}  {'TrBank':>8}  "
              f"{'VlLoss':>8}  {'VlCont':>8}  {'VlScr':>8}  "
              f"{'tA':>5}  {'tL':>5}  {'LR':>10}")
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        tm = train_one_epoch(model, train_loader, optimizer, args, device, memory_bank)
        vm = validate(model, val_loader, args, device)

        scheduler.step(vm["loss"])
        lr = optimizer.param_groups[0]["lr"]
        tau_arm = model.temperature_arm.item()
        tau_leg = model.temperature_leg.item()

        is_best = vm["loss"] < best_val_loss
        if is_best:
            best_val_loss = vm["loss"]
            best_epoch = epoch
            save_path = os.path.join(args.save_dir, f"best_{args.model_type}_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "args": vars(args),
                "model_type": args.model_type,
                "global_mean": train_ds.global_mean,
                "global_std": train_ds.global_std,
            }, save_path)

        mark = " *" if is_best else ""
        print(f"{epoch:4d}  {tm['loss']:8.4f}  {tm['cont_loss']:8.4f}  {tm['score_loss']:8.4f}  {tm['bank_loss']:8.4f}  "
              f"{vm['loss']:8.4f}  {vm['cont_loss']:8.4f}  {vm['score_loss']:8.4f}  "
              f"{tau_arm:5.3f}  {tau_leg:5.3f}  {lr:10.2e}{mark}")

    print("-" * len(header))
    print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Final temperatures: tau_arm={model.temperature_arm.item():.4f}, "
          f"tau_leg={model.temperature_leg.item():.4f}")
    print(f"Saved: {os.path.join(args.save_dir, f'best_{args.model_type}_model.pt')}")


if __name__ == "__main__":
    main()
