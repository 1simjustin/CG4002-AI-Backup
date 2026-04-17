"""
File-based inference for the open-set AQA Siamese Network.

Simulates streaming by replaying two saved CSVs (shifu + student)
through a SlidingWindowBuffer, triggering inference every --step_size
frames. This is useful for offline evaluation and debugging.

The inference pipeline:
    1. Load checkpoint (model weights + normalisation stats).
    2. Push frames from both CSVs into separate SlidingWindowBuffers.
    3. Every step_size frames, extract (128, 48) windows from both
       buffers.
    4. Apply the same two-stage normalisation used during training:
       (a) Per-sample mean subtraction (only non-padding rows)
       (b) Global z-score using stats from the training set
    5. Split each window into 4 limb tensors (12 channels each).
    6. Forward pass through the model (arm extractor for arms, leg
       extractor for legs) to get per-limb embeddings.
    7. Compute per-limb scores via exp(-dist^2 / tau_group^2).
    8. Aggregate into overall score via compute_overall_score()
       (geometric mean + worst-limb penalty -- fixed formula, not learned).

Also provides load_model(), SlidingWindowBuffer, run_inference(), and
other utilities imported by live_infer.py and eval.py.

Usage:
    python infer.py --shifu_csv recordings/kick_shifu_*.csv --student_csv recordings/kick_student_*.csv
    python infer.py --shifu_csv recordings/kick_shifu_*.csv --student_csv recordings/kick_student_*.csv --step_size 20
"""

import argparse
import os
import sys
from collections import deque

import numpy as np
import pandas as pd
import torch

from aqa_model import AQAModel, LIMB_NAMES, MODEL_CONFIGS, compute_overall_score, detect_model_type

# Must match training max_seq_len
WINDOW_SIZE = 128
NUM_SENSORS = 48

# Column index slices for each limb (0-indexed within the 48 sensor columns)
# left_arm: cols 0-11, right_arm: 12-23, left_leg: 24-35, right_leg: 36-47
LIMB_SLICES = {
    "left_arm":  slice(0, 12),
    "right_arm": slice(12, 24),
    "left_leg":  slice(24, 36),
    "right_leg": slice(36, 48),
}


# -- Sliding Window Buffer ------------------------------------------------

class SlidingWindowBuffer:
    """
    Fixed-capacity ring buffer for streaming IMU data.

    Stores up to `window_size` timesteps of `num_channels` sensor values.
    When full, new pushes evict the oldest frame (FIFO).
    """

    def __init__(self, window_size: int = WINDOW_SIZE, num_channels: int = NUM_SENSORS):
        self.window_size = window_size
        self.num_channels = num_channels
        self._buf: deque = deque(maxlen=window_size)

    def push(self, frame: np.ndarray) -> None:
        """Push a single timestep of shape (num_channels,)."""
        assert frame.shape == (self.num_channels,), (
            f"Expected ({self.num_channels},), got {frame.shape}"
        )
        self._buf.append(frame)

    @property
    def count(self) -> int:
        return len(self._buf)

    def get_window(self) -> np.ndarray:
        """
        Return the current window as (window_size, num_channels).
        Left-zero-pads if fewer than window_size frames are available.
        """
        if self.count == 0:
            return np.zeros((self.window_size, self.num_channels), dtype=np.float32)

        data = np.array(self._buf, dtype=np.float32)  # (count, C)
        if data.shape[0] < self.window_size:
            pad = np.zeros((self.window_size - data.shape[0], self.num_channels), dtype=np.float32)
            data = np.concatenate([pad, data], axis=0)
        return data

    def reset(self) -> None:
        self._buf.clear()


# -- Preprocessing ---------------------------------------------------------

def normalize_window(
    window: np.ndarray,
    global_mean: np.ndarray,
    global_std: np.ndarray,
) -> np.ndarray:
    """
    Apply the same two-stage normalisation used during training:
        1. Per-sample mean subtraction (removes gravity/orientation offset)
        2. Global z-score (puts accel and gyro on the same scale)

    Args:
        window: (128, 48) raw sensor values.
        global_mean: (48,) from training set (post mean-subtraction).
        global_std: (48,) from training set (post mean-subtraction).

    Returns:
        (128, 48) normalised array.
    """
    # Only subtract mean from non-padding rows (non-zero)
    non_pad = np.any(window != 0, axis=1)
    if non_pad.any():
        sample_mean = window[non_pad].mean(axis=0, keepdims=True)
        out = window.copy()
        out[non_pad] = out[non_pad] - sample_mean
    else:
        out = window.copy()
    out = (out - global_mean) / global_std
    # Re-zero the padding rows
    out[~non_pad] = 0.0
    return out


def window_to_limb_tensors(
    window: np.ndarray,
    device: torch.device,
    global_mean: np.ndarray = None,
    global_std: np.ndarray = None,
) -> dict:
    """
    Convert a (128, 48) numpy window into 4 limb tensors of shape (1, 128, 12).
    Applies normalisation if stats are provided.
    """
    if global_mean is not None:
        window = normalize_window(window, global_mean, global_std)

    tensors = {}
    for limb, sl in LIMB_SLICES.items():
        t = torch.tensor(window[:, sl], dtype=torch.float32)  # (128, 12)
        tensors[limb] = t.unsqueeze(0).to(device)              # (1, 128, 12)
    return tensors


def detect_missing_limbs(raw_window: np.ndarray) -> dict:
    """
    Detect which limbs have no data (both segments all-zero across the window).

    Must be called on the RAW window before normalization, since z-score
    transforms zeros into non-zero values.

    Args:
        raw_window: (128, 48) raw sensor values.

    Returns:
        Dict mapping limb name -> bool (True if missing).
    """
    missing = {}
    for limb, sl in LIMB_SLICES.items():
        limb_data = raw_window[:, sl]           # (128, 12)
        missing[limb] = not np.any(limb_data)   # True if all zeros
    return missing


# -- Inference step --------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: AQAModel,
    shifu_buf: SlidingWindowBuffer,
    student_buf: SlidingWindowBuffer,
    device: torch.device,
    global_mean: np.ndarray = None,
    global_std: np.ndarray = None,
) -> dict:
    """
    Extract windows from both buffers, preprocess, run model, return scores.

    Limbs with missing data (both segments all-zero) in either the shifu
    or student stream get a sentinel score of -1.0 and are excluded from
    the overall score computation.
    """
    shifu_window = shifu_buf.get_window()
    student_window = student_buf.get_window()

    # Detect missing limbs on raw windows BEFORE normalization
    shifu_missing = detect_missing_limbs(shifu_window)
    student_missing = detect_missing_limbs(student_window)
    missing = {l: shifu_missing[l] or student_missing[l] for l in LIMB_NAMES}

    shifu_limbs = window_to_limb_tensors(shifu_window, device, global_mean, global_std)
    student_limbs = window_to_limb_tensors(student_window, device, global_mean, global_std)

    model_input = {}
    for limb in LIMB_NAMES:
        model_input[f"shifu_{limb}"] = shifu_limbs[limb]
        model_input[f"student_{limb}"] = student_limbs[limb]

    output = model(**model_input)
    limb_scores = output["limb_scores"].squeeze(0).cpu().numpy()     # (4,)

    # Override missing limbs with sentinel and compute overall from valid only
    for i, limb in enumerate(LIMB_NAMES):
        if missing[limb]:
            limb_scores[i] = -1.0

    valid_indices = [i for i, l in enumerate(LIMB_NAMES) if not missing[l]]
    if valid_indices:
        valid_t = torch.tensor([[limb_scores[i] for i in valid_indices]])
        overall_score = compute_overall_score(valid_t).squeeze().item()
    else:
        overall_score = -1.0

    return {
        "overall": overall_score,
        "left_arm": limb_scores[0],
        "right_arm": limb_scores[1],
        "left_leg": limb_scores[2],
        "right_leg": limb_scores[3],
    }


def _fmt_score(s: float) -> str:
    """Format a limb/overall score, showing N/A for missing (-1)."""
    return " N/A" if s < 0 else f"{s:.3f}"


def print_scores(frame_idx: int, filled: int, scores: dict) -> None:
    """Pretty-print inference results."""
    print(f"  Frame {frame_idx:>4d} | buf {filled:>3d}/{WINDOW_SIZE} | "
          f"Overall: {_fmt_score(scores['overall'])} | "
          f"LA: {_fmt_score(scores['left_arm'])}  RA: {_fmt_score(scores['right_arm'])}  "
          f"LL: {_fmt_score(scores['left_leg'])}  RL: {_fmt_score(scores['right_leg'])}")


# -- File helpers ----------------------------------------------------------

def load_sensor_columns(csv_path: str) -> np.ndarray:
    """Load CSV, drop timestamp_ms, return (num_rows, 48) numpy array."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["timestamp_ms"], errors="ignore")
    assert df.shape[1] == NUM_SENSORS, (
        f"Expected {NUM_SENSORS} sensor columns, got {df.shape[1]} in {csv_path}"
    )
    return df.values.astype(np.float32)


def load_model(checkpoint_path, device, model_type="auto"):
    """
    Load model and normalisation stats from a training checkpoint.

    The checkpoint contains:
        - model_state_dict: weights for arm_extractor, leg_extractor,
          and per-group temperatures (log_temperature_arm, log_temperature_leg).
        - global_mean / global_std: (48,) arrays for z-score normalisation,
          computed from the training set after per-sample mean subtraction.
        - args: training hyperparameters for reproducibility.

    Args:
        checkpoint_path: Path to a .pt checkpoint file.
        device: torch.device to load weights onto.
        model_type: Which architecture to instantiate.
            'auto'     -- detect automatically from checkpoint state-dict keys
                          (arm_extractor.lstm.* -> 'original',
                           arm_extractor.temporal.* -> 'slim').
            'slim'     -- dilated-conv temporal aggregator (slim-model branch).
            'original' -- Bi-LSTM temporal aggregator (masking branch).

    Uses strict=False to allow loading old checkpoints that may have a
    different key layout (e.g. single shared extractor, global temperature).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if model_type == "auto":
        model_type = detect_model_type(checkpoint["model_state_dict"])
        print(f"Auto-detected model type: '{model_type}'")

    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         f"Choose from: {list(MODEL_CONFIGS)}")

    model = AQAModel(**MODEL_CONFIGS[model_type])
    try:
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False)
    except RuntimeError as exc:
        # Shape mismatches happen when the wrong model_type is forced
        # (e.g. --model_type original on a slim checkpoint).
        raise RuntimeError(
            f"Shape mismatch loading '{model_type}' architecture from "
            f"'{checkpoint_path}'. The checkpoint was likely saved with a "
            f"different architecture. Try --model_type auto to detect "
            f"automatically.\n\nOriginal error: {exc}"
        ) from None
    if missing:
        print(f"Note: Using defaults for missing keys: {missing}")
    if unexpected:
        print(f"Note: Ignoring unexpected keys from old checkpoint: {unexpected}")
    model.to(device)
    model.eval()
    print(f"Model type: '{model_type}'")

    global_mean = checkpoint.get("global_mean", None)
    global_std = checkpoint.get("global_std", None)

    return model, checkpoint, global_mean, global_std


# -- Main ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="File-based AQA inference")
    p.add_argument("--shifu_csv", type=str, required=True,
                   help="Path to shifu CSV")
    p.add_argument("--student_csv", type=str, required=True,
                   help="Path to student CSV")
    p.add_argument("--checkpoint", type=str,
                   default=os.path.join(os.path.dirname(__file__), "checkpoints", "best_original_model.pt"),
                   help="Path to saved model checkpoint")
    p.add_argument("--step_size", type=int, default=10,
                   help="Frames between inference steps")
    p.add_argument("--device", type=str, default="auto",
                   help="'cpu', 'cuda', or 'auto'")
    p.add_argument("--model_type", type=str, default="auto",
                   choices=["auto", "slim", "original"],
                   help="Architecture variant to load. 'auto' detects from "
                        "checkpoint keys (default). 'slim' = dilated-conv "
                        "temporal block; 'original' = Bi-LSTM.")
    args = p.parse_args()

    if not os.path.isfile(args.shifu_csv):
        sys.exit(f"Error: shifu CSV not found: {args.shifu_csv}")
    if not os.path.isfile(args.student_csv):
        sys.exit(f"Error: student CSV not found: {args.student_csv}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Error: checkpoint not found: {args.checkpoint}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    model, checkpoint, global_mean, global_std = load_model(
        args.checkpoint, device, args.model_type)
    val_metric = checkpoint.get('val_loss', checkpoint.get('val_mae', None))
    metric_name = 'val_loss' if 'val_loss' in checkpoint else 'val_mae'
    print(f"Model loaded (trained epoch {checkpoint['epoch']}, "
          f"{metric_name} {val_metric:.4f})")
    if global_mean is not None:
        print("Normalisation stats loaded from checkpoint\n")
    else:
        print("WARNING: No normalisation stats in checkpoint (old model?)\n")

    # -- Run --
    print(f"Loading shifu:   {args.shifu_csv}")
    print(f"Loading student: {args.student_csv}")

    shifu_data = load_sensor_columns(args.shifu_csv)
    student_data = load_sensor_columns(args.student_csv)
    print(f"Shifu frames: {len(shifu_data)}, Student frames: {len(student_data)}")

    shifu_buf = SlidingWindowBuffer()
    student_buf = SlidingWindowBuffer()

    total_frames = max(len(shifu_data), len(student_data))
    frames_since_last = 0

    print(f"\nStreaming {total_frames} frames (step_size={args.step_size})...\n")

    for i in range(total_frames):
        if i < len(shifu_data):
            shifu_buf.push(shifu_data[i])
        if i < len(student_data):
            student_buf.push(student_data[i])

        frames_since_last += 1

        if frames_since_last >= args.step_size:
            scores = run_inference(model, shifu_buf, student_buf, device,
                                   global_mean, global_std)
            filled = min(shifu_buf.count, student_buf.count)
            print_scores(i, filled, scores)
            frames_since_last = 0

    if frames_since_last > 0:
        scores = run_inference(model, shifu_buf, student_buf, device,
                               global_mean, global_std)
        filled = min(shifu_buf.count, student_buf.count)
        print(f"\n  [Final window]")
        print_scores(total_frames - 1, filled, scores)


if __name__ == "__main__":
    main()
