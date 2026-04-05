# Open-Set Action Quality Assessment (AQA)

A Multi-Branch Siamese Network that compares two streams of 8-IMU motion
data (expert "shifu" vs student) and outputs per-limb quality scores. The
model uses **contrastive learning with L2 distance scoring**, making it
**open-set** -- it generalises to movements never seen during training.

---

## Project Structure

```
AI/
├── recordings/              # IMU data
│   ├── labels.json          # Metadata: movement, role, score, duration
│   └── *.csv                # ~192 CSV files (shifu + student recordings)
├── checkpoints/
│   └── best_model.pt        # Saved model weights (created by training)
├── aqa_model.py             # Siamese network architecture (arm/leg extractors)
├── aqa_dataset.py           # Dataset, pairing, normalisation, augmentation
├── train.py                 # Training loop, losses, memory bank
├── infer.py                 # File-based inference (CSV replay)
├── live_infer.py            # Live MQTT inference (real-time streaming)
├── eval.py                  # Batch evaluation script
├── train_dtw.py             # Alternative DTW + classical ML approach
├── ultra96.py               # FPGA acceleration integration
└── README.md
```

## Prerequisites

```
pip install torch pandas numpy
# For live inference:
pip install paho-mqtt
```

---

## Data Format

### CSV Files

Each CSV has a `timestamp_ms` column followed by **48 sensor columns**
(4 limbs x 2 segments x 6 axes: ax, ay, az, gx, gy, gz):

| Columns 1-12 | Columns 13-24 | Columns 25-36 | Columns 37-48 |
|---|---|---|---|
| Left Arm (upper + forearm) | Right Arm (upper + forearm) | Left Leg (thigh + shin) | Right Leg (thigh + shin) |

Each limb has 2 IMU nodes (e.g. upper arm + forearm), each providing
6 channels (ax, ay, az, gx, gy, gz) = 12 channels per limb, 48 total.

### labels.json

```json
{
  "kick_shifu_20260328_140249.csv": {
    "movement": "kick",
    "role": "shifu",
    "score": null,
    "duration": 5.1
  },
  "kick_student_20260328_140303.csv": {
    "movement": "kick",
    "role": "student",
    "score": 1.0,
    "duration": 5.2
  }
}
```

- **role**: `"shifu"` (expert reference) or `"student"` (person being assessed)
- **score**: `0.1` (poor), `0.5` (moderate), or `1.0` (expert-level) for students; `null` for shifu
- **movement**: one of 5 training movements (wave_hands, push, kick, block, hug)

---

## Architecture Overview

```
                ┌─────────────────────┐     ┌─────────────────────┐
                │   arm_extractor     │     │   leg_extractor     │
                │  Conv1d(k=7) stem   │     │  Conv1d(k=3) stem   │
                │  2x ResBlock1D      │     │  2x ResBlock1D      │
                │  Bi-LSTM            │     │  Bi-LSTM            │
                │  Recency-Weighted   │     │  Recency-Weighted   │
                │  Average Pooling    │     │  Average Pooling    │
                └────────┬────────────┘     └────────┬────────────┘
                         │                           │
         ┌───────────────┼───────────────┐  ┌────────┼────────────┐
         │               │               │  │        │            │
    shifu_LA      shifu_RA         student_LA  shifu_LL      shifu_RL
    student_LA    student_RA       student_RA  student_LL    student_RL
         │               │               │  │        │            │
         ▼               ▼               ▼  ▼        ▼            ▼
    Per-limb L2 distance: dist_i = ||shifu_emb_i - student_emb_i||
         │
         ▼
    Per-group temperature scaling:
        score_arm_i = exp(-dist_i^2 / tau_arm^2)
        score_leg_i = exp(-dist_i^2 / tau_leg^2)
         │
         ▼
    4 limb scores in [0, 1]
         │
         ▼
    Post-hoc aggregation (NOT learned):
        geo_mean = (s0 * s1 * s2 * s3)^(1/4)
        penalty  = exp(-1.5 * (1 - min(s_i))^2)
        overall  = geo_mean * penalty
```

### Key Design Decisions

- **Separate arm/leg extractors**: Arms and legs have fundamentally
  different motion profiles. The arm extractor uses a wider stem kernel
  (k=7) for sweeping upper-body motions; the leg extractor uses a
  narrower kernel (k=3) for sharp kick/step dynamics.

- **Recency-Weighted Average Pooling (RWAP)**: Instead of flat GAP,
  the LSTM output is pooled with exponential weights favouring recent
  frames (decay=0.03, oldest frame at ~2% weight of newest). This
  allows fast score recovery when the student returns to good form --
  old bad frames fade out exponentially rather than persisting at full
  weight until evicted from the 128-frame sliding window.

- **Per-group learned temperatures**: tau_arm and tau_leg independently
  control how quickly scores decay with embedding distance. Arms and
  legs can have different distance scales in embedding space.

- **Gentle worst-limb penalty** (sharpness=1.5): The exponential
  penalty on the worst limb is moderate rather than aggressive, so
  overall scores recover smoothly as limb scores improve.

- **No classification head**: The model never learns to identify specific
  movements. It only learns similarity, which transfers to unseen movements.

- **No temporal alignment**: Movement speed is part of quality. A slow
  movement produces different embeddings than a fast one, which is
  correctly scored as lower quality.

**Parameters**: ~35K (lightweight for edge deployment)

---

## Training

### Quick Start

```bash
python train.py
```

Trains for 100 epochs with all defaults. Best model (by validation loss)
saved to `checkpoints/best_model.pt`.

### Full Command

```bash
python train.py \
  --recordings_dir recordings \
  --save_dir checkpoints \
  --epochs 100 \
  --batch_size 8 \
  --max_seq_len 128 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --contrastive_margin 1.5 \
  --score_loss_weight 0.3 \
  --bank_loss_weight 0.1 \
  --bank_capacity 256 \
  --negative_pair_prob 0.35 \
  --pseudo_shifu_ratio 0.5 \
  --val_split 0.2 \
  --patience 15 \
  --seed 42
```

### Training Parameters

| Parameter | Default | Description |
|---|---|---|
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 8 | Samples per batch |
| `--max_seq_len` | 128 | Fixed sequence length (pad/truncate) |
| `--lr` | 1e-3 | AdamW learning rate |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--grad_clip` | 1.0 | Max gradient norm (stabilises LSTM) |
| `--contrastive_margin` | 1.5 | Target distance for score=0 pairs and mismatch push threshold |
| `--contrastive_floor` | 0.5 | Minimum target distance for score=1.0 pairs (natural variation floor) |
| `--score_loss_weight` | 1.0 | Weight for supervised score MSE calibration loss |
| `--bank_loss_weight` | 0.1 | Weight for memory bank cross-contrastive loss |
| `--bank_capacity` | 256 | Number of cached embeddings in the memory bank |
| `--negative_pair_prob` | 0.35 | Fraction of training pairs replaced with cross-movement mismatches |
| `--pseudo_shifu_ratio` | 0.5 | Fraction of score=1.0 students promoted to pseudo-shifu references |
| `--val_split` | 0.2 | Fraction of data reserved for validation |
| `--patience` | 15 | Epochs without improvement before LR is halved |
| `--seed` | 42 | Random seed for reproducibility |

### Multi-Task Loss Function

```
L_total = L_contrastive + 1.0 * L_score_mse + 0.1 * L_bank
```

1. **L_contrastive** (primary signal): Per-limb target-distance contrastive
   loss on raw embeddings.
   - *Matched pairs*: `loss = (dist - target_dist)^2` where `target_dist = floor + (margin - floor) * (1 - score)`
     - score=1.0 -> target dist = floor (0.5) — acknowledges natural variation
     - score=0.5 -> moderate distance (midpoint between floor and margin)
     - score=0.1 -> push far apart (target near margin)
   - *Mismatched pairs*: `loss = max(0, margin - dist)^2`
     - Pushes cross-movement embeddings apart until distance >= margin

2. **L_score_mse** (supervised calibration): MSE between model limb scores
   and ground-truth, applied only to matched pairs. Excludes mismatches to
   avoid biasing toward low scores.

3. **L_bank** (memory bank negatives): Cross-contrastive push-apart loss
   between current student embeddings and cached shifu embeddings from
   recent batches. Provides ~256 additional negative comparisons per step.

### Data Augmentation (Training Only)

Applied independently per limb and per role (shifu/student):

| Augmentation | Probability | Description |
|---|---|---|
| Jitter | 50% | Gaussian noise (std=0.02) |
| Scaling | 50% | Uniform multiplier [0.9, 1.1] |
| Random rotation | 40% | Small SO(3) rotation (up to 10 deg) on accel/gyro triples |
| Channel dropout | 30% | Zero out individual channels (15% per-channel) |
| Temporal cutout | 30% | Zero out contiguous window (up to 15% of sequence) |

### Training Output

```
  Ep    TrLoss    TrCont     TrScr   TrBank    VlLoss    VlCont     VlScr     tA     tL          LR
-------------------------------------------------------------------------------------------------------
   1    0.4521    0.3812    0.2363    0.0001    0.3815    0.3102    0.2377  1.000  1.000    1.00e-03 *
   2    0.3967    0.3254    0.2378    0.0012    0.3641    0.2953    0.2293  1.012  0.987    1.00e-03 *
```

- `*` marks epochs where a new best model was saved
- `tA`/`tL`: learned arm/leg temperatures
- Best model selected by **validation loss**

### Checkpoint Contents

```python
{
    "epoch": 42,
    "model_state_dict": ...,       # arm_extractor, leg_extractor, temperatures
    "optimizer_state_dict": ...,
    "val_loss": 0.1234,
    "args": { ... },               # all training hyperparameters
    "global_mean": np.array(48,),  # normalisation stats
    "global_std": np.array(48,),
}
```

---

## Inference

### File Mode (CSV Replay)

Simulates streaming by replaying two saved CSVs through sliding-window
buffers:

```bash
python infer.py \
  --shifu_csv recordings/kick_shifu_20260328_140249.csv \
  --student_csv recordings/kick_student_20260328_140303.csv \
  --step_size 10
```

### Live Mode (MQTT Streaming)

Receives real-time IMU data from shifu/student via MQTT:

```bash
python live_infer.py
python live_infer.py --step_size 20
python live_infer.py --save_streams  # record to CSV for later replay
```

MQTT topics:
- Subscribe: `sensor/+/aggregated` (+ = "shifu" or "student")
- Publish: `inference/result` (overall + per-node scores)
- Publish: `inference/student/<limb>` (per-limb breakdown)

### Inference Parameters

| Parameter | Default | Description |
|---|---|---|
| `--shifu_csv` | (required) | Shifu CSV path (file mode) |
| `--student_csv` | (required) | Student CSV path (file mode) |
| `--checkpoint` | `checkpoints/best_model.pt` | Path to trained model |
| `--step_size` | 10 | Frames between inference triggers (~0.5s at 20Hz) |
| `--device` | auto | `cpu`, `cuda`, or `auto` |

### Inference Output

```
  Frame    9 | buf  10/128 | Overall: 0.485 | LA: 0.999  RA: 0.998  LL: 0.999  RL: 0.999
  Frame   19 | buf  20/128 | Overall: 0.485 | LA: 0.999  RA: 0.997  LL: 0.998  RL: 0.999
  ...
  [Final window]
  Frame   83 | buf  81/128 | Overall: 0.483 | LA: 0.998  RA: 0.960  LL: 0.968  RL: 0.998
```

- **buf N/128**: real frames in the buffer (rest is zero-padded)
- **Overall**: geometric mean + worst-limb penalty aggregation
- **LA/RA/LL/RL**: per-limb L2-distance-based scores in [0, 1]

### Sliding Window Behaviour

- Buffer capacity: **128 frames** (matches training `max_seq_len`)
- Early windows (< 128 frames): **left-zero-padded** to 128
- Once full: oldest frame evicted on each push (FIFO ring buffer)
- Unequal CSV lengths: shorter stream stops pushing; buffer retains last frames

---

## Batch Evaluation

```bash
python eval.py                                    # matched pairs only
python eval.py --mismatches                       # include cross-movement pairs
python eval.py --mismatches --csv results.csv     # export to CSV
```

- **Matched pairs**: first shifu of each movement vs all students of same movement
- **Mismatched pairs**: shifu vs best student of other movements (should score ~0)

---

## Open-Set Design

The model generalises to movements never seen during training because:

1. **L2 distance scoring** is geometry-based, not movement-specific. The
   model only measures how close two embeddings are, regardless of what
   movement they represent.

2. **No classification head**. The model never learns to identify movements.
   It only learns similarity in embedding space, which transfers to any
   movement type.

3. **Separate arm/leg extractors** learn body-region-specific motion features
   rather than whole-body movement signatures.

4. **Per-group temperatures** let the model calibrate distance sensitivity
   independently for upper and lower body.

5. **Synthetic mismatches** (35% of training) teach the model that
   cross-movement comparisons should score near zero, establishing a
   strong open-set boundary in embedding space.

6. **Memory bank** provides additional cross-batch negatives during training,
   strengthening the push-apart signal between dissimilar embeddings.

7. **Speed is quality**: no temporal alignment is used, so a slow performance
   of a movement naturally produces a different embedding and lower score.
