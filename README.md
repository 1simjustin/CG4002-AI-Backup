# Open-Set Action Quality Assessment — Model Implementation

A Multi-Branch Siamese Network that compares two streams of 8-IMU motion
data (expert "shifu" vs student) and produces per-limb quality scores. The
model uses contrastive learning with L2 distance scoring, making it
open-set — it generalises to movements never seen during training.

This document details the design choices behind every component of the AI
pipeline: data handling, model architecture, training, inference, live
deployment, and evaluation.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Pipeline — aqa_dataset.py](#data-pipeline--aqa_datasetpy)
3. [Model Architecture — aqa_model.py](#model-architecture--aqa_modelpy)
4. [Training — train.py](#training--trainpy)
5. [Inference — infer.py](#inference--inferpy)
6. [Live Inference — live_infer.py](#live-inference--live_inferpy)
7. [Evaluation — eval.py](#evaluation--evalpy)

---

## System Overview

The system assesses how well a student performs a martial arts movement
compared to an expert ("shifu"). Eight IMU sensors (4 limbs x 2 segments
each) capture accelerometer and gyroscope data at each timestep, producing
48 channels of motion data. The AI compares the student's motion pattern
against the expert's and outputs per-limb quality scores in [0, 1].

```
8 IMU sensors (48 channels)
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  Normalisation: per-sample mean subtraction + z-score  │
└────────────────┬───────────────────────────────────────┘
                 │
         ┌───────┴───────┐
         ▼               ▼
   ┌──────────┐    ┌──────────┐
   │   Arm    │    │   Leg    │
   │Extractor │    │Extractor │
   │ (k=7)    │    │ (k=3)    │
   └────┬─────┘    └────┬─────┘
        │               │
        ▼               ▼
   4 limb embeddings (shifu + student)
        │
        ▼
   Per-limb L2 distance → exp(-d²/τ²) → 4 scores
        │
        ▼
   Overall = geometric mean × worst-limb penalty
```

### Why Open-Set?

The model never learns to classify or identify specific movements. It only
learns to measure similarity between two motion streams in embedding space.
This means it generalises to any movement type — including ones never seen
during training — because similarity measurement is movement-agnostic.

This is a deliberate design choice driven by the deployment context: the
system must assess arbitrary martial arts forms without retraining every
time a new movement is introduced.

---

## Data Pipeline — aqa_dataset.py

### Data Format

Each recording is a CSV with a `timestamp_ms` column followed by 48 sensor
columns (4 limbs x 2 segments x 6 axes):

| Columns 0–11 | Columns 12–23 | Columns 24–35 | Columns 36–47 |
|---|---|---|---|
| Left Arm | Right Arm | Left Leg | Right Leg |
| (upper arm + forearm) | (upper arm + forearm) | (thigh + shin) | (thigh + shin) |

Each limb has 12 channels: 2 IMU segments x (ax, ay, az, gx, gy, gz).

Metadata in `labels.json` maps each CSV to its movement type, role
(shifu/student), and quality score (0.1 = poor, 0.5 = moderate, 1.0 =
expert-level). Shifu recordings have score `null`.

### Pairing Strategy

Training requires shifu-student pairs. The pairing logic makes two
non-obvious choices:

**1. Pseudo-shifu references.** A configurable fraction (default 50%) of
score=1.0 student recordings are promoted to "pseudo-shifu" status and
used as reference recordings alongside actual shifu recordings. Each
reference is then paired with every student of the same movement.

*Why:* If only the single shifu recording per movement is used as the
reference, the model risks memorising session-specific artifacts of that
one recording (sensor drift, exact placement, ambient noise) rather than
learning the movement's actual motion pattern. Pseudo-shifu creates
cross-session reference diversity — the model sees different physical
people performing the same movement as reference, forcing it to learn
the movement pattern itself rather than a specific session's signature.

**2. Synthetic mismatches.** 35% of training pairs are replaced at
load-time with cross-movement mismatches: a shifu from one movement
paired with a student from a different movement, score forced to 0.0.

*Why:* Without explicit negative signal, the model has no reason to push
embeddings of different movements apart. It could learn a trivially
collapsed embedding where everything maps to the same point and scores
are uniformly high. Synthetic mismatches teach the model that completely
different movements should score near zero, establishing an open-set
boundary in embedding space. The 35% ratio was chosen empirically — too
low and the boundary is weak; too high and the model sees too few real
matched pairs to calibrate its score range.

### Two-Stage Normalisation

Raw IMU data has two problems for neural network consumption:

1. **Orientation offset.** The absolute sensor values encode gravity
   direction and the sensor's physical mounting angle. Two identical
   movements performed with slightly different sensor placement produce
   wildly different raw values. The model should see *motion deltas*, not
   absolute orientation.

2. **Scale mismatch.** Accelerometer values (in g or m/s^2) and gyroscope
   values (in deg/s or rad/s) live on different numeric scales. Without
   normalisation, whichever modality has larger absolute values dominates
   the learned features.

The solution is a two-stage normalisation applied consistently across
training, file inference, and live inference:

**Stage 1 — Per-sample mean subtraction.** For each recording (or
sliding window at inference time), subtract the per-channel mean computed
over non-padding timesteps. This removes the gravity/orientation offset so
the model sees deviations from the session's baseline rather than absolute
sensor values. Non-padding masking is important: during early inference
when the sliding window is mostly zero-padded, including the zeros in the
mean computation would corrupt the subtraction for the real frames.

**Stage 2 — Global z-score.** Apply (x - global_mean) / global_std using
statistics pre-computed from the entire training set (after Stage 1). This
puts accelerometer and gyroscope channels on a comparable scale. The
global statistics are saved into the model checkpoint so that inference
uses the exact same normalisation as training — no need for a separate
calibration step.

Constant channels (std < 1e-8) have their std clamped to 1.0 to avoid
division by zero. This is a safety measure for channels that might be
constant across the training set (e.g. a sensor axis that never varies).

### Data Augmentation

Five augmentations are applied independently per limb and per role
(shifu/student), each with its own activation probability. The design
philosophy is to simulate real-world variability rather than generic image
augmentation patterns:

| Augmentation | Probability | Parameters | Rationale |
|---|---|---|---|
| Jitter | 50% | Gaussian noise, std=0.02 | Simulates sensor noise. The std was tuned to be below the actual noise floor of the IMU so it adds variability without corrupting signal. |
| Scaling | 50% | Uniform multiplier [0.9, 1.1] | Teaches amplitude invariance. A movement performed with slightly more or less force should not change the quality score. |
| Random rotation | 40% | SO(3) rotation up to 10 deg on accel/gyro triples | Simulates slight sensor misplacement between sessions. The same rotation is applied to both accel and gyro of each segment to maintain physical consistency. Principal-axis rotation (not arbitrary Euler) keeps it computationally cheap. |
| Channel dropout | 30% | 15% per-channel zero probability | Simulates partial sensor noise or intermittent connection drops. Forces the model to use redundant information across channels rather than relying on any single channel. |
| Temporal cutout | 30% | Zero out up to 15% of sequence contiguously | Forces the model to not rely on any single phase of the movement. If the middle of a kick is masked, the model must still assess quality from the windup and follow-through. |

Augmentations are applied to both shifu and student recordings
independently. Augmenting the shifu is important — it prevents the model
from treating the reference as a fixed template and instead forces it to
learn the underlying motion pattern.

---

## Model Architecture — aqa_model.py

### Why Siamese?

A Siamese architecture processes the reference and student through the
same feature extractor with shared weights, then compares their
embeddings. Alternatives considered and rejected:

- **Direct regression on student-only input:** This would require the
  model to implicitly memorise what "correct" looks like for every
  movement. It cannot generalise to unseen movements because the
  reference is baked into the weights rather than provided as input.

- **Classification head:** A discrete good/bad classifier cannot provide
  fine-grained quality scores and cannot handle open-set movements
  without retraining the output layer.

- **Attention-based cross-stream fusion:** While more expressive, this
  dramatically increases parameter count and makes it harder to reason
  about what the model has learned. L2 distance in embedding space is
  interpretable — closer means more similar — and constrains the model
  to learn geometrically meaningful representations.

### Separate Arm and Leg Extractors

The model uses two independent `FeatureExtractor` instances with different
stem kernel sizes:

```
arm_extractor: stem Conv1d(12, 32, kernel_size=7) → 2x ResBlock1D → Bi-LSTM → Pooling
leg_extractor: stem Conv1d(12, 32, kernel_size=3) → 2x ResBlock1D → Bi-LSTM → Pooling
```

*Why separate extractors?* Arms and legs have fundamentally different
motion dynamics. Upper body movements (waves, pushes, hugs) tend to be
broader and slower — a wider kernel (k=7) captures these sweeping temporal
patterns. Lower body movements (kicks, stances, steps) tend to be sharper
and faster — a narrower kernel (k=3) captures these rapid dynamics without
over-smoothing. Sharing a single extractor would force a compromise
receptive field that is suboptimal for both.

*Why not four separate extractors (one per limb)?* With only ~190
training recordings, four extractors would quadruple the parameter count
(~35K to ~140K) while quartering the effective training data per
extractor. Two groups (arm/leg) is the sweet spot between specialisation
and data efficiency.

Both extractors share the same topology — the only difference is the
stem kernel size. This means the ResBlock and LSTM layers are identically
structured but learn independent weights.

### FeatureExtractor Pipeline

Each extractor processes one limb at a time:

**Input:** (batch, 12, seq_len) — 12 channels from 2 IMU segments.

**1. Conv1d stem.** Projects 12 input channels to 32 feature channels.
The stem kernel size (7 for arms, 3 for legs) sets the initial receptive
field. Same-padding preserves temporal resolution.

**2. 2x ResBlock1D.** Pre-activation residual blocks
(BN → ReLU → Conv → BN → ReLU → Dropout → Conv + skip). The residual
connection (x + block(x)) ensures gradient flow even with the 0.3 dropout,
which is relatively aggressive for a small model. Pre-activation
(BN-ReLU before Conv) rather than post-activation gives better gradient
flow in practice.

**3. Bi-LSTM.** A single-layer bidirectional LSTM (hidden=8, output
dim=16). Captures temporal dependencies in both directions — the quality
of a follow-through depends on the preceding motion, and vice versa.
The LSTM is deliberately small (8 hidden units) to prevent overfitting
on the limited training data.

**4. Dual-mode pooling.** This is one of the most important design
choices:

- **Training: Global Average Pooling (GAP).** All frames contribute
  equally to the embedding. This produces stable, low-variance
  embeddings that are ideal for contrastive loss — the loss landscape is
  smooth and the model can learn well-calibrated distance relationships.

- **Inference: Recency-Weighted Average Pooling (RWAP).** An exponential
  weighting scheme where recent frames have more influence:
  ```
  w_t = exp(decay * (t - (T-1)))
  ```
  With decay=0.1 and T=128, the oldest frame has ~0.0003% the weight of
  the newest (exp(-0.1 * 127) ≈ 0.000003).

  *Why not use RWAP during training?* RWAP makes the embedding sensitive
  to which frames are at the end of the window, introducing variance that
  fights the contrastive loss. GAP during training gives the model a
  stable target to learn against. RWAP at inference gives the live system
  fast score recovery — when a student returns to good form, the bad
  frames from earlier exponentially fade out rather than persisting at
  full weight until evicted from the sliding window.

  *Why not use attention pooling?* Attention adds learnable parameters
  and makes pooling data-dependent. RWAP is a fixed, interpretable
  scheme that doesn't need to be learned and doesn't risk overfitting.
  The exponential decay directly encodes the desired behaviour: "recent
  frames matter more for real-time feedback."

### Per-Group Learned Temperatures

L2 distances are mapped to [0, 1] scores via:

```
score = exp(-dist² / τ²)
```

where τ (tau) is a learned parameter, stored as log(τ) for unconstrained
optimisation. There are two independent temperatures: τ_arm and τ_leg.

*Why per-group temperatures?* Arms and legs operate in different regions
of embedding space with different distance scales. A push movement might
produce arm embeddings that are naturally closer together than leg
embeddings for a kick. A single shared temperature would force one body
region to compromise its score calibration for the other. Two temperatures
let each region independently control its sensitivity.

*Why not per-limb temperatures (4 total)?* Four temperatures on ~190
recordings risks overfitting. Two groups provide enough flexibility
while keeping the parameter count minimal.

*Why exp(-d²/τ²) instead of a learned MLP?* The Gaussian RBF kernel is
monotonic (further = lower score), bounded in [0, 1], and has a single
interpretable parameter. An MLP could learn non-monotonic score
functions, which would be physically meaningless — more similar should
always mean higher quality.

### Overall Score Aggregation

The overall score is NOT a learned parameter. It is a fixed statistical
formula applied post-hoc:

```
geo_mean = (s0 * s1 * s2 * s3) ^ (1/4)
worst_penalty = exp(-1.1 * (1 - min(s_i))²)
overall = geo_mean * worst_penalty
```

*Why geometric mean instead of arithmetic?* The geometric mean penalises
low outliers more naturally. With arithmetic mean, [0.1, 0.9, 0.9, 0.9]
averages to 0.7 — misleadingly high for a student with one badly
performed limb. Geometric mean gives 0.547, better reflecting the weak
link.

*Why the additional worst-limb penalty?* Even the geometric mean can be
too generous when one limb is very bad. The exponential penalty provides
a targeted dropoff:

| Limb Scores | Geo Mean | Penalty | Overall |
|---|---|---|---|
| [0.90, 0.90, 0.90, 0.90] | 0.900 | 0.989 | 0.890 |
| [0.50, 0.50, 0.50, 0.50] | 0.500 | 0.757 | 0.379 |
| [0.10, 0.90, 0.90, 0.90] | 0.547 | 0.418 | 0.229 |
| [0.90, 0.90, 0.90, 0.50] | 0.796 | 0.757 | 0.603 |

Sharpness=1.1 was tuned to be gentle enough that scores recover
smoothly during live streaming as limb scores improve, but sharp enough
that one catastrophically bad limb visibly drags down the overall score.

*Why not learn the aggregation?* Learning it would couple the overall
score to the training data distribution, hurting open-set generalisation.
A fixed formula means the overall score behaves predictably on any
movement type, including ones never seen during training.

### Model Size

The full model has ~35K parameters. This is deliberately small:

- With ~190 training recordings, a larger model would overfit
- The model runs on an Ultra96 FPGA in production — small size enables
  hardware synthesis
- Inference latency matters for real-time feedback — fewer operations
  means faster response

---

## Training — train.py

### Multi-Task Loss

The total loss combines three components:

```
L_total = L_contrastive + 1.0 * L_score_mse + 0.1 * L_bank
```

The weights (1.0 and 0.1) were tuned empirically. The contrastive loss
is the primary learning signal; the other two are calibration aids.

#### L_contrastive — Target-Distance Contrastive Loss

This is the core loss. For each of the 4 limbs independently:

**Matched pairs (same movement):**
```
target_dist = floor + (margin - floor) * (1 - score)
loss = (dist - target_dist)²
```

The target distance is a linear function of the quality score:
- score=1.0 → target = floor (0.5) — close but not zero
- score=0.5 → target = midpoint (1.0)
- score=0.1 → target = near margin (1.4)

*Why a floor instead of pulling score=1.0 pairs to dist=0?* A shifu and
a perfect student are different people performing different physical
instances of the same movement. Demanding zero distance between them is
unachievable — there will always be natural variation in sensor placement,
body proportions, and movement execution. Without the floor, the model
wastes capacity trying to collapse distinct recordings to a single point,
leading to an undertrained temperature (because the model never achieves
the zero-distance target, the gradient signal on tau is noisy) and
compressed score ranges. The floor of 0.5 acknowledges natural variation
and gives the model a reachable target.

*Why continuous target distances instead of binary (same/different)?*
A binary contrastive loss (e.g., score=1.0 → pull, score=0.1 → push)
gives no gradient signal for the quality *degree*. A student with
score=0.5 gets the same loss as score=0.1, so the model cannot
distinguish moderate from poor quality. Continuous targets let every
score level contribute proportional gradient.

**Mismatched pairs (different movements):**
```
loss = max(0, margin - dist)²
```

Simple hinge loss: push embeddings apart until distance >= margin. Once
they are far enough apart, the loss is zero and no further gradient is
wasted pushing them to infinity. The margin (default 1.5) defines the
radius of the open-set decision boundary.

#### L_score_mse — Supervised Score Calibration

```
L_score = MSE(limb_scores, target) for matched pairs only
```

Each of the 4 limb scores is independently pushed toward the ground-truth
target. This provides direct supervision on the output score range,
preventing drift that contrastive loss alone might cause (contrastive loss
shapes the embedding geometry, but the temperature mapping from distance
to score could drift if not anchored).

*Why exclude mismatched pairs?* Mismatched pairs have target 0.0 and
comprise 35% of the training data. Including them in the MSE loss
creates a massive bias toward low scores — the loss would be minimised
by predicting low scores for everything. Excluding them lets the MSE
loss focus on calibrating the meaningful score range (0.1 to 1.0).

#### L_bank — Memory Bank Cross-Contrastive Loss

The memory bank is a FIFO buffer of 256 detached embeddings from recent
batches. At each training step, pairwise L2 distances are computed
between the current batch's student embeddings and all banked shifu
embeddings. All cross-batch pairs are treated as negatives (push apart).

*Why a memory bank?* Contrastive learning benefits from seeing many
negatives per step. With batch_size=8, each sample only sees 7 other
samples as potential negatives. The memory bank provides ~256 additional
negative comparisons per step without increasing GPU memory proportionally
(banked embeddings are detached — no gradient computation).

*Why only push-apart loss for bank pairs?* We don't know the ground-truth
score relationship between a current-batch student and a banked shifu from
a different batch. Treating them all as negatives is conservative but
safe — even if some happen to be the same movement, pushing them slightly
apart is a much smaller error than pulling genuinely different movements
together.

*Why weight 0.1?* The bank loss is a regulariser, not a primary signal.
Too high a weight would dominate the contrastive loss and collapse the
embedding space (everything pushed maximally apart). 0.1 provides gentle
additional separation without disrupting the matched-pair geometry.

### Optimiser and Scheduler

**AdamW** with weight decay 1e-4 and gradient clipping (norm <= 1.0).
Gradient clipping is essential because the Bi-LSTM can produce exploding
gradients, especially early in training when the embedding space is
unstructured and distances are large.

**ReduceLROnPlateau** halves the learning rate if validation loss stalls
for 15 epochs. This is conservative — the model trains for 100 epochs
total, so the LR typically drops 1-2 times. More aggressive scheduling
(e.g., cosine annealing) was tested but found to underperform on this
small dataset, likely because the model needs sustained gradient signal
to calibrate the temperatures.

### Validation Strategy

The dataset is split 80/20 by shuffled indices (seeded for
reproducibility). The validation set uses the same normalisation
statistics as the training set (computed from training data) but has
augmentation disabled and no synthetic mismatches. This means validation
measures the model's ability to score clean, unaugmented pairs — the
same setting it encounters at inference time.

The memory bank loss is excluded from validation because the bank is
populated during training only.

### Checkpoint Contents

The best model (lowest validation loss) is saved with:

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

Embedding the normalisation statistics in the checkpoint is a deliberate
choice — it ensures that inference always uses exactly the same
normalisation as training, even if the training data changes later.

---

## Inference — infer.py

### SlidingWindowBuffer

A fixed-capacity ring buffer (FIFO deque, capacity 128) that holds the
most recent timesteps of sensor data.

*Why 128 frames?* This matches `max_seq_len` from training. The model
was trained on 128-frame windows, so inference must present the same
temporal context. At 20Hz sensor rate, 128 frames ≈ 6.4 seconds —
sufficient to capture most martial arts movements.

*Why left-zero-padding instead of right?* When the buffer has fewer than
128 frames (early in a session), the real data is placed at the end
(most recent) and the beginning is zero-padded. This is consistent with
the model's RWAP pooling which weights recent frames most heavily — the
zero-padded early frames have minimal influence on the embedding.

### Missing Limb Detection

Before normalisation, each limb's channels are checked for all-zero data
across the entire window. If both segments of a limb are zero (meaning
no sensor data was received), that limb gets a sentinel score of -1.0
and is excluded from the overall score computation.

*Why detect before normalisation?* After z-score normalisation,
zero-channel data becomes (0 - mean) / std, which is non-zero. Detection
must happen on raw data.

*Why not impute missing data?* Imputing sensor data from other limbs
would inject false similarity between the student and shifu, producing
artificially high scores. A sentinel value is honest — it tells
downstream consumers that the score is unavailable rather than
fabricating one.

### Normalisation at Inference

The same two-stage normalisation from training is applied:

1. Per-sample mean subtraction (over non-padding rows of the window)
2. Global z-score using the statistics from the checkpoint

The non-padding mask is reapplied after z-scoring to re-zero the padded
frames. Without this, the z-score would transform padding zeros into
non-zero values that the model interprets as signal.

### File-Based Replay

`infer.py` also provides a standalone mode that loads two CSVs and
simulates streaming by pushing frames one at a time into
SlidingWindowBuffers, triggering inference every `step_size` frames
(default 10). This is useful for debugging and offline evaluation without
needing an MQTT connection.

---

## Live Inference — live_infer.py

### MQTT Integration

The live system subscribes to `sensor/+/aggregated` (wildcard matching
both `sensor/shifu/aggregated` and `sensor/student/aggregated`). Each
MQTT message contains one timestep of 8-node IMU data as JSON.

*Why QoS 0?* At 20Hz sensor rate, the system receives ~40 messages per
second (shifu + student). QoS 1 or 2 would add acknowledgement overhead
for data that is immediately stale — a dropped frame is better handled
by the sliding window's buffering than by retransmission.

*Why TLS with `tls_insecure_set(True)`?* The CA cert provides transport
encryption, but the `insecure` flag skips hostname verification. This is
a pragmatic choice for the deployment environment where the broker IP
may not match the certificate's CN/SAN. Encryption is preserved; only
hostname validation is relaxed.

### Frame Parsing

`parse_frame()` converts the nested JSON structure into a flat (48,)
numpy array matching the training CSV column order:

```json
{
  "nodes": {
    "left_arm": {
      "left_arm_upper_arm": {"accel": [x,y,z], "gyro": [x,y,z]},
      "left_arm_forearm": {"accel": [x,y,z], "gyro": [x,y,z]}
    },
    ...
  }
}
```

The `NODE_ORDER` list defines the canonical ordering of the 8 IMU nodes.
This order must match the CSV column layout used during training — if
the order is wrong, the model receives scrambled limb data and produces
meaningless scores.

Missing nodes (absent from the JSON) are silently filled with zeros. This
is detected downstream by the missing limb detection in `run_inference()`.

### Inference Cadence

Inference is triggered every `step_size` student frames (default 10) after
both buffers have accumulated at least `MIN_FRAMES` (20) frames. Shifu
frames are buffered silently without triggering inference.

*Why trigger on student frames only?* The shifu stream is a reference
that typically starts first and runs continuously. The student stream
represents the active performance being assessed. Triggering on student
frames ensures inference runs at a cadence proportional to the student's
actual activity.

*Why MIN_FRAMES=20?* With fewer than 20 real frames in a 128-frame
buffer, the window is >85% zero-padding. The model's embeddings at this
point are dominated by the zero-padding rather than real motion data,
producing unreliable scores. 20 frames (~1 second at 20Hz) provides
enough real signal for a meaningful first score.

### Result Publishing

Results are published to two topic patterns:

1. `inference/result` — Overall score + per-node scores (8 nodes, where
   both segments of a limb share the same score since the model operates
   per-limb)
2. `inference/student/<limb>` — Per-limb breakdown (4 topics)

The per-node expansion (4 limb scores → 8 node scores) exists because
downstream consumers (dashboard, FPGA) may expect per-sensor-node
granularity. The model cannot distinguish quality at the segment level
(upper arm vs forearm), so both segments receive the limb's score.

### Stream Recording

The `--save_streams` flag enables `StreamRecorder`, which writes every
incoming frame to timestamped CSVs in `live_recordings/`. Timestamps are
relative to session start (milliseconds). This creates recordings in the
exact same format as the training data, enabling later offline replay
via `infer.py` or incorporation into the training set.

---

## Evaluation — eval.py

### Test Pair Construction

The evaluation script constructs two types of test pairs:

**Matched pairs:** For each movement, the first shifu recording is
paired with every student of that movement. This tests the model's
ability to assign appropriate scores across the quality range (0.1, 0.5,
1.0). Expected behaviour:
- score=1.0 students → model scores ~0.8+
- score=0.5 students → model scores ~0.4–0.7
- score=0.1 students → model scores ~0.0–0.2

**Mismatched pairs (optional, `--mismatches`):** Each shifu is paired
with the first score=1.0 student of every *other* movement. These should
all produce near-zero overall scores regardless of quality. If mismatch
scores are high, the model is not discriminating between movement types
and the open-set boundary has failed.

*Why use the best (score=1.0) student for mismatches?* This is the
hardest test case. If even an expertly performed *wrong* movement gets a
low score, then the model is truly comparing movement patterns rather
than just motion amplitude or general activity level.

### Full-Buffer Evaluation

Unlike the streaming inference in `infer.py` (which reports at intervals),
`eval.py` fills both buffers completely before running a single inference
pass. This gives the model the maximum 128 frames of context, producing
the most reliable score for each pair. It represents the model's
"best-case" assessment — the score you would get if you let the entire
movement play out before judging.

### Output

Results are printed as a formatted table grouped by movement, with an
optional CSV export for further analysis:

```
Test                           GT   Over     LA     RA     LL     RL  Student CSV
-----------------------------------------------------------------------------------------------
kick 1.0                      1.0  0.891  0.912  0.876  0.898  0.889  kick_student_1.csv
kick 0.5                      0.5  0.431  0.512  0.389  0.445  0.401  kick_student_2.csv
kick 0.1                      0.1  0.089  0.102  0.078  0.091  0.085  kick_student_3.csv

kick vs wave_hands             0.0  0.012  0.015  0.009  0.018  0.011  wave_student_1.csv
```

The side-by-side ground-truth and model score columns make it easy to
assess calibration — whether the model's output scale matches the
intended 0-to-1 quality range.
