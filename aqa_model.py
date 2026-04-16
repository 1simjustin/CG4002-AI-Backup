"""
Multi-Branch Siamese Network for open-set Action Quality Assessment.

The model compares two streams of 8-IMU data (reference "shifu" vs
"student") and produces per-limb quality scores in [0, 1]. It is
open-set: the model never learns to identify specific movements --
it only learns how similar two motion patterns are, so it generalises
to movements never seen during training.

Architecture
------------
Two specialised FeatureExtractors with independent weights:

    arm_extractor  (for left_arm, right_arm):
        Conv1d stem (kernel=7) -> 1x ResBlock1D -> Dilated Temporal Conv -> GAP
        Wider stem kernel captures broader, sweeping upper-body motions
        (e.g. wave, push, hug).

    leg_extractor  (for left_leg, right_leg):
        Conv1d stem (kernel=3) -> 1x ResBlock1D -> Dilated Temporal Conv -> GAP
        Narrower stem kernel is tuned for fast, sharp lower-body
        dynamics (e.g. kick, step, stance).

    Both extractors share the same topology but learn independent weights.

    The temporal aggregator is a stack of two dilated 1D convolutions
    (dilations 1 and 2, kernel 3) instead of a Bi-LSTM. Convs parallelise
    across time, which gives a large speedup on CPU inference where the
    sequential LSTM was the dominant latency cost. Receptive field is
    comparable to what the tiny Bi-LSTM effectively used.

    Input per limb:  (batch, 12, seq_len)
        12 = 2 segments x 6 axes (ax, ay, az, gx, gy, gz)
    Output per limb: (batch, embed_dim)
        embed_dim defaults to 16

Scoring pipeline:
    1. Extract 8 embeddings (4 shifu + 4 student) using the appropriate
       group extractor (arm or leg).
    2. Per-limb L2 distance: dist = ||shifu_emb - student_emb||
    3. Map to [0, 1] via per-group temperature:
           score_i = exp(-dist_i^2 / tau_group^2)
       tau_arm and tau_leg are independently learned parameters that
       calibrate how quickly scores decay with embedding distance.
       Different body regions can have different distance scales.
    4. Return 4 limb scores + raw embeddings (for contrastive loss).

Overall score aggregation (post-hoc, NOT learned):
    compute_overall_score() uses geometric mean + exponential worst-limb
    penalty. This is a fixed statistical formula applied only at inference.

Output dict:
    'limb_scores':     (batch, 4)        -- per-limb scores in [0, 1]
    'embeddings':      dict of 8 tensors -- each (batch, embed_dim)
    'temperature':     scalar            -- mean of tau_arm, tau_leg
    'temperature_arm': scalar            -- learned arm temperature
    'temperature_leg': scalar            -- learned leg temperature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -- 1D Residual Block ----------------------------------------------------

class ResBlock1D(nn.Module):
    """
    Pre-activation 1D residual block (BN -> ReLU -> Conv -> BN -> ReLU -> Conv).

    Maintains spatial dimensions via same-padding. The residual connection
    (x + block(x)) ensures gradient flow through deep stacks.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# -- Feature Extractor ----------------------------------------------------

class FeatureExtractor(nn.Module):
    """
    1D-ResNet + Dilated Temporal Conv + pooling (GAP for training, RWAP
    for inference).

    Processes a single limb's IMU tensor (2 segments x 6 axes = 12 channels)
    into a fixed-size embedding vector for similarity comparison.

    Pipeline:
        Conv1d stem (12 -> conv_channels) -> N x ResBlock1D
        -> Dilated Temporal Conv (2 layers, dilations 1 and 2) -> Pooling

    The stem_kernel_size parameter controls the initial receptive field,
    allowing arm vs leg extractors to capture different motion profiles:
        - Wider kernel (7): captures broader temporal patterns (arms)
        - Narrower kernel (3): captures sharp, fast dynamics (legs)

    The dilated conv block captures medium-range temporal dependencies
    (receptive field ~7 timesteps) and replaces the previous Bi-LSTM. It
    is fully parallel across time, which makes CPU inference several
    times faster -- 8 extractor forwards run per inference step in the
    live MQTT pipeline, and the Bi-LSTM's sequential loop over 128
    frames was the dominant latency cost.

    Pooling strategy:
        - **Training** (self.training=True): flat Global Average Pooling (GAP).
          All frames contribute equally, producing stable embeddings with low
          variance across recordings of the same movement. This helps the
          contrastive loss learn a well-calibrated embedding space.
        - **Inference** (self.training=False): Recency-Weighted Average Pooling
          (RWAP) with exponential weights favouring recent frames:
              w_t = exp(decay * (t - T+1))   for t in [0, T-1]
          With decay=0.03 and T=128, the oldest frame has ~2% the weight of
          the newest. This allows fast score recovery when the student returns
          to good form during live streaming -- old bad frames fade out
          exponentially rather than persisting at full weight until evicted
          from the 128-frame sliding window.

    Speed information is preserved in both modes: a slow movement produces
    different activations than a fast one, which is desired since movement
    tempo is part of quality assessment.
    """

    def __init__(
        self,
        in_channels: int = 12,
        conv_channels: int = 20,
        num_res_blocks: int = 1,
        embed_dim: int = 16,
        dropout: float = 0.3,
        stem_kernel_size: int = 5,
        recency_decay: float = 0.1,
        # Deprecated: kept for backward-compat with callers that still pass
        # lstm_hidden/lstm_layers. If lstm_hidden is given, embed_dim is
        # inferred as lstm_hidden * 2 (matching old bidirectional shape).
        lstm_hidden: int = None,
        lstm_layers: int = None,
    ):
        super().__init__()
        if lstm_hidden is not None:
            embed_dim = lstm_hidden * 2
        self.embed_dim = embed_dim
        self.recency_decay = recency_decay

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=stem_kernel_size,
                      padding=stem_kernel_size // 2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock1D(conv_channels, dropout=dropout) for _ in range(num_res_blocks)]
        )

        # Dilated temporal conv block (replaces Bi-LSTM).
        # Two Conv1d layers with dilations 1 and 2, kernel=3 -> receptive
        # field of 7 timesteps. Fully parallel over the time axis, so this
        # is much cheaper than an LSTM on CPU.
        self.temporal = nn.Sequential(
            nn.Conv1d(conv_channels, embed_dim, kernel_size=3,
                      padding=1, dilation=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3,
                      padding=2, dilation=2),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)

    def _recency_weighted_mean(self, h: torch.Tensor) -> torch.Tensor:
        """
        Exponentially-weighted mean over the sequence dimension.

        Args:
            h: (B, T, D) -- LSTM output sequence.

        Returns:
            (B, D) -- weighted embedding favouring recent frames.

        Weight for frame t (0-indexed, T-1 is newest):
            w_t = exp(decay * (t - (T-1)))
        So w_{T-1} = 1.0 (newest), w_0 = exp(-decay * (T-1)) (oldest).
        Weights are normalised to sum to 1.
        """
        T = h.size(1)
        # positions: [-(T-1), -(T-2), ..., -1, 0]
        positions = torch.arange(T, device=h.device, dtype=h.dtype) - (T - 1)
        weights = torch.exp(self.recency_decay * positions)  # (T,)
        weights = weights / weights.sum()                     # normalise
        # (B, T, D) * (1, T, 1) -> sum over T -> (B, D)
        return (h * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 12, seq_len)
        Returns:
            (batch, embed_dim)
        """
        h = self.stem(x)          # (B, conv_ch, seq)
        h = self.res_blocks(h)    # (B, conv_ch, seq)
        h = self.temporal(h)      # (B, embed_dim, seq)
        h = h.transpose(1, 2)     # (B, seq, embed_dim)
        h = self.dropout(h)

        if self.training:
            return h.mean(dim=1)                   # GAP -> stable embeddings
        return self._recency_weighted_mean(h)      # RWAP -> fast recovery


# -- Main Model -----------------------------------------------------------

LIMB_NAMES = ["left_arm", "right_arm", "left_leg", "right_leg"]

# Which extractor group each limb belongs to
LIMB_GROUP = {
    "left_arm": "arm",
    "right_arm": "arm",
    "left_leg": "leg",
    "right_leg": "leg",
}


def compute_overall_score(
    limb_scores: torch.Tensor,
    sharpness: float = 1.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Statistical aggregation of 4 limb scores into a single overall score.
    NOT learned -- applied at inference only.

    Formula:
        geo_mean = (s0 * s1 * s2 * s3) ^ (1/4)
        worst_penalty = exp(-sharpness * (1 - min(s_i))^2 )
        overall = geo_mean * worst_penalty

    The geometric mean already penalises low outliers more than arithmetic
    mean. The exponential penalty adds a moderate dropoff when any single
    limb is performing badly. Sharpness=1.1 provides gentle penalisation,
    allowing faster recovery when the student returns to good form
    (previous values of 1.5 and 3.0 were too aggressive and caused overall
    scores to stay low long after the offending limb had improved).

    Examples (sharpness=1.1):
        All 0.90 -> geo=0.900, penalty=0.989, overall=0.890
        All 0.50 -> geo=0.500, penalty=0.757, overall=0.379
        [0.10, 0.90, 0.90, 0.90] -> geo=0.547, penalty=0.418, overall=0.229
        [0.90, 0.90, 0.90, 0.50] -> geo=0.796, penalty=0.757, overall=0.603

    Args:
        limb_scores: (..., 4) tensor of per-limb scores in [0, 1].
        sharpness: Controls how aggressively a bad limb drags the score down.
                   Higher = steeper exponential dropoff. Default 1.1.
        eps: Small constant to avoid log(0).

    Returns:
        (...,) tensor of overall scores in [0, 1].
    """
    clamped = limb_scores.clamp(min=eps)
    geo_mean = clamped.prod(dim=-1).pow(1.0 / clamped.size(-1))

    worst_score = limb_scores.min(dim=-1).values
    penalty = torch.exp(-sharpness * (1.0 - worst_score).pow(2))

    return geo_mean * penalty


class AQAModel(nn.Module):
    """
    Multi-Branch Siamese Network for open-set AQA.

    Uses separate feature extractors for arms and legs:
    - Arm extractor: wider stem kernel (7) for sweeping upper-body motions
    - Leg extractor: narrower stem kernel (3) for sharp lower-body dynamics

    Per-limb-group learned temperatures (tau_arm, tau_leg) control score
    sensitivity independently -- arms and legs can have different distance
    scales in embedding space.

    Scoring uses L2 distance mapped through exp(-dist^2 / tau_group^2) to
    [0, 1]. The overall score is computed post-hoc via compute_overall_score()
    -- a fixed statistical formula, not a learned head.
    """

    def __init__(
        self,
        in_channels: int = 12,
        conv_channels: int = 20,
        num_res_blocks: int = 1,
        embed_dim: int = 16,
        dropout: float = 0.3,
        init_temperature: float = 1.5,
    ):
        super().__init__()

        # Arm extractor: wider stem kernel for broader sweeping motions
        self.arm_extractor = FeatureExtractor(
            in_channels=in_channels,
            conv_channels=conv_channels,
            num_res_blocks=num_res_blocks,
            embed_dim=embed_dim,
            dropout=dropout,
            stem_kernel_size=7,
        )

        # Leg extractor: narrower stem kernel for sharp kick/step dynamics
        self.leg_extractor = FeatureExtractor(
            in_channels=in_channels,
            conv_channels=conv_channels,
            num_res_blocks=num_res_blocks,
            embed_dim=embed_dim,
            dropout=dropout,
            stem_kernel_size=3,
        )

        # Per-limb-group learned temperatures: arms and legs can have
        # different distance scales in embedding space.
        # Stored as log(tau) for unconstrained optimisation; tau always positive.
        self.log_temperature_arm = nn.Parameter(
            torch.tensor(init_temperature).log()
        )
        self.log_temperature_leg = nn.Parameter(
            torch.tensor(init_temperature).log()
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Mean temperature across groups (for backward-compatible logging)."""
        return (self.log_temperature_arm.exp() + self.log_temperature_leg.exp()) / 2.0

    @property
    def temperature_arm(self) -> torch.Tensor:
        return self.log_temperature_arm.exp()

    @property
    def temperature_leg(self) -> torch.Tensor:
        return self.log_temperature_leg.exp()

    def _get_extractor(self, limb: str) -> FeatureExtractor:
        """Return the appropriate extractor for a limb."""
        if LIMB_GROUP[limb] == "arm":
            return self.arm_extractor
        return self.leg_extractor

    def _get_tau_sq(self, limb: str) -> torch.Tensor:
        """Return tau^2 for the appropriate limb group."""
        if LIMB_GROUP[limb] == "arm":
            return self.temperature_arm.pow(2)
        return self.temperature_leg.pow(2)

    def forward(
        self,
        shifu_left_arm: torch.Tensor,
        shifu_right_arm: torch.Tensor,
        shifu_left_leg: torch.Tensor,
        shifu_right_leg: torch.Tensor,
        student_left_arm: torch.Tensor,
        student_right_arm: torch.Tensor,
        student_left_leg: torch.Tensor,
        student_right_leg: torch.Tensor,
    ) -> dict:
        """
        Forward pass: extract embeddings, compute per-limb similarity scores.

        Args:
            8 limb tensors, each (batch, seq_len, 12):
                12 = 2 IMU segments x 6 axes (ax, ay, az, gx, gy, gz)
                Arm limbs are routed to arm_extractor, legs to leg_extractor.

        Returns:
            dict with:
                'limb_scores':     (B, 4) in [0, 1] -- per-limb quality scores
                'embeddings':      dict of 8 tensors, each (B, embed_dim)
                'temperature':     scalar -- mean temperature (for logging)
                'temperature_arm': scalar -- arm group temperature
                'temperature_leg': scalar -- leg group temperature
        """
        shifu_inputs = {
            "left_arm": shifu_left_arm,
            "right_arm": shifu_right_arm,
            "left_leg": shifu_left_leg,
            "right_leg": shifu_right_leg,
        }
        student_inputs = {
            "left_arm": student_left_arm,
            "right_arm": student_right_arm,
            "left_leg": student_left_leg,
            "right_leg": student_right_leg,
        }

        # Extract embeddings using limb-group-specific extractors
        embeddings = {}
        for limb in LIMB_NAMES:
            extractor = self._get_extractor(limb)
            # (B, seq, 12) -> (B, 12, seq) for Conv1d
            embeddings[f"shifu_{limb}"] = extractor(
                shifu_inputs[limb].transpose(1, 2)
            )
            embeddings[f"student_{limb}"] = extractor(
                student_inputs[limb].transpose(1, 2)
            )

        # Per-limb L2 distance, mapped to [0, 1] via exp(-dist^2 / tau^2)
        # tau is per-limb-group -- arms and legs learn independent scales.
        limb_scores = []
        for limb in LIMB_NAMES:
            tau_sq = self._get_tau_sq(limb)
            diff = embeddings[f"shifu_{limb}"] - embeddings[f"student_{limb}"]
            dist_sq = (diff * diff).sum(dim=-1)  # (B,)
            limb_scores.append(torch.exp(-dist_sq / tau_sq))

        limb_scores = torch.stack(limb_scores, dim=1)  # (B, 4)

        return {
            "limb_scores": limb_scores,  # (B, 4) in [0, 1]
            "embeddings": embeddings,    # 8 x (B, embed_dim)
            "temperature": self.temperature.detach(),  # for logging
            "temperature_arm": self.temperature_arm.detach(),
            "temperature_leg": self.temperature_leg.detach(),
        }


# -- Smoke test -----------------------------------------------------------

if __name__ == "__main__":
    import os
    from aqa_dataset import get_aqa_dataloader

    RECORDINGS = os.path.join(os.path.dirname(__file__), "recordings")

    model = AQAModel()
    total_params = sum(p.numel() for p in model.parameters())
    arm_params = sum(p.numel() for p in model.arm_extractor.parameters())
    leg_params = sum(p.numel() for p in model.leg_extractor.parameters())
    print(f"Parameters: {total_params:,} (arm extractor: {arm_params:,}, leg extractor: {leg_params:,})")
    print(model)

    loader = get_aqa_dataloader(RECORDINGS, batch_size=4, max_seq_len=128)
    batch = next(iter(loader))

    limb_inputs = {k: batch[k] for k in batch if k not in ("score", "is_mismatch", "movement")}
    output = model(**limb_inputs)

    overall = compute_overall_score(output["limb_scores"])

    print(f"\nOutput shapes:")
    print(f"  limb_scores: {output['limb_scores'].shape}")
    print(f"  embeddings:  {len(output['embeddings'])} x {next(iter(output['embeddings'].values())).shape}")
    print(f"\nSample output:")
    print(f"  limb_scores:   {output['limb_scores'][0].detach()}")
    print(f"  overall_score: {overall[0].detach().item():.4f}")
    print(f"  temperature (mean): {output['temperature'].item():.4f}")
    print(f"  temperature_arm:    {output['temperature_arm'].item():.4f}")
    print(f"  temperature_leg:    {output['temperature_leg'].item():.4f}")

    # Demo the exponential dropoff behaviour
    print(f"\nOverall score examples (sharpness=1.1):")
    test_cases = [
        [0.90, 0.90, 0.90, 0.90],
        [0.50, 0.50, 0.50, 0.50],
        [0.10, 0.90, 0.90, 0.90],
        [0.90, 0.90, 0.90, 0.50],
        [0.00, 0.90, 0.90, 0.90],
    ]
    for case in test_cases:
        t = torch.tensor([case])
        s = compute_overall_score(t).item()
        print(f"  {case} -> {s:.3f}")
