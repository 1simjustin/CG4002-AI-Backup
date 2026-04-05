"""
Multi-Branch Siamese Network for open-set Action Quality Assessment.

Architecture
------------
FeatureExtractor (shared weights):
    Conv1d stem -> 2x ResBlock1D -> Bi-LSTM -> Global Average Pooling
    Input:  (batch, 12, seq_len)
    Output: (batch, 16)  -- low-dim for L2 distance spread

AQAModel:
    8 limb tensors -> shared FeatureExtractor -> 8 embeddings
    Per-limb L2 distance -> exp(-dist^2 / τ^2) -> 4 limb scores in [0, 1]
    τ (temperature) is a learned parameter that calibrates score spread.

Overall score is computed post-hoc via compute_overall_score() using a
geometric-mean with exponential penalty for weak limbs. This is NOT
learned -- it is a fixed statistical aggregation applied at inference.

Output dict:
    'limb_scores':   (batch, 4)
    'embeddings':    dict of 8 tensors, each (batch, embed_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -- 1D Residual Block ----------------------------------------------------

class ResBlock1D(nn.Module):
    """Pre-activation 1D residual block with optional downsampling."""

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
    1D-ResNet + Bi-LSTM + Global Average Pooling.

    Processes a single limb tensor into a fixed-size embedding.
    """

    def __init__(
        self,
        in_channels: int = 12,
        conv_channels: int = 32,
        num_res_blocks: int = 2,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed_dim = lstm_hidden * 2  # bidirectional

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock1D(conv_channels, dropout=dropout) for _ in range(num_res_blocks)]
        )

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 12, seq_len)
        Returns:
            (batch, embed_dim)
        """
        h = self.stem(x)          # (B, conv_ch, seq)
        h = self.res_blocks(h)    # (B, conv_ch, seq)
        h = h.transpose(1, 2)     # (B, seq, conv_ch)
        h, _ = self.lstm(h)       # (B, seq, embed_dim)
        h = self.dropout(h)
        return h.mean(dim=1)      # GAP -> (B, embed_dim)


# -- Main Model -----------------------------------------------------------

LIMB_NAMES = ["left_arm", "right_arm", "left_leg", "right_leg"]


def compute_overall_score(
    limb_scores: torch.Tensor,
    sharpness: float = 3.0,
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
    mean. The exponential penalty adds a steep dropoff when any single limb
    is performing badly.

    Examples (sharpness=3.0):
        All 0.90 -> geo=0.900, penalty=0.970, overall=0.873
        All 0.50 -> geo=0.500, penalty=0.472, overall=0.236
        [0.10, 0.90, 0.90, 0.90] -> geo=0.547, penalty=0.067, overall=0.037
        [0.90, 0.90, 0.90, 0.50] -> geo=0.796, penalty=0.472, overall=0.376

    Args:
        limb_scores: (..., 4) tensor of per-limb scores in [0, 1].
        sharpness: Controls how aggressively a bad limb drags the score down.
                   Higher = steeper exponential dropoff.
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

    Scoring uses L2 distance mapped through exp(-dist^2 / τ^2) to [0, 1],
    where τ is a learned temperature parameter that calibrates the score
    spread. This allows the model to learn the optimal distance-to-score
    mapping rather than relying on a fixed exponential curve.

    The overall score is computed post-hoc via compute_overall_score()
    -- a fixed statistical formula, not a learned head.
    """

    def __init__(
        self,
        in_channels: int = 12,
        conv_channels: int = 32,
        num_res_blocks: int = 2,
        lstm_hidden: int = 8,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        init_temperature: float = 1.0,
    ):
        super().__init__()

        self.feature_extractor = FeatureExtractor(
            in_channels=in_channels,
            conv_channels=conv_channels,
            num_res_blocks=num_res_blocks,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )

        # Learned temperature for score calibration: exp(-dist^2 / τ^2)
        # Stored as log(τ) for unconstrained optimisation; τ is always positive.
        self.log_temperature = nn.Parameter(
            torch.tensor(init_temperature).log()
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Current temperature value (always positive)."""
        return self.log_temperature.exp()

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
        All inputs: (batch, seq_len, 12) from the dataloader.

        Returns:
            dict with 'limb_scores', 'embeddings', and 'temperature'.
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

        # Extract embeddings for all 8 limb tensors
        embeddings = {}
        for limb in LIMB_NAMES:
            # (B, seq, 12) -> (B, 12, seq) for Conv1d
            embeddings[f"shifu_{limb}"] = self.feature_extractor(
                shifu_inputs[limb].transpose(1, 2)
            )
            embeddings[f"student_{limb}"] = self.feature_extractor(
                student_inputs[limb].transpose(1, 2)
            )

        # Per-limb L2 distance, mapped to [0, 1] via exp(-dist^2 / τ^2)
        # τ (temperature) is learned -- controls how quickly scores decay
        # with distance. Larger τ -> gentler decay -> higher scores for
        # moderate distances. Smaller τ -> sharper decay.
        tau_sq = self.temperature.pow(2)
        limb_scores = []
        for limb in LIMB_NAMES:
            diff = embeddings[f"shifu_{limb}"] - embeddings[f"student_{limb}"]
            dist_sq = (diff * diff).sum(dim=-1)  # (B,)
            limb_scores.append(torch.exp(-dist_sq / tau_sq))

        limb_scores = torch.stack(limb_scores, dim=1)  # (B, 4)

        return {
            "limb_scores": limb_scores,  # (B, 4) in [0, 1]
            "embeddings": embeddings,    # 8 x (B, embed_dim)
            "temperature": self.temperature.detach(),  # for logging
        }


# -- Smoke test -----------------------------------------------------------

if __name__ == "__main__":
    import os
    from aqa_dataset import get_aqa_dataloader

    RECORDINGS = os.path.join(os.path.dirname(__file__), "recordings")

    model = AQAModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
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
    print(f"  temperature:   {output['temperature'].item():.4f}")

    # Demo the exponential dropoff behaviour
    print(f"\nOverall score examples (sharpness=3.0):")
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
