"""Autoencoder for IoT IDS anomaly detection.

Architecture:
  Encoder: [input_dim] → [64, ReLU] → [bottleneck_dim, ReLU]
  Decoder: [bottleneck_dim] → [64, ReLU] → [input_dim]

Trained on benign-only data. Anomaly score = per-sample MSE reconstruction error.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Unsupervised autoencoder for anomaly detection via reconstruction error.

    This is the canonical AE architecture used for training and inference.
    It should be trained on benign-only traffic so that attack traffic produces
    higher reconstruction error (anomaly score).

    The architecture mirrors the inline definitions in:
      - scripts/05_train_models.py (AutoencoderModule)
      - scripts/06_calibrate_risk.py (AutoencoderModule)
      - scripts/10_run_adversarial_evaluation.py (AutoencoderModule)
    """

    def __init__(self, input_dim: int, bottleneck_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        code = self.encoder(x)
        return self.decoder(code)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the bottleneck encoding."""
        return self.encoder(x)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Computes per-sample MSE reconstruction error as anomaly score."""
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            mse = torch.mean((x - recon) ** 2, dim=1)
        return mse
