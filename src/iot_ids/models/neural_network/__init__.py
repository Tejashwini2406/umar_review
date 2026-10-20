"""MLP Classifier for IoT IDS.

Architecture: [input_dim] → [128, BN, ReLU, Dropout] → [64, BN, ReLU, Dropout] → [32, ReLU] → [1]
Output: raw logits (apply sigmoid for probability)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """Supervised MLP binary classifier for intrusion detection.

    This is the canonical MLP architecture used for both training and inference.
    The architecture uses BatchNorm + ReLU + Dropout for regularization.
    Output is a single raw logit; apply torch.sigmoid() for probability.
    """

    def __init__(self, input_dim: int, hidden_layers: list[int] | None = None, dropout: float = 0.2):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(in_dim, h))
            if i < len(hidden_layers) - 1:
                # BatchNorm + ReLU + Dropout for all but last hidden layer
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            else:
                # Last hidden layer: just ReLU, no BN/Dropout
                layers.append(nn.ReLU())
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
