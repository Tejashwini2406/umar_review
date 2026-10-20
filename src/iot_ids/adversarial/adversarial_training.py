"""Adversarial training module for MLP classifier.

Implements adversarial training where PGD-generated adversarial examples
are mixed with clean training data to improve model robustness.

The adversarial training protocol:
  1. For each mini-batch, generate PGD adversarial examples from the current model
  2. Mix clean and adversarial examples (default 50/50)
  3. Train on the mixed batch
  4. Repeat for each epoch

This is a standard approach from Madry et al. (2018):
  "Towards Deep Learning Models Resistant to Adversarial Attacks"
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from iot_ids.adversarial.attacks import pgd_attack


def train_mlp_adversarial(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    epsilon: float = 0.1,
    pgd_steps: int = 7,
    adv_ratio: float = 0.5,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 0.001,
    discrete_feature_names: set[str] | None = None,
    x_min: np.ndarray | None = None,
    x_max: np.ndarray | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Trains MLP with adversarial training using PGD perturbations.

    Args:
        model: MLP model to train (will be modified in-place).
        X_train: Training features (n_samples, n_features).
        y_train: Training labels (n_samples,).
        feature_names: Ordered list of feature names.
        epsilon: Maximum perturbation magnitude (L∞ norm).
        pgd_steps: Number of PGD steps for adversarial example generation.
        adv_ratio: Fraction of each batch that is adversarial (0.0 = clean only, 1.0 = adversarial only).
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.
        discrete_feature_names: Set of feature names that should NOT be perturbed.
        x_min: Per-feature minimum values for domain clamping.
        x_max: Per-feature maximum values for domain clamping.
        verbose: Print per-epoch training metrics.

    Returns:
        Dict with training history (per-epoch clean loss, adversarial loss, accuracy).
    """
    if discrete_feature_names is None:
        discrete_feature_names = {
            "proto_tcp", "proto_udp", "proto_icmp", "proto_other",
            "is_well_known_port", "mqtt_msgtype", "mbtcp_unit_id",
            "conn_state_encoded", "http_method_encoded",
        }

    # Build continuous feature mask
    cont_mask_list = [0.0 if col in discrete_feature_names else 1.0 for col in feature_names]
    continuous_mask = torch.tensor(cont_mask_list, dtype=torch.float32).unsqueeze(0)

    # Domain bounds for clamping
    if x_min is None:
        x_min_t = torch.tensor(X_train.min(axis=0), dtype=torch.float32).unsqueeze(0)
    else:
        x_min_t = torch.tensor(x_min, dtype=torch.float32).unsqueeze(0)

    if x_max is None:
        x_max_t = torch.tensor(X_train.max(axis=0), dtype=torch.float32).unsqueeze(0)
    else:
        x_max_t = torch.tensor(x_max, dtype=torch.float32).unsqueeze(0)

    # Create DataLoader
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "epoch": [],
        "clean_loss": [],
        "adv_loss": [],
        "mixed_loss": [],
        "clean_accuracy": [],
    }

    for epoch in range(epochs):
        model.train()
        epoch_clean_loss = 0.0
        epoch_adv_loss = 0.0
        epoch_mixed_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_batches = 0

        for bx, by in loader:
            bs = bx.size(0)
            n_adv = max(1, int(bs * adv_ratio))
            n_clean = bs - n_adv

            # Split batch into clean and adversarial portions
            bx_clean = bx[:n_clean]
            by_clean = by[:n_clean]

            bx_adv_orig = bx[n_clean:]
            by_adv = by[n_clean:]

            # Generate PGD adversarial examples against current model
            model.eval()
            bx_adv = pgd_attack(
                model=model,
                X=bx_adv_orig,
                y=by_adv,
                epsilon=epsilon,
                continuous_mask=continuous_mask,
                x_min=x_min_t,
                x_max=x_max_t,
                steps=pgd_steps,
            )
            model.train()

            # Combine clean and adversarial
            bx_mixed = torch.cat([bx_clean, bx_adv], dim=0)
            by_mixed = torch.cat([by_clean, by_adv], dim=0)

            # Forward pass on mixed batch
            optimizer.zero_grad()
            logits = model(bx_mixed)
            loss = criterion(logits, by_mixed.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_mixed_loss += loss.item()

            # Track clean and adversarial loss separately (for monitoring)
            with torch.no_grad():
                if n_clean > 0:
                    clean_logits = model(bx_clean)
                    cl = criterion(clean_logits, by_clean.unsqueeze(1)).item()
                    epoch_clean_loss += cl
                if n_adv > 0:
                    adv_logits = model(bx_adv)
                    al = criterion(adv_logits, by_adv.unsqueeze(1)).item()
                    epoch_adv_loss += al

                # Clean accuracy
                with torch.no_grad():
                    preds = (torch.sigmoid(model(bx[:bs])).squeeze() >= 0.5).float()
                    epoch_correct += (preds == by[:bs]).sum().item()
                    epoch_total += bs

            n_batches += 1

        avg_clean = epoch_clean_loss / max(n_batches, 1)
        avg_adv = epoch_adv_loss / max(n_batches, 1)
        avg_mixed = epoch_mixed_loss / max(n_batches, 1)
        acc = epoch_correct / max(epoch_total, 1)

        history["epoch"].append(epoch + 1)
        history["clean_loss"].append(avg_clean)
        history["adv_loss"].append(avg_adv)
        history["mixed_loss"].append(avg_mixed)
        history["clean_accuracy"].append(acc)

        if verbose:
            print(
                f"  Epoch {epoch + 1:3d}/{epochs} | "
                f"Clean Loss: {avg_clean:.4f} | "
                f"Adv Loss: {avg_adv:.4f} | "
                f"Mixed Loss: {avg_mixed:.4f} | "
                f"Clean Acc: {acc:.4f}"
            )

    return history
