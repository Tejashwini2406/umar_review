from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def fgsm_attack(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    continuous_mask: torch.Tensor,
    x_min: torch.Tensor,
    x_max: torch.Tensor,
) -> torch.Tensor:
    """Fast Gradient Sign Method (FGSM) attack with discrete feature masking and domain clamping."""
    model.eval()
    X_adv = X.clone().detach().requires_grad_(True)

    criterion = nn.BCEWithLogitsLoss()
    logits = model(X_adv)
    loss = criterion(logits, y.unsqueeze(1))
    loss.backward()

    with torch.no_grad():
        grad_sign = X_adv.grad.sign() * continuous_mask
        X_adv = X_adv + epsilon * grad_sign
        X_adv = torch.max(torch.min(X_adv, x_max), x_min)

    return X_adv.detach()


def pgd_attack(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    continuous_mask: torch.Tensor,
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    steps: int = 10,
    alpha: float | None = None,
) -> torch.Tensor:
    """Projected Gradient Descent (PGD) attack with discrete feature masking and domain clamping."""
    model.eval()
    if alpha is None:
        alpha = epsilon / 4.0

    X_orig = X.clone().detach()
    X_adv = X.clone().detach()

    criterion = nn.BCEWithLogitsLoss()

    for step in range(steps):
        X_adv.requires_grad = True
        logits = model(X_adv)
        loss = criterion(logits, y.unsqueeze(1))
        
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad_sign = X_adv.grad.sign() * continuous_mask
            X_adv = X_adv + alpha * grad_sign
            
            # Project onto epsilon-ball around original sample
            eta = torch.clamp(X_adv - X_orig, min=-epsilon, max=epsilon)
            X_adv = X_orig + eta * continuous_mask
            
            # Clamp to physical domain min/max
            X_adv = torch.max(torch.min(X_adv, x_max), x_min)

    return X_adv.detach()
