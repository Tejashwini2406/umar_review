from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def explain_mlp(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
) -> dict:
    """Computes Permutation Feature Importance for PyTorch MLP classifier."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        base_logits = model(X_tensor).squeeze().numpy()
        base_preds = (torch.sigmoid(torch.tensor(base_logits)).numpy() >= 0.5).astype(int)
        base_acc = float((base_preds == y).mean())

    perm_importance = {}
    rng = np.random.default_rng(42)

    for i, col_name in enumerate(feature_names):
        acc_drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            with torch.no_grad():
                logits = model(torch.tensor(X_perm, dtype=torch.float32)).squeeze().numpy()
                preds = (torch.sigmoid(torch.tensor(logits)).numpy() >= 0.5).astype(int)
                acc = float((preds == y).mean())
                acc_drops.append(base_acc - acc)
        perm_importance[col_name] = float(np.mean(acc_drops))

    sorted_importance = dict(sorted(perm_importance.items(), key=lambda x: x[1], reverse=True))

    return {
        "base_accuracy": base_acc,
        "permutation_importance": sorted_importance,
    }
