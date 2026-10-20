from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def explain_ae_reconstruction(
    model: nn.Module,
    X: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Computes per-feature reconstruction error decomposition: e_i = (x_i - hat_x_i)^2."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        recon = model(X_tensor).numpy()

    # Per-sample, per-feature squared error
    sq_err = (X - recon) ** 2
    total_mse_per_sample = np.mean(sq_err, axis=1)

    # Average feature contribution across samples
    mean_feat_mse = np.mean(sq_err, axis=0)
    feat_importance = dict(zip(feature_names, mean_feat_mse.tolist()))

    return {
        "mean_mse_per_feature": dict(sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)),
        "squared_errors": sq_err,
        "total_mse": total_mse_per_sample,
    }
