"""SHAP-based Explainability for IoT IDS models.

Provides SHAP (SHapley Additive exPlanations) values for:
  - Random Forest via TreeExplainer (exact Shapley values)
  - MLP via GradientExplainer (gradient-based approximation)

SHAP values satisfy the Shapley axioms (efficiency, symmetry, dummy, additivity)
and provide both local (per-sample) and global (aggregated) explanations.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def explain_rf_shap(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    max_samples: int = 1000,
) -> dict:
    """Computes SHAP TreeExplainer values for Random Forest.

    TreeExplainer computes exact Shapley values in polynomial time for
    tree-based models, making it both fast and exact.

    Args:
        model: Trained sklearn RandomForestClassifier.
        X: Feature matrix (n_samples, n_features).
        feature_names: Ordered list of feature names.
        max_samples: Max samples to explain (for speed).

    Returns:
        Dict with shap_values, mean_abs_shap (global importance),
        and per-feature statistics.
    """
    import shap

    # Subsample if needed for computational efficiency
    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X_explain = X[idx]
    else:
        X_explain = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    # For binary classification, handle list of arrays, 3D array, or 2D array
    if isinstance(shap_values, list) and len(shap_values) == 2:
        sv = shap_values[1]  # Class 1 = attack
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, 1] if shap_values.shape[2] == 2 else shap_values[:, :, 0]
    else:
        sv = shap_values

    # Global importance: mean |SHAP value| per feature
    mean_abs = np.mean(np.abs(sv), axis=0)
    importance_dict = dict(zip(feature_names, mean_abs.tolist()))
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    # Per-feature statistics
    feature_stats = {}
    for i, fname in enumerate(feature_names):
        feature_stats[fname] = {
            "mean_abs_shap": float(mean_abs[i]),
            "mean_shap": float(np.mean(sv[:, i])),
            "std_shap": float(np.std(sv[:, i])),
            "max_abs_shap": float(np.max(np.abs(sv[:, i]))),
        }

    return {
        "shap_values": sv,
        "mean_abs_shap": sorted_importance,
        "feature_stats": feature_stats,
        "n_samples_explained": len(X_explain),
        "explainer_type": "TreeExplainer",
    }


def explain_mlp_shap(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    background_samples: int = 100,
    max_samples: int = 500,
) -> dict:
    """Computes SHAP GradientExplainer values for PyTorch MLP.

    GradientExplainer uses expected gradients (an extension of integrated
    gradients) to approximate Shapley values for neural networks.

    Args:
        model: Trained PyTorch MLP (nn.Module).
        X: Feature matrix (n_samples, n_features).
        feature_names: Ordered list of feature names.
        background_samples: Number of background samples for the explainer.
        max_samples: Max samples to explain.

    Returns:
        Dict with shap_values, mean_abs_shap (global importance),
        and per-feature statistics.
    """
    import shap
    import torch

    model.eval()

    rng = np.random.default_rng(42)

    # Background data for GradientExplainer
    bg_idx = rng.choice(len(X), size=min(background_samples, len(X)), replace=False)
    background = torch.tensor(X[bg_idx], dtype=torch.float32)

    # Samples to explain
    if len(X) > max_samples:
        explain_idx = rng.choice(len(X), size=max_samples, replace=False)
        X_explain = X[explain_idx]
    else:
        X_explain = X

    X_tensor = torch.tensor(X_explain, dtype=torch.float32)

    try:
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(X_tensor)
    except Exception:
        # Fallback to KernelExplainer if GradientExplainer fails
        def model_predict(x_np):
            with torch.no_grad():
                t = torch.tensor(x_np, dtype=torch.float32)
                logits = model(t).squeeze(-1)
                return torch.sigmoid(logits).numpy()

        bg_np = X[bg_idx]
        explainer = shap.KernelExplainer(model_predict, bg_np)
        shap_values = explainer.shap_values(X_explain, nsamples=200)

    if isinstance(shap_values, list):
        sv = shap_values[0] if len(shap_values) == 1 else shap_values[1]
    else:
        sv = shap_values
        if sv.ndim == 3:
            sv = sv[:, :, 0]  # Take first output dimension

    mean_abs = np.mean(np.abs(sv), axis=0)
    importance_dict = dict(zip(feature_names, mean_abs.tolist()))
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    feature_stats = {}
    for i, fname in enumerate(feature_names):
        feature_stats[fname] = {
            "mean_abs_shap": float(mean_abs[i]),
            "mean_shap": float(np.mean(sv[:, i])),
            "std_shap": float(np.std(sv[:, i])),
            "max_abs_shap": float(np.max(np.abs(sv[:, i]))),
        }

    return {
        "shap_values": sv,
        "mean_abs_shap": sorted_importance,
        "feature_stats": feature_stats,
        "n_samples_explained": len(X_explain),
        "explainer_type": "GradientExplainer",
    }


def compare_shap_rankings(
    shap_result_1: dict,
    shap_result_2: dict,
    label_1: str = "Model A",
    label_2: str = "Model B",
) -> dict:
    """Compares SHAP-based feature importance rankings between two models.

    Uses Spearman rank correlation to measure agreement between two
    models' feature importance orderings.

    Args:
        shap_result_1: Output from explain_rf_shap or explain_mlp_shap.
        shap_result_2: Output from explain_rf_shap or explain_mlp_shap.
        label_1: Label for the first model.
        label_2: Label for the second model.

    Returns:
        Dict with Spearman rho, p-value, and per-feature comparison table.
    """
    from scipy.stats import spearmanr

    imp1 = shap_result_1["mean_abs_shap"]
    imp2 = shap_result_2["mean_abs_shap"]

    common_features = [f for f in imp1 if f in imp2]
    if len(common_features) < 3:
        return {"spearman_rho": 0.0, "p_value": 1.0, "common_features": len(common_features)}

    vec1 = [imp1[f] for f in common_features]
    vec2 = [imp2[f] for f in common_features]

    result = spearmanr(vec1, vec2)
    rho = float(result.statistic) if not np.isnan(result.statistic) else 0.0
    p_val = float(result.pvalue) if not np.isnan(result.pvalue) else 1.0

    # Build comparison table
    comparison = []
    rank1 = {f: r + 1 for r, f in enumerate(sorted(common_features, key=lambda f: imp1[f], reverse=True))}
    rank2 = {f: r + 1 for r, f in enumerate(sorted(common_features, key=lambda f: imp2[f], reverse=True))}
    for f in common_features:
        comparison.append({
            "feature": f,
            f"{label_1}_shap": imp1[f],
            f"{label_1}_rank": rank1[f],
            f"{label_2}_shap": imp2[f],
            f"{label_2}_rank": rank2[f],
            "rank_diff": abs(rank1[f] - rank2[f]),
        })

    return {
        "spearman_rho": rho,
        "p_value": p_val,
        "common_features": len(common_features),
        "comparison": sorted(comparison, key=lambda x: x["rank_diff"], reverse=True),
        "model_labels": [label_1, label_2],
    }
