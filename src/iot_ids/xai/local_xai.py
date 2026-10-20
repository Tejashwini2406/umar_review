"""Local feature-ablation explanations for the complete Option C ensemble."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


def explain_local_sample(
    rf_model: Any,
    mlp_model: Any,
    x_single: np.ndarray,
    feature_names: list[str],
    top_k: int = 5,
    decision_threshold: float = 0.5,
    rf_background_X: np.ndarray | None = None,
    mlp_background_X: np.ndarray | None = None,
) -> dict[str, Any]:
    """Explain the fused probability by ablating each feature to a common baseline.

    This is model-faithful single-feature ablation, not a Shapley attribution:
    feature interactions are not allocated across features.
    """
    x_2d = np.atleast_2d(np.asarray(x_single, dtype=np.float64))
    if x_2d.ndim != 2 or x_2d.shape[1] != len(feature_names):
        actual = x_2d.shape[1] if x_2d.ndim == 2 else 0
        raise ValueError(f"Feature count mismatch: got {actual}, expected {len(feature_names)}")
    if not np.isfinite(x_2d).all():
        raise ValueError("x_single must contain only finite values")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    background_arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (rf_background_X, mlp_background_X)
        if values is not None
    ]
    for background in background_arrays:
        if background.ndim != 2 or background.shape[1] != len(feature_names):
            raise ValueError("Background arrays must have shape (n_samples, n_features)")
        if background.shape[0] == 0 or not np.isfinite(background).all():
            raise ValueError("Background arrays must be non-empty and finite")
    if background_arrays:
        baseline = np.median(np.vstack(background_arrays), axis=0)
    else:
        baseline = np.zeros(len(feature_names), dtype=np.float64)

    def predict_option_c_probability(values: np.ndarray) -> np.ndarray:
        rf_probability = np.asarray(rf_model.predict_proba(values))[:, 1]
        mlp_model.eval()
        with torch.no_grad():
            logits = mlp_model(torch.as_tensor(values, dtype=torch.float32)).reshape(-1)
            mlp_probability = torch.sigmoid(logits).cpu().numpy()
        probabilities = 0.7 * rf_probability + 0.3 * mlp_probability
        if not np.isfinite(probabilities).all():
            raise ValueError("The fused predictor returned non-finite probabilities")
        return probabilities

    rf_probability = float(np.asarray(rf_model.predict_proba(x_2d))[0, 1])
    mlp_model.eval()
    with torch.no_grad():
        mlp_probability = float(
            torch.sigmoid(
                mlp_model(torch.as_tensor(x_2d, dtype=torch.float32)).reshape(-1)[0]
            ).item()
        )
    option_c_probability = 0.7 * rf_probability + 0.3 * mlp_probability
    baseline_probability = float(
        predict_option_c_probability(baseline.reshape(1, -1))[0]
    )
    final_decision = (
        "ATTACK" if option_c_probability >= decision_threshold else "BENIGN"
    )

    effects: list[tuple[int, float, float, float]] = []
    for index in range(len(feature_names)):
        ablated = x_2d.copy()
        ablated[:, index] = baseline[index]
        rf_ablated = float(np.asarray(rf_model.predict_proba(ablated))[0, 1])
        with torch.no_grad():
            mlp_ablated = float(
                torch.sigmoid(
                    mlp_model(torch.as_tensor(ablated, dtype=torch.float32)).reshape(-1)[0]
                ).item()
            )
        fused_ablated = 0.7 * rf_ablated + 0.3 * mlp_ablated
        effects.append(
            (
                index,
                option_c_probability - fused_ablated,
                rf_probability - rf_ablated,
                mlp_probability - mlp_ablated,
            )
        )

    effects.sort(key=lambda item: abs(item[1]), reverse=True)
    top_features = []
    for index, fused_delta, rf_delta, mlp_delta in effects[: min(top_k, len(feature_names))]:
        top_features.append(
            {
                "feature": feature_names[index],
                "raw_value": float(x_2d[0, index]),
                "baseline_value": float(baseline[index]),
                "fused_probability_delta": float(fused_delta),
                "rf_probability_delta": float(rf_delta),
                "mlp_probability_delta": float(mlp_delta),
                "direction": (
                    "PUSHES_ATTACK"
                    if fused_delta > 0.0
                    else "PUSHES_BENIGN"
                    if fused_delta < 0.0
                    else "NO_CHANGE"
                ),
            }
        )

    return {
        "option_c_probability": float(option_c_probability),
        "rf_probability": rf_probability,
        "robust_mlp_probability": mlp_probability,
        "final_decision": final_decision,
        "top_k_features": top_features,
        "baseline_probability": baseline_probability,
        "methodology": (
            "single-feature ablation on the complete Option C probability; "
            "not SHAP and not additive across interacting features"
        ),
    }
