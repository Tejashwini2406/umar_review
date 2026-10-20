from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def explain_rf(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
) -> dict:
    """Computes global Gini feature importances and Permutation Feature Importances for Random Forest."""
    # 1. Native Gini Feature Importance
    gini_importance = dict(zip(feature_names, model.feature_importances_.tolist()))

    # 2. Permutation Feature Importance
    perm = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    perm_importance = dict(zip(feature_names, perm.importances_mean.tolist()))

    return {
        "gini_importance": dict(sorted(gini_importance.items(), key=lambda x: x[1], reverse=True)),
        "permutation_importance": dict(sorted(perm_importance.items(), key=lambda x: x[1], reverse=True)),
    }
