from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def compute_model_consensus(
    importance1: dict[str, float],
    importance2: dict[str, float],
) -> dict:
    """Computes Spearman rank correlation coefficient (rho) between two feature importance dicts."""
    common_keys = [k for k in importance1 if k in importance2]
    if not common_keys:
        return {"spearman_rho": 0.0, "p_value": 1.0, "common_features": 0}

    vec1 = [importance1[k] for k in common_keys]
    vec2 = [importance2[k] for k in common_keys]

    res = spearmanr(vec1, vec2)
    rho = float(res.statistic) if not np.isnan(res.statistic) else 0.0
    p_val = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0

    return {
        "spearman_rho": rho,
        "p_value": p_val,
        "common_features": len(common_keys),
    }
