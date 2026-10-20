"""Stage 6 Paired Effect Size Calculation Module.

Calculates Cohen's dz for paired samples, Hedges' g correction, Wilcoxon signed-rank statistics,
mean/median paired differences, and improvement directions using the 12 source->target transfer directions as paired units.
"""
from __future__ import annotations

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats


def calculate_paired_cohens_dz(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates Cohen's dz for paired samples: mean(diff) / std(diff)."""
    diff = a - b
    std_diff = float(np.std(diff, ddof=1))
    if std_diff == 0.0:
        return 0.0
    return float(np.mean(diff) / std_diff)


def calculate_hedges_g_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Applies small-sample Hedges' g correction factor J(N-1) to paired Cohen's dz."""
    dz = calculate_paired_cohens_dz(a, b)
    n = len(a)
    if n <= 1:
        return dz
    j_factor = 1.0 - (3.0 / (4.0 * (n - 1) - 1.0))
    return float(dz * j_factor)


def compute_paired_comparison_stats(
    a: np.ndarray, b: np.ndarray, label_a: str, label_b: str, metric_name: str
) -> Dict[str, Any]:
    """Computes comprehensive paired statistical summary across paired experimental units."""
    n = len(a)
    diff = a - b
    mean_diff = float(np.mean(diff))
    median_diff = float(np.median(diff))
    std_diff = float(np.std(diff, ddof=1))

    dz = calculate_paired_cohens_dz(a, b)
    g = calculate_hedges_g_paired(a, b)

    # Paired t-test
    try:
        t_stat, p_val = stats.ttest_rel(a, b)
    except Exception:
        t_stat, p_val = 0.0, 1.0

    # Wilcoxon signed-rank test
    try:
        w_stat, w_pval = stats.wilcoxon(a, b)
    except Exception:
        w_stat, w_pval = 0.0, 1.0

    direction = "SUPERIOR" if mean_diff > 0 else ("INFERIOR" if mean_diff < 0 else "EQUAL")

    return {
        "comparison": f"{label_a} vs {label_b}",
        "metric": metric_name,
        "n_units": n,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_paired_diff": mean_diff,
        "median_paired_diff": median_diff,
        "std_paired_diff": std_diff,
        "cohens_dz": dz,
        "hedges_g": g,
        "ttest_stat": float(t_stat),
        "ttest_pvalue": float(p_val),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pvalue": float(w_pval),
        "direction": direction,
    }
