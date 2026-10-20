"""Stage 6 Non-Parametric Bootstrap 95% Confidence Interval Module.

Estimates 95% bootstrap confidence intervals (percentile method) for means and paired differences
over the 12 source->target transfer directions as paired experimental units.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def bootstrap_mean_ci(
    data: np.ndarray, n_boot: int = 10000, ci_level: float = 0.95, seed: int = 42
) -> Tuple[float, float, float]:
    """Computes bootstrap mean and 95% percentile confidence interval (ci_lower, ci_upper)."""
    if len(data) == 0:
        return 0.0, 0.0, 0.0

    np.random.seed(seed)
    n = len(data)
    boot_means = np.zeros(n_boot)

    for i in range(n_boot):
        sample = np.random.choice(data, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - ci_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    mean_val = float(np.mean(data))
    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))
    return mean_val, ci_lower, ci_upper


def bootstrap_paired_difference_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10000, ci_level: float = 0.95, seed: int = 42
) -> Tuple[float, float, float]:
    """Computes bootstrap mean paired difference and 95% percentile confidence interval."""
    diff = a - b
    return bootstrap_mean_ci(diff, n_boot=n_boot, ci_level=ci_level, seed=seed)
