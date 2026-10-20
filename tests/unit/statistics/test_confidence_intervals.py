"""Unit tests for Stage 6 confidence intervals calculation."""

import pytest
import numpy as np
from iot_ids.statistics.confidence_intervals import (
    bootstrap_mean_ci,
    bootstrap_paired_difference_ci,
)


def test_bootstrap_mean_ci():
    data = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.75, 0.65, 0.55, 0.95, 0.88, 0.92])
    mean_val, ci_lower, ci_upper = bootstrap_mean_ci(data, n_boot=1000, seed=42)

    assert ci_lower <= mean_val <= ci_upper
    assert 0.40 <= ci_lower <= 0.80
    assert 0.75 <= ci_upper <= 1.00


def test_bootstrap_paired_difference_ci():
    a = np.array([0.8, 0.85, 0.90, 0.95, 0.99, 0.88, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99])
    b = np.array([0.5, 0.55, 0.60, 0.65, 0.70, 0.58, 0.62, 0.64, 0.66, 0.67, 0.68, 0.69])

    mean_diff, ci_lower, ci_upper = bootstrap_paired_difference_ci(a, b, n_boot=1000, seed=42)
    assert ci_lower > 0.20
    assert ci_lower <= mean_diff <= ci_upper
