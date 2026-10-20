"""Unit tests for Stage 6 effect sizes calculation."""

import pytest
import numpy as np
from iot_ids.statistics.effect_sizes import (
    calculate_paired_cohens_dz,
    calculate_hedges_g_paired,
    compute_paired_comparison_stats,
)


def test_calculate_paired_cohens_dz():
    a = np.array([0.8, 0.85, 0.90, 0.95, 0.99])
    b = np.array([0.5, 0.55, 0.60, 0.65, 0.70])

    dz = calculate_paired_cohens_dz(a, b)
    assert dz > 10.0  # Large effect size for consistent difference


def test_compute_paired_comparison_stats():
    a = np.array([0.8, 0.82, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99])
    b = np.array([0.4, 0.42, 0.45, 0.48, 0.50, 0.52, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59])

    stats = compute_paired_comparison_stats(a, b, "ProfileA", "ProfileB", "roc_auc")
    assert stats["comparison"] == "ProfileA vs ProfileB"
    assert stats["metric"] == "roc_auc"
    assert stats["n_units"] == 12
    assert stats["mean_paired_diff"] > 0.35
    assert stats["cohens_dz"] > 5.0
    assert stats["ttest_pvalue"] < 0.001
    assert stats["direction"] == "SUPERIOR"
