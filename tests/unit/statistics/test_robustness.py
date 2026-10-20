"""Unit tests for Stage 6 robustness and Holm-Bonferroni correction."""

import pytest
import numpy as np
from iot_ids.statistics.robustness import (
    apply_holm_bonferroni_correction,
    count_direction_wins_losses,
    FULL_MULTILEVEL_NO_PROTO,
    FULL_MULTILEVEL_NO_RATES,
    SEMANTIC_FEATURES_ONLY,
)


def test_holm_bonferroni_correction():
    p_vals = [0.001, 0.01, 0.04, 0.20, 0.50]
    significant = apply_holm_bonferroni_correction(p_vals)

    assert len(significant) == 5
    assert bool(significant[0]) is True  # 0.001 <= 0.05/5 = 0.01
    assert bool(significant[1]) is True  # 0.01 <= 0.05/4 = 0.0125
    assert bool(significant[3]) is False


def test_count_direction_wins_losses():
    a = np.array([0.8, 0.85, 0.90, 0.60, 0.70])
    b = np.array([0.5, 0.55, 0.60, 0.65, 0.70])

    counts = count_direction_wins_losses(a, b, threshold=1e-4)
    assert counts["wins"] == 3  # 0.8>0.5, 0.85>0.55, 0.90>0.60
    assert counts["losses"] == 1  # 0.60<0.65
    assert counts["ties"] == 1  # 0.70==0.70
    assert counts["total"] == 5


def test_shortcut_ablation_feature_lists():
    assert len(FULL_MULTILEVEL_NO_PROTO) == 17
    assert len(FULL_MULTILEVEL_NO_RATES) == 19
    assert len(SEMANTIC_FEATURES_ONLY) == 15
