"""Stage 6 Statistical Robustness & Domain Shortcut Diagnostic Module.

Provides Holm-Bonferroni multiplicity correction, direction-level win/loss counts,
shortcut feature ablations, and feature distribution shift diagnostic correlations.
"""
from __future__ import annotations

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from iot_ids.features.canonical.flow_builder import CANONICAL_18_FEATURE_NAMES


# Feature groupings for shortcut ablation diagnostics
PROTOCOL_FEATURES = ["proto_tcp", "proto_udp", "proto_icmp", "proto_other"]
RAW_RATE_FEATURES = ["flow_bytes_per_sec", "flow_pkts_per_sec"]

FULL_MULTILEVEL_NO_PROTO = [f for f in CANONICAL_18_FEATURE_NAMES if f not in PROTOCOL_FEATURES]
FULL_MULTILEVEL_NO_RATES = [f for f in CANONICAL_18_FEATURE_NAMES if f not in RAW_RATE_FEATURES]
SEMANTIC_FEATURES_ONLY = [f for f in CANONICAL_18_FEATURE_NAMES if f not in PROTOCOL_FEATURES and f not in RAW_RATE_FEATURES]


def apply_holm_bonferroni_correction(p_values: List[float]) -> List[bool]:
    """Applies Holm-Bonferroni step-down correction for multiple hypothesis testing (alpha=0.05)."""
    n = len(p_values)
    if n == 0:
        return []

    sorted_indices = np.argsort(p_values)
    sorted_pvals = np.array(p_values)[sorted_indices]

    significant = np.zeros(n, dtype=bool)
    for k in range(n):
        adj_alpha = 0.05 / (n - k)
        if sorted_pvals[k] <= adj_alpha:
            significant[sorted_indices[k]] = True
        else:
            break
    return list(significant)


def count_direction_wins_losses(
    a: np.ndarray, b: np.ndarray, threshold: float = 1e-4
) -> Dict[str, int]:
    """Counts win/loss/tie outcomes across paired transfer directions."""
    diff = a - b
    wins = int(np.sum(diff > threshold))
    losses = int(np.sum(diff < -threshold))
    ties = int(np.sum(np.abs(diff) <= threshold))
    return {"wins": wins, "losses": losses, "ties": ties, "total": len(a)}
