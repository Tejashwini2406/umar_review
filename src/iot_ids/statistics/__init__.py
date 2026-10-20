"""Stage 6 Statistical Robustness & Effect Sizes Module."""

from iot_ids.statistics.effect_sizes import (
    calculate_paired_cohens_dz,
    calculate_hedges_g_paired,
    compute_paired_comparison_stats,
)
from iot_ids.statistics.confidence_intervals import (
    bootstrap_mean_ci,
    bootstrap_paired_difference_ci,
)
from iot_ids.statistics.robustness import (
    apply_holm_bonferroni_correction,
    count_direction_wins_losses,
    PROTOCOL_FEATURES,
    RAW_RATE_FEATURES,
    FULL_MULTILEVEL_NO_PROTO,
    FULL_MULTILEVEL_NO_RATES,
    SEMANTIC_FEATURES_ONLY,
)

__all__ = [
    "calculate_paired_cohens_dz",
    "calculate_hedges_g_paired",
    "compute_paired_comparison_stats",
    "bootstrap_mean_ci",
    "bootstrap_paired_difference_ci",
    "apply_holm_bonferroni_correction",
    "count_direction_wins_losses",
    "PROTOCOL_FEATURES",
    "RAW_RATE_FEATURES",
    "FULL_MULTILEVEL_NO_PROTO",
    "FULL_MULTILEVEL_NO_RATES",
    "SEMANTIC_FEATURES_ONLY",
]
