"""Feature Schema Validation for Runtime Inference.

Provides fail-fast input validation for canonical multi-level flow features,
ensuring strict column counts, feature ordering, type safety, and zero missing values
before inference execution.
"""
from __future__ import annotations

from typing import Dict, List, Union, Any
import numpy as np
import pandas as pd

# 21-column standardized canonical feature names
STANDARDIZED_21_FEATURES: List[str] = [
    "flow_duration",
    "flow_bytes_per_sec",
    "flow_pkts_per_sec",
    "mean_pkt_size",
    "payload_byte_ratio",
    "pkt_count_ratio",
    "tcp_syn_ratio",
    "proto_tcp",
    "proto_udp",
    "proto_icmp",
    "proto_other",
    "temporal_iat_mean",
    "temporal_iat_cv",
    "temporal_flow_rate_ewma",
    "temporal_byte_rate_ewma",
    "temporal_syn_rate_ewma",
    "behavioral_dst_diversity",
    "behavioral_port_entropy",
    "behavioral_fanout_ratio",
    "behavioral_unanswered_ratio",
    "behavioral_src_activity_ewma",
]


def validate_input_schema(
    inputs: Union[pd.DataFrame, Dict[str, Any], np.ndarray, List[Dict[str, Any]]],
    profile: str = "standardized_21",
    expected_feature_names: Union[List[str], None] = None,
) -> pd.DataFrame:
    """Validates runtime input features against the expected profile schema.

    Args:
        inputs: Input flow data as a pandas DataFrame, dictionary, list of dictionaries, or numpy array.
        profile: Feature profile name ('standardized_21' or 'historical_13').
        expected_feature_names: Optional explicit list of expected feature names.

    Returns:
        pd.DataFrame containing canonicalized, ordered, validated feature values.

    Raises:
        ValueError: If feature count, column presence, feature names, or values violate schema.
    """
    if profile not in {"standardized_21", "historical_13"}:
        raise ValueError(f"Runtime Schema Error: Unsupported profile '{profile}'. Must be 'standardized_21' or 'historical_13'.")

    if expected_feature_names is None:
        if profile == "standardized_21":
            expected_feature_names = STANDARDIZED_21_FEATURES
        else:
            raise ValueError("Runtime Schema Error: Explicit expected_feature_names list required for 'historical_13' profile.")

    expected_count = len(expected_feature_names)

    # 1. Convert input to pandas DataFrame
    if isinstance(inputs, dict):
        df = pd.DataFrame([inputs])
    elif isinstance(inputs, list):
        if not inputs:
            raise ValueError("Runtime Schema Error: Input list of samples is empty.")
        if isinstance(inputs[0], dict):
            df = pd.DataFrame(inputs)
        else:
            df = pd.DataFrame(inputs, columns=expected_feature_names if len(inputs[0]) == expected_count else None)
    elif isinstance(inputs, np.ndarray):
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        if inputs.shape[1] != expected_count:
            raise ValueError(f"Runtime Schema Error: Input array shape {inputs.shape} does not match expected feature count ({expected_count}).")
        df = pd.DataFrame(inputs, columns=expected_feature_names)
    elif isinstance(inputs, pd.DataFrame):
        df = inputs.copy()
    else:
        raise ValueError(f"Runtime Schema Error: Unsupported input type '{type(inputs)}'. Expected DataFrame, Dict, List[Dict], or ndarray.")

    # 2. Check feature count
    actual_cols = list(df.columns)

    if len(actual_cols) != expected_count:
        missing_feats = [col for col in expected_feature_names if col not in actual_cols]
        extra_feats = [col for col in actual_cols if col not in expected_feature_names]
        err_msg = (
            f"Runtime Schema Error: [{profile}] requires exactly {expected_count} canonical features, "
            f"but received {len(actual_cols)} columns."
        )
        if missing_feats:
            err_msg += f"\n  Missing required features ({len(missing_feats)}): {missing_feats}"
        if extra_feats:
            err_msg += f"\n  Unexpected extra features ({len(extra_feats)}): {extra_feats}"
        raise ValueError(err_msg)

    # 3. Check column presence
    missing_feats = [col for col in expected_feature_names if col not in df.columns]
    if missing_feats:
        raise ValueError(f"Runtime Schema Error: [{profile}] missing required features: {missing_feats}")

    # 4. Enforce deterministic canonical feature ordering
    df_canonical = df[expected_feature_names].copy()

    # 5. Numeric validation (NaN and Inf checks)
    num_array = df_canonical.to_numpy()

    if np.isnan(num_array).any():
        nan_cols = df_canonical.columns[df_canonical.isna().any()].tolist()
        raise ValueError(f"Runtime Schema Error: Input features contain NaN values in columns: {nan_cols}")

    if np.isinf(num_array).any():
        inf_cols = df_canonical.columns[np.isinf(df_canonical).any()].tolist()
        raise ValueError(f"Runtime Schema Error: Input features contain Infinite values in columns: {inf_cols}")

    return df_canonical
