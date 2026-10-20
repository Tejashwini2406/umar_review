from __future__ import annotations

import numpy as np
import pandas as pd


def add_causal_rolling_count(
    frame: pd.DataFrame,
    group_col: str,
    order_col: str,
    output_col: str,
    window: int,
    prior_counts: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if group_col not in frame or order_col not in frame:
        result = frame.copy()
        result[output_col] = pd.NA
        return result, prior_counts or {}
    if prior_counts is None:
        prior_counts = {}
    ordered = frame.sort_values([group_col, order_col], kind="mergesort").copy()
    
    # 0-based rank within group for current chunk
    raw_cumcount = ordered.groupby(group_col, sort=False).cumcount()
    
    # Carry-forward offset from prior chunks
    offsets = ordered[group_col].map(lambda s: prior_counts.get(s, 0))
    
    # 1-based global count of observations seen so far for this group
    global_k = raw_cumcount + offsets + 1
    
    # Bounded rolling count window: min(global_k, window)
    values = np.minimum(global_k, window).astype(float)
    ordered[output_col] = values
    
    # Update prior_counts with total rows seen so far
    updated_counts = dict(prior_counts)
    group_sizes = ordered.groupby(group_col, sort=False).size()
    for grp, size in group_sizes.items():
        updated_counts[grp] = prior_counts.get(grp, 0) + size
        
    return ordered.sort_index(), updated_counts


def assert_temporal_causality(
    original: pd.DataFrame,
    transformed: pd.DataFrame,
    feature_col: str,
    order_col: str,
) -> None:
    changed_future = original.copy()
    last_idx = changed_future[order_col].idxmax()
    changed_future.loc[last_idx, order_col] = changed_future[order_col].max() + 10_000
    if not transformed.loc[original[order_col] < original[order_col].max(), feature_col].equals(
        transformed.loc[original[order_col] < original[order_col].max(), feature_col]
    ):
        raise AssertionError("Temporal feature depends on future observations.")

