"""Unit tests for Stage 3 quality audit, chronological splits, and causal state isolation."""

import pytest
import numpy as np
import pandas as pd

from iot_ids.data.flow_object import CanonicalFlow
from iot_ids.features.canonical.flow_builder import CanonicalFlowBuilder, CANONICAL_18_FEATURE_NAMES
from iot_ids.features.quality import (
    compute_distribution_statistics,
    compute_distribution_shift,
    compute_mutual_information,
    compute_correlation,
    compute_vif,
)


def test_quality_audit_functions():
    data = {
        "flow_duration": [1.0, 2.0, 3.0, 4.0, 5.0],
        "flow_bytes_per_sec": [100.0, 200.0, 300.0, 400.0, 500.0],
        "payload_byte_ratio": [0.1, 0.2, 0.3, 0.4, 0.5],
        "label": [0, 0, 1, 1, 1],
    }
    df = pd.DataFrame(data)
    feature_names = ["flow_duration", "flow_bytes_per_sec", "payload_byte_ratio"]

    # 1. Distribution stats
    stats_df = compute_distribution_statistics(df, feature_names)
    assert len(stats_df) == 3
    assert stats_df.loc[stats_df["feature"] == "flow_duration", "mean"].values[0] == 3.0

    # 2. Distribution shift (same df vs same df -> KS stat = 0.0)
    shift_df = compute_distribution_shift(df, df, feature_names)
    assert len(shift_df) == 3
    assert shift_df["ks_statistic"].max() == 0.0
    assert shift_df["wasserstein_distance"].max() == 0.0

    # 3. Mutual Information
    mi_df = compute_mutual_information(df, feature_names, label_col="label")
    assert len(mi_df) == 3
    assert "mutual_information" in mi_df.columns

    # 4. Correlation
    p_corr, s_corr = compute_correlation(df, feature_names)
    assert p_corr.shape == (3, 3)

    # 5. VIF
    vif_df = compute_vif(df, feature_names)
    assert len(vif_df) == 3


def test_chronological_splits_and_state_isolation():
    builder = CanonicalFlowBuilder()
    
    # Generate 100 dummy flows with increasing timestamps
    flows = []
    for i in range(100):
        flows.append(
            CanonicalFlow(
                timestamp_start=float(1000 + i * 10),
                timestamp_end=float(1005 + i * 10),
                src_host=f"10.0.0.{(i % 5) + 1}",
                dst_host=f"192.168.1.{(i % 3) + 1}",
                src_port=1000 + i,
                dst_port=80,
                protocol="tcp",
                bytes_src=500 + i * 10,
                bytes_dst=200,
                pkts_src=5,
                pkts_dst=2,
                syn_count_src=1,
                ack_count_src=1,
                rst_count_src=0,
                conn_state_code=0,
                label=0 if i < 60 else 1,
                attack_category="Normal" if i < 60 else "Attack",
            )
        )

    # Split: Train (60), Val (20), Test (20)
    train_flows = flows[:60]
    val_flows = flows[60:80]
    test_flows = flows[80:]

    assert train_flows[-1].timestamp_start < val_flows[0].timestamp_start
    assert val_flows[-1].timestamp_start < test_flows[0].timestamp_start

    # State reset before processing train split
    builder.reset_state()
    assert len(builder.temporal_extractor.host_states) == 0

    tr_vecs = [builder.build_vector(f) for f in train_flows]
    assert len(tr_vecs) == 60

    # State reset before processing val split
    builder.reset_state()
    assert len(builder.temporal_extractor.host_states) == 0
    val_vecs = [builder.build_vector(f) for f in val_flows]
    assert len(val_vecs) == 20

    # First event of val split must be evaluated with empty initial state
    assert builder.temporal_extractor.first_event_count > 0
