"""Unit tests for CanonicalFlowBuilder and 18-dimensional vector materialization."""

import pytest
import numpy as np
from iot_ids.data.flow_object import CanonicalFlow
from iot_ids.features.canonical.flow_builder import CanonicalFlowBuilder, CANONICAL_18_FEATURE_NAMES


def test_flow_builder_18_dimensional_vector():
    builder = CanonicalFlowBuilder()

    flow = CanonicalFlow(
        timestamp_start=100.0,
        timestamp_end=102.0,
        src_host="10.0.0.1",
        dst_host="10.0.0.2",
        src_port=1234,
        dst_port=80,
        protocol="tcp",
        bytes_src=1000,
        bytes_dst=500,
        pkts_src=10,
        pkts_dst=5,
        syn_count_src=1,
        ack_count_src=1,
        rst_count_src=0,
        conn_state_code=0,
        label=0,
        attack_category="Normal",
    )

    vec = builder.build_vector(flow)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (21,)  # 11 Level A (8 core + 4 proto one-hot) + 5 Level B + 5 Level C
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()

    grouped = builder.build_features(flow)
    assert "instant" in grouped
    assert "temporal" in grouped
    assert "behavioral" in grouped


def test_cross_split_state_reset():
    builder = CanonicalFlowBuilder()

    f1 = CanonicalFlow(
        timestamp_start=100.0,
        timestamp_end=102.0,
        src_host="10.0.0.1",
        dst_host="10.0.0.2",
        src_port=1234,
        dst_port=80,
        protocol="tcp",
        bytes_src=1000,
        bytes_dst=500,
        pkts_src=10,
        pkts_dst=5,
        syn_count_src=1,
        ack_count_src=1,
        rst_count_src=0,
        conn_state_code=0,
        label=0,
        attack_category="Normal",
    )

    builder.build_vector(f1)
    # State accumulated
    assert len(builder.temporal_extractor.host_states) == 1
    assert len(builder.behavioral_extractor.host_states) == 1

    # Reset state for split boundary or cross-domain transfer
    builder.reset_state()
    assert len(builder.temporal_extractor.host_states) == 0
    assert len(builder.behavioral_extractor.host_states) == 0
