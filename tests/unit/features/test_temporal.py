"""Unit tests for Level B Causal Temporal Feature Extractor."""

import pytest
from iot_ids.data.flow_object import CanonicalFlow
from iot_ids.features.canonical.temporal import TemporalFeatureExtractor, TEMPORAL_FEATURE_NAMES


def test_temporal_causal_execution_and_initialization():
    extractor = TemporalFeatureExtractor(alpha=0.1)

    f1 = CanonicalFlow(
        timestamp_start=100.0,
        timestamp_end=101.0,
        src_host="10.0.0.5",
        dst_host="10.0.0.1",
        src_port=100,
        dst_port=80,
        protocol="tcp",
        bytes_src=500,
        bytes_dst=200,
        pkts_src=5,
        pkts_dst=2,
        syn_count_src=1,
        ack_count_src=1,
        rst_count_src=0,
        conn_state_code=0,
        label=0,
        attack_category="Normal",
    )

    # Event 1: First event initialization policy
    feat1 = extractor.extract_and_update(f1)
    assert feat1["temporal_iat_mean"] == 0.0
    assert feat1["temporal_iat_cv"] == 0.0
    assert feat1["temporal_flow_rate_ewma"] == 0.0
    assert extractor.first_event_count == 1

    # Event 2: Should read state from Event 1 (READ BEFORE WRITE)
    f2 = CanonicalFlow(
        timestamp_start=105.0,  # delta_t = 5.0s
        timestamp_end=106.0,
        src_host="10.0.0.5",
        dst_host="10.0.0.2",
        src_port=101,
        dst_port=80,
        protocol="tcp",
        bytes_src=300,
        bytes_dst=100,
        pkts_src=3,
        pkts_dst=1,
        syn_count_src=1,
        ack_count_src=1,
        rst_count_src=0,
        conn_state_code=0,
        label=0,
        attack_category="Normal",
    )

    feat2 = extractor.extract_and_update(f2)
    # Event 2 features must read previous state (iat_mean from state updated after f1)
    assert feat2["temporal_flow_rate_ewma"] > 0.0
    assert feat2["temporal_byte_rate_ewma"] > 0.0


def test_temporal_state_reset():
    extractor = TemporalFeatureExtractor()
    f1 = CanonicalFlow(
        timestamp_start=100.0,
        timestamp_end=101.0,
        src_host="10.0.0.5",
        dst_host="10.0.0.1",
        src_port=100,
        dst_port=80,
        protocol="tcp",
        bytes_src=500,
        bytes_dst=200,
        pkts_src=5,
        pkts_dst=2,
        syn_count_src=1,
        ack_count_src=1,
        rst_count_src=0,
        conn_state_code=0,
        label=0,
        attack_category="Normal",
    )
    extractor.extract_and_update(f1)
    assert len(extractor.host_states) == 1

    extractor.reset_state()
    assert len(extractor.host_states) == 0
