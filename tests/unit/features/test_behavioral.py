"""Unit tests for Level C Causal Behavioral Feature Extractor."""

import pytest
from iot_ids.data.flow_object import CanonicalFlow
from iot_ids.features.canonical.behavioral import BehavioralFeatureExtractor, BEHAVIORAL_FEATURE_NAMES


def test_behavioral_window_eviction_and_metrics():
    extractor = BehavioralFeatureExtractor(window_seconds=30.0)

    # Event 1 at t = 100.0
    f1 = CanonicalFlow(
        timestamp_start=100.0,
        timestamp_end=101.0,
        src_host="192.168.1.50",
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

    feat1 = extractor.extract_and_update(f1)
    # First event has no prior window buffer -> 0.0
    assert feat1["behavioral_dst_diversity"] == 0.0
    assert feat1["behavioral_port_entropy"] == 0.0

    # Event 2 at t = 110.0 -> target distinct destination
    f2 = CanonicalFlow(
        timestamp_start=110.0,
        timestamp_end=111.0,
        src_host="192.168.1.50",
        dst_host="10.0.0.2",
        src_port=101,
        dst_port=443,
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
    # Reads state from Event 1 -> 1 unique destination (10.0.0.1)
    assert feat2["behavioral_dst_diversity"] == 1.0

    # Event 3 at t = 135.0 (cutoff = 135 - 30 = 105.0) -> Event 1 (t=100) evicted (<105), Event 2 (t=110) retained (>=105)
    f3 = CanonicalFlow(
        timestamp_start=135.0,
        timestamp_end=136.0,
        src_host="192.168.1.50",
        dst_host="10.0.0.3",
        src_port=102,
        dst_port=8080,
        protocol="tcp",
        bytes_src=200,
        bytes_dst=100,
        pkts_src=2,
        pkts_dst=1,
        syn_count_src=1,
        ack_count_src=1,
        rst_count_src=0,
        conn_state_code=0,
        label=0,
        attack_category="Normal",
    )

    feat3 = extractor.extract_and_update(f3)
    # Event 1 (t=100) was evicted, so only Event 2 (t=110) remains in window -> 1 unique destination (10.0.0.2)
    assert feat3["behavioral_dst_diversity"] == 1.0
    assert extractor.total_evictions >= 1
