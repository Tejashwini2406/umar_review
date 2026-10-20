"""Unit tests for Level A Instantaneous Feature Extractor."""

import pytest
from iot_ids.data.flow_object import CanonicalFlow
from iot_ids.features.canonical.instant import InstantaneousFeatureExtractor, INSTANT_FEATURE_NAMES


def test_instant_feature_extraction():
    extractor = InstantaneousFeatureExtractor()

    flow = CanonicalFlow(
        timestamp_start=100.0,
        timestamp_end=102.0,  # duration = 2.0s
        src_host="192.168.1.5",
        dst_host="192.168.1.1",
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

    features = extractor.extract(flow)

    assert features["flow_duration"] == 2.0
    assert pytest.approx(features["flow_bytes_per_sec"], 0.01) == 1500.0 / 2.0
    assert pytest.approx(features["flow_pkts_per_sec"], 0.01) == 15.0 / 2.0
    assert pytest.approx(features["mean_pkt_size"], 0.01) == 1500.0 / 15.0
    assert pytest.approx(features["payload_byte_ratio"], 0.01) == 1000.0 / 1500.0
    assert pytest.approx(features["pkt_count_ratio"], 0.01) == 10.0 / 15.0
    assert pytest.approx(features["tcp_syn_ratio"], 0.01) == 1.0 / 15.0
    assert features["proto_tcp"] == 1.0
    assert features["proto_udp"] == 0.0

    vec = extractor.extract_vector(flow)
    assert len(vec) == len(INSTANT_FEATURE_NAMES)
