"""Unit tests for CanonicalFlow object."""

import pytest
from iot_ids.data.flow_object import CanonicalFlow


def test_canonical_flow_creation_and_properties():
    flow = CanonicalFlow(
        timestamp_start=100.0,
        timestamp_end=105.0,
        src_host="192.168.1.10",
        dst_host="192.168.1.1",
        src_port=443,
        dst_port=8080,
        protocol="tcp",
        bytes_src=1000,
        bytes_dst=500,
        pkts_src=10,
        pkts_dst=5,
        syn_count_src=1,
        ack_count_src=1,
        rst_count_src=0,
        conn_state_code=0,
        label=1,
        attack_category="DDoS",
    )

    assert flow.duration == 5.0
    assert flow.total_bytes == 1500
    assert flow.total_pkts == 15
    assert flow.label == 1
    assert flow.attack_category == "DDoS"

    d = flow.to_dict()
    assert d["src_host"] == "192.168.1.10"
    assert d["duration"] == 5.0


def test_canonical_flow_validation_raises():
    with pytest.raises(ValueError, match="Invalid duration"):
        CanonicalFlow(
            timestamp_start=105.0,
            timestamp_end=100.0,  # invalid: end < start
            src_host="A",
            dst_host="B",
            src_port=80,
            dst_port=80,
            protocol="tcp",
            bytes_src=0,
            bytes_dst=0,
            pkts_src=0,
            pkts_dst=0,
            syn_count_src=0,
            ack_count_src=0,
            rst_count_src=0,
            conn_state_code=0,
            label=0,
            attack_category="Normal",
        )

    with pytest.raises(ValueError, match="Invalid protocol"):
        CanonicalFlow(
            timestamp_start=100.0,
            timestamp_end=101.0,
            src_host="A",
            dst_host="B",
            src_port=80,
            dst_port=80,
            protocol="invalid_proto",
            bytes_src=0,
            bytes_dst=0,
            pkts_src=0,
            pkts_dst=0,
            syn_count_src=0,
            ack_count_src=0,
            rst_count_src=0,
            conn_state_code=0,
            label=0,
            attack_category="Normal",
        )
