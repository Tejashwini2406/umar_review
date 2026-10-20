"""Unit tests for FlowAggregator engine."""

import pytest
from iot_ids.data.flow_aggregator import FlowAggregator, normalize_5tuple


def test_normalize_5tuple():
    key1, is_fw1 = normalize_5tuple("1.1.1.1", "2.2.2.2", 100, 200, "tcp")
    key2, is_fw2 = normalize_5tuple("2.2.2.2", "1.1.1.1", 200, 100, "tcp")

    assert key1 == key2
    assert is_fw1 != is_fw2


def test_flow_aggregator_reconstruction():
    aggregator = FlowAggregator(inactivity_timeout=5.0, max_flow_duration=10.0)

    # Frame 1: Forward SYN
    f1 = aggregator.process_frame(
        timestamp=100.0,
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        src_port=1234,
        dst_port=80,
        proto="tcp",
        length=60,
        syn=1,
        ack=0,
        label=1,
        attack_category="DDoS",
    )
    assert f1 is None

    # Frame 2: Reverse ACK (Response)
    f2 = aggregator.process_frame(
        timestamp=101.0,
        src_ip="10.0.0.2",
        dst_ip="10.0.0.1",
        src_port=80,
        dst_port=1234,
        proto="tcp",
        length=100,
        syn=0,
        ack=1,
        label=1,
        attack_category="DDoS",
    )
    assert f2 is None

    # Frame 3: Expired by inactivity timeout (t = 110.0 > 101.0 + 5.0)
    f3 = aggregator.process_frame(
        timestamp=110.0,
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        src_port=1234,
        dst_port=80,
        proto="tcp",
        length=50,
        label=1,
        attack_category="DDoS",
    )
    assert f3 is not None
    assert f3.pkts_src == 1
    assert f3.pkts_dst == 1
    assert f3.bytes_src == 60
    assert f3.bytes_dst == 100
    assert f3.duration == 1.0

    # Flush remaining flows
    flushed = aggregator.flush()
    assert len(flushed) == 1
    assert flushed[0].bytes_src == 50
