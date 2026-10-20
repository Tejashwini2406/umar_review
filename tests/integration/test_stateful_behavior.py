"""Stateful Stream, Packet Ordering, Expiration, and Reset Behavioral Tests."""

from pathlib import Path
import pytest
import numpy as np

from iot_ids.data.flow_object import CanonicalFlow
from iot_ids.data.flow_aggregator import FlowAggregator
from iot_ids.features.canonical.flow_builder import CanonicalFlowBuilder
from iot_ids.pipeline.system import IDSSystemPipeline
from iot_ids.inference.predictor import IDSPredictor
from iot_ids.registry.manager import ModelRegistry


def test_aggregator_packet_expiration_and_flushing():
    """Verifies FlowAggregator evicts flows upon inactivity timeout (15s)."""
    aggregator = FlowAggregator(inactivity_timeout=15.0)

    # Frame 1 at t=10.0
    f1 = aggregator.process_frame(
        timestamp=10.0, src_ip="192.168.1.10", dst_ip="10.0.0.1",
        src_port=1234, dst_port=80, proto="tcp", length=100, syn=1
    )
    assert f1 is None  # Active flow in buffer

    # Frame 2 at t=30.0 (idle time 20s > 15s timeout)
    f2 = aggregator.process_frame(
        timestamp=30.0, src_ip="192.168.1.10", dst_ip="10.0.0.1",
        src_port=1234, dst_port=80, proto="tcp", length=200, ack=1
    )
    assert f2 is not None  # Expired flow emitted
    assert f2.total_bytes == 100

    # Flush remaining flows
    remaining = aggregator.flush()
    assert len(remaining) == 1
    assert remaining[0].total_bytes == 200


def test_multi_host_fanout_and_port_entropy():
    """Verifies BehavioralFeatureExtractor tracks multi-host fanout and port entropy."""
    builder = CanonicalFlowBuilder(window_seconds=30.0)

    flow1 = CanonicalFlow(
        timestamp_start=10.0, timestamp_end=11.0,
        src_host="192.168.1.50", dst_host="10.0.0.1",
        src_port=5000, dst_port=80, protocol="tcp",
        bytes_src=100, bytes_dst=100, pkts_src=2, pkts_dst=2,
        syn_count_src=1, ack_count_src=1, rst_count_src=0,
        conn_state_code=0, label=0, attack_category="Normal"
    )

    flow2 = CanonicalFlow(
        timestamp_start=12.0, timestamp_end=13.0,
        src_host="192.168.1.50", dst_host="10.0.0.2",
        src_port=5001, dst_port=443, protocol="tcp",
        bytes_src=100, bytes_dst=100, pkts_src=2, pkts_dst=2,
        syn_count_src=1, ack_count_src=1, rst_count_src=0,
        conn_state_code=0, label=0, attack_category="Normal"
    )

    feats1 = builder.build_features(flow1)
    feats2 = builder.build_features(flow2)

    div1 = feats1["behavioral"]["behavioral_dst_diversity"]
    div2 = feats2["behavioral"]["behavioral_dst_diversity"]

    assert div2 > div1, f"Destination diversity should increase (div1={div1}, div2={div2})"


def test_real_artifact_loading_reproducibility():
    """Verifies loading real exported model artifact from models/final/deployable_artifact."""
    artifact_dir = Path("models/final/deployable_artifact")
    if not artifact_dir.exists():
        pytest.skip(f"Deployable artifact directory not found: {artifact_dir}")

    predictor = IDSPredictor.from_artifact(artifact_dir)

    flow = CanonicalFlow(
        timestamp_start=100.0, timestamp_end=101.0,
        src_host="192.168.1.10", dst_host="10.0.0.1",
        src_port=12345, dst_port=80, protocol="tcp",
        bytes_src=200, bytes_dst=400, pkts_src=5, pkts_dst=4,
        syn_count_src=1, ack_count_src=1, rst_count_src=0,
        conn_state_code=0, label=0, attack_category="Normal"
    )

    alert1 = predictor.predict_flow(flow)
    predictor.reset_state()
    alert2 = predictor.predict_flow(flow)

    assert alert1["prediction_prob"] == alert2["prediction_prob"]
    assert alert1["risk_level"] == alert2["risk_level"]
    assert alert1["is_anomaly"] == alert2["is_anomaly"]
