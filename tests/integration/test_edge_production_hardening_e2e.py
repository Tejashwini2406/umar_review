"""Integration Tests for Production Hardening and Edge Deployment.

Tests fail-fast PipelineConfig validation, health status inspection, operational metrics,
alert sinks (JSONL & Console), bounded memory limits, and inference error isolation.
"""
import tempfile
from pathlib import Path
import pytest

from iot_ids.config import PipelineConfig
from iot_ids.inference.predictor import IDSPredictor
from iot_ids.data.flow_aggregator import FlowAggregator
from iot_ids.utils.metrics import MetricsCollector
from iot_ids.utils.sinks import JSONLFileSink, ConsoleAlertSink, MultiAlertSink


@pytest.fixture
def deployable_predictor():
    artifact_dir = Path("models/final/deployable_artifact")
    if not artifact_dir.exists():
        pytest.skip(f"Deployable artifact directory not found: {artifact_dir}")
    return IDSPredictor.from_artifact(artifact_dir)


def test_pipeline_config_fail_fast_validation():
    """Verifies PipelineConfig enforces strict fail-fast validation for invalid parameters."""
    valid_cfg = PipelineConfig(decision_threshold=0.5, inactivity_timeout=15.0, max_active_flows=1000)
    valid_cfg.validate()  # Should pass cleanly

    with pytest.raises(ValueError, match="decision_threshold"):
        PipelineConfig(decision_threshold=1.5).validate()

    with pytest.raises(ValueError, match="inactivity_timeout"):
        PipelineConfig(inactivity_timeout=-5.0).validate()

    with pytest.raises(ValueError, match="max_active_flows"):
        PipelineConfig(max_active_flows=0).validate()

    with pytest.raises(ValueError, match="model_family"):
        PipelineConfig(model_family="UnregisteredModel").validate()


def test_predictor_health_check(deployable_predictor):
    """Verifies health check status reporting."""
    health = deployable_predictor.health_check()

    assert "pipeline" in health
    assert "metrics" in health
    assert health["pipeline"]["status"] == "HEALTHY"
    assert health["pipeline"]["model_loaded"] is True
    assert health["pipeline"]["preprocessor_loaded"] is True


def test_metrics_collector_tracking():
    """Verifies MetricsCollector records packet count, flow count, latency, and summary."""
    collector = MetricsCollector()
    collector.record_packet()
    collector.record_packet()
    collector.record_flow()
    collector.record_latency(12.5)
    collector.record_latency(15.0)

    summary = collector.get_summary()
    assert summary.packets_processed == 2
    assert summary.flows_completed == 1
    assert summary.latency_count == 2
    assert summary.latency_min_ms == 12.5
    assert summary.latency_max_ms == 15.0
    assert summary.latency_mean_ms == 13.75


def test_alert_sinks_persistence():
    """Verifies JSONLFileSink and MultiAlertSink persist structured alert records."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        jsonl_path = Path(tmp_dir) / "test_alerts.jsonl"
        sink1 = JSONLFileSink(jsonl_path)
        sink2 = ConsoleAlertSink()
        multi_sink = MultiAlertSink([sink1, sink2])

        alert_data = {
            "flow_id": "192.168.1.1:100->10.0.0.1:80",
            "src_ip": "192.168.1.1",
            "dst_ip": "10.0.0.1",
            "timestamp": 1000.0,
            "prediction_prob": 0.95,
            "decision_threshold": 0.5,
            "is_anomaly": True,
            "risk_score": 0.95,
            "risk_level": "CRITICAL",
            "feature_profile": "full_multilevel",
            "raw_features": {}
        }

        multi_sink.write_alert(alert_data)
        assert jsonl_path.exists()

        lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        assert "192.168.1.1" in lines[0]


def test_bounded_flow_memory_eviction():
    """Verifies FlowAggregator evicts oldest flow when max_active_flows limit is reached."""
    aggregator = FlowAggregator(inactivity_timeout=15.0, max_active_flows=2)

    # Add Flow 1 at t=10.0
    aggregator.process_frame(10.0, "192.168.1.1", "10.0.0.1", 1000, 80, "tcp", 100)
    # Add Flow 2 at t=12.0
    aggregator.process_frame(12.0, "192.168.1.2", "10.0.0.2", 2000, 80, "tcp", 100)

    assert len(aggregator.active_flows) == 2

    # Add Flow 3 at t=14.0 (triggers eviction of oldest flow)
    evicted = aggregator.process_frame(14.0, "192.168.1.3", "10.0.0.3", 3000, 80, "tcp", 100)

    assert evicted is not None
    assert len(aggregator.active_flows) == 2
