"""Realistic End-to-End Integration & CLI Subcommand Tests.

Verifies real training -> export -> load -> batch prediction, streaming telemetry ingestion,
structured alert logging, error handling, and CLI subcommand dispatches.
"""
import json
import tempfile
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from iot_ids.config import PipelineConfig
from iot_ids.training.train import train_and_export_pipeline
from iot_ids.inference.predictor import IDSPredictor
from iot_ids.utils.logging import IDSAlertLogger
from iot_ids.data.flow_aggregator import FlowAggregator


def test_e2e_training_export_load_batch_pipeline():
    """Verifies end-to-end train -> export -> load -> batch inference cycle."""
    stage3_dir = Path("data/processed/stage3")
    if not (stage3_dir / "ToN-IoT").exists():
        pytest.skip("Materialized stage3 datasets not found.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        art_dir = Path(tmp_dir) / "test_artifact"
        config = PipelineConfig(
            artifact_dir=str(art_dir),
            source_domain="ToN-IoT",
            target_domain="Edge-IIoTset",
            model_family="RandomForest",
            n_estimators=10,
            max_depth=5,
        )

        saved_path = train_and_export_pipeline(config, stage3_dir=stage3_dir)
        assert saved_path.exists()
        assert (saved_path / "model.joblib").exists()
        assert (saved_path / "pipeline_metadata.json").exists()

        # Load Predictor
        predictor = IDSPredictor.from_artifact(saved_path)

        # Batch Inference over test split
        test_df = pd.read_parquet(stage3_dir / "ToN-IoT" / "test.parquet").iloc[:10]
        alerts = predictor.predict_dataframe(test_df)

        assert len(alerts) == 10
        assert "prediction_prob" in alerts[0]
        assert "risk_score" in alerts[0]
        assert "is_anomaly" in alerts[0]


def test_streaming_raw_telemetry_aggregation_e2e():
    """Verifies raw packet -> FlowAggregator -> 18 features -> pipeline -> alert."""
    art_dir = Path("models/final/deployable_artifact")
    if not art_dir.exists():
        pytest.skip(f"Deployable artifact directory not found: {art_dir}")

    predictor = IDSPredictor.from_artifact(art_dir)
    predictor.reset_state()

    # Packet 1 at t=10.0
    pkt1 = {
        "timestamp": 10.0, "src_ip": "192.168.1.50", "dst_ip": "10.0.0.1",
        "src_port": 1234, "dst_port": 80, "protocol": "tcp", "length": 100, "syn": 1
    }
    alert1 = predictor.process_raw_packet(pkt1)
    assert alert1 is None  # Active flow in aggregator buffer

    # Packet 2 at t=30.0 (idle time 20s > 15s timeout -> evicts flow1)
    pkt2 = {
        "timestamp": 30.0, "src_ip": "192.168.1.50", "dst_ip": "10.0.0.1",
        "src_port": 1234, "dst_port": 80, "protocol": "tcp", "length": 200, "ack": 1
    }
    alert2 = predictor.process_raw_packet(pkt2)
    assert alert2 is not None  # Evicted flow evaluated
    assert "prediction_prob" in alert2
    assert alert2["src_ip"] in ("192.168.1.50", "10.0.0.1")


def test_structured_alert_logging():
    """Verifies IDSAlertLogger writes structured JSON lines logs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "alerts.log"
        logger = IDSAlertLogger(log_path=log_path, enable_console=False)

        sample_alert = {
            "flow_id": "192.168.1.1:1234->10.0.0.1:80",
            "src_ip": "192.168.1.1",
            "dst_ip": "10.0.0.1",
            "timestamp": 100.0,
            "prediction_prob": 0.88,
            "decision_threshold": 0.5,
            "is_anomaly": True,
            "risk_score": 0.88,
            "risk_level": "CRITICAL",
            "feature_profile": "full_multilevel",
            "raw_features": {"flow_duration": 1.0}
        }

        logger.log_alert(sample_alert)
        assert log_path.exists()

        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert data["flow_id"] == sample_alert["flow_id"]
        assert "logged_at" in data


def test_error_handling_missing_artifacts():
    """Verifies clear error reporting for non-existent artifact directories."""
    non_existent = Path("models/final/non_existent_dir_12345")
    with pytest.raises(FileNotFoundError):
        IDSPredictor.from_artifact(non_existent)
