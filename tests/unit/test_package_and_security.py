"""Unit Tests for Package Entrypoints, Security Auditing, and Reliability."""

import json
import tempfile
from pathlib import Path
import pytest

from iot_ids.config import PipelineConfig
from iot_ids.registry.manager import ModelRegistry
from iot_ids.utils.sinks import JSONLFileSink


def test_registry_security_validations():
    """Verifies ModelRegistry validates artifact paths and metadata integrity securely."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Pointing to non-existent directory -> FileNotFoundError
        missing_dir = tmp_path / "missing_artifact_dir"
        with pytest.raises(FileNotFoundError):
            ModelRegistry.load_pipeline(missing_dir)

        # 2. Pointing to a file instead of a directory -> NotADirectoryError
        fake_file = tmp_path / "fake_file.txt"
        fake_file.write_text("not a dir", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            ModelRegistry.load_pipeline(fake_file)

        # 3. Directory missing metadata -> FileNotFoundError
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="pipeline_metadata.json"):
            ModelRegistry.load_pipeline(empty_dir)

        # 4. Corrupted JSON metadata -> ValueError
        meta_file = empty_dir / "pipeline_metadata.json"
        meta_file.write_text("{invalid_json: ", encoding="utf-8")
        with pytest.raises(ValueError, match="Corrupted or invalid"):
            ModelRegistry.load_pipeline(empty_dir)


def test_pipeline_config_json_loader():
    """Verifies PipelineConfig.save_json and load_json with validation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_file = Path(tmp_dir) / "config.json"
        config = PipelineConfig(
            artifact_dir="models/final/deployable_artifact",
            decision_threshold=0.65,
            source_domain="ToN-IoT",
            target_domain="Edge-IIoTset"
        )
        config.save_json(cfg_file)
        assert cfg_file.exists()

        loaded_cfg = PipelineConfig.load_json(cfg_file)
        assert loaded_cfg.decision_threshold == 0.65
        assert loaded_cfg.source_domain == "ToN-IoT"


def test_sink_file_handling():
    """Verifies JSONLFileSink creates directories and writes records cleanly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        jsonl_file = Path(tmp_dir) / "nested" / "dir" / "alerts.jsonl"
        sink = JSONLFileSink(jsonl_file)

        sink.write_alert({"flow_id": "f1", "prediction_prob": 0.99, "is_anomaly": True})
        assert jsonl_file.exists()

        records = jsonl_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(records) == 1
        data = json.loads(records[0])
        assert data["flow_id"] == "f1"
