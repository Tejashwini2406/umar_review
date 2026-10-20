"""Operational System Configuration Module for Multi-Level IoT IDS.

Provides PipelineConfig dataclass with fail-fast validation and JSON/YAML management for
deployable model artifacts, feature profiles, domain settings, thresholds, and logging.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
from typing import Dict, Any, Optional

from iot_ids.features.canonical.flow_builder import CANONICAL_18_FEATURE_NAMES


@dataclass
class PipelineConfig:
    """Production configuration parameters for multi-level IoT IDS deployment."""

    artifact_dir: str = "models/final/deployable_artifact"
    feature_profile: str = "full_multilevel"
    source_domain: str = "ToN-IoT"
    target_domain: Optional[str] = "Edge-IIoTset"
    decision_threshold: float = 0.50
    model_family: str = "RandomForest"
    n_estimators: int = 100
    max_depth: Optional[int] = 12
    feature_names: list[str] = field(default_factory=lambda: list(CANONICAL_18_FEATURE_NAMES))
    inactivity_timeout: float = 15.0
    window_seconds: float = 30.0
    max_active_flows: int = 10000
    interface: Optional[str] = None
    bpf_filter: Optional[str] = "ip"
    capture_duration: Optional[float] = None
    pcap_file: Optional[str] = None
    log_level: str = "INFO"

    def validate(self) -> None:
        """Enforces fail-fast configuration validation for production deployment."""
        if not self.artifact_dir or not isinstance(self.artifact_dir, str):
            raise ValueError("Config error: artifact_dir must be a non-empty string path.")

        if not (0.01 <= self.decision_threshold <= 0.99):
            raise ValueError(f"Config error: decision_threshold ({self.decision_threshold}) must be in range [0.01, 0.99].")

        if self.inactivity_timeout <= 0.0:
            raise ValueError(f"Config error: inactivity_timeout ({self.inactivity_timeout}) must be positive.")

        if self.window_seconds <= 0.0:
            raise ValueError(f"Config error: window_seconds ({self.window_seconds}) must be positive.")

        if self.max_active_flows <= 0:
            raise ValueError(f"Config error: max_active_flows ({self.max_active_flows}) must be positive.")

        valid_models = {"randomforest", "rf", "logisticregression", "lr"}
        if self.model_family.lower() not in valid_models:
            raise ValueError(f"Config error: Unsupported model_family '{self.model_family}'. Must be one of {valid_models}.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary representation."""
        return asdict(self)

    def save_json(self, json_path: Path) -> Path:
        """Save configuration to JSON file."""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return json_path

    @classmethod
    def load_json(cls, json_path: Path) -> PipelineConfig:
        """Load configuration from JSON file and validate."""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = cls(**data)
        config.validate()
        return config
