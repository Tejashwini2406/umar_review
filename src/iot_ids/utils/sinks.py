"""Configurable Alert Sink Infrastructure for Multi-Level IoT IDS.

Provides extensible alert persistence sinks (JSONL File, Console, Multi-Sink)
while preserving the existing IDSAlertOutput contract 100% unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class BaseAlertSink(ABC):
    """Abstract base class for alert sinks."""

    @abstractmethod
    def write_alert(self, alert_dict: Dict[str, Any]) -> None:
        """Writes a single alert dictionary to the destination sink."""

    def write_batch(self, alerts: List[Dict[str, Any]]) -> None:
        """Batch writes multiple alert dictionaries."""
        for alert in alerts:
            self.write_alert(alert)


class JSONLFileSink(BaseAlertSink):
    """Appends structured alert records to a JSON lines file."""

    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def write_alert(self, alert_dict: Dict[str, Any]) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **alert_dict
        }
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class ConsoleAlertSink(BaseAlertSink):
    """Prints formatted alert notifications to stdout."""

    def __init__(self, min_level: str = "LOW"):
        self.min_level = min_level.upper()
        self.level_weights = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

    def write_alert(self, alert_dict: Dict[str, Any]) -> None:
        level = str(alert_dict.get("risk_level", "LOW")).upper()
        is_anom = bool(alert_dict.get("is_anomaly", False))
        if is_anom and self.level_weights.get(level, 0) >= self.level_weights.get(self.min_level, 0):
            print(
                f"[ALERT {level}] "
                f"Flow: {alert_dict.get('flow_id')} | "
                f"Prob: {alert_dict.get('prediction_prob', 0.0):.4f} | "
                f"Thresh: {alert_dict.get('decision_threshold', 0.5):.2f}"
            )


class MultiAlertSink(BaseAlertSink):
    """Multiplexes alert writes across multiple registered alert sinks."""

    def __init__(self, sinks: List[BaseAlertSink]):
        self.sinks = sinks

    def add_sink(self, sink: BaseAlertSink) -> None:
        self.sinks.append(sink)

    def write_alert(self, alert_dict: Dict[str, Any]) -> None:
        for sink in self.sinks:
            sink.write_alert(alert_dict)
