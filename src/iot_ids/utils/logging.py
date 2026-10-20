"""Structured Observability and Alert Logging Module.

Provides IDSAlertLogger to emit structured JSON lines log outputs for prediction alerts,
performance tracking, and security auditing.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class IDSAlertLogger:
    """Structured JSON lines logger for operational alert outputs and audit telemetry."""

    def __init__(self, log_path: Optional[Path] = None, enable_console: bool = True):
        self.log_path = Path(log_path) if log_path else None
        self.enable_console = enable_console

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_alert(self, alert_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Logs a single prediction alert record with an ISO timestamp."""
        log_record = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            **alert_dict
        }

        log_json = json.dumps(log_record)

        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(log_json + "\n")

        if self.enable_console and alert_dict.get("is_anomaly", False):
            print(
                f"[ALERT {alert_dict.get('risk_level', 'HIGH')}] "
                f"Flow: {alert_dict.get('flow_id')} | "
                f"Prob: {alert_dict.get('prediction_prob', 0.0):.4f} | "
                f"Thresh: {alert_dict.get('decision_threshold', 0.5):.2f}"
            )

        return log_record

    def log_batch(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Logs a batch list of prediction alert records."""
        return [self.log_alert(a) for a in alerts]
