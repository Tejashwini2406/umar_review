"""Production Inference Engine for Multi-Level IoT Intrusion Detection.

Provides a clean predictor API for canonical flows, raw packet streams, and batch DataFrames,
delegating pipeline execution to IDSSystemPipeline while tracking operational metrics and health status.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd

from iot_ids.data.flow_object import CanonicalFlow
from iot_ids.pipeline.system import IDSSystemPipeline
from iot_ids.registry.manager import ModelRegistry
from iot_ids.utils.metrics import MetricsCollector


class IDSPredictor:
    """Deployable inference predictor for multi-level IoT intrusion detection."""

    def __init__(self, pipeline: IDSSystemPipeline):
        self.pipeline = pipeline
        self.metrics = MetricsCollector()

    @classmethod
    def from_artifact(cls, artifact_dir: Union[str, Path]) -> IDSPredictor:
        """Instantiates an IDSPredictor from a saved ModelRegistry pipeline directory."""
        pipeline = ModelRegistry.load_pipeline(Path(artifact_dir))
        return cls(pipeline=pipeline)

    def health_check(self) -> Dict[str, Any]:
        """Returns health status inspection combining pipeline state and operational metrics."""
        pipe_health = self.pipeline.health_check()
        summary = self.metrics.to_dict()
        return {
            "pipeline": pipe_health,
            "metrics": summary,
        }

    def reset_state(self) -> None:
        """Resets temporal and behavioral state buffers and operational metrics."""
        self.pipeline.reset_state()
        self.metrics.reset()

    def predict_flow(self, flow: CanonicalFlow) -> Dict[str, Any]:
        """Runs end-to-end inference on a single CanonicalFlow object with latency tracking."""
        t0 = time.time()
        try:
            alert = self.pipeline.process_flow(flow)
            t_lat = (time.time() - t0) * 1000.0
            self.metrics.record_flow()
            self.metrics.record_latency(t_lat)
            if alert.get("is_anomaly", False):
                self.metrics.record_alert()
            return alert
        except Exception:
            self.metrics.record_error()
            raise

    def predict_dataframe(self, df_flows: pd.DataFrame) -> List[Dict[str, Any]]:
        """Runs batch inference over a DataFrame of canonical flow attributes."""
        t0 = time.time()
        try:
            alerts = self.pipeline.process_dataframe(df_flows)
            t_lat = (time.time() - t0) * 1000.0 / max(len(alerts), 1)
            for a in alerts:
                self.metrics.record_flow()
                self.metrics.record_latency(t_lat)
                if a.get("is_anomaly", False):
                    self.metrics.record_alert()
            return alerts
        except Exception:
            self.metrics.record_error()
            raise

    def process_raw_packet(self, packet_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Ingests a raw packet dictionary into the FlowAggregator.
        
        Returns an alert decision if the packet completes/evicts a flow; otherwise returns None.
        """
        self.metrics.record_packet()
        try:
            completed_flow = self.pipeline.flow_aggregator.add_packet(packet_dict)
            self.metrics.update_active_flows(len(self.pipeline.flow_aggregator.active_flows))
        except Exception:
            self.metrics.record_error()
            raise
        if completed_flow is not None:
            return self.predict_flow(completed_flow)
        return None
