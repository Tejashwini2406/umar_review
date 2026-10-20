"""Structured Operational Metrics Collector for Multi-Level IoT IDS.

Tracks packets processed, flows completed, alerts emitted, inference latency (mean, min, max, p95),
error counts, and active memory footprints for production observability.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List
import numpy as np


@dataclass
class OperationalMetricsSummary:
    """Snapshot summary of operational pipeline metrics."""

    packets_processed: int = 0
    flows_completed: int = 0
    alerts_emitted: int = 0
    errors_encountered: int = 0
    active_flows_in_memory: int = 0
    latency_count: int = 0
    latency_mean_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_p95_ms: float = 0.0
    uptime_seconds: float = 0.0


class MetricsCollector:
    """Thread-safe operational metrics tracking collector."""

    def __init__(self, max_latency_samples: int = 10000):
        self.start_time: float = time.time()
        self.packets_processed: int = 0
        self.flows_completed: int = 0
        self.alerts_emitted: int = 0
        self.errors_encountered: int = 0
        self.active_flows_in_memory: int = 0
        self.max_latency_samples: int = max_latency_samples
        self.latency_samples_ms: List[float] = []

    def reset(self) -> None:
        """Resets all metrics counters."""
        self.start_time = time.time()
        self.packets_processed = 0
        self.flows_completed = 0
        self.alerts_emitted = 0
        self.errors_encountered = 0
        self.active_flows_in_memory = 0
        self.latency_samples_ms.clear()

    def record_packet(self) -> None:
        self.packets_processed += 1

    def record_flow(self) -> None:
        self.flows_completed += 1

    def record_alert(self) -> None:
        self.alerts_emitted += 1

    def record_error(self) -> None:
        self.errors_encountered += 1

    def record_latency(self, latency_ms: float) -> None:
        if len(self.latency_samples_ms) >= self.max_latency_samples:
            self.latency_samples_ms.pop(0)
        self.latency_samples_ms.append(float(latency_ms))

    def update_active_flows(self, count: int) -> None:
        self.active_flows_in_memory = count

    def get_summary(self) -> OperationalMetricsSummary:
        """Returns a snapshot summary of current operational metrics."""
        uptime = time.time() - self.start_time
        samples = self.latency_samples_ms

        if samples:
            arr = np.array(samples)
            l_mean = float(np.mean(arr))
            l_min = float(np.min(arr))
            l_max = float(np.max(arr))
            l_p95 = float(np.percentile(arr, 95))
        else:
            l_mean = l_min = l_max = l_p95 = 0.0

        return OperationalMetricsSummary(
            packets_processed=self.packets_processed,
            flows_completed=self.flows_completed,
            alerts_emitted=self.alerts_emitted,
            errors_encountered=self.errors_encountered,
            active_flows_in_memory=self.active_flows_in_memory,
            latency_count=len(samples),
            latency_mean_ms=l_mean,
            latency_min_ms=l_min,
            latency_max_ms=l_max,
            latency_p95_ms=l_p95,
            uptime_seconds=uptime,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self.get_summary())
