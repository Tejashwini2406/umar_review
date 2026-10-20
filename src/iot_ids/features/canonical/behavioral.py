"""Level C: Causal Behavioral Representation Extractor.

Tracks host interaction topology, destination diversity, destination port entropy,
fan-out ratios, and unanswered connection ratios over a 30-second sliding time window.

Strict Causal Execution Invariant:
  1. EVICT OLD EVENTS (t < current_t - 30.0s)
  2. READ PREVIOUS STATE (H_{<t_i})
  3. COMPUTE CURRENT FEATURES
  4. UPDATE STATE (H_{t_i})
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple
import numpy as np

from iot_ids.data.flow_object import CanonicalFlow

BEHAVIORAL_FEATURE_NAMES = [
    "behavioral_dst_diversity",
    "behavioral_port_entropy",
    "behavioral_fanout_ratio",
    "behavioral_unanswered_ratio",
    "behavioral_src_activity_ewma",
]


class HostBehavioralState:
    """Sliding time window state buffer for host interaction context (default window = 30.0s)."""

    def __init__(self, window_seconds: float = 30.0, alpha: float = 0.1):
        self.window_seconds = window_seconds
        self.alpha = alpha
        # Event store: list of tuples (timestamp, dst_host, dst_port, conn_state_code)
        self.events: List[Tuple[float, str, int, int]] = []
        self.activity_ewma: float = 0.0

    def evict_expired(self, current_time: float) -> int:
        """Removes events older than (current_time - window_seconds). Returns eviction count."""
        cutoff = current_time - self.window_seconds
        orig_len = len(self.events)
        self.events = [e for e in self.events if e[0] >= cutoff]
        return orig_len - len(self.events)

    def compute_metrics(self) -> Tuple[float, float, float, float, float]:
        """Computes behavioral features from current active window buffer."""
        if not self.events:
            return 0.0, 0.0, 0.0, 0.0, float(self.activity_ewma)

        eps = 1e-6
        total_flows = float(len(self.events))

        dst_hosts = {e[1] for e in self.events}
        dst_diversity = float(len(dst_hosts))

        dst_ports = [e[2] for e in self.events]
        port_counts = Counter(dst_ports)
        entropy = 0.0
        for p, count in port_counts.items():
            prob = float(count) / total_flows
            entropy -= prob * np.log2(prob + eps)

        fanout = dst_diversity / (total_flows + eps)

        unanswered = sum(1 for e in self.events if e[3] in (1, 2))
        unanswered_ratio = float(unanswered) / (total_flows + eps)

        return dst_diversity, float(entropy), float(fanout), float(unanswered_ratio), float(self.activity_ewma)

    def add_event(self, timestamp: float, dst_host: str, dst_port: int, conn_state_code: int) -> None:
        """Adds event to active buffer AFTER feature computation."""
        self.events.append((timestamp, dst_host, dst_port, conn_state_code))
        self.activity_ewma = (1.0 - self.alpha) * self.activity_ewma + self.alpha * 1.0


class BehavioralFeatureExtractor:
    """Extracts Level C Causal Behavioral Features with 30s sliding window state tracking."""

    def __init__(self, window_seconds: float = 30.0, alpha: float = 0.1):
        self.window_seconds = window_seconds
        self.alpha = alpha
        self.host_states: Dict[str, HostBehavioralState] = {}
        self.total_evictions: int = 0
        self.first_event_count: int = 0
        self.state_update_count: int = 0

    def reset_state(self) -> None:
        """Clears all behavioral host states. Call at split boundaries and domain transfers."""
        self.host_states.clear()
        self.total_evictions = 0
        self.first_event_count = 0
        self.state_update_count = 0

    def extract_and_update(self, flow: CanonicalFlow) -> Dict[str, float]:
        """Evicts expired events, reads prior state, computes features, then appends current event."""
        src_ip = flow.src_host
        curr_time = flow.timestamp_start

        # Step 1: Initialize host state if first event
        if src_ip not in self.host_states:
            self.host_states[src_ip] = HostBehavioralState(
                window_seconds=self.window_seconds, alpha=self.alpha
            )
            self.first_event_count += 1

        state = self.host_states[src_ip]

        # Step 2: Evict expired events outside 30.0s window
        evicted = state.evict_expired(curr_time)
        self.total_evictions += evicted

        # Step 3: READ PREVIOUS STATE & COMPUTE FEATURES
        dst_div, port_ent, fanout, unans_ratio, act_ewma = state.compute_metrics()

        features = {
            "behavioral_dst_diversity": dst_div,
            "behavioral_port_entropy": port_ent,
            "behavioral_fanout_ratio": float(np.clip(fanout, 0.0, 1.0)),
            "behavioral_unanswered_ratio": float(np.clip(unans_ratio, 0.0, 1.0)),
            "behavioral_src_activity_ewma": float(act_ewma),
        }

        # Step 4: UPDATE STATE AFTER feature computation
        state.add_event(
            timestamp=curr_time,
            dst_host=flow.dst_host,
            dst_port=flow.dst_port,
            conn_state_code=flow.conn_state_code,
        )
        self.state_update_count += 1

        return features

    def extract_vector(self, flow: CanonicalFlow) -> List[float]:
        """Returns ordered float vector matching BEHAVIORAL_FEATURE_NAMES."""
        d = self.extract_and_update(flow)
        return [d[k] for k in BEHAVIORAL_FEATURE_NAMES]
