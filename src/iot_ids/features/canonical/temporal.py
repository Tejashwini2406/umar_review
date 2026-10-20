"""Level B: Causal Temporal Representation Extractor.

Tracks causal arrival dynamics, inter-arrival time (IAT) variance, and EWMA throughput/flow rates.

Strict Causal Execution Invariant:
  1. READ PREVIOUS STATE (H_{<t_i})
  2. COMPUTE CURRENT FEATURES
  3. UPDATE STATE (H_{t_i})
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from iot_ids.data.flow_object import CanonicalFlow

TEMPORAL_FEATURE_NAMES = [
    "temporal_iat_mean",
    "temporal_iat_cv",
    "temporal_flow_rate_ewma",
    "temporal_byte_rate_ewma",
    "temporal_syn_rate_ewma",
]


class HostTemporalState:
    """State memory tracker for a single source host entity."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.last_timestamp: Optional[float] = None
        self.iat_mean: float = 0.0
        self.iat_variance: float = 0.0
        self.flow_rate_ewma: float = 0.0
        self.byte_rate_ewma: float = 0.0
        self.syn_rate_ewma: float = 0.0
        self.event_count: int = 0

    def update(self, timestamp: float, duration: float, total_bytes: float, syn_count: float) -> None:
        """Update state with current event parameters AFTER feature extraction."""
        if self.last_timestamp is not None:
            delta_t = max(0.0, timestamp - self.last_timestamp)
            # Recursive EWMA update for IAT mean and variance
            diff = delta_t - self.iat_mean
            self.iat_mean += self.alpha * diff
            self.iat_variance = (1.0 - self.alpha) * (self.iat_variance + self.alpha * (diff ** 2))

            curr_flow_rate = 1.0 / (delta_t + 1e-6)
            self.flow_rate_ewma = (1.0 - self.alpha) * self.flow_rate_ewma + self.alpha * curr_flow_rate
        else:
            self.iat_mean = 0.0
            self.iat_variance = 0.0
            self.flow_rate_ewma = 1.0 / (duration + 1e-6)

        self.last_timestamp = timestamp
        curr_byte_rate = total_bytes / (duration + 1e-6)
        self.byte_rate_ewma = (1.0 - self.alpha) * self.byte_rate_ewma + self.alpha * curr_byte_rate

        curr_syn_flag = 1.0 if syn_count > 0 else 0.0
        self.syn_rate_ewma = (1.0 - self.alpha) * self.syn_rate_ewma + self.alpha * curr_syn_flag

        self.event_count += 1


class TemporalFeatureExtractor:
    """Extracts Level B Causal Temporal Features using host state tracking."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.host_states: Dict[str, HostTemporalState] = {}
        self.first_event_count: int = 0
        self.state_update_count: int = 0

    def reset_state(self) -> None:
        """Clears all host state memory. Call at split boundaries and domain transfers."""
        self.host_states.clear()
        self.first_event_count = 0
        self.state_update_count = 0

    def extract_and_update(self, flow: CanonicalFlow) -> Dict[str, float]:
        """Extracts temporal features strictly reading previous state, then updates state."""
        src_ip = flow.src_host
        eps = 1e-6

        # Step 1: Read previous state (or initialize first-event state)
        if src_ip not in self.host_states:
            self.host_states[src_ip] = HostTemporalState(alpha=self.alpha)
            self.first_event_count += 1
            # First-event explicit initialization policy
            iat_mean = 0.0
            iat_cv = 0.0
            flow_rate_ewma = 0.0
            byte_rate_ewma = 0.0
            syn_rate_ewma = 0.0
        else:
            state = self.host_states[src_ip]
            iat_mean = state.iat_mean
            std_dev = np.sqrt(max(0.0, state.iat_variance))
            iat_cv = std_dev / (iat_mean + eps)
            flow_rate_ewma = state.flow_rate_ewma
            byte_rate_ewma = state.byte_rate_ewma
            syn_rate_ewma = state.syn_rate_ewma

        features = {
            "temporal_iat_mean": float(iat_mean),
            "temporal_iat_cv": float(np.clip(iat_cv, 0.0, 100.0)),
            "temporal_flow_rate_ewma": float(flow_rate_ewma),
            "temporal_byte_rate_ewma": float(byte_rate_ewma),
            "temporal_syn_rate_ewma": float(syn_rate_ewma),
        }

        # Step 2: Update state AFTER feature computation
        self.host_states[src_ip].update(
            timestamp=flow.timestamp_start,
            duration=flow.duration,
            total_bytes=float(flow.total_bytes),
            syn_count=float(flow.syn_count_src),
        )
        self.state_update_count += 1

        return features

    def extract_vector(self, flow: CanonicalFlow) -> List[float]:
        """Returns ordered float vector matching TEMPORAL_FEATURE_NAMES."""
        d = self.extract_and_update(flow)
        return [d[k] for k in TEMPORAL_FEATURE_NAMES]
