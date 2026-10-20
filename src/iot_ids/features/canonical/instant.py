"""Level A: Instantaneous Flow Representation Extractor.

Computes stateless, event-level rate, ratio, size, and protocol metrics from a single CanonicalFlow record.
"""
from __future__ import annotations

from typing import Dict, List
import numpy as np

from iot_ids.data.flow_object import CanonicalFlow

INSTANT_FEATURE_NAMES = [
    "flow_duration",
    "flow_bytes_per_sec",
    "flow_pkts_per_sec",
    "mean_pkt_size",
    "payload_byte_ratio",
    "pkt_count_ratio",
    "tcp_syn_ratio",
    "proto_tcp",
    "proto_udp",
    "proto_icmp",
    "proto_other",
]


class InstantaneousFeatureExtractor:
    """Extracts Level A instantaneous features from a CanonicalFlow record."""

    def extract(self, flow: CanonicalFlow) -> Dict[str, float]:
        """Computes Level A instantaneous feature dictionary from flow event.
        
        No future info, no host state memory, no dataset labels used.
        """
        dur = max(0.0, flow.duration)
        eps = 1e-6

        total_bytes = float(flow.total_bytes)
        total_pkts = float(flow.total_pkts)

        bytes_per_sec = total_bytes / (dur + eps)
        pkts_per_sec = total_pkts / (dur + eps)
        mean_size = total_bytes / (total_pkts + eps)

        byte_ratio = float(flow.bytes_src) / (total_bytes + eps)
        pkt_ratio = float(flow.pkts_src) / (total_pkts + eps)
        syn_ratio = float(flow.syn_count_src) / (total_pkts + eps)

        pr = flow.protocol.lower()
        proto_tcp = 1.0 if pr == "tcp" else 0.0
        proto_udp = 1.0 if pr == "udp" else 0.0
        proto_icmp = 1.0 if pr == "icmp" else 0.0
        proto_other = 1.0 if pr not in ("tcp", "udp", "icmp") else 0.0

        return {
            "flow_duration": dur,
            "flow_bytes_per_sec": bytes_per_sec,
            "flow_pkts_per_sec": pkts_per_sec,
            "mean_pkt_size": mean_size,
            "payload_byte_ratio": float(np.clip(byte_ratio, 0.0, 1.0)),
            "pkt_count_ratio": float(np.clip(pkt_ratio, 0.0, 1.0)),
            "tcp_syn_ratio": float(np.clip(syn_ratio, 0.0, 1.0)),
            "proto_tcp": proto_tcp,
            "proto_udp": proto_udp,
            "proto_icmp": proto_icmp,
            "proto_other": proto_other,
        }

    def extract_vector(self, flow: CanonicalFlow) -> List[float]:
        """Returns ordered float vector matching INSTANT_FEATURE_NAMES."""
        d = self.extract(flow)
        return [d[k] for k in INSTANT_FEATURE_NAMES]
