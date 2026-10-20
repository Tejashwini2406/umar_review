"""Privacy-Aware Data Minimization Module.

Enforces data minimization principles across the IoT IDS pipeline:
  - Operates strictly on 21 standardized flow-level statistical metrics (X in R^21)
  - Eliminates raw packet payload storage, user payload inspection, and PII retention
  - Verifies input vectors against raw payload leakages
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np


class DataMinimizationPolicy:
    """Formal definition of Privacy-Aware Data Minimization policy for IoT IDS."""

    ALLOWED_FEATURE_CATEGORIES: ClassVar[dict[str, str]] = {
        "flow_duration": "Flow duration statistic",
        "flow_bytes_per_sec": "Aggregated byte throughput rate",
        "flow_pkts_per_sec": "Aggregated packet throughput rate",
        "mean_pkt_size": "Statistical mean packet size",
        "payload_byte_ratio": "Aggregated payload size ratio",
        "pkt_count_ratio": "Aggregated packet-count ratio",
        "tcp_syn_ratio": "Control flag frequency statistic",
        "proto_tcp": "One-hot protocol identifier",
        "proto_udp": "One-hot protocol identifier",
        "proto_icmp": "One-hot protocol identifier",
        "proto_other": "One-hot protocol identifier",
        "temporal_iat_mean": "Inter-arrival timing statistic",
        "temporal_iat_cv": "Inter-arrival timing variability",
        "temporal_flow_rate_ewma": "Behavioral flow-rate EWMA",
        "temporal_byte_rate_ewma": "Behavioral byte-rate EWMA",
        "temporal_syn_rate_ewma": "Behavioral SYN-rate EWMA",
        "behavioral_dst_diversity": "Statistical destination diversity",
        "behavioral_port_entropy": "Statistical port entropy",
        "behavioral_fanout_ratio": "Behavioral fanout ratio",
        "behavioral_unanswered_ratio": "Behavioral unanswered-flow ratio",
        "behavioral_src_activity_ewma": "Behavioral source-activity EWMA",
    }

    FORBIDDEN_DATA_TYPES: ClassVar[list[str]] = [
        "raw_payload_bytes",
        "http_body_text",
        "user_passwords",
        "application_payload_hex",
        "device_mac_addresses",
        "raw_pcap_frames",
    ]

    @classmethod
    def verify_schema_minimization(cls, feature_names: list[str]) -> dict[str, Any]:
        """Require exactly the approved, unique 21-feature flow representation."""
        normalized = [name.casefold() for name in feature_names]
        allowed = {name.casefold() for name in cls.ALLOWED_FEATURE_CATEGORIES}
        duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
        unknown = sorted(name for name in set(normalized) if name not in allowed)
        missing = sorted(allowed - set(normalized))
        forbidden = [
            name
            for name in feature_names
            if name.casefold() in cls.FORBIDDEN_DATA_TYPES
            or "payload_content" in name.casefold()
            or "raw_hex" in name.casefold()
        ]
        violations = sorted(set(forbidden + unknown))
        is_compliant = (
            len(feature_names) == len(cls.ALLOWED_FEATURE_CATEGORIES)
            and not violations
            and not missing
            and not duplicates
        )
        return {
            "is_compliant": is_compliant,
            "total_features": len(feature_names),
            "allowed_categories": len(cls.ALLOWED_FEATURE_CATEGORIES),
            "violations_found": violations,
            "unknown_features": unknown,
            "missing_features": missing,
            "duplicate_features": duplicates,
            "policy_summary": "Requires the approved 21 flow-level features without raw payload storage.",
        }


def extract_minimized_representation(raw_flow_dict: dict[str, float]) -> np.ndarray:
    """Extracts standardized 21-D flow feature vector complying with data minimization."""
    ordered_cols = list(DataMinimizationPolicy.ALLOWED_FEATURE_CATEGORIES.keys())
    vec = np.zeros(len(ordered_cols), dtype=np.float32)

    for i, col in enumerate(ordered_cols):
        vec[i] = float(raw_flow_dict.get(col, 0.0))

    return vec
