"""Canonical Flow Representation Object for Multi-Dataset Ingestion.

Defines the single standardized dataclass used across all telemetry sources
(ToN-IoT, Edge-IIoTset, NF-ToN-IoT-v2, CICIoT2023).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


VALID_PROTOCOLS = {"tcp", "udp", "icmp", "other"}
VALID_LABELS = {0, 1}


@dataclass(frozen=True)
class CanonicalFlow:
    """Canonical Bi-Directional Flow Record.
    
    Attributes:
        timestamp_start: Flow start UNIX timestamp in seconds (float64).
        timestamp_end: Flow end UNIX timestamp in seconds (float64).
        src_host: Anonymized/hashed source host identifier.
        dst_host: Anonymized/hashed destination host identifier.
        src_port: Transport layer source port (0-65535).
        dst_port: Transport layer destination port (0-65535).
        protocol: Transport protocol string ('tcp', 'udp', 'icmp', 'other').
        bytes_src: Total payload/header bytes sent by source host.
        bytes_dst: Total payload/header bytes sent by destination host.
        pkts_src: Total packet count sent by source host.
        pkts_dst: Total packet count sent by destination host.
        syn_count_src: TCP SYN flag count sent by source host.
        ack_count_src: TCP ACK flag count sent by source host.
        rst_count_src: TCP RST flag count sent by source host.
        conn_state_code: Connection status code (0: Normal/SF, 1: Rejected/REJ, 2: Reset, 3: Attempt/S0).
        label: Binary attack label (0: Benign, 1: Attack).
        attack_category: Standardized attack category taxonomy string (e.g., 'Normal', 'DDoS', 'Scanning').
    """

    timestamp_start: float
    timestamp_end: float
    src_host: str
    dst_host: str
    src_port: int
    dst_port: int
    protocol: str
    bytes_src: int
    bytes_dst: int
    pkts_src: int
    pkts_dst: int
    syn_count_src: int
    ack_count_src: int
    rst_count_src: int
    conn_state_code: int
    label: int
    attack_category: str

    def __post_init__(self) -> None:
        """Validate canonical flow attributes to guarantee dataset-agnostic integrity."""
        if self.timestamp_end < self.timestamp_start:
            raise ValueError(
                f"Invalid duration: timestamp_end ({self.timestamp_end}) < "
                f"timestamp_start ({self.timestamp_start})"
            )
        if self.bytes_src < 0 or self.bytes_dst < 0:
            raise ValueError(f"Negative byte count: bytes_src={self.bytes_src}, bytes_dst={self.bytes_dst}")
        if self.pkts_src < 0 or self.pkts_dst < 0:
            raise ValueError(f"Negative packet count: pkts_src={self.pkts_src}, pkts_dst={self.pkts_dst}")
        if self.protocol not in VALID_PROTOCOLS:
            raise ValueError(f"Invalid protocol: {self.protocol}. Must be one of {VALID_PROTOCOLS}")
        if self.label not in VALID_LABELS:
            raise ValueError(f"Invalid binary label: {self.label}. Must be 0 or 1.")

    @property
    def duration(self) -> float:
        """Flow duration in seconds."""
        return max(0.0, self.timestamp_end - self.timestamp_start)

    @property
    def total_bytes(self) -> int:
        """Total flow payload bytes (source + destination)."""
        return self.bytes_src + self.bytes_dst

    @property
    def total_pkts(self) -> int:
        """Total flow packets (source + destination)."""
        return self.pkts_src + self.pkts_dst

    def to_dict(self) -> Dict[str, Any]:
        """Convert flow record to a standard dictionary representation."""
        res = asdict(self)
        res["duration"] = self.duration
        res["total_bytes"] = self.total_bytes
        res["total_pkts"] = self.total_pkts
        return res
