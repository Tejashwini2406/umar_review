"""Flow Reconstruction Engine for Packet/Frame-Level Telemetry.

Aggregates individual network frame records (e.g., Edge-IIoTset Wireshark dumps)
into bi-directional CanonicalFlow records using strict 5-tuple grouping,
inactivity timeouts, and maximum duration boundaries.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import pandas as pd

from iot_ids.data.flow_object import CanonicalFlow


FiveTuple = Tuple[str, str, int, int, str]


def normalize_5tuple(
    src_ip: str, dst_ip: str, src_port: int, dst_port: int, proto: str
    
) -> Tuple[FiveTuple, bool]:
    """Sorts 5-tuple canonical direction so (A->B) and (B->A) map to the same flow key.
    
    Returns:
        ((canonical_src, canonical_dst, canonical_src_port, canonical_dst_port, proto), is_forward)
    """
    forward_key = (src_ip, dst_ip, src_port, dst_port, proto)
    reverse_key = (dst_ip, src_ip, dst_port, src_port, proto)

    if forward_key <= reverse_key:
        return forward_key, True
    return reverse_key, False


class ActiveFlowState:
    """Internal mutable tracking buffer for an ongoing bi-directional flow."""

    def __init__(self, key: FiveTuple, first_timestamp: float, label: int, attack_category: str):
        self.key = key
        self.src_host, self.dst_host, self.src_port, self.dst_port, self.protocol = key
        self.timestamp_start = first_timestamp
        self.timestamp_end = first_timestamp
        self.bytes_src = 0
        self.bytes_dst = 0
        self.pkts_src = 0
        self.pkts_dst = 0
        self.syn_count_src = 0
        self.ack_count_src = 0
        self.rst_count_src = 0
        self.has_rst = False
        self.has_rej = False
        self.label = label
        self.attack_category = attack_category

    def update(
        self,
        timestamp: float,
        is_forward: bool,
        length: int,
        syn: int = 0,
        ack: int = 0,
        rst: int = 0,
    ) -> None:
        """Update flow state with a new frame observation."""
        self.timestamp_end = max(self.timestamp_end, timestamp)
        if is_forward:
            self.bytes_src += length
            self.pkts_src += 1
            self.syn_count_src += syn
            self.ack_count_src += ack
            self.rst_count_src += rst
        else:
            self.bytes_dst += length
            self.pkts_dst += 1

        if rst > 0:
            self.has_rst = True

    def finalize(self) -> CanonicalFlow:
        """Convert active state into an immutable CanonicalFlow record."""
        conn_code = 0  # Normal / SF
        if self.has_rst:
            conn_code = 2  # Reset
        elif self.pkts_dst == 0 and self.pkts_src > 0:
            conn_code = 1  # Unanswered / Rejected

        return CanonicalFlow(
            timestamp_start=float(self.timestamp_start),
            timestamp_end=float(self.timestamp_end),
            src_host=self.src_host,
            dst_host=self.dst_host,
            src_port=int(self.src_port),
            dst_port=int(self.dst_port),
            protocol=self.protocol,
            bytes_src=int(self.bytes_src),
            bytes_dst=int(self.bytes_dst),
            pkts_src=int(self.pkts_src),
            pkts_dst=int(self.pkts_dst),
            syn_count_src=int(self.syn_count_src),
            ack_count_src=int(self.ack_count_src),
            rst_count_src=int(self.rst_count_src),
            conn_state_code=int(conn_code),
            label=int(self.label),
            attack_category=str(self.attack_category),
        )


class FlowAggregator:
    """Bi-directional flow reconstructor for frame-level packet logs.
    
    Args:
        inactivity_timeout: Maximum idle duration (seconds) before a flow is closed. Default: 15.0s.
        max_flow_duration: Maximum total duration (seconds) before a flow is closed. Default: 120.0s.
    """

    def __init__(self, inactivity_timeout: float = 15.0, max_flow_duration: float = 120.0, max_active_flows: int = 10000):
        self.inactivity_timeout = inactivity_timeout
        self.max_flow_duration = max_flow_duration
        self.max_active_flows = max_active_flows
        self.active_flows: Dict[FiveTuple, ActiveFlowState] = {}

    def _evict_oldest_flow_if_full(self) -> Optional[CanonicalFlow]:
        """Evicts the oldest flow if the active flow memory limit is reached."""
        if len(self.active_flows) >= self.max_active_flows:
            oldest_key = min(self.active_flows.keys(), key=lambda k: self.active_flows[k].timestamp_end)
            oldest_flow = self.active_flows.pop(oldest_key)
            return oldest_flow.finalize()
        return None

    def clear(self) -> None:
        """Clears all active flows from internal state buffer."""
        self.active_flows.clear()

    def add_packet(self, packet_dict: Dict[str, Any]) -> Optional[CanonicalFlow]:
        """Convenience method wrapping process_frame for dictionary packet records."""
        return self.process_frame(
            timestamp=float(packet_dict.get("timestamp", 0.0)),
            src_ip=str(packet_dict.get("src_ip", packet_dict.get("src_host", "0.0.0.0"))),
            dst_ip=str(packet_dict.get("dst_ip", packet_dict.get("dst_host", "0.0.0.0"))),
            src_port=int(packet_dict.get("src_port", 0)),
            dst_port=int(packet_dict.get("dst_port", 0)),
            proto=str(packet_dict.get("protocol", packet_dict.get("proto", "tcp"))).lower(),
            length=int(packet_dict.get("length", packet_dict.get("bytes", 0))),
            syn=int(packet_dict.get("syn", 0)),
            ack=int(packet_dict.get("ack", 0)),
            rst=int(packet_dict.get("rst", 0)),
            label=int(packet_dict.get("label", 0)),
            attack_category=str(packet_dict.get("attack_category", "Normal")),
        )

    def process_frame(
        self,
        timestamp: float,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        proto: str,
        length: int,
        syn: int = 0,
        ack: int = 0,
        rst: int = 0,
        label: int = 0,
        attack_category: str = "Normal",
    ) -> Optional[CanonicalFlow]:
        """Ingests a single packet frame and returns a finalized CanonicalFlow if expired."""
        key, is_forward = normalize_5tuple(src_ip, dst_ip, src_port, dst_port, proto)
        expired_flow: Optional[CanonicalFlow] = None

        if key in self.active_flows:
            flow = self.active_flows[key]
            idle_time = timestamp - flow.timestamp_end
            active_time = timestamp - flow.timestamp_start

            if idle_time > self.inactivity_timeout or active_time > self.max_flow_duration:
                expired_flow = flow.finalize()
                # Initialize new flow state for current frame
                self.active_flows[key] = ActiveFlowState(key, timestamp, label, attack_category)
                self.active_flows[key].update(timestamp, is_forward, length, syn, ack, rst)
            else:
                flow.update(timestamp, is_forward, length, syn, ack, rst)
        else:
            if len(self.active_flows) >= self.max_active_flows:
                expired_flow = self._evict_oldest_flow_if_full()
            self.active_flows[key] = ActiveFlowState(key, timestamp, label, attack_category)
            self.active_flows[key].update(timestamp, is_forward, length, syn, ack, rst)

        return expired_flow

    def flush(self) -> List[CanonicalFlow]:
        """Flushes all remaining active flows into finalized CanonicalFlow records."""
        finalized = [flow.finalize() for flow in self.active_flows.values()]
        self.active_flows.clear()
        return finalized
