"""Production Scapy Packet Capture and PCAP Replay Ingestion Layer.

Converts Scapy packets from live interfaces or offline .pcap/.pcapng files
into canonical packet dictionaries, feeding the existing FlowAggregator -> CanonicalFlowBuilder -> IDSPredictor pipeline.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import scapy.all as scapy
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.inet6 import IPv6

from iot_ids.inference.predictor import IDSPredictor
from iot_ids.utils.logging import IDSAlertLogger


def scapy_to_canonical_packet(pkt: Any) -> Optional[Dict[str, Any]]:
    """Converts a Scapy packet object into a canonical packet dictionary.
    
    Returns None if packet is not IP/IPv6 or is malformed.
    """
    try:
        timestamp = float(getattr(pkt, "time", time.time()))
        length = int(len(pkt))

        if pkt.haslayer(IP):
            src_ip = str(pkt[IP].src)
            dst_ip = str(pkt[IP].dst)
        elif pkt.haslayer(IPv6):
            src_ip = str(pkt[IPv6].src)
            dst_ip = str(pkt[IPv6].dst)
        else:
            return None  # Ignore non-IP (ARP, STP, etc.)

        src_port = 0
        dst_port = 0
        protocol = "other"
        syn = 0
        ack = 0
        rst = 0

        if pkt.haslayer(TCP):
            protocol = "tcp"
            src_port = int(pkt[TCP].sport)
            dst_port = int(pkt[TCP].dport)
            flags = str(pkt[TCP].flags)
            syn = 1 if "S" in flags else 0
            ack = 1 if "A" in flags else 0
            rst = 1 if "R" in flags else 0
        elif pkt.haslayer(UDP):
            protocol = "udp"
            src_port = int(pkt[UDP].sport)
            dst_port = int(pkt[UDP].dport)
        elif pkt.haslayer(ICMP):
            protocol = "icmp"

        return {
            "timestamp": timestamp,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "length": length,
            "syn": syn,
            "ack": ack,
            "rst": rst,
            "label": 0,
            "attack_category": "Normal",
        }
    except Exception:
        return None  # Handle malformed packets gracefully


class PacketCaptureEngine:
    """Production capture engine supporting offline PCAP replay and live interface sniffing."""

    def __init__(self, predictor: IDSPredictor, logger: Optional[IDSAlertLogger] = None):
        self.predictor = predictor
        self.logger = logger or IDSAlertLogger(enable_console=True)

    def replay_pcap(
        self,
        pcap_path: Path | str,
        bpf_filter: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Replays an offline .pcap / .pcapng file through the operational IDS pipeline."""
        pcap_path = Path(pcap_path)
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or None")
        if not pcap_path.exists():
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

        self.predictor.reset_state()
        emitted_alerts: List[Dict[str, Any]] = []
        packets_processed = 0
        if limit == 0:
            return emitted_alerts

        try:
            reader = scapy.PcapReader(str(pcap_path))
            for pkt in reader:
                if limit is not None and packets_processed >= limit:
                    break

                packets_processed += 1
                canon_pkt = scapy_to_canonical_packet(pkt)
                if canon_pkt is None:
                    continue

                alert = self.predictor.process_raw_packet(canon_pkt)
                if alert is not None:
                    logged = self.logger.log_alert(alert)
                    emitted_alerts.append(logged)
            reader.close()
        except Exception as e:
            if packets_processed == 0 and "empty" in str(e).lower():
                return []
            raise RuntimeError(f"Error reading PCAP file {pcap_path}: {e}") from e

        # Flush remaining active flows at end of PCAP replay
        flushed_flows = self.predictor.pipeline.flow_aggregator.flush()
        for flow in flushed_flows:
            alert = self.predictor.predict_flow(flow)
            logged = self.logger.log_alert(alert)
            emitted_alerts.append(logged)

        return emitted_alerts

    def capture_live(
        self,
        interface: Optional[str] = None,
        bpf_filter: Optional[str] = None,
        duration: Optional[float] = None,
        count: Optional[int] = None,
        stop_filter: Optional[Callable[[Any], bool]] = None
    ) -> List[Dict[str, Any]]:
        """Captures live packets from a network interface and feeds the IDS pipeline."""
        if count is not None and count < 0:
            raise ValueError("count must be non-negative or None")
        if duration is not None and duration < 0:
            raise ValueError("duration must be non-negative or None")

        self.predictor.reset_state()
        emitted_alerts: List[Dict[str, Any]] = []
        if count == 0 or duration == 0:
            return emitted_alerts

        def packet_callback(pkt: Any) -> None:
            canon_pkt = scapy_to_canonical_packet(pkt)
            if canon_pkt is not None:
                alert = self.predictor.process_raw_packet(canon_pkt)
                if alert is not None:
                    logged = self.logger.log_alert(alert)
                    emitted_alerts.append(logged)

        try:
            scapy.sniff(
                iface=interface,
                filter=bpf_filter,
                prn=packet_callback,
                timeout=duration,
                count=count if count is not None else 0,
                stop_filter=stop_filter,
                store=0
            )
        except KeyboardInterrupt:
            print("\nLive capture interrupted by user.")
        except Exception as e:
            print(f"Live capture error on interface {interface}: {e}")

        # Flush active flows on capture shutdown
        flushed_flows = self.predictor.pipeline.flow_aggregator.flush()
        for flow in flushed_flows:
            alert = self.predictor.predict_flow(flow)
            logged = self.logger.log_alert(alert)
            emitted_alerts.append(logged)

        return emitted_alerts
