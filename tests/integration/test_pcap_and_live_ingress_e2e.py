"""Integration Tests for Production Scapy Packet Capture and PCAP Replay.

Tests offline .pcap replay, synthetic packet generation, non-IP/ARP filtering,
malformed packet handling, empty captures, and mocked live network interface sniffing.
"""
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

try:
    import scapy.all as scapy
    from scapy.layers.inet import IP, TCP
    from scapy.layers.l2 import Ether, ARP
    from iot_ids.data.packet_capture import scapy_to_canonical_packet, PacketCaptureEngine
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

pytestmark = pytest.mark.skipif(not HAS_SCAPY, reason="Optional dependency 'scapy' is not installed")

from iot_ids.inference.predictor import IDSPredictor
from iot_ids.utils.logging import IDSAlertLogger


@pytest.fixture
def deployable_predictor():
    artifact_dir = Path("models/final/deployable_artifact")
    if not artifact_dir.exists():
        pytest.skip(f"Deployable artifact directory not found: {artifact_dir}")
    return IDSPredictor.from_artifact(artifact_dir)


def test_scapy_to_canonical_packet_parsing():
    """Verifies Scapy IP/TCP packet conversion to canonical dictionary format."""
    pkt = Ether() / IP(src="192.168.1.100", dst="10.0.0.1") / TCP(sport=12345, dport=80, flags="S")
    canon = scapy_to_canonical_packet(pkt)

    assert canon is not None
    assert canon["src_ip"] == "192.168.1.100"
    assert canon["dst_ip"] == "10.0.0.1"
    assert canon["src_port"] == 12345
    assert canon["dst_port"] == 80
    assert canon["protocol"] == "tcp"
    assert canon["syn"] == 1


def test_scapy_non_ip_packet_rejection():
    """Verifies non-IP packets (such as ARP) return None without errors."""
    arp_pkt = Ether() / ARP(pdst="192.168.1.1")
    assert scapy_to_canonical_packet(arp_pkt) is None


def test_pcap_replay_inference_cycle(deployable_predictor):
    """Verifies real PCAP file creation -> replay -> FlowAggregator -> IDSPredictor -> Alert."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pcap_path = Path(tmp_dir) / "test_stream.pcap"

        # Generate synthetic packets: Flow 1 at t=10.0, Flow 2 at t=30.0 (causes expiration)
        pkts = [
            Ether() / IP(src="192.168.1.10", dst="10.0.0.1") / TCP(sport=1234, dport=80, flags="S"),
            Ether() / IP(src="192.168.1.10", dst="10.0.0.1") / TCP(sport=1234, dport=80, flags="A"),
        ]

        # Force packet timestamps
        pkts[0].time = 10.0
        pkts[1].time = 30.0  # 20s gap evicts Flow 1

        scapy.wrpcap(str(pcap_path), pkts)
        assert pcap_path.exists()

        engine = PacketCaptureEngine(predictor=deployable_predictor, logger=IDSAlertLogger(enable_console=False))
        alerts = engine.replay_pcap(pcap_path)

        assert len(alerts) >= 1
        assert "prediction_prob" in alerts[0]
        assert "risk_score" in alerts[0]
        assert "is_anomaly" in alerts[0]


def test_empty_pcap_file_handling(deployable_predictor):
    """Verifies empty PCAP file returns empty list cleanly without crashing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        empty_pcap = Path(tmp_dir) / "empty.pcap"
        scapy.wrpcap(str(empty_pcap), [])

        engine = PacketCaptureEngine(predictor=deployable_predictor, logger=IDSAlertLogger(enable_console=False))
        alerts = engine.replay_pcap(empty_pcap)
        assert len(alerts) == 0


def test_mocked_live_capture(deployable_predictor):
    """Verifies live sniffing callback handling with mocked Scapy sniff."""
    engine = PacketCaptureEngine(predictor=deployable_predictor, logger=IDSAlertLogger(enable_console=False))

    pkt = Ether() / IP(src="192.168.1.20", dst="10.0.0.2") / TCP(sport=5555, dport=443, flags="S")
    pkt.time = 100.0

    def fake_sniff(iface=None, filter=None, prn=None, timeout=None, count=None, stop_filter=None, store=0):
        if prn:
            prn(pkt)

    with patch("scapy.all.sniff", side_effect=fake_sniff):
        alerts = engine.capture_live(interface="mock0", duration=1.0)
        assert isinstance(alerts, list)


def test_zero_packet_limits_process_no_packets():
    predictor = Mock()
    engine = PacketCaptureEngine(
        predictor=predictor,
        logger=IDSAlertLogger(enable_console=False),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        pcap_path = Path(tmp_dir) / "not_read.pcap"
        pcap_path.write_bytes(b"")
        assert engine.replay_pcap(pcap_path, limit=0) == []

    with patch("scapy.all.sniff") as sniff:
        assert engine.capture_live(count=0) == []

    sniff.assert_not_called()
    assert predictor.reset_state.call_count == 2


def test_packet_capture_rejects_negative_limits():
    engine = PacketCaptureEngine(
        predictor=Mock(),
        logger=IDSAlertLogger(enable_console=False),
    )

    with pytest.raises(ValueError, match="limit"):
        engine.replay_pcap("missing.pcap", limit=-1)
    with pytest.raises(ValueError, match="count"):
        engine.capture_live(count=-1)
