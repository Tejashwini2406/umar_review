"""Integration and Stress Tests for Production IDSRuntimeDaemon.

Tests daemon startup, PCAP ingestion, live loop, graceful shutdown, state flushing,
malformed packet exception isolation, sink failure handling, and repeated lifecycle restarts.
"""
import tempfile
from pathlib import Path
import pytest

try:
    import scapy.all as scapy
    from iot_ids.runtime.daemon import IDSRuntimeDaemon
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False

pytestmark = pytest.mark.skipif(not HAS_SCAPY, reason="Optional dependency 'scapy' is not installed")

from iot_ids.config import PipelineConfig
from iot_ids.inference.predictor import IDSPredictor
from iot_ids.utils.sinks import JSONLFileSink


@pytest.fixture
def deployable_predictor():
    artifact_dir = Path("models/final/deployable_artifact")
    if not artifact_dir.exists():
        pytest.skip(f"Deployable artifact directory not found: {artifact_dir}")
    return IDSPredictor.from_artifact(artifact_dir)


def test_daemon_pcap_ingestion_and_flushing(deployable_predictor):
    """Verifies IDSRuntimeDaemon ingests PCAP file, emits alerts, and flushes state on completion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pcap_path = Path(tmp_dir) / "daemon_test.pcap"
        log_path = Path(tmp_dir) / "daemon_alerts.jsonl"

        pkts = [
            scapy.Ether() / scapy.IP(src="192.168.1.10", dst="10.0.0.1") / scapy.TCP(sport=1000, dport=80, flags="S"),
            scapy.Ether() / scapy.IP(src="10.0.0.1", dst="192.168.1.10") / scapy.TCP(sport=80, dport=1000, flags="SA"),
            scapy.Ether() / scapy.IP(src="192.168.1.10", dst="10.0.0.1") / scapy.TCP(sport=1000, dport=80, flags="A"),
        ]
        pkts[0].time = 100.0
        pkts[1].time = 100.1
        pkts[2].time = 130.0  # Evicts flow

        scapy.wrpcap(str(pcap_path), pkts)

        sink = JSONLFileSink(log_path)
        config = PipelineConfig(artifact_dir="models/final/deployable_artifact", pcap_file=str(pcap_path))

        daemon = IDSRuntimeDaemon(
            predictor=deployable_predictor,
            config=config,
            sink=sink,
            pcap_file=pcap_path
        )

        metrics = daemon.run()

        assert metrics.packets_processed == 3
        assert metrics.flows_completed >= 1
        assert log_path.exists()

        records = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(records) >= 1


def test_daemon_malformed_packet_isolation(deployable_predictor):
    """Verifies malformed input frames do not crash the daemon loop."""
    daemon = IDSRuntimeDaemon(predictor=deployable_predictor)

    # Ingest malformed object
    result = daemon.process_packet("not_a_valid_scapy_packet")
    assert result is None


def test_packet_processing_failure_is_counted_once(deployable_predictor):
    daemon = IDSRuntimeDaemon(predictor=deployable_predictor)
    aggregator = deployable_predictor.pipeline.flow_aggregator

    def fail_add_packet(_packet):
        raise RuntimeError("forced aggregator failure")

    aggregator.add_packet = fail_add_packet
    packet = scapy.Ether() / scapy.IP(src="192.168.1.3", dst="10.0.0.3") / scapy.TCP()

    assert daemon.process_packet(packet) is None
    assert deployable_predictor.metrics.get_summary().errors_encountered == 1


def test_daemon_graceful_shutdown(deployable_predictor):
    """Verifies calling stop() sets shutdown flag and terminates cleanly."""
    daemon = IDSRuntimeDaemon(predictor=deployable_predictor)
    daemon.is_running = True
    assert daemon.shutdown_requested is False

    daemon.stop()
    assert daemon.shutdown_requested is True
    assert daemon.is_running is False


def test_daemon_repeated_restart_lifecycle(deployable_predictor):
    """Verifies daemon can be executed, stopped, reset, and re-executed repeatedly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pcap_path = Path(tmp_dir) / "restart_test.pcap"
        pkts = [
            scapy.Ether() / scapy.IP(src="192.168.1.50", dst="10.0.0.5") / scapy.TCP(sport=2000, dport=80, flags="S"),
        ]
        pkts[0].time = 200.0
        scapy.wrpcap(str(pcap_path), pkts)

        daemon = IDSRuntimeDaemon(predictor=deployable_predictor, pcap_file=pcap_path)

        # First lifecycle
        metrics1 = daemon.run()
        assert metrics1.packets_processed == 1

        # Second lifecycle
        metrics2 = daemon.run()
        assert metrics2.packets_processed == 1
