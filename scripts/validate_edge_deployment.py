"""Reproducible End-to-End Edge Deployment Validation Script.

Executes a complete end-to-end edge deployment validation cycle:
  - Fail-fast PipelineConfig validation
  - Predictor artifact health checking
  - Packet / PCAP ingestion -> FlowAggregator -> 18 features -> preprocessing -> alignment -> RF -> calibration -> IDSAlertOutput -> JSONL alert sink
  - Operational metrics inspection (latency mean, min, max, p95, packets, flows, alerts, errors, active memory)
"""
from __future__ import annotations

import json
from pathlib import Path
import scapy.all as scapy
from scapy.layers.inet import IP, TCP

from iot_ids.config import PipelineConfig
from iot_ids.inference.predictor import IDSPredictor
from iot_ids.data.packet_capture import PacketCaptureEngine
from iot_ids.utils.sinks import MultiAlertSink, JSONLFileSink, ConsoleAlertSink


def validate_edge_deployment():
    print("==========================================================================")
    print("=== STARTING PHASE D EDGE DEPLOYMENT & PRODUCTION HARDENING VALIDATION ===")
    print("==========================================================================")

    # 1. Config Fail-Fast Validation
    config = PipelineConfig(
        artifact_dir="models/final/deployable_artifact",
        decision_threshold=0.50,
        inactivity_timeout=15.0,
        max_active_flows=5000,
    )
    config.validate()
    print("[PASS] PipelineConfig fail-fast validation PASSED.")

    # 2. Predictor Health Inspection
    artifact_path = Path(config.artifact_dir)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Deployable artifact path not found: {artifact_path}")

    predictor = IDSPredictor.from_artifact(artifact_path)
    health = predictor.health_check()
    print(f"[PASS] Health Check: {health['pipeline']['status']} | Model Loaded: {health['pipeline']['model_loaded']}")
    assert health['pipeline']['status'] == "HEALTHY"

    # 3. Create Alert Sink Pipeline (JSONL File + Console)
    sink_path = Path("reports/stage12/edge_deployment_alerts.jsonl")
    if sink_path.exists():
        sink_path.unlink()

    jsonl_sink = JSONLFileSink(sink_path)
    console_sink = ConsoleAlertSink(min_level="LOW")
    multi_sink = MultiAlertSink([jsonl_sink, console_sink])

    # 4. Generate Synthetic Stream & Replay
    pcap_path = Path("reports/stage12/edge_validation_stream.pcap")
    pcap_path.parent.mkdir(parents=True, exist_ok=True)

    pkts = [
        scapy.Ether() / scapy.IP(src="192.168.1.100", dst="10.0.0.1") / scapy.TCP(sport=5000, dport=80, flags="S"),
        scapy.Ether() / scapy.IP(src="10.0.0.1", dst="192.168.1.100") / scapy.TCP(sport=80, dport=5000, flags="SA"),
        scapy.Ether() / scapy.IP(src="192.168.1.100", dst="10.0.0.1") / scapy.TCP(sport=5000, dport=80, flags="A"),
        scapy.Ether() / scapy.IP(src="192.168.1.100", dst="10.0.0.1") / scapy.TCP(sport=5000, dport=80, flags="PA"),
    ]
    pkts[0].time = 100.0
    pkts[1].time = 100.1
    pkts[2].time = 130.0  # 29.9s gap triggers flow eviction
    pkts[3].time = 130.5

    scapy.wrpcap(str(pcap_path), pkts)

    # 5. Execute PCAP Replay Pipeline
    engine = PacketCaptureEngine(predictor=predictor, logger=None)
    predictor.reset_state()

    reader = scapy.PcapReader(str(pcap_path))
    alerts_emitted = []
    for pkt in reader:
        canon_pkt = engine.scapy_to_canonical_packet(pkt) if hasattr(engine, "scapy_to_canonical_packet") else None
        if canon_pkt is None:
            from iot_ids.data.packet_capture import scapy_to_canonical_packet
            canon_pkt = scapy_to_canonical_packet(pkt)
        if canon_pkt:
            alert = predictor.process_raw_packet(canon_pkt)
            if alert:
                multi_sink.write_alert(alert)
                alerts_emitted.append(alert)
    reader.close()

    # Flush remaining active flows at EOF
    flushed = predictor.pipeline.flow_aggregator.flush()
    for flow in flushed:
        alert = predictor.predict_flow(flow)
        multi_sink.write_alert(alert)
        alerts_emitted.append(alert)

    # 6. Verify Sink Persistence
    assert sink_path.exists()
    saved_lines = sink_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(saved_lines) == len(alerts_emitted)
    print(f"[PASS] Alert Sink Persistence: Saved {len(saved_lines)} structured alert records to {sink_path}")

    # 7. Metrics Inspection
    metrics_summary = predictor.metrics.get_summary()
    print("\n--- Operational Metrics Summary ---")
    print(f"  Packets Processed: {metrics_summary.packets_processed}")
    print(f"  Flows Completed:   {metrics_summary.flows_completed}")
    print(f"  Alerts Emitted:    {metrics_summary.alerts_emitted}")
    print(f"  Errors Encountered:{metrics_summary.errors_encountered}")
    print(f"  Mean Latency:      {metrics_summary.latency_mean_ms:.3f} ms/flow")
    print(f"  P95 Latency:       {metrics_summary.latency_p95_ms:.3f} ms/flow")
    print(f"  Active Flows Mem:  {metrics_summary.active_flows_in_memory}")

    assert metrics_summary.packets_processed == 4
    assert metrics_summary.flows_completed >= 2
    assert metrics_summary.errors_encountered == 0

    print("\n==========================================================================")
    print("=== PHASE D EDGE DEPLOYMENT VALIDATION: 100% SUCCESSFUL & READY ===")
    print("==========================================================================")


if __name__ == "__main__":
    validate_edge_deployment()
