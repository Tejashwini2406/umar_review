"""Production Operational Runtime Daemon for Multi-Level IoT IDS.

Manages continuous packet capture ingestion, real-time multi-level flow evaluation,
alert sink persistence, operational metrics tracking, and graceful signal shutdown.
"""
from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import scapy.all as scapy

from iot_ids.config import PipelineConfig
from iot_ids.inference.predictor import IDSPredictor
from iot_ids.data.packet_capture import scapy_to_canonical_packet
from iot_ids.utils.sinks import BaseAlertSink, JSONLFileSink, ConsoleAlertSink, MultiAlertSink
from iot_ids.utils.metrics import OperationalMetricsSummary


class IDSRuntimeDaemon:
    """Production long-running operational IDS runtime daemon."""

    def __init__(
        self,
        predictor: IDSPredictor,
        config: Optional[PipelineConfig] = None,
        sink: Optional[BaseAlertSink] = None,
        interface: Optional[str] = None,
        pcap_file: Optional[Path | str] = None,
        bpf_filter: str = "ip",
    ):
        self.predictor = predictor
        self.config = config or PipelineConfig()
        self.config.validate()

        self.interface = interface or self.config.interface
        self.pcap_file = Path(pcap_file) if pcap_file else (Path(self.config.pcap_file) if self.config.pcap_file else None)
        self.bpf_filter = bpf_filter or self.config.bpf_filter or "ip"

        if sink:
            self.sink = sink
        else:
            jsonl_sink = JSONLFileSink("reports/runtime_alerts.jsonl")
            console_sink = ConsoleAlertSink(min_level="LOW")
            self.sink = MultiAlertSink([jsonl_sink, console_sink])

        self.is_running: bool = False
        self.shutdown_requested: bool = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Registers SIGINT and SIGTERM handlers for graceful daemon shutdown."""
        def handle_signal(signum, frame):
            print(f"\n[DAEMON] Received signal {signum}. Initiating graceful shutdown...")
            self.stop()

        try:
            signal.signal(signal.SIGINT, handle_signal)
            signal.signal(signal.SIGTERM, handle_signal)
        except (ValueError, AttributeError):
            # Signal handling might be restricted on non-main threads or specific platforms
            pass

    def stop(self) -> None:
        """Requests daemon shutdown and flushes active flows and sinks."""
        self.shutdown_requested = True
        self.is_running = False

    def process_packet(self, pkt: Any) -> Optional[Dict[str, Any]]:
        """Processes a single raw packet through the daemon pipeline with exception isolation."""
        try:
            canon_pkt = scapy_to_canonical_packet(pkt)
            if canon_pkt is None:
                return None

            alert = self.predictor.process_raw_packet(canon_pkt)
            if alert is not None:
                try:
                    self.sink.write_alert(alert)
                except Exception as sink_err:
                    print(f"[DAEMON WARNING] Alert sink write error: {sink_err}", file=sys.stderr)
                return alert
            return None
        except Exception as err:
            print(f"[DAEMON ERROR] Isolated packet processing error: {err}", file=sys.stderr)
            return None

    def run(self, max_packets: Optional[int] = None, duration: Optional[float] = None) -> OperationalMetricsSummary:
        """Executes the long-running operational capture and prediction daemon loop."""
        if max_packets is not None and max_packets < 0:
            raise ValueError("max_packets must be non-negative or None")
        if duration is not None and duration < 0:
            raise ValueError("duration must be non-negative or None")

        self.is_running = True
        self.shutdown_requested = False
        self.predictor.reset_state()

        print("==========================================================================")
        print("=== MULTI-LEVEL IOT IDS OPERATIONAL RUNTIME DAEMON STARTED ===")
        print("==========================================================================")
        health = self.predictor.health_check()
        thresh = self.predictor.pipeline.effective_decision_threshold
        print(f"Health Status: {health['pipeline']['status']} | Active Model: {self.config.model_family}")
        print(f"Feature Profile: {self.config.feature_profile} | Calibrated Threshold: {thresh:.2f}")

        start_time = time.time()
        packets_processed = 0

        # Mode A: PCAP File Replay Daemon
        if max_packets == 0 or duration == 0:
            pass
        elif self.pcap_file and self.pcap_file.exists():
            print(f"[DAEMON] Ingesting offline PCAP stream: {self.pcap_file}")
            try:
                reader = scapy.PcapReader(str(self.pcap_file))
                for pkt in reader:
                    if self.shutdown_requested:
                        break
                    if max_packets is not None and packets_processed >= max_packets:
                        break
                    if duration is not None and (time.time() - start_time) >= duration:
                        break

                    self.process_packet(pkt)
                    packets_processed += 1
                reader.close()
            except Exception as e:
                print(f"[DAEMON ERROR] Error reading PCAP file: {e}", file=sys.stderr)

        # Mode B: Live Sniffing Daemon or Mock Capture
        else:
            print(f"[DAEMON] Listening on live interface: '{self.interface or 'default'}' (filter: '{self.bpf_filter}')")

            def packet_callback(pkt: Any) -> None:
                nonlocal packets_processed
                if self.shutdown_requested:
                    return
                self.process_packet(pkt)
                packets_processed += 1

            def stop_condition(pkt: Any) -> bool:
                if self.shutdown_requested:
                    return True
                if max_packets is not None and packets_processed >= max_packets:
                    return True
                if duration is not None and (time.time() - start_time) >= duration:
                    return True
                return False

            try:
                scapy.sniff(
                    iface=self.interface,
                    filter=self.bpf_filter,
                    prn=packet_callback,
                    stop_filter=stop_condition,
                    timeout=duration,
                    store=0,
                )
            except Exception as e:
                print(f"[DAEMON ERROR] Live capture error: {e}", file=sys.stderr)

        # Flush state on shutdown
        self._flush_and_cleanup()
        return self.predictor.metrics.get_summary()

    def _flush_and_cleanup(self) -> None:
        """Flushes remaining active flows and outputs final operational summary."""
        print("\n[DAEMON] Flushing remaining active flow buffers...")
        try:
            flushed_flows = self.predictor.pipeline.flow_aggregator.flush()
            for flow in flushed_flows:
                alert = self.predictor.predict_flow(flow)
                try:
                    self.sink.write_alert(alert)
                except Exception as sink_err:
                    print(f"[DAEMON WARNING] Flush sink write error: {sink_err}", file=sys.stderr)
        except Exception as flush_err:
            print(f"[DAEMON ERROR] Flush error: {flush_err}", file=sys.stderr)

        summary = self.predictor.metrics.get_summary()
        print("\n==========================================================================")
        print("=== DAEMON SHUTDOWN COMPLETE: FINAL METRICS SUMMARY ===")
        print("==========================================================================")
        print(f"Packets Processed: {summary.packets_processed}")
        print(f"Flows Completed:   {summary.flows_completed}")
        print(f"Alerts Emitted:    {summary.alerts_emitted}")
        print(f"Errors Encountered:{summary.errors_encountered}")
        print(f"Mean Latency:      {summary.latency_mean_ms:.3f} ms/flow")
        print(f"P95 Latency:       {summary.latency_p95_ms:.3f} ms/flow")
        print(f"Uptime Duration:   {summary.uptime_seconds:.2f} seconds")
        self.is_running = False
