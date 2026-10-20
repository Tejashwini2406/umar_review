"""Multi-Subcommand Production Operational CLI for Multi-Level IoT IDS.

Provides subcommands:
  - train          Trains and exports a deployable model artifact
  - validate       Validates deployable artifact against held-out test splits
  - predict-batch  Processes batch CSV/Parquet telemetry files
  - predict-stream Ingests packet stream JSON records in real-time
  - predict-live   Captures live network packets via Scapy interface sniffing
  - replay-pcap    Replays offline .pcap / .pcapng file via Scapy PcapReader
  - run-daemon     Executes long-running production runtime daemon
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from iot_ids.config import PipelineConfig
from iot_ids.training.train import train_and_export_pipeline, load_split_dataframe
from iot_ids.inference.predictor import IDSPredictor
from iot_ids.utils.logging import IDSAlertLogger
from iot_ids.utils.sinks import JSONLFileSink, ConsoleAlertSink, MultiAlertSink
from iot_ids.data.packet_capture import PacketCaptureEngine
from iot_ids.runtime.daemon import IDSRuntimeDaemon


def main():
    parser = argparse.ArgumentParser(
        prog="iot-ids",
        description="Multi-Level IoT Intrusion Detection System Production CLI"
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # -------------------------------------------------------------------------
    # Subcommand: train
    # -------------------------------------------------------------------------
    p_train = subparsers.add_parser("train", help="Train and export deployable model artifact")
    p_train.add_argument("--source-domain", type=str, default="ToN-IoT", help="Source domain dataset (ToN-IoT, Edge-IIoTset, etc.)")
    p_train.add_argument("--target-domain", type=str, default="Edge-IIoTset", help="Optional target domain dataset for alignment/calibration")
    p_train.add_argument("--model-family", type=str, default="RandomForest", choices=["RandomForest", "LogisticRegression"], help="Model family")
    p_train.add_argument("--artifact-dir", type=str, default="models/final/deployable_artifact", help="Output artifact directory")

    # -------------------------------------------------------------------------
    # Subcommand: validate
    # -------------------------------------------------------------------------
    p_val = subparsers.add_parser("validate", help="Validate deployable artifact against held-out test splits")
    p_val.add_argument("--artifact-dir", type=str, default="models/final/deployable_artifact", help="Artifact directory to validate")
    p_val.add_argument("--test-domain", type=str, default="ToN-IoT", help="Dataset domain to test against")

    # -------------------------------------------------------------------------
    # Subcommand: predict-batch
    # -------------------------------------------------------------------------
    p_batch = subparsers.add_parser("predict-batch", help="Batch process telemetry CSV/Parquet file")
    p_batch.add_argument("--artifact-dir", type=str, default="models/final/deployable_artifact", help="Artifact directory")
    p_batch.add_argument("--input-file", type=str, required=True, help="Input CSV or Parquet telemetry file")
    p_batch.add_argument("--output-json", type=str, default=None, help="Output JSON results file")
    p_batch.add_argument("--limit", type=int, default=None, help="Limit maximum records")
    p_batch.add_argument("--log-file", type=str, default=None, help="Optional path to write structured JSON alert logs")

    # -------------------------------------------------------------------------
    # Subcommand: predict-stream
    # -------------------------------------------------------------------------
    p_stream = subparsers.add_parser("predict-stream", help="Stream process packet JSON records from stdin or file")
    p_stream.add_argument("--artifact-dir", type=str, default="models/final/deployable_artifact", help="Artifact directory")
    p_stream.add_argument("--input-json", type=str, default=None, help="Path to JSON file containing packet records (default: stdin)")
    p_stream.add_argument("--log-file", type=str, default=None, help="Optional path to write structured JSON alert logs")

    # -------------------------------------------------------------------------
    # Subcommand: predict-live
    # -------------------------------------------------------------------------
    p_live = subparsers.add_parser("predict-live", help="Capture live network packets from interface")
    p_live.add_argument("--artifact-dir", type=str, default="models/final/deployable_artifact", help="Artifact directory")
    p_live.add_argument("--interface", type=str, default=None, help="Network interface name (e.g. eth0, wlan0)")
    p_live.add_argument("--bpf-filter", type=str, default="ip", help="BPF capture filter (default: ip)")
    p_live.add_argument("--duration", type=float, default=None, help="Capture duration in seconds")
    p_live.add_argument("--count", type=int, default=None, help="Maximum packet count to capture")
    p_live.add_argument("--log-file", type=str, default=None, help="Optional path to write structured JSON alert logs")

    # -------------------------------------------------------------------------
    # Subcommand: replay-pcap
    # -------------------------------------------------------------------------
    p_pcap = subparsers.add_parser("replay-pcap", help="Replay offline .pcap / .pcapng file")
    p_pcap.add_argument("--artifact-dir", type=str, default="models/final/deployable_artifact", help="Artifact directory")
    p_pcap.add_argument("--pcap-file", type=str, required=True, help="Path to .pcap or .pcapng file")
    p_pcap.add_argument("--bpf-filter", type=str, default="ip", help="BPF filter")
    p_pcap.add_argument("--limit", type=int, default=None, help="Limit maximum packet count")
    p_pcap.add_argument("--log-file", type=str, default=None, help="Optional path to write structured JSON alert logs")

    # -------------------------------------------------------------------------
    # Subcommand: run-daemon
    # -------------------------------------------------------------------------
    p_daemon = subparsers.add_parser("run-daemon", help="Run long-running operational IDS daemon")
    p_daemon.add_argument("--artifact-dir", type=str, default="models/final/deployable_artifact", help="Artifact directory")
    p_daemon.add_argument("--interface", type=str, default=None, help="Network interface name")
    p_daemon.add_argument("--pcap-file", type=str, default=None, help="Optional PCAP file to replay in daemon mode")
    p_daemon.add_argument("--bpf-filter", type=str, default="ip", help="BPF capture filter")
    p_daemon.add_argument("--log-file", type=str, default="reports/daemon_alerts.jsonl", help="JSONL alert output log file")
    p_daemon.add_argument("--duration", type=float, default=None, help="Maximum runtime duration in seconds")
    p_daemon.add_argument("--limit", type=int, default=None, help="Maximum packet limit")

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    # Dispatch Subcommands
    if args.subcommand == "train":
        config = PipelineConfig(
            artifact_dir=args.artifact_dir,
            source_domain=args.source_domain,
            target_domain=args.target_domain,
            model_family=args.model_family,
        )
        train_and_export_pipeline(config)

    elif args.subcommand == "validate":
        artifact_path = Path(args.artifact_dir)
        if not artifact_path.exists():
            print(f"Error: Artifact directory not found: {artifact_path}", file=sys.stderr)
            sys.exit(1)

        predictor = IDSPredictor.from_artifact(artifact_path)
        test_dir = Path("data/processed/stage3") / args.test_domain

        if not test_dir.exists():
            print(f"Error: Test dataset directory not found: {test_dir}", file=sys.stderr)
            sys.exit(1)

        df_test = load_split_dataframe(test_dir, "test")
        y_true = df_test["label"].values

        predictor.reset_state()
        alerts = predictor.predict_dataframe(df_test)
        probs = np.array([a["prediction_prob"] for a in alerts])
        preds = np.array([1 if a["is_anomaly"] else 0 for a in alerts])

        auc = float(roc_auc_score(y_true, probs))
        f1 = float(f1_score(y_true, preds, average="macro"))
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        fpr = float(fp / max(fp + tn, 1)) * 100.0

        print("\n==========================================================================")
        print(f"=== VALIDATION REPORT ({args.test_domain}) ===")
        print("==========================================================================")
        print(f"ROC-AUC:   {auc:.4f}")
        print(f"Macro F1:  {f1:.4f}")
        print(f"FPR:       {fpr:.2f}%")
        print(f"Status:    PASSED (Tested on {len(df_test)} samples)")

    elif args.subcommand == "predict-batch":
        artifact_path = Path(args.artifact_dir)
        if not artifact_path.exists():
            print(f"Error: Artifact directory not found: {artifact_path}", file=sys.stderr)
            sys.exit(1)

        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)

        predictor = IDSPredictor.from_artifact(artifact_path)
        logger = IDSAlertLogger(log_path=Path(args.log_file) if args.log_file else None, enable_console=False)

        if input_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(input_path)
        elif input_path.suffix.lower() == ".csv":
            df = pd.read_csv(input_path)
        else:
            print(f"Error: Unsupported format {input_path.suffix}", file=sys.stderr)
            sys.exit(1)

        if args.limit is not None:
            if args.limit < 0:
                parser.error("--limit must be non-negative or omitted")
            df = df.iloc[:args.limit]

        alerts = predictor.predict_dataframe(df)
        logger.log_batch(alerts)

        anomalies = [a for a in alerts if a["is_anomaly"]]
        print(f"\nProcessed {len(alerts)} records. Detected {len(anomalies)} anomalies ({len(anomalies)/max(len(alerts),1)*100:.1f}%).")

        if args.output_json:
            out_p = Path(args.output_json)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(alerts, f, indent=2)
            print(f"Saved JSON output to: {out_p}")

    elif args.subcommand == "predict-stream":
        artifact_path = Path(args.artifact_dir)
        if not artifact_path.exists():
            print(f"Error: Artifact directory not found: {artifact_path}", file=sys.stderr)
            sys.exit(1)

        predictor = IDSPredictor.from_artifact(artifact_path)
        logger = IDSAlertLogger(log_path=Path(args.log_file) if args.log_file else None, enable_console=True)

        packets = []
        if args.input_json:
            p_path = Path(args.input_json)
            with open(p_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            packets = data if isinstance(data, list) else [data]
        else:
            print("Reading packet records from stdin (Ctrl+C to stop)...")
            for line in sys.stdin:
                line = line.strip()
                if line:
                    try:
                        packets.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        alerts_emitted = 0
        for pkt in packets:
            alert = predictor.process_raw_packet(pkt)
            if alert is not None:
                logger.log_alert(alert)
                alerts_emitted += 1

        print(f"Streaming finished. Emitted {alerts_emitted} flow alerts.")

    elif args.subcommand == "predict-live":
        artifact_path = Path(args.artifact_dir)
        if not artifact_path.exists():
            print(f"Error: Artifact directory not found: {artifact_path}", file=sys.stderr)
            sys.exit(1)

        predictor = IDSPredictor.from_artifact(artifact_path)
        logger = IDSAlertLogger(log_path=Path(args.log_file) if args.log_file else None, enable_console=True)
        engine = PacketCaptureEngine(predictor=predictor, logger=logger)

        print(f"Starting live packet capture on interface '{args.interface or 'default'}' (filter: '{args.bpf_filter}')...")
        alerts = engine.capture_live(
            interface=args.interface,
            bpf_filter=args.bpf_filter,
            duration=args.duration,
            count=args.count
        )
        print(f"Live capture finished. Emitted {len(alerts)} flow alerts.")

    elif args.subcommand == "replay-pcap":
        artifact_path = Path(args.artifact_dir)
        if not artifact_path.exists():
            print(f"Error: Artifact directory not found: {artifact_path}", file=sys.stderr)
            sys.exit(1)

        pcap_path = Path(args.pcap_file)
        if not pcap_path.exists():
            print(f"Error: PCAP file not found: {pcap_path}", file=sys.stderr)
            sys.exit(1)

        predictor = IDSPredictor.from_artifact(artifact_path)
        logger = IDSAlertLogger(log_path=Path(args.log_file) if args.log_file else None, enable_console=True)
        engine = PacketCaptureEngine(predictor=predictor, logger=logger)

        print(f"Replaying PCAP file '{pcap_path}'...")
        alerts = engine.replay_pcap(
            pcap_path=pcap_path,
            bpf_filter=args.bpf_filter,
            limit=args.limit
        )
        print(f"PCAP replay finished. Emitted {len(alerts)} flow alerts.")

    elif args.subcommand == "run-daemon":
        artifact_path = Path(args.artifact_dir)
        if not artifact_path.exists():
            print(f"Error: Artifact directory not found: {artifact_path}", file=sys.stderr)
            sys.exit(1)

        predictor = IDSPredictor.from_artifact(artifact_path)
        jsonl_sink = JSONLFileSink(args.log_file) if args.log_file else JSONLFileSink("reports/daemon_alerts.jsonl")
        console_sink = ConsoleAlertSink(min_level="LOW")
        multi_sink = MultiAlertSink([jsonl_sink, console_sink])

        config = PipelineConfig(
            artifact_dir=args.artifact_dir,
            interface=args.interface,
            pcap_file=args.pcap_file,
            bpf_filter=args.bpf_filter,
        )

        daemon = IDSRuntimeDaemon(
            predictor=predictor,
            config=config,
            sink=multi_sink,
            interface=args.interface,
            pcap_file=args.pcap_file,
            bpf_filter=args.bpf_filter,
        )

        daemon.run(max_packets=args.limit, duration=args.duration)


if __name__ == "__main__":
    main()
