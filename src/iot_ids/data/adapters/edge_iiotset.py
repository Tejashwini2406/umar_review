"""Edge-IIoTset Dataset Aggregator Adapter.

Ingests Edge-IIoTset frame telemetry (`ML-EdgeIIoT-dataset.csv`) and aggregates frames
into bi-directional CanonicalFlow records using FlowAggregator.
"""
from __future__ import annotations

from pathlib import Path
from typing import Generator, List
import pandas as pd
import numpy as np

from iot_ids.data.adapters.base import DatasetAdapter, IngestionAuditReport
from iot_ids.data.flow_aggregator import FlowAggregator
from iot_ids.data.flow_object import CanonicalFlow


EDGE_CATEGORY_MAP = {
    "normal": "Normal",
    "ddos_udp": "DDoS",
    "ddos_http": "DDoS",
    "ddos_icmp": "DDoS",
    "ddos_tcp": "DDoS",
    "sql_injection": "Injection",
    "vulnerability_scanner": "Scanning",
    "password": "Password",
    "backdoor": "Backdoor",
    "ransomware": "Ransomware",
    "mitm": "MITM",
    "uploading": "Injection",
    "xss": "XSS",
    "port_scanning": "Scanning",
}


def parse_edge_timestamp(val: str, default_ts: float) -> float:
    """Parses Wireshark frame.time string into a float UNIX timestamp."""
    try:
        dt = pd.to_datetime(val, errors="coerce")
        if pd.isna(dt):
            return default_ts
        return dt.timestamp()
    except Exception:
        return default_ts


class EdgeIIoTsetAggregatorAdapter(DatasetAdapter):
    """Adapter for Edge-IIoTset packet telemetry (ML-EdgeIIoT-dataset.csv)."""

    name = "Edge-IIoTset"

    def get_source_files(self) -> List[Path]:
        if self.raw_path.is_file():
            return [self.raw_path]
        target = (
            self.raw_path
            / "Edge-IIoTset dataset"
            / "Selected dataset for ML and DL"
            / "ML-EdgeIIoT-dataset.csv"
        )
        return [target] if target.exists() else sorted(self.raw_path.glob("*.csv"))

    def stream_canonical_flows(
        self, chunksize: int = 50000
    ) -> Generator[CanonicalFlow, None, IngestionAuditReport]:
        files = self.get_source_files()
        report = IngestionAuditReport(
            dataset_name=self.name,
            source_files=[str(p) for p in files],
        )

        aggregator = FlowAggregator(inactivity_timeout=15.0, max_flow_duration=120.0)
        curr_ts = 1640995200.0  # Baseline epoch timestamp (2022-01-01)

        for file_path in files:
            reader = pd.read_csv(file_path, chunksize=chunksize, low_memory=False)
            for chunk in reader:
                report.raw_records_read += len(chunk)

                def get_col(col_name, default_val=0):
                    if col_name in chunk.columns:
                        return chunk[col_name]
                    return pd.Series([default_val] * len(chunk))

                src_ip_v = get_col("ip.src_host", "0.0.0.0").astype(str).values
                dst_ip_v = get_col("ip.dst_host", "0.0.0.0").astype(str).values
                src_port_v = pd.to_numeric(get_col("tcp.srcport", 0), errors="coerce").fillna(0).astype(int).values
                dst_port_v = pd.to_numeric(get_col("tcp.dstport", 0), errors="coerce").fillna(0).astype(int).values
                udp_port_v = pd.to_numeric(get_col("udp.port", 0), errors="coerce").fillna(0).astype(int).values
                icmp_v = pd.to_numeric(get_col("icmp.checksum", 0), errors="coerce").fillna(0).astype(int).values

                frame_len_v = pd.to_numeric(get_col("tcp.len", 0), errors="coerce").fillna(0).astype(int).values
                syn_v = pd.to_numeric(get_col("tcp.flags.syn", 0), errors="coerce").fillna(0).astype(int).values
                ack_v = pd.to_numeric(get_col("tcp.flags.ack", 0), errors="coerce").fillna(0).astype(int).values
                rst_v = pd.to_numeric(get_col("tcp.flags.reset", 0), errors="coerce").fillna(0).astype(int).values

                label_v = pd.to_numeric(get_col("Attack_label", 0), errors="coerce").fillna(0).astype(int).values
                type_v = get_col("Attack_type", "Normal").astype(str).values
                time_v = get_col("frame.time", "").astype(str).values

                for i in range(len(chunk)):
                    ts = parse_edge_timestamp(time_v[i], curr_ts)
                    curr_ts = max(curr_ts, ts + 0.0001)

                    sp = int(src_port_v[i])
                    dp = int(dst_port_v[i])
                    up = int(udp_port_v[i])
                    ic = int(icmp_v[i])

                    if sp > 0 or dp > 0:
                        proto = "tcp"
                    elif up > 0:
                        proto = "udp"
                    elif ic > 0:
                        proto = "icmp"
                    else:
                        proto = "other"

                    lbl = 1 if label_v[i] > 0 else 0
                    cat_raw = type_v[i].lower().replace(" ", "_")
                    cat_norm = EDGE_CATEGORY_MAP.get(cat_raw, "Normal" if lbl == 0 else "Attack")

                    flow = aggregator.process_frame(
                        timestamp=curr_ts,
                        src_ip=src_ip_v[i],
                        dst_ip=dst_ip_v[i],
                        src_port=sp if sp > 0 else up,
                        dst_port=dp if dp > 0 else up,
                        proto=proto,
                        length=int(frame_len_v[i]),
                        syn=int(syn_v[i]),
                        ack=int(ack_v[i]),
                        rst=int(rst_v[i]),
                        label=lbl,
                        attack_category=cat_norm,
                    )
                    if flow is not None:
                        report.record_flow(flow)
                        report.reconstructed_flows += 1
                        yield flow

        # Flush remaining flows
        for flow in aggregator.flush():
            report.record_flow(flow)
            report.reconstructed_flows += 1
            yield flow

        return report
