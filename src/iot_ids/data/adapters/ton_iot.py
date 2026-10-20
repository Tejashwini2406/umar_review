"""ToN-IoT Network Dataset Adapter.

Ingests ToN-IoT Zeek flow telemetry (`train_test_network.csv`) into CanonicalFlow records.
"""
from __future__ import annotations

from pathlib import Path
from typing import Generator, List
import pandas as pd
import numpy as np

from iot_ids.data.adapters.base import DatasetAdapter, IngestionAuditReport
from iot_ids.data.flow_object import CanonicalFlow


CATEGORY_MAP = {
    "normal": "Normal",
    "ddos": "DDoS",
    "dos": "DoS",
    "reconnaissance": "Scanning",
    "scanning": "Scanning",
    "backdoor": "Backdoor",
    "injection": "Injection",
    "xss": "XSS",
    "password": "Password",
    "mitm": "MITM",
}


class ToNIoTAdapter(DatasetAdapter):
    """Adapter for ToN-IoT Zeek flow logs (train_test_network.csv)."""

    name = "ToN-IoT"

    def get_source_files(self) -> List[Path]:
        if self.raw_path.is_file():
            return [self.raw_path]
        target = self.raw_path / "train_test_network.csv"
        return [target] if target.exists() else sorted(self.raw_path.glob("*.csv"))

    def stream_canonical_flows(
        self, chunksize: int = 100000
    ) -> Generator[CanonicalFlow, None, IngestionAuditReport]:
        files = self.get_source_files()
        report = IngestionAuditReport(
            dataset_name=self.name,
            source_files=[str(p) for p in files],
        )

        global_ts = 1546300800.0  # Synthetic UNIX epoch baseline (2019-01-01)

        for file_path in files:
            reader = pd.read_csv(file_path, chunksize=chunksize, low_memory=False)
            for chunk in reader:
                report.raw_records_read += len(chunk)
                
                # Coerce numeric columns
                duration_s = pd.to_numeric(chunk.get("duration", 0), errors="coerce").fillna(0.0).clip(lower=0.0).values
                src_bytes_v = pd.to_numeric(chunk.get("src_bytes", 0), errors="coerce").fillna(0).clip(lower=0).astype(int).values
                dst_bytes_v = pd.to_numeric(chunk.get("dst_bytes", 0), errors="coerce").fillna(0).clip(lower=0).astype(int).values
                src_pkts_v = pd.to_numeric(chunk.get("src_pkts", 0), errors="coerce").fillna(0).clip(lower=0).astype(int).values
                dst_pkts_v = pd.to_numeric(chunk.get("dst_pkts", 0), errors="coerce").fillna(0).clip(lower=0).astype(int).values

                src_ip_v = chunk.get("src_ip", "0.0.0.0").astype(str).values
                dst_ip_v = chunk.get("dst_ip", "0.0.0.0").astype(str).values
                src_port_v = pd.to_numeric(chunk.get("src_port", 0), errors="coerce").fillna(0).astype(int).values
                dst_port_v = pd.to_numeric(chunk.get("dst_port", 0), errors="coerce").fillna(0).astype(int).values
                proto_v = chunk.get("proto", "tcp").astype(str).str.lower().values

                conn_state_v = chunk.get("conn_state", "SF").astype(str).values
                label_v = pd.to_numeric(chunk.get("label", 0), errors="coerce").fillna(0).astype(int).values
                type_v = chunk.get("type", "normal").astype(str).str.lower().values

                for i in range(len(chunk)):
                    dur = float(duration_s[i])
                    ts_start = global_ts
                    ts_end = global_ts + dur
                    global_ts += max(0.001, dur * 0.1)  # Increment global time stream

                    p_str = proto_v[i]
                    proto_norm = p_str if p_str in ("tcp", "udp", "icmp") else "other"

                    state_str = conn_state_v[i].upper()
                    conn_code = 0  # Normal SF / S1
                    if any(tok in state_str for tok in ("REJ", "S0", "RSTO", "RSTR", "RSTOS0")):
                        conn_code = 1  # Unanswered / Rejected / Reset

                    lbl = 1 if label_v[i] > 0 else 0
                    cat_raw = type_v[i]
                    cat_norm = CATEGORY_MAP.get(cat_raw, "Normal" if lbl == 0 else "Attack")

                    try:
                        flow = CanonicalFlow(
                            timestamp_start=ts_start,
                            timestamp_end=ts_end,
                            src_host=src_ip_v[i],
                            dst_host=dst_ip_v[i],
                            src_port=int(src_port_v[i]),
                            dst_port=int(dst_port_v[i]),
                            protocol=proto_norm,
                            bytes_src=int(src_bytes_v[i]),
                            bytes_dst=int(dst_bytes_v[i]),
                            pkts_src=int(src_pkts_v[i]),
                            pkts_dst=int(dst_pkts_v[i]),
                            syn_count_src=1 if proto_norm == "tcp" else 0,
                            ack_count_src=1 if conn_code == 0 and proto_norm == "tcp" else 0,
                            rst_count_src=1 if conn_code == 1 and proto_norm == "tcp" else 0,
                            conn_state_code=conn_code,
                            label=lbl,
                            attack_category=cat_norm,
                        )
                        report.record_flow(flow)
                        yield flow
                    except Exception:
                        report.invalid_records_dropped += 1

        return report
