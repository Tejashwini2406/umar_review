"""NF-ToN-IoT-v2 Dataset Adapter.

High-performance memory-efficient streaming adapter for NF-ToN-IoT-v2 NetFlow records
(NF-ToN-IoT-V2.parquet, ~13.1 million flows).
"""
from __future__ import annotations

from pathlib import Path
from typing import Generator, List
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

from iot_ids.data.adapters.base import DatasetAdapter, IngestionAuditReport
from iot_ids.data.flow_object import CanonicalFlow


NF_CATEGORY_MAP = {
    "benign": "Normal",
    "ddos": "DDoS",
    "dos": "DoS",
    "reconnaissance": "Scanning",
    "scanning": "Scanning",
    "backdoor": "Backdoor",
    "injection": "Injection",
    "xss": "XSS",
    "password": "Password",
    "mitm": "MITM",
    "ransomware": "Ransomware",
}


class NFToNIoTAdapter(DatasetAdapter):
    """Adapter for NF-ToN-IoT-v2 NetFlow Parquet records."""

    name = "NF-ToN-IoT-v2"

    def get_source_files(self) -> List[Path]:
        if self.raw_path.is_file():
            return [self.raw_path]
        target = self.raw_path / "NF-ToN-IoT-V2.parquet"
        return [target] if target.exists() else sorted(self.raw_path.glob("*.parquet"))

    def stream_canonical_flows(
        self, chunksize: int = 250000
    ) -> Generator[CanonicalFlow, None, IngestionAuditReport]:
        files = self.get_source_files()
        report = IngestionAuditReport(
            dataset_name=self.name,
            source_files=[str(p) for p in files],
        )

        global_ts = 1577836800.0  # Synthetic UNIX epoch baseline (2020-01-01)

        for file_path in files:
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=chunksize):
                df = batch.to_pandas()
                report.raw_records_read += len(df)

                src_port_v = df.get("L4_SRC_PORT", 0).astype(int).values
                dst_port_v = df.get("L4_DST_PORT", 0).astype(int).values
                proto_v = df.get("PROTOCOL", 6).astype(int).values

                in_bytes_v = df.get("IN_BYTES", 0).astype(int).values
                out_bytes_v = df.get("OUT_BYTES", 0).astype(int).values
                in_pkts_v = df.get("IN_PKTS", 0).astype(int).values
                out_pkts_v = df.get("OUT_PKTS", 0).astype(int).values

                duration_ms_v = df.get("FLOW_DURATION_MILLISECONDS", 0).astype(float).values
                tcp_flags_v = df.get("TCP_FLAGS", 0).astype(int).values

                label_v = df.get("Label", 0).astype(int).values
                attack_v = df.get("Attack", "benign").astype(str).str.lower().values

                for i in range(len(df)):
                    dur_s = max(0.0, duration_ms_v[i] / 1000.0)
                    ts_start = global_ts
                    ts_end = global_ts + dur_s
                    global_ts += max(0.0001, dur_s * 0.05)

                    pr_code = proto_v[i]
                    if pr_code == 6:
                        proto_norm = "tcp"
                    elif pr_code == 17:
                        proto_norm = "udp"
                    elif pr_code == 1:
                        proto_norm = "icmp"
                    else:
                        proto_norm = "other"

                    flags = tcp_flags_v[i]
                    syn_flag = 1 if (flags & 0x02) else 0
                    ack_flag = 1 if (flags & 0x10) else 0
                    rst_flag = 1 if (flags & 0x04) else 0

                    conn_code = 0
                    if rst_flag > 0:
                        conn_code = 2  # Reset
                    elif out_pkts_v[i] == 0 and in_pkts_v[i] > 0:
                        conn_code = 1  # Unanswered

                    lbl = 1 if label_v[i] > 0 else 0
                    cat_raw = attack_v[i]
                    cat_norm = NF_CATEGORY_MAP.get(cat_raw, "Normal" if lbl == 0 else "Attack")

                    try:
                        flow = CanonicalFlow(
                            timestamp_start=ts_start,
                            timestamp_end=ts_end,
                            src_host=f"host_src_{src_port_v[i] % 1000}",
                            dst_host=f"host_dst_{dst_port_v[i] % 1000}",
                            src_port=int(src_port_v[i]),
                            dst_port=int(dst_port_v[i]),
                            protocol=proto_norm,
                            bytes_src=int(in_bytes_v[i]),
                            bytes_dst=int(out_bytes_v[i]),
                            pkts_src=int(in_pkts_v[i]),
                            pkts_dst=int(out_pkts_v[i]),
                            syn_count_src=syn_flag,
                            ack_count_src=ack_flag,
                            rst_count_src=rst_flag,
                            conn_state_code=conn_code,
                            label=lbl,
                            attack_category=cat_norm,
                        )
                        report.record_flow(flow)
                        yield flow
                    except Exception:
                        report.invalid_records_dropped += 1

        return report
