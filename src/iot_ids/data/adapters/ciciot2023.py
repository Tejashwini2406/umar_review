"""CICIoT2023 Dataset Adapter.

Ingests CICIoT2023 flow statistics CSV telemetry into CanonicalFlow records.
"""
from __future__ import annotations

import re
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd

from iot_ids.data.adapters.base import DatasetAdapter, IngestionAuditReport
from iot_ids.data.flow_object import CanonicalFlow

CIC_CATEGORY_MAP = {
    "benign": "Normal",
    "ddos": "DDoS",
    "dos": "DoS",
    "vulnerability_scan": "Scanning",
    "recon": "Scanning",
    "spoofing": "MITM",
    "web": "Injection",
    "bruteforce": "Password",
    "mirai": "DDoS",
}


def map_cic_label_category(label_str: str) -> tuple[int, str]:
    """Map raw CICIoT2023 label string to binary label and category taxonomy."""
    lbl_clean = str(label_str).strip().lower()
    if "benign" in lbl_clean:
        return 0, "Normal"

    lbl_binary = 1
    cat_norm = "Attack"
    for key, cat in CIC_CATEGORY_MAP.items():
        if key in lbl_clean:
            cat_norm = cat
            break
    return lbl_binary, cat_norm


class CICIoT2023Adapter(DatasetAdapter):
    """Adapter for CICIoT2023 flow summary CSV records."""

    name = "CICIoT2023"

    def __init__(self, raw_path: Path, partition: str | None = None):
        super().__init__(raw_path)
        if partition not in (None, "train", "validation", "test"):
            raise ValueError(
                "partition must be one of None, 'train', 'validation', or 'test'"
            )
        self.partition = partition

    def _path_partitions(self, path: Path) -> set[str]:
        partitions: set[str] = set()
        relative_path = (
            path.relative_to(self.raw_path)
            if self.raw_path.is_dir()
            else Path(path.name)
        )
        for part in relative_path.parts:
            tokens = set(re.split(r"[^a-z0-9]+", part.lower()))
            if "train" in tokens:
                partitions.add("train")
            if tokens.intersection({"validation", "valid", "val"}):
                partitions.add("validation")
            if "test" in tokens:
                partitions.add("test")
        return partitions

    def get_source_files(self) -> list[Path]:
        files = [self.raw_path] if self.raw_path.is_file() else sorted(self.raw_path.rglob("*.csv"))
        if self.partition is None:
            return files

        selected = [
            path for path in files if self.partition in self._path_partitions(path)
        ]
        if not selected:
            raise FileNotFoundError(
                f"No CICIoT2023 source files found for the official "
                f"{self.partition!r} partition under {self.raw_path}"
            )
        return selected

    def stream_canonical_flows(
        self, chunksize: int = 100000
    ) -> Generator[CanonicalFlow, None, IngestionAuditReport]:
        files = self.get_source_files()
        report = IngestionAuditReport(
            dataset_name=self.name,
            source_files=[str(p) for p in files],
        )

        global_ts = 1672531200.0  # Synthetic UNIX epoch baseline (2023-01-01)

        for file_path in files:
            reader = pd.read_csv(file_path, chunksize=chunksize, low_memory=False)
            for chunk in reader:
                report.raw_records_read += len(chunk)

                duration_v = pd.to_numeric(chunk.get("flow_duration", 0), errors="coerce").fillna(0.0).clip(lower=0.0).values
                proto_code_v = pd.to_numeric(chunk.get("Protocol Type", 6), errors="coerce").fillna(6).astype(int).values
                
                tot_sum_v = pd.to_numeric(chunk.get("Tot sum", 0), errors="coerce").fillna(0).clip(lower=0).astype(int).values
                avg_size_v = pd.to_numeric(chunk.get("AVG", 0), errors="coerce").fillna(0).clip(lower=0).values
                num_pkts_v = pd.to_numeric(chunk.get("Number", 1), errors="coerce").fillna(1).clip(lower=1).astype(int).values
                
                srate_v = pd.to_numeric(chunk.get("Srate", 0), errors="coerce").fillna(0.0).clip(lower=0.0).values
                drate_v = pd.to_numeric(chunk.get("Drate", 0), errors="coerce").fillna(0.0).clip(lower=0.0).values
                
                syn_cnt_v = pd.to_numeric(chunk.get("syn_count", 0), errors="coerce").fillna(0).astype(int).values
                ack_cnt_v = pd.to_numeric(chunk.get("ack_count", 0), errors="coerce").fillna(0).astype(int).values
                rst_cnt_v = pd.to_numeric(chunk.get("rst_count", 0), errors="coerce").fillna(0).astype(int).values
                
                labels_v = chunk.get("label", "BENIGN").astype(str).values

                for i in range(len(chunk)):
                    dur = float(duration_v[i])
                    ts_start = global_ts
                    ts_end = global_ts + dur
                    global_ts += max(0.001, dur * 0.1)

                    pr = proto_code_v[i]
                    if pr == 6:
                        proto_norm = "tcp"
                    elif pr == 17:
                        proto_norm = "udp"
                    elif pr == 1:
                        proto_norm = "icmp"
                    else:
                        proto_norm = "other"

                    tot_p = int(num_pkts_v[i])
                    sr = float(srate_v[i])
                    dr = float(drate_v[i])
                    total_rate = sr + dr

                    if total_rate > 0:
                        src_p = int(np.clip(round(tot_p * (sr / total_rate)), 1, tot_p))
                    else:
                        src_p = max(1, tot_p // 2)
                    dst_p = max(0, tot_p - src_p)

                    avg_sz = float(avg_size_v[i])
                    tot_b = int(tot_sum_v[i])
                    if tot_b == 0 and avg_sz > 0:
                        tot_b = int(avg_sz * tot_p)

                    src_b = int(avg_sz * src_p)
                    dst_b = max(0, tot_b - src_b)

                    lbl, cat_norm = map_cic_label_category(labels_v[i])
                    rst = int(rst_cnt_v[i])
                    conn_code = 2 if rst > 0 else (1 if dst_p == 0 else 0)

                    try:
                        flow = CanonicalFlow(
                            timestamp_start=ts_start,
                            timestamp_end=ts_end,
                            src_host=f"host_cic_{i % 500}",
                            dst_host=f"host_cic_{(i + 1) % 500}",
                            src_port=80 if proto_norm == "tcp" else 53,
                            dst_port=8080 if proto_norm == "tcp" else 53,
                            protocol=proto_norm,
                            bytes_src=src_b,
                            bytes_dst=dst_b,
                            pkts_src=src_p,
                            pkts_dst=dst_p,
                            syn_count_src=int(syn_cnt_v[i]),
                            ack_count_src=int(ack_cnt_v[i]),
                            rst_count_src=rst,
                            conn_state_code=conn_code,
                            label=lbl,
                            attack_category=cat_norm,
                        )
                        report.record_flow(flow)
                        yield flow
                    except ValueError:
                        report.invalid_records_dropped += 1

        return report
