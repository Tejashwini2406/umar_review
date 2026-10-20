from __future__ import annotations

import re
from collections.abc import Generator
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from iot_ids.data.adapters.base import DatasetAdapter, IngestionAuditReport
from iot_ids.data.flow_object import CanonicalFlow
from iot_ids.data.schema.profile import DatasetAudit


class BoTIoTAdapter(DatasetAdapter):
    """Stream BoT-IoT CSV records as canonical flows."""

    name = "BoT-IoT"
    metadata_files: ClassVar[frozenset[str]] = frozenset({"data_names.csv"})
    required_columns: ClassVar[frozenset[str]] = frozenset(
        {
            "stime",
            "dur",
            "proto",
            "sport",
            "dport",
            "saddr",
            "daddr",
            "spkts",
            "dpkts",
            "sbytes",
            "dbytes",
            "attack",
            "category",
            "flgs",
            "state",
        }
    )

    def get_source_files(self) -> list[Path]:
        if self.raw_path.is_file():
            candidates = [self.raw_path]
        elif self.raw_path.is_dir():
            candidates = list(self.raw_path.rglob("*"))
        else:
            return []

        return sorted(
            path
            for path in candidates
            if path.is_file()
            and path.suffix.lower() == ".csv"
            and path.name.lower() not in self.metadata_files
        )

    def stream_canonical_flows(
        self, chunksize: int = 100000
    ) -> Generator[CanonicalFlow, None, IngestionAuditReport]:
        if chunksize < 1:
            raise ValueError("chunksize must be a positive integer")
        files = self.get_source_files()
        if not files:
            raise FileNotFoundError(f"No BoT-IoT CSV files found under {self.raw_path}")

        report = IngestionAuditReport(
            dataset_name=self.name,
            source_files=[str(path) for path in files],
        )
        last_timestamp: float | None = None

        for file_path in files:
            file_invalid = 0
            reader = pd.read_csv(
                file_path,
                chunksize=chunksize,
                low_memory=False,
                skipinitialspace=True,
            )
            for chunk in reader:
                chunk.columns = [str(column).strip().lower() for column in chunk.columns]
                if len(set(chunk.columns)) != len(chunk.columns):
                    raise ValueError(f"Duplicate normalized columns in {file_path}")
                missing = sorted(self.required_columns.difference(chunk.columns))
                if missing:
                    raise ValueError(
                        f"BoT-IoT source {file_path} is missing required columns: {missing}"
                    )

                report.raw_records_read += len(chunk)
                numeric = {
                    name: pd.to_numeric(chunk[name], errors="coerce").to_numpy(dtype=np.float64)
                    for name in (
                        "stime",
                        "dur",
                        "sport",
                        "dport",
                        "spkts",
                        "dpkts",
                        "sbytes",
                        "dbytes",
                        "attack",
                    )
                }
                protocols = chunk["proto"].fillna("").astype(str).str.strip().str.lower()
                categories = chunk["category"].fillna("").astype(str).str.strip()
                integer_fields = (
                    "sport",
                    "dport",
                    "spkts",
                    "dpkts",
                    "sbytes",
                    "dbytes",
                    "attack",
                )
                valid = np.ones(len(chunk), dtype=bool)
                for values in numeric.values():
                    valid &= np.isfinite(values)
                for field in integer_fields:
                    valid &= numeric[field] == np.floor(numeric[field])
                valid &= numeric["dur"] >= 0
                valid &= (numeric["sport"] >= 0) & (numeric["sport"] <= 65535)
                valid &= (numeric["dport"] >= 0) & (numeric["dport"] <= 65535)
                for field in ("spkts", "dpkts", "sbytes", "dbytes"):
                    valid &= numeric[field] >= 0
                valid &= np.isin(numeric["attack"], (0, 1))
                valid &= protocols.ne("").to_numpy()
                valid &= categories.ne("").to_numpy()

                source_hosts = chunk["saddr"].fillna("").astype(str).str.strip().to_numpy()
                destination_hosts = chunk["daddr"].fillna("").astype(str).str.strip().to_numpy()
                valid &= np.asarray(
                    [
                        bool(source) and bool(destination)
                        for source, destination in zip(source_hosts, destination_hosts)
                    ],
                    dtype=bool,
                )
                file_invalid += int((~valid).sum())

                flags = chunk["flgs"].fillna("").astype(str).to_numpy()
                states = chunk["state"].fillna("").astype(str).to_numpy()

                for index in np.flatnonzero(valid):
                    timestamp_start = float(numeric["stime"][index])
                    if (
                        last_timestamp is not None
                        and timestamp_start < last_timestamp
                    ):
                        raise ValueError(
                            "BoT-IoT records are not in chronological order near "
                            f"{file_path} (stream record "
                            f"{report.raw_records_read - len(chunk) + index + 1})"
                        )
                    last_timestamp = timestamp_start
                    protocol = protocols.iloc[index]
                    protocol = {
                        "6": "tcp",
                        "17": "udp",
                        "1": "icmp",
                    }.get(protocol, protocol)
                    if protocol not in {"tcp", "udp", "icmp"}:
                        protocol = "other"

                    label = int(numeric["attack"][index])
                    category = categories.iloc[index]
                    attack_category = self._attack_category(category, label)
                    flag_tokens = set(re.findall(r"[A-Z]+", flags[index].upper()))
                    state_tokens = set(re.findall(r"[A-Z0-9]+", states[index].upper()))
                    connection_state = self._connection_state(state_tokens)

                    flow = CanonicalFlow(
                        timestamp_start=timestamp_start,
                        timestamp_end=float(numeric["stime"][index] + numeric["dur"][index]),
                        src_host=source_hosts[index],
                        dst_host=destination_hosts[index],
                        src_port=int(numeric["sport"][index]),
                        dst_port=int(numeric["dport"][index]),
                        protocol=protocol,
                        bytes_src=int(numeric["sbytes"][index]),
                        bytes_dst=int(numeric["dbytes"][index]),
                        pkts_src=int(numeric["spkts"][index]),
                        pkts_dst=int(numeric["dpkts"][index]),
                        syn_count_src=int(bool(flag_tokens.intersection({"S", "SYN"}))),
                        ack_count_src=int(bool(flag_tokens.intersection({"A", "ACK"}))),
                        rst_count_src=int(bool(flag_tokens.intersection({"R", "RST"}))),
                        conn_state_code=connection_state,
                        label=label,
                        attack_category=attack_category,
                    )
                    report.record_flow(flow)
                    yield flow

            if file_invalid:
                report.invalid_records_dropped += file_invalid
                report.warnings.append(
                    f"{file_path}: dropped {file_invalid} rows with invalid numeric "
                    "values or missing endpoint addresses"
                )

        return report

    @staticmethod
    def _attack_category(category: str, label: int) -> str:
        if label == 0:
            return "Normal"
        normalized = category.strip().lower()
        mapping = {
            "ddos": "DDoS",
            "dos": "DoS",
            "reconnaissance": "Scanning",
            "scan": "Scanning",
            "theft": "Theft",
        }
        return mapping.get(normalized, category.strip() or "Attack")

    @staticmethod
    def _connection_state(tokens: set[str]) -> int:
        if tokens.intersection({"RST", "RSTR", "RSTO"}):
            return 2
        if "REJ" in tokens:
            return 1
        if tokens.intersection({"S0", "INT", "REQ"}):
            return 3
        return 0

    def dataset_notes(self, audit: DatasetAudit | None) -> list[str]:
        audit_root = Path(audit.root) if audit is not None else self.raw_path
        has_names = audit_root.is_dir() and (audit_root / "data_names.csv").is_file()
        return [
            f"data_names.csv present: {has_names}; treat it as metadata, not a feature source.",
            "The attack label comes from `attack`; category/subcategory are metadata only.",
            "BoT-IoT flgs stores aggregate flags, so SYN/ACK/RST fields are presence indicators, not packet counts.",
            "Input file and row order is preserved; non-monotonic timestamps are rejected.",
        ]
