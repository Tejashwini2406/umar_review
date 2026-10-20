"""Abstract Dataset Adapter Interface for Canonical Flow Ingestion.

Defines the contract for converting raw telemetry datasets (CSV, Parquet, PCAP)
into validated CanonicalFlow records.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List

from iot_ids.data.flow_object import CanonicalFlow


@dataclass
class IngestionAuditReport:
    """Audit summary for a dataset ingestion run."""

    dataset_name: str
    source_files: List[str] = field(default_factory=list)
    raw_records_read: int = 0
    canonical_flows_emitted: int = 0
    invalid_records_dropped: int = 0
    reconstructed_flows: int = 0
    timestamp_min: float = float("inf")
    timestamp_max: float = float("-inf")
    label_distribution: Dict[int, int] = field(default_factory=dict)
    attack_category_distribution: Dict[str, int] = field(default_factory=dict)
    missing_field_counts: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def record_flow(self, flow: CanonicalFlow) -> None:
        """Register a valid emitted flow into the audit summary."""
        self.canonical_flows_emitted += 1
        self.timestamp_min = min(self.timestamp_min, flow.timestamp_start)
        self.timestamp_max = max(self.timestamp_max, flow.timestamp_end)
        self.label_distribution[flow.label] = self.label_distribution.get(flow.label, 0) + 1
        cat = flow.attack_category
        self.attack_category_distribution[cat] = self.attack_category_distribution.get(cat, 0) + 1


from iot_ids.data.schema.profile import DatasetAudit, FileProfile


class DatasetAdapter(ABC):
    """Abstract base class for dataset-specific ingestion adapters."""

    name: str

    def __init__(self, raw_path: Path):
        self.raw_path = raw_path

    @abstractmethod
    def stream_canonical_flows(
        self, chunksize: int = 100000
    ) -> Generator[CanonicalFlow, None, IngestionAuditReport]:
        """Yields CanonicalFlow objects in strict chronological stream.
        
        Returns IngestionAuditReport upon completion.
        """

    @abstractmethod
    def get_source_files(self) -> List[Path]:
        """Returns list of raw source files managed by this adapter."""

    def discover_files(self) -> list[Path]:
        """Return source files using the discovery name used by materializers."""
        return self.get_source_files()

    def audit(self, sample_rows: int = 1000) -> DatasetAudit:
        """Runs a quick audit scan over dataset source files."""
        files = self.get_source_files()
        file_profiles = [FileProfile.from_path(p) for p in files if p.exists()]
        total_size = sum(fp.size_bytes for fp in file_profiles)
        return DatasetAudit(
            name=getattr(self, "name", "Dataset"),
            root=str(self.raw_path),
            status="AVAILABLE" if len(files) > 0 else "MISSING",
            files=file_profiles,
            total_size_bytes=total_size,
            total_files=len(files),
            source_total_size_bytes=total_size,
            source_total_files=len(files),
            notes=getattr(self, "dataset_notes", lambda audit: [])(None),
        )


