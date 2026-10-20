from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class FileProfile:
    path: str
    size_bytes: int
    suffix: str
    row_count: int | None = None
    rows_sampled: int = 0
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    label_candidates: list[str] = field(default_factory=list)
    label_distribution_sample: dict[str, int] = field(default_factory=dict)
    missing_sample: dict[str, int] = field(default_factory=dict)
    inf_sample: dict[str, int] = field(default_factory=dict)
    constant_columns_sample: list[str] = field(default_factory=list)
    high_cardinality_columns_sample: list[str] = field(default_factory=list)
    suspicious_leakage_columns: list[str] = field(default_factory=list)
    read_error: str | None = None

    @classmethod
    def from_path(cls, path: Path) -> "FileProfile":
        return cls(path=str(path), size_bytes=path.stat().st_size, suffix=path.suffix.lower())


@dataclass(slots=True)
class DatasetAudit:
    name: str
    root: str
    status: str
    files: list[FileProfile] = field(default_factory=list)
    total_size_bytes: int = 0
    total_files: int = 0
    source_total_size_bytes: int = 0
    source_total_files: int = 0
    schema_union: list[str] = field(default_factory=list)
    label_distribution: dict[str, dict[str, int]] = field(default_factory=dict)
    label_columns_seen: list[str] = field(default_factory=list)
    leakage_columns_seen: list[str] = field(default_factory=list)
    temporal_columns_seen: list[str] = field(default_factory=list)
    entity_columns_seen: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
