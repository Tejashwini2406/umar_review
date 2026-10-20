from __future__ import annotations

from pathlib import Path

from iot_ids.data.adapters.base import DatasetAdapter
from iot_ids.data.schema.profile import DatasetAudit


class CICIDS2017Adapter(DatasetAdapter):
    name = "CICIDS2017"

    def discover_files(self) -> list[Path]:
        return sorted(self.root.rglob("*.csv"))

    def dataset_notes(self, audit: DatasetAudit) -> list[str]:
        return [
            "Treat MachineLearningCSV daily/file boundaries as source metadata for leakage-aware analysis.",
            "Known leakage-prone fields include Flow ID, source/destination IPs, ports where they encode hosts, and Timestamp.",
        ]

