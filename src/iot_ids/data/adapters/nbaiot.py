from __future__ import annotations

from pathlib import Path

from iot_ids.data.adapters.base import DatasetAdapter
from iot_ids.data.schema.profile import DatasetAudit


class NBaIoTAdapter(DatasetAdapter):
    name = "N-BaIoT"
    metadata_files = {"data_summary.csv", "device_info.csv", "features.csv"}

    def discover_files(self) -> list[Path]:
        return sorted(
            p for p in self.root.rglob("*.csv") if p.name.lower() not in self.metadata_files
        )

    def audit(self) -> DatasetAudit:
        audit = super().audit()
        if audit.status != "ok":
            return audit

        label_counts: dict[str, int] = {}
        device_counts: dict[str, int] = {}
        for profile in audit.files:
            path = Path(profile.path)
            parts = path.name.split(".")
            if len(parts) < 2:
                continue
            device = parts[0]
            attack_family = parts[1].lower()
            subtype = parts[2].lower() if len(parts) > 2 else "unknown"
            label = "benign" if attack_family == "benign" else f"{attack_family}.{subtype}"
            rows = profile.row_count or 0
            label_counts[label] = label_counts.get(label, 0) + rows
            device_counts[device] = device_counts.get(device, 0) + rows

        audit.label_columns_seen = ["__filename_label__"]
        audit.label_distribution = {"__filename_label__": dict(sorted(label_counts.items()))}
        audit.metadata["device_row_counts"] = dict(sorted(device_counts.items()))
        return audit

    def dataset_notes(self, audit: DatasetAudit) -> list[str]:
        return [
            "Device identity, attack family, and subtype should be derived from filenames as metadata.",
            "Device identity is useful for grouped/domain splits but must not be a predictive feature by default.",
        ]
