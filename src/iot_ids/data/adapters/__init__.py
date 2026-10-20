"""Dataset Ingestion Adapters Module."""

from pathlib import Path

from iot_ids.data.adapters.base import DatasetAdapter, IngestionAuditReport
from iot_ids.data.adapters.bot_iot import BoTIoTAdapter
from iot_ids.data.adapters.ciciot2023 import CICIoT2023Adapter
from iot_ids.data.adapters.edge_iiotset import EdgeIIoTsetAggregatorAdapter
from iot_ids.data.adapters.nf_ton_iot import NFToNIoTAdapter
from iot_ids.data.adapters.ton_iot import ToNIoTAdapter


def build_default_adapters(raw_dir: Path, sample_rows: int = 1000) -> list[DatasetAdapter]:
    return [
        ToNIoTAdapter(raw_dir / "ToN-IoT"),
        EdgeIIoTsetAggregatorAdapter(raw_dir / "Edge-IIoTset"),
        NFToNIoTAdapter(raw_dir / "NF-ToN-IoT-v2"),
        CICIoT2023Adapter(raw_dir / "CICIoT2023"),
        BoTIoTAdapter(raw_dir / "BoT-IoT"),
    ]


__all__ = [
    "BoTIoTAdapter",
    "CICIoT2023Adapter",
    "DatasetAdapter",
    "EdgeIIoTsetAggregatorAdapter",
    "IngestionAuditReport",
    "NFToNIoTAdapter",
    "ToNIoTAdapter",
    "build_default_adapters",
]
