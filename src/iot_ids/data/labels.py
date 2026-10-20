from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


LABEL_MAPPING_VERSION = "label-taxonomy-v1"


@dataclass(frozen=True, slots=True)
class LabelRecord:
    dataset: str
    raw_label: str
    binary_label: str
    attack_family: str | None
    attack_subtype: str | None
    mapping_version: str = LABEL_MAPPING_VERSION


def normalize_binary_label(dataset: str, raw_label: object, source_path: str | Path | None = None) -> LabelRecord:
    raw = str(raw_label).strip()
    dataset_key = dataset.lower()

    if dataset_key == "cicids2017":
        benign = raw.upper() == "BENIGN"
        return LabelRecord(dataset, raw, "BENIGN" if benign else "ATTACK", None if benign else raw, raw)

    if dataset_key == "edge-iiotset":
        benign = raw in {"0", "0.0", "Normal", "normal", "BENIGN", "Benign"}
        return LabelRecord(dataset, raw, "BENIGN" if benign else "ATTACK", None if benign else raw, raw)

    if dataset_key == "bot-iot":
        benign = raw in {"0", "0.0", "Normal", "normal", "BENIGN", "Benign"}
        return LabelRecord(dataset, raw, "BENIGN" if benign else "ATTACK", None if benign else raw, raw)

    if dataset_key == "n-baiot":
        if source_path is None:
            raise ValueError("N-BaIoT label mapping requires source filename metadata.")
        label = nbaiot_label_from_filename(Path(source_path).name)
        benign = label == "benign"
        family, _, subtype = label.partition(".")
        return LabelRecord(
            dataset=dataset,
            raw_label=label,
            binary_label="BENIGN" if benign else "ATTACK",
            attack_family=None if benign else family,
            attack_subtype=None if benign else subtype or None,
        )

    raise ValueError(f"Unsupported dataset for label mapping: {dataset}")


def nbaiot_label_from_filename(filename: str) -> str:
    parts = filename.split(".")
    if len(parts) < 3:
        raise ValueError(f"Cannot derive N-BaIoT traffic label from filename: {filename}")
    family = parts[1].lower()
    if family == "benign":
        return "benign"
    return f"{family}.{parts[2].lower()}"

