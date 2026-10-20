from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import pandas as pd
import psutil
from iot_ids.features.schema import load_feature_schema


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def deep_schema_inventory() -> dict[str, object]:
    audit = load_json(ROOT / "reports" / "eda" / "dataset_audit.json")
    leakage = load_json(ROOT / "configs" / "features" / "leakage_metadata.json")
    inventory: dict[str, object] = {"version": "deep-schema-v1", "datasets": {}}
    for ds in audit["datasets"]:
        dataset = ds["name"]
        roles = leakage["datasets"].get(dataset, {})
        columns = []
        for name in ds["schema_union"]:
            role = roles.get(name, {}).get("role", "SAFE_FEATURE")
            normalized = name.lower()
            semantic = "unknown"
            if any(token in normalized for token in ("byte", "sbytes", "dbytes")):
                semantic = "byte statistic"
            elif any(token in normalized for token in ("pkt", "packet", "pkts", "spkts", "dpkts")):
                semantic = "packet statistic"
            elif any(token in normalized for token in ("time", "stime", "ltime", "iat", "jit")):
                semantic = "time/order statistic"
            elif any(token in normalized for token in ("addr", "host", "mac", "device")):
                semantic = "entity/address metadata"
            elif any(token in normalized for token in ("label", "attack", "category", "subcategory")):
                semantic = "label/taxonomy"
            elif any(token in normalized for token in ("port", "proto", "state", "service")):
                semantic = "protocol/service"
            elif any(token in normalized for token in ("mean", "std", "variance", "radius", "covariance", "weight")):
                semantic = "source-provided aggregate statistic"
            columns.append({"name": name, "semantic_category": semantic, "leakage_role": role})
        inventory["datasets"][dataset] = {
            "row_count": sum((f.get("row_count") or 0) for f in ds["files"]),
            "candidate_files": ds["total_files"],
            "source_files": ds.get("source_total_files"),
            "columns": columns,
            "label_columns": ds["label_columns_seen"],
            "temporal_columns": ds["temporal_columns_seen"],
            "entity_columns": ds["entity_columns_seen"],
            "label_distribution": ds.get("label_distribution", {}),
        }
    return inventory


def feature_availability() -> list[dict[str, object]]:
    features = load_feature_schema(ROOT / "configs" / "features" / "canonical_schema.json")
    rows = []
    for feature in features:
        row = {
            "canonical_name": feature.name,
            "level": feature.level,
            "valid_for": feature.valid_for,
            "leakage_role": feature.leakage_role,
        }
        for dataset in ("CICIDS2017", "Edge-IIoTset", "BoT-IoT", "N-BaIoT"):
            row[dataset] = "; ".join(feature.availability.get(dataset, []))
        rows.append(row)
    return rows


def write_feature_availability(rows: list[dict[str, object]]) -> None:
    write_json(ROOT / "reports" / "eda" / "feature_availability.json", rows)
    csv_path = ROOT / "reports" / "eda" / "feature_availability.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_path = ROOT / "reports" / "eda" / "feature_availability.md"
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")).replace("|", "/") for header in headers) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def capture_environment() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_total_bytes": psutil.virtual_memory().total,
        "dependencies": {
            package: __import__(package).__version__
            for package in ("pandas", "numpy", "pyarrow", "sklearn", "scipy", "joblib", "yaml", "pytest")
        },
    }


def main() -> None:
    inventory = deep_schema_inventory()
    write_json(ROOT / "reports" / "eda" / "deep_schema_inventory.json", inventory)
    rows = feature_availability()
    write_feature_availability(rows)
    label_mapping = load_json(ROOT / "configs" / "datasets" / "label_taxonomy.json")
    write_json(ROOT / "reports" / "eda" / "label_mapping.json", label_mapping)
    leakage = load_json(ROOT / "configs" / "features" / "leakage_metadata.json")
    write_json(ROOT / "reports" / "eda" / "leakage_analysis.json", leakage)
    write_json(ROOT / "reports" / "eda" / "environment.json", capture_environment())


if __name__ == "__main__":
    main()
