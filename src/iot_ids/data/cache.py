from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def file_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cache_fingerprint(
    source_files: Iterable[Path],
    schema_version: str,
    label_mapping_version: str,
    feature_version: str,
    config: dict[str, object] | None = None,
) -> str:
    payload = {
        "source_files": [file_fingerprint(path) for path in sorted(source_files)],
        "schema_version": schema_version,
        "label_mapping_version": label_mapping_version,
        "feature_version": feature_version,
        "config": config or {},
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]

