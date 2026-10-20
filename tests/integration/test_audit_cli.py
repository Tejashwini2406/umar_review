from __future__ import annotations

import json
from pathlib import Path

from iot_ids.data.audit import run_audit


def test_run_audit_writes_json_and_markdown(tmp_path: Path) -> None:
    data_root = tmp_path / "data" / "raw"
    output = tmp_path / "reports" / "eda"
    run_audit(data_root=data_root, output=output, sample_rows=5)

    payload = json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))
    assert "environment" in payload
    assert len(payload["datasets"]) == 5
    assert any(dataset["name"] == "BoT-IoT" for dataset in payload["datasets"])
    assert (output / "dataset_audit.md").exists()
