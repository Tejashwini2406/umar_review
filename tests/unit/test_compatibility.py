from __future__ import annotations

import json
from pathlib import Path


def test_nbaiot_cross_dataset_cells_are_explicitly_incompatible() -> None:
    payload = json.loads(Path("configs/experiments/cross_dataset_compatibility.json").read_text())
    cells = payload["cells"]
    for dataset in ("CICIDS2017", "Edge-IIoTset", "BoT-IoT"):
        assert cells[f"{dataset}->N-BaIoT"]["status"] == "N/A"
        assert cells[f"N-BaIoT->{dataset}"]["status"] == "N/A"
    assert cells["N-BaIoT->N-BaIoT"]["profile"] == "NBAIOT_SOURCE_AGGREGATE"


def test_flow_datasets_use_flow_compatible_profile() -> None:
    payload = json.loads(Path("configs/experiments/cross_dataset_compatibility.json").read_text())
    for source in ("CICIDS2017", "Edge-IIoTset", "BoT-IoT"):
        for target in ("CICIDS2017", "Edge-IIoTset", "BoT-IoT"):
            assert payload["cells"][f"{source}->{target}"]["profile"] == "FLOW_COMPATIBLE_C_E_B"
