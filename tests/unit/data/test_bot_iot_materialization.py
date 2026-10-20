from __future__ import annotations

import runpy
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    .parent
    / "scripts"
    / "03_materialize_and_audit_features.py"
)
MATERIALIZER = runpy.run_path(str(SCRIPT))["materialize_dataset"]
FILE_NAMES = (
    "UNSW_2018_IoT_Botnet_Full5pc_1.csv",
    "UNSW_2018_IoT_Botnet_Full5pc_2.csv",
    "UNSW_2018_IoT_Botnet_Full5pc_3.csv",
    "UNSW_2018_IoT_Botnet_Full5pc_4.csv",
)


def _write_source(path: Path, offset: int, attack_value: int | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as source:
        source.write(
            "stime,dur,proto,sport,dport,saddr,daddr,spkts,dpkts,sbytes,dbytes,"
            "attack,category,flgs,state\n"
        )
        for index in range(100):
            attack = index % 2 if attack_value is None else attack_value
            category = "DoS" if attack else "Normal"
            source.write(
                f"{1_700_000_000 + offset + index},0.1,tcp,1234,80,"
                f"10.0.0.1,10.0.0.2,1,1,20,40,{attack},{category},e s,CON\n"
            )


def test_bot_iot_materialization_uses_frozen_source_file_groups(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for file_index, filename in enumerate(FILE_NAMES):
        _write_source(raw_dir / filename, file_index * 1000)

    train, validation, test = MATERIALIZER(
        "BoT-IoT",
        runpy.run_path(str(SCRIPT))["BoTIoTAdapter"](raw_dir),
        sample_limit=60,
        output_dir=tmp_path / "processed",
    )

    assert len(train) == 36
    assert len(validation) == 12
    assert len(test) == 12
    manifest = (tmp_path / "processed" / "BoT-IoT" / "split_manifest.json").read_text(
        encoding="utf-8"
    )
    assert '"split_strategy": "bot_iot_source_file_holdout_v1"' in manifest
    assert '"split_protocol_frozen_before_label_audit": true' in manifest
    assert '"train": [\n      "UNSW_2018_IoT_Botnet_Full5pc_1.csv",\n      "UNSW_2018_IoT_Botnet_Full5pc_2.csv"' in manifest
    assert set(train["label"].unique()) == {0, 1}
    assert set(validation["label"].unique()) == {0, 1}
    assert set(test["label"].unique()) == {0, 1}


def test_bot_iot_materialization_rejects_unexpected_source_file_set(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_source(raw_dir / FILE_NAMES[0], 0)

    adapter = runpy.run_path(str(SCRIPT))["BoTIoTAdapter"](raw_dir)
    with pytest.raises(ValueError, match="do not match the pre-registered 5% subset"):
        MATERIALIZER(
            "BoT-IoT",
            adapter,
            sample_limit=60,
            output_dir=tmp_path / "processed",
        )


def test_bot_iot_materialization_fails_if_a_locked_partition_lacks_a_class(
    tmp_path: Path,
):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for file_index, filename in enumerate(FILE_NAMES):
        _write_source(
            raw_dir / filename,
            file_index * 1000,
            attack_value=1 if file_index == 3 else None,
        )

    adapter = runpy.run_path(str(SCRIPT))["BoTIoTAdapter"](raw_dir)
    with pytest.raises(ValueError, match="test split must contain both labels"):
        MATERIALIZER(
            "BoT-IoT",
            adapter,
            sample_limit=60,
            output_dir=tmp_path / "processed",
        )
