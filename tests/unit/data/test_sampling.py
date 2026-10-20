"""Tests for bounded sampling and leakage-aware split primitives."""
import json
import runpy
from pathlib import Path

import pandas as pd
import pytest

from iot_ids.data.adapters.ciciot2023 import CICIoT2023Adapter
from iot_ids.data.flow_object import CanonicalFlow
from iot_ids.data.sampling import reservoir_sample, stratified_random_split

STAGE3_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "03_materialize_and_audit_features.py"
)


def _flow(index: int, label: int) -> CanonicalFlow:
    timestamp = float(index)
    return CanonicalFlow(
        timestamp_start=timestamp,
        timestamp_end=timestamp,
        src_host=f"source-{index}",
        dst_host=f"destination-{index}",
        src_port=1000,
        dst_port=2000,
        protocol="tcp",
        bytes_src=1,
        bytes_dst=1,
        pkts_src=1,
        pkts_dst=1,
        syn_count_src=0,
        ack_count_src=0,
        rst_count_src=0,
        conn_state_code=0,
        label=label,
        attack_category="Normal" if label == 0 else "Attack",
    )


def test_reservoir_sample_is_bounded_reproducible_and_order_preserving():
    flows = [_flow(index, index % 2) for index in range(100)]

    sample = reservoir_sample(flows, sample_limit=20, seed=42)

    assert len(sample) == 20
    assert sample == reservoir_sample(flows, sample_limit=20, seed=42)
    assert [flow.timestamp_start for flow in sample] == sorted(
        flow.timestamp_start for flow in sample
    )


def test_stratified_split_is_disjoint_and_contains_both_classes():
    flows = [_flow(index, index % 2) for index in range(100)]

    splits = stratified_random_split(flows, seed=42)

    split_ids = [{id(flow) for flow in split} for split in splits]
    assert not split_ids[0] & split_ids[1]
    assert not split_ids[0] & split_ids[2]
    assert not split_ids[1] & split_ids[2]
    assert set.union(*split_ids) == {id(flow) for flow in flows}
    for split in splits:
        assert {flow.label for flow in split} == {0, 1}
        assert [flow.timestamp_start for flow in split] == sorted(
            flow.timestamp_start for flow in split
        )


def test_stratified_split_is_deterministic():
    flows = [_flow(index, index % 2) for index in range(100)]

    first = stratified_random_split(flows, seed=42)
    second = stratified_random_split(flows, seed=42)

    assert [[id(flow) for flow in split] for split in first] == [
        [id(flow) for flow in split] for split in second
    ]


def test_stratified_split_rejects_class_too_small_for_three_splits():
    flows = [_flow(0, 0), _flow(1, 0), _flow(2, 1)]

    with pytest.raises(ValueError, match="at least 3 sampled flows"):
        stratified_random_split(flows, seed=42)


def test_ciciot_adapter_selects_only_requested_official_partition(tmp_path):
    train = tmp_path / "train" / "train.csv"
    validation = tmp_path / "validation" / "validation.csv"
    test = tmp_path / "test" / "test.csv"
    for path in (train, validation, test):
        path.parent.mkdir(parents=True)
        path.touch()

    assert CICIoT2023Adapter(tmp_path, partition="train").get_source_files() == [
        train
    ]
    assert CICIoT2023Adapter(tmp_path, partition="validation").get_source_files() == [
        validation
    ]
    assert CICIoT2023Adapter(tmp_path, partition="test").get_source_files() == [test]


def test_ciciot_materialization_keeps_official_partitions_and_disables_fake_history(
    tmp_path,
):
    source_root = tmp_path / "ciciot"
    columns = {
        "flow_duration": [1.0] * 8,
        "Protocol Type": [6] * 8,
        "Tot sum": [100] * 8,
        "AVG": [50.0] * 8,
        "Number": [2] * 8,
        "Srate": [1.0] * 8,
        "Drate": [1.0] * 8,
        "syn_count": [1] * 8,
        "ack_count": [1] * 8,
        "rst_count": [0] * 8,
        "label": ["BENIGN", "DDoS"] * 4,
    }
    for partition in ("train", "validation", "test"):
        partition_dir = source_root / partition
        partition_dir.mkdir(parents=True)
        pd.DataFrame(columns).to_csv(partition_dir / f"{partition}.csv", index=False)

    materialize_dataset = runpy.run_path(str(STAGE3_SCRIPT))["materialize_dataset"]
    train, validation, test = materialize_dataset(
        "CICIoT2023",
        CICIoT2023Adapter(source_root),
        sample_limit=30,
        output_dir=tmp_path / "processed",
    )

    assert set(train["split"]) == {"train"}
    assert set(validation["split"]) == {"val"}
    assert set(test["split"]) == {"test"}
    for split in (train, validation, test):
        assert set(split["label"]) == {0, 1}
        assert (split["temporal_iat_mean"] == 0).all()
        assert (split["behavioral_dst_diversity"] == 0).all()

    manifest = json.loads(
        (tmp_path / "processed" / "CICIoT2023" / "split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["split_strategy"] == "official_source_partitions"
    assert manifest["source_partitions"]["train"] != manifest["source_partitions"]["test"]
