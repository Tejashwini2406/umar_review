"""Stage 3 Materialization and Statistical Audit Runner.

Ingests flows from ToN-IoT, Edge-IIoTset, NF-ToN-IoT-v2, CICIoT2023, and BoT-IoT,
materializes source-disjoint or seeded stratified train/validation/test splits,
and generates empirical statistical quality audit reports (KS, Wasserstein, MI, correlation, VIF).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import chain
from pathlib import Path

import pandas as pd

from iot_ids.data.adapters import (
    BoTIoTAdapter,
    CICIoT2023Adapter,
    EdgeIIoTsetAggregatorAdapter,
    NFToNIoTAdapter,
    ToNIoTAdapter,
)
from iot_ids.data.sampling import reservoir_sample, stratified_random_split
from iot_ids.features.canonical.behavioral import BEHAVIORAL_FEATURE_NAMES
from iot_ids.features.canonical.flow_builder import CANONICAL_18_FEATURE_NAMES, CanonicalFlowBuilder
from iot_ids.features.canonical.temporal import TEMPORAL_FEATURE_NAMES
from iot_ids.features.quality import (
    compute_correlation,
    compute_distribution_shift,
    compute_distribution_statistics,
    compute_mutual_information,
    compute_vif,
)

FEATURE_PROFILES = {
    "baseline_common": [
        "flow_duration", "flow_bytes_per_sec", "proto_tcp", "proto_udp", "proto_icmp"
    ],
    "instant_only": [
        "flow_duration", "flow_bytes_per_sec", "flow_pkts_per_sec", "mean_pkt_size",
        "payload_byte_ratio", "pkt_count_ratio", "tcp_syn_ratio",
        "proto_tcp", "proto_udp", "proto_icmp", "proto_other"
    ],
    "instant_temporal": [
        "flow_duration", "flow_bytes_per_sec", "flow_pkts_per_sec", "mean_pkt_size",
        "payload_byte_ratio", "pkt_count_ratio", "tcp_syn_ratio",
        "proto_tcp", "proto_udp", "proto_icmp", "proto_other",
        "temporal_iat_mean", "temporal_iat_cv", "temporal_flow_rate_ewma",
        "temporal_byte_rate_ewma", "temporal_syn_rate_ewma"
    ],
    "instant_behavioral": [
        "flow_duration", "flow_bytes_per_sec", "flow_pkts_per_sec", "mean_pkt_size",
        "payload_byte_ratio", "pkt_count_ratio", "tcp_syn_ratio",
        "proto_tcp", "proto_udp", "proto_icmp", "proto_other",
        "behavioral_dst_diversity", "behavioral_port_entropy", "behavioral_fanout_ratio",
        "behavioral_unanswered_ratio", "behavioral_src_activity_ewma"
    ],
    "full_multilevel": CANONICAL_18_FEATURE_NAMES,
}

RANDOM_SPLIT_SEED = 42
STATEFUL_FEATURE_DATASETS = {"Edge-IIoTset"}
STAGE3_OUTPUT_DIR = Path("data/processed/stage3_corrected")
STAGE3_REPORTS_DIR = Path("reports/stage3_corrected")
BOT_IOT_FILE_SPLIT = {
    "UNSW_2018_IoT_Botnet_Full5pc_1.csv": "train",
    "UNSW_2018_IoT_Botnet_Full5pc_2.csv": "train",
    "UNSW_2018_IoT_Botnet_Full5pc_3.csv": "validation",
    "UNSW_2018_IoT_Botnet_Full5pc_4.csv": "test",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_source_paths(adapter, paths: list[Path]) -> list[str]:
    if adapter.raw_path.is_file():
        return [path.name for path in paths]
    return [path.relative_to(adapter.raw_path).as_posix() for path in paths]


def materialize_dataset(
    dataset_name: str,
    adapter,
    sample_limit: int = 7000,
    output_dir: Path = STAGE3_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n==========================================================================")
    print(f"=== MATERIALIZING STAGE 3 DATASET: {dataset_name} ===")
    print("==========================================================================")
    
    t0 = time.time()
    if sample_limit < 15:
        raise ValueError("sample_limit must be at least 15 for stratified split coverage")

    manifest = {
        "dataset": dataset_name,
        "seed": RANDOM_SPLIT_SEED,
        "sample_limit": sample_limit,
        "sampling_method": "uniform_reservoir",
        "temporal_and_behavioral_features_supported": dataset_name
        in STATEFUL_FEATURE_DATASETS,
        "unsupported_state_features": []
        if dataset_name in STATEFUL_FEATURE_DATASETS
        else list(TEMPORAL_FEATURE_NAMES + BEHAVIORAL_FEATURE_NAMES),
    }
    if isinstance(adapter, BoTIoTAdapter):
        source_files = adapter.get_source_files()
        files_by_name = {path.name: path for path in source_files}
        if len(files_by_name) != len(source_files):
            raise ValueError("BoT-IoT source filenames must be unique")
        if set(files_by_name) != set(BOT_IOT_FILE_SPLIT):
            raise ValueError(
                "BoT-IoT source files do not match the pre-registered 5% subset: "
                f"expected {sorted(BOT_IOT_FILE_SPLIT)}, found {sorted(files_by_name)}"
            )
        partition_files = {
            split_name: [
                files_by_name[filename]
                for filename, assigned_split in BOT_IOT_FILE_SPLIT.items()
                if assigned_split == split_name
            ]
            for split_name in ("train", "validation", "test")
        }
        budgets = {
            "train": int(sample_limit * 0.6),
            "validation": int(sample_limit * 0.2),
            "test": sample_limit - int(sample_limit * 0.6) - int(sample_limit * 0.2),
        }
        partition_flows = {
            split_name: reservoir_sample(
                chain.from_iterable(
                    BoTIoTAdapter(path).stream_canonical_flows(chunksize=50000)
                    for path in paths
                ),
                budgets[split_name],
                RANDOM_SPLIT_SEED,
            )
            for split_name, paths in partition_files.items()
        }
        train_flows = partition_flows["train"]
        val_flows = partition_flows["validation"]
        test_flows = partition_flows["test"]
        manifest.update(
            {
                "split_strategy": "bot_iot_source_file_holdout_v1",
                "split_protocol_frozen_before_label_audit": True,
                "source_partitions": {
                    split_name: [
                        path.relative_to(adapter.raw_path).as_posix()
                        if adapter.raw_path.is_dir()
                        else path.name
                        for path in paths
                    ]
                    for split_name, paths in partition_files.items()
                },
                "source_file_sha256": {
                    path.name: _sha256(path) for path in source_files
                },
                "partition_sample_counts": {
                    split_name: len(flows)
                    for split_name, flows in partition_flows.items()
                },
            }
        )
    elif isinstance(adapter, CICIoT2023Adapter):
        budgets = {
            "train": int(sample_limit * 0.6),
            "validation": int(sample_limit * 0.2),
            "test": sample_limit - int(sample_limit * 0.6) - int(sample_limit * 0.2),
        }
        partition_flows = {}
        partition_files = {}
        for partition, budget in budgets.items():
            partition_adapter = CICIoT2023Adapter(adapter.raw_path, partition=partition)
            partition_files[partition] = partition_adapter.get_source_files()
            partition_flows[partition] = reservoir_sample(
                partition_adapter.stream_canonical_flows(chunksize=50000),
                budget,
                RANDOM_SPLIT_SEED,
            )
        file_sets = [set(paths) for paths in partition_files.values()]
        if any(
            file_sets[left].intersection(file_sets[right])
            for left in range(len(file_sets))
            for right in range(left + 1, len(file_sets))
        ):
            raise ValueError("CICIoT2023 source files overlap across official partitions")
        train_flows = partition_flows["train"]
        val_flows = partition_flows["validation"]
        test_flows = partition_flows["test"]
        manifest.update(
            {
                "split_strategy": "official_source_partitions",
                "source_partitions": {
                    partition: _relative_source_paths(adapter, paths)
                    for partition, paths in partition_files.items()
                },
                "partition_sample_counts": {
                    partition: len(flows)
                    for partition, flows in partition_flows.items()
                },
            }
        )
    else:
        sampled_flows = reservoir_sample(
            adapter.stream_canonical_flows(chunksize=50000),
            sample_limit,
            RANDOM_SPLIT_SEED,
        )
        train_flows, val_flows, test_flows = stratified_random_split(
            sampled_flows, RANDOM_SPLIT_SEED
        )
        manifest.update(
            {
                "split_strategy": "seeded_stratified_random",
                "source_files": _relative_source_paths(
                    adapter, adapter.get_source_files()
                ),
                "split_counts": {
                    "train": len(train_flows),
                    "validation": len(val_flows),
                    "test": len(test_flows),
                },
            }
        )

    manifest["label_counts"] = {
        split_name: {
            "benign": sum(flow.label == 0 for flow in flows),
            "attack": sum(flow.label == 1 for flow in flows),
        }
        for split_name, flows in (
            ("train", train_flows),
            ("validation", val_flows),
            ("test", test_flows),
        )
    }

    for split_name, flows in (
        ("train", train_flows),
        ("validation", val_flows),
        ("test", test_flows),
    ):
        labels = {flow.label for flow in flows}
        if labels != {0, 1}:
            raise ValueError(
                f"{dataset_name} {split_name} split must contain both labels; "
                f"found {sorted(labels)}"
            )

    n_total = len(train_flows) + len(val_flows) + len(test_flows)
    print(f"Total flows ingested: {n_total} in {time.time() - t0:.2f}s")
    print(f"{manifest['split_strategy']} -> Train: {len(train_flows)} (Benign: {sum(1 for f in train_flows if f.label==0)}, Attack: {sum(1 for f in train_flows if f.label==1)}), "
          f"Val: {len(val_flows)} (Benign: {sum(1 for f in val_flows if f.label==0)}, Attack: {sum(1 for f in val_flows if f.label==1)}), "
          f"Test: {len(test_flows)} (Benign: {sum(1 for f in test_flows if f.label==0)}, Attack: {sum(1 for f in test_flows if f.label==1)})")

    builder = CanonicalFlowBuilder(alpha=0.1, window_seconds=30.0)

    def extract_split(split_flows: list, split_name: str) -> pd.DataFrame:
        # Strict boundary state reset!
        builder.reset_state()
        rows = []
        for flow in split_flows:
            feat_dict = builder.instant_extractor.extract(flow)
            if dataset_name in STATEFUL_FEATURE_DATASETS:
                feat_dict.update(builder.temporal_extractor.extract_and_update(flow))
                feat_dict.update(builder.behavioral_extractor.extract_and_update(flow))
            else:
                feat_dict.update({name: 0.0 for name in TEMPORAL_FEATURE_NAMES})
                feat_dict.update({name: 0.0 for name in BEHAVIORAL_FEATURE_NAMES})
            
            # Metadata
            feat_dict["dataset"] = dataset_name
            feat_dict["split"] = split_name
            feat_dict["timestamp_start"] = flow.timestamp_start
            feat_dict["label"] = flow.label
            feat_dict["attack_category"] = flow.attack_category
            rows.append(feat_dict)
        return pd.DataFrame(rows)

    df_train = extract_split(train_flows, "train")
    df_val = extract_split(val_flows, "val")
    df_test = extract_split(test_flows, "test")

    # Save materialized parquet splits
    ds_dir = output_dir / dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(ds_dir / "train.parquet", index=False)
    df_val.to_parquet(ds_dir / "val.parquet", index=False)
    df_test.to_parquet(ds_dir / "test.parquet", index=False)
    (ds_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Saved materialized splits to {ds_dir}")

    return df_train, df_val, df_test


def run_stage3_audit(
    sample_limit: int = 10000, dataset_filter: str | None = None
) -> None:
    supported_datasets = {
        "ToN-IoT",
        "Edge-IIoTset",
        "NF-ToN-IoT-v2",
        "CICIoT2023",
        "BoT-IoT",
    }
    if sample_limit < 15:
        raise ValueError("sample_limit must be at least 15 for stratified split coverage")
    if dataset_filter is not None and dataset_filter not in supported_datasets:
        raise ValueError(f"Unsupported dataset filter: {dataset_filter}")

    raw_dir = Path("data/raw")
    output_dir = STAGE3_OUTPUT_DIR
    reports_dir = STAGE3_REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    adapters = [
        ("ToN-IoT", ToNIoTAdapter(raw_dir / "ToN-IoT")),
        ("Edge-IIoTset", EdgeIIoTsetAggregatorAdapter(raw_dir / "Edge-IIoTset")),
        ("NF-ToN-IoT-v2", NFToNIoTAdapter(raw_dir / "NF-ToN-IoT-v2")),
        ("CICIoT2023", CICIoT2023Adapter(raw_dir / "CICIOT23")),
        ("BoT-IoT", BoTIoTAdapter(raw_dir / "BoT-IoT")),
    ]
    if dataset_filter is not None:
        adapters = [item for item in adapters if item[0] == dataset_filter]

    dataset_dfs = {}
    stats_list = []
    mi_list = []

    for name, adapter in adapters:
        df_tr, df_v, df_te = materialize_dataset(
            name, adapter, sample_limit=sample_limit, output_dir=output_dir
        )
        full_df = pd.concat([df_tr, df_v, df_te], ignore_index=True)
        dataset_dfs[name] = full_df

        # 1. Distribution stats per split
        tr_stats = compute_distribution_statistics(df_tr, CANONICAL_18_FEATURE_NAMES)
        tr_stats["dataset"] = name
        tr_stats["split"] = "train"
        stats_list.append(tr_stats)

        # 2. Mutual Information (Training split only)
        mi_df = compute_mutual_information(df_tr, CANONICAL_18_FEATURE_NAMES, label_col="label")
        mi_df["dataset"] = name
        mi_list.append(mi_df)

    if dataset_filter is not None:
        stats_list[0].to_csv(
            reports_dir / f"{dataset_filter}_feature_statistics.csv", index=False
        )
        mi_list[0].to_csv(
            reports_dir / f"{dataset_filter}_mutual_information.csv", index=False
        )
        print(
            f"Generated per-dataset feature statistics and mutual-information "
            f"reports for {dataset_filter} in {reports_dir}"
        )
        return

    combined_stats = pd.concat(stats_list, ignore_index=True)
    combined_stats.to_csv(reports_dir / "stage3_feature_statistics.csv", index=False)

    combined_mi = pd.concat(mi_list, ignore_index=True)
    combined_mi.to_csv(reports_dir / "stage3_mutual_information.csv", index=False)

    # 3. Cross-Domain Distribution Shift (KS test & Wasserstein distance)
    names = list(dataset_dfs.keys())
    shift_list = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s_name, t_name = names[i], names[j]
            s_df, t_df = dataset_dfs[s_name], dataset_dfs[t_name]
            shift_df = compute_distribution_shift(s_df, t_df, CANONICAL_18_FEATURE_NAMES)
            shift_df["source_dataset"] = s_name
            shift_df["target_dataset"] = t_name
            shift_list.append(shift_df)

    combined_shift = pd.concat(shift_list, ignore_index=True)
    combined_shift.to_csv(reports_dir / "stage3_distribution_shift.csv", index=False)

    # 4. Correlation & VIF on reference dataset (ToN-IoT)
    ref_df = dataset_dfs["ToN-IoT"]
    pearson_df, spearman_df = compute_correlation(ref_df, CANONICAL_18_FEATURE_NAMES)
    pearson_df.to_csv(reports_dir / "stage3_correlation.csv")
    spearman_df.to_csv(reports_dir / "stage3_spearman_correlation.csv")

    vif_df = compute_vif(ref_df, CANONICAL_18_FEATURE_NAMES)
    vif_df.to_csv(reports_dir / "stage3_vif.csv", index=False)

    print("\n==========================================================================")
    print("=== STAGE 3 STATISTICAL AUDIT COMPLETED ===")
    print("==========================================================================")
    print(f"Generated Audit Reports in {reports_dir}:")
    print("  - stage3_feature_statistics.csv")
    print("  - stage3_distribution_shift.csv")
    print("  - stage3_mutual_information.csv")
    print("  - stage3_correlation.csv")
    print("  - stage3_spearman_correlation.csv")
    print("  - stage3_vif.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("ToN-IoT", "Edge-IIoTset", "NF-ToN-IoT-v2", "CICIoT2023", "BoT-IoT"),
        help="Materialize and audit only this dataset; do not overwrite aggregate audits.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10000,
        help="Maximum uniformly sampled flows (default: 10000).",
    )
    arguments = parser.parse_args()
    run_stage3_audit(
        sample_limit=arguments.sample_limit,
        dataset_filter=arguments.dataset,
    )
