"""Stage 5 Experimental Runner: Domain Adaptation & Target Calibration Benchmark.

Executes 840 controlled experiments across 12 source->target transfer directions, 5 feature profiles,
2 models, and 7 adaptation methods (zero_shot, unsupervised_alignment, unsupervised_correction,
threshold_calibration, limited_label_1pct, limited_label_5pct, limited_label_10pct).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import pandas as pd
import numpy as np

from iot_ids.experiments.preprocessing import FeaturePreprocessor
from iot_ids.experiments.runner import instantiate_model
from iot_ids.features.canonical.flow_builder import CANONICAL_18_FEATURE_NAMES
from iot_ids.adaptation.adaptation import run_adaptation_experiment, ADAPTATION_METHODS

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

DATASETS = ["ToN-IoT", "Edge-IIoTset", "NF-ToN-IoT-v2", "CICIoT2023"]
MODELS = ["LogisticRegression", "RandomForest"]


def load_dataset_splits(data_dir: Path, ds_name: str):
    ds_path = data_dir / ds_name
    return (
        pd.read_parquet(ds_path / "train.parquet"),
        pd.read_parquet(ds_path / "val.parquet"),
        pd.read_parquet(ds_path / "test.parquet"),
    )


def run_stage5_benchmarks(smoke_test: bool = False):
    data_dir = Path("data/processed/stage3")
    output_dir = Path("reports/stage5")
    cm_dir = output_dir / "stage4_confusion_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================================================")
    print(f"=== STAGE 5 DOMAIN ADAPTATION RUNNER (Smoke Test = {smoke_test}) ===")
    print("==========================================================================")

    target_ds_list = ["ToN-IoT", "Edge-IIoTset"] if smoke_test else DATASETS
    target_profiles = ["instant_only"] if smoke_test else list(FEATURE_PROFILES.keys())
    target_models = ["LogisticRegression"] if smoke_test else MODELS
    target_methods = ADAPTATION_METHODS if smoke_test else ADAPTATION_METHODS

    # Load splits
    datasets_data = {}
    for ds in target_ds_list:
        datasets_data[ds] = load_dataset_splits(data_dir, ds)

    all_experiments = []
    cm_records = []

    t0 = time.time()

    # 12 Source -> Target transfer directions
    for src in datasets_data.keys():
        for tgt in datasets_data.keys():
            if src == tgt:
                continue

            src_tr, src_v, _ = datasets_data[src]
            tgt_tr, _, tgt_te = datasets_data[tgt]

            for p_name in target_profiles:
                f_names = FEATURE_PROFILES[p_name]
                for m_name in target_models:
                    for method in target_methods:
                        exp_res = run_adaptation_experiment(
                            source_dataset=src,
                            target_dataset=tgt,
                            feature_profile=p_name,
                            feature_names=f_names,
                            model_name=m_name,
                            adaptation_method=method,
                            source_train_df=src_tr,
                            source_val_df=src_v,
                            target_train_df=tgt_tr,
                            target_test_df=tgt_te,
                            seed=42,
                        )
                        all_experiments.append(exp_res)
                        cm_records.append({
                            "experiment_id": exp_res["experiment_id"],
                            "source": src,
                            "target": tgt,
                            "profile": p_name,
                            "model": m_name,
                            "method": method,
                            "tn": exp_res["tn"],
                            "fp": exp_res["fp"],
                            "fn": exp_res["fn"],
                            "tp": exp_res["tp"],
                        })

    t1 = time.time()
    print(f"\nAll Stage 5 Adaptation Benchmarks Completed in {t1 - t0:.2f} seconds!")

    df_all = pd.DataFrame(all_experiments)
    df_all.to_csv(output_dir / "stage5_experiment_results.csv", index=False)

    # 1. Adaptation Summary per Method & Profile
    summary_df = df_all.groupby(["model_name", "feature_profile", "adaptation_method"])[
        ["roc_auc", "pr_auc", "macro_f1", "fpr", "fp_per_1000"]
    ].agg(["mean", "std", "median"]).reset_index()
    summary_df.to_csv(output_dir / "stage5_adaptation_summary.csv", index=False)

    # 2. Label Budget Results (0%, 1%, 5%, 10%)
    budget_methods = ["zero_shot", "limited_label_1pct", "limited_label_5pct", "limited_label_10pct"]
    budget_df = df_all[df_all["adaptation_method"].isin(budget_methods)]
    budget_df.to_csv(output_dir / "stage5_label_budget_results.csv", index=False)

    # 3. Direction Results (Individual 12 Source->Target Pairs)
    df_all.to_csv(output_dir / "stage5_direction_results.csv", index=False)

    # 4. FPR Comparison Matrix
    fpr_pivot = df_all.pivot_table(
        index=["source_domain", "target_domain", "model_name", "feature_profile"],
        columns="adaptation_method",
        values="fpr"
    ).reset_index()
    fpr_pivot.to_csv(output_dir / "stage5_fpr_comparison.csv", index=False)

    print(f"Exported Machine-Readable Stage 5 CSV Reports to {output_dir}:")
    print("  - stage5_experiment_results.csv")
    print("  - stage5_adaptation_summary.csv")
    print("  - stage5_label_budget_results.csv")
    print("  - stage5_direction_results.csv")
    print("  - stage5_fpr_comparison.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5 Domain Adaptation Benchmark Runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast pre-flight smoke test")
    args = parser.parse_args()
    run_stage5_benchmarks(smoke_test=args.smoke_test)
