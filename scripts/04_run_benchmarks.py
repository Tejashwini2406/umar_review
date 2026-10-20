"""Stage 4 Experimental Runner: Within-Domain & Cross-Domain Multi-Level Ablation Benchmark.

Executes 40 within-domain and 120 zero-shot cross-domain transfer experiments across 5 feature profiles
and 2 baseline models (Logistic Regression, Random Forest), optimizing thresholds on source validation
and evaluating frozen models directly on target test splits.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import pandas as pd

from iot_ids.experiments.runner import run_within_domain_experiment, run_cross_domain_experiment
from iot_ids.features.canonical.flow_builder import INSTANT_FEATURE_NAMES, TEMPORAL_FEATURE_NAMES, BEHAVIORAL_FEATURE_NAMES, CANONICAL_18_FEATURE_NAMES

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

DATASET_NAMES = ["ToN-IoT", "Edge-IIoTset", "NF-ToN-IoT-v2", "CICIoT2023"]
MODEL_NAMES = ["LogisticRegression", "RandomForest"]


def load_dataset_splits(data_dir: Path, dataset_name: str):
    ds_path = data_dir / dataset_name
    df_train = pd.read_parquet(ds_path / "train.parquet")
    df_val = pd.read_parquet(ds_path / "val.parquet")
    df_test = pd.read_parquet(ds_path / "test.parquet")
    return df_train, df_val, df_test


def run_benchmarks(smoke_test: bool = False):
    data_dir = Path("data/processed/stage3")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================================================")
    print(f"=== STAGE 4 EXPERIMENTAL RUNNER (Smoke Test = {smoke_test}) ===")
    print("==========================================================================")

    # 1. Load materialized dataset splits
    datasets_data = {}
    ds_to_load = ["ToN-IoT", "Edge-IIoTset"] if smoke_test else DATASET_NAMES
    for ds_name in ds_to_load:
        print(f"Loading materialized splits for: {ds_name}")
        datasets_data[ds_name] = load_dataset_splits(data_dir, ds_name)

    within_results = []
    cross_results = []
    confusion_matrices = []

    profiles_to_run = ["instant_only"] if smoke_test else list(FEATURE_PROFILES.keys())
    models_to_run = ["LogisticRegression"] if smoke_test else MODEL_NAMES

    t0 = time.time()

    # 2. Within-Domain Experiments
    print("\n--- Running Within-Domain Experiments ---")
    for ds_name in datasets_data.keys():
        df_tr, df_v, df_te = datasets_data[ds_name]
        for p_name in profiles_to_run:
            f_names = FEATURE_PROFILES[p_name]
            for m_name in models_to_run:
                exp_res = run_within_domain_experiment(
                    dataset_name=ds_name,
                    feature_profile=p_name,
                    feature_names=f_names,
                    model_name=m_name,
                    train_df=df_tr,
                    val_df=df_v,
                    test_df=df_te,
                )
                within_results.append(exp_res)
                confusion_matrices.append({
                    "experiment_type": "within_domain",
                    "source": ds_name,
                    "target": ds_name,
                    "profile": p_name,
                    "model": m_name,
                    "tn": exp_res["tn"],
                    "fp": exp_res["fp"],
                    "fn": exp_res["fn"],
                    "tp": exp_res["tp"],
                })

    # 3. Cross-Domain Zero-Shot Transfer Experiments
    print("\n--- Running Cross-Domain Zero-Shot Transfer Experiments ---")
    for src_name in datasets_data.keys():
        for tgt_name in datasets_data.keys():
            if src_name == tgt_name:
                continue

            src_tr, src_v, _ = datasets_data[src_name]
            _, _, tgt_te = datasets_data[tgt_name]

            for p_name in profiles_to_run:
                f_names = FEATURE_PROFILES[p_name]
                for m_name in models_to_run:
                    exp_res = run_cross_domain_experiment(
                        source_dataset=src_name,
                        target_dataset=tgt_name,
                        feature_profile=p_name,
                        feature_names=f_names,
                        model_name=m_name,
                        source_train_df=src_tr,
                        source_val_df=src_v,
                        target_test_df=tgt_te,
                    )
                    cross_results.append(exp_res)
                    confusion_matrices.append({
                        "experiment_type": "cross_domain",
                        "source": src_name,
                        "target": tgt_name,
                        "profile": p_name,
                        "model": m_name,
                        "tn": exp_res["tn"],
                        "fp": exp_res["fp"],
                        "fn": exp_res["fn"],
                        "tp": exp_res["tp"],
                    })

    t1 = time.time()
    print(f"\nCompleted Stage 4 Benchmarks in {t1 - t0:.2f} seconds!")

    df_within = pd.DataFrame(within_results)
    df_cross = pd.DataFrame(cross_results)
    df_cm = pd.DataFrame(confusion_matrices)

    df_within.to_csv(reports_dir / "stage4_results.csv", index=False)
    df_cross.to_csv(reports_dir / "stage4_cross_domain.csv", index=False)
    df_cm.to_csv(reports_dir / "stage4_confusion_matrices.csv", index=False)

    print(f"\nSaved benchmark outputs to {reports_dir}:")
    print("  - stage4_results.csv")
    print("  - stage4_cross_domain.csv")
    print("  - stage4_confusion_matrices.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4 Controlled Model Benchmarking Runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast pre-flight smoke test")
    args = parser.parse_args()
    run_benchmarks(smoke_test=args.smoke_test)
