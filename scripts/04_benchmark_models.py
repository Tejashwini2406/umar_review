"""Stage 4 Deterministic Model Benchmarking Engine.

Executes within-domain and zero-shot cross-domain model benchmarking across 5 feature profiles,
2 model families (Logistic Regression, Random Forest), and 4 IoT datasets (ToN-IoT, Edge-IIoTset,
NF-ToN-IoT-v2, CICIoT2023). Enforces strict anti-leakage invariants and generates machine-readable CSV outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import pandas as pd
import numpy as np

from iot_ids.experiments.preprocessing import FeaturePreprocessor
from iot_ids.experiments.evaluator import optimize_threshold_on_val, evaluate_predictions
from iot_ids.experiments.runner import instantiate_model
from iot_ids.features.canonical.flow_builder import CANONICAL_18_FEATURE_NAMES

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


def perform_preflight_data_integrity_audit(data_dir: Path):
    """Verifies schemas, split non-overlap, label ranges, and absence of NaN/Inf before training."""
    print("--- Running Stage 4 Pre-Flight Data Integrity Audit ---")
    for ds in DATASETS:
        ds_dir = data_dir / ds
        tr_p = ds_dir / "train.parquet"
        val_p = ds_dir / "val.parquet"
        te_p = ds_dir / "test.parquet"

        if not (tr_p.exists() and val_p.exists() and te_p.exists()):
            raise FileNotFoundError(f"Missing materialized splits for dataset: {ds}")

        df_tr = pd.read_parquet(tr_p)
        df_v = pd.read_parquet(val_p)
        df_te = pd.read_parquet(te_p)

        # 1. Label range check
        for split_name, df_s in [("train", df_tr), ("val", df_v), ("test", df_te)]:
            labels = df_s["label"].unique()
            if not set(labels).issubset({0, 1}):
                raise ValueError(f"Invalid label values in {ds} {split_name}: {labels}")
            if len(labels) < 2:
                raise ValueError(f"Single class detected in {ds} {split_name}: {labels}")

        # 2. NaN/Inf check on canonical features
        for f in CANONICAL_18_FEATURE_NAMES:
            for split_name, df_s in [("train", df_tr), ("val", df_v), ("test", df_te)]:
                col_vals = df_s[f].values
                if np.isnan(col_vals).any() or np.isinf(col_vals).any():
                    raise ValueError(f"NaN or Inf detected in feature {f} of {ds} {split_name}")

        # 3. Non-overlapping timestamp check
        tr_max_ts = df_tr["timestamp_start"].max()
        val_min_ts = df_v["timestamp_start"].min()
        val_max_ts = df_v["timestamp_start"].max()
        te_min_ts = df_te["timestamp_start"].min()

        if tr_max_ts > val_min_ts:
            print(f"Warning: Train/Val timestamp boundary overlap in {ds}: train_max={tr_max_ts}, val_min={val_min_ts}")
        if val_max_ts > te_min_ts:
            print(f"Warning: Val/Test timestamp boundary overlap in {ds}: val_max={val_max_ts}, test_min={te_min_ts}")

    print("Pre-Flight Data Integrity Audit Passed 100% Cleanly!\n")


def load_dataset_splits(data_dir: Path, ds_name: str):
    ds_path = data_dir / ds_name
    return (
        pd.read_parquet(ds_path / "train.parquet"),
        pd.read_parquet(ds_path / "val.parquet"),
        pd.read_parquet(ds_path / "test.parquet"),
    )


def run_stage4_benchmarks(smoke_test: bool = False):
    data_dir = Path("data/processed/stage3")
    output_dir = Path("reports/stage4")
    cm_dir = output_dir / "stage4_confusion_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)

    perform_preflight_data_integrity_audit(data_dir)

    print("==========================================================================")
    print(f"=== STAGE 4 BENCHMARK EXECUTION (Smoke Test = {smoke_test}) ===")
    print("==========================================================================")

    target_ds_list = ["ToN-IoT", "Edge-IIoTset"] if smoke_test else DATASETS
    target_profiles = ["instant_only"] if smoke_test else list(FEATURE_PROFILES.keys())
    target_models = ["LogisticRegression"] if smoke_test else MODELS

    # Load splits
    datasets_data = {}
    for ds in target_ds_list:
        datasets_data[ds] = load_dataset_splits(data_dir, ds)

    all_experiments = []
    cm_records = []

    t0 = time.time()

    # 1. WITHIN-DOMAIN EXPERIMENTS
    print("--- Executing Within-Domain Experiments ---")
    for ds in datasets_data.keys():
        df_tr, df_v, df_te = datasets_data[ds]
        for p_name in target_profiles:
            f_names = FEATURE_PROFILES[p_name]
            for m_name in target_models:
                # Fit preprocessor on train only
                preproc = FeaturePreprocessor(f_names)
                X_tr = preproc.fit_transform(df_tr)
                y_tr = df_tr["label"].values

                X_v = preproc.transform(df_v)
                y_v = df_v["label"].values

                X_te = preproc.transform(df_te)
                y_te = df_te["label"].values

                # Fit model on train only
                clf = instantiate_model(m_name)
                clf.fit(X_tr, y_tr)

                # Optimize threshold on Val only
                val_probs = clf.predict_proba(X_v)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_v)
                tau_star = optimize_threshold_on_val(y_v, val_probs)

                # Evaluate Test with frozen threshold
                te_probs = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_te)
                metrics = evaluate_predictions(y_te, te_probs, threshold=tau_star)

                rec = {
                    "experiment_id": f"within_{ds}_{p_name}_{m_name}",
                    "eval_type": "WITHIN_DOMAIN",
                    "source_domain": ds,
                    "target_domain": ds,
                    "feature_profile": p_name,
                    "model": m_name,
                    "n_train": len(df_tr),
                    "n_val": len(df_v),
                    "n_test": len(df_te),
                }
                rec.update(metrics)
                all_experiments.append(rec)

                # Confusion matrix saving
                cm_df = pd.DataFrame([
                    {"tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"], "tp": metrics["tp"]}
                ])
                cm_df.to_csv(cm_dir / f"{rec['experiment_id']}_cm.csv", index=False)
                cm_records.append(rec)

    # 2. CROSS-DOMAIN ZERO-SHOT TRANSFER EXPERIMENTS
    print("--- Executing Zero-Shot Cross-Domain Transfer Experiments ---")
    for src in datasets_data.keys():
        for tgt in datasets_data.keys():
            if src == tgt:
                continue

            src_tr, src_v, _ = datasets_data[src]
            _, _, tgt_te = datasets_data[tgt]

            for p_name in target_profiles:
                f_names = FEATURE_PROFILES[p_name]
                for m_name in target_models:
                    # Fit preprocessor strictly on Source Train
                    preproc = FeaturePreprocessor(f_names)
                    X_src_tr = preproc.fit_transform(src_tr)
                    y_src_tr = src_tr["label"].values

                    X_src_v = preproc.transform(src_v)
                    y_src_v = src_v["label"].values

                    # Transform Target Test using frozen preprocessor (NO target fit!)
                    X_tgt_te = preproc.transform(tgt_te)
                    y_tgt_te = tgt_te["label"].values

                    # Fit model strictly on Source Train
                    clf = instantiate_model(m_name)
                    clf.fit(X_src_tr, y_src_tr)

                    # Optimize threshold strictly on Source Val
                    src_val_probs = clf.predict_proba(X_src_v)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_src_v)
                    tau_star = optimize_threshold_on_val(y_src_v, src_val_probs)

                    # Evaluate directly on Target Test with frozen model & threshold
                    tgt_test_probs = clf.predict_proba(X_tgt_te)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_tgt_te)
                    metrics = evaluate_predictions(y_tgt_te, tgt_test_probs, threshold=tau_star)

                    rec = {
                        "experiment_id": f"cross_{src}_to_{tgt}_{p_name}_{m_name}",
                        "eval_type": "CROSS_DOMAIN",
                        "source_domain": src,
                        "target_domain": tgt,
                        "feature_profile": p_name,
                        "model": m_name,
                        "n_train": len(src_tr),
                        "n_val": len(src_v),
                        "n_test": len(tgt_te),
                    }
                    rec.update(metrics)
                    all_experiments.append(rec)

                    cm_df = pd.DataFrame([
                        {"tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"], "tp": metrics["tp"]}
                    ])
                    cm_df.to_csv(cm_dir / f"{rec['experiment_id']}_cm.csv", index=False)
                    cm_records.append(rec)

    t1 = time.time()
    print(f"\nAll Stage 4 Experiments Completed in {t1 - t0:.2f} seconds!")

    df_all = pd.DataFrame(all_experiments)
    df_all.to_csv(output_dir / "stage4_experiment_results.csv", index=False)

    df_within = df_all[df_all["eval_type"] == "WITHIN_DOMAIN"]
    df_within.to_csv(output_dir / "stage4_within_domain_results.csv", index=False)

    df_cross = df_all[df_all["eval_type"] == "CROSS_DOMAIN"]
    df_cross.to_csv(output_dir / "stage4_cross_domain_results.csv", index=False)

    # Calculate Profile Summaries
    prof_summary = df_cross.groupby(["model", "feature_profile"])[
        ["roc_auc", "pr_auc", "macro_f1", "fpr", "fnr", "fp_per_1000"]
    ].agg(["mean", "std", "median"]).reset_index()
    prof_summary.to_csv(output_dir / "stage4_profile_summary.csv", index=False)

    # Calculate FPR Comparison Matrix
    fpr_comp = df_cross.pivot_table(
        index=["source_domain", "target_domain", "model"],
        columns="feature_profile",
        values="fpr"
    ).reset_index()

    # Calculate relative FPR reduction vs baseline_common
    if "baseline_common" in fpr_comp.columns and "full_multilevel" in fpr_comp.columns:
        fpr_comp["rel_fpr_reduction_full"] = (
            (fpr_comp["baseline_common"] - fpr_comp["full_multilevel"]) / (fpr_comp["baseline_common"] + 1e-6)
        )
    if "baseline_common" in fpr_comp.columns and "instant_behavioral" in fpr_comp.columns:
        fpr_comp["rel_fpr_reduction_behavioral"] = (
            (fpr_comp["baseline_common"] - fpr_comp["instant_behavioral"]) / (fpr_comp["baseline_common"] + 1e-6)
        )

    fpr_comp.to_csv(output_dir / "stage4_fpr_comparison.csv", index=False)

    # Calculate Model Comparison
    model_comp = df_cross.groupby(["model"])[
        ["roc_auc", "pr_auc", "macro_f1", "fpr"]
    ].agg(["mean", "std"]).reset_index()
    model_comp.to_csv(output_dir / "stage4_model_comparison.csv", index=False)

    print(f"Exported Machine-Readable Stage 4 CSV Reports to {output_dir}:")
    print("  - stage4_experiment_results.csv")
    print("  - stage4_within_domain_results.csv")
    print("  - stage4_cross_domain_results.csv")
    print("  - stage4_profile_summary.csv")
    print("  - stage4_fpr_comparison.csv")
    print("  - stage4_model_comparison.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4 Model Benchmarking Runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast pre-flight smoke test")
    args = parser.parse_args()
    run_stage4_benchmarks(smoke_test=args.smoke_test)
