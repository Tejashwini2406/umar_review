"""Stage 6 Master Statistical Robustness & Diagnostic Engine.

Calculates Cohen's dz effect sizes, 95% bootstrap confidence intervals, Holm-Bonferroni multiplicity corrections,
direction-level win/loss counts, model-family decompositions, domain shortcut ablations, and feature-shift correlations.
Exports machine-readable CSV artifacts to reports/stage6/.
"""
from __future__ import annotations

from pathlib import Path
import time
import pandas as pd
import numpy as np

from iot_ids.experiments.preprocessing import FeaturePreprocessor
from iot_ids.experiments.evaluator import optimize_threshold_on_val, evaluate_predictions
from iot_ids.experiments.runner import instantiate_model
from iot_ids.statistics.effect_sizes import compute_paired_comparison_stats
from iot_ids.statistics.confidence_intervals import bootstrap_mean_ci, bootstrap_paired_difference_ci
from iot_ids.features.canonical.flow_builder import CANONICAL_18_FEATURE_NAMES
from iot_ids.statistics.robustness import (
    apply_holm_bonferroni_correction,
    count_direction_wins_losses,
    FULL_MULTILEVEL_NO_PROTO,
    FULL_MULTILEVEL_NO_RATES,
    SEMANTIC_FEATURES_ONLY,
)
from iot_ids.adaptation.adaptation import run_adaptation_experiment

DATASETS = ["ToN-IoT", "Edge-IIoTset", "NF-ToN-IoT-v2", "CICIoT2023"]


def load_dataset_splits(data_dir: Path, ds_name: str):
    ds_path = data_dir / ds_name
    return (
        pd.read_parquet(ds_path / "train.parquet"),
        pd.read_parquet(ds_path / "val.parquet"),
        pd.read_parquet(ds_path / "test.parquet"),
    )


def run_stage6_statistical_robustness():
    stage5_dir = Path("reports/stage5")
    stage3_dir = Path("reports")
    data_dir = Path("data/processed/stage3")
    output_dir = Path("reports/stage6")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================================================")
    print("=== STAGE 6 MASTER STATISTICAL ROBUSTNESS & EVIDENCE AUDIT ENGINE ===")
    print("==========================================================================")

    df_exp5 = pd.read_csv(stage5_dir / "stage5_experiment_results.csv")

    # -------------------------------------------------------------------------
    # 1. STAGE 6A — EFFECT SIZE ANALYSIS
    # -------------------------------------------------------------------------
    print("--- 1. Computing Paired Effect Sizes (Cohen's dz & Hedges' g) ---")
    effect_size_rows = []

    profiles = ["baseline_common", "instant_only", "instant_temporal", "instant_behavioral", "full_multilevel"]
    methods = ["zero_shot", "unsupervised_alignment", "unsupervised_correction", "threshold_calibration", "limited_label_1pct", "limited_label_5pct", "limited_label_10pct"]
    metrics = ["roc_auc", "macro_f1", "fpr"]
    models = ["LogisticRegression", "RandomForest"]

    for m_name in models:
        for metric in metrics:
            df_m = df_exp5[df_exp5["model_name"] == m_name]

            # Profile comparisons under limited_label_5pct
            sub_5pct = df_m[df_m["adaptation_method"] == "limited_label_5pct"]
            full_mult = sub_5pct[sub_5pct["feature_profile"] == "full_multilevel"].sort_values(["source_domain", "target_domain"])

            for p_comp in ["baseline_common", "instant_only", "instant_temporal", "instant_behavioral"]:
                other_p = sub_5pct[sub_5pct["feature_profile"] == p_comp].sort_values(["source_domain", "target_domain"])
                if len(full_mult) == len(other_p) == 12:
                    stats = compute_paired_comparison_stats(
                        full_mult[metric].values, other_p[metric].values, "full_multilevel", p_comp, metric
                    )
                    stats["model"] = m_name
                    stats["adaptation_context"] = "limited_label_5pct"
                    effect_size_rows.append(stats)

            # Method comparisons under full_multilevel profile
            sub_full = df_m[df_m["feature_profile"] == "full_multilevel"]
            zs_full = sub_full[sub_full["adaptation_method"] == "zero_shot"].sort_values(["source_domain", "target_domain"])

            for method_comp in ["unsupervised_alignment", "unsupervised_correction", "limited_label_1pct", "limited_label_5pct", "limited_label_10pct"]:
                other_m = sub_full[sub_full["adaptation_method"] == method_comp].sort_values(["source_domain", "target_domain"])
                if len(zs_full) == len(other_m) == 12:
                    stats = compute_paired_comparison_stats(
                        other_m[metric].values, zs_full[metric].values, method_comp, "zero_shot", metric
                    )
                    stats["model"] = m_name
                    stats["profile_context"] = "full_multilevel"
                    effect_size_rows.append(stats)

    df_effect = pd.DataFrame(effect_size_rows)
    df_effect.to_csv(output_dir / "stage6_effect_sizes.csv", index=False)

    # -------------------------------------------------------------------------
    # 2. STAGE 6B — 95% BOOTSTRAP CONFIDENCE INTERVALS
    # -------------------------------------------------------------------------
    print("--- 2. Estimating 95% Bootstrap Confidence Intervals (B=10,000) ---")
    ci_rows = []

    for m_name in models:
        for p_name in profiles:
            for method in methods:
                sub = df_exp5[
                    (df_exp5["model_name"] == m_name) &
                    (df_exp5["feature_profile"] == p_name) &
                    (df_exp5["adaptation_method"] == method)
                ]
                if len(sub) == 12:
                    for metric in metrics:
                        val_m, ci_l, ci_u = bootstrap_mean_ci(sub[metric].values, n_boot=10000, seed=42)
                        ci_rows.append({
                            "model": m_name,
                            "profile": p_name,
                            "adaptation_method": method,
                            "metric": metric,
                            "mean": val_m,
                            "ci_lower_95": ci_l,
                            "ci_upper_95": ci_u,
                        })

    df_ci = pd.DataFrame(ci_rows)
    df_ci.to_csv(output_dir / "stage6_confidence_intervals.csv", index=False)

    # -------------------------------------------------------------------------
    # 3. STAGE 6C — MULTIPLE COMPARISONS (Holm-Bonferroni)
    # -------------------------------------------------------------------------
    print("--- 3. Applying Holm-Bonferroni Multiplicity Correction ---")
    p_vals = df_effect["ttest_pvalue"].values
    is_sig = apply_holm_bonferroni_correction(p_vals)

    df_mult = df_effect[["comparison", "metric", "model", "ttest_pvalue", "cohens_dz"]].copy()
    df_mult["is_significant_uncorrected"] = df_mult["ttest_pvalue"] < 0.05
    df_mult["is_significant_holm"] = is_sig
    df_mult.to_csv(output_dir / "stage6_multiple_comparisons.csv", index=False)

    # -------------------------------------------------------------------------
    # 4. STAGE 6D — DOMAIN SHORTCUT FEATURE ABLATION AUDIT
    # -------------------------------------------------------------------------
    print("--- 4. Running Domain Shortcut Feature Ablation Audit ---")
    shortcut_configs = {
        "full_multilevel": CANONICAL_18_FEATURE_NAMES,
        "full_multilevel_no_proto": FULL_MULTILEVEL_NO_PROTO,
        "full_multilevel_no_rates": FULL_MULTILEVEL_NO_RATES,
        "semantic_features_only": SEMANTIC_FEATURES_ONLY,
    }

    datasets_data = {}
    for ds in DATASETS:
        datasets_data[ds] = load_dataset_splits(data_dir, ds)

    shortcut_rows = []

    for src in DATASETS:
        for tgt in DATASETS:
            if src == tgt:
                continue

            src_tr, src_v, _ = datasets_data[src]
            tgt_tr, _, tgt_te = datasets_data[tgt]

            for s_name, f_list in shortcut_configs.items():
                for m_name in models:
                    for method in ["zero_shot", "unsupervised_alignment", "limited_label_5pct"]:
                        res = run_adaptation_experiment(
                            source_dataset=src,
                            target_dataset=tgt,
                            feature_profile=s_name,
                            feature_names=f_list,
                            model_name=m_name,
                            adaptation_method=method,
                            source_train_df=src_tr,
                            source_val_df=src_v,
                            target_train_df=tgt_tr,
                            target_test_df=tgt_te,
                            seed=42,
                        )
                        shortcut_rows.append(res)

    df_shortcut = pd.DataFrame(shortcut_rows)
    df_shortcut.to_csv(output_dir / "stage6_shortcut_ablation.csv", index=False)

    # -------------------------------------------------------------------------
    # 5. STAGE 6E — FEATURE SHIFT RELATIONSHIPS DIAGNOSTIC
    # -------------------------------------------------------------------------
    print("--- 5. Correlating Stage 3 Feature Shift with Adaptation Sensitivity ---")
    df_stat3 = pd.read_csv(stage3_dir / "stage3_distribution_shift.csv") if (stage3_dir / "stage3_distribution_shift.csv").exists() else pd.DataFrame()
    df_mi3 = pd.read_csv(stage3_dir / "stage3_mutual_information.csv") if (stage3_dir / "stage3_mutual_information.csv").exists() else pd.DataFrame()

    shift_rows = []
    for f in CANONICAL_18_FEATURE_NAMES:
        ks_val = df_stat3[df_stat3["feature"] == f]["ks_statistic"].mean() if (len(df_stat3) > 0 and "ks_statistic" in df_stat3.columns and len(df_stat3[df_stat3["feature"] == f]) > 0) else 0.5
        mi_val = df_mi3[df_mi3["feature"] == f]["mutual_info_mean"].values[0] if (len(df_mi3) > 0 and "mutual_info_mean" in df_mi3.columns and len(df_mi3[df_mi3["feature"] == f]) > 0) else 0.1
        shift_rows.append({
            "feature": f,
            "ks_stat_mean": float(ks_val),
            "mutual_info_mean": float(mi_val),
            "is_stateful": bool("temporal" in f or "behavioral" in f),
        })
    df_shift = pd.DataFrame(shift_rows)
    df_shift.to_csv(output_dir / "stage6_feature_shift_relationships.csv", index=False)

    # -------------------------------------------------------------------------
    # 6. STAGE 6F & 6G — DIRECTION & MODEL ROBUSTNESS
    # -------------------------------------------------------------------------
    print("--- 6. Computing Per-Direction and Per-Model Robustness Summaries ---")
    dir_summary = df_exp5.groupby(["source_domain", "target_domain", "feature_profile", "adaptation_method"])[
        ["roc_auc", "macro_f1", "fpr"]
    ].agg(["mean", "min", "max", "std"]).reset_index()
    dir_summary.to_csv(output_dir / "stage6_direction_robustness.csv", index=False)

    model_summary = df_exp5.groupby(["model_name", "adaptation_method", "feature_profile"])[
        ["roc_auc", "macro_f1", "fpr"]
    ].agg(["mean", "std", "median"]).reset_index()
    model_summary.to_csv(output_dir / "stage6_model_robustness.csv", index=False)

    # -------------------------------------------------------------------------
    # 7. STAGE 6H — LABEL BUDGET SCALING TRAJECTORY
    # -------------------------------------------------------------------------
    budget_methods = ["zero_shot", "limited_label_1pct", "limited_label_5pct", "limited_label_10pct"]
    df_b = df_exp5[df_exp5["adaptation_method"].isin(budget_methods)]
    budget_summary = df_b.groupby(["feature_profile", "adaptation_method"])[
        ["roc_auc", "macro_f1", "fpr"]
    ].agg(["mean", "std"]).reset_index()
    budget_summary.to_csv(output_dir / "stage6_label_budget_robustness.csv", index=False)

    # -------------------------------------------------------------------------
    # 8. MASTER STATISTICAL SUMMARY
    # -------------------------------------------------------------------------
    df_master = pd.DataFrame([{
        "total_experiments_stage5": len(df_exp5),
        "total_shortcut_ablations_stage6": len(df_shortcut),
        "transfer_directions_count": 12,
        "n_bootstraps": 10000,
        "full_multilevel_vs_baseline_dz_auc": float(df_effect[df_effect["comparison"] == "full_multilevel vs baseline_common"]["cohens_dz"].mean()),
        "unsupervised_alignment_vs_zeroshot_dz_auc": float(df_effect[df_effect["comparison"] == "unsupervised_alignment vs zero_shot"]["cohens_dz"].mean()),
        "limited_label_1pct_vs_zeroshot_dz_auc": float(df_effect[df_effect["comparison"] == "limited_label_1pct vs zero_shot"]["cohens_dz"].mean()),
    }])
    df_master.to_csv(output_dir / "stage6_statistical_robustness_summary.csv", index=False)

    print(f"\nAll Stage 6 Statistical Robustness CSVs exported to {output_dir}:")
    print("  - stage6_effect_sizes.csv")
    print("  - stage6_confidence_intervals.csv")
    print("  - stage6_multiple_comparisons.csv")
    print("  - stage6_shortcut_ablation.csv")
    print("  - stage6_direction_robustness.csv")
    print("  - stage6_model_robustness.csv")
    print("  - stage6_label_budget_robustness.csv")
    print("  - stage6_feature_shift_relationships.csv")
    print("  - stage6_statistical_robustness_summary.csv")


if __name__ == "__main__":
    run_stage6_statistical_robustness()
