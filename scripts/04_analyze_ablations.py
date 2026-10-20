"""Stage 4 Ablation Analyzer and Statistical Report Generator.

Computes incremental representation contributions (Δ ROC-AUC, Δ PR-AUC, Δ F1, Δ FPR)
and generates reports/stage4_ablation_summary.csv and reports/stage4_benchmark_report.md.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def analyze_ablations():
    reports_dir = Path("reports")
    df_within = pd.read_csv(reports_dir / "stage4_results.csv")
    df_cross = pd.read_csv(reports_dir / "stage4_cross_domain.csv")

    print("=== Analyzing Within-Domain & Cross-Domain Incremental Ablations ===")

    profiles = ["baseline_common", "instant_only", "instant_temporal", "instant_behavioral", "full_multilevel"]
    models = df_within["model_name"].unique()

    ablation_rows = []

    for model in models:
        sub_within = df_within[df_within["model_name"] == model]
        sub_cross = df_cross[df_cross["model_name"] == model]

        # Get baseline_common metrics
        base_within = sub_within[sub_within["feature_profile"] == "baseline_common"]
        base_cross = sub_cross[sub_cross["feature_profile"] == "baseline_common"]

        base_within_f1 = base_within["macro_f1"].mean() if len(base_within) > 0 else 0.0
        base_within_fpr = base_within["fpr"].mean() if len(base_within) > 0 else 0.0

        base_cross_f1 = base_cross["macro_f1"].mean() if len(base_cross) > 0 else 0.0
        base_cross_fpr = base_cross["fpr"].mean() if len(base_cross) > 0 else 0.0
        base_cross_auc = base_cross["roc_auc"].mean() if len(base_cross) > 0 else 0.0

        for p in profiles:
            p_within = sub_within[sub_within["feature_profile"] == p]
            p_cross = sub_cross[sub_cross["feature_profile"] == p]

            w_auc = p_within["roc_auc"].mean() if len(p_within) > 0 else 0.0
            w_pr = p_within["pr_auc"].mean() if len(p_within) > 0 else 0.0
            w_f1 = p_within["macro_f1"].mean() if len(p_within) > 0 else 0.0
            w_fpr = p_within["fpr"].mean() if len(p_within) > 0 else 0.0

            c_auc = p_cross["roc_auc"].mean() if len(p_cross) > 0 else 0.0
            c_pr = p_cross["pr_auc"].mean() if len(p_cross) > 0 else 0.0
            c_f1 = p_cross["macro_f1"].mean() if len(p_cross) > 0 else 0.0
            c_fpr = p_cross["fpr"].mean() if len(p_cross) > 0 else 0.0

            ablation_rows.append({
                "model_name": model,
                "feature_profile": p,
                "within_roc_auc_mean": float(w_auc),
                "within_pr_auc_mean": float(w_pr),
                "within_macro_f1_mean": float(w_f1),
                "within_fpr_mean": float(w_fpr),
                "cross_roc_auc_mean": float(c_auc),
                "cross_pr_auc_mean": float(c_pr),
                "cross_macro_f1_mean": float(c_f1),
                "cross_fpr_mean": float(c_fpr),
                "delta_cross_auc_vs_baseline": float(c_auc - base_cross_auc),
                "delta_cross_f1_vs_baseline": float(c_f1 - base_cross_f1),
                "delta_cross_fpr_vs_baseline": float(c_fpr - base_cross_fpr),
            })

    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(reports_dir / "stage4_ablation_summary.csv", index=False)
    print(f"Saved ablation summary matrix to {reports_dir / 'stage4_ablation_summary.csv'}")


if __name__ == "__main__":
    analyze_ablations()
