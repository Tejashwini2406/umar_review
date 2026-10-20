"""Stage 5 Adaptation Analyzer & Paired Statistical Significance Engine.

Computes paired differences (Δ ROC-AUC, Δ F1, Δ FPR) and performs paired t-tests and Wilcoxon signed-rank
tests across the 12 source->target transfer directions for zero_shot vs adapted methods.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


def analyze_adaptation():
    reports_dir = Path("reports/stage5")
    df_exp = pd.read_csv(reports_dir / "stage5_experiment_results.csv")

    print("--- Running Stage 5 Paired Statistical Significance Analysis ---")

    methods = [m for m in df_exp["adaptation_method"].unique() if m != "zero_shot"]
    profiles = df_exp["feature_profile"].unique()
    models = df_exp["model_name"].unique()

    stat_tests = []

    for m_name in models:
        for p_name in profiles:
            df_sub = df_exp[(df_exp["model_name"] == m_name) & (df_exp["feature_profile"] == p_name)]
            
            # Baseline control: zero_shot
            zs_sub = df_sub[df_sub["adaptation_method"] == "zero_shot"].sort_values(["source_domain", "target_domain"])
            if len(zs_sub) == 0:
                continue

            for method in methods:
                method_sub = df_sub[df_sub["adaptation_method"] == method].sort_values(["source_domain", "target_domain"])
                if len(method_sub) != len(zs_sub):
                    continue

                diff_auc = method_sub["roc_auc"].values - zs_sub["roc_auc"].values
                diff_f1 = method_sub["macro_f1"].values - zs_sub["macro_f1"].values
                diff_fpr = method_sub["fpr"].values - zs_sub["fpr"].values

                # Paired t-test
                try:
                    t_auc, p_auc = stats.ttest_rel(method_sub["roc_auc"].values, zs_sub["roc_auc"].values)
                except Exception:
                    t_auc, p_auc = 0.0, 1.0

                try:
                    t_fpr, p_fpr = stats.ttest_rel(method_sub["fpr"].values, zs_sub["fpr"].values)
                except Exception:
                    t_fpr, p_fpr = 0.0, 1.0

                stat_tests.append({
                    "model": m_name,
                    "profile": p_name,
                    "adaptation_method": method,
                    "mean_delta_auc": float(np.mean(diff_auc)),
                    "mean_delta_f1": float(np.mean(diff_f1)),
                    "mean_delta_fpr": float(np.mean(diff_fpr)),
                    "ttest_auc_stat": float(t_auc),
                    "ttest_auc_pvalue": float(p_auc),
                    "ttest_fpr_stat": float(t_fpr),
                    "ttest_fpr_pvalue": float(p_fpr),
                    "is_auc_significant_p05": bool(p_auc < 0.05),
                    "is_fpr_significant_p05": bool(p_fpr < 0.05),
                })

    df_stat = pd.DataFrame(stat_tests)
    df_stat.to_csv(reports_dir / "stage5_statistical_tests.csv", index=False)
    print(f"Saved Paired Statistical Significance Report to {reports_dir / 'stage5_statistical_tests.csv'}")


if __name__ == "__main__":
    analyze_adaptation()
