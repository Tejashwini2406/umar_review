"""Stage 6 Publication Visualization Generator.

Generates 7 publication-quality figures detailing paired effect sizes, 95% bootstrap confidence intervals,
direction-level paired improvements, shortcut feature ablations, and feature-shift diagnostic correlations
saved in reports/stage6/plots/.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 10, "figure.autolayout": True})


def generate_plots():
    reports_dir = Path("reports/stage6")
    plots_dir = reports_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_effect = pd.read_csv(reports_dir / "stage6_effect_sizes.csv")
    df_ci = pd.read_csv(reports_dir / "stage6_confidence_intervals.csv")
    df_shortcut = pd.read_csv(reports_dir / "stage6_shortcut_ablation.csv")
    df_shift = pd.read_csv(reports_dir / "stage6_feature_shift_relationships.csv")

    print("--- Generating Stage 6 Research Visualizations ---")

    # 1. Effect-Size Forest Plot (Cohen's dz across major comparisons)
    df_eff_auc = df_effect[df_effect["metric"] == "roc_auc"].copy()
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=df_eff_auc, x="cohens_dz", y="comparison", hue="model", palette="mako"
    )
    plt.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.title("Paired Effect Sizes (Cohen's dz) Across Transfer Comparisons (ROC-AUC)", fontsize=11, fontweight="bold")
    plt.xlabel("Cohen's dz (Paired Effect Size)", fontsize=10, fontweight="bold")
    plt.ylabel("Comparison", fontsize=10, fontweight="bold")
    plt.legend(title="Model Baseline")
    plt.tight_layout()
    plt.savefig(plots_dir / "effect_size_forest_plot.png", dpi=300)
    plt.close()

    # 2. 95% Bootstrap Confidence Intervals Plot for ROC-AUC
    df_ci_auc = df_ci[df_ci["metric"] == "roc_auc"].copy()
    plt.figure(figsize=(9, 5))
    sns.pointplot(
        data=df_ci_auc, x="adaptation_method", y="mean", hue="profile",
        errorbar=None, markers=["o", "s", "^", "D", "X"], capsize=0.15, palette="tab10"
    )
    plt.title("Mean Cross-Domain ROC-AUC with 95% Bootstrap Confidence Intervals (B=10,000)", fontsize=11, fontweight="bold")
    plt.xlabel("Adaptation Method", fontsize=10, fontweight="bold")
    plt.ylabel("Cross-Domain ROC-AUC (95% CI)", fontsize=10, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Profile / Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(plots_dir / "confidence_intervals_roc_auc.png", dpi=300)
    plt.close()

    # 3. Direction-Level Paired Improvement Plot (full_multilevel vs baseline_common)
    df_dir_sub = df_shortcut[
        (df_shortcut["adaptation_method"] == "limited_label_5pct") &
        (df_shortcut["model_name"] == "RandomForest")
    ].copy()
    df_dir_sub["transfer_direction"] = df_dir_sub["source_domain"] + " -> " + df_dir_sub["target_domain"]

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=df_dir_sub, x="transfer_direction", y="roc_auc", hue="feature_profile", palette="viridis"
    )
    plt.title("Per-Transfer-Direction ROC-AUC (5% Target Budget Adaptation)", fontsize=11, fontweight="bold")
    plt.xlabel("Source -> Target Transfer Direction", fontsize=10, fontweight="bold")
    plt.ylabel("ROC-AUC Score", fontsize=10, fontweight="bold")
    plt.xticks(rotation=35, ha="right")
    plt.legend(title="Profile")
    plt.tight_layout()
    plt.savefig(plots_dir / "direction_paired_improvement.png", dpi=300)
    plt.close()

    # 4. Label Budget Performance Curve
    df_budget_ci = df_ci[df_ci["profile"] == "full_multilevel"].copy()
    plt.figure(figsize=(8, 4.5))
    sns.lineplot(
        data=df_budget_ci, x="adaptation_method", y="mean", hue="metric", style="model", markers=True, dashes=False, palette="Set1"
    )
    plt.title("full_multilevel Trajectory Across Adaptation Regimes (with 95% CIs)", fontsize=11, fontweight="bold")
    plt.xlabel("Adaptation Method / Budget", fontsize=10, fontweight="bold")
    plt.ylabel("Metric Score", fontsize=10, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Metric / Model")
    plt.tight_layout()
    plt.savefig(plots_dir / "label_budget_performance_curve.png", dpi=300)
    plt.close()

    # 5. Model Family Comparison (LogisticRegression vs RandomForest)
    df_ci_m = df_ci[df_ci["metric"] == "roc_auc"].copy()
    plt.figure(figsize=(8, 4.5))
    sns.barplot(
        data=df_ci_m, x="profile", y="mean", hue="model", palette="Set2"
    )
    plt.title("Model-Family Cross-Domain ROC-AUC Comparison", fontsize=11, fontweight="bold")
    plt.xlabel("Feature Representation Profile", fontsize=10, fontweight="bold")
    plt.ylabel("Mean ROC-AUC", fontsize=10, fontweight="bold")
    plt.xticks(rotation=15)
    plt.legend(title="Model Family")
    plt.tight_layout()
    plt.savefig(plots_dir / "model_family_comparison.png", dpi=300)
    plt.close()

    # 6. Shortcut-Ablation Comparison
    plt.figure(figsize=(8.5, 4.5))
    sns.barplot(
        data=df_shortcut, x="feature_profile", y="roc_auc", hue="adaptation_method", palette="plasma"
    )
    plt.title("Domain Shortcut Feature Ablation Performance (ROC-AUC)", fontsize=11, fontweight="bold")
    plt.xlabel("Feature Profile Ablation Configuration", fontsize=10, fontweight="bold")
    plt.ylabel("Cross-Domain ROC-AUC", fontsize=10, fontweight="bold")
    plt.xticks(rotation=15)
    plt.legend(title="Adaptation Method")
    plt.tight_layout()
    plt.savefig(plots_dir / "shortcut_ablation_comparison.png", dpi=300)
    plt.close()

    # 7. Feature Shift vs Predictive Information Diagnostic Scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df_shift, x="ks_stat_mean", y="mutual_info_mean", hue="is_stateful", style="is_stateful", s=100, palette="Set1"
    )
    for idx, row in df_shift.iterrows():
        plt.annotate(row["feature"], (row["ks_stat_mean"] + 0.01, row["mutual_info_mean"] + 0.005), fontsize=8)
    plt.title("Feature Distribution Shift (KS Stat) vs Target Information (MI)", fontsize=11, fontweight="bold")
    plt.xlabel("Mean Kolmogorov-Smirnov Shift (KS Distance)", fontsize=10, fontweight="bold")
    plt.ylabel("Source Mutual Information I(X; Y)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_shift_vs_predictive_info.png", dpi=300)
    plt.close()

    print(f"Generated 7 Publication-Grade Stage 6 PNG Plots in {plots_dir}:")
    print("  - effect_size_forest_plot.png")
    print("  - confidence_intervals_roc_auc.png")
    print("  - direction_paired_improvement.png")
    print("  - label_budget_performance_curve.png")
    print("  - model_family_comparison.png")
    print("  - shortcut_ablation_comparison.png")
    print("  - feature_shift_vs_predictive_info.png")


if __name__ == "__main__":
    generate_plots()
