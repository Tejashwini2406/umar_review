"""Stage 5 Research Visualization Generator.

Generates 8 publication-quality figures detailing domain adaptation gains, label-budget curves,
and ROC-AUC vs FPR trade-offs saved in reports/stage5/plots/.
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
    reports_dir = Path("reports/stage5")
    plots_dir = reports_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_exp = pd.read_csv(reports_dir / "stage5_experiment_results.csv")

    print("--- Generating Stage 5 Research Visualizations ---")

    # 1. Zero-Shot vs Adapted ROC-AUC
    plt.figure(figsize=(8, 4.5))
    sns.barplot(
        data=df_exp, x="adaptation_method", y="roc_auc", hue="model_name", palette="viridis", errorbar="se"
    )
    plt.title("ROC-AUC Comparison Across Domain Adaptation Methods", fontsize=11, fontweight="bold")
    plt.xlabel("Adaptation Method", fontsize=10, fontweight="bold")
    plt.ylabel("Cross-Domain ROC-AUC", fontsize=10, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Model Baseline")
    plt.tight_layout()
    plt.savefig(plots_dir / "zero_shot_vs_adapted_auc.png", dpi=300)
    plt.close()

    # 2. Zero-Shot vs Adapted FPR
    plt.figure(figsize=(8, 4.5))
    sns.barplot(
        data=df_exp, x="adaptation_method", y="fpr", hue="model_name", palette="rocket_r", errorbar="se"
    )
    plt.title("False Positive Rate (FPR) Suppression by Adaptation Method", fontsize=11, fontweight="bold")
    plt.xlabel("Adaptation Method", fontsize=10, fontweight="bold")
    plt.ylabel("Cross-Domain False Positive Rate (FPR)", fontsize=10, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Model Baseline")
    plt.tight_layout()
    plt.savefig(plots_dir / "zero_shot_vs_adapted_fpr.png", dpi=300)
    plt.close()

    # 3. FPR Reduction by Adaptation Method vs Zero-Shot
    zs_fpr = df_exp[df_exp["adaptation_method"] == "zero_shot"]["fpr"].mean()
    method_fpr = df_exp.groupby("adaptation_method")["fpr"].mean().reset_index()
    method_fpr["fpr_reduction_pct"] = ((zs_fpr - method_fpr["fpr"]) / (zs_fpr + 1e-6)) * 100.0

    plt.figure(figsize=(7.5, 4.5))
    sns.barplot(
        data=method_fpr, x="adaptation_method", y="fpr_reduction_pct", palette="magma"
    )
    plt.title("Relative False-Positive Reduction (%) vs Zero-Shot Control", fontsize=11, fontweight="bold")
    plt.xlabel("Adaptation Method", fontsize=10, fontweight="bold")
    plt.ylabel("Relative FPR Reduction (%)", fontsize=10, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "fpr_reduction_by_method.png", dpi=300)
    plt.close()

    # 4. Performance vs Target Label Budget (0%, 1%, 5%, 10%)
    budget_methods = ["zero_shot", "limited_label_1pct", "limited_label_5pct", "limited_label_10pct"]
    df_budget = df_exp[df_exp["adaptation_method"].isin(budget_methods)].copy()
    budget_map = {"zero_shot": 0, "limited_label_1pct": 1, "limited_label_5pct": 5, "limited_label_10pct": 10}
    df_budget["label_budget_pct"] = df_budget["adaptation_method"].map(budget_map)

    plt.figure(figsize=(8, 4.5))
    sns.lineplot(
        data=df_budget, x="label_budget_pct", y="macro_f1", hue="feature_profile", style="model_name", markers=True, dashes=False
    )
    plt.title("Macro-F1 Performance scaling vs Target-Label Budget (%)", fontsize=11, fontweight="bold")
    plt.xlabel("Target Domain Labeled Adaptation Budget (%)", fontsize=10, fontweight="bold")
    plt.ylabel("Cross-Domain Macro-F1", fontsize=10, fontweight="bold")
    plt.legend(title="Profile / Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(plots_dir / "performance_vs_label_budget.png", dpi=300)
    plt.close()

    # 5. Profile Comparison Before/After Adaptation (zero_shot vs unsupervised_alignment vs limited_label_5pct)
    df_p_sub = df_exp[df_exp["adaptation_method"].isin(["zero_shot", "unsupervised_alignment", "limited_label_5pct"])].copy()
    plt.figure(figsize=(8.5, 4.5))
    sns.barplot(
        data=df_p_sub, x="feature_profile", y="macro_f1", hue="adaptation_method", palette="deep", errorbar="se"
    )
    plt.title("Feature Profile Macro-F1 Before and After Adaptation", fontsize=11, fontweight="bold")
    plt.xlabel("Feature Representation Profile", fontsize=10, fontweight="bold")
    plt.ylabel("Macro-F1 Score", fontsize=10, fontweight="bold")
    plt.xticks(rotation=15)
    plt.legend(title="Adaptation Regime")
    plt.tight_layout()
    plt.savefig(plots_dir / "profile_comparison_before_after_adaptation.png", dpi=300)
    plt.close()

    # 6. Source->Target Transfer Heatmap (Unsupervised Alignment)
    df_align = df_exp[df_exp["adaptation_method"] == "unsupervised_alignment"]
    f1_pivot = df_align.pivot_table(index="source_domain", columns="target_domain", values="macro_f1", aggfunc="mean")
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(f1_pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True)
    plt.title("Adapted Cross-Domain Macro-F1 Matrix (Unsupervised Alignment)", fontsize=11, fontweight="bold")
    plt.xlabel("Target Domain (Test)", fontsize=10, fontweight="bold")
    plt.ylabel("Source Domain (Train)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "transfer_heatmap_adapted.png", dpi=300)
    plt.close()

    # 7. Adaptation Method Comparison Across Transfer Directions
    plt.figure(figsize=(9, 4.5))
    sns.boxplot(
        data=df_exp, x="adaptation_method", y="macro_f1", palette="Set3"
    )
    plt.title("Macro-F1 Distribution Across 12 Transfer Directions per Method", fontsize=11, fontweight="bold")
    plt.xlabel("Adaptation Method", fontsize=10, fontweight="bold")
    plt.ylabel("Macro-F1 Score", fontsize=10, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "adaptation_method_comparison_directions.png", dpi=300)
    plt.close()

    # 8. Trade-Off Between ROC-AUC and FPR
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df_exp, x="fpr", y="roc_auc", hue="adaptation_method", style="model_name", s=80, alpha=0.8, palette="tab10"
    )
    plt.title("Trade-off Between ROC-AUC and False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    plt.xlabel("False Positive Rate (FPR)", fontsize=10, fontweight="bold")
    plt.ylabel("ROC-AUC", fontsize=10, fontweight="bold")
    plt.legend(title="Method / Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(plots_dir / "roc_auc_vs_fpr_tradeoff.png", dpi=300)
    plt.close()

    print(f"Generated 8 Publication-Grade PNG Plots in {plots_dir}:")
    print("  - zero_shot_vs_adapted_auc.png")
    print("  - zero_shot_vs_adapted_fpr.png")
    print("  - fpr_reduction_by_method.png")
    print("  - performance_vs_label_budget.png")
    print("  - profile_comparison_before_after_adaptation.png")
    print("  - transfer_heatmap_adapted.png")
    print("  - adaptation_method_comparison_directions.png")
    print("  - roc_auc_vs_fpr_tradeoff.png")


if __name__ == "__main__":
    generate_plots()
