"""Stage 4 Research Visualization Generator.

Generates publication-quality figures for cross-domain Macro F1 heatmaps, FPR heatmaps,
feature profile comparisons, and false-positive reduction dynamics.
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
    reports_dir = Path("reports/stage4")
    plots_dir = reports_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_cross = pd.read_csv(reports_dir / "stage4_cross_domain_results.csv")

    print("--- Generating Stage 4 Research Visualizations ---")

    # 1. Cross-Domain Macro F1 Heatmap (LogisticRegression & RandomForest averaged)
    f1_pivot = df_cross.pivot_table(
        index="source_domain", columns="target_domain", values="macro_f1", aggfunc="mean"
    )
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(f1_pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True, vmin=0.2, vmax=0.6)
    plt.title("Zero-Shot Cross-Domain Macro-F1 Matrix (Mean across Profiles & Models)", fontsize=11, fontweight="bold")
    plt.xlabel("Target Domain (Test)", fontsize=10, fontweight="bold")
    plt.ylabel("Source Domain (Train)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "cross_domain_f1_heatmap.png", dpi=300)
    plt.close()

    # 2. Cross-Domain FPR Heatmap
    fpr_pivot = df_cross.pivot_table(
        index="source_domain", columns="target_domain", values="fpr", aggfunc="mean"
    )
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(fpr_pivot, annot=True, fmt=".3f", cmap="OrRd", cbar=True, vmin=0.2, vmax=1.0)
    plt.title("Zero-Shot Cross-Domain False Positive Rate (FPR) Matrix", fontsize=11, fontweight="bold")
    plt.xlabel("Target Domain (Test)", fontsize=10, fontweight="bold")
    plt.ylabel("Source Domain (Train)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "cross_domain_fpr_heatmap.png", dpi=300)
    plt.close()

    # 3. Profile Macro F1 Comparison Bar Chart
    plt.figure(figsize=(8, 4.5))
    sns.barplot(
        data=df_cross, x="feature_profile", y="macro_f1", hue="model", palette="viridis", errorbar="se"
    )
    plt.title("Cross-Domain Macro-F1 Score by Feature Representation Profile", fontsize=11, fontweight="bold")
    plt.xlabel("Feature Representation Profile", fontsize=10, fontweight="bold")
    plt.ylabel("Cross-Domain Macro-F1", fontsize=10, fontweight="bold")
    plt.xticks(rotation=15)
    plt.legend(title="Model Baseline")
    plt.tight_layout()
    plt.savefig(plots_dir / "profile_f1_comparison.png", dpi=300)
    plt.close()

    # 4. Profile FPR Comparison (Baseline vs Behavioral vs Multilevel)
    plt.figure(figsize=(8, 4.5))
    sns.barplot(
        data=df_cross, x="feature_profile", y="fpr", hue="model", palette="rocket_r", errorbar="se"
    )
    plt.title("Cross-Domain False Positive Rate (FPR) Suppression by Profile", fontsize=11, fontweight="bold")
    plt.xlabel("Feature Representation Profile", fontsize=10, fontweight="bold")
    plt.ylabel("Cross-Domain False Positive Rate (FPR)", fontsize=10, fontweight="bold")
    plt.xticks(rotation=15)
    plt.legend(title="Model Baseline")
    plt.tight_layout()
    plt.savefig(plots_dir / "profile_fpr_comparison.png", dpi=300)
    plt.close()

    # 5. Transfer Performance Breakdown by Target Dataset
    plt.figure(figsize=(8, 4.5))
    sns.boxplot(
        data=df_cross, x="target_domain", y="macro_f1", hue="feature_profile", palette="Set2"
    )
    plt.title("Cross-Domain Macro-F1 Breakdown by Target Domain", fontsize=11, fontweight="bold")
    plt.xlabel("Target Domain", fontsize=10, fontweight="bold")
    plt.ylabel("Macro-F1 Score", fontsize=10, fontweight="bold")
    plt.legend(title="Profile", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(plots_dir / "transfer_performance_by_target.png", dpi=300)
    plt.close()

    print(f"Generated 5 Publication-Grade PNG Plots in {plots_dir}:")
    print("  - cross_domain_f1_heatmap.png")
    print("  - cross_domain_fpr_heatmap.png")
    print("  - profile_f1_comparison.png")
    print("  - profile_fpr_comparison.png")
    print("  - transfer_performance_by_target.png")


if __name__ == "__main__":
    generate_plots()
