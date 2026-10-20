"""Stage 10 IEEE Technical Monochromatic Architecture & Evidence Chain Generator.

Generates a publication-grade black-and-white / monochromatic vector diagram
showing the Stages 1-9 Evidence Chain -> Stage 10 Final Submission Gate saved in reports/stage10/.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.style.use("default")
plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 9, "figure.autolayout": True})


def generate_ieee_architecture_figure():
    output_dir = Path("reports/stage10")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Title Banner
    ax.text(5.0, 5.7, "END-TO-END RESEARCH PIPELINE AND EVIDENCE RECONCILIATION CHAIN",
            ha="center", va="center", fontsize=11, fontweight="bold", family="sans-serif")
    ax.text(5.0, 5.4, "From Heterogeneous Telemetry Ingestion to IEEE Technical Submission Gate",
            ha="center", va="center", fontsize=9, fontstyle="italic", family="sans-serif")

    # Stage Box Drawing Helper
    def draw_box(x, y, w, h, title, subtitle, metrics, is_gate=False):
        edge_style = "-" if not is_gate else "--"
        lw = 1.2 if not is_gate else 1.8
        fill_color = "white" if not is_gate else "#f0f0f0"
        
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="square,pad=0.0",
            facecolor=fill_color, edgecolor="black", linestyle=edge_style, linewidth=lw
        )
        ax.add_patch(rect)
        
        # Header Box
        header_h = 0.4
        header_rect = patches.Rectangle((x, y + h - header_h), w, header_h, facecolor="#e0e0e0" if not is_gate else "#d0d0d0", edgecolor="black", linewidth=0.8)
        ax.add_patch(header_rect)
        
        ax.text(x + w / 2.0, y + h - header_h / 2.0, title, ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.text(x + w / 2.0, y + h - header_h - 0.15, subtitle, ha="center", va="center", fontsize=7.5, fontstyle="italic")
        
        # Metrics list
        start_y = y + h - header_h - 0.35
        for i, m in enumerate(metrics):
            ax.text(x + 0.1, start_y - i * 0.22, f"• {m}", ha="left", va="center", fontsize=7.5)

    # Box 1: Stage 1-3 Ingestion & Materialization
    draw_box(
        0.3, 3.2, 2.7, 1.8,
        "STAGES 1–3: INGESTION",
        "Canonical Flow Materialization",
        [
            "4 Datasets: ToN-IoT, Edge-IIoT...",
            "28,000 Materialized Flows",
            "18 Semantic Candidate Features",
            "60/20/20 Chronological Splits"
        ]
    )

    # Box 2: Stage 4 Benchmark
    draw_box(
        3.6, 3.2, 2.7, 1.8,
        "STAGE 4: BENCHMARK",
        "Supervised Feature Ablation",
        [
            "160 Benchmark Experiments",
            "5 Profiles x 2 Model Families",
            "In-Domain F1: 0.986 (FPR 1.6%)",
            "Zero-Shot AUC: 0.567 (Level A)"
        ]
    )

    # Box 3: Stage 5 Domain Adaptation
    draw_box(
        6.9, 3.2, 2.8, 1.8,
        "STAGE 5: ADAPTATION",
        "Target Calibration Matrix",
        [
            "840 Adaptation Experiments",
            "7 Regimes (0% to 10% Budget)",
            "Align Recovery: AUC 0.607",
            "5% Budget Recovery: AUC 0.992"
        ]
    )

    # Box 4: Stage 6 Statistical Robustness
    draw_box(
        0.3, 0.7, 2.7, 1.8,
        "STAGE 6: ROBUSTNESS",
        "Effect Sizes & Ablations",
        [
            "Paired Cohen's dz = 1.134 (p<.005)",
            "95% Bootstrap CI: [0.989, 0.996]",
            "Holm-Bonferroni Correction",
            "Shortcut Ablation: 97.2% Ret."
        ]
    )

    # Box 5: Stage 7-9 Compilation
    draw_box(
        3.6, 0.7, 2.7, 1.8,
        "STAGES 7–9: COMPILATION",
        "Manuscript & BibTeX Engine",
        [
            "Reconciled Evidence Matrix",
            "17 Headline Metrics Audit",
            "BibTeX references.bib (0 gaps)",
            "IEEE_paper_final.tex Draft"
        ]
    )

    # Box 6: Stage 10 Final Gate
    draw_box(
        6.9, 0.7, 2.8, 1.8,
        "STAGE 10: SUBMISSION GATE",
        "Adversarial Reviewer Audit",
        [
            "8 Reviewer Attack Vectors",
            "Terminology Precision Edits",
            "Verdict: SUBMISSION READY",
            "34/34 Unit Tests Passed"
        ],
        is_gate=True
    )

    # Connecting Arrows
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2, mutation_scale=12)
        )

    # Top row connections
    draw_arrow(3.0, 4.1, 3.6, 4.1)
    draw_arrow(6.3, 4.1, 6.9, 4.1)
    
    # Downward connection
    draw_arrow(8.3, 3.2, 8.3, 2.5)

    # Bottom row connections (right to left to gate)
    draw_arrow(6.9, 1.6, 6.3, 1.6)
    draw_arrow(3.6, 1.6, 3.0, 1.6)
    draw_arrow(1.65, 0.7, 8.3, 0.7) # Return loop indication

    plt.savefig(output_dir / "stage10_ieee_architecture_pipeline.png", dpi=300)
    plt.close()

    print(f"Generated IEEE Technical Monochromatic Architecture Diagram in {output_dir / 'stage10_ieee_architecture_pipeline.png'}")


if __name__ == "__main__":
    generate_ieee_architecture_figure()
