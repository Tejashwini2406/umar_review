"""
generate_architecture_diagram.py
Generates a publication-quality vector architecture diagram for Figure 1.
Outputs:
  - paper/figures/fig_architecture.pdf
  - paper/figures/fig_architecture.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_architecture_diagram():
    repo_root = Path(__file__).parent.parent.parent
    figures_dir = repo_root / "paper" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis('off')

    # Color palette - clean academic style
    c_blue = '#1f77b4'
    c_light_blue = '#e6f0fa'
    c_green = '#2ca02c'
    c_light_green = '#eafaf1'
    c_orange = '#ff7f0e'
    c_light_orange = '#fff5eb'
    c_purple = '#9467bd'
    c_light_purple = '#f3eef8'
    c_gray = '#444444'
    c_light_gray = '#f8f9fa'
    c_red = '#d62728'

    # Box drawer helper
    def draw_box(x, y, w, h, title, text_lines, bg_color, border_color, title_color='black', fontsize=9):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", 
                                  facecolor=bg_color, edgecolor=border_color, linewidth=1.5)
        ax.add_patch(box)
        
        # Title
        ax.text(x + w/2, y + h - 2.2, title, weight='bold', fontsize=fontsize+0.5, 
                ha='center', va='center', color=title_color)
        
        # Line text
        if text_lines:
            n_lines = len(text_lines)
            start_y = y + h - 5.5
            spacing = (h - 6) / max(n_lines, 1)
            for i, line in enumerate(text_lines):
                ax.text(x + w/2, start_y - i * spacing, line, fontsize=fontsize-1, 
                        ha='center', va='center', color=c_gray)

    # Arrow drawer helper
    def draw_arrow(x1, y1, x2, y2, label="", color=c_gray, ls='-'):
        ax.annotate(label, xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5, ls=ls),
                    fontsize=8, ha='center', va='center', color=color,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8) if label else None)

    # --- BLOCK 1: DATASETS & PREPROCESSING ---
    draw_box(2, 45, 20, 20, "IoT Datasets", 
             ["Edge-IIoTset", "(14 Attack + Benign)", "", "ToN-IoT Network", "(9 Attack + Benign)", "", "60% Train / 20% Val / 20% Test"],
             c_light_blue, c_blue)

    draw_box(26, 45, 20, 20, "Feature Hierarchy", 
             ["In-Domain (13 Features):", "• 6 Common (F_common)", "• 3 Temporal (Causal)", "• 2 Behavioral (Diversity)", "• 2 Dataset-Specific", "", "Cross-Domain (6 Features):", "• F_common only"],
             c_light_gray, c_gray)

    draw_arrow(22, 55, 26, 55, "Flow Data")

    # --- BLOCK 2: THREE-MODEL PATHWAYS ---
    # Top Branch: Supervised Path
    draw_box(52, 54, 20, 11, "Random Forest (RF)", 
             ["100 Trees, Max Depth 15", "Outputs Prob P_rf ∈ [0, 1]"], 
             c_light_green, c_green)

    draw_box(52, 38, 20, 11, "PyTorch MLP", 
             ["3 Hidden: 128→64→32", "BatchNorm + Dropout (0.2)", "Outputs Prob P_mlp ∈ [0, 1]"], 
             c_light_green, c_green)

    draw_arrow(46, 55, 52, 59.5, "13 / 6 Features")
    draw_arrow(46, 55, 52, 43.5, "13 / 6 Features")

    # Bottom Branch: Anomaly Path
    draw_box(52, 20, 20, 13, "Benign Autoencoder", 
             ["Encoder: Input→64→16", "Decoder: 16→64→Input", "Trained on Benign Only", "Score: S_ae via ECDF Calibration"], 
             c_light_orange, c_orange)

    draw_arrow(46, 55, 49, 55)
    draw_arrow(49, 55, 49, 26.5)
    draw_arrow(49, 26.5, 52, 26.5, "Benign Flow Features")

    # --- BLOCK 3: ENSEMBLE & RISK LAYER ---
    draw_box(76, 46, 22, 19, "Supervised Ensemble", 
             ["P_sup = 0.5·P_rf + 0.5·P_mlp", "", "Decision Rule:", "• Attack if P_sup ≥ 0.50", "• Benign if P_sup < 0.50"], 
             c_light_purple, c_purple)

    draw_arrow(72, 59.5, 76, 57)
    draw_arrow(72, 43.5, 76, 53)

    draw_box(76, 18, 22, 21, "Design B Risk Layer", 
             ["Inputs: P_sup and S_ae", "", "3-State Output:", "1. HIGH CONFIDENCE ATTACK", "   (P_sup ≥ 0.50)", "2. SUSPICIOUS / ANOMALOUS", "   (P_sup < 0.50 & S_ae ≥ 0.80)", "3. BENIGN", "   (P_sup < 0.50 & S_ae < 0.80)"], 
             c_light_purple, c_purple)

    draw_arrow(87, 46, 87, 39, "P_sup")
    draw_arrow(72, 26.5, 76, 26.5, "S_ae")

    # --- BLOCK 4: EVALUATION & ANALYSIS SUITE (BOTTOM STRIP) ---
    draw_box(2, 2, 96, 11, "Evaluation, Explainability, & Robustness Suite", 
             ["In-Domain Benchmark (13-Feat) | Zero-Adaptation Transfer (6-Feat) | Feature Importance Consensus (Spearman ρ) | FGSM & PGD-10 Evasion vs. AE Catch Rate"], 
             "#ffffff", c_gray, title_color=c_gray, fontsize=9)

    draw_arrow(87, 18, 87, 13)

    plt.tight_layout()
    
    # Save outputs
    pdf_path = figures_dir / "fig_architecture.pdf"
    png_path = figures_dir / "fig_architecture.png"
    
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Generated Figure 1 architecture diagram: {pdf_path.name} and {png_path.name}")

if __name__ == "__main__":
    generate_architecture_diagram()
