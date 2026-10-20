"""
Generate publication-quality performance bar plots from frozen evaluation results.
Reads: reports/tables/final_evaluation_results.json
Outputs: paper/figures/fig_indomain_performance.pdf
         paper/figures/fig_crossdomain_performance.pdf
         paper/figures/fig_ablation.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "reports" / "tables" / "final_evaluation_results.json"
OUT_DIR = REPO / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'Edge-IIoTset': '#2196F3',
    'ToN-IoT': '#FF9800',
    'edge_to_ton': '#4CAF50',
    'ton_to_edge': '#9C27B0',
}

with open(RESULTS) as f:
    data = json.load(f)


def fig_indomain():
    """Figure 3: In-domain performance comparison."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'macro_f1', 'roc_auc', 'pr_auc']
    labels = ['Accuracy', 'Precision', 'Recall', 'Attack\nF1', 'Macro\nF1', 'ROC\nAUC', 'PR\nAUC']

    edge = [data['in_domain']['Edge-IIoTset']['supervised_metrics'][m] for m in metrics]
    ton = [data['in_domain']['ToN-IoT']['supervised_metrics'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars1 = ax.bar(x - width/2, [v*100 for v in edge], width, label='Edge-IIoTset',
                   color=COLORS['Edge-IIoTset'], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, [v*100 for v in ton], width, label='ToN-IoT Network',
                   color=COLORS['ToN-IoT'], edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Score (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(60, 105)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

    # Add value labels on key bars (Attack F1)
    for bar, val in zip([bars1[3], bars2[3]], [edge[3]*100, ton[3]*100]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_indomain_performance.pdf')
    fig.savefig(OUT_DIR / 'fig_indomain_performance.png')
    print(f"Saved: fig_indomain_performance.pdf/png")
    plt.close(fig)


def fig_crossdomain():
    """Figure 4: Cross-domain transfer performance comparison."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'macro_f1', 'roc_auc', 'pr_auc']
    labels = ['Accuracy', 'Precision', 'Recall', 'Attack\nF1', 'Macro\nF1', 'ROC\nAUC', 'PR\nAUC']

    e2t = [data['cross_domain']['Edge-IIoTset_to_ToN-IoT'][m] for m in metrics]
    t2e = [data['cross_domain']['ToN-IoT_to_Edge-IIoTset'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars1 = ax.bar(x - width/2, [v*100 for v in e2t], width,
                   label='Edge → ToN', color=COLORS['edge_to_ton'],
                   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, [v*100 for v in t2e], width,
                   label='ToN → Edge', color=COLORS['ton_to_edge'],
                   edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Score (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(30, 105)
    ax.legend(loc='lower left', framealpha=0.9)

    # Value labels on Attack F1
    for bar, val in zip([bars1[3], bars2[3]], [e2t[3]*100, t2e[3]*100]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_crossdomain_performance.pdf')
    fig.savefig(OUT_DIR / 'fig_crossdomain_performance.png')
    print(f"Saved: fig_crossdomain_performance.pdf/png")
    plt.close(fig)


def fig_ablation():
    """Figure 7: Feature-level ablation study."""
    levels = ['A_Static_F_common', 'B_Static_plus_Temporal', 'C_Static_Temporal_Behavioral']
    level_labels = ['F_common\n(6 feat.)', '+ Temporal\n(9 feat.)', '+ Behavioral\n(11 feat.)']

    # Add full in-domain as the 4th bar
    edge_f1 = [data['ablation']['Edge-IIoTset'][l]['f1']*100 for l in levels]
    edge_f1.append(data['in_domain']['Edge-IIoTset']['supervised_metrics']['f1']*100)
    ton_f1 = [data['ablation']['ToN-IoT'][l]['f1']*100 for l in levels]
    ton_f1.append(data['in_domain']['ToN-IoT']['supervised_metrics']['f1']*100)
    full_labels = level_labels + ['Full\n(13 feat.)']

    x = np.arange(len(full_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(x - width/2, edge_f1, width, label='Edge-IIoTset',
           color=COLORS['Edge-IIoTset'], edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, ton_f1, width, label='ToN-IoT',
           color=COLORS['ToN-IoT'], edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Attack F1 (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(full_labels)
    ax.set_ylim(92, 99)
    ax.legend(framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_ablation.pdf')
    fig.savefig(OUT_DIR / 'fig_ablation.png')
    print(f"Saved: fig_ablation.pdf/png")
    plt.close(fig)


if __name__ == '__main__':
    fig_indomain()
    fig_crossdomain()
    fig_ablation()
    print("All performance plots generated.")
