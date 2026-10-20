"""
Generate publication-quality adversarial evaluation plots.
Reads: results/adversarial/adversarial_results.json
Outputs: paper/figures/fig_adversarial_evaluation.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
ADV = REPO / "results" / "adversarial" / "adversarial_results.json"
OUT_DIR = REPO / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

with open(ADV) as f:
    adv_data = json.load(f)


def plot_adversarial():
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

    datasets = ['Edge-IIoTset', 'ToN-IoT']
    attacks = ['FGSM', 'PGD-10']
    epsilons = [0.05, 0.1, 0.2, 0.3]
    colors_asr = '#E53935'
    colors_ae = '#43A047'

    for row, ds in enumerate(datasets):
        for col, atk in enumerate(attacks):
            ax = axes[row][col]
            subset = [r for r in adv_data if r['dataset'] == ds and r['attack'] == atk]
            subset.sort(key=lambda r: r['epsilon'])

            eps_vals = [r['epsilon'] for r in subset]
            asr_vals = [r['attack_success_rate'] * 100 for r in subset]
            ae_vals = [r['ae_catch_rate'] * 100 for r in subset]

            x = np.arange(len(eps_vals))
            width = 0.35

            bars1 = ax.bar(x - width/2, asr_vals, width, label='ASR (%)',
                          color=colors_asr, edgecolor='white', linewidth=0.5, alpha=0.85)
            bars2 = ax.bar(x + width/2, ae_vals, width, label='AE Catch (%)',
                          color=colors_ae, edgecolor='white', linewidth=0.5, alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels([f'ε={e}' for e in eps_vals])
            ax.set_title(f'{ds} — {atk}', fontsize=9)

            if col == 0:
                ax.set_ylabel('Rate (%)')
            if row == 0 and col == 0:
                ax.legend(loc='upper left', fontsize=6)

            # Add value labels
            for bar in bars1:
                h = bar.get_height()
                if h > 1:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                            f'{h:.1f}', ha='center', va='bottom', fontsize=6)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_adversarial_evaluation.pdf')
    fig.savefig(OUT_DIR / 'fig_adversarial_evaluation.png')
    print("Saved: fig_adversarial_evaluation.pdf/png")
    plt.close(fig)


if __name__ == '__main__':
    plot_adversarial()
