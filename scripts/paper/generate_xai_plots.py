"""
Generate publication-quality XAI feature importance plots.
Reads: reports/tables/xai_results.json
Outputs: paper/figures/fig_xai_importance.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
XAI = REPO / "reports" / "tables" / "xai_results.json"
OUT_DIR = REPO / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

with open(XAI) as f:
    xai = json.load(f)


def plot_xai():
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))

    for ax, ds_name in zip(axes, ['Edge-IIoTset', 'ToN-IoT']):
        ds = xai[ds_name]
        features = ds['in_domain_features']

        # Get RF permutation and MLP permutation importances
        rf_perm = [ds['rf_permutation_importance'].get(f, 0) for f in features]
        mlp_perm = [ds['mlp_permutation_importance'].get(f, 0) for f in features]

        # Sort by average importance
        avg_imp = [(rf_perm[i] + mlp_perm[i]) / 2 for i in range(len(features))]
        sorted_idx = np.argsort(avg_imp)[::-1]

        # Take top 8 features
        top_n = min(8, len(features))
        idx = sorted_idx[:top_n]

        y = np.arange(top_n)
        height = 0.35

        feat_labels = [features[i].replace('_', '\n') for i in idx]
        rf_vals = [rf_perm[i] for i in idx]
        mlp_vals = [mlp_perm[i] for i in idx]

        ax.barh(y + height/2, rf_vals, height, label='RF Permutation',
                color='#2196F3', edgecolor='white', linewidth=0.5)
        ax.barh(y - height/2, mlp_vals, height, label='MLP Permutation',
                color='#FF9800', edgecolor='white', linewidth=0.5)

        ax.set_yticks(y)
        ax.set_yticklabels(feat_labels)
        ax.invert_yaxis()
        ax.set_xlabel('Permutation Importance')
        ax.set_title(f'{ds_name}\n(ρ = {ds["consensus_spearman_rho"]:.3f}, '
                     f'p = {ds["consensus_p_value"]:.2e})')
        ax.legend(loc='lower right', fontsize=6)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig_xai_importance.pdf')
    fig.savefig(OUT_DIR / 'fig_xai_importance.png')
    print("Saved: fig_xai_importance.pdf/png")
    plt.close(fig)


if __name__ == '__main__':
    plot_xai()
