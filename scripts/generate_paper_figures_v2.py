import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# IEEE publication style parameters
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8.5
plt.rcParams['axes.labelsize'] = 8.5
plt.rcParams['axes.titlesize'] = 9.5
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7.5
plt.rcParams['figure.titlesize'] = 10

out_dir = r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\figures"
os.makedirs(out_dir, exist_ok=True)

# =========================================================
# Figure 1: Performance Comparison (ROC-AUC)
# =========================================================
datasets = ['Edge-IIoTset', 'NF-ToN-IoT-v2', 'ToN-IoT', 'CICIoT2023']
rf_auc = [0.999619, 0.993846, 1.000000, 0.996956]
option_b = [0.997430, 0.993759, 1.000000, 0.995915]
option_c = [0.998472, 0.994659, 1.000000, 0.996503]

x = np.arange(len(datasets))
width = 0.25

fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=300)
rects1 = ax.bar(x - width, rf_auc, width, label='Random Forest (Standalone)', color='#4D648D')
rects2 = ax.bar(x, option_b, width, label='Option B (RF + Std MLP)', color='#808080')
rects3 = ax.bar(x + width, option_c, width, label='Option C (RF + Robust MLP)', color='#283655')

ax.set_ylabel('ROC-AUC Score')
ax.set_title('Cross-Dataset Model ROC-AUC Benchmarks')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0.985, 1.001)
ax.legend(loc='lower right')
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "performance_comparison.pdf"), format='pdf', bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "performance_comparison.png"), format='png', bbox_inches='tight')
plt.close()

# =========================================================
# Figure 2: Adversarial ASR Comparison (PGD-10, eps=0.10)
# =========================================================
datasets_asr = ['Edge-IIoTset', 'NF-ToN-IoT-v2', 'ToN-IoT', 'CICIoT2023']
std_mlp_asr = [86.67, 7.91, 0.00, 0.20]
robust_mlp_asr = [73.13, 0.00, 0.00, 0.20]

x = np.arange(len(datasets_asr))
width = 0.35

fig, ax = plt.subplots(figsize=(6.5, 3.4), dpi=300)
rects1 = ax.bar(x - width/2, std_mlp_asr, width, label='Standard MLP Baseline', color='#D9534F')
rects2 = ax.bar(x + width/2, robust_mlp_asr, width, label='Robust MLP (PGD-7 Adv. Trained)', color='#4B8F8C')

ax.set_ylabel('PGD-10 Attack Success Rate (%)')
ax.set_title('Adversarial Robustness Comparison (PGD-10, $\epsilon=0.10$, $\\alpha=0.025$)')
ax.set_xticks(x)
ax.set_xticklabels(datasets_asr)
ax.set_ylim(-2, 100)
ax.legend(loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Value annotations on ALL bars with clear visibility for low/zero values
for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.2f}%',
                xy=(rect.get_x() + rect.get_width()/2, max(h, 0)),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=7.5, weight='bold')

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.2f}%',
                xy=(rect.get_x() + rect.get_width()/2, max(h, 0)),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=7.5, weight='bold', color='#2E5A57')

# Visual callout box highlighting low/zero ASR environments
ax.text(2.5, 38, "Low/Zero Evasion Environments:\n• ToN-IoT: 0.00% baseline / 0.00% robust\n• CICIoT2023: 0.20% baseline / 0.20% robust\n• NF-ToN-IoT-v2: 7.91% -> 0.00% (7.91 percentage points reduction)",
        fontsize=7.2, bbox=dict(boxstyle="round,pad=0.4", fc="#F8F9FA", ec="#CCCCCC", lw=0.8))

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "adversarial_asr_comparison.pdf"), format='pdf', bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "adversarial_asr_comparison.png"), format='png', bbox_inches='tight')
plt.close()

# =========================================================
# Figure 3: Runtime Performance Benchmark (NEW)
# =========================================================
# Batch sizes: 1 (cold), 1 (warm), 32, 64, 128, 256, 512, 1024
batch_sizes = [1, 32, 64, 128, 256, 512, 1024]
# Latencies in ms per sample: cold N=1 ~39.4 ms, N=32 ~3.2 ms, N=64 ~1.8 ms, N=128 ~0.95 ms, N=256 ~0.48 ms, N=512 ~0.18 ms, N=1024 ~0.045 ms
latencies = [39.4, 3.2, 1.8, 0.95, 0.48, 0.18, 0.045]

fig, ax1 = plt.subplots(figsize=(6.5, 3.4), dpi=300)

color_lat = '#1F77B4'
ax1.set_xlabel('Batch Size ($N$)')
ax1.set_ylabel('Inference Latency (ms / sample)', color=color_lat)
line1 = ax1.plot(batch_sizes, latencies, marker='o', color=color_lat, linewidth=1.8, markersize=5, label='Per-Sample Latency (ms)')
ax1.tick_params(axis='y', labelcolor=color_lat)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_xticks(batch_sizes)
ax1.set_xticklabels([str(b) for b in batch_sizes])

# Annotations for key measured benchmark values
# 1. Cold single sample N=1
ax1.annotate('Cold Start $N=1$\n37.4–41.4 ms\n(Framework Overhead)', xy=(1, 39.4), xytext=(2.2, 25),
             arrowprops=dict(arrowstyle="->", color='#555555', lw=0.9), fontsize=7.2, weight='bold')

# 2. Batched N=32 threshold
ax1.annotate('Batched $N=32$\n< 3.5 ms / sample', xy=(32, 3.2), xytext=(55, 6),
             arrowprops=dict(arrowstyle="->", color='#555555', lw=0.9), fontsize=7.2)

# 3. Warm Batch N=1024 throughput
ax1.annotate('Warm Batch $N=1024$\n0.045 ms / sample\n(~22,000 samples/sec)', xy=(1024, 0.045), xytext=(120, 0.08),
             arrowprops=dict(arrowstyle="->", color='#555555', lw=0.9), fontsize=7.2, weight='bold', color='#1F77B4')

ax1.set_title('Single-CPU Host Runtime Benchmark (Python/NumPy/PyTorch Engine)')
ax1.grid(True, which="both", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "runtime_performance.pdf"), format='pdf', bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "runtime_performance.png"), format='png', bbox_inches='tight')
plt.close()

print("Successfully generated all updated publication figures in ieee_paper_draft/figures/")
