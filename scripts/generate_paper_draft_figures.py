import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set IEEE publication style parameters
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 11

out_dir = r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\figures"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------
# Figure 1: Framework Architecture
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 3.5), dpi=300)
ax.axis('off')

# Color palette
c_input = '#E8EEF5'
c_feat = '#D0E1F9'
c_rf = '#4D648D'
c_mlp = '#283655'
c_fusion = '#1E1F26'
c_out = '#D9534F'

def draw_box(ax, x, y, w, h, text, color, text_color='black', fontsize=8, lw=1):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", ec="#333333", fc=color, lw=lw)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', color=text_color, fontsize=fontsize, weight='bold', multialignment='center')

# Input Stream
draw_box(ax, 0.02, 0.4, 0.14, 0.25, "IoT Network\nTraffic Stream\n(PCAP / NetFlow)", c_input, fontsize=7.5)

# Feature Materialization
draw_box(ax, 0.20, 0.4, 0.16, 0.25, "Feature Engine\nRaw (13) → Concept (18)\nML Vector (d = 21)", c_feat, fontsize=7.5)

# Dual Stream Branching
draw_box(ax, 0.40, 0.60, 0.18, 0.25, "Stream 1: Random Forest\nTree Ensemble (100 Trees)\nP_RF = Prob(Malicious)", c_rf, text_color='white', fontsize=7.5)
draw_box(ax, 0.40, 0.15, 0.18, 0.25, "Stream 2: Robust MLP\n(PGD-10 Adv. Trained)\nP_MLP = Prob(Malicious)", c_mlp, text_color='white', fontsize=7.5)

# Soft Voting Fusion
draw_box(ax, 0.63, 0.38, 0.16, 0.28, "Option C Fusion\n0.7 × P_RF +\n0.3 × P_MLP_Adv", c_fusion, text_color='white', fontsize=8)

# Output Alert
draw_box(ax, 0.84, 0.4, 0.14, 0.25, "Decision & Response\nLow-Latency Alert\n(< 3.5 ms CPU)", c_out, text_color='white', fontsize=7.5)

# Arrows
arrow_props = dict(arrowstyle="->", lw=1.2, color='#222222')
ax.annotate('', xy=(0.20, 0.525), xytext=(0.16, 0.525), arrowprops=arrow_props)
ax.annotate('', xy=(0.40, 0.725), xytext=(0.36, 0.525), arrowprops=arrow_props)
ax.annotate('', xy=(0.40, 0.275), xytext=(0.36, 0.525), arrowprops=arrow_props)
ax.annotate('', xy=(0.63, 0.525), xytext=(0.58, 0.725), arrowprops=arrow_props)
ax.annotate('', xy=(0.63, 0.525), xytext=(0.58, 0.275), arrowprops=arrow_props)
ax.annotate('', xy=(0.84, 0.525), xytext=(0.79, 0.525), arrowprops=arrow_props)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "framework_architecture.pdf"), format='pdf', bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "framework_architecture.png"), format='png', bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Figure 2: Feature Representation
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=300)
ax.axis('off')

draw_box(ax, 0.05, 0.35, 0.24, 0.35, "Layer 1: Raw Attributes (d = 13)\n• Src/Dst IPs & Ports\n• Packet Lengths & Duration\n• Raw Protocol Headers", '#E1E8ED', fontsize=7.5)
draw_box(ax, 0.38, 0.35, 0.26, 0.35, "Layer 2: Domain-Enriched (d = 18)\n• Window Statistics\n• Length Ratio Metrics\n• TCP/UDP/ICMP Frequencies", '#BBDEFB', fontsize=7.5)
draw_box(ax, 0.70, 0.35, 0.25, 0.35, "Layer 3: ML Vector (d = 21)\n• 18 Numeric Features\n• One-Hot Categorical Flags\n  (protocol_type_icmp/tcp/udp)", '#90CAF9', fontsize=7.5)

ax.annotate('', xy=(0.38, 0.525), xytext=(0.29, 0.525), arrowprops=arrow_props)
ax.annotate('', xy=(0.70, 0.525), xytext=(0.64, 0.525), arrowprops=arrow_props)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "feature_representation.pdf"), format='pdf', bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "feature_representation.png"), format='png', bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Figure 3: Adversarial Pipeline
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=300)
ax.axis('off')

draw_box(ax, 0.02, 0.35, 0.18, 0.35, "Original Vector X\n(d = 21)", '#EFEBE9', fontsize=8)
draw_box(ax, 0.25, 0.35, 0.22, 0.35, "Domain Masking (m)\nFreeze: Protocol Headers,\nPacket Counts\nAllow: Payload/Delay Mod", '#D7CCC8', fontsize=7.5)
draw_box(ax, 0.52, 0.35, 0.22, 0.35, "PGD-10 Iterations\nx^(t+1) = Clip_{X,eps}(\nx^t + alpha * m * sign(grad L))", '#BCAAA4', text_color='white', fontsize=7.5)
draw_box(ax, 0.79, 0.35, 0.19, 0.35, "Adversarial Sample X_adv\n(ε = 0.10 Bound)", '#A1887F', text_color='white', fontsize=8)

ax.annotate('', xy=(0.25, 0.525), xytext=(0.20, 0.525), arrowprops=arrow_props)
ax.annotate('', xy=(0.52, 0.525), xytext=(0.47, 0.525), arrowprops=arrow_props)
ax.annotate('', xy=(0.79, 0.525), xytext=(0.74, 0.525), arrowprops=arrow_props)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "adversarial_pipeline.pdf"), format='pdf', bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "adversarial_pipeline.png"), format='png', bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Figure 4: Performance Comparison (ROC-AUC)
# ---------------------------------------------------------
datasets = ['Edge-IIoTset', 'NF-ToN-IoT-v2', 'ToN-IoT', 'CICIoT2023']
rf_auc = [0.999619, 0.993846, 1.000000, 0.996956]
option_b = [0.997430, 0.993759, 1.000000, 0.995915]
option_c = [0.998472, 0.994659, 1.000000, 0.996503]

x = np.arange(len(datasets))
width = 0.25

fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=300)
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

# ---------------------------------------------------------
# Figure 5: Adversarial ASR Comparison
# ---------------------------------------------------------
datasets_asr = ['Edge-IIoTset', 'NF-ToN-IoT-v2', 'ToN-IoT', 'CICIoT2023']
std_mlp_asr = [86.67, 7.91, 0.00, 0.20]
robust_mlp_asr = [73.13, 0.00, 0.00, 0.20]

x = np.arange(len(datasets_asr))
width = 0.35

fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=300)
rects1 = ax.bar(x - width/2, std_mlp_asr, width, label='Standard MLP (Baseline)', color='#D9534F')
rects2 = ax.bar(x + width/2, robust_mlp_asr, width, label='Robust MLP (Adv. Trained)', color='#5CB85C')

ax.set_ylabel('PGD-10 Attack Success Rate (%)')
ax.set_title('Adversarial Robustness (ε = 0.10 PGD-10 Attack)')
ax.set_xticks(x)
ax.set_xticklabels(datasets_asr)
ax.set_ylim(0, 100)
ax.legend(loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Value annotations
for rect in rects1:
    h = rect.get_height()
    if h > 0:
        ax.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5)
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "adversarial_asr_comparison.pdf"), format='pdf', bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "adversarial_asr_comparison.png"), format='png', bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# Figure 6: Runtime Engine Execution Flow
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=300)
ax.axis('off')

draw_box(ax, 0.05, 0.35, 0.24, 0.35, "Packet Ingestion\n• Low-overhead Listener\n• Sub-ms Parsing", '#E8F5E9', fontsize=7.5)
draw_box(ax, 0.38, 0.35, 0.26, 0.35, "NumPy / C++ Engine\n• Exact Parity (atol < 10^-6)\n• Pre-allocated Tensors", '#C8E6C9', fontsize=7.5)
draw_box(ax, 0.70, 0.35, 0.25, 0.35, "Real-Time Action\n• Latency < 3.5 ms\n• 22/22 Tests Verified", '#A5D6A7', fontsize=7.5)

ax.annotate('', xy=(0.38, 0.525), xytext=(0.29, 0.525), arrowprops=arrow_props)
ax.annotate('', xy=(0.70, 0.525), xytext=(0.64, 0.525), arrowprops=arrow_props)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "runtime_architecture.pdf"), format='pdf', bbox_inches='tight')
plt.savefig(os.path.join(out_dir, "runtime_architecture.png"), format='png', bbox_inches='tight')
plt.close()

print("Successfully generated all 6 PDF & PNG vector figure assets in ieee_paper_draft/figures/")
