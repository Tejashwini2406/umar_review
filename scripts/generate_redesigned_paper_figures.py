import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

OUT_DIR = "reports/paper_redesign/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Datasets
datasets = ["ToN-IoT", "Edge-IIoTset", "NF-ToN-IoT-v2", "CICIoT2023"]

# Palette
colors = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#9467bd',
    'info': '#8c564b',
    'dark': '#333333',
    'light': '#f4f6f9',
    'rf': '#2b5c8f',
    'std_mlp': '#d95f02',
    'robust_mlp': '#7570b3',
    'option_b': '#e7298a',
    'option_c': '#1b9e77'
}

print("Generating Figure 1: Framework Architecture...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Draw boxes for pipeline
# Box 1: Heterogeneous IoT Datasets
box1 = patches.FancyBboxPatch((0.05, 0.70), 0.25, 0.22, boxstyle="round,pad=0.03", ec="#2b5c8f", fc="#eef4fb", lw=1.5)
ax.add_patch(box1)
ax.text(0.175, 0.87, "Heterogeneous IoT Datasets", ha="center", va="center", weight="bold", fontsize=10, color="#1a365d")
ax.text(0.175, 0.77, "• ToN-IoT (Zeek Logs)\n• Edge-IIoTset (Industrial PCAP)\n• NF-ToN-IoT-v2 (NetFlow v2)\n• CICIoT2023 (Sub-Flow Agg.)", ha="center", va="center", fontsize=8)

# Box 2: Multi-Level Feature Engine
box2 = patches.FancyBboxPatch((0.375, 0.70), 0.25, 0.22, boxstyle="round,pad=0.03", ec="#7570b3", fc="#f3f0f9", lw=1.5)
ax.add_patch(box2)
ax.text(0.50, 0.87, "Multi-Level Representation", ha="center", va="center", weight="bold", fontsize=10, color="#3c2f5b")
ax.text(0.50, 0.77, "• 13 Core In-Domain\n• 18 Conceptual Semantic\n• 21-Column Continuous ML Vector\n   X ∈ R²¹ (One-Hot Encoded)", ha="center", va="center", fontsize=8)

# Box 3: Model Architectures
box3 = patches.FancyBboxPatch((0.70, 0.70), 0.25, 0.22, boxstyle="round,pad=0.03", ec="#2ca02c", fc="#f0f9f0", lw=1.5)
ax.add_patch(box3)
ax.text(0.825, 0.87, "Base Model Architectures", ha="center", va="center", weight="bold", fontsize=10, color="#1e561e")
ax.text(0.825, 0.77, "• Random Forest (100 Trees)\n• Standard MLP (21→128→64→32→1)\n• Robust MLP (Masked PGD-10)\n• Autoencoder Anomaly Layer", ha="center", va="center", fontsize=8)

# Arrow row 1 to row 2
ax.annotate("", xy=(0.375, 0.81), xytext=(0.30, 0.81), arrowprops=dict(arrowstyle="->", lw=2, color="#555555"))
ax.annotate("", xy=(0.70, 0.81), xytext=(0.625, 0.81), arrowprops=dict(arrowstyle="->", lw=2, color="#555555"))

# Box 4: Ensemble Strategies
box4 = patches.FancyBboxPatch((0.15, 0.30), 0.32, 0.26, boxstyle="round,pad=0.03", ec="#e7298a", fc="#fdf0f5", lw=1.5)
ax.add_patch(box4)
ax.text(0.31, 0.51, "Probability Ensemble Formulations", ha="center", va="center", weight="bold", fontsize=10, color="#7d1649")
ax.text(0.31, 0.40, "Option B (Standard Ensemble):\n  P = 0.7 P_RF + 0.3 P_MLP_Std (μ_AUC = 0.9968)\nOption C (Selected Robust Ensemble):\n  P = 0.7 P_RF + 0.3 P_MLP_Adv (μ_AUC = 0.9974)\n  Minimal Cross-Domain Variance (σ² = 0.000004)", ha="center", va="center", fontsize=8.5)

# Box 5: Fail-Fast Runtime Engine
box5 = patches.FancyBboxPatch((0.53, 0.30), 0.32, 0.26, boxstyle="round,pad=0.03", ec="#1b9e77", fc="#edf7f4", lw=1.5)
ax.add_patch(box5)
ax.text(0.69, 0.51, "Production Runtime Engine", ha="center", va="center", weight="bold", fontsize=10, color="#0d503c")
ax.text(0.69, 0.40, "• Schema Validation (Strict 21-Col)\n• Profile Routing (Profile-Aware)\n• 100% Numerical Parity (atol < 10⁻⁶)\n• Sub-Millisecond Batch Latency\n• SHAP Explainability Verification", ha="center", va="center", fontsize=8.5)

# Connector to bottom row
ax.annotate("", xy=(0.31, 0.56), xytext=(0.825, 0.70), arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555", connectionstyle="arc3,rad=0.2"))
ax.annotate("", xy=(0.69, 0.56), xytext=(0.825, 0.70), arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555", connectionstyle="arc3,rad=-0.2"))

# Output box
box6 = patches.FancyBboxPatch((0.30, 0.03), 0.40, 0.15, boxstyle="round,pad=0.03", ec="#333333", fc="#333333", lw=1.5)
ax.add_patch(box6)
ax.text(0.50, 0.105, "Secure & Robust IoT Intrusion Decision Output\n(Attack vs Benign Probability + SHAP Attribution)", ha="center", va="center", weight="bold", fontsize=9.5, color="#ffffff")

ax.annotate("", xy=(0.50, 0.18), xytext=(0.31, 0.30), arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))
ax.annotate("", xy=(0.50, 0.18), xytext=(0.69, 0.30), arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))

plt.title("Figure 1: Overall End-to-End Proposed Intrusion Detection Architecture", fontsize=12, pad=15, weight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "framework_architecture.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, "framework_architecture.png"), bbox_inches='tight')
plt.close(fig)


print("Generating Figure 2: Multi-Level Feature Representation...")
fig, ax = plt.subplots(figsize=(9, 5))
ax.axis('off')

# Flow of features: 13 -> 18 -> 21
box_raw = patches.FancyBboxPatch((0.05, 0.65), 0.26, 0.25, boxstyle="round,pad=0.02", ec="#1f77b4", fc="#ebf3fb", lw=1.5)
ax.add_patch(box_raw)
ax.text(0.18, 0.84, "Level A: Instantaneous\n(Flow & Transport)", ha="center", va="center", weight="bold", fontsize=9.5, color="#12466b")
ax.text(0.18, 0.72, "8 Conceptual Features:\n• flow_duration\n• flow_bytes_per_sec\n• flow_pkts_per_sec\n• mean_pkt_size\n• payload_byte_ratio\n• pkt_count_ratio\n• tcp_syn_ratio\n• protocol (categorical)", ha="center", va="center", fontsize=7.5)

box_temp = patches.FancyBboxPatch((0.37, 0.65), 0.26, 0.25, boxstyle="round,pad=0.02", ec="#ff7f0e", fc="#fff5eb", lw=1.5)
ax.add_patch(box_temp)
ax.text(0.50, 0.84, "Level B: Causal Temporal\n(Inter-Arrival & EWMA)", ha="center", va="center", weight="bold", fontsize=9.5, color="#994c08")
ax.text(0.50, 0.72, "5 Conceptual Features:\n• temporal_iat_mean\n• temporal_iat_cv\n• temporal_flow_rate_ewma\n• temporal_byte_rate_ewma\n• temporal_syn_rate_ewma", ha="center", va="center", fontsize=7.5)

box_behav = patches.FancyBboxPatch((0.69, 0.65), 0.26, 0.25, boxstyle="round,pad=0.02", ec="#2ca02c", fc="#f0f9f0", lw=1.5)
ax.add_patch(box_behav)
ax.text(0.82, 0.84, "Level C: Causal Behavioral\n(Host Interaction Topology)", ha="center", va="center", weight="bold", fontsize=9.5, color="#175417")
ax.text(0.82, 0.72, "5 Conceptual Features:\n• behavioral_dst_diversity\n• behavioral_port_entropy\n• behavioral_fanout_ratio\n• behavioral_unanswered_ratio\n• behavioral_src_activity_ewma", ha="center", va="center", fontsize=7.5)

# Summarizing banner
box_conceptual = patches.FancyBboxPatch((0.15, 0.35), 0.70, 0.18, boxstyle="round,pad=0.02", ec="#666666", fc="#f5f5f5", lw=1.5)
ax.add_patch(box_conceptual)
ax.text(0.50, 0.47, "18 Conceptual Semantic Feature Hierarchy (13 Core In-Domain Representation)", ha="center", va="center", weight="bold", fontsize=9.5)
ax.text(0.50, 0.40, "Combines Instantaneous (8), Temporal Causal (5), and Behavioral Host Topology (5)", ha="center", va="center", fontsize=8.5)

# Downward mapping to 21
ax.annotate("", xy=(0.50, 0.22), xytext=(0.50, 0.35), arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))

box_21 = patches.FancyBboxPatch((0.10, 0.04), 0.80, 0.18, boxstyle="round,pad=0.02", ec="#d62728", fc="#fdf2f2", lw=1.5)
ax.add_patch(box_21)
ax.text(0.50, 0.16, "Uniform Continuous Machine Learning Input Vector: X ∈ R²¹", ha="center", va="center", weight="bold", fontsize=10.5, color="#911a1b")
ax.text(0.50, 0.08, "17 Continuous Numerical Features + 4 One-Hot Protocol Columns [proto_tcp, proto_udp, proto_icmp, proto_other]\nEnables zero-code-change inference across ToN-IoT, Edge-IIoTset, NF-ToN-IoT-v2, and CICIoT2023", ha="center", va="center", fontsize=8.5)

plt.title("Figure 2: Multi-Level Feature Representation Architecture (13 → 18 → 21 Mapping)", fontsize=11, pad=15, weight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "feature_representation.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, "feature_representation.png"), bbox_inches='tight')
plt.close(fig)


print("Generating Figure 3: Adversarial Training Pipeline...")
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.axis('off')

# Pipeline steps
steps = [
    ("Clean Input\nSample X ∈ R²¹", "#e1f5fe", "#0288d1"),
    ("Feature Masking\n& Constraints", "#fff3e0", "#f57c00"),
    ("Domain-Bounded\nPGD-10 Perturbation", "#ffebee", "#d32f2f"),
    ("Clamped Adversarial\nSample X_adv", "#f3e5f5", "#7b1fa2"),
    ("Robust MLP Loss\n& Model Weights", "#e8f5e9", "#388e3c")
]

for i, (title, bg, border) in enumerate(steps):
    x_pos = 0.03 + i * 0.19
    box = patches.FancyBboxPatch((x_pos, 0.30), 0.16, 0.45, boxstyle="round,pad=0.02", ec=border, fc=bg, lw=1.5)
    ax.add_patch(box)
    ax.text(x_pos + 0.08, 0.55, title, ha="center", va="center", weight="bold", fontsize=8.5, color=border)
    
    if i == 1:
        ax.text(x_pos + 0.08, 0.37, "proto_* masked\ncontinuous_mask=0\nx_min ≤ x ≤ x_max", ha="center", va="center", fontsize=7)
    elif i == 2:
        ax.text(x_pos + 0.08, 0.37, "ε = 0.10\n10 Gradient\nSteps (PGD)", ha="center", va="center", fontsize=7)
    elif i == 3:
        ax.text(x_pos + 0.08, 0.37, "Preserves valid\nnetwork domain\nsemantics", ha="center", va="center", fontsize=7)
    elif i == 4:
        ax.text(x_pos + 0.08, 0.37, "Updates weights\non clean + adv\nsamples", ha="center", va="center", fontsize=7)

    if i < 4:
        ax.annotate("", xy=(x_pos + 0.19, 0.525), xytext=(x_pos + 0.16, 0.525), arrowprops=dict(arrowstyle="->", lw=2, color="#555555"))

plt.title("Figure 3: Domain-Constrained Masked PGD-10 Adversarial Training Pipeline", fontsize=11, pad=15, weight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "adversarial_pipeline.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, "adversarial_pipeline.png"), bbox_inches='tight')
plt.close(fig)


print("Generating Figure 4: Runtime Architecture...")
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.axis('off')

r_steps = [
    ("Incoming IoT Traffic\nRecord", "#f4f6f7", "#34495e"),
    ("Schema Validation\n(STANDARDIZED_21)", "#e8f8f5", "#16a085"),
    ("Profile Routing &\nTransformation", "#ebf5fb", "#2980b9"),
    ("Option C Supervised\nInference (RF+MLP)", "#fef9e7", "#f39c12"),
    ("Parity Verified Output\n(atol < 10⁻⁶)", "#eaf2f8", "#1f618d")
]

for i, (title, bg, border) in enumerate(r_steps):
    x_pos = 0.03 + i * 0.19
    box = patches.FancyBboxPatch((x_pos, 0.30), 0.16, 0.45, boxstyle="round,pad=0.02", ec=border, fc=bg, lw=1.5)
    ax.add_patch(box)
    ax.text(x_pos + 0.08, 0.55, title, ha="center", va="center", weight="bold", fontsize=8.5, color=border)
    
    if i == 1:
        ax.text(x_pos + 0.08, 0.38, "Fail-fast check\nRejects missing/NaN\n22/22 Tests Pass", ha="center", va="center", fontsize=7)
    elif i == 2:
        ax.text(x_pos + 0.08, 0.38, "Profile-aware\n.transform() only\nNo .fit() in runtime", ha="center", va="center", fontsize=7)
    elif i == 3:
        ax.text(x_pos + 0.08, 0.38, "P_RF + P_MLP_Adv\nFusion α = 0.7", ha="center", va="center", fontsize=7)
    elif i == 4:
        ax.text(x_pos + 0.08, 0.38, "Latency < 3.5ms\nThroughput > 100k\nsamples/sec", ha="center", va="center", fontsize=7)

    if i < 4:
        ax.annotate("", xy=(x_pos + 0.19, 0.525), xytext=(x_pos + 0.16, 0.525), arrowprops=dict(arrowstyle="->", lw=2, color="#555555"))

plt.title("Figure 4: Fail-Fast Profile-Aware Production Runtime Engine Architecture", fontsize=11, pad=15, weight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "runtime_architecture.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, "runtime_architecture.png"), bbox_inches='tight')
plt.close(fig)


print("Generating Figure 5: Option B vs Option C Architecture Comparison...")
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.axis('off')

# Left side: Option B
box_opt_b = patches.FancyBboxPatch((0.05, 0.15), 0.40, 0.70, boxstyle="round,pad=0.03", ec="#e7298a", fc="#fdf0f5", lw=1.5)
ax.add_patch(box_opt_b)
ax.text(0.25, 0.78, "Option B (Standard Ensemble)", ha="center", va="center", weight="bold", fontsize=10.5, color="#7d1649")
ax.text(0.25, 0.68, "Formulation:\n P_B = 0.7 P_RF + 0.3 P_MLP_Std", ha="center", va="center", fontsize=8.5, weight="bold")
ax.text(0.25, 0.50, "Components:\n• Random Forest (RF)\n• Standard MLP (mlp_model.pt)\n\nPerformance:\n• Mean AUC = 0.996776 (~0.9968)\n• Cross-Dataset Variance = 0.00000517\n• High vulnerability under PGD attacks", ha="center", va="center", fontsize=8)

# Right side: Option C
box_opt_c = patches.FancyBboxPatch((0.55, 0.15), 0.40, 0.70, boxstyle="round,pad=0.03", ec="#1b9e77", fc="#edf7f4", lw=2)
ax.add_patch(box_opt_c)
ax.text(0.75, 0.78, "Option C (Selected Robust Ensemble)", ha="center", va="center", weight="bold", fontsize=10.5, color="#0d503c")
ax.text(0.75, 0.68, "Formulation:\n P_C = 0.7 P_RF + 0.3 P_MLP_Adv", ha="center", va="center", fontsize=8.5, weight="bold")
ax.text(0.75, 0.50, "Components:\n• Random Forest (RF)\n• Robust MLP (mlp_adversarial.pt)\n\nPerformance:\n• Mean AUC = 0.997408 (~0.9974)\n• Cross-Dataset Variance = 0.00000406\n• Superior Adversarial Resilience", ha="center", va="center", fontsize=8)

plt.title("Figure 5: Experimental Architecture Lineage & Option C Selection", fontsize=11, pad=15, weight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "option_ensemble_comparison.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, "option_ensemble_comparison.png"), bbox_inches='tight')
plt.close(fig)


print("Generating Plot 1 & 2: ROC-AUC Benchmark Comparison...")
rf_auc = [1.0000, 0.999619, 0.993846, 0.996956]
std_mlp_auc = [0.795011, 0.979963, 0.984741, 0.977755]
opt_b_auc = [1.0000, 0.997430, 0.993759, 0.995915]
opt_c_auc = [1.000000, 0.998472, 0.994659, 0.996503]

x = np.arange(len(datasets))
width = 0.20

fig, ax = plt.subplots(figsize=(10, 5))
rects1 = ax.bar(x - 1.5*width, rf_auc, width, label='Random Forest (RF)', color=colors['rf'])
rects2 = ax.bar(x - 0.5*width, std_mlp_auc, width, label='Standard MLP', color=colors['std_mlp'])
rects3 = ax.bar(x + 0.5*width, opt_b_auc, width, label='Option B (RF + Std MLP)', color=colors['option_b'])
rects4 = ax.bar(x + 1.5*width, opt_c_auc, width, label='Option C (Selected RF + Robust MLP)', color=colors['option_c'])

ax.set_ylabel('ROC-AUC Score', weight='bold')
ax.set_title('Plot 1: Cross-Dataset ROC-AUC Performance Across Four IoT Observation Layers', weight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(datasets, weight='bold')
ax.set_ylim(0.75, 1.02)
ax.legend(loc='lower right', frameon=True)

# Value annotations
for rect in rects4:
    height = rect.get_height()
    ax.annotate(f'{height:.4f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=7, weight='bold', color=colors['option_c'])

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "auc_comparison.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, "auc_comparison.png"), bbox_inches='tight')
plt.close(fig)


print("Generating Plot 3: Adversarial Attack Success Rate Comparison...")
std_asr = [0.0, 86.67, 7.91, 0.20]
rob_asr = [0.0, 73.13, 0.0, 0.20]

fig, ax = plt.subplots(figsize=(9, 4.5))
rects1 = ax.bar(x - width/2, std_asr, width, label='Standard MLP PGD ASR', color=colors['std_mlp'])
rects2 = ax.bar(x + width/2, rob_asr, width, label='Robust MLP PGD ASR (Masked PGD-10)', color=colors['option_c'])

ax.set_ylabel('Attack Success Rate (ASR %)', weight='bold')
ax.set_title('Plot 3: PGD-10 Adversarial Attack Success Rate (ε = 0.10, Continuous Feature Perturbation)', weight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(datasets, weight='bold')
ax.set_ylim(0, 100)
ax.legend(loc='upper right', frameon=True)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha='center', fontsize=8)

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0,3), textcoords="offset points", ha='center', fontsize=8, weight='bold', color=colors['option_c'])

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "adversarial_asr_comparison.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, "adversarial_asr_comparison.png"), bbox_inches='tight')
plt.close(fig)


print("Generating Plot 4: SHAP Spearman Rank Agreement...")
shap_rho = [0.507308, 0.691333, 0.629389, 0.772549]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(datasets, shap_rho, color=colors['primary'], width=0.45, edgecolor='#12466b', lw=1.2)
ax.set_ylabel("Spearman Rank Correlation (ρ)", weight='bold')
ax.set_title("Plot 4: SHAP Attribution Consensus Between Decision Trees and Neural Networks (N=1,000)", weight='bold', pad=15)
ax.set_ylim(0.0, 1.0)

for bar in bars:
    h = bar.get_height()
    ax.annotate(f'ρ = {h:.4f}', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha='center', weight='bold', fontsize=9)

ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Moderate Consensus Threshold (ρ ≥ 0.5)')
ax.legend(loc='lower right')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "shap_rank_agreement.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, "shap_rank_agreement.png"), bbox_inches='tight')
plt.close(fig)


print("Generating Plot 5: Performance Stability & Variance Visualization...")
models = ["Standard MLP", "Option B Ensemble", "Option C (Selected)"]
mean_aucs = [0.934368, 0.996776, 0.997408]
variances = [0.006480, 0.00000517, 0.00000406]

fig, ax1 = plt.subplots(figsize=(8, 4.5))

color = 'tab:blue'
ax1.set_xlabel('Architecture Model Formulation', weight='bold')
ax1.set_ylabel('Cross-Dataset Mean AUC', color=color, weight='bold')
bars1 = ax1.bar(np.arange(len(models)) - 0.15, mean_aucs, width=0.3, color=color, alpha=0.8, label='Mean AUC')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0.90, 1.00)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Cross-Dataset Variance (σ²)', color=color, weight='bold')
bars2 = ax2.bar(np.arange(len(models)) + 0.15, variances, width=0.3, color=color, alpha=0.8, label='AUC Variance (σ²)')
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_xticks(np.arange(len(models)))
ax1.set_xticklabels(models, weight='bold')

plt.title("Plot 5: Cross-Dataset Performance Stability & Variance Comparison", weight='bold', pad=15)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ensemble_stability.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, "ensemble_stability.png"), bbox_inches='tight')
plt.close(fig)

print("All 9 figures and plots successfully generated!")
