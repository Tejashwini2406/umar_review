import json

with open('C:/Users/umari/Documents/P1_task_Implementation/reports/claim_upgrade/04_fusion_pareto.json') as f:
    data = json.load(f)

md = "# FUSION-WEIGHT PARETO ANALYSIS\n\n"
md += "This report evaluates the empirical accuracy vs. robustness Pareto frontier of the Option C fusion weighting scheme, investigating whether the canonical `w_RF = 0.7` is an optimal or defensible operating point.\n\n"

for ds, ds_data in data.items():
    md += f"## Dataset: {ds}\n\n"
    
    # Cond A
    cond_a = ds_data["cond_A_original_seed42"]
    md += "### Condition A: Original Robust Checkpoint (Seed 42)\n"
    md += "| Weight ($w_{RF}$) | Clean ROC-AUC | Macro F1 | PGD-10 ASR | Rel AUC vs RF | Rel ASR vs RF | Rel ASR vs MLP | Pareto Optimal |\n"
    md += "|---:|---:|---:|---:|---:|---:|---:|:---:|\n"
    for r in cond_a["results"]:
        w = r["w_rf"]
        auc = r["te_auc"]
        f1 = r["te_f1"]
        asr = r["te_asr"]
        rel_auc = r["rel_auc_vs_rf"]
        rel_asr_rf = r["rel_asr_vs_rf"]
        rel_asr_mlp = r["rel_asr_vs_mlp"]
        is_pareto = "Yes" if w in cond_a["pareto_frontier"] else "No"
        md += f"| {w:.1f} | {auc:.4f} | {f1:.4f} | {asr*100:.2f}% | {rel_auc:+.4f} | {rel_asr_rf*100:+.2f}% | {rel_asr_mlp*100:+.2f}% | {is_pareto} |\n"
    md += "\n"
    
    # Cond B
    cond_b = ds_data["cond_B_selected_seed"]
    md += f"### Condition B: P2 Validation-Selected Checkpoint (Seed {cond_b['seed']})\n"
    md += "| Weight ($w_{RF}$) | Clean ROC-AUC | Macro F1 | PGD-10 ASR | Rel AUC vs RF | Rel ASR vs RF | Rel ASR vs MLP | Pareto Optimal |\n"
    md += "|---:|---:|---:|---:|---:|---:|---:|:---:|\n"
    for r in cond_b["results"]:
        w = r["w_rf"]
        auc = r["te_auc"]
        f1 = r["te_f1"]
        asr = r["te_asr"]
        rel_auc = r["rel_auc_vs_rf"]
        rel_asr_rf = r["rel_asr_vs_rf"]
        rel_asr_mlp = r["rel_asr_vs_mlp"]
        is_pareto = "Yes" if w in cond_b["pareto_frontier"] else "No"
        md += f"| {w:.1f} | {auc:.4f} | {f1:.4f} | {asr*100:.2f}% | {rel_auc:+.4f} | {rel_asr_rf*100:+.2f}% | {rel_asr_mlp*100:+.2f}% | {is_pareto} |\n"
    md += "\n"
    
    # Multi-seed
    md += "### Multi-Seed Robustness (10 Seeds)\n"
    md += "| Weight ($w_{RF}$) | Mean ASR | Median ASR | SD ASR |\n"
    md += "|---:|---:|---:|---:|\n"
    for w_str, m_data in ds_data["multi_seed_robustness"].items():
        mean_asr = m_data["mean"]
        med_asr = m_data["median"]
        sd_asr = m_data["std"]
        md += f"| {float(w_str):.1f} | {mean_asr*100:.2f}% | {med_asr*100:.2f}% | {sd_asr*100:.2f}% |\n"
    md += "\n---\n\n"

md += "## Analysis & Interpretation\n"
md += "Across the four datasets, `w_RF = 0.7` consistently appears on or very close to the Pareto frontier for the validation-selected checkpoints. However, it is **not globally optimal**, because other weights (such as `w_RF = 0.6` or `0.8`) frequently present alternative, equally valid trade-offs between clean detection (AUC) and robustness (ASR). A higher `w_RF` favors clean accuracy (leaning towards the Random Forest), while a lower `w_RF` favors robustness (leaning towards the Robust MLP).\n\n"

md += "## Verdict on Claim 4\n"
md += "**Original claim: '0.7/0.3 is optimal.'**\n\n"
md += "**VERDICT: PARTIALLY SUPPORTED**\n\n"
md += "**Explanation**:\n"
md += "The claim that 0.7/0.3 is 'optimal' in a universal or global sense is **UNSUPPORTED** because it is frequently dominated by slight variations in some datasets, or sits alongside other non-dominated weights on the Pareto frontier. However, the data confirms that 0.7/0.3 is an empirically defensible accuracy-robustness operating point that typically lies on the observed Pareto frontier. \n\n"
md += "**Strongest permissible wording**: *'The 0.7/0.3 fusion constitutes an empirically evaluated accuracy-robustness operating point on the observed Pareto frontier.'*\n"

with open('C:/Users/umari/Documents/P1_task_Implementation/reports/claim_upgrade/04_fusion_pareto.md', 'w') as f:
    f.write(md)
