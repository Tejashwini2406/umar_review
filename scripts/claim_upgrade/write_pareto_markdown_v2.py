import json

with open('C:/Users/umari/Documents/P1_task_Implementation/reports/claim_upgrade/04_fusion_pareto.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

md = '# FUSION-WEIGHT PARETO ANALYSIS\n\n'
md += 'This report evaluates the empirical clean-AUC versus ASR trade-off under the fixed neural-gradient PGD evaluation protocol for the Option C fusion weighting scheme. The goal is to investigate whether the canonical `w_RF = 0.7` is an optimal or defensible operating point.\n\n'

md += '## Methodological Note on Threat Model\n'
md += 'The PGD attack generates perturbations exclusively against the differentiable neural branch and evaluates those perturbations against the entire fused predictor. Therefore, these results must NOT be described as a true worst-case adversarial robustness Pareto frontier for each fusion weight. Specifically, the `w=1.0` (pure RF) ASR is NOT a white-box RF robustness measurement; it is merely the RF response to perturbations generated against the neural branch (a transfer-style attack).\n\n'
md += 'The current test-set sweep is strictly an exploratory ablation. We do not use these test-set Pareto computations to select or tune a new fusion weight for deployment.\n\n'

for ds, ds_data in data.items():
    md += f'## Dataset: {ds}\n\n'
    
    # Cond A
    cond_a = ds_data['cond_A_original_seed42']
    md += '### Condition A: Original Robust Checkpoint (Seed 42)\n'
    md += '| Weight ($w_{RF}$) | Clean ROC-AUC | Macro F1 | PGD-10 ASR | Rel AUC vs RF | Rel ASR vs RF | Rel ASR vs MLP | Pareto Optimal |\n'
    md += '|---:|---:|---:|---:|---:|---:|---:|:---:|\n'
    for r in cond_a['results']:
        w = r['w_rf']
        auc = r['te_auc']
        f1 = r['te_f1']
        asr = r['te_asr']
        rel_auc = r['rel_auc_vs_rf']
        rel_asr_rf = r['rel_asr_vs_rf']
        rel_asr_mlp = r['rel_asr_vs_mlp']
        is_pareto = 'Yes' if w in cond_a['pareto_frontier'] else 'No'
        md += f'| {w:.1f} | {auc:.4f} | {f1:.4f} | {asr*100:.2f}% | {rel_auc:+.4f} | {rel_asr_rf*100:+.2f}% | {rel_asr_mlp*100:+.2f}% | {is_pareto} |\n'
    md += '\n'
    
    # Cond B
    cond_b = ds_data['cond_B_selected_seed']
    md += f'### Condition B: P2 Validation-Selected Checkpoint (Seed {cond_b["seed"]})\n'
    md += '| Weight ($w_{RF}$) | Clean ROC-AUC | Macro F1 | PGD-10 ASR | Rel AUC vs RF | Rel ASR vs RF | Rel ASR vs MLP | Pareto Optimal |\n'
    md += '|---:|---:|---:|---:|---:|---:|---:|:---:|\n'
    for r in cond_b['results']:
        w = r['w_rf']
        auc = r['te_auc']
        f1 = r['te_f1']
        asr = r['te_asr']
        rel_auc = r['rel_auc_vs_rf']
        rel_asr_rf = r['rel_asr_vs_rf']
        rel_asr_mlp = r['rel_asr_vs_mlp']
        is_pareto = 'Yes' if w in cond_b['pareto_frontier'] else 'No'
        md += f'| {w:.1f} | {auc:.4f} | {f1:.4f} | {asr*100:.2f}% | {rel_auc:+.4f} | {rel_asr_rf*100:+.2f}% | {rel_asr_mlp*100:+.2f}% | {is_pareto} |\n'
    md += '\n'
    
    # Multi-seed
    md += '### Multi-Seed Robustness (10 Seeds)\n'
    md += '| Weight ($w_{RF}$) | Mean ASR | Median ASR | SD ASR |\n'
    md += '|---:|---:|---:|---:|\n'
    for w_str, m_data in ds_data['multi_seed_robustness'].items():
        mean_asr = m_data['mean']
        med_asr = m_data['median']
        sd_asr = m_data['std']
        md += f'| {float(w_str):.1f} | {mean_asr*100:.2f}% | {med_asr*100:.2f}% | {sd_asr*100:.2f}% |\n'
    md += '\n---\n\n'

md += '## Analysis & Interpretation\n'
md += 'The analysis reveals that `w_RF = 0.7` is frequently dominated by other fusion weights depending on the dataset and the precise robust checkpoint. \n\n'
md += 'For instance:\n'
md += '- On **CICIoT2023** (Condition B), `w=0.7` has AUC=0.9964 and ASR=1.1%, but `w=0.8` has AUC=0.9965 and ASR=1.1%. Therefore, `w=0.8` strictly dominates `w=0.7` under the Pareto criteria.\n'
md += '- On **NF-ToN-IoT-v2** (Condition A), `w=0.7` is entirely excluded from the Pareto frontier.\n\n'
md += 'The multi-seed robustness spread further reinforces that 0.7 is not universally optimal:\n'
md += '- **Edge-IIoTset**: `w=0.5` yields a mean ASR of 0.35% and `w=0.6` yields 1.87%, while `w=0.7` jumps significantly to 8.39%.\n'
md += '- **NF-ToN-IoT-v2**: `w=0.7` averages 12.36% ASR, whereas `w=0.8` surprisingly lowers the mean ASR to 4.13%.\n'
md += '- **CICIoT2023**: `w=0.7` yields 1.51%, while pure RF (`w=1.0`) evaluates to 0.87% against neural perturbations.\n\n'

md += '## Verdict on Claim 2\n'
md += '**Original claim: "0.7/0.3 is optimal."**\n\n'
md += '**VERDICT: UNSUPPORTED**\n\n'
md += '**Supported qualified finding:**\n'
md += '*"The 0.7/0.3 fusion is an empirically evaluated operating point that provides a favorable clean-detection/robustness trade-off under the evaluated neural-gradient PGD protocol on some datasets/checkpoints; however, no globally optimal fusion weight was established, and the preferred weight varies with dataset and neural checkpoint."*\n'

with open('C:/Users/umari/Documents/P1_task_Implementation/reports/claim_upgrade/04_fusion_pareto.md', 'w', encoding='utf-8') as f:
    f.write(md)
