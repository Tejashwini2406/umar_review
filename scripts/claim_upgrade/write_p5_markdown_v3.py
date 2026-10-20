import json

with open('C:/Users/umari/Documents/P1_task_Implementation/reports/claim_upgrade/05_blackbox_attack.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

md = '# QUERY-BASED BLACK-BOX ATTACK AGAINST OPTION C ENSEMBLE\n\n'

md += '## 1. Executive Summary\n'
md += 'This report evaluates the empirical robustness of the canonical Option C fusion model (`0.7 * RF + 0.3 * RobustMLP`) against a genuine query-based black-box attack. Unlike the neural-gradient PGD transfer evaluation in P4, this attack directly optimizes against the complete ensemble output using gradient estimation. The results reveal that under a sufficient query budget, the black-box attack achieves significantly higher Attack Success Rates (ASR) than the neural-gradient evaluation, especially against the operational 0.71 detection threshold, indicating that the existing threat model understated the empirical evasion risk of the fusion scheme.\n\n'

md += '## 2. Threat Model\n'
md += '- **Adversary Knowledge**: Black-box (no access to model weights, gradients, or internal RF structure).\n'
md += '- **Observable Output**: The final fused probability $P_C(x)$.\n'
md += '- **Constraint**: True $L_\infty$ bounded perturbations with $\epsilon = 0.10$.\n'
md += '- **Objective**: Cause a malicious sample to be classified as benign. Evaluated under two thresholds: strict attack success ($P_C < 0.50$) and operational evasion ($P_C < 0.71$).\n\n'

md += '## 3. Target Model\n'
md += 'The exact, frozen, canonical Option C implementation:\n'
md += '`OptionC(x) = 0.7 * RF_Prob(x) + 0.3 * MLP_Prob(x)`\n'
md += '(Seed 42 default models, with raw RobustScaler).\n\n'

md += '## 4. Attack Algorithm\n'
md += 'An Antithetic Natural Evolution Strategies (NES) gradient estimator was used. For each active sample in an iteration, 20 queries (10 true antithetic pairs: $+\sigma u$ and $-\sigma u$) were sampled from a Gaussian distribution ($\sigma=0.01$) to estimate the gradient of the complete Option C output. The estimated gradient guided an $L_\infty$ projected gradient descent step ($\alpha=0.025$). The minimum probability observed across all queries ($p_{min\_so\_far}$) is used to determine final attack success.\n\n'

md += '## 5. Query Budget\n'
md += 'Fixed maximum query ladders were pre-defined and strictly enforced per sample: **100**, **250**, and **500** queries. Initial probability checks consumed 1 query, and each NES iteration precisely consumed 21 queries (20 directional gradient estimation queries + 1 step evaluation query). Samples that reached the success threshold were frozen to preserve queries.\n\n'

md += '## 6. Feature Constraints\n'
md += 'The attack was strictly constrained to continuous features. The 4 canonical protocol features were completely frozen using a binary mask, identical to the standard PGD evaluation constraints.\n\n'

md += '## 7. Dataset and Test Protocol\n'
md += 'Evaluated on the canonical 20% held-out test set for all four datasets. Test labels were used only to define the malicious evaluation subset (exactly 1000 samples per dataset) and were not used for optimization, tuning, or early stopping. Crucially, the ASR denominator ($N_{50}$ and $N_{71}$) strictly excludes samples already misclassified by the clean model (i.e. starting below the target threshold).\n\n'

md += '## 8. Results\n'
for ds, ds_data in data.items():
    md += f'### Dataset: {ds}\n'
    md += '| Budget | Strict ASR (< 0.50) | Operational ASR (< 0.71) | Mean Queries | Max Queries |\n'
    md += '|---:|---:|---:|---:|---:|\n'
    for b in [100, 250, 500]:
        r = ds_data[f'budget_{b}']
        n50, s50 = r['eligible_50'], r['successful_50']
        n71, s71 = r['eligible_71'], r['successful_71']
        md += f"| {b} | {s50}/{n50} ({r['asr_50']*100:.2f}%) | {s71}/{n71} ({r['asr_71']*100:.2f}%) | {r['mean_queries']:.1f} | {r['max_queries']} |\n"
    md += '\n'
    if ds == "Edge-IIoTset":
        md += '> **Important Limitation**: For Edge-IIoTset, $N_{71} = 5$. This indicates that 995 out of 1000 malicious test samples were *already* misclassified as benign by the clean Option C predictor at the 0.71 threshold. The 100% operational ASR applies only to the 5 initially correctly classified samples.\n\n'

md += '## 9. Comparison with Neural-Gradient PGD\n'
md += 'The neural-gradient PGD transfer attack (P4) reported raw strict ASRs of approximately 8.0% (Edge) and 12.4% (NF). However, adjusting these to an apples-to-apples basis (excluding the 28 Edge and 63 NF samples natively misclassified by clean Option C from both the numerator and denominator) yields corrected P4 strict ASRs of approximately **5.35%** (Edge) and **6.51%** (NF).\n\n'
md += 'In contrast, the query-based black-box attack evaluated here achieved apples-to-apples strict ASRs under the 500 query budget of **18.31%** (Edge) and **36.07%** (NF). This confirms that neural-gradient transfer substantially understated the empirical evasion risk.\n\n'

md += '## 10. Comparison with Adaptive Surrogate\n'
md += 'Previous evaluations used a surrogate to proxy the RF component. While useful for rapid gradient generation, true query-based optimization against the final Option C score avoids the approximation error of the surrogate, exposing direct evasion paths in the fused landscape.\n\n'

md += '## 11. Query-Budget Sensitivity\n'
md += 'The ASR scales monotonically with the query budget. For example, on NF-ToN-IoT-v2, strict ASR rises from 16.8% (100 queries) to 28.8% (250 queries) to 36.1% (500 queries). This indicates that the attack systematically traverses the fusion loss surface rather than exploiting trivial random directions.\n\n'

md += '## 12. Limitations\n'
md += '- **Query Plausibility**: A 500-query limit per flow may be noisy and detectable by anomaly-based or rate-limiting defense layers in a real network environment.\n'
md += '- **Hyperparameters**: Alpha (0.025) and NES Sigma (0.01) were fixed a priori. Tuning these on a validation set might yield even higher ASRs.\n'
md += '- **Not a Worst-Case Guarantee**: These empirical results represent the evasion rate found by a specific black-box optimizer within a fixed budget; they do not represent an architecture-wide worst-case robustness limit.\n\n'

md += '## 13. Claim Assessment\n'
md += '**Question**: Does P5 establish stronger evidence about Option C under a query-based black-box threat model?\n\n'
md += '**VERDICT: SUPPORTED**\n\n'
md += '**Strongest supported claim**:\n'
md += '*"The frozen Option C ensemble was evaluated under a genuine query-based black-box NES attack with an $L_\infty$ budget of $\epsilon=0.10$ and fixed query budgets up to 500. Under this evaluated threat model, the observed strict ASR (excluding clean errors) reached up to 36.1% on NF-ToN-IoT-v2. These results establish that the previously employed neural-gradient transfer threat model substantially understated the ensemble\'s empirical evasion risk to direct black-box optimization."*\n\n'

md += '## 14. Reproducibility\n'
md += '- Attack executed via: `scripts/claim_upgrade/05_blackbox_attack_batched.py`\n'
md += '- Output artifacts: `reports/claim_upgrade/05_blackbox_attack.json`\n'
md += '- Attack seed: Fixed at `42`.\n'

with open('C:/Users/umari/Documents/P1_task_Implementation/reports/claim_upgrade/05_blackbox_attack.md', 'w', encoding='utf-8') as f:
    f.write(md)
