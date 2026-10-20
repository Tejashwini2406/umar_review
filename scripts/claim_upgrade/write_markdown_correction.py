import json
import numpy as np

with open('C:/Users/umari/Documents/P1_task_Implementation/reports/claim_upgrade/02_multiseed_robustness.json') as f:
    data = json.load(f)

md_content = '# MULTI-SEED ROBUSTNESS & CHECKPOINT SELECTION\n\n'
md_content += '## 1. Experimental Overview\n'
md_content += 'This experiment evaluates whether the seed instability observed in previous audits can be mitigated through either **Robustness-Aware Checkpoint Selection** (via a validation-only metric) or a **Robust MLP Ensemble**.\n\n'
md_content += 'Selection Criterion: `Validation Score = Validation AUC - 1.0 * Validation PGD ASR`\n\n'

for ds, res in data.items():
    md_content += f'## {ds}\n'
    
    seeds = res['baseline_seeds']
    asrs = [s['te_asr'] for s in seeds]
    aucs = [s['te_auc'] for s in seeds]
    
    mean_asr = np.mean(asrs)
    med_asr = np.median(asrs)
    std_asr = np.std(asrs, ddof=1)
    min_asr = np.min(asrs)
    max_asr = np.max(asrs)
    mean_auc = np.mean(aucs)
    
    sel = res['selected_checkpoint']
    ens = res['ensemble']
    
    md_content += '### Baseline (10 Independent Seeds)\n'
    md_content += f'- **Clean ROC-AUC**: {mean_auc:.4f} (Mean)\n'
    md_content += f'- **PGD-10 ASR (Mean)**: {mean_asr:.4f} ({mean_asr*100:.2f}%)\n'
    md_content += f'- **PGD-10 ASR (Median)**: {med_asr:.4f} ({med_asr*100:.2f}%)\n'
    md_content += f'- **Std Dev**: {std_asr:.4f}\n'
    md_content += f'- **Min/Max ASR**: {min_asr:.4f} / {max_asr:.4f}\n'
    md_content += f'- **Number of seeds**: 10\n'
    md_content += f'- **Inference overhead**: 1.0x\n\n'
    
    md_content += '### Robustness-Selected Checkpoint\n'
    md_content += f'- **Selected Seed**: {sel["seed"]}\n'
    md_content += f'- **Validation Score**: {sel["selection_score"]:.4f}\n'
    md_content += f'- **Test Clean ROC-AUC**: {sel["te_auc"]:.4f}\n'
    md_content += f'- **Test Macro F1**: {sel["te_f1"]:.4f}\n'
    md_content += f'- **Test PGD-10 ASR**: {sel["te_asr"]:.4f} ({sel["te_asr"]*100:.2f}%)\n'
    md_content += f'- **Number of checkpoints**: 1\n'
    md_content += f'- **Inference overhead**: 1.0x\n\n'
    
    md_content += '### Robust MLP Ensemble (10 checkpts averaged)\n'
    md_content += f'- **Test Clean ROC-AUC**: {ens["te_auc"]:.4f}\n'
    md_content += f'- **Test Macro F1**: {ens["te_f1"]:.4f}\n'
    md_content += f'- **Test PGD-10 ASR**: {ens["te_asr"]:.4f} ({ens["te_asr"]*100:.2f}%)\n'
    md_content += f'- **Number of checkpoints**: 10\n'
    md_content += f'- **Inference overhead**: {ens["overhead_multiplier"]:.1f}x\n\n'

md_content += '## 2. Attack Implementation Documentation\n'
md_content += 'The PGD evaluation targets only the differentiable neural component (whether a single MLP or the 10-model mean ensemble). The attack generates adversarial perturbations strictly against the neural branch, ignoring the RF branch during gradient computation because the RF is non-differentiable. The generated adversarial examples are then evaluated on the complete Option C fused predictor.\n\n'

md_content += '## 3. Analysis by Dataset\n'
md_content += '- **NF-ToN-IoT-v2**: Strong evidence of benefit from validation robustness selection. The random-seed baseline distribution exhibited severe instability (max ASR ~47.5%). The validation-selected checkpoint achieved a 1.2% test ASR, successfully avoiding the catastrophic failure without observing the test set.\n'
md_content += '- **CICIoT2023**: Modest benefit. Baseline ASR is generally low, and selection yields slight improvements.\n'
md_content += '- **ToN-IoT**: Robustness is already stable across all seeds (~0.0% ASR); selection is unnecessary.\n'
md_content += '- **Edge-IIoTset**: No demonstrated improvement from the tested selection or ensemble procedure. The baseline mean ASR is 8.39%, while the validation-selected checkpoint ASR is 10.70%, and the ensemble ASR is 10.90%.\n\n'

md_content += '## 4. Verdict on Claim 3\n'
md_content += '**Original claim: UNSUPPORTED.**\n\n'
md_content += '**Conditional finding**: A validation-only robustness selection procedure substantially reduced seed-dependent adversarial failures on NF-ToN-IoT-v2 in the evaluated experiment, achieving 1.2% test ASR for the selected checkpoint without test-set selection. This does not establish a universal ASR guarantee or seed-independent architectural robustness.\n'

with open('C:/Users/umari/Documents/P1_task_Implementation/reports/claim_upgrade/02_multiseed_robustness.md', 'w') as f:
    f.write(md_content)
