# MULTI-SEED ROBUSTNESS & CHECKPOINT SELECTION

## 1. Experimental Overview
This experiment evaluates whether the seed instability observed in previous audits can be mitigated through either **Robustness-Aware Checkpoint Selection** (via a validation-only metric) or a **Robust MLP Ensemble**.

Selection Criterion: `Validation Score = Validation AUC - 1.0 * Validation PGD ASR`

## Edge-IIoTset
### Baseline (10 Independent Seeds)
- **Clean ROC-AUC**: 0.9988 (Mean)
- **PGD-10 ASR (Mean)**: 0.0839 (8.39%)
- **PGD-10 ASR (Median)**: 0.0920 (9.20%)
- **Std Dev**: 0.0482
- **Min/Max ASR**: 0.0030 / 0.1520
- **Number of seeds**: 10
- **Inference overhead**: 1.0x

### Robustness-Selected Checkpoint
- **Selected Seed**: 44
- **Validation Score**: 0.9976
- **Test Clean ROC-AUC**: 0.9989
- **Test Macro F1**: 0.9827
- **Test PGD-10 ASR**: 0.1070 (10.70%)
- **Number of checkpoints**: 1
- **Inference overhead**: 1.0x

### Robust MLP Ensemble (10 checkpts averaged)
- **Test Clean ROC-AUC**: 0.9989
- **Test Macro F1**: 0.9751
- **Test PGD-10 ASR**: 0.1090 (10.90%)
- **Number of checkpoints**: 10
- **Inference overhead**: 10.0x

## NF-ToN-IoT-v2
### Baseline (10 Independent Seeds)
- **Clean ROC-AUC**: 0.9919 (Mean)
- **PGD-10 ASR (Mean)**: 0.1236 (12.36%)
- **PGD-10 ASR (Median)**: 0.0450 (4.50%)
- **Std Dev**: 0.1791
- **Min/Max ASR**: 0.0070 / 0.4750
- **Number of seeds**: 10
- **Inference overhead**: 1.0x

### Robustness-Selected Checkpoint
- **Selected Seed**: 43
- **Validation Score**: 0.9494
- **Test Clean ROC-AUC**: 0.9927
- **Test Macro F1**: 0.9285
- **Test PGD-10 ASR**: 0.0120 (1.20%)
- **Number of checkpoints**: 1
- **Inference overhead**: 1.0x

### Robust MLP Ensemble (10 checkpts averaged)
- **Test Clean ROC-AUC**: 0.9907
- **Test Macro F1**: 0.9295
- **Test PGD-10 ASR**: 0.0450 (4.50%)
- **Number of checkpoints**: 10
- **Inference overhead**: 10.0x

## ToN-IoT
### Baseline (10 Independent Seeds)
- **Clean ROC-AUC**: 1.0000 (Mean)
- **PGD-10 ASR (Mean)**: 0.0001 (0.01%)
- **PGD-10 ASR (Median)**: 0.0000 (0.00%)
- **Std Dev**: 0.0003
- **Min/Max ASR**: 0.0000 / 0.0010
- **Number of seeds**: 10
- **Inference overhead**: 1.0x

### Robustness-Selected Checkpoint
- **Selected Seed**: 42
- **Validation Score**: 1.0000
- **Test Clean ROC-AUC**: 1.0000
- **Test Macro F1**: 1.0000
- **Test PGD-10 ASR**: 0.0000 (0.00%)
- **Number of checkpoints**: 1
- **Inference overhead**: 1.0x

### Robust MLP Ensemble (10 checkpts averaged)
- **Test Clean ROC-AUC**: 1.0000
- **Test Macro F1**: 1.0000
- **Test PGD-10 ASR**: 0.0000 (0.00%)
- **Number of checkpoints**: 10
- **Inference overhead**: 10.0x

## CICIoT2023
### Baseline (10 Independent Seeds)
- **Clean ROC-AUC**: 0.9962 (Mean)
- **PGD-10 ASR (Mean)**: 0.0151 (1.51%)
- **PGD-10 ASR (Median)**: 0.0145 (1.45%)
- **Std Dev**: 0.0052
- **Min/Max ASR**: 0.0110 / 0.0290
- **Number of seeds**: 10
- **Inference overhead**: 1.0x

### Robustness-Selected Checkpoint
- **Selected Seed**: 42
- **Validation Score**: 0.9952
- **Test Clean ROC-AUC**: 0.9964
- **Test Macro F1**: 0.9282
- **Test PGD-10 ASR**: 0.0110 (1.10%)
- **Number of checkpoints**: 1
- **Inference overhead**: 1.0x

### Robust MLP Ensemble (10 checkpts averaged)
- **Test Clean ROC-AUC**: 0.9962
- **Test Macro F1**: 0.9399
- **Test PGD-10 ASR**: 0.0130 (1.30%)
- **Number of checkpoints**: 10
- **Inference overhead**: 10.0x

## 2. Attack Implementation Documentation
The PGD evaluation targets only the differentiable neural component (whether a single MLP or the 10-model mean ensemble). The attack generates adversarial perturbations strictly against the neural branch, ignoring the RF branch during gradient computation because the RF is non-differentiable. The generated adversarial examples are then evaluated on the complete Option C fused predictor.

## 3. Analysis by Dataset
- **NF-ToN-IoT-v2**: Strong evidence of benefit from validation robustness selection. The random-seed baseline distribution exhibited severe instability (max ASR ~47.5%). The validation-selected checkpoint achieved a 1.2% test ASR, successfully avoiding the catastrophic failure without observing the test set.
- **CICIoT2023**: Modest benefit. Baseline ASR is generally low, and selection yields slight improvements.
- **ToN-IoT**: Robustness is already stable across all seeds (~0.0% ASR); selection is unnecessary.
- **Edge-IIoTset**: No demonstrated improvement from the tested selection or ensemble procedure. The baseline mean ASR is 8.39%, while the validation-selected checkpoint ASR is 10.70%, and the ensemble ASR is 10.90%.

## 4. Verdict on Claim 3
**Original claim: UNSUPPORTED.**

**Conditional finding**: A validation-only robustness selection procedure substantially reduced seed-dependent adversarial failures on NF-ToN-IoT-v2 in the evaluated experiment, achieving 1.2% test ASR for the selected checkpoint without test-set selection. This does not establish a universal ASR guarantee or seed-independent architectural robustness.
