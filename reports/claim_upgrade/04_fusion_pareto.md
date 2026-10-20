# FUSION-WEIGHT PARETO ANALYSIS

This report evaluates the empirical clean-AUC versus ASR trade-off under the fixed neural-gradient PGD evaluation protocol for the Option C fusion weighting scheme. The goal is to investigate whether the canonical `w_RF = 0.7` is an optimal or defensible operating point.

## Methodological Note on Threat Model
The PGD attack generates perturbations exclusively against the differentiable neural branch and evaluates those perturbations against the entire fused predictor. Therefore, these results must NOT be described as a true worst-case adversarial robustness Pareto frontier for each fusion weight. Specifically, the `w=1.0` (pure RF) ASR is NOT a white-box RF robustness measurement; it is merely the RF response to perturbations generated against the neural branch (a transfer-style attack).

The current test-set sweep is strictly an exploratory ablation. We do not use these test-set Pareto computations to select or tune a new fusion weight for deployment.

## Dataset: Edge-IIoTset

### Condition A: Original Robust Checkpoint (Seed 42)
| Weight ($w_{RF}$) | Clean ROC-AUC | Macro F1 | PGD-10 ASR | Rel AUC vs RF | Rel ASR vs RF | Rel ASR vs MLP | Pareto Optimal |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.0 | 0.8028 | 0.4167 | 0.00% | -0.1967 | -29.90% | +0.00% | No |
| 0.1 | 0.9971 | 0.4167 | 0.00% | -0.0024 | -29.90% | +0.00% | Yes |
| 0.2 | 0.9971 | 0.4167 | 0.00% | -0.0024 | -29.90% | +0.00% | No |
| 0.3 | 0.9971 | 0.9480 | 0.00% | -0.0024 | -29.90% | +0.00% | Yes |
| 0.4 | 0.9971 | 0.9912 | 0.20% | -0.0024 | -29.70% | +0.20% | No |
| 0.5 | 0.9971 | 0.9939 | 0.40% | -0.0024 | -29.50% | +0.40% | Yes |
| 0.6 | 0.9971 | 0.9861 | 0.70% | -0.0024 | -29.20% | +0.70% | Yes |
| 0.7 | 0.9988 | 0.9751 | 8.00% | -0.0007 | -21.90% | +8.00% | Yes |
| 0.8 | 0.9992 | 0.9336 | 21.00% | -0.0003 | -8.90% | +21.00% | Yes |
| 0.9 | 0.9993 | 0.9082 | 27.20% | -0.0002 | -2.70% | +27.20% | Yes |
| 1.0 | 0.9995 | 0.9014 | 29.90% | +0.0000 | +0.00% | +29.90% | Yes |

### Condition B: P2 Validation-Selected Checkpoint (Seed 44)
| Weight ($w_{RF}$) | Clean ROC-AUC | Macro F1 | PGD-10 ASR | Rel AUC vs RF | Rel ASR vs RF | Rel ASR vs MLP | Pareto Optimal |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.0 | 0.8502 | 0.4167 | 0.00% | -0.1493 | -29.10% | +0.00% | No |
| 0.1 | 0.9971 | 0.4167 | 0.00% | -0.0024 | -29.10% | +0.00% | Yes |
| 0.2 | 0.9971 | 0.4167 | 0.00% | -0.0024 | -29.10% | +0.00% | Yes |
| 0.3 | 0.9971 | 0.9422 | 0.00% | -0.0024 | -29.10% | +0.00% | No |
| 0.4 | 0.9971 | 0.9912 | 0.20% | -0.0024 | -28.90% | +0.20% | No |
| 0.5 | 0.9971 | 0.9921 | 0.40% | -0.0024 | -28.70% | +0.40% | Yes |
| 0.6 | 0.9973 | 0.9895 | 2.20% | -0.0022 | -26.90% | +2.20% | Yes |
| 0.7 | 0.9989 | 0.9827 | 10.70% | -0.0005 | -18.40% | +10.70% | Yes |
| 0.8 | 0.9992 | 0.9336 | 21.80% | -0.0002 | -7.30% | +21.80% | Yes |
| 0.9 | 0.9993 | 0.9128 | 24.60% | -0.0002 | -4.50% | +24.60% | Yes |
| 1.0 | 0.9995 | 0.9014 | 29.10% | +0.0000 | +0.00% | +29.10% | Yes |

### Multi-Seed Robustness (10 Seeds)
| Weight ($w_{RF}$) | Mean ASR | Median ASR | SD ASR |
|---:|---:|---:|---:|
| 0.5 | 0.35% | 0.40% | 0.10% |
| 0.6 | 1.87% | 1.50% | 1.57% |
| 0.7 | 8.39% | 9.20% | 4.82% |
| 0.8 | 17.26% | 20.75% | 8.97% |
| 0.9 | 21.24% | 25.90% | 11.03% |
| 1.0 | 23.92% | 29.15% | 12.11% |

---

## Dataset: NF-ToN-IoT-v2

### Condition A: Original Robust Checkpoint (Seed 42)
| Weight ($w_{RF}$) | Clean ROC-AUC | Macro F1 | PGD-10 ASR | Rel AUC vs RF | Rel ASR vs RF | Rel ASR vs MLP | Pareto Optimal |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.0 | 0.8396 | 0.5784 | 53.80% | -0.1544 | +49.60% | +0.00% | No |
| 0.1 | 0.8697 | 0.5914 | 53.80% | -0.1243 | +49.60% | +0.00% | No |
| 0.2 | 0.9536 | 0.8718 | 48.00% | -0.0404 | +43.80% | -5.80% | No |
| 0.3 | 0.9782 | 0.9106 | 48.00% | -0.0158 | +43.80% | -5.80% | No |
| 0.4 | 0.9822 | 0.9156 | 47.80% | -0.0118 | +43.60% | -6.00% | No |
| 0.5 | 0.9895 | 0.9265 | 46.90% | -0.0045 | +42.70% | -6.90% | No |
| 0.6 | 0.9904 | 0.9841 | 45.80% | -0.0036 | +41.60% | -8.00% | No |
| 0.7 | 0.9927 | 0.9859 | 44.80% | -0.0013 | +40.60% | -9.00% | No |
| 0.8 | 0.9926 | 0.9877 | 4.10% | -0.0014 | -0.10% | -49.70% | No |
| 0.9 | 0.9929 | 0.9885 | 4.10% | -0.0010 | -0.10% | -49.70% | Yes |
| 1.0 | 0.9940 | 0.9903 | 4.20% | +0.0000 | +0.00% | -49.60% | Yes |

### Condition B: P2 Validation-Selected Checkpoint (Seed 43)
| Weight ($w_{RF}$) | Clean ROC-AUC | Macro F1 | PGD-10 ASR | Rel AUC vs RF | Rel ASR vs RF | Rel ASR vs MLP | Pareto Optimal |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.0 | 0.8253 | 0.5782 | 53.70% | -0.1687 | +52.20% | +0.00% | No |
| 0.1 | 0.8746 | 0.8671 | 47.90% | -0.1194 | +46.40% | -5.80% | No |
| 0.2 | 0.9570 | 0.9044 | 7.20% | -0.0370 | +5.70% | -46.50% | No |
| 0.3 | 0.9810 | 0.9156 | 7.20% | -0.0130 | +5.70% | -46.50% | No |
| 0.4 | 0.9865 | 0.9156 | 6.60% | -0.0075 | +5.10% | -47.10% | No |
| 0.5 | 0.9897 | 0.9265 | 5.50% | -0.0043 | +4.00% | -48.20% | No |
| 0.6 | 0.9919 | 0.9265 | 1.30% | -0.0021 | -0.20% | -52.40% | No |
| 0.7 | 0.9927 | 0.9285 | 1.20% | -0.0013 | -0.30% | -52.50% | Yes |
| 0.8 | 0.9923 | 0.9877 | 1.20% | -0.0017 | -0.30% | -52.50% | No |
| 0.9 | 0.9930 | 0.9885 | 1.40% | -0.0010 | -0.10% | -52.30% | Yes |
| 1.0 | 0.9940 | 0.9903 | 1.50% | +0.0000 | +0.00% | -52.20% | Yes |

### Multi-Seed Robustness (10 Seeds)
| Weight ($w_{RF}$) | Mean ASR | Median ASR | SD ASR |
|---:|---:|---:|---:|
| 0.5 | 13.99% | 6.25% | 17.53% |
| 0.6 | 12.53% | 4.55% | 18.07% |
| 0.7 | 12.36% | 4.50% | 17.91% |
| 0.8 | 4.13% | 4.10% | 2.01% |
| 0.9 | 4.15% | 4.10% | 1.99% |
| 1.0 | 4.18% | 4.20% | 2.07% |

---

## Dataset: ToN-IoT

### Condition A: Original Robust Checkpoint (Seed 42)
| Weight ($w_{RF}$) | Clean ROC-AUC | Macro F1 | PGD-10 ASR | Rel AUC vs RF | Rel ASR vs RF | Rel ASR vs MLP | Pareto Optimal |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.0 | 0.7772 | 0.5474 | 2.40% | -0.2228 | +2.40% | +0.00% | No |
| 0.1 | 0.9699 | 0.5865 | 1.60% | -0.0301 | +1.60% | -0.80% | No |
| 0.2 | 0.9851 | 0.6123 | 1.10% | -0.0149 | +1.10% | -1.30% | No |
| 0.3 | 0.9962 | 0.9939 | 0.70% | -0.0038 | +0.70% | -1.70% | No |
| 0.4 | 1.0000 | 0.9983 | 0.30% | -0.0000 | +0.30% | -2.10% | No |
| 0.5 | 1.0000 | 0.9991 | 0.00% | +0.0000 | +0.00% | -2.40% | Yes |
| 0.6 | 1.0000 | 1.0000 | 0.00% | +0.0000 | +0.00% | -2.40% | Yes |
| 0.7 | 1.0000 | 1.0000 | 0.00% | +0.0000 | +0.00% | -2.40% | Yes |
| 0.8 | 1.0000 | 1.0000 | 0.00% | +0.0000 | +0.00% | -2.40% | Yes |
| 0.9 | 1.0000 | 1.0000 | 0.00% | +0.0000 | +0.00% | -2.40% | Yes |
| 1.0 | 1.0000 | 1.0000 | 0.00% | +0.0000 | +0.00% | -2.40% | Yes |

### Condition B: P2 Validation-Selected Checkpoint (Seed 42)
| Weight ($w_{RF}$) | Clean ROC-AUC | Macro F1 | PGD-10 ASR | Rel AUC vs RF | Rel ASR vs RF | Rel ASR vs MLP | Pareto Optimal |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.0 | 0.7772 | 0.5474 | 2.40% | -0.2228 | +2.40% | +0.00% | No |
| 0.1 | 0.9699 | 0.5865 | 1.60% | -0.0301 | +1.60% | -0.80% | No |
| 0.2 | 0.9851 | 0.6123 | 1.10% | -0.0149 | +1.10% | -1.30% | No |
| 0.3 | 0.9962 | 0.9939 | 0.70% | -0.0038 | +0.70% | -1.70% | No |
| 0.4 | 1.0000 | 0.9983 | 0.30% | -0.0000 | +0.30% | -2.10% | No |
| 0.5 | 1.0000 | 0.9991 | 0.00% | +0.0000 | +0.00% | -2.40% | Yes |
| 0.6 | 1.0000 | 1.0000 | 0.00% | +0.0000 | +0.00% | -2.40% | Yes |
| 0.7 | 1.0000 | 1.0000 | 0.00% | +0.0000 | +0.00% | -2.40% | Yes |
| 0.8 | 1.0000 | 1.0000 | 0.00% | +0.0000 | +0.00% | -2.40% | Yes |
| 0.9 | 1.0000 | 1.0000 | 0.00% | +0.0000 | +0.00% | -2.40% | Yes |
| 1.0 | 1.0000 | 1.0000 | 0.00% | +0.0000 | +0.00% | -2.40% | Yes |

### Multi-Seed Robustness (10 Seeds)
| Weight ($w_{RF}$) | Mean ASR | Median ASR | SD ASR |
|---:|---:|---:|---:|
| 0.5 | 0.01% | 0.00% | 0.03% |
| 0.6 | 0.01% | 0.00% | 0.03% |
| 0.7 | 0.01% | 0.00% | 0.03% |
| 0.8 | 0.00% | 0.00% | 0.00% |
| 0.9 | 0.00% | 0.00% | 0.00% |
| 1.0 | 0.00% | 0.00% | 0.00% |

---

## Dataset: CICIoT2023

### Condition A: Original Robust Checkpoint (Seed 42)
| Weight ($w_{RF}$) | Clean ROC-AUC | Macro F1 | PGD-10 ASR | Rel AUC vs RF | Rel ASR vs RF | Rel ASR vs MLP | Pareto Optimal |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.0 | 0.9881 | 0.9199 | 1.40% | -0.0087 | -0.30% | +0.00% | No |
| 0.1 | 0.9932 | 0.9209 | 1.40% | -0.0036 | -0.30% | +0.00% | No |
| 0.2 | 0.9944 | 0.9209 | 1.40% | -0.0024 | -0.30% | +0.00% | No |
| 0.3 | 0.9951 | 0.9199 | 1.40% | -0.0017 | -0.30% | +0.00% | No |
| 0.4 | 0.9956 | 0.9226 | 1.30% | -0.0012 | -0.40% | -0.10% | No |
| 0.5 | 0.9959 | 0.9263 | 1.20% | -0.0009 | -0.50% | -0.20% | No |
| 0.6 | 0.9961 | 0.9273 | 1.20% | -0.0007 | -0.50% | -0.20% | No |
| 0.7 | 0.9964 | 0.9282 | 1.10% | -0.0004 | -0.60% | -0.30% | No |
| 0.8 | 0.9965 | 0.9351 | 1.10% | -0.0003 | -0.60% | -0.30% | Yes |
| 0.9 | 0.9966 | 0.9341 | 1.20% | -0.0002 | -0.50% | -0.20% | Yes |
| 1.0 | 0.9968 | 0.9377 | 1.70% | +0.0000 | +0.00% | +0.30% | Yes |

### Condition B: P2 Validation-Selected Checkpoint (Seed 42)
| Weight ($w_{RF}$) | Clean ROC-AUC | Macro F1 | PGD-10 ASR | Rel AUC vs RF | Rel ASR vs RF | Rel ASR vs MLP | Pareto Optimal |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.0 | 0.9881 | 0.9199 | 1.40% | -0.0087 | -0.30% | +0.00% | No |
| 0.1 | 0.9932 | 0.9209 | 1.40% | -0.0036 | -0.30% | +0.00% | No |
| 0.2 | 0.9944 | 0.9209 | 1.40% | -0.0024 | -0.30% | +0.00% | No |
| 0.3 | 0.9951 | 0.9199 | 1.40% | -0.0017 | -0.30% | +0.00% | No |
| 0.4 | 0.9956 | 0.9226 | 1.30% | -0.0012 | -0.40% | -0.10% | No |
| 0.5 | 0.9959 | 0.9263 | 1.20% | -0.0009 | -0.50% | -0.20% | No |
| 0.6 | 0.9961 | 0.9273 | 1.20% | -0.0007 | -0.50% | -0.20% | No |
| 0.7 | 0.9964 | 0.9282 | 1.10% | -0.0004 | -0.60% | -0.30% | No |
| 0.8 | 0.9965 | 0.9351 | 1.10% | -0.0003 | -0.60% | -0.30% | Yes |
| 0.9 | 0.9966 | 0.9341 | 1.20% | -0.0002 | -0.50% | -0.20% | Yes |
| 1.0 | 0.9968 | 0.9377 | 1.70% | +0.0000 | +0.00% | +0.30% | Yes |

### Multi-Seed Robustness (10 Seeds)
| Weight ($w_{RF}$) | Mean ASR | Median ASR | SD ASR |
|---:|---:|---:|---:|
| 0.5 | 1.99% | 1.75% | 1.16% |
| 0.6 | 1.75% | 1.55% | 0.82% |
| 0.7 | 1.51% | 1.45% | 0.52% |
| 0.8 | 1.28% | 1.25% | 0.34% |
| 0.9 | 1.14% | 1.20% | 0.21% |
| 1.0 | 0.87% | 0.70% | 0.44% |

---

## Analysis & Interpretation
The analysis reveals that `w_RF = 0.7` is frequently dominated by other fusion weights depending on the dataset and the precise robust checkpoint. 

For instance:
- On **CICIoT2023** (Condition B), `w=0.7` has AUC=0.9964 and ASR=1.1%, but `w=0.8` has AUC=0.9965 and ASR=1.1%. Therefore, `w=0.8` strictly dominates `w=0.7` under the Pareto criteria.
- On **NF-ToN-IoT-v2** (Condition A), `w=0.7` is entirely excluded from the Pareto frontier.

The multi-seed robustness spread further reinforces that 0.7 is not universally optimal:
- **Edge-IIoTset**: `w=0.5` yields a mean ASR of 0.35% and `w=0.6` yields 1.87%, while `w=0.7` jumps significantly to 8.39%.
- **NF-ToN-IoT-v2**: `w=0.7` averages 12.36% ASR, whereas `w=0.8` surprisingly lowers the mean ASR to 4.13%.
- **CICIoT2023**: `w=0.7` yields 1.51%, while pure RF (`w=1.0`) evaluates to 0.87% against neural perturbations.

## Verdict on Claim 2
**Original claim: "0.7/0.3 is optimal."**

**VERDICT: UNSUPPORTED**

**Supported qualified finding:**
*"The 0.7/0.3 fusion is an empirically evaluated operating point that provides a favorable clean-detection/robustness trade-off under the evaluated neural-gradient PGD protocol on some datasets/checkpoints; however, no globally optimal fusion weight was established, and the preferred weight varies with dataset and neural checkpoint."*
