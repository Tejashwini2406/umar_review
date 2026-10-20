# QUERY-BASED BLACK-BOX ATTACK AGAINST OPTION C ENSEMBLE

> **Historical result; superseded for the current corrected run.** The referenced script uses an absolute path under another user's profile, loads legacy Stage 3 data, refits a scaler and models, and does not verify the current corrected split/model hashes. Its numbers below must not be presented as results for `models/corrected_run`. The current reproducible entry point is `scripts/25_evaluate_black_box_ensemble_attack.py`; its output is marked exploratory and checkpoint-specific.

## 1. Executive Summary
This report evaluates the empirical robustness of the canonical Option C fusion model (`0.7 * RF + 0.3 * RobustMLP`) against a genuine query-based black-box attack. Unlike the neural-gradient PGD transfer evaluation in P4, this attack directly optimizes against the complete ensemble output using gradient estimation. The results reveal that under a sufficient query budget, the black-box attack achieves significantly higher Attack Success Rates (ASR) than the neural-gradient evaluation, especially against the operational 0.71 detection threshold, indicating that the existing threat model understated the empirical evasion risk of the fusion scheme.

## 2. Threat Model
- **Adversary Knowledge**: Black-box (no access to model weights, gradients, or internal RF structure).
- **Observable Output**: The final fused probability $P_C(x)$.
- **Constraint**: True $L_\infty$ bounded perturbations with $\epsilon = 0.10$.
- **Objective**: Cause a malicious sample to be classified as benign. Evaluated under two thresholds: strict attack success ($P_C < 0.50$) and operational evasion ($P_C < 0.71$).

## 3. Target Model
The exact, frozen, canonical Option C implementation:
`OptionC(x) = 0.7 * RF_Prob(x) + 0.3 * MLP_Prob(x)`
(Seed 42 default models, with raw RobustScaler).

## 4. Attack Algorithm
An Antithetic Natural Evolution Strategies (NES) gradient estimator was used. For each active sample in an iteration, 20 queries (10 true antithetic pairs: $+\sigma u$ and $-\sigma u$) were sampled from a Gaussian distribution ($\sigma=0.01$) to estimate the gradient of the complete Option C output. The estimated gradient guided an $L_\infty$ projected gradient descent step ($lpha=0.025$). The minimum probability observed across all queries ($p_{min\_so\_far}$) is used to determine final attack success.

## 5. Query Budget
Fixed maximum query ladders were pre-defined and strictly enforced per sample: **100**, **250**, and **500** queries. Initial probability checks consumed 1 query, and each NES iteration precisely consumed 21 queries (20 directional gradient estimation queries + 1 step evaluation query). Samples that reached the success threshold were frozen to preserve queries.

## 6. Feature Constraints
The attack was strictly constrained to continuous features. The 4 canonical protocol features were completely frozen using a binary mask, identical to the standard PGD evaluation constraints.

## 7. Dataset and Test Protocol
Evaluated on the canonical 20% held-out test set for all four datasets. Test labels were used only to define the malicious evaluation subset (exactly 1000 samples per dataset) and were not used for optimization, tuning, or early stopping. Crucially, the ASR denominator ($N_{50}$ and $N_{71}$) strictly excludes samples already misclassified by the clean model (i.e. starting below the target threshold).

## 8. Results
### Dataset: Edge-IIoTset
| Budget | Strict ASR (< 0.50) | Operational ASR (< 0.71) | Mean Queries | Max Queries |
|---:|---:|---:|---:|---:|
| 100 | 86/972 (8.85%) | 2/5 (40.00%) | 81.2 | 85 |
| 250 | 132/972 (13.58%) | 4/5 (80.00%) | 211.4 | 232 |
| 500 | 178/972 (18.31%) | 5/5 (100.00%) | 423.1 | 484 |

> **Important Limitation**: For Edge-IIoTset, $N_{71} = 5$. This indicates that 995 out of 1000 malicious test samples were *already* misclassified as benign by the clean Option C predictor at the 0.71 threshold. The 100% operational ASR applies only to the 5 initially correctly classified samples.

### Dataset: NF-ToN-IoT-v2
| Budget | Strict ASR (< 0.50) | Operational ASR (< 0.71) | Mean Queries | Max Queries |
|---:|---:|---:|---:|---:|
| 100 | 168/998 (16.83%) | 744/994 (74.85%) | 79.5 | 85 |
| 250 | 287/998 (28.76%) | 802/994 (80.68%) | 192.9 | 232 |
| 500 | 360/998 (36.07%) | 834/994 (83.90%) | 362.3 | 484 |

### Dataset: ToN-IoT
| Budget | Strict ASR (< 0.50) | Operational ASR (< 0.71) | Mean Queries | Max Queries |
|---:|---:|---:|---:|---:|
| 100 | 1/1000 (0.10%) | 4/999 (0.40%) | 85.0 | 85 |
| 250 | 1/1000 (0.10%) | 7/999 (0.70%) | 231.9 | 232 |
| 500 | 1/1000 (0.10%) | 12/999 (1.20%) | 483.6 | 484 |

### Dataset: CICIoT2023
| Budget | Strict ASR (< 0.50) | Operational ASR (< 0.71) | Mean Queries | Max Queries |
|---:|---:|---:|---:|---:|
| 100 | 3/992 (0.30%) | 200/983 (20.35%) | 84.9 | 85 |
| 250 | 3/992 (0.30%) | 282/983 (28.69%) | 231.4 | 232 |
| 500 | 4/992 (0.40%) | 325/983 (33.06%) | 482.5 | 484 |

## 9. Comparison with Neural-Gradient PGD
The neural-gradient PGD transfer attack (P4) reported raw strict ASRs of approximately 8.0% (Edge) and 12.4% (NF). However, adjusting these to an apples-to-apples basis (excluding the 28 Edge and 63 NF samples natively misclassified by clean Option C from both the numerator and denominator) yields corrected P4 strict ASRs of approximately **5.35%** (Edge) and **6.51%** (NF).

In contrast, the query-based black-box attack evaluated here achieved apples-to-apples strict ASRs under the 500 query budget of **18.31%** (Edge) and **36.07%** (NF). This confirms that neural-gradient transfer substantially understated the empirical evasion risk.

## 10. Comparison with Adaptive Surrogate
Previous evaluations used a surrogate to proxy the RF component. While useful for rapid gradient generation, true query-based optimization against the final Option C score avoids the approximation error of the surrogate, exposing direct evasion paths in the fused landscape.

## 11. Query-Budget Sensitivity
The ASR scales monotonically with the query budget. For example, on NF-ToN-IoT-v2, strict ASR rises from 16.8% (100 queries) to 28.8% (250 queries) to 36.1% (500 queries). This indicates that the attack systematically traverses the fusion loss surface rather than exploiting trivial random directions.

## 12. Limitations
- **Query Plausibility**: A 500-query limit per flow may be noisy and detectable by anomaly-based or rate-limiting defense layers in a real network environment.
- **Hyperparameters**: Alpha (0.025) and NES Sigma (0.01) were fixed a priori. Tuning these on a validation set might yield even higher ASRs.
- **Not a Worst-Case Guarantee**: These empirical results represent the evasion rate found by a specific black-box optimizer within a fixed budget; they do not represent an architecture-wide worst-case robustness limit.

## 13. Claim Assessment
**Question**: Does P5 establish stronger evidence about Option C under a query-based black-box threat model?

**VERDICT: SUPPORTED**

**Strongest supported claim**:
*"The frozen Option C ensemble was evaluated under a genuine query-based black-box NES attack with an $L_\infty$ budget of $\epsilon=0.10$ and fixed query budgets up to 500. Under this evaluated threat model, the observed strict ASR (excluding clean errors) reached up to 36.1% on NF-ToN-IoT-v2. These results establish that the previously employed neural-gradient transfer threat model substantially understated the ensemble's empirical evasion risk to direct black-box optimization."*

## 14. Reproducibility
- Attack executed via: `scripts/claim_upgrade/05_blackbox_attack_batched.py`
- Output artifacts: `reports/claim_upgrade/05_blackbox_attack.json`
- Attack seed: Fixed at `42`.
