# INDEPENDENT FORENSIC VERIFICATION REPORT
**Date:** 2026-10-05 05:58:36
**Repository HEAD Commit:** `c8ffa15f03e61fe601994bf53396b8614d9d3596`
**Working Tree Clean:** `False`

## 1. Executive Summary & Phase Verdicts
| Phase | Description | Verdict | Primary Evidence |
| :---: | :--- | :---: | :--- |
| **Phase 0** | Repository Discovery & Checkpoints | **PASS** | 16 golden model SHA-256 hashes verified 100% identical. |
| **Phase 1** | Technical Consistency Audit | **PASS** | 21 canonical features, protocol mask 7:11, flow timeouts 15s/120s verified. |
| **Phase 2** | Fusion-Weight Ablation | **PASS** | Evaluated w in [0.0, 1.0]. $w=0.7$ confirmed as high-clean/favorable robustness operating point. |
| **Phase 3** | Bootstrap Confidence Intervals | **PASS** | $B=1000$, seed=42, 95% CIs verified for clean AUC, Macro F1, and malicious PGD ASR. |
| **Phase 4** | XAI Attack-Family & Local Case Studies | **PASS** | Categorical attributions and local flow case studies verified. |
| **Phase 5** | Adaptive Surrogate-Gradient Attack | **PASS** | Grey-box adaptive surrogate attack evaluated on actual frozen Option C. |
| **Phase 6** | Final Forensic Audit | **PASS** | 15/15 audit areas passed; zero checkpoint or golden manifest drift. |
| **Phase 7** | Paper Evidence Package | **PASS** | 6 complete evidence package files compiled in `reports/paper_evidence/`. |

## 2. Independent Clean Performance Recomputation Results
| Dataset | RF ROC-AUC | Standard MLP ROC-AUC | Robust MLP ROC-AUC | Option C ROC-AUC | RF Macro F1 | Option C Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Edge-IIoTset** | 0.9995 | 0.8443 | 0.8002 | 0.9988 | 0.9014 | 0.9751 |
| **NF-ToN-IoT-v2** | 0.9940 | 0.8398 | 0.8299 | 0.9927 | 0.9903 | 0.9285 |
| **ToN-IoT** | 1.0000 | 0.7745 | 0.8063 | 1.0000 | 1.0000 | 1.0000 |
| **CICIoT2023** | 0.9968 | 0.9776 | 0.9884 | 0.9965 | 0.9377 | 0.9292 |
| **Mean** | **0.9976** | **0.8590** | **0.8562** | **0.9970** | -- | -- |

*Recomputation Confirmation:* Pure RF clean mean AUC (**0.9976**) is slightly higher than Option C (**0.9970**). Paper text must NOT claim fusion improves clean detection.

## 3. Paper Claims Requiring Revision & Safe Replacements
| Section / Claim | Issue Identified | Safe Replacement Wording |
| :--- | :--- | :--- |
| **Abstract & Intro (Clean Accuracy)** | Claims fusion improves clean detection. | *'Option C preserves near-RF clean detection performance (0.9970 vs 0.9976) while enabling neural privacy and robustness studies.'* |
| **Section V-B (Threat Model)** | Labels attack as 'white-box ensemble attack'. | *'This is a surrogate-based adaptive attack, not a true differentiable white-box attack through the Random Forest.'* |
| **Section III-C (Privacy Scope)** | Ambiguous differential privacy scope. | *'Differential privacy is guaranteed exclusively on the neural stream branch (\varepsilon=2.37); the Random Forest classifier is non-private.'* |
| **Section III-D (XAI Definition)** | Describes XAI as SHAP for fused model. | *'Model explainability is delivered via Weighted Component Attribution Aggregation ($0.7 \text{RF} + 0.3 \text{MLP}$), not exact game-theoretic SHAP for the non-linear fused predictor.'* |
| **Section V-D (Runtime Scope)** | Labels throughput as line-rate 10Gbps. | *'Single-CPU host classifier inference throughput reaches 16,505 samples/sec at batch size N=1024, excluding network ingress and feature extraction.'* |

## 4. Final Verdict & Publication Readiness
### VERDICT: GO (READY FOR IEEE PAPER REWRITE)
- **Repository Integrity:** 100% Verified.
- **Model Checkpoints:** 100% Unaltered.
- **Evidence Completeness:** 100% Traceable.
- **Next Step:** Proceed to update `final_ieee_paper/main.tex` according to the evidence package in `reports/paper_evidence/`.