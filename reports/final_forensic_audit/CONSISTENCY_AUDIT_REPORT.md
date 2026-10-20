# Technical Consistency Audit Report
**Date:** 2026-10-04 20:37:52
**Repository Commit:** `c8ffa15f03e61fe601994bf53396b8614d9d3596`
**Target Document:** `final_ieee_paper/main.tex`

## Audit Executive Summary
- **Total Checked Items:** 19
- **PASS:** 18
- **FAIL:** 0
- **NOT VERIFIED:** 1

## Audit Item Ledger
| ID | Parameter / Claim | Status | Code Evidence | Paper Statement | Recommendation / Notes |
|---|---|---|---|---|---|
| 1 | 21-Feature Representation | **PASS** | 21 features in FEATURE_COLS | Mentions 21-feature statistical flow representation | Verified across pipeline code and paper abstract/Section III-A. |
| 2 | Feature Ordering & Protocol Mask | **PASS** | Order matches canonical list 0-20. Protocol mask 7:11 (proto_tcp, proto_udp, proto_icmp, proto_other) | Paper Section III-A lists features in 4 metric groups. | Protocol mask 7:11 verified in scripts/run_golden_pipeline.py. |
| 3 | Flow Timeout Parameters | **NOT VERIFIED** | inactivity_timeout=15.0s, max_flow_duration=120s in iot_ids/config.py and flow_builder.py | Paper describes 21 flow metrics but omits explicit 15s/120s timeout values in text. | Recommendation: Add explicit 15s inactivity and 120s max duration mention in Section III-A of paper. |
| 4 | Chronological 60/20/20 Split | **PASS** | generate_v3_splits.py enforces chronological 60/20/20 | Paper line 115 explicitly states 'standardized 60/20/20 chronological train/validation/test splits'. |  |
| 5 | Dataset Sample Counts | **PASS** | 7,000 flows total (4,200 train / 1,400 val / 1,400 test) per dataset | Paper Section IV-A lists 4,200 train / 1,400 validation / 1,400 test for each dataset. |  |
| 6 | State Reset at Split Boundaries | **PASS** | Flow generator resets state between train, val, and test splits | Paper line 115 explicitly states 'with detector state reset at split boundaries to prevent data leakage'. |  |
| 7 | Train-Only Scaler Fitting | **PASS** | scaler.fit_transform(X_tr); scaler.transform(X_va); scaler.transform(X_te) | Paper line 70 states 'clamped using robust scaler parameters fitted on benign training distributions'. |  |
| 8 | Random Forest Hyperparameters | **PASS** | n_estimators=100, max_depth=15, random_state=42 | Paper line 75 states 'ensemble of 100 decision trees'. (Paper omits max_depth=15). | Recommendation: Paper can optionally specify max_depth=15 in Section III-B. |
| 9 | Standard MLP Architecture | **PASS** | 21->128->64->32->1, BatchNorm, ReLU, Dropout(0.2), Adam lr=0.001, epochs=15, batch_size=256 | Paper line 76 states '21 -> 128 -> 64 -> 32 -> 1 with Batch Normalization, ReLU activations, and Dropout (p=0.2)'. |  |
| 10 | Robust MLP & PGD-7 Training | **PASS** | PGD-7, eps=0.10, alpha=0.025, adv_ratio=0.5 | Paper line 82-86 details PGD-7 training with epsilon=0.1, alpha=0.025, k=7. |  |
| 11 | Option C Probability Fusion | **PASS** | P_opt_c = 0.7 * p_rf + 0.3 * p_mlp_rob | Paper Equation (1): P_Option C = 0.7 * P_RF + 0.3 * P_MLP. |  |
| 12 | Differential Privacy Bounds & Scope | **PASS** | Neural stream only, C=1.0, sigma=1.0, batch_size=64, delta=1e-5, Opacus PRV accountant, eps=2.37 | Paper Section III-C states (eps=2.37, delta=10^-5)-DP under Opacus PRV accountant, neural stream exclusively. |  |
| 13 | XAI Weighted Component Attribution | **PASS** | 0.7 * norm(RF SHAP/MDI) + 0.3 * norm(MLP Grad) | Paper line 98-101 defines Weighted Component Attribution Aggregation with 0.7 RF + 0.3 MLP normalized weights. |  |
| 14 | PGD-10 Adversarial Evaluation | **PASS** | PGD-10, eps=0.10, alpha=0.025, 10 steps | Paper Section IV-B and Section V-B state PGD-10 attack at epsilon=0.1, alpha=0.025, 10 iterations. |  |
| 15 | Adversarial Evaluation Scope Clarification | **PASS** | Gradients computed on neural stream, perturbed samples evaluated on RF and Option C | Paper line 177-178 explicitly states PGD-10 perturbations were generated using neural stream gradients and evaluated on RF and Option C. |  |
| 16 | Runtime Latency & Scope | **PASS** | N=1024 -> 16,504.7 samples/s (0.0606 ms/sample); N=1 -> 69.40 ms. Classifier-only host benchmark. | Paper Section V-D reports single-CPU host throughput of 16,505 samples/sec (0.0606 ms/sample) at N=1024. |  |
| 17 | Dataset Names & Counts | **PASS** | Edge-IIoTset, NF-ToN-IoT-v2, ToN-IoT, CICIoT2023 (7,000 flows each) | Paper lists four datasets with 7,000 flows each. |  |
| 18 | Software & Framework Specifications | **PASS** | Python 3.12, PyTorch 2.x, scikit-learn, Opacus, joblib | Paper cites Opacus PRV accountant, PyTorch, and scikit-learn algorithms. |  |
| 19 | Golden Benchmark Numerical Reconciliation | **PASS** | Option C AUCs: Edge=0.9988, NF=0.9927, ToN=1.0000, CIC=0.9965, Mean=0.9970 | Paper Table I & Abstract report exact matching values: 0.9988, 0.9927, 1.0000, 0.9965 (Mean 0.9970). |  |

## Detailed Discrepancies & Recommendations
1. **Item 3 (Flow Timeout Parameters):** Code uses `inactivity_timeout=15.0s` and `max_flow_duration=120s`. Paper describes the 21 metrics but omits explicit 15s/120s numbers. *Recommendation:* Add explicit mention of 15s inactivity and 120s max flow window in Section III-A.
2. **Item 8 (Random Forest Max Depth):** Code trains RF with `max_depth=15`. Paper states '100 decision trees' but omits max_depth. *Recommendation:* Optionally add `max_depth=15` to Section III-B.

## Conclusion
The technical implementation and paper LaTeX source exhibit **100% numerical and structural alignment** across all primary benchmark metrics, dataset splits, neural architectures, privacy parameters, and evaluation protocols. Two minor non-critical parameter omissions (flow timeout values and RF max_depth) were identified for narrative refinement in the paper.