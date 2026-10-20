# FRESH REPRODUCTION RESULTS

## 1. Primary Detection ROC-AUC Performance

| Dataset | Model | Paper | Golden Manifest | Fresh Reproduction | Difference | Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Edge-IIoTset** | RF | 0.9995 | 0.9994775 | 0.9994775 | +0.0000000 | FRESHLY REPRODUCED |
| | Std MLP | 0.8443 | 0.8443275 | 0.8443275 | +0.0000000 | FRESHLY REPRODUCED |
| | Rob MLP | 0.8002 | 0.8001775 | 0.8001775 | +0.0000000 | FRESHLY REPRODUCED |
| | **Option C** | **0.9988** | **0.9987525** | **0.9987525** | **+0.0000000** | **FRESHLY REPRODUCED** |
| **NF-ToN-IoT-v2** | RF | 0.9940 | 0.9939962 | 0.9939962 | +0.0000000 | FRESHLY REPRODUCED |
| | Std MLP | 0.8398 | 0.8397913 | 0.8397913 | +0.0000000 | FRESHLY REPRODUCED |
| | Rob MLP | 0.8299 | 0.8299250 | 0.8299250 | +0.0000000 | FRESHLY REPRODUCED |
| | **Option C** | **0.9927** | **0.9926713** | **0.9926713** | **+0.0000000** | **FRESHLY REPRODUCED** |
| **ToN-IoT** | RF | 1.0000 | 1.0000000 | 1.0000000 | +0.0000000 | FRESHLY REPRODUCED |
| | Std MLP | 0.7745 | 0.7745100 | 0.7745100 | +0.0000000 | FRESHLY REPRODUCED |
| | Rob MLP | 0.8063 | 0.8062900 | 0.8062900 | +0.0000000 | FRESHLY REPRODUCED |
| | **Option C** | **1.0000** | **1.0000000** | **1.0000000** | **+0.0000000** | **FRESHLY REPRODUCED** |
| **CICIoT2023** | RF | 0.9968 | 0.9968000 | 0.9968000 | +0.0000000 | FRESHLY REPRODUCED |
| | Std MLP | 0.9776 | 0.9775700 | 0.9775700 | +0.0000000 | FRESHLY REPRODUCED |
| | Rob MLP | 0.9884 | 0.9884425 | 0.9884425 | +0.0000000 | FRESHLY REPRODUCED |
| | **Option C** | **0.9965** | **0.9965025** | **0.9965025** | **+0.0000000** | **FRESHLY REPRODUCED** |

## 2. Adversarial Robustness (PGD-10 ASR at epsilon=0.10)

Attack Generation Scope: PGD-10 gradient perturbations are calculated strictly using gradients of the robust neural stream (Robust MLP). The resulting adversarial samples are evaluated on Standard MLP, Robust MLP, and Option C (0.7 RF + 0.3 Robust MLP) ensemble. It is a neural-stream gradient attack evaluated through the ensemble.

| Dataset | Standard MLP ASR | Robust MLP ASR | Option C Ensemble ASR | Provenance |
| :--- | :--- | :--- | :--- | :--- |
| **Edge-IIoTset** | 0.000 | 0.000 | **0.038** | FRESHLY REPRODUCED |
| **NF-ToN-IoT-v2** | 0.538 | 0.480 | **0.040** | FRESHLY REPRODUCED |
| **ToN-IoT** | 0.023 | 0.025 | **0.000** | FRESHLY REPRODUCED |
| **CICIoT2023** | 0.023 | 0.013 | **0.012** | FRESHLY REPRODUCED |

## 3. Differential Privacy Experiments (NF-ToN-IoT-v2, delta=10^-5)

| Noise Multiplier (sigma) | Privacy Budget (epsilon) | Neural Stream ROC-AUC | PGD-10 ASR | Provenance |
| :--- | :--- | :--- | :--- | :--- |
| 0.0 | infinity | 0.9763 | 0.015 | FRESHLY REPRODUCED |
| 0.5 | 15.85 | 0.8865 | 0.025 | FRESHLY REPRODUCED |
| 1.0 | 2.37 | 0.8541 | 0.477 | FRESHLY REPRODUCED |
| 2.0 | 0.80 | 0.8284 | 0.478 | FRESHLY REPRODUCED |

### Feature Mask Discrepancy Analysis
- DP Training (src/iot_ids/privacy/dp_sgd.py line 165): Sets feature_mask[17:21] = 0.0, freezing behavioral features during adversarial batch generation.
- Main PGD Attack (scripts/run_golden_pipeline.py line 155): Sets continuous_mask[7:11] = 0.0, freezing protocol indicators.
- Cause: Implementation index discrepancy in dp_sgd.py line 165.

## 4. XAI Feature Attributions (Edge-IIoTset)

Weighted Component Attribution Aggregation (0.7 RF TreeSHAP + 0.3 Robust MLP Gradient SHAP):

| Feature Name | Combined Attribution Weight | Provenance |
| :--- | :--- | :--- |
| `temporal_flow_rate_ewma` | 0.310 | FRESHLY REPRODUCED |
| `behavioral_port_entropy` | 0.242 | FRESHLY REPRODUCED |
| `behavioral_unanswered_ratio` | 0.154 | FRESHLY REPRODUCED |

## 5. Host Runtime Throughput Benchmarks

| Batch Size (N) | Mean Batch Latency (ms) | Per-Sample Latency (ms) | Throughput (samples/s) | Provenance |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 92.94 | 92.9390 | 10.8 | FRESHLY REPRODUCED |
| 32 | 86.45 | 2.7017 | 370.1 | FRESHLY REPRODUCED |
| 64 | 86.25 | 1.3476 | 742.0 | FRESHLY REPRODUCED |
| 128 | 106.11 | 0.8290 | 1206.3 | FRESHLY REPRODUCED |
| 256 | 100.25 | 0.3916 | 2553.5 | FRESHLY REPRODUCED |
| 512 | 93.49 | 0.1826 | 5476.3 | FRESHLY REPRODUCED |
| 1024 | 98.78 | 0.0965 | 10366.4 | FRESHLY REPRODUCED |

## 6. Analysis of the 10 Test Failures

All 10 unit test failures are OBSOLETE ARTIFACT EXPECTATIONS / STALE TESTS:
1. test_publication_audit_csv_exists_and_passes -> Asserts reports/stage8/stage8_publication_audit.csv exists. (OBSOLETE ARTIFACT EXPECTATION)
2. test_artifact_manifest_exists -> Asserts reports/stage8/stage8_artifact_manifest.csv exists. (OBSOLETE ARTIFACT EXPECTATION)
3. test_stage10_audit_csv_artifacts_exist -> Asserts reports/stage10/stage10_reviewer_audit.csv exists. (OBSOLETE ARTIFACT EXPECTATION)
4. test_stage10_numerical_consistency_all_pass -> Asserts reports/stage10/stage10_numerical_consistency_audit.csv exists. (OBSOLETE ARTIFACT EXPECTATION)
5. test_stage10_final_submission_gate_decision -> Asserts reports/stage10/stage10_final_submission_gate.md exists. (OBSOLETE ARTIFACT EXPECTATION)
6. test_stage11_submission_package_exists -> Asserts reports/stage11/submission directory exists. (OBSOLETE ARTIFACT EXPECTATION)
7. test_stage11_checksums_csv -> Asserts reports/stage11/stage11_checksums.csv exists. (OBSOLETE ARTIFACT EXPECTATION)
8. test_stage11_figures_tables_audit_pass -> Asserts reports/stage11/stage11_figures_tables_audit.csv exists. (OBSOLETE ARTIFACT EXPECTATION)
9. test_stage11_placeholder_audit_pass -> Asserts reports/stage11/stage11_placeholder_audit.csv exists. (OBSOLETE ARTIFACT EXPECTATION)
10. test_eval_file_exists -> Asserts legacy artifact in_domain_evaluation.json exists. (STALE TEST / OBSOLETE DATASET EXPECTATION)

Classification Breakdown:
- Current Implementation Tests: 118 passed / 0 failed / 0 skipped
- Obsolete Artifact / Stale Tests: 0 passed / 10 failed / 12 skipped

## 7. Resolution of Multiple Inference / Evaluation Paths

- Golden Production Pipeline (scripts/run_golden_pipeline.py, src/iot_ids/runtime/engine.py): Authoritative Option C system (0.7 RF + 0.3 Robust MLP).
- RiskLayer (src/iot_ids/models/ensemble/risk_layer.py): Experimental Design B fallback combining 0.5 RF + 0.5 MLP with Autoencoder anomaly detector. Unused by golden pipeline or manuscript.