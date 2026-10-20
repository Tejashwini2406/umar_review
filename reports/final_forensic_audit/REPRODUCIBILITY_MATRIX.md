# REPRODUCIBILITY MATRIX

| Experiment / Claim | Implementation Location | Command / Script | Dataset | Seed / Config | Expected Value | Fresh Reproduction Value | Status | Evidence Location |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Option C ROC-AUC (Edge-IIoTset) | `src/iot_ids/models/` | Historical run | Edge-IIoTset | 42 | 0.9987525 | 0.9987525 | WITHDRAWN — rebuild split protocol | `reports/golden_run_manifest.json` |
| Option C ROC-AUC (NF-ToN-IoT-v2) | `src/iot_ids/models/` | Historical run | NF-ToN-IoT-v2 | 42 | 0.9926712 | 0.9926712 | WITHDRAWN — rebuild split protocol | `reports/golden_run_manifest.json` |
| Option C ROC-AUC (ToN-IoT) | `src/iot_ids/models/` | Historical run | ToN-IoT | 42 | 1.0000000 | 1.0000000 | WITHDRAWN — rebuild split protocol | `reports/golden_run_manifest.json` |
| Option C ROC-AUC (CICIoT2023) | `src/iot_ids/models/` | Historical run | CICIoT2023 | 42 | 0.9965025 | 0.9965025 | INVALIDATED — sample can derive from official test partition | `reports/golden_run_manifest.json` |
| Option C Mean ROC-AUC | `src/iot_ids/models/` | Historical run | 4 Datasets | 42 | 0.9969816 | 0.9969816 | WITHDRAWN — rebuild split protocol | `reports/golden_run_manifest.json` |
| NF-ToN-IoT-v2 Std MLP PGD-10 ASR | `src/iot_ids/adversarial/` | Historical PGD-10 eps=0.1 | NF-ToN-IoT-v2 | 42 | 0.538 | 0.538 | WITHDRAWN — rerun after split repair | `reports/golden_run_manifest.json` |
| NF-ToN-IoT-v2 Rob MLP PGD-10 ASR | `src/iot_ids/adversarial/` | Historical PGD-10 eps=0.1 | NF-ToN-IoT-v2 | 42 | 0.480 | 0.480 | WITHDRAWN — rerun after split repair | `reports/golden_run_manifest.json` |
| NF-ToN-IoT-v2 Option C PGD-10 ASR | `src/iot_ids/adversarial/` | PGD-10 eps=0.1 | NF-ToN-IoT-v2 | 42 | 0.040 | 0.040 | WITHDRAWN — rerun after split repair | `reports/golden_run_manifest.json` |
| DP Privacy Accounting (sigma=1.0) | `src/iot_ids/privacy/` | Opacus PRV Accountant | NF-ToN-IoT-v2 | delta=1e-5 | eps = 2.37 | eps = 2.37 | VERIFIED | `reports/golden_run_manifest.json` |
| DP Privacy Accounting (sigma=0.5) | `src/iot_ids/privacy/` | Opacus PRV Accountant | NF-ToN-IoT-v2 | delta=1e-5 | eps = 15.85 | eps = 15.85 | ACCOUNTING ONLY — revalidate utility | `reports/golden_run_manifest.json` |
| DP Privacy Accounting (sigma=2.0) | `src/iot_ids/privacy/` | Opacus PRV Accountant | NF-ToN-IoT-v2 | delta=1e-5 | eps = 0.80 | eps = 0.80 | ACCOUNTING ONLY — revalidate utility | `reports/golden_run_manifest.json` |
| XAI Edge-IIoTset Top Feature 1 | `src/iot_ids/xai/` | Historical Weighted Component Attribution | Edge-IIoTset | 0.7 RF + 0.3 MLP | temporal_flow_rate_ewma (0.310) | 0.310 | WITHDRAWN — historical model output | `reports/xai/xai_evidence_summary.json` |
| XAI Edge-IIoTset Top Feature 2 | `src/iot_ids/xai/` | Historical Weighted Component Attribution | Edge-IIoTset | 0.7 RF + 0.3 MLP | behavioral_port_entropy (0.242) | 0.242 | WITHDRAWN — historical model output | `reports/xai/xai_evidence_summary.json` |
| XAI Edge-IIoTset Top Feature 3 | `src/iot_ids/xai/` | Historical Weighted Component Attribution | Edge-IIoTset | 0.7 RF + 0.3 MLP | behavioral_unanswered_ratio (0.154) | 0.154 | WITHDRAWN — historical model output | `reports/xai/xai_evidence_summary.json` |
| Pure Inference Throughput (N=1024) | `src/iot_ids/runtime/` | Historical detector-only benchmark | Batch 1024 | Single-CPU Host | ~16,505 samples/s | 16,504.7 samples/s | WITHDRAWN — benchmark BCL-inclusive path | `reports/runtime/runtime_overhead_summary.json` |
| Single-Sample Latency (N=1) | `src/iot_ids/runtime/` | Historical detector-only benchmark | Batch 1 | Single-CPU Host | 69.40 ms | 69.40 ms | WITHDRAWN — benchmark BCL-inclusive path | `reports/runtime/runtime_overhead_summary.json` |
