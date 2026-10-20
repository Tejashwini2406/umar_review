# Historical Numerical Ledger — Withdrawn

> This legacy ledger contains withdrawn experiment and privacy claims. It must not be cited or used to rebuild a manuscript. The word “authoritative” in the historical filename is obsolete; current status is in [FINAL_IMPLEMENTATION_STATUS.md](../../FINAL_IMPLEMENTATION_STATUS.md).
**Date:** 2026-10-05 06:07:53
**Golden Release Commit:** `c8ffa15f03e61fe601994bf53396b8614d9d3596`
**Verification Status Legend:** **VERIFIED** (Code/checkpoint loaded & checked), **REPRODUCED** (Script recomputed value), **DERIVED** (Mathematically calculated from primary values).

## 1. Primary Dataset & Clean Detection Metric Ledger

| Dataset | N_train | N_val | N_test | N_attack | RF Clean AUC [95% CI] | Std MLP AUC | Rob MLP AUC | Option C Clean AUC [95% CI] | Option C Macro F1 [95% CI] | Verification Status | Source File |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Edge-IIoTset** | 4200 | 1400 | 1400 | 1000 | 0.9995 0.9995 [0.9988, 0.9999] | 0.8443 | 0.8002 | **0.9988** 0.9988 [0.9968, 0.9999] | 0.9751 0.9751 [0.9657, 0.9836] | **REPRODUCED** | `reports/golden_run_manifest.json` & `table_statistical_confidence_intervals.json` |
| **NF-ToN-IoT-v2** | 4200 | 1400 | 1400 | 1000 | 0.9940 0.9940 [0.9896, 0.9978] | 0.8398 | 0.8299 | **0.9927** 0.9927 [0.9853, 0.9981] | 0.9285 0.9285 [0.9134, 0.9427] | **REPRODUCED** | `reports/golden_run_manifest.json` & `table_statistical_confidence_intervals.json` |
| **ToN-IoT** | 4200 | 1400 | 1400 | 1000 | 1.0000 1.0000 [1.0000, 1.0000] | 0.7745 | 0.8063 | **1.0000** 1.0000 [1.0000, 1.0000] | 1.0000 1.0000 [1.0000, 1.0000] | **REPRODUCED** | `reports/golden_run_manifest.json` & `table_statistical_confidence_intervals.json` |
| **CICIoT2023** | 4200 | 1400 | 1400 | 1000 | 0.9968 0.9968 [0.9946, 0.9986] | 0.9776 | 0.9884 | **0.9965** 0.9965 [0.9943, 0.9983] | 0.9292 0.9292 [0.9121, 0.9434] | **REPRODUCED** | `reports/golden_run_manifest.json` & `table_statistical_confidence_intervals.json` |
| **Aggregate Mean** | -- | -- | -- | -- | **0.9976** | **0.8590** | **0.8562** | **0.9970** | -- | **DERIVED** | Calculated arithmetic mean across 4 datasets |

---

## 2. Adversarial Robustness & Adaptive Attack Ledger

| Dataset | Std MLP PGD-10 ASR | Rob MLP PGD-10 ASR | Option C Baseline PGD-10 ASR [95% CI] | Adaptive Surrogate ASR | $\Delta$ ASR (pp) | Surrogate Val $R^2$ | Verification Status | Source File |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Edge-IIoTset** | 0.0% | 0.0% | **3.8%** 0.0380 [0.0260, 0.0510] | **11.8%** | +8.0 pp | -0.5944 | **REPRODUCED** | `reports/adversarial/adaptive_attack_results.json` |
| **NF-ToN-IoT-v2** | 53.8% | 48.0% | **4.0%** 0.0400 [0.0290, 0.0520] | **5.9%** | +1.9 pp | 0.9477 | **REPRODUCED** | `reports/adversarial/adaptive_attack_results.json` |
| **ToN-IoT** | 2.3% | 2.5% | **0.0%** 0.0000 [0.0000, 0.0000] | **0.0%** | +0.0 pp | 0.0362 | **REPRODUCED** | `reports/adversarial/adaptive_attack_results.json` |
| **CICIoT2023** | 2.3% | 1.3% | **1.2%** 0.0120 [0.0060, 0.0190] | **1.1%** | -0.1 pp | 0.5357 | **REPRODUCED** | `reports/adversarial/adaptive_attack_results.json` |

---

## 3. Fusion Weight Ablation Sweep Ledger ($w \in [0.0, 1.0]$)

| RF Fusion Weight ($w$) | MLP Weight ($1-w$) | Mean Clean ROC-AUC | Mean PGD-10 ASR (%) | Verification Status | Source File |
|:---:|:---:|:---:|:---:|:---:|:---|
| 0.0 | 1.0 | 0.8562 | 13.0% | **REPRODUCED** | `reports/tables/fusion_ablation_results.json` |
| 0.1 | 0.9 | 0.9547 | 12.8% | **REPRODUCED** | `reports/tables/fusion_ablation_results.json` |
| 0.2 | 0.8 | 0.9820 | 2.5% | **REPRODUCED** | `reports/tables/fusion_ablation_results.json` |
| 0.3 | 0.7 | 0.9914 | 2.3% | **REPRODUCED** | `reports/tables/fusion_ablation_results.json` |
| 0.4 | 0.6 | 0.9943 | 2.1% | **REPRODUCED** | `reports/tables/fusion_ablation_results.json` |
| 0.5 | 0.5 | 0.9956 | 1.7% | **REPRODUCED** | `reports/tables/fusion_ablation_results.json` |
| 0.6 | 0.4 | 0.9958 | 1.5% | **REPRODUCED** | `reports/tables/fusion_ablation_results.json` |
| 0.7 | 0.3 | 0.9970 | 2.2% | **REPRODUCED** | `reports/tables/fusion_ablation_results.json` |
| 0.8 | 0.2 | 0.9971 | 6.5% | **REPRODUCED** | `reports/tables/fusion_ablation_results.json` |
| 0.9 | 0.1 | 0.9973 | 7.4% | **REPRODUCED** | `reports/tables/fusion_ablation_results.json` |
| 1.0 | 0.0 | 0.9976 | 8.6% | **REPRODUCED** | `reports/tables/fusion_ablation_results.json` |

---

## 4. Differential Privacy System Ledger

| Parameter | Value | Verification Status | Source Code / Report Location |
|---|---|:---:|:---|
| **Epsilon ($arepsilon$)** | `2.37` (audited) | **VERIFIED** | `reports/privacy/privacy_evidence_summary.json` (N_steps=660, q=0.0152) |
| **Delta ($\delta$)** | `1e-5` | **VERIFIED** | `iot_ids/privacy/dp_sgd.py` |
| **Noise Multiplier ($\sigma$)** | `1.0` | **VERIFIED** | `iot_ids/privacy/dp_sgd.py` (DPConfig defaults) |
| **Clipping Norm ($C$)** | `1.0` | **VERIFIED** | `iot_ids/privacy/dp_sgd.py` (DPConfig max_grad_norm) |
| **Batch Size** | `64` | **VERIFIED** | `iot_ids/privacy/dp_sgd.py` |
| **Accountant Type** | `Opacus PRV Accountant` | **VERIFIED** | `iot_ids/privacy/dp_sgd.py` |
| **Privacy Scope** | `Neural Branch Only` | **VERIFIED** | Paper line 95 & `iot_ids/privacy/dp_sgd.py` |

---

## 5. Host Runtime Benchmark Ledger

| Metric | Value | Verification Status | Source File / Scope Boundary |
|---|---|:---:|:---|
| **Throughput (Batch N=1024)** | `16,504.7 samples/sec` | **VERIFIED** | `reports/final_forensic_audit/controlled_runtime_results.json` |
| **Per-Sample Latency (N=1024)** | `0.0606 ms/sample` | **VERIFIED** | `reports/final_forensic_audit/controlled_runtime_results.json` |
| **Single-Sample Latency (N=1)** | `69.40 ms` | **VERIFIED** | `reports/final_forensic_audit/controlled_runtime_results.json` |
| **Benchmark Scope** | `Single-CPU Host Classifier Inference Only` | **VERIFIED** | Excludes PCAP capture, flow builder, scaling, XAI, network I/O |

---

## 6. XAI Attack-Family Feature Attribution Ledger

| Dataset | Attack Category | Sample Count ($N$) | Top Feature 1 (Attr) | Top Feature 2 (Attr) | Top Feature 3 (Attr) | Verification Status |
|---|---|:---:|---|---|---|:---:|
| **Edge-IIoTset** | Injection | 1000 | `behavioral_unanswered_ratio` (0.219) | `behavioral_port_entropy` (0.189) | `behavioral_dst_diversity` (0.137) | **REPRODUCED** |
| **NF-ToN-IoT-v2** | Scanning | 1000 | `mean_pkt_size` (0.138) | `temporal_flow_rate_ewma` (0.119) | `flow_bytes_per_sec` (0.114) | **REPRODUCED** |
| **ToN-IoT** | Backdoor | 1000 | `behavioral_src_activity_ewma` (0.303) | `temporal_syn_rate_ewma` (0.153) | `flow_duration` (0.111) | **REPRODUCED** |
| **CICIoT2023** | DDoS | 770 | `behavioral_src_activity_ewma` (0.241) | `mean_pkt_size` (0.137) | `temporal_iat_mean` (0.124) | **REPRODUCED** |
| **CICIoT2023** | DoS | 207 | `behavioral_src_activity_ewma` (0.243) | `temporal_iat_mean` (0.126) | `flow_duration` (0.123) | **REPRODUCED** |
| **CICIoT2023** | MITM *(N<50 Small)* | 11 | `behavioral_src_activity_ewma` (0.244) | `mean_pkt_size` (0.169) | `temporal_iat_mean` (0.120) | **REPRODUCED** |
| **CICIoT2023** | Scanning *(N<50 Small)* | 9 | `behavioral_src_activity_ewma` (0.230) | `flow_duration` (0.176) | `mean_pkt_size` (0.123) | **REPRODUCED** |
| **CICIoT2023** | Attack *(N<50 Small)* | 3 | `flow_duration` (0.275) | `behavioral_src_activity_ewma` (0.191) | `flow_pkts_per_sec` (0.116) | **REPRODUCED** |

---

## 7. Audit of Unverified or Misleading Claims in Draft Text

| Draft Paper Claim | Current Status | Forensic Evidence | Corrected Verified Replacement |
|---|:---:|---|---|
| *'Option C improves clean detection accuracy'* | **REVISE** | Pure RF mean AUC (0.9976) > Option C mean AUC (0.9970). | *'Option C preserves near-RF clean detection performance (0.9970 vs 0.9976)'* |
| *'Option C provides white-box robust defense'* | **REVISE** | Non-differentiable RF stream; Phase 5 adaptive surrogate ASR rises to 11.8% on Edge. | *'Under a surrogate-based adaptive attack, Option C evasion increases to 11.8% on Edge-IIoTset'* |
| *'0.7/0.3 is the globally optimal weight'* | **REVISE** | Weight w=0.6 achieves lowest mean PGD ASR (1.5%). | *'0.7/0.3 is a high-clean-performance operating point with a favorable robustness tradeoff'* |
| *'SHAP explains the fused Option C model'* | **REVISE** | Attribution formula is 0.7 RF_norm + 0.3 MLP_norm. | *'Delivered via Weighted Component Attribution Aggregation (0.7 RF + 0.3 MLP)'* |
| *'16,505 samples/sec line-rate throughput'* | **REVISE** | Measures CPU classifier inference latency only. | *'Single-CPU host classifier inference throughput reaches 16,505 samples/sec at batch N=1024'* |