# FINAL END-TO-END RELEASE VERIFICATION REPORT

**Project:** Industrial IIoT Intrusion Detection System (Option C 21-Feature Architecture)  
**Date:** October 18, 2026
**Final Release Status:** **NOT READY FOR RESEARCH SUBMISSION**

The operational test suite passes. Corrected split manifests, bounded corrected datasets, retrained BCL baseline artifacts, and exploratory BCL evaluations now exist locally for all four datasets. The evaluations do not establish confirmatory research evidence: ToN-IoT's score was revised after test inspection, other datasets show weak utility or order sensitivity, and three datasets use row-level random rather than host-, campaign-, or time-disjoint splits. Do not cite historical metrics or current BCL results as confirmatory.

---

## 1. Environment & Packaging Verification

- **Python Version:** `3.13.1` (Primary environment: `.\.venv\`)
- **Package Installation:** Package dependencies (`numpy`, `pandas`, `pyarrow`, `scikit-learn`, `scipy`, `joblib`, `pydantic`, `PyYAML`, `matplotlib`, `seaborn`, `psutil`, `pypdf`, `scapy`, `torch`, `shap`, `opacus`, `streamlit`) are installed in `.\.venv\`.
- **`iot_ids` Import Verification:** `import iot_ids` works when running via `pytest` (using `pythonpath = ["src"]` in `pyproject.toml`) or when inserting `src` into `sys.path`. *Note:* Running `pip install -e .` in the virtual environment completes global CLI installation without manual path insertion.
- **scikit-learn Artifact Warning Audit:** The current full-suite run reports `InconsistentVersionWarning` for model artifacts produced with scikit-learn 1.7.2 and 1.9.0 while the selected environment uses 1.9.1. These are compatibility warnings, not proof of cross-version safety; reproducibility-sensitive results should be rerun with the artifact-producing version.

---

## 2. Complete Pytest Suite Verification (`python -m pytest -q`)

Run from the repository root using the project's Python 3.13.1 virtual environment after the BoT-IoT adapter and registry changes:

- **PASSED:** 161
- **SKIPPED:** 8
- **FAILED:** 0
- **WARNINGS:** 148
- **Runtime:** 34.27 seconds

Focused BCL and corrected-split tests: **21 passed**. BoT-IoT adapter, source-split materialization, and data-layer tests: **21 passed**. The skips are conditional checks for optional dependencies, locally available datasets/deployment artifacts, and legacy publication artifacts. Warnings include scikit-learn model-artifact version mismatches, a deprecated feature-builder import, and SHAP pending deprecations; these remain reproducibility/maintenance caveats, not test failures.

### Corrected Sample Runs (Exploratory, Not Confirmatory)

The old Stage 3 outputs were created by first-record class quotas and could source all CICIoT2023 splits from its official test partition. Corrected sample runs have now been materialized: CICIoT2023 uses separate official source partitions and has 50,000 sampled flows (30,000/10,000/10,000); the other datasets use 10,000-flow seeded stratified row-level random splits with bounded reservoir sampling. Each run writes a split manifest. State-derived temporal/behavioral features are disabled except for Edge-IIoTset because the other adapters synthesize timestamps and/or hosts.

Corrected RF/MLP/autoencoder artifacts and BCL evaluation outputs have been generated for all four local datasets. These are bounded samples, not full-source evaluations. ToN-IoT is exploratory because its BCL score was revised after test inspection; Edge-IIoTset has substantial input-order sensitivity; NF-ToN-IoT-v2 and CICIoT2023 have low BCL attack detection at the calibrated threshold. For datasets without official partitions, row-level random splitting does not establish host-, campaign-, or time-disjoint generalization.

---

## 3. CLI Demonstration Verification (`python demo/cli_demo.py`)

- **Execution Status:** **VERIFIED (100% SUCCESS)**
- **Datasets Tested:** All 4 datasets (`Edge-IIoTset`, `NF-ToN-IoT-v2`, `ToN-IoT`, `CICIoT2023`).
- **Inference Verification:**
  - Evaluates 21 continuous features per flow
  - Executes preprocessor, Random Forest (100 trees), and PyTorch Robust MLP
  - Computes probability fusion: $P_{\text{OptionC}} = 0.7 P_{\text{RF}} + 0.3 P_{\text{MLP}}$
  - Applies $0.50$ decision threshold $\to$ `ATTACK` / `BENIGN` classification
  - Calculates local XAI attributions live from model outputs
- **Hardcoding Check:** Outputs are dynamically computed from `InferenceEngine` execution on parquet test splits.

---

## 4. Streamlit Application Audit (`demo/app.py`)

- **Startup Verification:** Starts without import or runtime errors.
- **Mode Classification:**

| Demonstration Mode | Classification | Operational Status |
|---|---|---|
| **📊 Held-Out Flow Replay** | **LIVE COMPUTATION** | Evaluates locally available historical test artifacts through `InferenceEngine`, RF, PyTorch MLP, and XAI explainer; not valid current paper evidence. |
| **🔌 PCAP / Packet Replay** | **LIVE COMPUTATION** | Parses Scapy packets from `.pcap` live, aggregates 5-tuple bidirectional flows, computes 21 features, and outputs alerts. |
| **🎯 Controlled Test Flow** | **LIVE COMPUTATION** | Evaluates preset or user-configured 21-feature vectors live. |
| **🛡️ Adversarial Evasion (PGD-10)** | **LIVE COMPUTATION** | Computes PGD-10 gradient perturbations against the neural stream live ($\epsilon=0.10, \alpha=0.025$, 10 steps) with protocol indicators (indices 7–10) frozen. |
| **💡 XAI Attribution Analysis** | **LIVE COMPUTATION** | Computes RF SHAP and PyTorch gradient attributions live using $0.7 \text{RF}_{\text{norm}} + 0.3 \text{MLP}_{\text{norm}}$. |
| **📈 Research Results Dashboard** | **PRECOMPUTED HISTORICAL RESULT** | Displays withdrawn paper/golden-run benchmarks; do not cite as current evidence. |

---

## 5. End-to-End Packet Pipeline Evidence

- **Test Input:** `data/sample_reproduce_stream.pcap`
- **Execution Log:**
  - **Packets Ingested:** 4 packets
  - **Flows Constructed:** 1 bidirectional flow
  - **5-Tuple:** `10.0.0.5:80 -> 192.168.1.105:54321 (tcp)`
  - **21-Feature Vector:** 21 continuous features computed
  - **$P_{\text{RF}}$:** `0.6400`
  - **$P_{\text{MLP}}$:** `0.9996`
  - **$P_{\text{OptionC}}$:** `0.7479`
  - **Mathematical Verification:** $0.7(0.64) + 0.3(0.999561) = 0.448 + 0.299868 = 0.747868$ (**100% Exact Match**)
  - **Final Classification:** `ATTACK` (since $0.7479 \ge 0.50$)

---

## 6. PGD-10 Adversarial Attack Verification

- **Execution Parameters:** $\epsilon = 0.10, \alpha = 0.025, \text{steps} = 10$.
- **Feature Mask Check:** `continuous_mask[7:11] = 0.0` strictly freezes protocol indicators (`proto_tcp`, `proto_udp`, `proto_icmp`, `proto_other`). Verified zero delta (`[0. 0. 0. 0.]`) on indices 7–10.
- **Gradient Origin & Ensemble Evaluation:** Attack gradients originate from the robust neural stream (`engine.robust_mlp_model`); perturbed samples are subsequently evaluated through the Option C ensemble ($P_{\text{OptionC}} = 0.7 P_{\text{RF}} + 0.3 P_{\text{MLP}}$).
- **Research-Evidence Status:** The mask implementation check is a software fact only. The reported golden-run ASR values are withdrawn because the underlying data split provenance is invalid for independent evaluation; they are not current research evidence.

---

## 7. Historical Metrics Reconciliation (Withdrawn)

- **Historical Artifact:** `reports/golden_run_manifest.json` generated by an earlier `scripts/run_golden_pipeline.py` run.
- **Historical Consistency:** `final_ieee_paper/main.tex` Table I matches `golden_run_manifest.json` to 4 decimal places:
  - Edge-IIoTset: RF 0.9995, Std MLP 0.8443, Rob MLP 0.8002, Option C **0.9988**
  - NF-ToN-IoT-v2: RF 0.9940, Std MLP 0.8398, Rob MLP 0.8299, Option C **0.9927**
  - ToN-IoT: RF 1.0000, Std MLP 0.7745, Rob MLP 0.8063, Option C **1.0000**
  - CICIoT2023: RF 0.9968, Std MLP 0.9776, Rob MLP 0.9884, Option C **0.9965**
  - Mean: RF 0.9976, Std MLP 0.8590, Rob MLP 0.8562, Option C **0.9970**
- **Current Status:** File-to-file consistency does not establish split independence or scientific validity. These values are withdrawn and must not be reused as publication results; the bounded corrected BCL results are tracked separately above and remain exploratory.

---

## 8. Differential Privacy & XAI Verification

- **Differential Privacy Scope:** DP-SGD implementation applies to the neural stream. The historical accountant configuration ($C=1.0, \delta=10^{-5}, \text{batch}=64, \text{epochs}=10, \text{steps}=660, q=64/4200$) reported $\sigma=1.0 \implies \varepsilon \approx 2.37$; this accounting value alone does not establish model utility or a BCL result.
- **Historical XAI Method:** The code implements weighted attribution aggregation $0.7 \times \text{RF}_{\text{norm}} + 0.3 \times \text{MLP}_{\text{norm}}$; historical model-specific outputs are withdrawn.
- **Withdrawn Edge-IIoTset Values:**
  - `temporal_flow_rate_ewma` $\approx 0.310$
  - `behavioral_port_entropy` $\approx 0.242$
  - `behavioral_unanswered_ratio` $\approx 0.154$
  - `behavioral_dst_diversity` $\approx 0.146$
  - These historical values are not evidence for the selected BCL contribution.

---

## 9. Paper-vs-Code Consistency Summary

The prior claim that all paper methodology claims were verified is withdrawn. Model architecture and software-path checks do not validate data provenance or scientific results; the split and timestamp limitations above invalidate the current evaluation claims pending reruns.

---

## 10. Software Demonstration Notes

The current demo can show inference, packet parsing, and the existing model interfaces. Its saved datasets, models, dashboards, and historical PGD/XAI outputs are not valid evidence for the selected paper. Do not present old ROC-AUC, ASR, privacy-utility, attribution, or throughput values as current results. The BCL experiment runner is separate from the Streamlit demo. Corrected bounded data, models, and exploratory evaluations exist for the four datasets currently in the workspace. The official BoT-IoT listing exposes the 5% subset, but its file-download endpoint returned HTTP 403 here; BoT-IoT has no local files, split, model, or evaluation.

---

## 11. Final Verdict

**NOT READY FOR RESEARCH SUBMISSION**

The software test suite passes, and corrected bounded materialization, model training, and BCL evaluation have run for the four local datasets. A locked BoT-IoT source-file split and its materializer/trainer/evaluator path are implemented and unit-tested, but actual data access is still blocked by the official SharePoint endpoint's HTTP 403. The project is **not ready for research submission**: current results are exploratory, no BoT-IoT split/model/evaluation has run, adaptive-evasion analysis is absent, and there is no venue-appropriate manuscript or Open Science Appendix. The existing IEEE draft and all historical empirical metrics must not be submitted.
