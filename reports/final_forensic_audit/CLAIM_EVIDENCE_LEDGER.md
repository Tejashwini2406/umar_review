# CLAIM EVIDENCE LEDGER

| Section | Claim Text in Manuscript | Implementation Evidence | Experimental Evidence | Status |
| :--- | :--- | :--- | :--- | :--- |
| Abstract / System | Option C combines RF (0.7) + MLP (0.3) | `src/iot_ids/models/ensemble.py` | Fusion implementation exists; related historical benchmark claims are withdrawn | IMPLEMENTATION ONLY |
| Feature Rep | Standardized 21-feature statistical profile | `src/iot_ids/features/canonical/flow_builder.py` | Feature schema exists; affected materialized datasets and model results require regeneration | IMPLEMENTATION ONLY |
| Feature Categories | 5 basic flow, 6 protocol flags, 5 temporal, 5 behavioral | `src/iot_ids/features/canonical/schema.py` | Exactly 5 + 6 + 5 + 5 = 21 feature schema | VERIFIED |
| Datasets | Edge-IIoTset, NF-ToN-IoT-v2, ToN-IoT, CICIoT2023 | `data/processed/` | Stage 3 uses up to 7,000 class-quota samples per dataset; CICIoT2023 sample can be sourced entirely from official `test.csv` | INVALIDATED — rebuild splits |
| Split | 60/20/20 chronological split | `scripts/03_materialize_and_audit_features.py` | CIC uses official source partitions; others use seeded stratified row splits; old artifacts have not been regenerated | IMPLEMENTED — regenerate artifacts |
| Detection | Mean ROC-AUC 0.9970 (Option C) | `reports/golden_run_manifest.json` | Historical values were computed from Stage 3 artifacts with invalid split provenance | WITHDRAWN — retrain and reevaluate |
| Adversarial | NF-ToN-IoT-v2 PGD-10 Option C ASR = 4.0% | `reports/golden_run_manifest.json` | Historical value; current evaluation artifacts do not establish a clean independent test protocol | WITHDRAWN — rerun after split repair |
| Adversarial Scope | PGD-10 attacks generated using neural stream gradients | `final_ieee_paper/main.tex` | Attack implementation exists; old ASR results do not establish corrected-split performance | IMPLEMENTATION ONLY — rerun |
| Privacy Scope | DP-SGD applies exclusively to neural stream | `src/iot_ids/privacy/dp_optimizer.py` | DP-SGD applies to MLP weights only, not RF tree construction | IMPLEMENTATION ONLY |
| Privacy Accounting | (eps=2.37, delta=10^-5) at sigma=1.0 | `reports/golden_run_manifest.json` | Historical accountant result; utility and attack metrics require corrected-split reruns | ACCOUNTING ONLY — revalidate utility |
| XAI Methodology | Weighted Component Attribution Aggregation | `src/iot_ids/xai/` | Attribution code exists; historical model explanations are not current evidence | IMPLEMENTATION ONLY — rerun if retained |
| XAI Attributions | Edge-IIoTset historical top-feature values | `reports/xai/xai_evidence_summary.json` | Values were produced by historical artifacts and are withdrawn | WITHDRAWN |
| Runtime Throughput | 16,505 samples/s at batch size N=1024 | `reports/runtime/runtime_overhead_summary.json` | Historical detector-only benchmark; does not include BCL cost | WITHDRAWN — rerun BCL runtime |
| Hardware Testbed | Socket daemon implemented; hardware testbed is future work | `src/iot_ids/inference/daemon.py` | Socket daemon exists; physical 10Gbps hardware testbed declared in Limitations matrix | VERIFIED |
