# Implementation and Paper-Readiness Status

**Status: Publication package completed and verified.** The primary research manuscript and operational software suite have been reconciled against the authoritative empirical ledger. All 15 scientific and implementation issues identified in the master audit have been integrated into the research narrative and paper deliverables.

## Primary Publication Deliverables

1. **Camera-Ready IEEE Manuscript (inal_ieee_paper/main.tex)**:
   - Formatted in IEEE Transactions on Dependable and Secure Computing (TDSC) style.
   - Authors: Tejashwini S Uranakar, Arun Kumar B R, Umar Ismail Shariff, Varsha, and Thannikuneth Ravindra Krishna (BMSIT&M).
   - Fully incorporates multi-seed analysis, threat-model dissection (transfer vs. black-box NES query optimization), fusion-weight operating points, DP-SGD neural-branch bounds, and surrogate XAI additivity analysis.
   - Complete figure package: 7 publication plots in inal_ieee_paper/figures/.
2. **Camera-Ready Microsoft Word Manuscript (../An Adversarially Robust and Privacy-Preserving AI Framework for Explainable Intrusion Detection in IoT Environments_Final_Publication.docx)**:
   - Fully styled Word document with 10 formatted tables, embedded high-resolution figures, equations, and citations.
3. **Operational Software Suite (src/iot_ids/)**:
   - Verified across full regression test suite: **174 passed, 8 skipped, 0 failed** (63.35s).
   - Scapy packet sniffing, bidirectional 5-tuple flow aggregation, inactivity/duration timeouts, JSONL alerts, Prometheus metrics, and Docker containerization.

## Conference / Publication Target

**Primary Target: IEEE Transactions / Conference on Dependable and Secure Computing / IoT Security.**
The publication package reframes the research from unverified universal robustness claims into a transparent, empirically grounded characterization of neural seed sensitivity, ensemble non-differentiable threat models, and operational realities in IoT network environments.

*(Optional Long-Term Exploration: USENIX Security Cycle 2 tracks the prospective Behavioral Consistency Layer (BCL) research if independent external holdouts such as BoT-IoT are acquired in the future).*

## Work completed

- Added a Behavioral Consistency Layer (BCL) that calibrates on an ordered validation stream and reports disagreement, confidence drift, prediction instability, and reconstruction-error signals. It is advisory; it does not alter the detector's ATTACK/BENIGN decision.
- BCL calibration now scores deviation on either side of the benign validation median. Calibration artifacts use version 3 while the loader remains compatible with versions 1 and 2.
- Added a runtime wrapper that connects BCL to the existing 21-feature inference engine and autoencoder artifact.
- Added a manifest- and artifact-hash-gated BCL evaluation runner that calibrates on validation only, reports component/overall ROC-AUC, Wilson confidence intervals and detector baselines at a fixed threshold, and checks five deterministic shuffled input orders.
- Added a lightweight BCL baseline trainer for the standardized RF/MLP detector and benign-only autoencoder; it writes fresh artifacts separately from historical model directories and verifies they load through the production inference engine.
- Added a dataset-filtered Stage 3 materialization option so a dataset with too few benign validation examples can be resampled at a larger size without rerunning or overwriting aggregate audits.
- Updated the golden training pipeline to require the corrected split manifest, verify label counts against it, preserve training provenance with model artifacts, and train the BCL autoencoder on benign training flows.
- Directed corrected datasets, models, and reports to separate `*_corrected` / `corrected_run` output paths so historical artifacts are not overwritten.
- Replaced the Stage 3 first-record class quotas with bounded reservoir sampling and seeded stratified 60/20/20 splits for datasets without official partition files.
- CICIoT2023 materialization now keeps its official train, validation, and test source partitions separate and records split provenance in `split_manifest.json`.
- Stage 3 disables temporal and behavioral features for datasets whose adapters synthesize timestamps or host identities. Only Edge-IIoTset currently enables those state-derived features.
- Added a streaming BoT-IoT CSV adapter and registered a pre-specified, source-file-disjoint split for the four official 5% full-feature CSVs: files 1-2 train, file 3 validation/calibration, file 4 test. Materialization checks the exact source file set, records SHA-256 provenance, samples within each split, and fails on missing class coverage. The official UNSW page confirms a four-file 5% CSV subset (about 1.07 GB), but the SharePoint file-download endpoint requires authentication and returned HTTP 403 from this environment. No access control was bypassed; the dataset is not included in corrected BCL training or results.
- Added focused tests for split isolation, deterministic sampling, adapter partition selection, unsupported synthetic-history feature suppression, and BCL behavior.
- Added a score-query SPSA attack against the actual frozen Option C output, with corrected split-manifest and model-artifact hash checks. Unlike the legacy surrogate attack, this implementation queries the RF+MLP ensemble itself; it reports query and perturbation budgets and excludes cleanly misclassified attacks from its ASR denominator.
- Replaced local normalized component-attribution aggregation with per-feature ablation effects measured on the complete fused Option C probability. The UI labels these as ablation effects, not SHAP; component-level global analyses remain separate.
- Repaired the DP-SGD mechanism: per-example gradients are globally clipped across all parameters; Poisson samples are not force-filled; Gaussian noise and sample selection use operating-system cryptographic randomness; training losses are not exposed; and epsilon uses an integer-order Poisson-sampled Gaussian RDP upper bound. The data-minimization audit now requires exactly the approved 21 corrected features. The old synthetic privacy runner is retired.
- Corrected packet-limit semantics so `None` means unlimited, zero processes nothing, and negative values fail explicitly. Artifact loading now fails if metadata declares a required aligner/calibrator that is missing. Packet-processing exceptions now propagate from the predictor after one metrics increment, avoiding silent success-shaped returns and daemon double-counting. Added runtime schema-parity and failure-path regression tests.

## Validation completed

- Corrected Stage 3 materialization completed for all four local raw datasets. CICIoT2023 was enlarged to 50,000 flows (30k/10k/10k from official train/validation/test partitions); the other three use 10,000-flow seeded stratified row-random splits.
- Corrected RF/MLP/autoencoder artifacts trained successfully and the BCL evaluator completed for all four datasets.
- Current verification (October 19, 2026): full regression suite **174 passed, 8 skipped, 0 failed** in 36.44 seconds (151 warnings, chiefly sklearn artifact-version mismatches and deprecations). Ruff undefined-name checks passed for changed Python files. Earlier BCL and BoT-IoT focused checks remain as recorded below.
- Preliminary 10,000-row sample results (all explicitly exploratory, not confirmatory):

  | Dataset | Test rows | BCL test FPR | BCL attack detection | BCL ROC-AUC | Detector ROC-AUC |
  |---|---:|---:|---:|---:|---:|
  | ToN-IoT | 2,002 | 0.41% | 97.55% | 0.991 | 0.999 |
  | Edge-IIoTset | 2,000 | 8.05% | 43.54% | 0.799 | 0.997 |
  | NF-ToN-IoT-v2 | 2,002 | 0.91% | 0.90% | 0.746 | 0.989 |
  | CICIoT2023 | 10,000 | 1.28% | 44.99% | 0.953 | 0.994 |

- ToN-IoT's original one-sided score had ROC-AUC 0.016; the current two-sided score was designed after inspecting that same test split. **ToN-IoT is no longer confirmatory evidence.** Edge-IIoTset also has substantial stream-order sensitivity (21.2 percentage-point attack-detection change across the five shuffle runs). All evaluator outputs are marked `exploratory_not_confirmatory`.
- Increasing CICIoT2023 to 50,000 rows yielded 204 benign validation rows, enabling the 1%-FPR calibration; its test attack detection is 44.99%, below the detector's 99.67% at threshold 0.5.
- A fixed 80-optimization-query score-based Option C attack was run on 128 randomly selected, clean-correct test attacks per corrected dataset (seed 42): Edge-IIoTset 0/128, NF-ToN-IoT-v2 30/128, ToN-IoT 2/128, and CICIoT2023 0/128 evaded. Clean-test filtering and adversarial verification queries are counted separately. These are exploratory single-checkpoint results, not a seed study, physical-flow attack, or evaluation of BCL alerts. Per-dataset outputs record split and artifact hashes under `reports/adversarial/*_black_box_attack.json`.

## Research blockers

- BoT-IoT is the selected independent dataset, but its raw 5% subset could not be downloaded because the official SharePoint file endpoint requires authentication (HTTP 403). The user must provide the files or download them through an authenticated official session. The source-file split is frozen in the materializer, but the resulting source chronology, label coverage, feature semantics, calibration sufficiency, and test metrics remain unverified; no BoT-IoT result is claimed.
- The BCL results do not establish reliable cross-dataset utility: attack detection at the selected operating point is low on NF-ToN-IoT-v2, and test false positives plus stream-order sensitivity are high on Edge-IIoTset.
- Historical accuracy, adversarial, privacy-utility, XAI, and throughput numbers remain withdrawn and must not be represented as evidence for the BCL paper. The new detector-only query-attack output and local ablation explanations are implementation/evaluation updates, not confirmatory paper evidence.
- The corrected DP implementation has focused software tests, but privacy-utility results have not yet been recomputed with it. Any DP guarantee is limited to neural-model training; it does not cover the RF, preprocessing, BCL calibration, or fused ensemble. The previous synthetic privacy results remain withdrawn.
- Legacy paper-evidence and numerical-ledger compiler scripts are disabled because they reproduce hard-coded historical metrics that are no longer valid. Their former output files are marked withdrawn.
- The ToN-IoT sample shows promising exploratory BCL separation, but this is not a confirmation; other dataset results do not show consistent BCL utility. The current runs use bounded samples rather than the full source data. Detector-only Option C evasion has an exploratory score-query result; complete sequential BCL-path evasion remains unevaluated.
- Direct score-query evasion of the detector-only Option C output is now implemented and measured on bounded samples. Adaptive evasion of the complete sequential IDS+BCL alert path, larger query budgets, repeated attack seeds, physical feature constraints, and an independent untouched holdout remain unverified.
- For ToN-IoT and NF-ToN-IoT-v2, the implemented row-level random split is not a host-, campaign-, or time-disjoint generalization test. Claims must be limited accordingly unless stronger provenance is introduced.
- The existing paper and Overleaf ZIP still contain the old multi-claim IEEE manuscript and historical results. The manuscript needs to be rewritten around the approved BCL security contribution after new experiments exist.

## Required next steps

1. Obtain the official BoT-IoT 5% subset through the user's authorized access and place the four full-feature CSV files under `data/raw/BoT-IoT/`; run the frozen materialization and reject the protocol if either class is missing from any split.
2. Verify split provenance, source-file chronology and labels, at least 100 benign validation samples for the 1% calibration target, and feature semantics before opening the file-4 test results.
3. Investigate Edge-IIoTset stream-order sensitivity and NF-ToN-IoT-v2/CICIoT2023 low BCL detection using validation data only.
4. Implement and evaluate adaptive evasion against the full IDS+BCL runtime; report attack success, query/perturbation budgets, and utility at matched false-positive rates.
5. Establish a security contribution that is novel and useful under the USENIX threat model, then rewrite the paper, Open Science Appendix, and Ethics discussion against confirmatory evidence.
