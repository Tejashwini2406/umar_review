# Publication Readiness: Remaining Work

**Status: NOT READY TO SUBMIT.** This checklist tracks evidence and paper work still required for the selected USENIX Security '27 Cycle 2 target. Passing software tests does not establish scientific validity or acceptance readiness.

## Current verified state

- The current BCL implementation, corrected bounded experiments on four locally available datasets, and their outputs are exploratory. All four test sets have been inspected; ToN-IoT's scoring rule was revised after looking at its test results.
- BCL utility is weak on NF-ToN-IoT-v2 and CICIoT2023, and Edge-IIoTset is sensitive to stream order.
- A BoT-IoT adapter and fixed source-file split path are implemented and unit-tested. The official 5% data files are not present because the official SharePoint download endpoint returned HTTP 403 in this environment.
- Latest software verification (October 19, 2026): **174 passed, 8 skipped, 0 failed**; Ruff undefined-name checks passed for changed Python files. This software verification does not validate research claims or substitute for the missing independent dataset.
- The existing IEEE manuscript is withdrawn and contains historical claims. Do not submit it or reuse its historical metrics.

## Blocking work

### 1. Obtain and audit the independent dataset

- [ ] Download the official BoT-IoT 5% full-feature CSV subset using authorized access.
- [ ] Place only the four extracted CSV files under `data/raw/BoT-IoT/`. Do not commit raw dataset files.
- [ ] Verify filenames, file sizes, schemas, row counts, label values, duplicate records, timestamp order, and source-file SHA-256 hashes.
- [ ] Confirm licensing/citation requirements and document the exact dataset release and access date.
- [ ] Treat `flgs`, `state`, `category`, and `subcategory` according to their real semantics; do not leak label/category fields into predictive features.
- [ ] Run the fixed partition materializer without viewing the test outcomes first:

  ```powershell
  .\.venv\Scripts\python.exe scripts\03_materialize_and_audit_features.py --dataset BoT-IoT --sample-limit 50000
  ```

- [ ] Confirm that each frozen split has both classes and that validation has at least 100 benign examples for the configured 1% calibration target. If a preassigned source file lacks required coverage, record the protocol failure and design a new protocol using training/validation information only; do not change groups after opening test results.

### 2. Freeze, train, and evaluate

- [ ] Record the threat model, attacker access, assumptions, allowed perturbations, budgets, and primary outcomes before running the holdout evaluation.
- [ ] Freeze BCL signals, history behavior, threshold procedure, baseline, seeds, feature preprocessing, and metrics using training/validation data only.
- [ ] Train and evaluate with the pre-registered commands:

  ```powershell
  .\.venv\Scripts\python.exe scripts\16_train_bcl_baselines.py --dataset BoT-IoT
  .\.venv\Scripts\python.exe scripts\15_evaluate_bcl.py --dataset BoT-IoT
  ```

- [ ] Verify all split and model provenance hashes and report test metrics with uncertainty that reflects grouping/dependence rather than relying only on row-level intervals.
- [ ] Evaluate the complete IDS+BCL path against adaptive, query-based, and transfer evasion attacks. Detector-only Option C now has an exploratory score-query SPSA run on 128 clean-correct test attacks per corrected dataset (80 optimization queries per flow; clean-filtering and final-scoring queries are separate); this does not optimize the sequential BCL alert path. Use a pre-registered, untouched holdout and report budgets, clean operating points, attack success, and benign false positives.
- [ ] Compare against detector-only and appropriate simple baselines at matched false-positive rates; include ablations for each BCL signal and stream history.
- [ ] Use validation data only to investigate Edge-IIoTset order sensitivity and the weak results on NF-ToN-IoT-v2/CICIoT2023. Preserve all negative and null outcomes.
- [ ] If the source split or BCL utility fails the predeclared gates, revise or stop the claim. Do not move the test boundary or retrofit the method to test outcomes.

### 3. Rebuild the research paper and artifacts

- [ ] Establish the actual security contribution and novelty against current related work; the current engineering implementation alone is not enough to establish a top-tier security contribution.
- [ ] Write a new anonymized USENIX-format manuscript from validated evidence; replace the withdrawn IEEE draft rather than editing its unsupported claims into a new venue template.
- [ ] Include limitations, threat model, dataset provenance/licensing, ethics, responsible data handling, and negative results.
- [ ] Prepare the required Open Science Appendix and an anonymized, reproducible artifact package with environment lock, exact commands, data acquisition instructions, hashes, and expected outputs. Do not redistribute data without permission.
- [ ] Build the paper and artifact from a clean environment; rerun tests, lint, reproducibility checks, and verify no author identity or local-path leakage in the anonymous submission.
- [ ] Check the live USENIX CFP for deadlines and formatting requirements immediately before submission. Current tracked Cycle 2 dates are registration January 19, 2027; paper January 26, 2027; artifacts January 29, 2027.

## Submission go/no-go gate

Submit only if an independent evaluation has remained untouched until the protocol and method were frozen; provenance is reproducible; the adaptive security evaluation supports a clear, useful claim over baselines; the manuscript follows the live CFP; and the artifact package reproduces the reported results. Otherwise, do not submit to this cycle—continue the research or choose a more appropriate venue.
