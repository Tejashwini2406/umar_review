# Historical Claim-Evidence Matrix — Superseded

> This legacy matrix referred to historical experiment artifacts. Do not use its old PASS/REVISE labels or numerical values as current evidence. See [current implementation status](../../FINAL_IMPLEMENTATION_STATUS.md) and [publication remaining work](../../PUBLICATION_REMAINING_WORK.md).

| Topic | Current evidence status | Permitted wording |
|---|---|---|
| Ensemble score-query evasion | Exploratory SPSA evaluations exist for four corrected checkpoints and bounded samples; not a seed study, physical-flow attack, or BCL-path attack | Report the exact dataset, checkpoint hashes, selected sample count, and query/perturbation budgets; do not claim general robustness |
| Fusion weight | 0.7/0.3 is the evaluated Option C operating point, not a proven global optimum | Call it the selected configuration under the documented experiment |
| BCL utility | Exploratory results have weak utility on some datasets, stream-order sensitivity, and test-guided ToN-IoT method revision | Preserve the negative findings and do not call the evidence confirmatory |
| Differential privacy | The DP-SGD implementation is corrected and unit-tested; old synthetic utility values and epsilon claims are withdrawn | State only that the code implements a neural-model DP-SGD mechanism; no current benchmark utility result is available |
| Local explainability | Current local output is fused-probability single-feature ablation, not SHAP | Call it a local fused-model ablation effect; do not claim additive SHAP values |
| Runtime performance | Existing high-throughput number measures classifier-only host inference | State the measured batch and hardware scope; do not equate it to network or enterprise throughput |
| Independent generalization | BoT-IoT source files are not available in the local raw-data folder | Make no independent-holdout or unseen-domain generalization claim |
