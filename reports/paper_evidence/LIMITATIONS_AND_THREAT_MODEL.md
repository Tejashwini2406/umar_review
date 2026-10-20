# Current Threat Model and Evidence Limitations

This file supersedes the historical threat-model text in this directory. Current evaluation status and commands are tracked in [FINAL_IMPLEMENTATION_STATUS.md](../../FINAL_IMPLEMENTATION_STATUS.md) and [PUBLICATION_REMAINING_WORK.md](../../PUBLICATION_REMAINING_WORK.md).

## Evaluated implementation scope

- The detector-only score-query SPSA experiment queries the actual frozen RF+MLP Option C probability. It is bounded, checkpoint-specific, and exploratory; it does not represent physical flow realizability or attack the sequential BCL alert state.
- The separate advisory BCL alert path has not undergone a full adaptive/query/transfer evasion study.
- Local explanations are fused-probability single-feature ablation effects. They are not exact SHAP values and are not additive where features interact.
- DP-SGD is scoped to neural-model training only. The Random Forest, preprocessing fit, BCL calibration, and fused system are outside that guarantee. Archived synthetic utility numbers and old epsilon claims are withdrawn.
- The current sample splits do not establish host-, campaign-, time-, or unseen-domain generalization. The planned independent BoT-IoT source files are unavailable locally.
- Classifier-only CPU throughput is not packet-to-alert throughput or enterprise-scale evidence.

## Reporting requirements

Report attack knowledge, target output, perturbation and query budgets, sample eligibility, checkpoint and split hashes, and benign false-positive behavior. Preserve negative results. Do not interpret exploratory or test-informed results as confirmatory security evidence.
