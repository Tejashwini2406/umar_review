# Figure & Table Source Mapping Ledger

> Historical source map only. Most mapped artifacts are withdrawn or superseded; do not use this ledger as current paper evidence. The DP row and archived summary are explicitly withdrawn.

| IEEE Paper Element | Document Section | Source Script | Generated Output Path | Underlying JSON Evidence Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **Table I: Clean Detection ROC-AUC** | Section V-A | `scripts/run_golden_pipeline.py` | `reports/tables/latex_ci_table.tex` | `reports/golden_run_manifest.json` |
| **Table II: Baseline PGD-10 ASR** | Section V-B | `scripts/run_golden_pipeline.py` | `reports/tables/latex_ci_table.tex` | `reports/golden_run_manifest.json` |
| **Table III: Adaptive Surrogate Attack** | Section V-B | `scripts/15_adaptive_ensemble_attack.py` | `reports/figures/fig_adaptive_attack_asr.pdf` | `reports/adversarial/adaptive_attack_results.json` |
| **Table IV: Fusion Weight Sweep** | Section V-A | `scripts/18_fusion_weight_ablation.py` | `reports/figures/fig_fusion_ablation_pareto.pdf` | `reports/tables/fusion_ablation_results.json` |
| **Table V: Bootstrap 95% CIs** | Section V-A / App. | `scripts/17_bootstrap_confidence_intervals.py` | `reports/tables/latex_ci_table.tex` | `reports/tables/table_statistical_confidence_intervals.json` |
| **Table VI: DP Utility Sweep** | Withdrawn | Legacy synthetic runner retired | Do not include until rerun on corrected real data | `reports/privacy/privacy_evidence_summary.json` (withdrawn) |
| **Table VII: Host Inference Throughput** | Section V-D | `scripts/benchmark_runtime_xai_privacy.py` | Table in Section V-D | `reports/final_forensic_audit/controlled_runtime_results.json` |
| **Figure 4: Fusion Pareto Plot** | Section V-A | `scripts/18_fusion_weight_ablation.py` | `reports/figures/fig_fusion_ablation_pareto.pdf` | `reports/tables/fusion_ablation_results.json` |
| **Figure 5: XAI Local Case Study** | Section V-E | `scripts/16_xai_attack_breakdown.py` | `reports/figures/fig_xai_local_case_study.pdf` | `reports/xai/xai_attack_breakdown.json` |
| **Figure 6: Adaptive Attack ASR** | Section V-B | `scripts/15_adaptive_ensemble_attack.py` | `reports/figures/fig_adaptive_attack_asr.pdf` | `reports/adversarial/adaptive_attack_results.json` |
