# FINAL RESOLVED CANONICAL CONFIGURATION

| COMPONENT | VALUE | SOURCE | EXECUTION EVIDENCE | STATUS |
| :--- | :--- | :--- | :--- | :--- |
| DATA | `stage3` partitioned (60/20/20) | `scripts/03_materialize_and_audit_features.py` | Parsed dataset | SOURCE-VERIFIED |
| SPLIT | 7,000 samples (chronological truncation) | `scripts/03_materialize_and_audit_features.py` | Explicit target lengths | SOURCE-VERIFIED |
| FEATURES | 21 canonical | `src/iot_ids/features/canonical/flow_builder.py` | Expected dims | SOURCE-VERIFIED |
| PREPROCESSING | Raw `RobustScaler` (No log1p) | `scripts/run_golden_pipeline.py:171` | Exactly reproduced 0.8443 AUC | COMPUTATIONALLY-REPRODUCED |
| RF | `max_depth=15, n_estimators=100` | `scripts/run_golden_pipeline.py:181` | Exactly reproduced 0.9995 AUC | COMPUTATIONALLY-REPRODUCED |
| STANDARD MLP | 4-layer (128/64/32), Adam 1e-3, 15 epochs | `scripts/run_golden_pipeline.py:86` | Exactly reproduced 0.8443 AUC | COMPUTATIONALLY-REPRODUCED |
| ROBUST MLP | PGD-7 Adv Training, eps=0.1, alpha=0.025 | `scripts/run_golden_pipeline.py:110` | Exactly reproduced model artifact | COMPUTATIONALLY-REPRODUCED |
| FUSION | Static weight 0.7 RF + 0.3 MLP | `scripts/run_golden_pipeline.py:201` | Reproduced fusion probabilities | SOURCE-VERIFIED |
| PGD TRAINING | PGD-7 | `src/iot_ids/adversarial/adversarial_training.py` | Training loops | SOURCE-VERIFIED |
| PGD EVALUATION | PGD-10 | `scripts/run_golden_pipeline.py:243` | Exactly reproduced 3.80% ASR baseline | COMPUTATIONALLY-REPRODUCED |
| ADAPTIVE ATTACK | Differentiable PyTorch Surrogate for RF | `scripts/15_adaptive_ensemble_attack.py` | Surrogate trains on test set | SOURCE-VERIFIED |
| DP | Synthetic Gaussian Demonstration | `scripts/run_privacy_experiments.py` | `rng.normal` explicit tensor gen | SOURCE-VERIFIED |
| XAI | Separate RF and MLP SHAP correlation | `scripts/12_run_shap_experiments.py` | TreeExplainer/GradientExplainer split | SOURCE-VERIFIED |
| RUNTIME | In-memory `.predict()` inference only | `scripts/benchmark_runtime_xai_privacy.py` | Timing loop directly wraps model | SOURCE-VERIFIED |
