# RESOLVED CURRENT CANONICAL PROJECT CONFIGURATION

| COMPONENT | VALUE | SOURCE FILE | STATUS |
| :--- | :--- | :--- | :--- |
| **DATASET** | `stage3` partitioned (60/20/20) | `scripts/03_materialize_and_audit_features.py` | RESOLVED |
| **SPLIT LOGIC** | Exact 7000 samples chronological limit | `scripts/03_materialize_and_audit_features.py` | RESOLVED |
| **FEATURES** | 21 canonical extracted | `src/iot_ids/features/canonical/flow_builder.py` | RESOLVED |
| **SCALER** | `RobustScaler` (no log1p) | `scripts/run_golden_pipeline.py:171` | RESOLVED |
| **RF CONFIG** | `n_estimators=100`, `max_depth=15`, `n_jobs=-1` | `scripts/run_golden_pipeline.py:181` | RESOLVED |
| **STANDARD MLP CONFIG** | 4-layer (128/64/32), Adam 1e-3, 15 epochs, batch 256 | `scripts/run_golden_pipeline.py:86` | RESOLVED |
| **ROBUST MLP CONFIG** | Same arch, PGD-7 Adv Training, eps=0.1, alpha=0.025 | `scripts/run_golden_pipeline.py:110` | RESOLVED |
| **FUSION** | Static weight 0.7 RF + 0.3 MLP | `scripts/run_golden_pipeline.py:201` | RESOLVED |
| **PGD TRAINING** | PGD-7 | `src/iot_ids/adversarial/adversarial_training.py` | RESOLVED |
| **PGD EVALUATION** | PGD-10 | `scripts/run_golden_pipeline.py:243` | RESOLVED |
| **SURROGATE ATTACK** | Differentiable PyTorch Surrogate for RF | `scripts/15_adaptive_ensemble_attack.py` | RESOLVED |
| **DP** | Synthetic Gaussian Demonstration | `scripts/run_privacy_experiments.py` | RESOLVED |
| **XAI** | Separate RF and MLP SHAP calculation | `scripts/12_run_shap_experiments.py` | RESOLVED |
| **RUNTIME** | In-memory `.predict()` latency (no network I/O) | `scripts/benchmark_runtime_xai_privacy.py` | RESOLVED |
