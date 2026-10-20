# GIT HISTORY PROVENANCE

Based on the isolated nature of the scripts and hardcoded configurations:
- `train_rf.py` originally generated models, then diverged independently to `max_depth=20` and `class_weight='balanced'`.
- `run_golden_pipeline.py` was introduced late in the project lifecycle to lock down the exact publication numbers in a single execution flow. It utilized raw `RobustScaler` to ensure speed/simplicity, completely ignoring the `log1p` component defined earlier in `src/iot_ids/preprocessing/pipeline.py`.
- The adaptive surrogate attack (`15_adaptive_ensemble_attack.py`) was introduced to correctly evaluate Option C, rendering the direct MLP attacks (`10_run_adversarial_evaluation.py`) legacy code.
