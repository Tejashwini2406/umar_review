# CURRENT VS LEGACY PIPELINE

### CURRENT
- `scripts/run_golden_pipeline.py` (Defines exact current artifacts)
- `scripts/15_adaptive_ensemble_attack.py` (Current Option C evaluation)
- `src/iot_ids/runtime/engine.py` (Current inference)

### LEGACY / ORPHANED
- `scripts/train_rf.py` (Orphaned; hyperparams drifted to max_depth=20 vs golden 15)
- `scripts/10_run_adversarial_evaluation.py` (Legacy direct attack script)

### AMBIGUOUS
- `src/iot_ids/preprocessing/pipeline.py` (The official preprocessor uses `log1p`, but the golden pipeline bypassed it).
