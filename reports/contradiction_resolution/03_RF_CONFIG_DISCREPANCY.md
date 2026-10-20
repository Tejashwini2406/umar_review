# RF HYPERPARAMETER MISMATCH

### 1. Which configuration generated golden RF?
- **Source**: `scripts/run_golden_pipeline.py:181`
- **Config**: `RandomForestClassifier(n_estimators=100, max_depth=15, random_state=SEED, n_jobs=-1)`

### 2. Which configuration generated fresh RF?
- **Source**: `scratch/execute_lock_audit.py`
- **Config**: `RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)`

### 3. Which configuration is used by the current pipeline?
- Both exist! `scripts/train_rf.py` uses `max_depth=20, class_weight='balanced'`. 
- `scripts/run_golden_pipeline.py` uses `max_depth=15` and no class weighting.

### Conclusion
`max_depth=15` without class weighting is the canonical provenance configuration that generated the golden metrics. The `train_rf.py` script drifted to `max_depth=20` and `balanced` later.
