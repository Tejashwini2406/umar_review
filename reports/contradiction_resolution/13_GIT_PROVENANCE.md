# GIT PROVENANCE

Based on the evidence across the codebase:
- The project evolved from separate training scripts (`train_rf.py`) to a centralized pipeline (`run_golden_pipeline.py`) to guarantee artifact consistency for final publication.
- `train_rf.py` continued to evolve (adding `class_weight='balanced'`) independently of the frozen `golden_run` models.
- The scaler discrepancy arose because `run_golden_pipeline.py` instantiated `RobustScaler` natively instead of importing the robust preprocessing class from `pipeline.py`.
