# STANDARD SCALER VS ROBUST SCALER

- **StandardScaler**: Only appears in the `execute_lock_audit.py` (the fresh reproduction script created dynamically).
- **RobustScaler**: Exists universally in the canonical repository (`run_golden_pipeline.py`, `src/iot_ids/preprocessing/pipeline.py`).
- **Inconsistency**: While `RobustScaler` is canonical, the *way* it is used differs. `src/iot_ids/preprocessing/pipeline.py` uses it wrapped in a `log1p` transformation. `run_golden_pipeline.py` uses it natively on raw features.

**Verdict**: The official canonical pipeline (`run_golden_pipeline.py`) used raw `RobustScaler` across RF and MLP.
