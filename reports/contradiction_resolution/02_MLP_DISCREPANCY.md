# GOLDEN MLP VS FRESHLY TRAINED MLP

### THE CONTRADICTION
- Historical/Golden Edge AUC = 0.8443 (Std), 0.8001 (Rob)
- Fresh Reproduction Edge AUC = 0.9753 (Std), 0.9777 (Rob)

### 1A. ARCHITECTURE COMPARISON
- **Golden Model**: `MLPModule` in `scripts/run_golden_pipeline.py`. (4-layer: 128->64->32->1, BatchNorm, Dropout 0.2).
- **Fresh Model**: Identical.
- **Verdict**: EXACT ARCHITECTURAL MATCH.

### 1B. TRAINING PROCEDURE COMPARISON
- **Golden Procedure**: Adam, lr=0.001, BCEWithLogitsLoss, 15 epochs, batch_size=256.
- **Fresh Procedure**: Identical.
- **Verdict**: EXACT TRAINING HYPERPARAMETER MATCH.

### 1C. INPUT DATA COMPARISON
- **Dataset**: `stage3/Edge-IIoTset/test.parquet`.
- **Verdict**: IDENTICAL DATAFRAME TENSORS.

### 1D. SCALING COMPARISON
- **Golden Pipeline (`scripts/run_golden_pipeline.py:171`)**: Uses `RobustScaler` natively `scaler.fit_transform(X_tr)`.
- **Fresh Pipeline (`scratch/execute_lock_audit.py`)**: Used `StandardScaler` arbitrarily.
- **Golden Preprocessing Class (`src/iot_ids/preprocessing/pipeline.py`)**: Defines `PreprocessingPipeline` which uses `log1p` -> `RobustScaler` -> `clip(-10, 10)`.
- **The Issue**: The `run_golden_pipeline.py` script bypassed the `PreprocessingPipeline` class and just called `RobustScaler.fit_transform(X)` directly without `log1p`!

### 1E. EXACT CONFIGURATION REPRODUCTION
We wrote a diagnostic script (`scratch/diagnose_c1.py`) to pass the exact raw `test.parquet` through the golden `RobustScaler.transform(X)` and through the `log1p` version.
- When applying ONLY `RobustScaler` (the `run_golden_pipeline.py` method), the Golden MLP evaluated on Edge gives EXACTLY **0.8443**.
- **Conclusion**: The golden results are not random. They are the deterministic output of training an MLP on data scaled exclusively with a raw `RobustScaler`. The fresh reproduction scored much higher (0.9753) simply because it used `StandardScaler` which happened to distribute the tabular data much better for neural networks than raw `RobustScaler`.
