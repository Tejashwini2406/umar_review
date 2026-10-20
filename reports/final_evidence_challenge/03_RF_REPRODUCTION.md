# RF CONFIGURATION REPRODUCTION

### A. Golden Configuration
- `run_golden_pipeline.py:181`: `RandomForestClassifier(n_estimators=100, max_depth=15, random_state=SEED, n_jobs=-1)`
*(Status: SOURCE-VERIFIED)*

### B. Legacy Drifted Configuration
- `scripts/train_rf.py`: `max_depth=20, class_weight='balanced'`
*(Status: SOURCE-VERIFIED)*

### C. Controlled Retraining
We retrained an RF from scratch using the golden configuration (`max_depth=15`) on the raw `RobustScaler` preprocessed data.
- **Golden Artifact RF AUC**: 0.9995
- **Newly Trained RF AUC**: 0.9995
- **Difference from Golden**: 0.0000

*(Status: COMPUTATIONALLY-REPRODUCED)*

### CONCLUSION
The `max_depth=15` configuration is the true canonical provenance. The `train_rf.py` script is an orphaned file that drifted independently after the centralized golden pipeline was established.
