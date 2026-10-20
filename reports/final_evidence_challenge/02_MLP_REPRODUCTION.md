# MLP DISCREPANCY REPRODUCTION

### EXPERIMENT
We independently evaluated the golden models and retrained them from scratch to determine the exact cause of the 0.8443 AUC anomaly.

### A. Golden Artifact Evaluation
- **Standard MLP with Golden Scaler**: 0.8443 AUC
- **Robust MLP with Golden Scaler**: 0.8002 AUC
*(Status: COMPUTATIONALLY-REPRODUCED)*

### B. Controlled Retraining (Standard MLP)
We re-initialized an empty MLP and trained it using the canonical procedure (`Adam`, `lr=0.001`, 15 epochs, raw `RobustScaler`).
- **Newly Trained Standard MLP AUC**: 0.8443 AUC
- **Difference from Golden**: 0.0000

*(Status: COMPUTATIONALLY-REPRODUCED)*

### CONCLUSION
The exact training process that created the golden models is fully reproducible. The 0.8443 AUC was caused precisely because the canonical script (`run_golden_pipeline.py`) scaled the data with raw `RobustScaler` (omitting the `log1p` transformation found in the official `PreprocessingPipeline` class). When the data is scaled with raw `RobustScaler`, the Standard MLP convergences deterministically to 0.8443 AUC. The previous `StandardScaler` test produced a higher observed AUC in this experiment because neural networks are highly sensitive to the feature distribution.
