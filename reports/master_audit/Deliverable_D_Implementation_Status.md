# Deliverable D: Implementation Status Matrix

| Component | Status | Evidence/Notes |
|---|---|---|
| Data Cleaning & Splits | Implemented and execution-verified | Chronological split logic verified in P1 audit. |
| Feature Scaling | Implemented and execution-verified | RobustScaler is saved and loaded correctly during inference. |
| Model Training (RF, MLP) | Implemented and execution-verified | Artifacts exist in `models/` and can be loaded. |
| DP-SGD Integration | Implemented and execution-verified | Opacus PRV applied correctly to the neural stream (P3). |
| PGD-7 Adversarial Training | Implemented and execution-verified | Implemented in neural stream training loop. |
| Option C Fusion (0.7/0.3) | Implemented and execution-verified | Found in `InferenceEngine.predict()`. |
| Black-box NES Attack | Implemented and execution-verified | Batched NES successfully evades at 36.1% ASR (P5). |
| Exact Ensemble SHAP | Broken or contradicted by evidence | Math intractable ($2^{21}$ queries); uses linear surrogate instead (P6). |
| End-to-End Live Throughput | Planned or scaffolded only | Scapy dependencies missing; benchmark only tests offline inference (P7). |
| Federated Training | Planned or scaffolded only | Mentioned in limitations, no execution evidence found. |
