# XAI AUDIT

> **Historical audit note:** The local explanation implementation has since changed to single-feature ablation on the complete fused Option C probability. Current local outputs are not SHAP and do not use weighted component normalization. The global component-level SHAP analysis described below remains component-level.

The project uses TreeSHAP for RF and Gradient-based attribution for the MLP.
The fusion attributes combine them linearly (`0.7 * RF_SHAP_norm + 0.3 * MLP_SHAP_norm`).
This is a **custom weighted attribution**, NOT "exact SHAP". Claiming exact SHAP for the non-linear fused ensemble is incorrect.
