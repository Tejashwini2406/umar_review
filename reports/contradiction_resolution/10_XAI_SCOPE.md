# XAI ATTRIBUTION

- **Source**: `scripts/12_run_shap_experiments.py`
- **Implementation**: 
  - `explain_rf_shap()` uses exact TreeExplainer for RF.
  - `explain_mlp_shap()` uses GradientExplainer for MLP.
  - It compares them via Spearman Rank Correlation (`compare_shap_rankings`).
- **Verdict**: A (TreeSHAP for RF) and B (Gradient attribution for MLP) are calculated separately. It NEVER computes the mathematically integrated SHAP of the complete fused predictor (Option C).
