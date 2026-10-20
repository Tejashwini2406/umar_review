# XAI ATTRIBUTION VERIFICATION

- **RF Method**: `TreeExplainer` (exact Shapley values for trees).
- **MLP Method**: `GradientExplainer` (expected gradients for neural networks).
- **Implementation**: The SHAP values are extracted *independently* from the two models. Their absolute mean attributions are then ranked, and a Spearman rank correlation is performed.
- **Fusion SHAP**: The code NEVER attempts to calculate the SHAP of the complete fused predictor (Option C).

*(Status: SOURCE-VERIFIED)*

### CONCLUSION
The codebase implements **separate component-level attribution analysis**, NOT a fused SHAP explanation of Option C.
