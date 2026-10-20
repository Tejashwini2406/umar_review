# XAI SCOPE

- **Mechanism**: The XAI implementation extracts `TreeExplainer` and `GradientExplainer` attributions independently on frozen components.
- **Limitation**: It does NOT compute a joint game-theoretic SHAP across the fusion boundary.
- **Status**: It performs component-level attribution analysis, not ensemble-level.
