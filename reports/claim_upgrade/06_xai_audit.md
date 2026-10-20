# EXACT SHAP AUDIT FOR FUSED PREDICTOR (P6)

## 1. Executive Summary
This report assesses the mathematical integrity of the component-wise SHAP approximation currently implemented in the Option C framework. By constructing a true end-to-end black-box wrapper for the $f_C(x) = 0.7 * P_{RF}(x) + 0.3 * P_{MLP}(x)$ probability surface and executing direct KernelSHAP, we confirm that the current linear combination method violates the SHAP additivity axiom and introduces significant mathematical distortion. Furthermore, exhaustive exact Shapley enumeration is proven computationally intractable at scale.

## 2. Current XAI Implementation Parameters
- **Attribution Game**: Linear weighted average of independent component attributions ($0.7 * SHAP_{RF} + 0.3 * SHAP_{MLP}$).
- **Explainer**: `TreeExplainer` (exact) for Random Forest, and `GradientExplainer` (expected gradients) for Robust MLP.
- **Output Space**: Currently mixed. TreeExplainer typically evaluates margin/log-odds by default unless strictly calibrated, while GradientExplainer evaluates probabilities.
- **Background Dataset**: 100 random samples from the training set.
- **Normalization**: Global importance averages are L1 normalized before combining.

## 3. Computational Benchmark (Exact Shapley Enumeration)
For an exhaustive exact Shapley value computation over the 21 canonical features, exactly $2^21 = 2,097,152$ feature coalitions must be evaluated per sample.
- **Measured Cost**: 3.53 seconds per single sample.
- **Extrapolation**: Explaining the standard 1,400-sample test set exactly would require roughly 1.37 hours. Applying this exactly to the 200,000+ validation flows used in multi-seed evaluation is strictly intractable.

## 4. Empirical Evaluation vs Direct Fused KernelSHAP
Because exact enumeration is too costly for broad deployment, we compared the current component-wise linear approximation against highly-sampled direct KernelSHAP ($N_{samples} = 10,000$) on the complete Option C wrapper.

### Comparison Metrics (10 controlled training samples)
- **Mean Absolute Error (MAE)**: 0.0434
- **Maximum Absolute Error**: 5.0783
- **Spearman Rank Correlation ($\\rho$)**: Mean = 0.845, Min = 0.603, Max = 0.909
- **Fused Model Additivity Residual**: 2.98e-09 (Mathematically exact)
- **Current Linear Additivity Residual**: 0.8348 (Violates additivity axiom)

## 5. Claim Assessment
**Question**: Is the current XAI method exact game-theoretic SHAP for the fused predictor?

**VERDICT: FAIL**

**Justification**:
The current approach calculates exact or gradient-approximated SHAP on the individual models in isolated output spaces and linearly averages them. Our audit proves this method violates the Shapley additivity axiom (mean residual $\\approx 0.83$) because it ignores non-linear interactions across the final ensemble voting mechanism and potentially mixes margin space with probability space. While the feature importance rankings are highly correlated (mean $\\rho = 0.84$), it cannot mathematically be claimed as "exact SHAP for the fused predictor."

We recommend replacing the claim with an acknowledgement that the current XAI provides a computationally efficient component-wise surrogate explanation, as exact direct SHAP estimation is computationally prohibitive.

