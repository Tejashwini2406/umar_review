import json

with open('C:/Users/umari/Documents/P1_task_Implementation/reports/claim_upgrade/06_xai_audit.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

md = '# EXACT SHAP AUDIT FOR FUSED PREDICTOR (P6)\n\n'
md += '## 1. Executive Summary\n'
md += 'This report assesses the mathematical integrity of the component-wise SHAP approximation currently implemented in the Option C framework. By constructing a true end-to-end black-box wrapper for the $f_C(x) = 0.7 * P_{RF}(x) + 0.3 * P_{MLP}(x)$ probability surface and executing direct KernelSHAP, we confirm that the current linear combination method violates the SHAP additivity axiom and introduces significant mathematical distortion. Furthermore, exhaustive exact Shapley enumeration is proven computationally intractable at scale.\n\n'

md += '## 2. Current XAI Implementation Parameters\n'
md += '- **Attribution Game**: Linear weighted average of independent component attributions ($0.7 * SHAP_{RF} + 0.3 * SHAP_{MLP}$).\n'
md += '- **Explainer**: `TreeExplainer` (exact) for Random Forest, and `GradientExplainer` (expected gradients) for Robust MLP.\n'
md += '- **Output Space**: Currently mixed. TreeExplainer typically evaluates margin/log-odds by default unless strictly calibrated, while GradientExplainer evaluates probabilities.\n'
md += '- **Background Dataset**: 100 random samples from the training set.\n'
md += '- **Normalization**: Global importance averages are L1 normalized before combining.\n\n'

md += '## 3. Computational Benchmark (Exact Shapley Enumeration)\n'
md += f'For an exhaustive exact Shapley value computation over the 21 canonical features, exactly $2^{21} = 2,097,152$ feature coalitions must be evaluated per sample.\n'
md += f'- **Measured Cost**: {d["benchmarks"]["time_to_evaluate_exact_coalitions_per_sample"]:.2f} seconds per single sample.\n'
md += f'- **Extrapolation**: Explaining the standard 1,400-sample test set exactly would require roughly 1.37 hours. Applying this exactly to the 200,000+ validation flows used in multi-seed evaluation is strictly intractable.\n\n'

md += '## 4. Empirical Evaluation vs Direct Fused KernelSHAP\n'
md += 'Because exact enumeration is too costly for broad deployment, we compared the current component-wise linear approximation against highly-sampled direct KernelSHAP ($N_{samples} = 10,000$) on the complete Option C wrapper.\n\n'
md += '### Comparison Metrics (10 controlled training samples)\n'
metrics = d['comparison_metrics']
md += f'- **Mean Absolute Error (MAE)**: {metrics["mae_linear_vs_fused"]:.4f}\n'
md += f'- **Maximum Absolute Error**: {metrics["max_error_linear_vs_fused"]:.4f}\n'
md += f'- **Spearman Rank Correlation ($\\\\rho$)**: Mean = {metrics["spearman_rho_mean"]:.3f}, Min = {metrics["spearman_rho_min"]:.3f}, Max = {metrics["spearman_rho_max"]:.3f}\n'
md += f'- **Fused Model Additivity Residual**: {metrics["fused_additivity_residual_mae"]:.2e} (Mathematically exact)\n'
md += f'- **Current Linear Additivity Residual**: {metrics["linear_additivity_residual_mae"]:.4f} (Violates additivity axiom)\n\n'

md += '## 5. Claim Assessment\n'
md += '**Question**: Is the current XAI method exact game-theoretic SHAP for the fused predictor?\n\n'
md += '**VERDICT: FAIL**\n\n'
md += '**Justification**:\n'
md += 'The current approach calculates exact or gradient-approximated SHAP on the individual models in isolated output spaces and linearly averages them. Our audit proves this method violates the Shapley additivity axiom (mean residual $\\\\approx 0.83$) because it ignores non-linear interactions across the final ensemble voting mechanism and potentially mixes margin space with probability space. While the feature importance rankings are highly correlated (mean $\\\\rho = 0.84$), it cannot mathematically be claimed as "exact SHAP for the fused predictor."\n\n'
md += 'We recommend replacing the claim with an acknowledgement that the current XAI provides a computationally efficient component-wise surrogate explanation, as exact direct SHAP estimation is computationally prohibitive.\n\n'

with open('C:/Users/umari/Documents/P1_task_Implementation/reports/claim_upgrade/06_xai_audit.md', 'w', encoding='utf-8') as f:
    f.write(md)
