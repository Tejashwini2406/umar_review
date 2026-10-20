"""Global Explainability Module for IoT IDS Option C Framework.

Computes global feature importances, TreeExplainer / GradientExplainer SHAP attributions,
feature-ranking agreement between Random Forest and Robust MLP, and cross-dataset stability metrics.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from scipy.stats import spearmanr

from iot_ids.xai.rf_explainer import explain_rf
from iot_ids.xai.mlp_explainer import explain_mlp
from iot_ids.xai.shap_explainer import explain_rf_shap, explain_mlp_shap


def analyze_dataset_global_xai(
    rf_model: Any,
    mlp_model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    dataset_name: str,
    max_shap_samples: int = 500,
    background_samples: int = 100,
) -> Dict[str, Any]:
    """Computes comprehensive global XAI metrics for a single dataset schema.

    Returns dictionary containing:
      - RF Gini importance and SHAP TreeExplainer importance
      - Robust MLP permutation importance and SHAP GradientExplainer importance
      - Spearman rank correlation (rho) between RF and Robust MLP feature rankings
      - Isolated zero-IQR preprocessing flags
    """
    # 1. Random Forest global importances
    rf_perm = explain_rf(rf_model, X, y, feature_names, n_repeats=3)
    rf_shap = explain_rf_shap(rf_model, X, feature_names, max_samples=max_shap_samples)

    # 2. Robust MLP global importances
    mlp_perm = explain_mlp(mlp_model, X, y, feature_names, n_repeats=3)
    mlp_shap = explain_mlp_shap(
        mlp_model, X, feature_names, background_samples=background_samples, max_samples=max_shap_samples
    )

    # 3. Model Feature Ranking Correlation (RF SHAP vs Robust MLP SHAP)
    rf_shap_imp = rf_shap["mean_abs_shap"]
    mlp_shap_imp = mlp_shap["mean_abs_shap"]

    rf_vector = [rf_shap_imp.get(f, 0.0) for f in feature_names]
    mlp_vector = [mlp_shap_imp.get(f, 0.0) for f in feature_names]

    rho_res = spearmanr(rf_vector, mlp_vector)
    rho = float(rho_res.statistic) if not np.isnan(rho_res.statistic) else 0.0
    p_val = float(rho_res.pvalue) if not np.isnan(rho_res.pvalue) else 1.0

    # 4. Zero-IQR Feature Pathology Detection
    iqr_per_feature = {}
    for i, fname in enumerate(feature_names):
        col = X[:, i]
        q75, q25 = np.percentile(col, [75, 25])
        iqr = float(q75 - q25)
        iqr_per_feature[fname] = iqr

    zero_iqr_features = [f for f, iqr in iqr_per_feature.items() if iqr == 0.0]

    return {
        "dataset_name": dataset_name,
        "feature_names": feature_names,
        "rf_gini": rf_perm["gini_importance"],
        "rf_permutation": rf_perm["permutation_importance"],
        "rf_shap": rf_shap["mean_abs_shap"],
        "mlp_permutation": mlp_perm["permutation_importance"],
        "mlp_shap": mlp_shap["mean_abs_shap"],
        "spearman_rho_rf_vs_mlp": rho,
        "spearman_p_value": p_val,
        "zero_iqr_features": zero_iqr_features,
        "n_samples_analyzed": len(X),
    }


def compute_cross_dataset_stability(
    dataset_results: Dict[str, Dict[str, Any]],
    feature_names: List[str],
) -> Dict[str, Any]:
    """Computes cross-dataset feature attribution stability metrics across datasets.

    Aggregates mean absolute attributions, calculates overall feature ranks,
    and measures rank correlation stability across dataset pairs.
    """
    dataset_names = list(dataset_results.keys())

    # Aggregate mean SHAP per feature across datasets for Option C weighted models
    # Option C attribution = 0.7 * RF_SHAP_norm + 0.3 * MLP_SHAP_norm
    combined_attributions = {ds: {} for ds in dataset_names}
    rankings_per_dataset = {}

    for ds_name, res in dataset_results.items():
        rf_shap = res["rf_shap"]
        mlp_shap = res["mlp_shap"]

        # Normalize per dataset
        rf_sum = sum(rf_shap.values()) if sum(rf_shap.values()) > 0 else 1.0
        mlp_sum = sum(mlp_shap.values()) if sum(mlp_shap.values()) > 0 else 1.0

        comb = {}
        for f in feature_names:
            rf_norm = rf_shap.get(f, 0.0) / rf_sum
            mlp_norm = mlp_shap.get(f, 0.0) / mlp_sum
            comb[f] = 0.7 * rf_norm + 0.3 * mlp_norm
        
        combined_attributions[ds_name] = comb
        
        # Rank features for this dataset
        sorted_feats = sorted(feature_names, key=lambda f: comb[f], reverse=True)
        rankings_per_dataset[ds_name] = {f: r + 1 for r, f in enumerate(sorted_feats)}

    # Mean attribution and mean rank across all datasets
    mean_combined_attr = {}
    mean_rank = {}

    for f in feature_names:
        attrs = [combined_attributions[ds][f] for ds in dataset_names]
        ranks = [rankings_per_dataset[ds][f] for ds in dataset_names]
        mean_combined_attr[f] = float(np.mean(attrs))
        mean_rank[f] = float(np.mean(ranks))

    sorted_global_features = sorted(feature_names, key=lambda f: mean_combined_attr[f], reverse=True)

    # Pairwise cross-dataset rank correlations
    pairwise_rhos = {}
    for i in range(len(dataset_names)):
        for j in range(i + 1, len(dataset_names)):
            ds1, ds2 = dataset_names[i], dataset_names[j]
            v1 = [combined_attributions[ds1][f] for f in feature_names]
            v2 = [combined_attributions[ds2][f] for f in feature_names]
            r = spearmanr(v1, v2)
            pairwise_rhos[f"{ds1}_vs_{ds2}"] = float(r.statistic) if not np.isnan(r.statistic) else 0.0

    mean_cross_dataset_rho = float(np.mean(list(pairwise_rhos.values()))) if pairwise_rhos else 1.0

    return {
        "dataset_names": dataset_names,
        "combined_attributions": combined_attributions,
        "rankings_per_dataset": rankings_per_dataset,
        "global_mean_attribution": dict(sorted(mean_combined_attr.items(), key=lambda x: x[1], reverse=True)),
        "global_mean_rank": dict(sorted(mean_rank.items(), key=lambda x: x[1])),
        "sorted_global_features": sorted_global_features,
        "pairwise_rank_correlations": pairwise_rhos,
        "mean_cross_dataset_spearman_rho": mean_cross_dataset_rho,
    }
