from __future__ import annotations

from iot_ids.xai.rf_explainer import explain_rf
from iot_ids.xai.mlp_explainer import explain_mlp
from iot_ids.xai.ae_explainer import explain_ae_reconstruction
from iot_ids.xai.consensus import compute_model_consensus
from iot_ids.xai.shap_explainer import explain_rf_shap, explain_mlp_shap, compare_shap_rankings
from iot_ids.xai.global_xai import analyze_dataset_global_xai, compute_cross_dataset_stability
from iot_ids.xai.local_xai import explain_local_sample

__all__ = [
    "explain_rf",
    "explain_mlp",
    "explain_ae_reconstruction",
    "compute_model_consensus",
    "explain_rf_shap",
    "explain_mlp_shap",
    "compare_shap_rankings",
    "analyze_dataset_global_xai",
    "compute_cross_dataset_stability",
    "explain_local_sample",
]
