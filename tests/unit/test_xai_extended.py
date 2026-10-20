"""Unit tests for Global and Local XAI modules."""
import numpy as np
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn

from iot_ids.xai.global_xai import analyze_dataset_global_xai, compute_cross_dataset_stability
from iot_ids.xai.local_xai import explain_local_sample


class DummyMLP(nn.Module):
    def __init__(self, input_dim: int = 21):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def xai_setup():
    rng = np.random.default_rng(42)
    feature_names = [f"feat_{i}" for i in range(21)]

    X = rng.normal(0, 1, size=(100, 21))
    y = rng.choice([0, 1], size=100)

    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X, y)

    mlp = DummyMLP(21)

    return rf, mlp, X, y, feature_names


def test_global_xai_analysis(xai_setup):
    rf, mlp, X, y, feature_names = xai_setup

    res = analyze_dataset_global_xai(
        rf_model=rf,
        mlp_model=mlp,
        X=X,
        y=y,
        feature_names=feature_names,
        dataset_name="TestDataset",
        max_shap_samples=50,
        background_samples=20,
    )

    assert res["dataset_name"] == "TestDataset"
    assert len(res["rf_gini"]) == 21
    assert len(res["rf_shap"]) == 21
    assert len(res["mlp_shap"]) == 21
    assert -1.0 <= res["spearman_rho_rf_vs_mlp"] <= 1.0


def test_cross_dataset_stability(xai_setup):
    rf, mlp, X, y, feature_names = xai_setup

    res1 = analyze_dataset_global_xai(rf, mlp, X, y, feature_names, "DS1", 30, 10)
    res2 = analyze_dataset_global_xai(rf, mlp, X, y, feature_names, "DS2", 30, 10)

    stability = compute_cross_dataset_stability({"DS1": res1, "DS2": res2}, feature_names)

    assert "global_mean_attribution" in stability
    assert len(stability["sorted_global_features"]) == 21
    assert -1.0 <= stability["mean_cross_dataset_spearman_rho"] <= 1.0


def test_local_xai_explanation(xai_setup):
    rf, mlp, X, _, feature_names = xai_setup

    sample = X[0]
    local_exp = explain_local_sample(
        rf_model=rf,
        mlp_model=mlp,
        x_single=sample,
        feature_names=feature_names,
        top_k=5,
    )

    assert "option_c_probability" in local_exp
    assert "rf_probability" in local_exp
    assert "robust_mlp_probability" in local_exp
    assert local_exp["final_decision"] in ["ATTACK", "BENIGN"]
    assert "single-feature ablation" in local_exp["methodology"]
    assert len(local_exp["top_k_features"]) == 5

    explanation_by_feature = {
        item["feature"]: item for item in local_exp["top_k_features"]
    }
    top_feature = local_exp["top_k_features"][0]["feature"]
    feature_index = feature_names.index(top_feature)
    ablated = sample.copy()
    ablated[feature_index] = 0.0
    expected_probability = (
        0.7 * rf.predict_proba(ablated.reshape(1, -1))[0, 1]
        + 0.3
        * torch.sigmoid(mlp(torch.tensor(ablated.reshape(1, -1), dtype=torch.float32)))
        .item()
    )
    assert explanation_by_feature[top_feature]["fused_probability_delta"] == pytest.approx(
        local_exp["option_c_probability"] - expected_probability
    )

    for item in local_exp["top_k_features"]:
        assert "feature" in item
        assert "direction" in item
        assert item["direction"] in ["PUSHES_ATTACK", "PUSHES_BENIGN", "NO_CHANGE"]
