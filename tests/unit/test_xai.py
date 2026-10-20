from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

from iot_ids.xai.rf_explainer import explain_rf
from iot_ids.xai.mlp_explainer import explain_mlp
from iot_ids.xai.ae_explainer import explain_ae_reconstruction
from iot_ids.xai.consensus import compute_model_consensus


class DummyMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        return self.fc(x)


class DummyAE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc(x)


def test_rf_explainer() -> None:
    X = np.random.randn(50, 4)
    y = (X[:, 0] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    res = explain_rf(rf, X, y, ["f1", "f2", "f3", "f4"])
    assert "gini_importance" in res
    assert "permutation_importance" in res
    assert len(res["gini_importance"]) == 4


def test_mlp_explainer() -> None:
    X = np.random.randn(50, 4)
    y = (X[:, 0] > 0).astype(int)
    mlp = DummyMLP(4)

    res = explain_mlp(mlp, X, y, ["f1", "f2", "f3", "f4"])
    assert "permutation_importance" in res
    assert len(res["permutation_importance"]) == 4


def test_ae_explainer() -> None:
    X = np.random.randn(50, 4)
    ae = DummyAE(4)

    res = explain_ae_reconstruction(ae, X, ["f1", "f2", "f3", "f4"])
    assert "mean_mse_per_feature" in res
    assert "total_mse" in res
    assert len(res["mean_mse_per_feature"]) == 4


def test_consensus() -> None:
    imp1 = {"a": 0.8, "b": 0.5, "c": 0.1}
    imp2 = {"a": 0.9, "b": 0.4, "c": 0.2}

    res = compute_model_consensus(imp1, imp2)
    assert res["spearman_rho"] > 0.8
