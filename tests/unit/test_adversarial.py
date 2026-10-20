from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

from iot_ids.adversarial.attacks import fgsm_attack, pgd_attack
from iot_ids.adversarial.evaluator import evaluate_adversarial_sample
from iot_ids.models.ensemble.risk_layer import RiskLayer


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


def test_fgsm_and_pgd_discrete_masking() -> None:
    # 4 features: 2 continuous, 2 discrete
    X = torch.tensor([[1.0, 2.0, 0.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([1.0], dtype=torch.float32)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    x_min = torch.tensor([[-5.0, -5.0, 0.0, 0.0]], dtype=torch.float32)
    x_max = torch.tensor([[5.0, 5.0, 1.0, 1.0]], dtype=torch.float32)

    mlp = DummyMLP(4)

    adv_fgsm = fgsm_attack(mlp, X, y, epsilon=0.5, continuous_mask=mask, x_min=x_min, x_max=x_max)
    adv_pgd = pgd_attack(mlp, X, y, epsilon=0.5, continuous_mask=mask, x_min=x_min, x_max=x_max, steps=5)

    # Discrete features (index 2 and 3) must remain 100% unchanged!
    assert float(adv_fgsm[0, 2]) == 0.0
    assert float(adv_fgsm[0, 3]) == 1.0
    assert float(adv_pgd[0, 2]) == 0.0
    assert float(adv_pgd[0, 3]) == 1.0


def test_adversarial_evaluator() -> None:
    X_clean = np.random.randn(20, 4)
    y_true = np.array([0, 1] * 10)

    rf = RandomForestClassifier(n_estimators=5, random_state=42)
    rf.fit(X_clean, y_true)

    mlp = DummyMLP(4)
    ae = DummyAE(4)
    risk_layer = RiskLayer(tau_sup=0.5, tau_ae=0.8)

    X_adv = X_clean + 0.1

    res = evaluate_adversarial_sample(rf, mlp, ae, risk_layer, X_clean, X_adv, y_true)
    assert "attack_success_rate" in res
    assert "ae_catch_rate" in res
