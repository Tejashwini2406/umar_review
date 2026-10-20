"""Unit tests for Privacy-Aware Data Minimization and DP-SGD modules."""

import numpy as np
import pytest
import torch
from torch import nn

from iot_ids.privacy.data_minimization import (
    DataMinimizationPolicy,
    extract_minimized_representation,
)
from iot_ids.privacy.dp_sgd import (
    DPConfig,
    compute_dp_epsilon,
    train_robust_mlp_dp,
    train_step_dp_sgd,
)


class DummyMLP(nn.Module):
    def __init__(self, input_dim: int = 21):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


def test_data_minimization_compliance():
    feature_names = list(DataMinimizationPolicy.ALLOWED_FEATURE_CATEGORIES.keys())
    audit = DataMinimizationPolicy.verify_schema_minimization(feature_names)

    assert audit["is_compliant"] is True
    assert audit["total_features"] == 21
    assert len(audit["violations_found"]) == 0

    # Test forbidden type violation
    bad_features = feature_names + ["raw_payload_bytes"]
    bad_audit = DataMinimizationPolicy.verify_schema_minimization(bad_features)
    assert bad_audit["is_compliant"] is False
    assert "raw_payload_bytes" in bad_audit["violations_found"]


def test_minimized_representation_extraction():
    raw_dict = {"flow_duration": 1.25, "proto_tcp": 1.0}
    vec = extract_minimized_representation(raw_dict)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (21,)
    assert vec[0] == 1.25
    assert vec[7] == 1.0


def test_dp_epsilon_accounting():
    eps_low_noise = compute_dp_epsilon(epochs=10, sample_rate=0.05, noise_multiplier=0.5)
    eps_high_noise = compute_dp_epsilon(epochs=10, sample_rate=0.05, noise_multiplier=2.0)

    assert eps_low_noise > eps_high_noise
    assert eps_high_noise > 0.0
    assert compute_dp_epsilon(epochs=10, sample_rate=0.05, noise_multiplier=0.0) == float("inf")


def test_dp_sgd_training_loop():
    rng = np.random.default_rng(42)
    X_tr = rng.normal(0, 1, size=(100, 21))
    y_tr = rng.choice([0, 1], size=100)

    model = DummyMLP(21)
    dp_cfg = DPConfig(max_grad_norm=1.0, noise_multiplier=1.0, epochs=2, batch_size=32)

    res = train_robust_mlp_dp(model, X_tr, y_tr, dp_cfg, adv_steps=2)

    assert "privacy_epsilon" in res
    assert res["privacy_epsilon"] > 0.0
    assert res["privacy_delta"] == 1e-5
    assert len(res["history"]) == 2
    assert res["accounted_steps"] == 8
    assert res["noise_source"] == "operating-system cryptographic randomness"
    assert all("loss" not in epoch for epoch in res["history"])


class TwoWeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Parameter(torch.tensor(0.0))
        self.second = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.first * x[:, :1] + self.second * x[:, 1:2]


def test_dp_step_clips_global_per_example_gradient_norm():
    model = TwoWeightModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    config = DPConfig(max_grad_norm=0.25, noise_multiplier=0.0)

    train_step_dp_sgd(
        model,
        optimizer,
        nn.BCEWithLogitsLoss(),
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([0.0]),
        config,
    )

    expected = 0.25 / np.sqrt(2.0)
    assert np.isclose(model.first.item(), -expected, atol=1e-6)
    assert np.isclose(model.second.item(), -expected, atol=1e-6)


def test_dp_accounting_handles_edges_and_rejects_invalid_values():
    assert compute_dp_epsilon(steps=0, sample_rate=0.1) == 0.0
    assert compute_dp_epsilon(steps=10, sample_rate=0.1, noise_multiplier=0.0) == float("inf")
    with pytest.raises(ValueError, match="sample_rate"):
        compute_dp_epsilon(steps=1, sample_rate=1.1)
    with pytest.raises(ValueError, match="max_grad_norm"):
        DPConfig(max_grad_norm=0.0)


def test_data_minimization_rejects_unknown_and_duplicate_schemas():
    allowed = list(DataMinimizationPolicy.ALLOWED_FEATURE_CATEGORIES)
    unknown = allowed[:-1] + ["unapproved_statistic"]
    duplicate = allowed[:-1] + [allowed[0]]

    unknown_audit = DataMinimizationPolicy.verify_schema_minimization(unknown)
    duplicate_audit = DataMinimizationPolicy.verify_schema_minimization(duplicate)

    assert not unknown_audit["is_compliant"]
    assert "unapproved_statistic" in unknown_audit["unknown_features"]
    assert not duplicate_audit["is_compliant"]
    assert duplicate_audit["duplicate_features"] == [allowed[0].casefold()]


def test_empty_poisson_draws_use_noise_without_forcing_samples(monkeypatch):
    from iot_ids.privacy import dp_sgd

    monkeypatch.setattr(
        dp_sgd,
        "_poisson_sample_mask",
        lambda dataset_size, sample_rate: np.zeros(dataset_size, dtype=bool),
    )
    monkeypatch.setattr(
        dp_sgd,
        "_secure_gaussian_like",
        lambda reference: torch.zeros_like(reference),
    )
    model = DummyMLP()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    config = DPConfig(epochs=2, batch_size=3, noise_multiplier=1.0)
    result = train_robust_mlp_dp(
        model,
        np.ones((8, 21), dtype=np.float32),
        np.zeros(8, dtype=np.float32),
        config,
        adv_steps=0,
    )

    assert result["accounted_steps"] == 6
    assert result["optimizer_updates"] == 6
    assert all(
        torch.equal(initial, current) for initial, current in zip(before, model.parameters())
    )
