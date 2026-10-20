import numpy as np
import pytest

from iot_ids.adversarial.black_box import spsa_attack


def test_spsa_attack_reduces_actual_score_within_constraints():
    inputs = np.asarray([[0.8, 0.0, 1.0], [0.7, 0.0, 0.0]])

    def score(values):
        return values[:, 0] + 0.25 * values[:, 1]

    result = spsa_attack(
        score,
        inputs,
        epsilon=0.4,
        alpha=0.05,
        steps=12,
        probe_scale=0.02,
        feature_mask=np.asarray([1, 1, 0]),
        x_min=np.asarray([-1.0, -1.0, 0.0]),
        x_max=np.asarray([1.0, 1.0, 1.0]),
        seed=7,
    )

    assert np.all(score(result.adversarial) < score(inputs))
    assert np.all(np.abs(result.adversarial - inputs) <= 0.4 + 1e-12)
    assert np.array_equal(result.adversarial[:, 2], inputs[:, 2])
    assert result.queries_per_sample == 24
    assert result.total_queries == 48


def test_spsa_attack_is_deterministic_for_fixed_seed():
    values = np.asarray([[0.5, -0.2]])
    kwargs = {
        "epsilon": 0.2,
        "alpha": 0.025,
        "steps": 4,
        "probe_scale": 0.01,
        "feature_mask": np.ones(2),
        "x_min": np.full(2, -1.0),
        "x_max": np.full(2, 1.0),
        "seed": 19,
    }
    first = spsa_attack(lambda x: x[:, 0] - x[:, 1], values, **kwargs)
    second = spsa_attack(lambda x: x[:, 0] - x[:, 1], values, **kwargs)

    np.testing.assert_array_equal(first.adversarial, second.adversarial)


def test_spsa_attack_rejects_invalid_oracle_output():
    with pytest.raises(ValueError, match="one score per input"):
        spsa_attack(
            lambda values: np.asarray([0.5]),
            np.zeros((2, 1)),
            epsilon=0.1,
            alpha=0.01,
            steps=1,
            probe_scale=0.01,
            feature_mask=np.ones(1),
            x_min=np.full(1, -1.0),
            x_max=np.full(1, 1.0),
        )
