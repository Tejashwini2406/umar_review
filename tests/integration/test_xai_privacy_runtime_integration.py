"""Integration test for InferenceEngine with XAI local explanation generation."""
import pytest
import numpy as np

from iot_ids.runtime.engine import InferenceEngine
from iot_ids.runtime.schema import STANDARDIZED_21_FEATURES


def test_inference_engine_xai_integration():
    engine = InferenceEngine(profile="standardized_21", dataset="ToN-IoT")

    # Sample 21-D telemetry vector matching STANDARDIZED_21_FEATURES
    sample_dict = {feat: 1.0 for feat in STANDARDIZED_21_FEATURES}

    # Run inference with explain=True
    res = engine.predict(sample_dict, explain=True, top_k=5)

    assert isinstance(res, dict)
    assert "prediction" in res
    assert "probability" in res
    assert "rf_probability" in res
    assert "robust_mlp_probability" in res
    assert "explanation" in res
    assert res["explanation"]["option_c_probability"] == pytest.approx(
        res["probability"]
    )

    exp = res["explanation"]
    assert "option_c_probability" in exp
    assert "final_decision" in exp
    assert "top_k_features" in exp
    assert len(exp["top_k_features"]) == 5

    for item in exp["top_k_features"]:
        assert "feature" in item
        assert "fused_probability_delta" in item
        assert "direction" in item
