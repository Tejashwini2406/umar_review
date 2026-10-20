"""Comprehensive Unit & Integration Test Suite for Production Runtime Inference Engine.

Verifies:
- Group A: Artifact loading and 21-dimensional compatibility across all 4 datasets.
- Group B: Standardized Option C ensemble inference and exact arithmetic (P_ensemble = 0.7*P_RF + 0.3*P_MLP_Adv).
- Group C: Historical 13-feature backward compatibility.
- Group D: Fail-fast schema validation (missing/extra columns, wrong counts, invalid profiles/datasets).
- Group E: Preprocessor immutability (guaranteeing .fit() and .fit_transform() are NEVER invoked).
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from iot_ids.runtime.schema import validate_input_schema, STANDARDIZED_21_FEATURES
from iot_ids.runtime.engine import InferenceEngine
from iot_ids.runtime.predictor import IDSPredictor


# ---------------------------------------------------------------------------
# Test Group A: Standardized Artifact Loading & Dimension Checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset_name", ["ToN-IoT", "Edge-IIoTset", "NF-ToN-IoT-v2", "CICIoT2023"])
def test_group_a_artifact_loading(dataset_name: str):
    """Verifies that InferenceEngine loads all required artifacts and validates 21 dimensions."""
    engine = InferenceEngine(profile="standardized_21", dataset=dataset_name)
    assert engine.profile == "standardized_21"
    assert engine.dataset == dataset_name
    assert engine.expected_dim == 21
    assert engine.preprocessor is not None
    assert engine.rf_model is not None
    assert engine.robust_mlp_model is not None
    assert engine.rf_model.n_features_in_ == 21


# ---------------------------------------------------------------------------
# Test Group B: Standardized Inference & Ensemble Arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset_name", ["ToN-IoT", "Edge-IIoTset", "NF-ToN-IoT-v2", "CICIoT2023"])
def test_group_b_standardized_inference_and_arithmetic(dataset_name: str):
    """Verifies Option C probability ensemble arithmetic (P_ensemble = 0.7*P_RF + 0.3*P_MLP_Adv)."""
    engine = InferenceEngine(profile="standardized_21", dataset=dataset_name)

    # Load small controlled sample from test split
    test_pq = repo_root / "data" / "processed" / "stage3" / dataset_name / "test.parquet"
    assert test_pq.exists(), f"Missing test split for {dataset_name}"
    df_sample = pd.read_parquet(test_pq).head(10)[STANDARDIZED_21_FEATURES]

    # Predict via engine
    results = engine.predict(df_sample)
    assert isinstance(results, list)
    assert len(results) == 10

    for res in results:
        assert res["profile"] == "standardized_21"
        assert res["dataset"] == dataset_name
        assert res["feature_dimension"] == 21
        assert 0.0 <= res["probability"] <= 1.0
        assert 0.0 <= res["rf_probability"] <= 1.0
        assert 0.0 <= res["robust_mlp_probability"] <= 1.0

        # Exact Ensemble Arithmetic Validation
        expected_p = 0.7 * res["rf_probability"] + 0.3 * res["robust_mlp_probability"]
        np.testing.assert_allclose(res["probability"], expected_p, atol=1e-6)
        assert res["prediction"] == int(res["probability"] >= 0.5)


# ---------------------------------------------------------------------------
# Test Group C: Historical 13-Feature Compatibility
# ---------------------------------------------------------------------------

def test_group_c_historical_compatibility():
    """Verifies that profile='historical_13' successfully loads 13-feature legacy pipeline."""
    engine = InferenceEngine(profile="historical_13", dataset="ToN-IoT")
    assert engine.profile == "historical_13"
    assert engine.dataset == "ToN-IoT"
    assert engine.expected_dim == 13

    # Synthetic 13-feature input dict
    hist_cols = engine.feature_names
    assert len(hist_cols) == 13
    sample_dict = {col: 1.0 for col in hist_cols}

    res = engine.predict(sample_dict)
    assert isinstance(res, dict)
    assert res["profile"] == "historical_13"
    assert res["feature_dimension"] == 13
    assert 0.0 <= res["probability"] <= 1.0


# ---------------------------------------------------------------------------
# Test Group D: Fail-Fast Schema Validation
# ---------------------------------------------------------------------------

def test_group_d_schema_validation():
    """Tests schema validator failure cases (missing/extra columns, invalid profile/dataset)."""
    # 1. Invalid Profile
    with pytest.raises(ValueError, match="Unsupported profile"):
        InferenceEngine(profile="invalid_profile", dataset="ToN-IoT")

    # 2. Invalid Dataset for standardized profile
    with pytest.raises(ValueError, match="Dataset 'invalid_dataset' is not supported"):
        InferenceEngine(profile="standardized_21", dataset="invalid_dataset")

    # 3. Feature count mismatch (20 features instead of 21)
    df_20 = pd.DataFrame([{col: 1.0 for col in STANDARDIZED_21_FEATURES[:20]}])
    with pytest.raises(ValueError, match="requires exactly 21 canonical features"):
        validate_input_schema(df_20, profile="standardized_21")

    # 4. Extra unexpected feature
    dict_extra = {col: 1.0 for col in STANDARDIZED_21_FEATURES}
    dict_extra["unauthorized_column"] = 99.0
    with pytest.raises(ValueError, match="requires exactly 21 canonical features"):
        validate_input_schema(pd.DataFrame([dict_extra]), profile="standardized_21")

    # 5. Missing required feature
    dict_missing = {col: 1.0 for col in STANDARDIZED_21_FEATURES[1:]}
    dict_missing["fake_feature"] = 1.0
    with pytest.raises(ValueError, match="missing required features"):
        validate_input_schema(pd.DataFrame([dict_missing]), profile="standardized_21")

    # 6. NaN values
    dict_nan = {col: 1.0 for col in STANDARDIZED_21_FEATURES}
    dict_nan[STANDARDIZED_21_FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match="contain NaN values"):
        validate_input_schema(dict_nan, profile="standardized_21")

    # 7. Inf values
    dict_inf = {col: 1.0 for col in STANDARDIZED_21_FEATURES}
    dict_inf[STANDARDIZED_21_FEATURES[0]] = np.inf
    with pytest.raises(ValueError, match="contain Infinite values"):
        validate_input_schema(dict_inf, profile="standardized_21")


# ---------------------------------------------------------------------------
# Test Group E: Preprocessor Immutability (NO FIT)
# ---------------------------------------------------------------------------

def test_group_e_preprocessor_immutability(monkeypatch):
    """Demonstrates that runtime inference never calls fit() or fit_transform()."""
    engine = InferenceEngine(profile="standardized_21", dataset="ToN-IoT")

    fit_called = False
    fit_transform_called = False

    def forbidden_fit(*args, **kwargs):
        nonlocal fit_called
        fit_called = True
        raise RuntimeError("FORBIDDEN: fit() called during runtime inference!")

    def forbidden_fit_transform(*args, **kwargs):
        nonlocal fit_transform_called
        fit_transform_called = True
        raise RuntimeError("FORBIDDEN: fit_transform() called during runtime inference!")

    if hasattr(engine.preprocessor, "fit"):
        monkeypatch.setattr(engine.preprocessor, "fit", forbidden_fit)
    if hasattr(engine.preprocessor, "fit_transform"):
        monkeypatch.setattr(engine.preprocessor, "fit_transform", forbidden_fit_transform)

    # Execute prediction
    dict_sample = {col: 1.0 for col in STANDARDIZED_21_FEATURES}
    res = engine.predict(dict_sample)

    assert not fit_called, "fit() was improperly called during inference!"
    assert not fit_transform_called, "fit_transform() was improperly called during inference!"
    assert res["probability"] >= 0.0


# ---------------------------------------------------------------------------
# Test IDSPredictor Wrapper API
# ---------------------------------------------------------------------------

def test_ids_predictor_api():
    """Tests high-level IDSPredictor wrapper API."""
    predictor = IDSPredictor.from_profile_and_dataset(profile="standardized_21", dataset="CICIoT2023")
    health = predictor.health_check()

    assert health["status"] == "HEALTHY"
    assert health["profile"] == "standardized_21"
    assert health["dataset"] == "CICIoT2023"
    assert health["feature_dimension"] == 21
    assert health["rf_model_loaded"] is True
    assert health["robust_mlp_loaded"] is True

    dict_sample = {col: 1.0 for col in STANDARDIZED_21_FEATURES}
    alert = predictor.predict_flow(dict_sample)
    assert alert["dataset"] == "CICIoT2023"
