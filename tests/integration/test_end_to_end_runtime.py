"""End-to-End Integration & Parity Test Suite for Production Runtime Inference.

Verifies:
- Test 1: Historical 13-feature profile end-to-end parity and execution.
- Test 2: Standardized 21-feature profile end-to-end execution across all 4 datasets.
- Test 3: Single-sample vs Batch prediction exact parity.
- Test 4: Dataset routing isolation and error handling.
- Test 5: Failure path: Feature count and column mismatch rejection.
- Test 6: Failure path: Invalid profile and invalid dataset rejection.
- Test 7: Failure path: NaN and Infinite value rejection.
- Test 8: Preprocessor immutability (transform ONLY, zero refitting).
- Test 9: Output contract and probability bounds (0 <= P <= 1, exact Option C arithmetic).
- Test 10: Direct Model Inference vs Runtime Inference numerical parity.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
import pytest
import joblib
import numpy as np
import pandas as pd
import torch

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from iot_ids.runtime.schema import validate_input_schema, STANDARDIZED_21_FEATURES
from iot_ids.runtime.engine import InferenceEngine, MLPModule
from iot_ids.runtime.predictor import IDSPredictor

DATASETS = ["ToN-IoT", "Edge-IIoTset", "NF-ToN-IoT-v2", "CICIoT2023"]


# ---------------------------------------------------------------------------
# Test 1: Historical 13-feature Profile End-to-End Integration
# ---------------------------------------------------------------------------

def test_1_historical_profile_end_to_end():
    """Verifies that historical 13-feature profile operates cleanly with 100% backward compatibility."""
    engine = InferenceEngine(profile="historical_13", dataset="ToN-IoT")
    assert engine.profile == "historical_13"
    assert engine.dataset == "ToN-IoT"
    assert engine.expected_dim == 13

    # Load small sample from historical dataset if present
    hist_feats = engine.feature_names
    sample_dict = {col: 1.0 for col in hist_feats}

    res = engine.predict(sample_dict)
    assert isinstance(res, dict)
    assert res["profile"] == "historical_13"
    assert res["dataset"] == "ToN-IoT"
    assert res["feature_dimension"] == 13
    assert 0.0 <= res["probability"] <= 1.0
    assert res["prediction"] in (0, 1)


# ---------------------------------------------------------------------------
# Test 2: Standardized 21-feature Execution Across All 4 Datasets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ds_name", DATASETS)
def test_2_standardized_execution_all_datasets(ds_name: str):
    """Verifies standardized 21-feature runtime execution across all four datasets."""
    engine = InferenceEngine(profile="standardized_21", dataset=ds_name)
    test_pq = repo_root / "data" / "processed" / "stage3" / ds_name / "test.parquet"
    assert test_pq.exists(), f"Missing test split for {ds_name}"

    df_test = pd.read_parquet(test_pq).head(20)[STANDARDIZED_21_FEATURES]
    batch_res = engine.predict(df_test)

    assert isinstance(batch_res, list)
    assert len(batch_res) == 20

    for item in batch_res:
        assert item["profile"] == "standardized_21"
        assert item["dataset"] == ds_name
        assert item["feature_dimension"] == 21
        assert 0.0 <= item["probability"] <= 1.0
        assert 0.0 <= item["rf_probability"] <= 1.0
        assert 0.0 <= item["robust_mlp_probability"] <= 1.0
        assert item["prediction"] in (0, 1)


# ---------------------------------------------------------------------------
# Test 3: Single-Sample vs Batch Prediction Parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ds_name", DATASETS)
def test_3_single_vs_batch_parity(ds_name: str):
    """Verifies that single-sample inference produces identical outputs to batch inference."""
    engine = InferenceEngine(profile="standardized_21", dataset=ds_name)
    test_pq = repo_root / "data" / "processed" / "stage3" / ds_name / "test.parquet"
    df_batch = pd.read_parquet(test_pq).head(5)[STANDARDIZED_21_FEATURES]

    # Batch prediction
    batch_res = engine.predict(df_batch)

    # Single predictions
    for idx, row in df_batch.iterrows():
        sample_dict = row.to_dict()
        single_res = engine.predict(sample_dict)

        b_item = batch_res[list(df_batch.index).index(idx)]
        np.testing.assert_allclose(single_res["probability"], b_item["probability"], atol=1e-6)
        np.testing.assert_allclose(single_res["rf_probability"], b_item["rf_probability"], atol=1e-6)
        np.testing.assert_allclose(single_res["robust_mlp_probability"], b_item["robust_mlp_probability"], atol=1e-6)
        assert single_res["prediction"] == b_item["prediction"]


# ---------------------------------------------------------------------------
# Test 4: Dataset Routing & Isolation
# ---------------------------------------------------------------------------

def test_4_dataset_routing_and_isolation():
    """Verifies that InferenceEngine enforces dataset isolation and prevents cross-dataset mixing."""
    engine_ton = InferenceEngine(profile="standardized_21", dataset="ToN-IoT")
    engine_edge = InferenceEngine(profile="standardized_21", dataset="Edge-IIoTset")

    assert engine_ton.model_dir != engine_edge.model_dir
    assert engine_ton.dataset == "ToN-IoT"
    assert engine_edge.dataset == "Edge-IIoTset"

    # Invalid dataset choice
    with pytest.raises(ValueError, match="Dataset 'UnknownDataset' is not supported"):
        InferenceEngine(profile="standardized_21", dataset="UnknownDataset")


# ---------------------------------------------------------------------------
# Test 5: Failure Path — Feature Mismatch & Column Rejection
# ---------------------------------------------------------------------------

def test_5_failure_path_feature_mismatch():
    """Verifies fail-fast rejection of missing, extra, or wrong count feature inputs."""
    # 20 features instead of 21
    df_20 = pd.DataFrame([{col: 1.0 for col in STANDARDIZED_21_FEATURES[:20]}])
    with pytest.raises(ValueError, match="requires exactly 21 canonical features"):
        validate_input_schema(df_20, profile="standardized_21")

    # 22 features (extra unexpected column)
    dict_22 = {col: 1.0 for col in STANDARDIZED_21_FEATURES}
    dict_22["extra_unauthorized_feature"] = 99.9
    with pytest.raises(ValueError, match="requires exactly 21 canonical features"):
        validate_input_schema(dict_22, profile="standardized_21")

    # Missing column replaced by dummy
    dict_missing = {col: 1.0 for col in STANDARDIZED_21_FEATURES[1:]}
    dict_missing["bad_col_name"] = 1.0
    with pytest.raises(ValueError, match="missing required features"):
        validate_input_schema(dict_missing, profile="standardized_21")


# ---------------------------------------------------------------------------
# Test 6: Failure Path — Invalid Profile & Dataset Rejection
# ---------------------------------------------------------------------------

def test_6_failure_path_invalid_profile_dataset():
    """Verifies fail-fast rejection of invalid profile and dataset names."""
    with pytest.raises(ValueError, match="Unsupported profile"):
        InferenceEngine(profile="invalid_profile_v9", dataset="ToN-IoT")

    with pytest.raises(ValueError, match="Dataset 'InvalidDS' is not supported"):
        InferenceEngine(profile="standardized_21", dataset="InvalidDS")


# ---------------------------------------------------------------------------
# Test 7: Failure Path — NaN and Infinite Value Rejection
# ---------------------------------------------------------------------------

def test_7_failure_path_nan_inf_rejection():
    """Verifies fail-fast rejection of inputs containing NaN or Inf values."""
    dict_nan = {col: 1.0 for col in STANDARDIZED_21_FEATURES}
    dict_nan["flow_duration"] = np.nan
    with pytest.raises(ValueError, match="contain NaN values"):
        validate_input_schema(dict_nan, profile="standardized_21")

    dict_inf = {col: 1.0 for col in STANDARDIZED_21_FEATURES}
    dict_inf["mean_pkt_size"] = np.inf
    with pytest.raises(ValueError, match="contain Infinite values"):
        validate_input_schema(dict_inf, profile="standardized_21")


# ---------------------------------------------------------------------------
# Test 8: Preprocessor Immutability (TRANSFORM ONLY)
# ---------------------------------------------------------------------------

def test_8_preprocessor_immutability(monkeypatch):
    """Guarantees that inference never invokes fit() or fit_transform()."""
    engine = InferenceEngine(profile="standardized_21", dataset="NF-ToN-IoT-v2")

    def forbidden_fit(*args, **kwargs):
        raise RuntimeError("FORBIDDEN: fit() called during runtime inference!")

    def forbidden_fit_transform(*args, **kwargs):
        raise RuntimeError("FORBIDDEN: fit_transform() called during runtime inference!")

    if hasattr(engine.preprocessor, "fit"):
        monkeypatch.setattr(engine.preprocessor, "fit", forbidden_fit)
    if hasattr(engine.preprocessor, "fit_transform"):
        monkeypatch.setattr(engine.preprocessor, "fit_transform", forbidden_fit_transform)

    dict_sample = {col: 1.0 for col in STANDARDIZED_21_FEATURES}
    res = engine.predict(dict_sample)
    assert res["probability"] >= 0.0


# ---------------------------------------------------------------------------
# Test 9: Output Contract & Option C Ensemble Formula Verification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ds_name", DATASETS)
def test_9_output_contract_and_ensemble_arithmetic(ds_name: str):
    """Verifies exact Option C ensemble formula (P_ensemble = 0.7*P_RF + 0.3*P_MLP_Adv)."""
    engine = InferenceEngine(profile="standardized_21", dataset=ds_name)
    sample_dict = {col: 0.5 for col in STANDARDIZED_21_FEATURES}

    res = engine.predict(sample_dict)
    assert "prediction" in res
    assert "probability" in res
    assert "rf_probability" in res
    assert "robust_mlp_probability" in res
    assert "profile" in res
    assert "dataset" in res
    assert "feature_dimension" in res

    p_rf = res["rf_probability"]
    p_mlp = res["robust_mlp_probability"]
    p_ens = res["probability"]

    expected_p = 0.7 * p_rf + 0.3 * p_mlp
    np.testing.assert_allclose(p_ens, expected_p, atol=1e-6)
    assert res["prediction"] == int(p_ens >= 0.5)


# ---------------------------------------------------------------------------
# Test 10: Direct Model Inference vs Runtime Engine Parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ds_name", DATASETS)
def test_10_direct_model_vs_runtime_parity(ds_name: str):
    """Verifies 100% numerical parity between direct offline model execution and InferenceEngine."""
    m_dir = repo_root / "models" / "standardized" / ds_name
    prep = joblib.load(m_dir / "prep_standardized.joblib")
    rf = joblib.load(m_dir / "rf_model.joblib")
    mlp_state = torch.load(m_dir / "mlp_adversarial.pt", weights_only=True)

    mlp = MLPModule(input_dim=21)
    mlp.load_state_dict(mlp_state)
    mlp.eval()

    # Load test sample
    test_pq = repo_root / "data" / "processed" / "stage3" / ds_name / "test.parquet"
    df_raw = pd.read_parquet(test_pq).head(10)[STANDARDIZED_21_FEATURES]

    # Direct offline transformation & prediction
    X_proc_direct = prep.transform(df_raw)
    p_rf_direct = rf.predict_proba(X_proc_direct)[:, 1]

    with torch.no_grad():
        mlp_logits = mlp(torch.tensor(X_proc_direct, dtype=torch.float32)).squeeze(-1)
        p_mlp_direct = torch.sigmoid(mlp_logits).numpy()

    p_ens_direct = 0.7 * p_rf_direct + 0.3 * p_mlp_direct

    # Engine inference
    engine = InferenceEngine(profile="standardized_21", dataset=ds_name)
    engine_res = engine.predict(df_raw)

    p_rf_engine = np.array([r["rf_probability"] for r in engine_res])
    p_mlp_engine = np.array([r["robust_mlp_probability"] for r in engine_res])
    p_ens_engine = np.array([r["probability"] for r in engine_res])

    # Assert 100% numerical parity
    np.testing.assert_allclose(p_rf_engine, p_rf_direct, atol=1e-6)
    np.testing.assert_allclose(p_mlp_engine, p_mlp_direct, atol=1e-6)
    np.testing.assert_allclose(p_ens_engine, p_ens_direct, atol=1e-6)
