from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iot_ids.models.ensemble.bcl import BehavioralConsistencyLayer
from iot_ids.runtime.bcl_engine import BCLInferenceEngine
from iot_ids.runtime.engine import InferenceEngine
from iot_ids.runtime.schema import STANDARDIZED_21_FEATURES


def _calibrated_bcl() -> BehavioralConsistencyLayer:
    size = 200
    probabilities = np.full(size, 0.05)
    return BehavioralConsistencyLayer(
        target_false_positive_rate=0.01,
        history_size=5,
    ).fit(
        rf_probabilities=probabilities,
        mlp_probabilities=probabilities,
        ae_errors=np.linspace(0.01, 0.20, size),
        labels=np.zeros(size, dtype=np.int8),
    )


def test_bcl_calibrates_only_on_benign_validation_rows() -> None:
    size = 200
    labels = np.zeros(size, dtype=np.int8)
    labels[-20:] = 1
    probabilities = np.full(size, 0.05)
    mlp_probabilities = probabilities.copy()
    mlp_probabilities[-20:] = 0.95

    ae_errors = np.linspace(0.01, 0.20, size)
    ae_errors[-20:] = 1000.0
    bcl = BehavioralConsistencyLayer(target_false_positive_rate=0.01).fit(
        probabilities,
        mlp_probabilities,
        ae_errors,
        labels,
    )

    assert bcl.threshold is not None
    assert bcl.calibration_false_positive_rate is not None
    assert bcl.calibration_false_positive_rate <= 0.01
    assert bcl.predict_one(0.05, 0.05, 1000.0).suspicious


def test_bcl_flags_disagreement_and_confidence_drift() -> None:
    bcl = _calibrated_bcl()

    disagreement = bcl.predict_one(0.05, 0.95, 0.01)
    assert disagreement.suspicious
    assert disagreement.model_disagreement == pytest.approx(0.9)

    bcl.reset_state()
    bcl.predict_one(0.05, 0.05, 0.01)
    drift = bcl.predict_one(0.95, 0.95, 0.01)
    assert drift.suspicious
    assert drift.confidence_drift == pytest.approx(0.9)


def test_bcl_flags_signals_below_the_benign_calibration_range() -> None:
    bcl = BehavioralConsistencyLayer(
        target_false_positive_rate=0.01,
        use_stream_history=False,
    ).fit(
        np.full(200, 0.2),
        np.full(200, 0.4),
        np.full(200, 0.1),
        np.zeros(200, dtype=np.int8),
    )

    decision = bcl.predict_one(0.2, 0.2, 0.1)

    assert decision.model_disagreement == 0.0
    assert decision.suspicious


def test_bcl_reset_clears_history_but_keeps_calibration() -> None:
    bcl = _calibrated_bcl()
    bcl.predict_one(0.05, 0.05, 0.01)
    bcl.reset_state()

    first_after_reset = bcl.predict_one(0.05, 0.05, 0.01)
    assert first_after_reset.confidence_drift == 0.0
    assert first_after_reset.prediction_instability == 0.0
    assert first_after_reset.threshold == bcl.threshold


def test_bcl_can_disable_unverified_input_order_dependence() -> None:
    bcl = BehavioralConsistencyLayer(
        target_false_positive_rate=0.01,
        history_size=5,
        use_stream_history=False,
    )
    probabilities = np.full(200, 0.05)
    bcl.fit(
        probabilities,
        probabilities,
        np.linspace(0.01, 0.20, 200),
        np.zeros(200, dtype=np.int8),
    )
    first = bcl.predict_one(0.05, 0.05, 0.1)
    bcl.reset_state()
    bcl.predict_one(0.95, 0.95, 0.1)
    second = bcl.predict_one(0.05, 0.05, 0.1)

    assert first.score == second.score
    assert first.confidence_drift == 0.0
    assert first.prediction_instability == 0.0
    assert second.confidence_drift == 0.0
    assert second.prediction_instability == 0.0


def test_bcl_rejects_unusable_calibration_and_invalid_scores() -> None:
    bcl = BehavioralConsistencyLayer(target_false_positive_rate=0.01)
    with pytest.raises(ValueError, match="At least 100 benign"):
        bcl.fit(
            np.full(99, 0.1),
            np.full(99, 0.1),
            np.full(99, 0.1),
            np.zeros(99),
        )

    bcl = _calibrated_bcl()
    with pytest.raises(ValueError, match="Model probabilities"):
        bcl.predict_one(1.1, 0.5, 0.1)
    with pytest.raises(ValueError, match="non-negative"):
        bcl.predict_one(0.1, 0.5, -0.1)


def test_bcl_requires_calibration_before_prediction() -> None:
    bcl = BehavioralConsistencyLayer()
    with pytest.raises(RuntimeError, match="must be fitted"):
        bcl.predict_one(0.1, 0.1, 0.01)


def test_bcl_calibration_round_trip(tmp_path) -> None:
    bcl = _calibrated_bcl()
    calibration_path = tmp_path / "bcl.json"
    bcl.save(calibration_path)
    loaded = BehavioralConsistencyLayer.load(calibration_path)

    assert loaded.threshold == bcl.threshold
    assert loaded.calibration_false_positive_rate == bcl.calibration_false_positive_rate
    assert loaded.predict_one(0.05, 0.95, 0.01).suspicious


def test_bcl_calibration_round_trip_preserves_history_setting(tmp_path) -> None:
    bcl = BehavioralConsistencyLayer(
        target_false_positive_rate=0.01,
        use_stream_history=False,
    ).fit(
        np.full(200, 0.05),
        np.full(200, 0.05),
        np.linspace(0.01, 0.20, 200),
        np.zeros(200, dtype=np.int8),
    )
    calibration_path = tmp_path / "stateless-bcl.json"
    bcl.save(calibration_path)

    loaded = BehavioralConsistencyLayer.load(calibration_path)

    assert not loaded.use_stream_history
    assert loaded.predict_one(0.05, 0.05, 0.01).confidence_drift == 0.0


def test_bcl_loads_legacy_scale_calibration(tmp_path) -> None:
    calibration_path = tmp_path / "legacy-bcl.json"
    calibration_path.write_text(
        """{
          "version": 2,
          "target_false_positive_rate": 0.01,
          "history_size": 5,
          "use_stream_history": false,
          "threshold": 1.0,
          "signal_scales": {
            "model_disagreement": 1.0,
            "confidence_drift": 1.0,
            "prediction_instability": 1.0,
            "autoencoder_error": 1.0
          },
          "calibration_false_positive_rate": 0.0
        }""",
        encoding="utf-8",
    )

    loaded = BehavioralConsistencyLayer.load(calibration_path)

    assert loaded.predict_one(0.05, 0.95, 0.01).score == pytest.approx(0.9)


def test_bcl_runtime_calibrates_and_augments_predictions() -> None:
    dataset = "ToN-IoT"
    engine = InferenceEngine(profile="standardized_21", dataset=dataset)
    layer = BehavioralConsistencyLayer(target_false_positive_rate=0.01)
    runtime = BCLInferenceEngine(engine, layer)

    data_dir = engine.model_dir.parents[2] / "data" / "processed" / "stage3" / dataset
    validation = pd.read_parquet(data_dir / "val.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    runtime.calibrate(
        validation[STANDARDIZED_21_FEATURES],
        validation["label"].to_numpy(),
    )
    results = runtime.predict(test[STANDARDIZED_21_FEATURES].head(3))

    assert isinstance(results, list)
    assert len(results) == 3
    assert "behavioral_consistency" in results[0]
    assert "autoencoder_deviation" in results[0]["behavioral_consistency"]
