"""Tests for BCL evidence evaluation guards and metric output."""
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iot_ids.runtime.schema import STANDARDIZED_21_FEATURES

EVALUATION_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "15_evaluate_bcl.py"
)
EVALUATION = runpy.run_path(str(EVALUATION_SCRIPT))


def test_bcl_metric_summary_reports_false_positive_and_attack_detection_rates():
    labels = np.asarray([0, 0, 1, 1])
    results = [
        {
            "prediction": 0,
            "probability": 0.1,
            "behavioral_consistency": {
                "suspicious": False,
                "score": 0.1,
                "model_disagreement": 0.1,
                "confidence_drift": 0.0,
                "prediction_instability": 0.0,
                "autoencoder_deviation": 0.1,
            },
        },
        {
            "prediction": 0,
            "probability": 0.3,
            "behavioral_consistency": {
                "suspicious": True,
                "score": 0.8,
                "model_disagreement": 0.2,
                "confidence_drift": 0.0,
                "prediction_instability": 0.0,
                "autoencoder_deviation": 0.2,
            },
        },
        {
            "prediction": 1,
            "probability": 0.7,
            "behavioral_consistency": {
                "suspicious": False,
                "score": 0.2,
                "model_disagreement": 0.3,
                "confidence_drift": 0.0,
                "prediction_instability": 0.0,
                "autoencoder_deviation": 0.3,
            },
        },
        {
            "prediction": 1,
            "probability": 0.9,
            "behavioral_consistency": {
                "suspicious": True,
                "score": 0.9,
                "model_disagreement": 0.4,
                "confidence_drift": 0.0,
                "prediction_instability": 0.0,
                "autoencoder_deviation": 0.4,
            },
        },
    ]

    metrics = EVALUATION["_decision_metrics"](labels, results)

    assert metrics["benign_false_positive_rate"] == pytest.approx(0.5)
    assert metrics["benign_false_positive_rate_95ci"][0] < 0.5
    assert metrics["benign_false_positive_rate_95ci"][1] > 0.5
    assert metrics["attack_detection_rate"] == pytest.approx(0.5)
    assert metrics["attack_detection_rate_95ci"][0] < 0.5
    assert metrics["attack_detection_rate_95ci"][1] > 0.5
    assert metrics["detector_benign_false_positive_rate_at_0_5"] == pytest.approx(0.0)
    assert metrics["detector_attack_detection_rate_at_0_5"] == pytest.approx(1.0)
    assert metrics["bcl_score_roc_auc"] == pytest.approx(0.75)
    assert metrics["detector_roc_auc"] == pytest.approx(1.0)
    assert metrics["component_roc_auc"]["model_disagreement"] == pytest.approx(1.0)


def test_bcl_evaluation_refuses_splits_without_corrected_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing corrected split manifest"):
        EVALUATION["_read_verified_splits"](
            "ToN-IoT",
            tmp_path / "processed",
            tmp_path / "models" / "ToN-IoT",
        )


def test_bcl_evaluation_accepts_matching_corrected_provenance(tmp_path):
    data_dir = tmp_path / "processed"
    model_dir = tmp_path / "models" / "ToN-IoT"
    dataset_dir = data_dir / "ToN-IoT"
    dataset_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    manifest = {
        "sampling_method": "uniform_reservoir",
        "split_strategy": "seeded_stratified_random",
        "label_counts": {
            "validation": {"benign": 2, "attack": 2},
            "test": {"benign": 2, "attack": 2},
        },
    }
    manifest_json = json.dumps(manifest)
    manifest_path = dataset_dir / "split_manifest.json"
    manifest_path.write_text(manifest_json, encoding="utf-8")
    (model_dir / "training_split_manifest.json").write_text(
        manifest_json, encoding="utf-8"
    )
    artifact_names = (
        "prep_standardized.joblib",
        "rf_model.joblib",
        "mlp_adversarial.pt",
        "ae_model.pt",
    )
    for filename in artifact_names:
        (model_dir / filename).write_bytes(filename.encode("utf-8"))
    artifact_hashes = {
        filename: EVALUATION["_sha256"](model_dir / filename)
        for filename in artifact_names
    }
    training_config = {
        "dataset": "ToN-IoT",
        "feature_names": list(STANDARDIZED_21_FEATURES),
        "split_manifest_sha256": EVALUATION["_sha256"](manifest_path),
        "artifact_sha256": artifact_hashes,
    }
    (model_dir / "training_config.json").write_text(
        json.dumps(training_config), encoding="utf-8"
    )
    features = pd.DataFrame(
        np.zeros((4, len(STANDARDIZED_21_FEATURES))),
        columns=STANDARDIZED_21_FEATURES,
    )
    for split_name in ("val", "test"):
        split = features.copy()
        split["label"] = [0, 0, 1, 1]
        split.to_parquet(dataset_dir / f"{split_name}.parquet", index=False)

    validation, test = EVALUATION["_read_verified_splits"](
        "ToN-IoT", data_dir, model_dir
    )

    assert len(validation) == 4
    assert len(test) == 4
