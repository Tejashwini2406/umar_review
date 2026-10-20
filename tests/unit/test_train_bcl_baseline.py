"""End-to-end tests for the lightweight BCL model artifact trainer."""
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from iot_ids.models.ensemble.bcl import BehavioralConsistencyLayer
from iot_ids.runtime.bcl_engine import BCLInferenceEngine
from iot_ids.runtime.engine import InferenceEngine
from iot_ids.runtime.schema import STANDARDIZED_21_FEATURES

TRAIN_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "16_train_bcl_baselines.py"
TRAINER = runpy.run_path(str(TRAIN_SCRIPT))["train_bcl_baseline"]


def _split_frame(labels: np.ndarray, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(len(labels), len(STANDARDIZED_21_FEATURES)))
    features[:, 0] += labels * 2.0
    frame = pd.DataFrame(features, columns=STANDARDIZED_21_FEATURES)
    frame["label"] = labels
    return frame


def test_training_artifacts_load_and_run_through_bcl(tmp_path):
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    dataset_dir = data_dir / "ToN-IoT"
    dataset_dir.mkdir(parents=True)
    train_labels = np.tile([0, 1], 20)
    validation_labels = np.tile([0, 1], 10)
    test_labels = np.tile([0, 1], 10)
    manifest = {
        "sampling_method": "uniform_reservoir",
        "split_strategy": "seeded_stratified_random",
        "label_counts": {
            "train": {"benign": 20, "attack": 20},
            "validation": {"benign": 10, "attack": 10},
            "test": {"benign": 10, "attack": 10},
        },
    }
    (dataset_dir / "split_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    _split_frame(train_labels, 1).to_parquet(dataset_dir / "train.parquet")
    _split_frame(validation_labels, 2).to_parquet(dataset_dir / "val.parquet")
    _split_frame(test_labels, 3).to_parquet(dataset_dir / "test.parquet")

    config = TRAINER(
        "ToN-IoT",
        data_dir,
        model_dir,
        seed=42,
        epochs=1,
        batch_size=4,
        trees=5,
    )

    assert config["ae_training_rows"] == 20
    assert (model_dir / "ToN-IoT" / "ae_model.pt").is_file()
    assert (model_dir / "ToN-IoT" / "training_split_manifest.json").is_file()

    engine = InferenceEngine(
        profile="standardized_21",
        dataset="ToN-IoT",
        models_base_dir=model_dir,
    )
    bcl_runtime = BCLInferenceEngine(
        engine,
        BehavioralConsistencyLayer(target_false_positive_rate=0.1, history_size=3),
    )
    validation = pd.read_parquet(dataset_dir / "val.parquet")
    bcl_runtime.calibrate(
        validation[list(STANDARDIZED_21_FEATURES)],
        validation["label"].to_numpy(),
    )
    test = pd.read_parquet(dataset_dir / "test.parquet")
    predictions = bcl_runtime.predict(test[list(STANDARDIZED_21_FEATURES)].head(4))

    assert isinstance(predictions, list)
    assert all("behavioral_consistency" in prediction for prediction in predictions)
