"""Verification helpers for corrected, model-bound experiment splits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from iot_ids.runtime.schema import STANDARDIZED_21_FEATURES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_verified_splits(
    dataset: str, data_dir: Path, model_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load validation/test data only after checking manifests and model hashes."""
    dataset_dir = data_dir / dataset
    manifest_path = dataset_dir / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing corrected split manifest: {manifest_path}; regenerate Stage 3 first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sampling_method") != "uniform_reservoir":
        raise ValueError(f"Unsupported sampling protocol in {manifest_path}")
    expected_strategy = (
        "official_source_partitions"
        if dataset == "CICIoT2023"
        else "bot_iot_source_file_holdout_v1"
        if dataset == "BoT-IoT"
        else "seeded_stratified_random"
    )
    if manifest.get("split_strategy") != expected_strategy:
        raise ValueError(
            f"Expected {expected_strategy!r} for {dataset}, found "
            f"{manifest.get('split_strategy')!r}"
        )

    model_manifest_path = model_dir / "training_split_manifest.json"
    if not model_manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing model training provenance: {model_manifest_path}; retrain models first"
        )
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if model_manifest != manifest:
        raise ValueError(
            "Model training manifest does not match the requested Stage 3 split manifest"
        )
    training_config_path = model_dir / "training_config.json"
    if not training_config_path.is_file():
        raise FileNotFoundError(
            f"Missing model training configuration: {training_config_path}; retrain models first"
        )
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    if (
        training_config.get("dataset") != dataset
        or training_config.get("feature_names") != list(STANDARDIZED_21_FEATURES)
        or training_config.get("split_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("Model training configuration does not match split provenance")
    expected_artifacts = {
        "prep_standardized.joblib",
        "rf_model.joblib",
        "mlp_adversarial.pt",
        "ae_model.pt",
    }
    artifact_hashes = training_config.get("artifact_sha256", {})
    if set(artifact_hashes) != expected_artifacts or any(
        not (model_dir / name).is_file()
        or sha256_file(model_dir / name) != artifact_hashes[name]
        for name in expected_artifacts
    ):
        raise ValueError("Trained model artifacts do not match their recorded hashes")

    validation = pd.read_parquet(dataset_dir / "val.parquet")
    test = pd.read_parquet(dataset_dir / "test.parquet")
    for split_name, split in (("validation", validation), ("test", test)):
        missing = [
            name
            for name in (*STANDARDIZED_21_FEATURES, "label")
            if name not in split.columns
        ]
        if missing:
            raise ValueError(f"{dataset} {split_name} split is missing columns: {missing}")
        if set(split["label"].unique()) != {0, 1}:
            raise ValueError(
                f"{dataset} {split_name} split must contain both binary labels"
            )
        split_key = "validation" if split_name == "validation" else "test"
        counts = manifest["label_counts"][split_key]
        actual_counts = {
            "benign": int((split["label"] == 0).sum()),
            "attack": int((split["label"] == 1).sum()),
        }
        if counts != actual_counts:
            raise ValueError(
                f"{dataset} {split_name} labels do not match the split manifest"
            )
    return validation, test
