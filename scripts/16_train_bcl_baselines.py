"""Train the minimal frozen detector and benign-only AE needed for BCL."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from iot_ids.models.autoencoder import Autoencoder
from iot_ids.runtime.engine import InferenceEngine, MLPModule
from iot_ids.runtime.schema import STANDARDIZED_21_FEATURES
from iot_ids.utils.paths import REPO_ROOT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_training_split(dataset: str, data_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_dir = data_dir / dataset
    manifest_path = dataset_dir / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing corrected split manifest: {manifest_path}; run Stage 3 first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_strategy = (
        "official_source_partitions"
        if dataset == "CICIoT2023"
        else "bot_iot_source_file_holdout_v1"
        if dataset == "BoT-IoT"
        else "seeded_stratified_random"
    )
    if (
        manifest.get("sampling_method") != "uniform_reservoir"
        or manifest.get("split_strategy") != expected_strategy
    ):
        raise ValueError(f"{dataset} split manifest does not match the corrected protocol")

    train = pd.read_parquet(dataset_dir / "train.parquet")
    missing = [
        name
        for name in (*STANDARDIZED_21_FEATURES, "label")
        if name not in train.columns
    ]
    if missing:
        raise ValueError(f"{dataset} training split is missing columns: {missing}")
    if set(train["label"].unique()) != {0, 1}:
        raise ValueError(f"{dataset} training split must contain both binary labels")
    actual_counts = {
        "benign": int((train["label"] == 0).sum()),
        "attack": int((train["label"] == 1).sum()),
    }
    if actual_counts != manifest["label_counts"]["train"]:
        raise ValueError(f"{dataset} training labels do not match the split manifest")
    return train, manifest


def _train_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
) -> MLPModule:
    model = MLPModule(input_dim=features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.float32).unsqueeze(1),
    )
    generator = torch.Generator().manual_seed(seed)
    drop_last = len(dataset) % batch_size == 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=drop_last,
    )

    model.train()
    for _ in range(epochs):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def _train_autoencoder(
    benign_features: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Autoencoder:
    if len(benign_features) < 2:
        raise ValueError("At least two benign training samples are required for AE training")
    model = Autoencoder(input_dim=benign_features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.as_tensor(benign_features, dtype=torch.float32))
    generator = torch.Generator().manual_seed(seed)
    drop_last = len(dataset) % batch_size == 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=drop_last,
    )

    model.train()
    for _ in range(epochs):
        for (batch_features,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_features)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def train_bcl_baseline(
    dataset: str,
    data_dir: Path,
    models_base_dir: Path,
    seed: int = 42,
    epochs: int = 15,
    batch_size: int = 256,
    trees: int = 100,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 2 or trees < 1:
        raise ValueError("epochs, trees, and batch size must be positive (batch size >= 2)")
    train, split_manifest = _load_training_split(dataset, data_dir)
    feature_frame = train[list(STANDARDIZED_21_FEATURES)]
    labels = train["label"].to_numpy(dtype=np.int8)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(feature_frame)
    scaler.feature_names = list(STANDARDIZED_21_FEATURES)

    rf = RandomForestClassifier(
        n_estimators=trees,
        max_depth=15,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(scaled_features, labels)
    mlp = _train_mlp(scaled_features, labels, epochs, batch_size, seed)
    autoencoder = _train_autoencoder(
        scaled_features[labels == 0], epochs, batch_size, seed + 1
    )

    model_dir = models_base_dir / dataset
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / "prep_standardized.joblib")
    joblib.dump(rf, model_dir / "rf_model.joblib")
    torch.save(mlp.state_dict(), model_dir / "mlp_adversarial.pt")
    torch.save(autoencoder.state_dict(), model_dir / "ae_model.pt")
    (model_dir / "training_split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )

    config = {
        "dataset": dataset,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "rf_estimators": trees,
        "rf_max_depth": 15,
        "ae_training_rows": int((labels == 0).sum()),
        "train_rows": len(train),
        "split_strategy": split_manifest["split_strategy"],
        "feature_names": list(STANDARDIZED_21_FEATURES),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "pytorch_version": torch.__version__,
        "split_manifest_sha256": _sha256(data_dir / dataset / "split_manifest.json"),
        "artifact_sha256": {
            filename: _sha256(model_dir / filename)
            for filename in (
                "prep_standardized.joblib",
                "rf_model.joblib",
                "mlp_adversarial.pt",
                "ae_model.pt",
            )
        },
    }
    (model_dir / "training_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    # Fail fast if the saved artifacts cannot be loaded by the production inference path.
    InferenceEngine(
        profile="standardized_21",
        dataset=dataset,
        models_base_dir=models_base_dir,
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("Edge-IIoTset", "NF-ToN-IoT-v2", "ToN-IoT", "CICIoT2023", "BoT-IoT"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "stage3_corrected",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=REPO_ROOT / "models" / "corrected_run"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--trees", type=int, default=100)
    args = parser.parse_args()
    result = train_bcl_baseline(
        args.dataset,
        args.data_dir,
        args.models_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        trees=args.trees,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
