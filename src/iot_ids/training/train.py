"""Reproducible Multi-Level Pipeline Training Engine.

Trains Random Forest or Logistic Regression classifiers on canonical materialized datasets,
fits preprocessor, aligner, and calibrator, and exports deployable model artifacts via ModelRegistry.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from iot_ids.config import PipelineConfig
from iot_ids.features.canonical.flow_builder import CANONICAL_18_FEATURE_NAMES
from iot_ids.experiments.preprocessing import FeaturePreprocessor
from iot_ids.adaptation.alignment import UnsupervisedFeatureAligner
from iot_ids.adaptation.calibration import TargetThresholdCalibrator
from iot_ids.pipeline.system import IDSSystemPipeline
from iot_ids.registry.manager import ModelRegistry


def load_split_dataframe(dataset_dir: Path, split: str = "train") -> pd.DataFrame:
    """Loads split DataFrame from Parquet or CSV."""
    parquet_path = dataset_dir / f"{split}.parquet"
    csv_path = dataset_dir / f"{split}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Missing split file for {dataset_dir}: {split}")


def train_and_export_pipeline(config: PipelineConfig, stage3_dir: Optional[Path] = None) -> Path:
    """Trains a multi-level operational pipeline and exports it as a deployable ModelRegistry artifact."""
    stage3_root = Path(stage3_dir) if stage3_dir else Path("data/processed/stage3")
    src_dir = stage3_root / config.source_domain

    if not src_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {src_dir}")

    print(f"Loading source dataset ({config.source_domain}) from: {src_dir}")
    df_src_train = load_split_dataframe(src_dir, "train")

    # 1. Feature Preprocessing & One-Hot Expansion
    preprocessor = FeaturePreprocessor(feature_names=config.feature_names)
    X_src_proc = preprocessor.fit_transform(df_src_train)
    y_src = df_src_train["label"].values

    print(f"Training {config.model_family} on {X_src_proc.shape[0]} flows across {X_src_proc.shape[1]} columns...")
    if config.model_family.lower() in ("randomforest", "rf"):
        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=42,
            n_jobs=-1
        )
    elif config.model_family.lower() in ("logisticregression", "lr"):
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unsupported model family: {config.model_family}")

    model.fit(X_src_proc, y_src)

    # 2. Optional Feature Alignment
    aligner = None
    if config.target_domain:
        tgt_dir = stage3_root / config.target_domain
        if tgt_dir.exists():
            print(f"Fitting UnsupervisedFeatureAligner for target domain: {config.target_domain}")
            df_tgt_train = load_split_dataframe(tgt_dir, "train")
            aligner = UnsupervisedFeatureAligner(feature_names=config.feature_names)
            aligner.fit(df_src_train, df_tgt_train)

    # 3. Optional Target Threshold Calibration
    calibrator = None
    if config.target_domain:
        tgt_dir = stage3_root / config.target_domain
        if tgt_dir.exists():
            print(f"Calibrating target threshold on validation split: {tgt_dir}")
            df_tgt_val = load_split_dataframe(tgt_dir, "val")
            if aligner is not None:
                X_val_proc = aligner.transform(df_tgt_val)
            else:
                X_val_proc = preprocessor.transform(df_tgt_val)

            y_val = df_tgt_val["label"].values
            probs_val = model.predict_proba(X_val_proc)[:, 1]
            calibrator = TargetThresholdCalibrator()
            calibrator.fit(probs_val, y_val)

    # 4. Assemble and Save Artifact
    pipeline = IDSSystemPipeline(
        model=model,
        preprocessor=preprocessor,
        feature_profile=config.feature_profile,
        feature_names=config.feature_names,
        aligner=aligner,
        calibrator=calibrator,
        decision_threshold=config.decision_threshold,
        alpha=0.1,
        window_seconds=config.window_seconds,
    )

    metadata = {
        "version": "1.0.0",
        "config": config.to_dict(),
        "num_semantic_features": len(config.feature_names),
        "num_matrix_columns": X_src_proc.shape[1],
    }

    export_path = Path(config.artifact_dir)
    saved_dir = ModelRegistry.save_pipeline(pipeline, export_path, metadata=metadata)
    print(f"Successfully exported deployable pipeline artifact to: {saved_dir}")

    return saved_dir
