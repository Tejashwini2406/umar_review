"""Real Deployable Pipeline Model Export Script.

Trains a Random Forest classifier on materialized ToN-IoT telemetry (parquet),
fits FeaturePreprocessor, UnsupervisedFeatureAligner, and TargetThresholdCalibrator,
and exports the unified deployable pipeline artifact to models/final/deployable_artifact/.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from iot_ids.features.canonical.flow_builder import CANONICAL_18_FEATURE_NAMES
from iot_ids.experiments.preprocessing import FeaturePreprocessor
from iot_ids.adaptation.alignment import UnsupervisedFeatureAligner
from iot_ids.adaptation.calibration import TargetThresholdCalibrator
from iot_ids.pipeline.system import IDSSystemPipeline
from iot_ids.registry.manager import ModelRegistry


def load_dataset_split(dataset_dir: Path, split: str = "train") -> pd.DataFrame:
    parquet_path = dataset_dir / f"{split}.parquet"
    csv_path = dataset_dir / f"{split}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Neither {parquet_path} nor {csv_path} exists.")


def export_deployable_model_artifact():
    stage3_dir = Path("data/processed/stage3")
    export_dir = Path("models/final/deployable_artifact")
    
    src_dir = stage3_dir / "ToN-IoT"
    tgt_dir = stage3_dir / "Edge-IIoTset"

    print(f"Loading source dataset from: {src_dir}")
    df_src = load_dataset_split(src_dir, "train")

    # 1. Feature Preprocessing & One-Hot Expansion
    preprocessor = FeaturePreprocessor(feature_names=CANONICAL_18_FEATURE_NAMES)
    X_src_proc = preprocessor.fit_transform(df_src)
    y_src = df_src["label"].values

    print(f"Training Random Forest on {X_src_proc.shape[0]} source flows across {X_src_proc.shape[1]} columns...")
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_src_proc, y_src)

    # 2. Unsupervised Feature Alignment (if target dataset available)
    aligner = None
    if tgt_dir.exists():
        print(f"Fitting UnsupervisedFeatureAligner on target dataset: {tgt_dir}")
        df_tgt_train = load_dataset_split(tgt_dir, "train")
        aligner = UnsupervisedFeatureAligner(feature_names=CANONICAL_18_FEATURE_NAMES)
        aligner.fit(df_src, df_tgt_train)

    # 3. Target Threshold Calibration (using target val split)
    calibrator = None
    if tgt_dir.exists():
        print(f"Calibrating target decision threshold on validation set: {tgt_dir}")
        df_tgt_val = load_dataset_split(tgt_dir, "val")
        if aligner is not None:
            X_val_proc = aligner.transform(df_tgt_val)
        else:
            X_val_proc = preprocessor.transform(df_tgt_val)
            
        y_val = df_tgt_val["label"].values
        probs_val = model.predict_proba(X_val_proc)[:, 1]
        calibrator = TargetThresholdCalibrator()
        calibrator.fit(probs_val, y_val)

    # 4. Assemble & Save IDSSystemPipeline Artifact
    pipeline = IDSSystemPipeline(
        model=model,
        preprocessor=preprocessor,
        feature_profile="full_multilevel",
        feature_names=CANONICAL_18_FEATURE_NAMES,
        aligner=aligner,
        calibrator=calibrator,
        decision_threshold=0.5,
    )

    metadata = {
        "version": "1.0.0",
        "source_dataset": "ToN-IoT",
        "target_dataset": "Edge-IIoTset",
        "model_family": "RandomForest",
        "n_estimators": 100,
        "feature_profile": "full_multilevel",
        "num_semantic_features": len(CANONICAL_18_FEATURE_NAMES),
        "num_matrix_columns": X_src_proc.shape[1],
    }

    out_path = ModelRegistry.save_pipeline(pipeline, export_dir, metadata=metadata)
    print(f"Successfully exported deployable pipeline artifact to: {out_path}")


if __name__ == "__main__":
    export_deployable_model_artifact()
