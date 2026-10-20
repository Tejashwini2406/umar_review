"""Trained Multi-Level Pipeline Model Registry & Artifact Serialization Manager.

Saves and loads deployable pipeline artifacts containing model estimators, preprocessors,
aligners, threshold calibrators, and metadata independently of training code.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional
import joblib

from iot_ids.pipeline.system import IDSSystemPipeline


class ModelRegistry:
    """Registry manager for serializing and deserializing multi-level IDS pipeline artifacts."""

    @staticmethod
    def save_pipeline(
        pipeline: IDSSystemPipeline,
        save_dir: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Saves a complete IDSSystemPipeline artifact to a directory."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save component joblibs
        joblib.dump(pipeline.model, save_dir / "model.joblib")
        joblib.dump(pipeline.preprocessor, save_dir / "preprocessor.joblib")

        if pipeline.aligner is not None:
            joblib.dump(pipeline.aligner, save_dir / "aligner.joblib")

        if pipeline.calibrator is not None:
            joblib.dump(pipeline.calibrator, save_dir / "calibrator.joblib")

        # Save metadata configuration JSON
        meta_payload = {
            "feature_profile": pipeline.feature_profile,
            "feature_names": pipeline.feature_names,
            "decision_threshold": pipeline.decision_threshold,
            "has_aligner": pipeline.aligner is not None,
            "has_calibrator": pipeline.calibrator is not None,
            "user_metadata": metadata or {},
        }

        with open(save_dir / "pipeline_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

        return save_dir

    @staticmethod
    def load_pipeline(load_dir: Path) -> IDSSystemPipeline:
        """Loads an IDSSystemPipeline artifact from a saved directory."""
        load_dir = Path(load_dir).resolve()
        if not load_dir.exists():
            raise FileNotFoundError(f"Pipeline artifact directory not found: {load_dir}")
        if not load_dir.is_dir():
            raise NotADirectoryError(f"Pipeline artifact path is not a directory: {load_dir}")

        meta_path = load_dir / "pipeline_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing pipeline_metadata.json in artifact directory: {load_dir}")

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            raise ValueError(f"Corrupted or invalid pipeline_metadata.json in {load_dir}: {e}") from e

        model_path = load_dir / "model.joblib"
        preproc_path = load_dir / "preprocessor.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing required model.joblib in artifact directory: {load_dir}")
        if not preproc_path.exists():
            raise FileNotFoundError(f"Missing required preprocessor.joblib in artifact directory: {load_dir}")

        model = joblib.load(model_path)
        preprocessor = joblib.load(preproc_path)

        aligner_path = load_dir / "aligner.joblib"
        calib_path = load_dir / "calibrator.joblib"

        if meta.get("has_aligner"):
            if not aligner_path.is_file():
                raise FileNotFoundError(
                    f"Metadata requires aligner.joblib but it is missing from {load_dir}"
                )
            aligner = joblib.load(aligner_path)
        else:
            aligner = None

        if meta.get("has_calibrator"):
            if not calib_path.is_file():
                raise FileNotFoundError(
                    f"Metadata requires calibrator.joblib but it is missing from {load_dir}"
                )
            calibrator = joblib.load(calib_path)
        else:
            calibrator = None

        pipeline = IDSSystemPipeline(
            model=model,
            preprocessor=preprocessor,
            feature_profile=meta.get("feature_profile", "full_multilevel"),
            feature_names=meta.get("feature_names"),
            aligner=aligner,
            calibrator=calibrator,
            decision_threshold=meta.get("decision_threshold", 0.5),
        )
        return pipeline
