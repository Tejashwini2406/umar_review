"""Runtime adapter combining the frozen IDS ensemble, autoencoder, and BCL."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from iot_ids.models.autoencoder import Autoencoder
from iot_ids.models.ensemble.bcl import BehavioralConsistencyLayer
from iot_ids.runtime.engine import InferenceEngine
from iot_ids.runtime.schema import validate_input_schema


class BCLInferenceEngine:
    """Adds calibrated behavioral-consistency alerts to frozen model predictions."""

    def __init__(
        self,
        engine: InferenceEngine,
        consistency_layer: BehavioralConsistencyLayer,
        autoencoder: Autoencoder | None = None,
    ):
        if engine.profile != "standardized_21":
            raise ValueError("BCL inference requires the standardized_21 profile")

        self.engine = engine
        self.consistency_layer = consistency_layer
        self.autoencoder = autoencoder or self._load_autoencoder()
        self.autoencoder.eval()

    @classmethod
    def from_artifacts(
        cls,
        dataset: str,
        calibration_path: str | Path,
        models_base_dir: str | Path | None = None,
    ) -> BCLInferenceEngine:
        """Load dataset model artifacts and a separately calibrated BCL artifact."""
        engine = InferenceEngine(
            profile="standardized_21",
            dataset=dataset,
            models_base_dir=models_base_dir,
        )
        consistency_layer = BehavioralConsistencyLayer.load(calibration_path)
        return cls(engine, consistency_layer)

    def calibrate(
        self,
        validation_inputs: pd.DataFrame | dict[str, Any] | list[dict[str, Any]] | np.ndarray,
        validation_labels: np.ndarray,
    ) -> BehavioralConsistencyLayer:
        """Fit the false-positive threshold on an ordered validation stream."""
        _, rf, mlp, ae = self._component_scores(validation_inputs)
        return self.consistency_layer.fit(rf, mlp, ae, validation_labels)

    def predict(
        self,
        inputs: pd.DataFrame | dict[str, Any] | list[dict[str, Any]] | np.ndarray,
        explain: bool = False,
        top_k: int = 5,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Return ordinary IDS predictions augmented with BCL signal details."""
        if self.consistency_layer.threshold is None:
            raise RuntimeError("Calibrate the BCL with validation data before prediction")
        base_results, rf, mlp, ae = self._component_scores(inputs, explain, top_k)
        single = isinstance(base_results, dict)
        results = [base_results] if single else base_results

        for result, rf_probability, mlp_probability, ae_error in zip(results, rf, mlp, ae):
            decision = self.consistency_layer.predict_one(
                float(rf_probability),
                float(mlp_probability),
                float(ae_error),
            )
            result["behavioral_consistency"] = asdict(decision)

        return results[0] if single else results

    def reset_state(self) -> None:
        """Reset sequential BCL state while retaining the fitted calibration."""
        self.consistency_layer.reset_state()

    def _component_scores(
        self,
        inputs: pd.DataFrame | dict[str, Any] | list[dict[str, Any]] | np.ndarray,
        explain: bool = False,
        top_k: int = 5,
    ) -> tuple[dict[str, Any] | list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
        predictions = self.engine.predict(inputs, explain=explain, top_k=top_k)
        results = [predictions] if isinstance(predictions, dict) else predictions
        rf = np.asarray([item["rf_probability"] for item in results], dtype=np.float64)
        mlp = np.asarray(
            [item["robust_mlp_probability"] for item in results],
            dtype=np.float64,
        )

        canonical = validate_input_schema(
            inputs,
            profile=self.engine.profile,
            expected_feature_names=self.engine.feature_names,
        )
        transformed = self.engine._transform_preprocessor_frozen(canonical)
        tensor = torch.as_tensor(transformed, dtype=torch.float32)
        with torch.no_grad():
            reconstruction = self.autoencoder(tensor)
            errors = torch.mean((tensor - reconstruction) ** 2, dim=1).cpu().numpy()
        return predictions, rf, mlp, errors

    def _load_autoencoder(self) -> Autoencoder:
        model_path = self.engine.model_dir / "ae_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing Autoencoder artifact: {model_path}")
        model = Autoencoder(input_dim=self.engine.expected_dim)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model
