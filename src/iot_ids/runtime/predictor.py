"""Production IDSPredictor API Wrapper for Runtime Inference Engine.

Delegates single flow and batch predictions to InferenceEngine while tracking
health inspection status and operational state.
"""
from __future__ import annotations

from typing import Dict, Any, List, Union, Optional
import pandas as pd

from iot_ids.runtime.engine import InferenceEngine


class IDSPredictor:
    """Production deployable predictor wrapper for multi-level IoT intrusion detection."""

    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    @classmethod
    def from_profile_and_dataset(
        cls,
        profile: str = "standardized_21",
        dataset: str = "ToN-IoT",
        models_base_dir: Optional[str] = None,
        decision_threshold: float = 0.5,
    ) -> IDSPredictor:
        """Instantiates an IDSPredictor from profile and dataset identifiers."""
        engine = InferenceEngine(
            profile=profile,
            dataset=dataset,
            models_base_dir=models_base_dir,
            decision_threshold=decision_threshold,
        )
        return cls(engine=engine)

    def health_check(self) -> Dict[str, Any]:
        """Returns runtime health inspection metadata."""
        return {
            "status": "HEALTHY",
            "profile": self.engine.profile,
            "dataset": self.engine.dataset,
            "feature_dimension": self.engine.expected_dim,
            "rf_model_loaded": self.engine.rf_model is not None,
            "robust_mlp_loaded": self.engine.robust_mlp_model is not None,
            "preprocessor_loaded": self.engine.preprocessor is not None,
            "decision_threshold": self.engine.decision_threshold,
        }

    def predict_flow(self, flow: Dict[str, Any]) -> Dict[str, Any]:
        """Runs inference on a single flow dictionary."""
        return self.engine.predict(flow)

    def predict_dataframe(self, df_flows: pd.DataFrame) -> List[Dict[str, Any]]:
        """Runs batch inference over a DataFrame of canonical flow features."""
        return self.engine.predict(df_flows)

    def reset_state(self) -> None:
        """Resets predictor state buffers."""
        pass
