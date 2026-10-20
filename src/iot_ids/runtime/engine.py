"""Production Profile-Aware Multi-Dataset Inference Engine.

Loads trained model artifact suites for standardized_21 or historical_13 profiles,
validates feature dimensionalities, executes frozen preprocessor transformations (NO refitting),
and evaluates probability fusion ensembles for deployable multi-level IoT intrusion detection.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from iot_ids.runtime.schema import validate_input_schema, STANDARDIZED_21_FEATURES
from iot_ids.utils.paths import REPO_ROOT


class MLPModule(nn.Module):
    """PyTorch MLP Classifier Architecture [input_dim -> 128 -> 64 -> 32 -> 1]."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InferenceEngine:
    """Production-grade profile-aware inference engine supporting Option C ensemble prediction."""

    SUPPORTED_STANDARDIZED_DATASETS = {"ToN-IoT", "Edge-IIoTset", "NF-ToN-IoT-v2", "CICIoT2023"}
    SUPPORTED_HISTORICAL_DATASETS = {"ToN-IoT", "Edge-IIoTset"}

    def __init__(
        self,
        profile: str = "standardized_21",
        dataset: str = "ToN-IoT",
        models_base_dir: Optional[Union[str, Path]] = None,
        decision_threshold: float = 0.5,
    ):
        self.profile = profile
        self.dataset = dataset
        self.decision_threshold = decision_threshold

        if self.profile == "standardized_21":
            if self.dataset not in self.SUPPORTED_STANDARDIZED_DATASETS:
                raise ValueError(
                    f"InferenceEngine Error: Dataset '{self.dataset}' is not supported for profile 'standardized_21'. "
                    f"Supported datasets: {sorted(list(self.SUPPORTED_STANDARDIZED_DATASETS))}"
                )
            base_dir = Path(models_base_dir) if models_base_dir else REPO_ROOT / "models" / "standardized"
            self.model_dir = base_dir / self.dataset
            self.expected_dim = 21
            self.feature_names = STANDARDIZED_21_FEATURES
        elif self.profile == "historical_13":
            if self.dataset not in self.SUPPORTED_HISTORICAL_DATASETS:
                raise ValueError(
                    f"InferenceEngine Error: Dataset '{self.dataset}' is not supported for profile 'historical_13'. "
                    f"Supported datasets: {sorted(list(self.SUPPORTED_HISTORICAL_DATASETS))}"
                )
            base_dir = Path(models_base_dir) if models_base_dir else REPO_ROOT / "models" / "final"
            self.model_dir = base_dir / self.dataset
            self.expected_dim = 13
            self.feature_names = None
        else:
            raise ValueError(f"InferenceEngine Error: Unsupported profile '{self.profile}'. Must be 'standardized_21' or 'historical_13'.")

        self.preprocessor = None
        self.rf_model = None
        self.robust_mlp_model = None
        self.historical_model = None

        self._load_and_validate_artifacts()

    def _load_and_validate_artifacts(self) -> None:
        """Loads model artifacts and validates feature input dimensions fail-fast."""
        if not self.model_dir.exists():
            raise FileNotFoundError(f"InferenceEngine Error: Model directory not found: {self.model_dir}")

        if self.profile == "standardized_21":
            prep_path = self.model_dir / "prep_standardized.joblib"
            rf_path = self.model_dir / "rf_model.joblib"
            mlp_path = self.model_dir / "mlp_adversarial.pt"

            if not prep_path.exists():
                raise FileNotFoundError(f"Missing preprocessor artifact: {prep_path}")
            if not rf_path.exists():
                raise FileNotFoundError(f"Missing Random Forest artifact: {rf_path}")
            if not mlp_path.exists():
                raise FileNotFoundError(f"Missing Adversarial MLP artifact: {mlp_path}")

            self.preprocessor = joblib.load(prep_path)
            self.rf_model = joblib.load(rf_path)

            # Dimension Validation
            prep_feats = getattr(self.preprocessor, "feature_names", None)
            if prep_feats is not None and len(prep_feats) != self.expected_dim:
                raise ValueError(
                    f"InferenceEngine Dimension Mismatch: Preprocessor has {len(prep_feats)} features, "
                    f"expected {self.expected_dim} for profile '{self.profile}'."
                )

            rf_dim = getattr(self.rf_model, "n_features_in_", None)
            if rf_dim is not None and rf_dim != self.expected_dim:
                raise ValueError(
                    f"InferenceEngine Dimension Mismatch: Random Forest n_features_in_={rf_dim}, "
                    f"expected {self.expected_dim} for profile '{self.profile}'."
                )

            # Load PyTorch Adversarial MLP
            mlp_state = torch.load(mlp_path, weights_only=True)
            mlp_in_dim = mlp_state["net.0.weight"].shape[1]
            if mlp_in_dim != self.expected_dim:
                raise ValueError(
                    f"InferenceEngine Dimension Mismatch: Robust MLP input_dim={mlp_in_dim}, "
                    f"expected {self.expected_dim} for profile '{self.profile}'."
                )

            self.robust_mlp_model = MLPModule(input_dim=self.expected_dim)
            self.robust_mlp_model.load_state_dict(mlp_state)
            self.robust_mlp_model.eval()

        else:
            # Historical 13-feature profile loading
            prep_path = self.model_dir / "prep_indomain.joblib"
            rf_path = self.model_dir / "rf_model.joblib"

            if not prep_path.exists():
                raise FileNotFoundError(f"Missing historical preprocessor artifact: {prep_path}")
            if not rf_path.exists():
                raise FileNotFoundError(f"Missing historical Random Forest artifact: {rf_path}")

            self.preprocessor = joblib.load(prep_path)
            self.rf_model = joblib.load(rf_path)

            hist_feats = getattr(self.preprocessor, "numeric_cols", getattr(self.preprocessor, "feature_names", None))
            if hist_feats is not None:
                self.feature_names = list(hist_feats)
                if len(self.feature_names) != self.expected_dim:
                    raise ValueError(
                        f"InferenceEngine Dimension Mismatch: Historical preprocessor has {len(self.feature_names)} features, "
                        f"expected {self.expected_dim} for profile '{self.profile}'."
                    )

            rf_dim = getattr(self.rf_model, "n_features_in_", None)
            if rf_dim is not None and rf_dim != self.expected_dim:
                raise ValueError(
                    f"InferenceEngine Dimension Mismatch: Historical RF n_features_in_={rf_dim}, "
                    f"expected {self.expected_dim} for profile '{self.profile}'."
                )

    def _transform_preprocessor_frozen(self, df_canonical: pd.DataFrame) -> np.ndarray:
        """Executes transform() ONLY on preprocessor without fit() or refitting."""
        if hasattr(self.preprocessor, "transform"):
            return self.preprocessor.transform(df_canonical)
        raise AttributeError("Preprocessor object lacks transform() method.")

    def predict(
        self,
        inputs: Union[pd.DataFrame, Dict[str, Any], np.ndarray, List[Dict[str, Any]]],
        explain: bool = False,
        top_k: int = 5,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Runs end-to-end inference over input flow samples.

        Args:
            inputs: Sample inputs as DataFrame, Dict, List of Dicts, or Numpy array.
            explain: If True, attaches local XAI explanation dict to each prediction output.
            top_k: Number of top influential features to include when explain=True.

        Returns:
            Structured prediction dictionary (single sample) or list of prediction dicts (batch).
        """
        # 1. Schema Validation & Canonical Ordering
        df_canonical = validate_input_schema(
            inputs,
            profile=self.profile,
            expected_feature_names=self.feature_names,
        )

        is_single = isinstance(inputs, dict) or (isinstance(inputs, np.ndarray) and inputs.ndim == 1)

        # 2. Preprocessing Transformation (FROZEN - NO FIT)
        X_proc = self._transform_preprocessor_frozen(df_canonical)

        # 3. Model Inference Execution
        if self.profile == "standardized_21":
            # RF Probabilities
            rf_probs = self.rf_model.predict_proba(X_proc)[:, 1]

            # Robust PyTorch MLP Probabilities
            with torch.no_grad():
                mlp_logits = self.robust_mlp_model(torch.tensor(X_proc, dtype=torch.float32)).squeeze(-1)
                mlp_probs = torch.sigmoid(mlp_logits).numpy()
                if mlp_probs.ndim == 0:
                    mlp_probs = np.array([float(mlp_probs)])

            # Option C Probability Fusion: 0.7 * P_RF + 0.3 * P_MLP_Adv
            ensemble_probs = 0.7 * rf_probs + 0.3 * mlp_probs

            results = []
            for i in range(len(ensemble_probs)):
                p_ens = float(ensemble_probs[i])
                p_rf = float(rf_probs[i])
                p_mlp = float(mlp_probs[i])
                is_anom = int(p_ens >= self.decision_threshold)

                res = {
                    "prediction": is_anom,
                    "probability": p_ens,
                    "rf_probability": p_rf,
                    "robust_mlp_probability": p_mlp,
                    "profile": self.profile,
                    "dataset": self.dataset,
                    "feature_dimension": self.expected_dim,
                }

                if explain and self.feature_names:
                    from iot_ids.xai.local_xai import explain_local_sample
                    local_exp = explain_local_sample(
                        rf_model=self.rf_model,
                        mlp_model=self.robust_mlp_model,
                        x_single=X_proc[i],
                        feature_names=self.feature_names,
                        top_k=top_k,
                        decision_threshold=self.decision_threshold,
                    )
                    res["explanation"] = local_exp

                results.append(res)

        else:
            # Historical 13-feature profile (RF only)
            rf_probs = self.rf_model.predict_proba(X_proc)[:, 1]
            results = []
            for i in range(len(rf_probs)):
                p_rf = float(rf_probs[i])
                is_anom = int(p_rf >= self.decision_threshold)
                results.append({
                    "prediction": is_anom,
                    "probability": p_rf,
                    "rf_probability": p_rf,
                    "profile": self.profile,
                    "dataset": self.dataset,
                    "feature_dimension": self.expected_dim,
                })

        if is_single and len(results) == 1:
            return results[0]
        return results
