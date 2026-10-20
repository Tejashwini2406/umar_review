from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def get_model_feature_columns(dataset: str, profile: str = "in_domain", levels: list[str] | None = None) -> list[str]:
    """Returns the exact ordered feature column list from the canonical schema.
    
    Args:
        dataset: Dataset name (Edge-IIoTset, ToN-IoT).
        profile: 'in_domain' for full multi-level features, 'cross_domain' for
                 the strict 8-feature F_common profile.
        levels: Optional list of semantic levels (e.g., ["common", "temporal"]).
    """
    schema_path = Path(__file__).resolve().parents[3] / "configs" / "features" / "canonical_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    if profile == "cross_domain":
        cols = schema["profiles"]["F_COMMON"]
    elif profile == "in_domain":
        in_domain_map = {
            "Edge-IIoTset": "IN_DOMAIN_EDGE_IIOT",
            "ToN-IoT": "IN_DOMAIN_TON_IOT",
            "ton_iot": "IN_DOMAIN_TON_IOT",
            "edge-iiotset": "IN_DOMAIN_EDGE_IIOT"
        }
        profile_key = in_domain_map.get(dataset)
        if profile_key and profile_key in schema["profiles"]:
            cols = schema["profiles"][profile_key]
        else:
            raise ValueError(f"No in-domain profile defined for dataset: {dataset}")
    else:
        raise ValueError(f"Invalid profile requested: {profile}. Must be 'in_domain' or 'cross_domain'.")

    # Ablation filtering
    if levels is not None:
        feature_defs = {f["name"]: f["level"] for f in schema.get("features", [])}
        filtered_cols = []
        for c in cols:
            level = feature_defs.get(c, "common")
            if level in levels:
                filtered_cols.append(c)
        return filtered_cols

    return cols


class PreprocessingPipeline:
    """Robust preprocessing pipeline applying Log1p transformation and RobustScaler
    to numerical network quantities, fitted strictly on Train data.
    """

    def __init__(self, numeric_cols: list[str], clip_bounds: tuple[float, float] | None = (-10.0, 10.0)):
        self.numeric_cols = list(numeric_cols)
        self.scaler = RobustScaler()
        self.clip_bounds = clip_bounds
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        X_num = df[self.numeric_cols].astype(float).fillna(0.0).values
        X_log = np.log1p(np.maximum(0, X_num))
        X_scaled = self.scaler.fit_transform(X_log)
        clip_b = getattr(self, "clip_bounds", (-10.0, 10.0))
        if clip_b is not None:
            X_scaled = np.clip(X_scaled, clip_b[0], clip_b[1])
        self.fitted = True
        return X_scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("PreprocessingPipeline must be fitted before transform.")
        X_num = df[self.numeric_cols].astype(float).fillna(0.0).values
        X_log = np.log1p(np.maximum(0, X_num))
        X_scaled = self.scaler.transform(X_log)
        clip_b = getattr(self, "clip_bounds", (-10.0, 10.0))
        if clip_b is not None:
            X_scaled = np.clip(X_scaled, clip_b[0], clip_b[1])
        return X_scaled
