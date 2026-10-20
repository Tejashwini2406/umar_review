"""Stage 4 Preprocessing Engine.

Applies log1p transformation to heavy-tailed continuous rate metrics and fits RobustScaler
STRICTLY on source training data to prevent future or target leakage.
"""
from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Rate metrics subject to log1p scaling
LOG1P_CANDIDATES = [
    "flow_duration",
    "flow_bytes_per_sec",
    "flow_pkts_per_sec",
    "mean_pkt_size",
    "temporal_iat_mean",
    "temporal_flow_rate_ewma",
    "temporal_byte_rate_ewma",
]


class FeaturePreprocessor:
    """Train-only preprocessor applying log1p transformation and RobustScaler."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.log_features = [f for f in feature_names if f in LOG1P_CANDIDATES]
        self.scaler = RobustScaler()
        self.is_fitted = False

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        """Fits preprocessing pipeline ONLY on source training data and returns scaled array."""
        sub_df = train_df[self.feature_names].copy().fillna(0.0)

        # Apply log1p to non-negative continuous rate metrics
        for col in self.log_features:
            sub_df[col] = np.log1p(np.clip(sub_df[col].values, 0.0, None))

        X_train = sub_df.values
        X_scaled = self.scaler.fit_transform(X_train)
        self.is_fitted = True
        return X_scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Applies pre-fitted transformation to validation or test data without calling fit()."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePreprocessor must be fitted on training data before calling transform()")

        sub_df = df[self.feature_names].copy().fillna(0.0)

        for col in self.log_features:
            sub_df[col] = np.log1p(np.clip(sub_df[col].values, 0.0, None))

        X_val = sub_df.values
        return self.scaler.transform(X_val)
