"""Stage 5 Unsupervised Domain Adaptation Engine.

Provides UnsupervisedFeatureAligner and UnsupervisedFeatureCorrector to align target-domain
feature distributions with the source training distribution without using any target labels.
"""
from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, QuantileTransformer

from iot_ids.experiments.preprocessing import LOG1P_CANDIDATES


class UnsupervisedFeatureAligner:
    """Aligns unlabeled target feature distributions with the source training distribution."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.log_features = [f for f in feature_names if f in LOG1P_CANDIDATES]
        self.source_scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        self.is_fitted = False

    def fit(self, source_train_df: pd.DataFrame, target_unlabeled_df: pd.DataFrame) -> UnsupervisedFeatureAligner:
        """Fits source scaler on source training data and target scaler on unlabeled target data."""
        # 1. Transform source
        s_sub = source_train_df[self.feature_names].copy().fillna(0.0)
        for col in self.log_features:
            s_sub[col] = np.log1p(np.clip(s_sub[col].values, 0.0, None))
        self.source_scaler.fit(s_sub.values)

        # 2. Transform target (UNLABELED features only)
        t_sub = target_unlabeled_df[self.feature_names].copy().fillna(0.0)
        for col in self.log_features:
            t_sub[col] = np.log1p(np.clip(t_sub[col].values, 0.0, None))
        self.target_scaler.fit(t_sub.values)

        self.is_fitted = True
        return self

    def transform_source(self, df: pd.DataFrame) -> np.ndarray:
        sub = df[self.feature_names].copy().fillna(0.0)
        for col in self.log_features:
            sub[col] = np.log1p(np.clip(sub[col].values, 0.0, None))
        return self.source_scaler.transform(sub.values)

    def transform_target(self, target_df: pd.DataFrame) -> np.ndarray:
        """Applies target-fitted scaler to target test features without touching target labels."""
        if not self.is_fitted:
            raise RuntimeError("UnsupervisedFeatureAligner must be fitted before calling transform_target()")
        sub = target_df[self.feature_names].copy().fillna(0.0)
        for col in self.log_features:
            sub[col] = np.log1p(np.clip(sub[col].values, 0.0, None))
        
        # Scale target features using target statistics, then map to source scale
        X_tgt_norm = self.target_scaler.transform(sub.values)
        return X_tgt_norm

    def transform(self, target_df: pd.DataFrame) -> np.ndarray:
        """Alias for transform_target() to provide unified scikit-learn transformer compatibility."""
        return self.transform_target(target_df)


class UnsupervisedFeatureCorrector:
    """Applies feature-wise distribution correction to match source and target feature moments."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.log_features = [f for f in feature_names if f in LOG1P_CANDIDATES]
        self.src_mean = None
        self.src_std = None
        self.tgt_mean = None
        self.tgt_std = None
        self.is_fitted = False

    def fit(self, source_train_df: pd.DataFrame, target_unlabeled_df: pd.DataFrame) -> UnsupervisedFeatureCorrector:
        s_sub = source_train_df[self.feature_names].copy().fillna(0.0)
        for col in self.log_features:
            s_sub[col] = np.log1p(np.clip(s_sub[col].values, 0.0, None))
        
        t_sub = target_unlabeled_df[self.feature_names].copy().fillna(0.0)
        for col in self.log_features:
            t_sub[col] = np.log1p(np.clip(t_sub[col].values, 0.0, None))

        self.src_mean = s_sub.mean().values
        self.src_std = s_sub.std().replace(0.0, 1.0).values

        self.tgt_mean = t_sub.mean().values
        self.tgt_std = t_sub.std().replace(0.0, 1.0).values

        self.is_fitted = True
        return self

    def transform_source(self, df: pd.DataFrame) -> np.ndarray:
        sub = df[self.feature_names].copy().fillna(0.0)
        for col in self.log_features:
            sub[col] = np.log1p(np.clip(sub[col].values, 0.0, None))
        return (sub.values - self.src_mean) / self.src_std

    def transform_target(self, target_df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("UnsupervisedFeatureCorrector must be fitted before calling transform_target()")
        sub = target_df[self.feature_names].copy().fillna(0.0)
        for col in self.log_features:
            sub[col] = np.log1p(np.clip(sub[col].values, 0.0, None))

        # Standardize target features to zero-mean/unit-var, then project into source moment space
        X_tgt_norm = (sub.values - self.tgt_mean) / self.tgt_std
        X_corrected = X_tgt_norm * self.src_std + self.src_mean
        return X_corrected
