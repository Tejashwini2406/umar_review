"""Stage 3 Statistical Quality Audit & Feature Analysis Engine.

Provides distribution profiling, cross-domain distribution shift analysis (KS test & Wasserstein distance),
feature-to-label mutual information calculation, correlation matrix computation, and Variance Inflation Factor (VIF).
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif


def compute_distribution_statistics(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Calculates comprehensive summary statistics for each feature in df."""
    stats_list = []
    for f in feature_names:
        if f not in df.columns:
            continue
        col = df[f].values
        col_clean = col[~np.isnan(col) & ~np.isinf(col)]
        
        stats_list.append({
            "feature": f,
            "count": len(col),
            "mean": float(np.mean(col_clean)) if len(col_clean) > 0 else 0.0,
            "std": float(np.std(col_clean)) if len(col_clean) > 0 else 0.0,
            "min": float(np.min(col_clean)) if len(col_clean) > 0 else 0.0,
            "q25": float(np.percentile(col_clean, 25)) if len(col_clean) > 0 else 0.0,
            "median": float(np.median(col_clean)) if len(col_clean) > 0 else 0.0,
            "q75": float(np.percentile(col_clean, 75)) if len(col_clean) > 0 else 0.0,
            "q95": float(np.percentile(col_clean, 95)) if len(col_clean) > 0 else 0.0,
            "q99": float(np.percentile(col_clean, 99)) if len(col_clean) > 0 else 0.0,
            "max": float(np.max(col_clean)) if len(col_clean) > 0 else 0.0,
            "missing_count": int(np.sum(np.isnan(col))),
            "missing_pct": float(np.mean(np.isnan(col))),
            "inf_count": int(np.sum(np.isinf(col))),
            "zero_pct": float(np.mean(col == 0.0)),
            "unique_count": int(len(np.unique(col_clean))),
        })
    return pd.DataFrame(stats_list)


def compute_distribution_shift(
    source_df: pd.DataFrame, target_df: pd.DataFrame, feature_names: List[str]
) -> pd.DataFrame:
    """Calculates pairwise Kolmogorov-Smirnov (KS) test and Wasserstein distance for cross-domain shift."""
    shift_list = []
    for f in feature_names:
        if f not in source_df.columns or f not in target_df.columns:
            continue
        
        s_vals = source_df[f].dropna().values
        t_vals = target_df[f].dropna().values

        if len(s_vals) == 0 or len(t_vals) == 0:
            continue

        ks_res = stats.ks_2samp(s_vals, t_vals)
        wass_dist = stats.wasserstein_distance(s_vals, t_vals)

        shift_list.append({
            "feature": f,
            "ks_statistic": float(ks_res.statistic),
            "ks_pvalue": float(ks_res.pvalue),
            "wasserstein_distance": float(wass_dist),
        })
    return pd.DataFrame(shift_list)


def compute_mutual_information(
    train_df: pd.DataFrame, feature_names: List[str], label_col: str = "label"
) -> pd.DataFrame:
    """Calculates mutual information between features and target binary label on training data."""
    if label_col not in train_df.columns:
        raise ValueError(f"Label column {label_col} missing from DataFrame")

    X = train_df[feature_names].fillna(0.0).values
    y = train_df[label_col].values

    mi_scores = mutual_info_classif(X, y, random_state=42)

    mi_list = []
    for f, mi in zip(feature_names, mi_scores):
        mi_list.append({
            "feature": f,
            "mutual_information": float(mi),
        })

    df_mi = pd.DataFrame(mi_list).sort_values("mutual_information", ascending=False).reset_index(drop=True)
    df_mi["rank"] = df_mi.index + 1
    return df_mi


def compute_correlation(df: pd.DataFrame, feature_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes Pearson and Spearman correlation matrices for feature pairs."""
    sub_df = df[feature_names].fillna(0.0)
    pearson = sub_df.corr(method="pearson")
    spearman = sub_df.corr(method="spearman")
    return pearson, spearman


def compute_vif(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Calculates Variance Inflation Factors (VIF) for multicollinearity analysis."""
    sub_df = df[feature_names].fillna(0.0)
    # Standardize to avoid scale effects in VIF calculation
    means = sub_df.mean()
    stds = sub_df.std().replace(0.0, 1.0)
    norm_df = (sub_df - means) / stds

    vif_list = []
    X = norm_df.values
    n_features = X.shape[1]

    for i in range(n_features):
        f = feature_names[i]
        y_i = X[:, i]
        X_other = np.delete(X, i, axis=1)

        try:
            # Linear regression R^2 calculation
            slope, intercept, r_val, p_val, std_err = stats.linregress(
                X_other.mean(axis=1), y_i
            )
            r_squared = r_val ** 2
            vif = 1.0 / (1.0 - r_squared + 1e-6)
        except Exception:
            vif = 1.0

        vif_list.append({
            "feature": f,
            "vif": float(vif),
        })

    return pd.DataFrame(vif_list)
