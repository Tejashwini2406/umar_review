"""Stage 4 Experiment Execution Engine.

Executes within-domain and zero-shot cross-domain benchmark experiments across feature profiles and model baselines.
"""
from __future__ import annotations

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from iot_ids.experiments.preprocessing import FeaturePreprocessor
from iot_ids.experiments.evaluator import optimize_threshold_on_val, evaluate_predictions


def instantiate_model(model_name: str) -> Any:
    """Instantiates controlled baseline model with fixed random state."""
    if model_name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    elif model_name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unsupported model baseline: {model_name}")


def run_within_domain_experiment(
    dataset_name: str,
    feature_profile: str,
    feature_names: List[str],
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Runs within-domain train/val/test benchmarking."""
    # 1. Fit preprocessor ONLY on training split
    preprocessor = FeaturePreprocessor(feature_names)
    X_train = preprocessor.fit_transform(train_df)
    y_train = train_df["label"].values

    X_val = preprocessor.transform(val_df)
    y_val = val_df["label"].values

    X_test = preprocessor.transform(test_df)
    y_test = test_df["label"].values

    # 2. Fit model ONLY on training split
    model = instantiate_model(model_name)
    model.fit(X_train, y_train)

    # 3. Optimize decision threshold on Validation split
    val_probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_val)
    opt_tau = optimize_threshold_on_val(y_val, val_probs)

    # 4. Evaluate final predictions on Test split using frozen threshold
    test_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    metrics = evaluate_predictions(y_test, test_probs, threshold=opt_tau)

    res = {
        "experiment_type": "within_domain",
        "source_dataset": dataset_name,
        "target_dataset": dataset_name,
        "feature_profile": feature_profile,
        "model_name": model_name,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }
    res.update(metrics)
    return res


def run_cross_domain_experiment(
    source_dataset: str,
    target_dataset: str,
    feature_profile: str,
    feature_names: List[str],
    model_name: str,
    source_train_df: pd.DataFrame,
    source_val_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Runs genuine zero-shot cross-domain transfer experiment."""
    # 1. Fit preprocessor ONLY on Source Training split
    preprocessor = FeaturePreprocessor(feature_names)
    X_src_train = preprocessor.fit_transform(source_train_df)
    y_src_train = source_train_df["label"].values

    X_src_val = preprocessor.transform(source_val_df)
    y_src_val = source_val_df["label"].values

    # Target test features transformed using FROZEN source preprocessor (NO target fit)
    X_tgt_test = preprocessor.transform(target_test_df)
    y_tgt_test = target_test_df["label"].values

    # 2. Fit model ONLY on Source Training split
    model = instantiate_model(model_name)
    model.fit(X_src_train, y_src_train)

    # 3. Optimize threshold ONLY on Source Validation split
    src_val_probs = model.predict_proba(X_src_val)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_src_val)
    opt_tau = optimize_threshold_on_val(y_src_val, src_val_probs)

    # 4. Evaluate frozen model directly on Target Test split (NO target training)
    tgt_test_probs = model.predict_proba(X_tgt_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_tgt_test)
    metrics = evaluate_predictions(y_tgt_test, tgt_test_probs, threshold=opt_tau)

    res = {
        "experiment_type": "cross_domain",
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "feature_profile": feature_profile,
        "model_name": model_name,
        "n_train": len(source_train_df),
        "n_val": len(source_val_df),
        "n_test": len(target_test_df),
    }
    res.update(metrics)
    return res
