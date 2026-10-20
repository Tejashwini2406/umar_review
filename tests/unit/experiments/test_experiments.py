"""Unit tests for Stage 4 experimental framework and leakage safeguards."""

import pytest
import numpy as np
import pandas as pd

from iot_ids.experiments.preprocessing import FeaturePreprocessor
from iot_ids.experiments.evaluator import optimize_threshold_on_val, evaluate_predictions
from iot_ids.experiments.runner import run_within_domain_experiment, run_cross_domain_experiment


def test_preprocessor_train_only_fitting_safeguard():
    feature_names = ["flow_duration", "flow_bytes_per_sec", "payload_byte_ratio"]
    preprocessor = FeaturePreprocessor(feature_names)

    df_dummy = pd.DataFrame({
        "flow_duration": [1.0, 2.0, 3.0],
        "flow_bytes_per_sec": [100.0, 200.0, 300.0],
        "payload_byte_ratio": [0.1, 0.2, 0.3],
    })

    # transform without fit_transform must raise RuntimeError
    with pytest.raises(RuntimeError, match="must be fitted"):
        preprocessor.transform(df_dummy)

    # fit_transform on train
    X_train = preprocessor.fit_transform(df_dummy)
    assert X_train.shape == (3, 3)
    assert preprocessor.is_fitted

    # now transform on val succeeds
    X_val = preprocessor.transform(df_dummy)
    assert X_val.shape == (3, 3)


def test_threshold_optimization_and_evaluation():
    y_val = np.array([0, 0, 0, 1, 1, 1])
    val_probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    best_tau = optimize_threshold_on_val(y_val, val_probs)
    assert 0.3 <= best_tau <= 0.7

    metrics = evaluate_predictions(y_val, val_probs, threshold=best_tau)
    assert metrics["roc_auc"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["tn"] == 3
    assert metrics["tp"] == 3


def test_cross_domain_runner_zero_shot():
    feature_names = ["flow_duration", "flow_bytes_per_sec", "payload_byte_ratio"]

    # Source dataset
    src_train = pd.DataFrame({
        "flow_duration": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "flow_bytes_per_sec": [10.0, 20.0, 30.0, 400.0, 500.0, 600.0],
        "payload_byte_ratio": [0.1, 0.1, 0.1, 0.9, 0.9, 0.9],
        "label": [0, 0, 0, 1, 1, 1],
    })
    src_val = src_train.copy()

    # Target dataset
    tgt_test = pd.DataFrame({
        "flow_duration": [1.5, 2.5, 4.5, 5.5],
        "flow_bytes_per_sec": [15.0, 25.0, 450.0, 550.0],
        "payload_byte_ratio": [0.15, 0.15, 0.85, 0.85],
        "label": [0, 0, 1, 1],
    })

    res = run_cross_domain_experiment(
        source_dataset="DomainA",
        target_dataset="DomainB",
        feature_profile="instant_only",
        feature_names=feature_names,
        model_name="LogisticRegression",
        source_train_df=src_train,
        source_val_df=src_val,
        target_test_df=tgt_test,
    )

    assert res["experiment_type"] == "cross_domain"
    assert res["source_dataset"] == "DomainA"
    assert res["target_dataset"] == "DomainB"
    assert res["macro_f1"] > 0.80
    assert res["fpr"] == 0.0
