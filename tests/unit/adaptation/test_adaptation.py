"""Unit tests for Stage 5 Domain Adaptation engine and leakage safeguards."""

import pytest
import numpy as np
import pandas as pd

from iot_ids.adaptation.alignment import UnsupervisedFeatureAligner, UnsupervisedFeatureCorrector
from iot_ids.adaptation.calibration import TargetThresholdCalibrator
from iot_ids.adaptation.adaptation import run_adaptation_experiment, extract_target_label_budget


def test_unsupervised_feature_aligner_no_labels():
    f_names = ["flow_duration", "flow_bytes_per_sec", "payload_byte_ratio"]

    src_tr = pd.DataFrame({
        "flow_duration": [1.0, 2.0, 3.0, 4.0],
        "flow_bytes_per_sec": [10.0, 20.0, 30.0, 40.0],
        "payload_byte_ratio": [0.1, 0.2, 0.3, 0.4],
        "label": [0, 0, 1, 1],
    })

    # Unlabeled target train features
    tgt_tr = pd.DataFrame({
        "flow_duration": [10.0, 20.0, 30.0, 40.0],
        "flow_bytes_per_sec": [100.0, 200.0, 300.0, 400.0],
        "payload_byte_ratio": [0.5, 0.6, 0.7, 0.8],
    })

    aligner = UnsupervisedFeatureAligner(f_names)
    aligner.fit(src_tr, tgt_tr)
    assert aligner.is_fitted

    X_tgt_aligned = aligner.transform_target(tgt_tr)
    assert X_tgt_aligned.shape == (4, 3)


def test_target_threshold_calibrator_unsupervised():
    src_val_labels = np.array([0, 0, 0, 1, 1])
    src_val_probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    tgt_probs = np.array([0.05, 0.15, 0.25, 0.65, 0.85, 0.95])

    tau = TargetThresholdCalibrator.calibrate_unsupervised(src_val_labels, src_val_probs, tgt_probs)
    assert 0.10 <= tau <= 0.90


def test_extract_target_label_budget():
    tgt_tr = pd.DataFrame({
        "feat": np.arange(100),
        "label": np.random.choice([0, 1], size=100),
    })

    budget_df, rem_df = extract_target_label_budget(tgt_tr, budget_pct=0.05, seed=42)
    assert len(budget_df) == 5
    assert len(rem_df) == 95
    assert len(set(budget_df.index).intersection(set(rem_df.index))) == 0


def test_adaptation_experiment_zero_shot_and_alignment():
    f_names = ["flow_duration", "flow_bytes_per_sec", "payload_byte_ratio"]

    src_tr = pd.DataFrame({
        "flow_duration": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "flow_bytes_per_sec": [10.0, 20.0, 30.0, 400.0, 500.0, 600.0],
        "payload_byte_ratio": [0.1, 0.1, 0.1, 0.9, 0.9, 0.9],
        "label": [0, 0, 0, 1, 1, 1],
    })
    src_val = src_tr.copy()
    tgt_tr = src_tr.copy()

    tgt_te = pd.DataFrame({
        "flow_duration": [1.5, 2.5, 4.5, 5.5],
        "flow_bytes_per_sec": [15.0, 25.0, 450.0, 550.0],
        "payload_byte_ratio": [0.15, 0.15, 0.85, 0.85],
        "label": [0, 0, 1, 1],
    })

    # 1. Zero Shot
    res_zs = run_adaptation_experiment(
        source_dataset="DomainA",
        target_dataset="DomainB",
        feature_profile="instant_only",
        feature_names=f_names,
        model_name="LogisticRegression",
        adaptation_method="zero_shot",
        source_train_df=src_tr,
        source_val_df=src_val,
        target_train_df=tgt_tr,
        target_test_df=tgt_te,
    )
    assert res_zs["adaptation_method"] == "zero_shot"
    assert res_zs["macro_f1"] > 0.80

    # 2. Unsupervised Alignment
    res_align = run_adaptation_experiment(
        source_dataset="DomainA",
        target_dataset="DomainB",
        feature_profile="instant_only",
        feature_names=f_names,
        model_name="LogisticRegression",
        adaptation_method="unsupervised_alignment",
        source_train_df=src_tr,
        source_val_df=src_val,
        target_train_df=tgt_tr,
        target_test_df=tgt_te,
    )
    assert res_align["adaptation_method"] == "unsupervised_alignment"
    assert res_align["macro_f1"] > 0.80
