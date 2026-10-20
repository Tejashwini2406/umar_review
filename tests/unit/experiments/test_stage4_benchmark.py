"""Unit tests for Stage 4 benchmarking runner, data integrity audit, and leakage safeguards."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import importlib.util

spec = importlib.util.spec_from_file_location("benchmark_models", "scripts/04_benchmark_models.py")
benchmark_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_models)

perform_preflight_data_integrity_audit = benchmark_models.perform_preflight_data_integrity_audit
FEATURE_PROFILES = benchmark_models.FEATURE_PROFILES
from iot_ids.experiments.preprocessing import FeaturePreprocessor
from iot_ids.experiments.evaluator import optimize_threshold_on_val, evaluate_predictions


def test_stage4_preflight_integrity_audit():
    data_dir = Path("data/processed/stage3")
    perform_preflight_data_integrity_audit(data_dir)


def test_feature_profiles_consistency():
    assert "baseline_common" in FEATURE_PROFILES
    assert "instant_only" in FEATURE_PROFILES
    assert "instant_temporal" in FEATURE_PROFILES
    assert "instant_behavioral" in FEATURE_PROFILES
    assert "full_multilevel" in FEATURE_PROFILES

    # Instantaneous features subset check
    assert set(FEATURE_PROFILES["baseline_common"]).issubset(set(FEATURE_PROFILES["instant_only"]))
    assert set(FEATURE_PROFILES["instant_only"]).issubset(set(FEATURE_PROFILES["instant_temporal"]))
    assert set(FEATURE_PROFILES["instant_only"]).issubset(set(FEATURE_PROFILES["instant_behavioral"]))


def test_source_only_preprocessor_fitting_invariant():
    f_names = ["flow_duration", "flow_bytes_per_sec", "payload_byte_ratio"]
    preproc = FeaturePreprocessor(f_names)

    df_src_tr = pd.DataFrame({
        "flow_duration": [1.0, 2.0, 3.0, 4.0],
        "flow_bytes_per_sec": [10.0, 20.0, 30.0, 40.0],
        "payload_byte_ratio": [0.1, 0.2, 0.3, 0.4],
    })

    df_tgt_te = pd.DataFrame({
        "flow_duration": [10.0, 20.0],
        "flow_bytes_per_sec": [100.0, 200.0],
        "payload_byte_ratio": [0.5, 0.6],
    })

    X_tr = preproc.fit_transform(df_src_tr)
    assert preproc.is_fitted

    # Transform target test using frozen preprocessor
    X_tgt = preproc.transform(df_tgt_te)
    assert X_tgt.shape == (2, 3)
