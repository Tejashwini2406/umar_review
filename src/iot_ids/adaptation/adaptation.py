"""Stage 5 Adaptation Pipeline Coordinator.

Executes the 7 distinct adaptation methods (zero_shot, unsupervised_alignment, unsupervised_correction,
threshold_calibration, limited_label_1pct, limited_label_5pct, limited_label_10pct) while enforcing strict
anti-leakage invariants. Target test labels are NEVER used for adaptation.
"""
from __future__ import annotations

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

from iot_ids.experiments.preprocessing import FeaturePreprocessor
from iot_ids.experiments.evaluator import optimize_threshold_on_val, evaluate_predictions
from iot_ids.experiments.runner import instantiate_model
from iot_ids.adaptation.alignment import UnsupervisedFeatureAligner, UnsupervisedFeatureCorrector
from iot_ids.adaptation.calibration import TargetThresholdCalibrator

ADAPTATION_METHODS = [
    "zero_shot",
    "unsupervised_alignment",
    "unsupervised_correction",
    "threshold_calibration",
    "limited_label_1pct",
    "limited_label_5pct",
    "limited_label_10pct",
]


def extract_target_label_budget(
    target_train_df: pd.DataFrame, budget_pct: float, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits target training split into budget adaptation subset and remaining unused subset."""
    if budget_pct <= 0.0:
        return pd.DataFrame(), target_train_df

    n_samples = max(2, int(len(target_train_df) * budget_pct))
    np.random.seed(seed)
    indices = np.random.choice(len(target_train_df), size=n_samples, replace=False)
    
    budget_df = target_train_df.iloc[indices].copy()
    remaining_df = target_train_df.drop(target_train_df.index[indices]).copy()
    return budget_df, remaining_df


def run_adaptation_experiment(
    source_dataset: str,
    target_dataset: str,
    feature_profile: str,
    feature_names: List[str],
    model_name: str,
    adaptation_method: str,
    source_train_df: pd.DataFrame,
    source_val_df: pd.DataFrame,
    target_train_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
    seed: int = 42,
) -> Dict[str, Any]:
    """Executes a single Stage 5 adaptation experiment with strict leakage safeguards."""
    
    # -------------------------------------------------------------------------
    # METHOD 0: ZERO-SHOT CONTROL
    # -------------------------------------------------------------------------
    if adaptation_method == "zero_shot":
        preproc = FeaturePreprocessor(feature_names)
        X_src_tr = preproc.fit_transform(source_train_df)
        y_src_tr = source_train_df["label"].values

        X_src_v = preproc.transform(source_val_df)
        y_src_v = source_val_df["label"].values

        X_tgt_te = preproc.transform(target_test_df)
        y_tgt_te = target_test_df["label"].values

        clf = instantiate_model(model_name)
        clf.fit(X_src_tr, y_src_tr)

        src_val_probs = clf.predict_proba(X_src_v)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_src_v)
        tau_star = optimize_threshold_on_val(y_src_v, src_val_probs)

        tgt_test_probs = clf.predict_proba(X_tgt_te)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_tgt_te)
        metrics = evaluate_predictions(y_tgt_te, tgt_test_probs, threshold=tau_star)
        n_budget = 0

    # -------------------------------------------------------------------------
    # METHOD 1: UNSUPERVISED TARGET FEATURE ALIGNMENT
    # -------------------------------------------------------------------------
    elif adaptation_method == "unsupervised_alignment":
        aligner = UnsupervisedFeatureAligner(feature_names)
        aligner.fit(source_train_df, target_train_df)  # UNLABELED target train features

        X_src_tr = aligner.transform_source(source_train_df)
        y_src_tr = source_train_df["label"].values

        X_src_v = aligner.transform_source(source_val_df)
        y_src_v = source_val_df["label"].values

        X_tgt_te = aligner.transform_target(target_test_df)
        y_tgt_te = target_test_df["label"].values

        clf = instantiate_model(model_name)
        clf.fit(X_src_tr, y_src_tr)

        src_val_probs = clf.predict_proba(X_src_v)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_src_v)
        tau_star = optimize_threshold_on_val(y_src_v, src_val_probs)

        tgt_test_probs = clf.predict_proba(X_tgt_te)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_tgt_te)
        metrics = evaluate_predictions(y_tgt_te, tgt_test_probs, threshold=tau_star)
        n_budget = 0

    # -------------------------------------------------------------------------
    # METHOD 2: UNSUPERVISED FEATURE-WISE CORRECTION
    # -------------------------------------------------------------------------
    elif adaptation_method == "unsupervised_correction":
        corrector = UnsupervisedFeatureCorrector(feature_names)
        corrector.fit(source_train_df, target_train_df)

        X_src_tr = corrector.transform_source(source_train_df)
        y_src_tr = source_train_df["label"].values

        X_src_v = corrector.transform_source(source_val_df)
        y_src_v = source_val_df["label"].values

        X_tgt_te = corrector.transform_target(target_test_df)
        y_tgt_te = target_test_df["label"].values

        clf = instantiate_model(model_name)
        clf.fit(X_src_tr, y_src_tr)

        src_val_probs = clf.predict_proba(X_src_v)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_src_v)
        tau_star = optimize_threshold_on_val(y_src_v, src_val_probs)

        tgt_test_probs = clf.predict_proba(X_tgt_te)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_tgt_te)
        metrics = evaluate_predictions(y_tgt_te, tgt_test_probs, threshold=tau_star)
        n_budget = 0

    # -------------------------------------------------------------------------
    # METHOD 3: UNSUPERVISED THRESHOLD CALIBRATION
    # -------------------------------------------------------------------------
    elif adaptation_method == "threshold_calibration":
        preproc = FeaturePreprocessor(feature_names)
        X_src_tr = preproc.fit_transform(source_train_df)
        y_src_tr = source_train_df["label"].values

        X_src_v = preproc.transform(source_val_df)
        y_src_v = source_val_df["label"].values

        X_tgt_tr = preproc.transform(target_train_df)  # Unlabeled target train
        X_tgt_te = preproc.transform(target_test_df)
        y_tgt_te = target_test_df["label"].values

        clf = instantiate_model(model_name)
        clf.fit(X_src_tr, y_src_tr)

        tgt_tr_probs = clf.predict_proba(X_tgt_tr)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_tgt_tr)
        tau_calibrated = TargetThresholdCalibrator.calibrate_unsupervised(
            y_src_v, clf.predict_proba(X_src_v)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_src_v), tgt_tr_probs
        )

        tgt_test_probs = clf.predict_proba(X_tgt_te)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_tgt_te)
        metrics = evaluate_predictions(y_tgt_te, tgt_test_probs, threshold=tau_calibrated)
        n_budget = 0

    # -------------------------------------------------------------------------
    # METHODS 4-6: LIMITED-LABEL TARGET ADAPTATION (1%, 5%, 10%)
    # -------------------------------------------------------------------------
    elif adaptation_method in ("limited_label_1pct", "limited_label_5pct", "limited_label_10pct"):
        pct_map = {"limited_label_1pct": 0.01, "limited_label_5pct": 0.05, "limited_label_10pct": 0.10}
        budget_pct = pct_map[adaptation_method]

        budget_df, _ = extract_target_label_budget(target_train_df, budget_pct=budget_pct, seed=seed)
        n_budget = len(budget_df)

        preproc = FeaturePreprocessor(feature_names)
        X_src_tr = preproc.fit_transform(source_train_df)
        y_src_tr = source_train_df["label"].values

        X_budget = preproc.transform(budget_df)
        y_budget = budget_df["label"].values

        X_tgt_te = preproc.transform(target_test_df)
        y_tgt_te = target_test_df["label"].values

        # Combine source train + small target budget to fine-tune model
        X_combined = np.vstack([X_src_tr, X_budget])
        y_combined = np.concatenate([y_src_tr, y_budget])

        clf = instantiate_model(model_name)
        clf.fit(X_combined, y_combined)

        # Calibrate threshold on target budget
        budget_probs = clf.predict_proba(X_budget)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_budget)
        tau_budget = TargetThresholdCalibrator.calibrate_supervised_limited(y_budget, budget_probs)

        tgt_test_probs = clf.predict_proba(X_tgt_te)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_tgt_te)
        metrics = evaluate_predictions(y_tgt_te, tgt_test_probs, threshold=tau_budget)

    else:
        raise ValueError(f"Unsupported adaptation method: {adaptation_method}")

    res = {
        "experiment_id": f"{source_dataset}_to_{target_dataset}_{feature_profile}_{model_name}_{adaptation_method}",
        "source_domain": source_dataset,
        "target_domain": target_dataset,
        "feature_profile": feature_profile,
        "model_name": model_name,
        "adaptation_method": adaptation_method,
        "target_label_budget_n": n_budget,
        "n_src_train": len(source_train_df),
        "n_tgt_test": len(target_test_df),
    }
    res.update(metrics)
    return res
