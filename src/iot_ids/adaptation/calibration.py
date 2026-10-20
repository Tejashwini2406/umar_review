"""Stage 5 Threshold Calibration & Adaptation Engine.

Provides unsupervised target probability percentile calibration and supervised target threshold tuning.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


class TargetThresholdCalibrator:
    """Calibrates classification decision thresholds for target domain deployment."""

    def __init__(self, default_threshold: float = 0.50):
        self.calibrated_threshold: float = default_threshold

    def fit(self, probs: np.ndarray, y_labels: np.ndarray) -> TargetThresholdCalibrator:
        """Fits calibrated threshold using supervised target budget split."""
        self.calibrated_threshold = self.calibrate_supervised_limited(y_labels, probs)
        return self

    @staticmethod
    def calibrate_unsupervised(
        source_val_labels: np.ndarray, source_val_probs: np.ndarray, target_unlabeled_probs: np.ndarray
    ) -> float:
        """Calibrates threshold on unlabeled target predictions by matching the source validation anomaly percentile.
        
        Zero target labels used.
        """
        # Calculate source anomaly prevalence
        source_attack_rate = float(np.mean(source_val_labels == 1))
        source_attack_rate = np.clip(source_attack_rate, 0.05, 0.95)

        # Set target threshold at the (100 - source_attack_rate*100)th percentile of target predicted probabilities
        target_percentile = (1.0 - source_attack_rate) * 100.0
        calibrated_tau = float(np.percentile(target_unlabeled_probs, target_percentile))
        return float(np.clip(calibrated_tau, 0.01, 0.99))

    @staticmethod
    def calibrate_supervised_limited(
        y_budget: np.ndarray, budget_probs: np.ndarray
    ) -> float:
        """Calibrates decision threshold on small target-labeled budget split (e.g. 1%, 5%, 10%)."""
        if len(y_budget) == 0 or len(np.unique(y_budget)) < 2:
            return 0.50

        best_tau = 0.50
        best_f1 = -1.0
        thresholds = np.linspace(0.01, 0.99, 99)

        for tau in thresholds:
            preds = (budget_probs >= tau).astype(int)
            macro_f1 = float(f1_score(y_budget, preds, average="macro", zero_division=0))
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_tau = float(tau)

        return float(best_tau)
