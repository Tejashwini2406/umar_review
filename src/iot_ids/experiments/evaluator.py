"""Stage 4 Metrics Evaluator & Validation Threshold Optimizer.

Calculates threshold-independent metrics (ROC-AUC, PR-AUC), threshold-dependent metrics
(Accuracy, Macro F1, Attack F1, Benign F1, FPR, FNR, FP/1000), and source-validation threshold optimization.
"""
from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def optimize_threshold_on_val(y_val: np.ndarray, val_probs: np.ndarray) -> float:
    """Finds decision threshold tau* in [0.01, 0.99] maximizing Macro F1 on Source Validation data."""
    best_tau = 0.50
    best_f1 = -1.0

    thresholds = np.linspace(0.01, 0.99, 99)
    for tau in thresholds:
        preds = (val_probs >= tau).astype(int)
        macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_tau = float(tau)

    return best_tau


def evaluate_predictions(
    y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.50
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics given true labels and attack probability predictions."""
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)

    # Threshold-independent metrics
    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        roc_auc = 0.50

    try:
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = float(auc(rec_arr, prec_arr))
    except Exception:
        pr_auc = 0.0

    # Threshold-dependent metrics
    preds = (y_probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    acc = float(accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    attack_f1 = float(f1_score(y_true, preds, pos_label=1, zero_division=0))
    benign_f1 = float(f1_score(y_true, preds, pos_label=0, zero_division=0))

    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    fp_per_1000 = float(fpr * 1000.0)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": macro_f1,
        "attack_f1": attack_f1,
        "benign_f1": benign_f1,
        "fpr": fpr,
        "fnr": fnr,
        "fp_per_1000": fp_per_1000,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }
