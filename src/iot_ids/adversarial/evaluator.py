from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from iot_ids.models.ensemble.risk_layer import RiskLayer


def evaluate_adversarial_sample(
    rf_model: RandomForestClassifier,
    mlp_model: nn.Module,
    ae_model: nn.Module,
    risk_layer: RiskLayer,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Evaluates adversarial perturbation impact across RF, MLP, AE, and Design B Risk Layer."""
    mlp_model.eval()
    ae_model.eval()

    # 1. Clean Predictions
    p_rf_clean = rf_model.predict_proba(X_clean)[:, 1]
    with torch.no_grad():
        X_cl_t = torch.tensor(X_clean, dtype=torch.float32)
        p_mlp_clean = torch.sigmoid(mlp_model(X_cl_t)).squeeze().numpy()
        recon_clean = ae_model(X_cl_t).numpy()
        mse_clean = np.mean((X_clean - recon_clean) ** 2, axis=1)

    p_sup_clean = 0.5 * p_rf_clean + 0.5 * p_mlp_clean
    decisions_clean = risk_layer.predict(p_rf_clean, p_mlp_clean, mse_clean)

    # 2. Adversarial Predictions
    p_rf_adv = rf_model.predict_proba(X_adv)[:, 1]
    with torch.no_grad():
        X_adv_t = torch.tensor(X_adv, dtype=torch.float32)
        p_mlp_adv = torch.sigmoid(mlp_model(X_adv_t)).squeeze().numpy()
        recon_adv = ae_model(X_adv_t).numpy()
        mse_adv = np.mean((X_adv - recon_adv) ** 2, axis=1)

    p_sup_adv = 0.5 * p_rf_adv + 0.5 * p_mlp_adv
    decisions_adv = risk_layer.predict(p_rf_adv, p_mlp_adv, mse_adv)

    # 3. Filter for originally correctly classified ATTACK samples (Y=1)
    correct_attack_mask = (y_true == 1) & (p_sup_clean >= 0.5)
    n_attacks = int(correct_attack_mask.sum())

    if n_attacks == 0:
        return {
            "n_attacked": 0,
            "evaded_supervised": 0,
            "attack_success_rate": 0.0,
            "suspicious_rate": 0.0,
            "ae_catch_rate": 0.0,
        }

    # Evasion: P_sup drops below 0.5
    evaded_mask = correct_attack_mask & (p_sup_adv < 0.5)
    n_evaded = int(evaded_mask.sum())
    asr = float(n_evaded / n_attacks)

    # Risk state transitions among evaded samples
    states_adv = np.array([d.risk_state for d in decisions_adv])
    suspicious_evaded = int((evaded_mask & (states_adv == "SUSPICIOUS / ANOMALOUS")).sum())
    benign_evaded = int((evaded_mask & (states_adv == "BENIGN")).sum())
    
    # AE catch rate: Of the samples that fooled the supervised path, how many did AE flag as suspicious?
    ae_catch_rate = float(suspicious_evaded / n_evaded) if n_evaded > 0 else 0.0

    return {
        "n_attacked": n_attacks,
        "evaded_supervised": n_evaded,
        "attack_success_rate": asr,
        "benign_evaded_count": benign_evaded,
        "suspicious_evaded_count": suspicious_evaded,
        "ae_catch_rate": ae_catch_rate,
        "mean_p_sup_clean": float(p_sup_clean[correct_attack_mask].mean()),
        "mean_p_sup_adv": float(p_sup_adv[correct_attack_mask].mean()),
        "mean_mse_clean": float(mse_clean[correct_attack_mask].mean()),
        "mean_mse_adv": float(mse_adv[correct_attack_mask].mean()),
    }
