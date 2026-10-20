from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np


@dataclass
class RiskDecision:
    supervised_score: float
    anomaly_score: float
    risk_state: str  # BENIGN, HIGH CONFIDENCE ATTACK, SUSPICIOUS
    reason: str


class RiskLayer:
    """Design B Risk & Decision Layer.
    Combines Supervised Detector (RF+MLP average probability) with
    an Independent Calibrated Autoencoder Anomaly Detector.
    """

    def __init__(self, tau_sup: float = 0.5, tau_ae: float = 0.8):
        self.tau_sup = tau_sup
        self.tau_ae = tau_ae
        self.val_mse_sorted: np.ndarray | None = None

    def fit_ae_calibration(self, val_benign_mse: np.ndarray):
        """Fit empirical CDF scaling for Autoencoder reconstruction error on validation benign samples."""
        self.val_mse_sorted = np.sort(val_benign_mse)

    def calibrate_ae_score(self, mse: np.ndarray) -> np.ndarray:
        """Converts raw reconstruction MSE into a calibrated anomaly score in [0, 1]."""
        if self.val_mse_sorted is None or len(self.val_mse_sorted) == 0:
            # Min-max fallback if no validation calibration data exists
            min_v, max_v = mse.min(), mse.max()
            if max_v == min_v:
                return np.zeros_like(mse)
            return (mse - min_v) / (max_v - min_v + 1e-9)

        # Empirical CDF lookup
        indices = np.searchsorted(self.val_mse_sorted, mse, side="right")
        return indices / len(self.val_mse_sorted)

    def predict(
        self,
        p_rf: np.ndarray,
        p_mlp: np.ndarray,
        ae_mse: np.ndarray,
    ) -> list[RiskDecision]:
        p_sup = 0.5 * p_rf + 0.5 * p_mlp
        s_ae = self.calibrate_ae_score(ae_mse)

        decisions = []
        for ps, sae in zip(p_sup, s_ae):
            if ps >= self.tau_sup:
                state = "HIGH CONFIDENCE ATTACK"
                reason = f"Supervised threat probability ({ps:.3f}) >= threshold ({self.tau_sup:.3f})"
            elif sae >= self.tau_ae:
                state = "SUSPICIOUS / ANOMALOUS"
                reason = f"Supervised threat probability ({ps:.3f}) low, but anomaly score ({sae:.3f}) >= threshold ({self.tau_ae:.3f}); potential zero-day"
            else:
                state = "BENIGN"
                reason = f"Normal supervised confidence ({ps:.3f}) and low anomaly distance ({sae:.3f})"

            decisions.append(
                RiskDecision(
                    supervised_score=float(ps),
                    anomaly_score=float(sae),
                    risk_state=state,
                    reason=reason,
                )
            )
        return decisions

    def save(self, path: Path):
        data = {
            "tau_sup": self.tau_sup,
            "tau_ae": self.tau_ae,
            "val_mse_sorted": self.val_mse_sorted.tolist() if self.val_mse_sorted is not None else [],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> RiskLayer:
        data = json.loads(path.read_text(encoding="utf-8"))
        obj = cls(tau_sup=data["tau_sup"], tau_ae=data["tau_ae"])
        if data["val_mse_sorted"]:
            obj.val_mse_sorted = np.array(data["val_mse_sorted"])
        return obj
