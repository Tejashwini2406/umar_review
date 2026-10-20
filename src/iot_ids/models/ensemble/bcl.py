"""Calibration and streaming decisions for a Behavioral Consistency Layer."""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ConsistencyDecision:
    """Per-flow behavioral consistency signals and calibrated alert decision."""

    suspicious: bool
    score: float
    threshold: float
    model_disagreement: float
    confidence_drift: float
    prediction_instability: float
    autoencoder_deviation: float


class BehavioralConsistencyLayer:
    """Flags unusual detector behavior using validation-calibrated signals.

    The threshold is calibrated on benign validation samples to meet a requested
    false-positive budget. Signals are scored by their deviation on either side
    of the benign calibration median. Validation labels are used only to select
    benign rows.
    """

    def __init__(
        self,
        target_false_positive_rate: float = 0.01,
        history_size: int = 10,
        use_stream_history: bool = True,
    ):
        if not 0.0 < target_false_positive_rate < 1.0:
            raise ValueError("target_false_positive_rate must be strictly between 0 and 1")
        if history_size < 2:
            raise ValueError("history_size must be at least 2")

        self.target_false_positive_rate = float(target_false_positive_rate)
        self.history_size = int(history_size)
        self.use_stream_history = bool(use_stream_history)
        self.threshold: float | None = None
        self._signal_calibration: dict[str, dict[str, float]] | None = None
        self._history: deque[float] = deque(maxlen=self.history_size)
        self.calibration_false_positive_rate: float | None = None

    def reset_state(self) -> None:
        """Clear stream history without discarding the fitted calibration."""
        self._history.clear()

    def save(self, path: str | Path) -> None:
        """Persist calibration parameters without storing validation samples."""
        if self.threshold is None or self._signal_calibration is None:
            raise RuntimeError("BehavioralConsistencyLayer must be fitted before saving")
        payload = {
            "version": 3,
            "target_false_positive_rate": self.target_false_positive_rate,
            "history_size": self.history_size,
            "use_stream_history": self.use_stream_history,
            "threshold": self.threshold,
            "signal_calibration": self._signal_calibration,
            "calibration_false_positive_rate": self.calibration_false_positive_rate,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BehavioralConsistencyLayer:
        """Load a fitted calibration artifact and validate its structure."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload.get("version")
        if version not in (1, 2, 3):
            raise ValueError(f"Unsupported BCL calibration version: {payload.get('version')}")
        layer = cls(
            target_false_positive_rate=payload["target_false_positive_rate"],
            history_size=payload["history_size"],
            use_stream_history=payload.get("use_stream_history", True),
        )
        threshold = float(payload["threshold"])
        required_signals = {
            "model_disagreement",
            "confidence_drift",
            "prediction_instability",
            "autoencoder_error",
        }
        if version == 3:
            calibration = {
                name: {
                    key: float(value)
                    for key, value in parameters.items()
                }
                for name, parameters in payload["signal_calibration"].items()
            }
        else:
            calibration = {
                name: {
                    "center": 0.0,
                    "lower_scale": float(scale),
                    "upper_scale": float(scale),
                }
                for name, scale in payload["signal_scales"].items()
            }
        if set(calibration) != required_signals or any(
            set(parameters) != {"center", "lower_scale", "upper_scale"}
            or any(not np.isfinite(value) for value in parameters.values())
            or parameters["lower_scale"] <= 0.0
            or parameters["upper_scale"] <= 0.0
            for parameters in calibration.values()
        ):
            raise ValueError("BCL calibration artifact contains invalid signal calibration")
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("BCL calibration artifact contains an invalid threshold")

        layer.threshold = threshold
        layer._signal_calibration = calibration
        false_positive_rate = payload["calibration_false_positive_rate"]
        if false_positive_rate is not None:
            false_positive_rate = float(false_positive_rate)
            if not np.isfinite(false_positive_rate) or not 0.0 <= false_positive_rate <= 1.0:
                raise ValueError("BCL calibration artifact contains an invalid false-positive rate")
        layer.calibration_false_positive_rate = false_positive_rate
        return layer

    def fit(
        self,
        rf_probabilities: np.ndarray,
        mlp_probabilities: np.ndarray,
        ae_errors: np.ndarray,
        labels: np.ndarray,
    ) -> BehavioralConsistencyLayer:
        """Calibrate the alert threshold from an ordered validation stream."""
        rf, mlp, ae, y = self._validate_arrays(
            rf_probabilities, mlp_probabilities, ae_errors, labels
        )
        benign = y == 0
        n_benign = int(benign.sum())
        minimum_benign = math.ceil(1.0 / self.target_false_positive_rate)
        if n_benign < minimum_benign:
            raise ValueError(
                f"At least {minimum_benign} benign validation samples are required "
                f"for a {self.target_false_positive_rate:.3%} false-positive target; "
                f"received {n_benign}"
            )

        self.reset_state()
        raw_signals = [
            self._raw_signals(rf_i, mlp_i, ae_i)
            for rf_i, mlp_i, ae_i in zip(rf, mlp, ae)
        ]
        signal_names = tuple(raw_signals[0])
        benign_signals = {
            name: np.asarray(
                [
                    signals[name]
                    for signals, is_benign in zip(raw_signals, benign)
                    if is_benign
                ],
                dtype=np.float64,
            )
            for name in signal_names
        }
        self._signal_calibration = {}
        for name, values in benign_signals.items():
            center = float(np.median(values))
            lower_quantile, upper_quantile = np.quantile(
                values,
                [self.target_false_positive_rate / 2.0, 1.0 - self.target_false_positive_rate / 2.0],
                method="linear",
            )
            minimum_scale = 1e-8 if name == "autoencoder_error" else 1e-3
            self._signal_calibration[name] = {
                "center": center,
                "lower_scale": max(center - float(lower_quantile), minimum_scale),
                "upper_scale": max(float(upper_quantile) - center, minimum_scale),
            }
        scores = np.asarray(
            [
                self._score_signals(signals)
                for signals in raw_signals
            ],
            dtype=np.float64,
        )

        benign_scores = scores[benign]
        self.threshold = float(
            np.quantile(
                benign_scores,
                1.0 - self.target_false_positive_rate,
                method="higher",
            )
        )
        self.calibration_false_positive_rate = float(np.mean(benign_scores > self.threshold))
        self.reset_state()
        return self

    def predict_one(
        self, rf_probability: float, mlp_probability: float, ae_error: float
    ) -> ConsistencyDecision:
        """Score one sample and update history for the next sample in the stream."""
        if self.threshold is None or self._signal_calibration is None:
            raise RuntimeError("BehavioralConsistencyLayer must be fitted before prediction")

        rf, mlp, ae, _ = self._validate_arrays(
            np.asarray([rf_probability]),
            np.asarray([mlp_probability]),
            np.asarray([ae_error]),
            np.asarray([0]),
        )
        signals = self._raw_signals(float(rf[0]), float(mlp[0]), float(ae[0]))
        score = self._score_signals(signals)
        return ConsistencyDecision(
            suspicious=score > self.threshold,
            score=float(score),
            threshold=self.threshold,
            model_disagreement=signals["model_disagreement"],
            confidence_drift=signals["confidence_drift"],
            prediction_instability=signals["prediction_instability"],
            autoencoder_deviation=signals["autoencoder_error"],
        )

    def _score_signals(self, signals: dict[str, float]) -> float:
        if self._signal_calibration is None:
            raise RuntimeError("BehavioralConsistencyLayer must be fitted before scoring")
        deviations = []
        for name, value in signals.items():
            calibration = self._signal_calibration[name]
            center = calibration["center"]
            scale_name = "upper_scale" if value >= center else "lower_scale"
            deviations.append(abs(value - center) / calibration[scale_name])
        return max(deviations)

    @staticmethod
    def _validate_arrays(
        rf_probabilities: np.ndarray,
        mlp_probabilities: np.ndarray,
        ae_errors: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        arrays = tuple(
            np.asarray(values, dtype=np.float64).reshape(-1)
            for values in (rf_probabilities, mlp_probabilities, ae_errors, labels)
        )
        rf, mlp, ae, y = arrays
        if not len(y) or any(len(values) != len(y) for values in arrays[:3]):
            raise ValueError("All inputs must contain the same non-zero number of samples")
        if any(not np.isfinite(values).all() for values in arrays):
            raise ValueError("All inputs must contain only finite values")
        if np.any((rf < 0.0) | (rf > 1.0)) or np.any((mlp < 0.0) | (mlp > 1.0)):
            raise ValueError("Model probabilities must be in [0, 1]")
        if np.any(ae < 0.0):
            raise ValueError("Autoencoder reconstruction errors must be non-negative")
        if np.any((y != 0.0) & (y != 1.0)):
            raise ValueError("labels must be binary values 0 or 1")
        return rf, mlp, ae, y.astype(np.int8)

    def _raw_signals(
        self, rf_probability: float, mlp_probability: float, ae_error: float
    ) -> dict[str, float]:
        p_sup = 0.7 * float(rf_probability) + 0.3 * float(mlp_probability)
        disagreement = abs(float(rf_probability) - float(mlp_probability))

        if self.use_stream_history and self._history:
            history = np.asarray(self._history, dtype=np.float64)
            confidence_drift = abs(p_sup - float(history.mean()))
            prediction_instability = float(history.std())
        else:
            confidence_drift = 0.0
            prediction_instability = 0.0

        if self.use_stream_history:
            self._history.append(p_sup)
        return {
            "model_disagreement": disagreement,
            "confidence_drift": confidence_drift,
            "prediction_instability": prediction_instability,
            "autoencoder_error": float(ae_error),
        }
