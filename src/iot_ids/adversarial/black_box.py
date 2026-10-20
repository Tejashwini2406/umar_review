"""Score-based attacks against frozen, non-differentiable predictors."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

ScoreFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BlackBoxAttackResult:
    adversarial: np.ndarray
    queries_per_sample: int
    total_queries: int


def spsa_attack(
    score_function: ScoreFunction,
    inputs: np.ndarray,
    *,
    epsilon: float,
    alpha: float,
    steps: int,
    probe_scale: float,
    feature_mask: np.ndarray,
    x_min: np.ndarray,
    x_max: np.ndarray,
    seed: int = 42,
) -> BlackBoxAttackResult:
    """Minimize a score with SPSA using only queries to the supplied predictor.

    The L-infinity budget and box bounds are applied in the model-input space.
    Features disabled by ``feature_mask`` remain unchanged. Features outside the
    supplied box at the clean point are held fixed rather than silently clipped.
    """
    original = np.asarray(inputs, dtype=np.float64)
    if original.ndim != 2 or original.shape[0] == 0 or original.shape[1] == 0:
        raise ValueError("inputs must be a non-empty two-dimensional array")
    if not np.isfinite(original).all():
        raise ValueError("inputs must contain only finite values")
    if epsilon <= 0.0 or alpha <= 0.0 or steps < 1 or probe_scale <= 0.0:
        raise ValueError("epsilon, alpha, and probe_scale must be positive; steps must be at least 1")

    mask = np.asarray(feature_mask, dtype=bool)
    if mask.shape != (original.shape[1],):
        raise ValueError("feature_mask must have one entry per input feature")
    if not mask.any():
        raise ValueError("feature_mask must enable at least one feature")

    lower = np.broadcast_to(np.asarray(x_min, dtype=np.float64), original.shape)
    upper = np.broadcast_to(np.asarray(x_max, dtype=np.float64), original.shape)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("x_min and x_max must be finite")
    if np.any(lower > upper):
        raise ValueError("x_min must not exceed x_max")

    active = np.broadcast_to(mask, original.shape)
    feasible_lower = np.maximum(original - epsilon, lower)
    feasible_upper = np.minimum(original + epsilon, upper)
    # Empirical training bounds may not contain every held-out value. Do not
    # alter such a value just to force it inside those bounds.
    infeasible = feasible_lower > feasible_upper
    feasible_lower[infeasible] = original[infeasible]
    feasible_upper[infeasible] = original[infeasible]
    feasible_lower[~active] = original[~active]
    feasible_upper[~active] = original[~active]

    rng = np.random.default_rng(seed)
    adversarial = original.copy()
    for _ in range(steps):
        direction = rng.choice(np.asarray([-1.0, 1.0]), size=original.shape)
        direction *= active
        plus = np.clip(adversarial + probe_scale * direction, feasible_lower, feasible_upper)
        minus = np.clip(adversarial - probe_scale * direction, feasible_lower, feasible_upper)
        score_plus = _query_scores(score_function, plus)
        score_minus = _query_scores(score_function, minus)
        gradient_estimate = (
            (score_plus - score_minus)[:, None] / (2.0 * probe_scale)
        ) * direction
        adversarial = np.clip(
            adversarial - alpha * np.sign(gradient_estimate),
            feasible_lower,
            feasible_upper,
        )

    queries_per_sample = 2 * steps
    return BlackBoxAttackResult(
        adversarial=adversarial,
        queries_per_sample=queries_per_sample,
        total_queries=queries_per_sample * len(original),
    )


def _query_scores(score_function: ScoreFunction, inputs: np.ndarray) -> np.ndarray:
    scores = np.asarray(score_function(inputs), dtype=np.float64).reshape(-1)
    if scores.shape != (len(inputs),):
        raise ValueError(
            f"score_function must return one score per input; got {scores.shape} "
            f"for {len(inputs)} inputs"
        )
    if not np.isfinite(scores).all():
        raise ValueError("score_function returned non-finite scores")
    return scores
