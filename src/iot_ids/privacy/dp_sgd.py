"""Per-example clipped DP-SGD with Poisson sampling and RDP accounting."""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.special import gammaln, logsumexp
from torch import nn, optim


@dataclass
class DPConfig:
    """Configuration for DP-SGD training."""

    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    target_delta: float = 1e-5
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.005

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if not math.isfinite(self.noise_multiplier) or self.noise_multiplier < 0:
            raise ValueError("noise_multiplier must be finite and non-negative")
        if not 0 < self.target_delta < 1:
            raise ValueError("target_delta must be between 0 and 1")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")


def _validate_accounting_inputs(
    steps: int,
    sample_rate: float,
    noise_multiplier: float,
    target_delta: float,
) -> None:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not math.isfinite(sample_rate) or not 0 <= sample_rate <= 1:
        raise ValueError("sample_rate must be between 0 and 1")
    if not math.isfinite(noise_multiplier) or noise_multiplier < 0:
        raise ValueError("noise_multiplier must be finite and non-negative")
    if not 0 < target_delta < 1:
        raise ValueError("target_delta must be between 0 and 1")


def compute_dp_epsilon(
    steps: int | None = None,
    sample_rate: float = 0.01,
    noise_multiplier: float = 1.0,
    target_delta: float = 1e-5,
    epochs: int | None = None,
) -> float:
    """Return an integer-order RDP upper bound for Poisson-sampled Gaussian DP-SGD."""
    if steps is None:
        if epochs is not None:
            if epochs < 0:
                raise ValueError("epochs must be non-negative")
            steps = math.ceil(epochs / sample_rate) if sample_rate > 0 else 0
        else:
            steps = 656

    _validate_accounting_inputs(steps, sample_rate, noise_multiplier, target_delta)
    if steps == 0 or sample_rate == 0:
        return 0.0
    if noise_multiplier == 0:
        return float("inf")

    orders = np.arange(2, 257, dtype=np.float64)
    if sample_rate == 1:
        rdp_per_step = orders / (2.0 * noise_multiplier**2)
    else:
        i = np.arange(257, dtype=np.float64)
        log_binomial = (
            gammaln(orders[:, None] + 1)
            - gammaln(i[None, :] + 1)
            - gammaln(orders[:, None] - i[None, :] + 1)
        )
        valid = i[None, :] <= orders[:, None]
        log_q_terms = (
            log_binomial
            + i[None, :] * math.log(sample_rate)
            + (orders[:, None] - i[None, :]) * math.log1p(-sample_rate)
            + (i[None, :] ** 2 - i[None, :]) / (2.0 * noise_multiplier**2)
        )
        log_q_terms[~valid] = -np.inf
        rdp_per_step = logsumexp(log_q_terms, axis=1) / (orders - 1.0)

    epsilon = steps * rdp_per_step + math.log(1.0 / target_delta) / (orders - 1.0)
    return float(np.min(epsilon))


def _secure_uniforms(size: int) -> np.ndarray:
    raw = np.frombuffer(secrets.token_bytes(size * 8), dtype=np.uint64)
    return ((raw >> 11).astype(np.float64) + 0.5) / (2**53)


def _poisson_sample_mask(dataset_size: int, sample_rate: float) -> np.ndarray:
    return _secure_uniforms(dataset_size) < sample_rate


def _secure_gaussian_like(reference: torch.Tensor) -> torch.Tensor:
    count = reference.numel()
    raw = np.frombuffer(secrets.token_bytes(count * 16), dtype=np.uint64).reshape(2, count)
    uniform_1 = ((raw[0] >> 11).astype(np.float64) + 0.5) / (2**53)
    uniform_2 = ((raw[1] >> 11).astype(np.float64) + 0.5) / (2**53)
    normal = np.sqrt(-2.0 * np.log(uniform_1)) * np.cos(2.0 * np.pi * uniform_2)
    noise = torch.from_numpy(normal.reshape(tuple(reference.shape))).to(
        device=reference.device,
        dtype=reference.dtype,
    )
    return noise


def train_step_dp_sgd(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    dp_config: DPConfig,
    expected_batch_size: float | None = None,
) -> None:
    """Apply one globally clipped per-example DP-SGD update."""
    if x_batch.size(0) != y_batch.size(0):
        raise ValueError("x_batch and y_batch must have the same number of samples")

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")

    denominator = float(x_batch.size(0) if expected_batch_size is None else expected_batch_size)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("expected_batch_size must be finite and positive")

    x_batch = torch.clamp(x_batch, min=-10.0, max=10.0)
    was_training = model.training
    model.eval()
    per_parameter: list[list[torch.Tensor]] = [[] for _ in parameters]
    try:
        for index in range(x_batch.size(0)):
            logits = model(x_batch[index : index + 1]).reshape(-1)
            losses = criterion(logits, y_batch[index : index + 1].float().reshape(-1))
            if losses.numel() != 1:
                raise ValueError("criterion must produce one scalar loss per sample")
            gradients = torch.autograd.grad(
                losses.reshape(()),
                parameters,
                allow_unused=True,
            )
            for stored, parameter, gradient in zip(per_parameter, parameters, gradients):
                stored.append(
                    torch.zeros_like(parameter) if gradient is None else gradient.detach()
                )
    finally:
        model.train(was_training)

    per_example = [
        torch.stack(parameter_gradients, dim=0) if parameter_gradients else None
        for parameter_gradients in per_parameter
    ]
    squared_norms = torch.zeros(
        x_batch.size(0),
        device=x_batch.device,
        dtype=x_batch.dtype,
    )
    for gradients in per_example:
        if gradients is not None:
            squared_norms += gradients.reshape(x_batch.size(0), -1).pow(2).sum(dim=1)
    norms = torch.sqrt(squared_norms)
    clip_factors = torch.clamp(
        dp_config.max_grad_norm / (norms + 1e-12),
        max=1.0,
    )

    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for parameter, gradients in zip(parameters, per_example):
            if gradients is None:
                clipped_sum = torch.zeros_like(parameter)
            else:
                view_shape = (x_batch.size(0),) + (1,) * (gradients.ndim - 1)
                clipped_sum = (gradients * clip_factors.reshape(view_shape)).sum(dim=0)
            if dp_config.noise_multiplier > 0:
                clipped_sum = clipped_sum + (
                    _secure_gaussian_like(clipped_sum)
                    * dp_config.noise_multiplier
                    * dp_config.max_grad_norm
                )
            parameter.grad = clipped_sum / denominator
    optimizer.step()


def train_robust_mlp_dp(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    dp_config: DPConfig,
    adv_eps: float = 0.10,
    adv_alpha: float = 0.025,
    adv_steps: int = 7,
) -> dict[str, Any]:
    """Train with per-record PGD preprocessing and Poisson-sampled DP-SGD."""
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if X_train.ndim != 2 or len(X_train) == 0 or len(X_train) != len(y_train):
        raise ValueError("training features and labels must be non-empty and aligned")
    if not np.isfinite(X_train).all() or not np.isfinite(y_train).all():
        raise ValueError("training data must contain only finite values")
    if not np.isin(y_train, (0.0, 1.0)).all():
        raise ValueError("training labels must be binary")
    if adv_eps < 0 or adv_alpha < 0 or adv_steps < 0:
        raise ValueError("adversarial budget, step size, and steps must be non-negative")

    optimizer = optim.Adam(model.parameters(), lr=dp_config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    dataset_size = len(X_train)
    sample_rate = min(dp_config.batch_size / dataset_size, 1.0)
    expected_batch_size = sample_rate * dataset_size
    steps_per_epoch = math.ceil(1.0 / sample_rate)
    total_steps = dp_config.epochs * steps_per_epoch
    feature_mask = torch.ones(X_train.shape[1], dtype=torch.float32)
    feature_mask[7:11] = 0.0

    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train)
    epoch_steps: list[dict[str, int]] = []
    model.eval()

    optimizer_updates = 0
    for epoch in range(dp_config.epochs):
        for _ in range(steps_per_epoch):
            selected = _poisson_sample_mask(dataset_size, sample_rate)
            x_batch = X_tensor[selected].clone()
            y_batch = y_tensor[selected].clone()
            x_adv = x_batch.clone()
            if x_batch.size(0) > 0:
                for _ in range(adv_steps):
                    per_record_gradients = []
                    for record_index in range(x_adv.size(0)):
                        x_record = x_adv[record_index : record_index + 1].detach()
                        x_record.requires_grad_(True)
                        logits = model(x_record).reshape(-1)
                        loss = criterion(
                            logits,
                            y_batch[record_index : record_index + 1],
                        )
                        per_record_gradients.append(
                            torch.autograd.grad(loss, x_record)[0]
                        )
                    input_gradient = torch.cat(per_record_gradients, dim=0)
                    with torch.no_grad():
                        perturbation = (
                            torch.clamp(
                                x_adv + adv_alpha * input_gradient.sign() * feature_mask - x_batch,
                                min=-adv_eps,
                                max=adv_eps,
                            )
                            * feature_mask
                        )
                        x_adv = torch.clamp(
                            x_batch + perturbation,
                            min=-10.0,
                            max=10.0,
                        )

            train_step_dp_sgd(
                model,
                optimizer,
                criterion,
                x_adv.detach(),
                y_batch,
                dp_config,
                expected_batch_size=expected_batch_size,
            )
            optimizer_updates += 1

        epoch_steps.append({"epoch": epoch + 1, "scheduled_steps": steps_per_epoch})

    epsilon = compute_dp_epsilon(
        steps=total_steps,
        sample_rate=sample_rate,
        noise_multiplier=dp_config.noise_multiplier,
        target_delta=dp_config.target_delta,
    )
    model.eval()
    return {
        "privacy_epsilon": epsilon,
        "privacy_delta": dp_config.target_delta,
        "max_grad_norm": dp_config.max_grad_norm,
        "noise_multiplier": dp_config.noise_multiplier,
        "accounted_steps": total_steps,
        "optimizer_updates": optimizer_updates,
        "sample_rate": sample_rate,
        "sampling_rate": sample_rate,
        "epochs": dp_config.epochs,
        "history": epoch_steps,
        "privacy_scope": "neural model training only",
        "noise_source": "operating-system cryptographic randomness",
        "accountant": "integer-order Poisson-sampled Gaussian RDP upper bound",
    }
