"""Deterministic, bounded sampling and stratified split helpers."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeVar

import numpy as np

from iot_ids.data.flow_object import CanonicalFlow

T = TypeVar("T")


def reservoir_sample(items: Iterable[T], sample_limit: int, seed: int) -> list[T]:
    """Uniformly sample a bounded number of items without favoring source order."""
    if sample_limit < 1:
        raise ValueError("sample_limit must be positive")

    rng = np.random.default_rng(seed)
    reservoir: list[tuple[int, T]] = []
    for index, item in enumerate(items):
        if index < sample_limit:
            reservoir.append((index, item))
        else:
            replacement = int(rng.integers(0, index + 1))
            if replacement < sample_limit:
                reservoir[replacement] = (index, item)
    return [item for _, item in sorted(reservoir)]


def stratified_random_split(
    flows: Sequence[CanonicalFlow], seed: int
) -> tuple[list[CanonicalFlow], list[CanonicalFlow], list[CanonicalFlow]]:
    """Split sampled flows 60/20/20 within each class, retaining source order."""
    rng = np.random.default_rng(seed)
    assignments: list[list[int]] = [[], [], []]
    for label in (0, 1):
        class_indices = [index for index, flow in enumerate(flows) if flow.label == label]
        if len(class_indices) < 3:
            raise ValueError(
                f"Need at least 3 sampled flows for label {label} to populate "
                "train, validation, and test independently"
            )
        permutation = rng.permutation(len(class_indices))
        n_train = max(1, int(len(class_indices) * 0.6))
        n_val = max(1, int(len(class_indices) * 0.2))
        if n_train + n_val >= len(class_indices):
            n_train = len(class_indices) - 2
            n_val = 1
        boundaries = (n_train, n_train + n_val)
        for split_indices, positions in zip(
            assignments,
            (
                permutation[: boundaries[0]],
                permutation[boundaries[0] : boundaries[1]],
                permutation[boundaries[1] :],
            ),
        ):
            split_indices.extend(class_indices[int(position)] for position in positions)

    splits: list[list[CanonicalFlow]] = []
    for split_indices in assignments:
        selected_indices = set(split_indices)
        splits.append(
            [
                flow
                for index, flow in enumerate(flows)
                if index in selected_indices
            ]
        )
    return splits[0], splits[1], splits[2]
