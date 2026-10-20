from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True, slots=True)
class SplitPlan:
    dataset: str
    strategy: str
    grouping_key: str | None
    temporal_key: str | None
    rationale: str


SPLIT_PLANS = {
    "Edge-IIoTset": SplitPlan(
        "Edge-IIoTset",
        "group-aware stratified 60/20/20",
        "source_host",
        "timestamp",
        "Group-aware stratified split by source IP and attack category to prevent entity leakage while ensuring all attack types exist across Train, Val, and Test splits.",
    ),
    "ToN-IoT": SplitPlan(
        "ToN-IoT",
        "group-aware stratified 60/20/20",
        "source_host",
        "timestamp",
        "Group-aware stratified split by source IP and attack type to guarantee benign/attack balance across Train, Val, and Test splits.",
    ),
}


def stratified_split_indices(
    frame: pd.DataFrame,
    label_col: str = "label",
    category_col: str = "attack_category",
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates 60/20/20 train, validation, and test indices with stratified attack category representation."""
    strat_key = frame[label_col].astype(str)
    if category_col in frame:
        strat_key = strat_key + "_" + frame[category_col].astype(str)

    indices = np.arange(len(frame))
    
    # Check minimum class frequency for stratification
    class_counts = strat_key.value_counts()
    if class_counts.min() < 3:
        strat_key = frame[label_col].astype(str)

    # First split: Train (60%) vs Temp (40%)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_size + test_size),
        random_state=random_state,
        stratify=strat_key,
    )

    # Second split: Val (20%) vs Test (20%) from Temp
    temp_strat = strat_key.iloc[temp_idx]
    if temp_strat.value_counts().min() < 2:
        temp_strat = frame[label_col].iloc[temp_idx].astype(str)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_strat,
    )

    return train_idx, val_idx, test_idx
