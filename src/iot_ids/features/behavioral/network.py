from __future__ import annotations

import math
import pandas as pd


def entropy(values: pd.Series) -> float:
    counts = values.dropna().astype(str).value_counts()
    total = counts.sum()
    if total == 0:
        return 0.0
    return float(-sum((count / total) * math.log2(count / total) for count in counts))


def add_historical_destination_diversity(
    frame: pd.DataFrame,
    source_col: str,
    destination_col: str,
    output_col: str,
    window: int = 50,
    prior_history: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if source_col not in frame or destination_col not in frame:
        result = frame.copy()
        result[output_col] = 0.0
        return result, prior_history or {}

    result = frame.copy()
    sources = result[source_col].astype(str).values
    destinations = result[destination_col].astype(str).values

    history: dict[str, list[str]] = {}
    if prior_history:
        for k, v in prior_history.items():
            history[k] = list(v)

    values: list[float] = []
    for src, dst in zip(sources, destinations):
        recent = history.setdefault(src, [])
        values.append(float(len(set(recent[-window:]))))
        recent.append(dst)
        if len(recent) > window * 2:
            history[src] = recent[-window:]

    result[output_col] = values
    trimmed: dict[str, list[str]] = {k: v[-window:] for k, v in history.items()}
    return result, trimmed
