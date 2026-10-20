from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FeatureLevel = Literal["instant", "temporal", "behavioral"]
FeatureValidity = Literal["cross_dataset", "in_domain", "both"]


@dataclass(frozen=True, slots=True)
class CanonicalFeature:
    name: str
    dtype: str
    level: FeatureLevel
    semantic_definition: str
    units: str | None
    transformation: str
    availability: dict[str, list[str]]
    missing_behavior: str
    leakage_role: str
    valid_for: FeatureValidity


def load_feature_schema(path: str | Path) -> list[CanonicalFeature]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CanonicalFeature(**item) for item in payload["features"]]


def select_features(
    features: list[CanonicalFeature],
    levels: set[FeatureLevel] | None = None,
    valid_for: FeatureValidity | None = None,
) -> list[CanonicalFeature]:
    selected = features
    if levels is not None:
        selected = [feature for feature in selected if feature.level in levels]
    if valid_for is not None:
        selected = [
            feature
            for feature in selected
            if feature.valid_for == valid_for or feature.valid_for == "both"
        ]
    return selected

