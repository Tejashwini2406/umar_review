"""Canonical Multi-Level Feature Representation Builder.

Coordinates Level A Instantaneous, Level B Causal Temporal, and Level C Causal Behavioral
extractors to generate the candidate 18-dimensional feature vector.

Preserves semantic feature grouping and enforces state resets across split/domain boundaries.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from iot_ids.data.flow_object import CanonicalFlow
from iot_ids.features.canonical.instant import InstantaneousFeatureExtractor, INSTANT_FEATURE_NAMES
from iot_ids.features.canonical.temporal import TemporalFeatureExtractor, TEMPORAL_FEATURE_NAMES
from iot_ids.features.canonical.behavioral import BehavioralFeatureExtractor, BEHAVIORAL_FEATURE_NAMES

CANONICAL_18_FEATURE_NAMES = (
    INSTANT_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES + BEHAVIORAL_FEATURE_NAMES
)


class CanonicalFlowBuilder:
    """Multi-level canonical feature extraction coordinator."""

    def __init__(self, alpha: float = 0.1, window_seconds: float = 30.0):
        self.instant_extractor = InstantaneousFeatureExtractor()
        self.temporal_extractor = TemporalFeatureExtractor(alpha=alpha)
        self.behavioral_extractor = BehavioralFeatureExtractor(window_seconds=window_seconds, alpha=alpha)

    def reset_state(self) -> None:
        """Clears state buffers across temporal/behavioral extractors.
        
        Call at train/val/test split boundaries and target-domain transfer initialization.
        """
        self.temporal_extractor.reset_state()
        self.behavioral_extractor.reset_state()

    def build_features(self, flow: CanonicalFlow) -> Dict[str, Dict[str, float]]:
        """Returns feature dictionary partitioned by semantic level."""
        instant = self.instant_extractor.extract(flow)
        temporal = self.temporal_extractor.extract_and_update(flow)
        behavioral = self.behavioral_extractor.extract_and_update(flow)

        return {
            "instant": instant,
            "temporal": temporal,
            "behavioral": behavioral,
        }

    def build_vector(self, flow: CanonicalFlow) -> np.ndarray:
        """Returns 18-dimensional continuous numpy array matching CANONICAL_18_FEATURE_NAMES."""
        instant_vec = self.instant_extractor.extract_vector(flow)
        temporal_vec = self.temporal_extractor.extract_vector(flow)
        behavioral_vec = self.behavioral_extractor.extract_vector(flow)

        full_vec = instant_vec + temporal_vec + behavioral_vec
        return np.array(full_vec, dtype=np.float64)
