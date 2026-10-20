"""Unified Multi-Level Operational IDS Pipeline.

Composes tested pipeline components:
FlowAggregator -> CanonicalFlowBuilder (18 Multi-Level Features) -> FeaturePreprocessor ->
Optional UnsupervisedFeatureAligner -> Trained Model -> TargetThresholdCalibrator -> Risk/Alert Output.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from iot_ids.data.flow_object import CanonicalFlow
from iot_ids.data.flow_aggregator import FlowAggregator
from iot_ids.features.canonical.flow_builder import CanonicalFlowBuilder, CANONICAL_18_FEATURE_NAMES
from iot_ids.experiments.preprocessing import FeaturePreprocessor
from iot_ids.adaptation.alignment import UnsupervisedFeatureAligner
from iot_ids.adaptation.calibration import TargetThresholdCalibrator


@dataclass(frozen=True)
class IDSAlertOutput:
    """Production Alert Output Schema Contract for Multi-Level IoT IDS."""

    flow_id: str
    src_ip: str
    dst_ip: str
    timestamp: float
    prediction_prob: float
    decision_threshold: float
    is_anomaly: bool
    risk_score: float
    risk_level: str
    feature_profile: str
    raw_features: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Converts alert output to standard JSON/serializable dictionary."""
        return asdict(self)


class IDSSystemPipeline:
    """Production-grade operational pipeline for multi-level IoT intrusion detection."""

    def __init__(
        self,
        model: Any,
        preprocessor: FeaturePreprocessor,
        feature_profile: str = "full_multilevel",
        feature_names: Optional[List[str]] = None,
        aligner: Optional[UnsupervisedFeatureAligner] = None,
        calibrator: Optional[TargetThresholdCalibrator] = None,
        decision_threshold: float = 0.5,
        alpha: float = 0.1,
        window_seconds: float = 30.0,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_profile = feature_profile
        self.feature_names = feature_names or CANONICAL_18_FEATURE_NAMES
        self.aligner = aligner
        self.calibrator = calibrator
        self.decision_threshold = decision_threshold

        self.flow_aggregator = FlowAggregator(inactivity_timeout=15.0)
        self.flow_builder = CanonicalFlowBuilder(alpha=alpha, window_seconds=window_seconds)

    @property
    def effective_decision_threshold(self) -> float:
        """Returns the calibrated threshold if calibrator is present, otherwise configured decision_threshold."""
        if self.calibrator is not None and getattr(self.calibrator, "calibrated_threshold", None) is not None:
            return float(self.calibrator.calibrated_threshold)
        return float(self.decision_threshold)

    def health_check(self) -> Dict[str, Any]:
        """Performs production health inspection of the pipeline and loaded artifacts."""
        has_model = self.model is not None
        has_preproc = self.preprocessor is not None
        has_aligner = self.aligner is not None and getattr(self.aligner, "is_fitted", False)
        has_calibrator = self.calibrator is not None and getattr(self.calibrator, "calibrated_threshold", None) is not None

        status = "HEALTHY" if (has_model and has_preproc) else "UNHEALTHY"

        return {
            "status": status,
            "model_loaded": has_model,
            "preprocessor_loaded": has_preproc,
            "aligner_fitted": has_aligner,
            "calibrator_fitted": has_calibrator,
            "feature_profile": self.feature_profile,
            "num_feature_names": len(self.feature_names),
            "active_flows_in_memory": len(self.flow_aggregator.active_flows),
            "decision_threshold": self.effective_decision_threshold,
        }

    def reset_state(self) -> None:
        """Clears state buffers in the aggregator and multi-level feature builder."""
        self.flow_aggregator.clear()
        self.flow_builder.reset_state()

    def process_flow(self, flow: CanonicalFlow) -> Dict[str, Any]:
        """Processes a single CanonicalFlow object through the full multi-level inference pipeline."""
        # 1. Feature Extraction (18 Semantic Features)
        feats_dict = self.flow_builder.build_features(flow)
        
        flat_feats = {}
        for level_dict in feats_dict.values():
            flat_feats.update(level_dict)
            
        df_single = pd.DataFrame([flat_feats])
        
        present_cols = [c for c in self.feature_names if c in df_single.columns]
        if len(present_cols) == 0:
            present_cols = [c for c in df_single.columns if c in CANONICAL_18_FEATURE_NAMES]
            
        X_df = df_single[present_cols]
        
        # 2. Preprocessing & Optional Alignment
        if self.aligner is not None and getattr(self.aligner, "is_fitted", False):
            X_proc = self.aligner.transform(X_df)
        else:
            X_proc = self.preprocessor.transform(X_df)
            
        # 3. Model Prediction
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_proc)[:, 1]
            prob = float(probs[0])
        else:
            decision = self.model.predict(X_proc)
            prob = float(decision[0])
            
        # 4. Threshold Calibration / Risk Score
        thresh = self.decision_threshold
        if self.calibrator is not None and getattr(self.calibrator, "calibrated_threshold", None) is not None:
            thresh = self.calibrator.calibrated_threshold

        is_anomaly = prob >= thresh
        risk_score = float(np.clip(prob / max(thresh * 2.0, 1e-4), 0.0, 1.0))
        risk_level = "CRITICAL" if prob >= 0.85 else ("HIGH" if prob >= thresh else ("MEDIUM" if prob >= thresh * 0.5 else "LOW"))

        alert_obj = IDSAlertOutput(
            flow_id=f"{flow.src_host}:{flow.src_port}->{flow.dst_host}:{flow.dst_port}",
            src_ip=flow.src_host,
            dst_ip=flow.dst_host,
            timestamp=flow.timestamp_start,
            prediction_prob=prob,
            decision_threshold=thresh,
            is_anomaly=bool(is_anomaly),
            risk_score=risk_score,
            risk_level=risk_level,
            feature_profile=self.feature_profile,
            raw_features=flat_feats,
        )
        return alert_obj.to_dict()

    def process_dataframe(self, df_flows: pd.DataFrame) -> List[Dict[str, Any]]:
        """Batch processes a DataFrame of flow parameters or pre-extracted materialized features."""
        results = []

        # Fast path: DataFrame already contains extracted feature columns
        feat_match = [c for c in CANONICAL_18_FEATURE_NAMES if c in df_flows.columns]
        if len(feat_match) >= 10:
            if self.aligner is not None and getattr(self.aligner, "is_fitted", False):
                X_proc = self.aligner.transform(df_flows)
            else:
                X_proc = self.preprocessor.transform(df_flows)

            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_proc)[:, 1]
            else:
                probs = self.model.predict(X_proc).astype(float)

            thresh = self.decision_threshold
            if self.calibrator is not None and getattr(self.calibrator, "calibrated_threshold", None) is not None:
                thresh = self.calibrator.calibrated_threshold

            for idx, prob in enumerate(probs):
                row = df_flows.iloc[idx]
                p_val = float(prob)
                is_anom = p_val >= thresh
                r_score = float(np.clip(p_val / max(thresh * 2.0, 1e-4), 0.0, 1.0))
                r_level = "CRITICAL" if p_val >= 0.85 else ("HIGH" if p_val >= thresh else ("MEDIUM" if p_val >= thresh * 0.5 else "LOW"))

                alert = IDSAlertOutput(
                    flow_id=str(row.get("flow_id", f"flow_{idx}")),
                    src_ip=str(row.get("src_host", row.get("src_ip", "0.0.0.0"))),
                    dst_ip=str(row.get("dst_host", row.get("dst_ip", "0.0.0.0"))),
                    timestamp=float(row.get("timestamp_start", row.get("timestamp", 0.0))),
                    prediction_prob=p_val,
                    decision_threshold=thresh,
                    is_anomaly=is_anom,
                    risk_score=r_score,
                    risk_level=r_level,
                    feature_profile=self.feature_profile,
                    raw_features={c: float(row[c]) for c in feat_match if c in row and isinstance(row[c], (int, float, np.number))},
                )
                results.append(alert.to_dict())
            return results

        # Slow path: Raw flow fields needing CanonicalFlow construction & 18-feature extraction
        for _, row in df_flows.iterrows():
            t_start = float(row.get("timestamp_start", 0.0))
            t_end = float(row.get("timestamp_end", t_start + 1.0))
            if t_end < t_start:
                t_end = t_start + 1.0

            flow = CanonicalFlow(
                timestamp_start=t_start,
                timestamp_end=t_end,
                src_host=str(row.get("src_host", "192.168.1.1")),
                dst_host=str(row.get("dst_host", "10.0.0.1")),
                src_port=int(row.get("src_port", 12345)),
                dst_port=int(row.get("dst_port", 80)),
                protocol=str(row.get("protocol", "tcp")).lower(),
                bytes_src=int(row.get("bytes_src", 100)),
                bytes_dst=int(row.get("bytes_dst", 100)),
                pkts_src=int(row.get("pkts_src", 2)),
                pkts_dst=int(row.get("pkts_dst", 2)),
                syn_count_src=int(row.get("syn_count_src", 1)),
                ack_count_src=int(row.get("ack_count_src", 1)),
                rst_count_src=int(row.get("rst_count_src", 0)),
                conn_state_code=int(row.get("conn_state_code", 0)),
                label=int(row.get("label", 0)),
                attack_category=str(row.get("attack_category", "Normal")),
            )
            results.append(self.process_flow(flow))
        return results
