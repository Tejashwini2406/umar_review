"""End-to-End Integration Tests for Multi-Level Operational Pipeline.

Verifies raw telemetry -> FlowAggregator -> 18-feature extraction -> Preprocessor ->
Model inference -> Threshold decision -> Structured alert output, including save/load round-trips.
"""
import tempfile
import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from iot_ids.data.flow_object import CanonicalFlow
from iot_ids.features.canonical.flow_builder import CANONICAL_18_FEATURE_NAMES
from iot_ids.experiments.preprocessing import FeaturePreprocessor
from iot_ids.pipeline.system import IDSSystemPipeline
from iot_ids.registry.manager import ModelRegistry
from iot_ids.inference.predictor import IDSPredictor


@pytest.fixture
def trained_pipeline():
    """Creates a small, deterministic IDSSystemPipeline fixture."""
    np.random.seed(42)
    n_samples = 50
    df_train = pd.DataFrame(
        np.random.rand(n_samples, len(CANONICAL_18_FEATURE_NAMES)),
        columns=CANONICAL_18_FEATURE_NAMES
    )
    df_train["proto_tcp"] = 1.0
    df_train["proto_udp"] = 0.0
    df_train["proto_icmp"] = 0.0
    df_train["proto_other"] = 0.0

    y_train = np.random.randint(0, 2, size=n_samples)

    preprocessor = FeaturePreprocessor(feature_names=CANONICAL_18_FEATURE_NAMES)
    X_train_proc = preprocessor.fit_transform(df_train)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train_proc, y_train)

    pipeline = IDSSystemPipeline(
        model=model,
        preprocessor=preprocessor,
        feature_profile="full_multilevel",
        decision_threshold=0.5,
    )
    return pipeline


def create_sample_flow(timestamp=100.0, dst_host="10.0.0.1", label=0):
    return CanonicalFlow(
        timestamp_start=timestamp,
        timestamp_end=timestamp + 1.5,
        src_host="192.168.1.10",
        dst_host=dst_host,
        src_port=12345,
        dst_port=80,
        protocol="tcp",
        bytes_src=200,
        bytes_dst=400,
        pkts_src=5,
        pkts_dst=4,
        syn_count_src=1,
        ack_count_src=1,
        rst_count_src=0,
        conn_state_code=0,
        label=label,
        attack_category="Normal" if label == 0 else "DDoS",
    )


def test_e2e_normal_flow_inference(trained_pipeline):
    """Verifies flow -> 18 features -> preprocessor -> model -> alert dict."""
    flow = create_sample_flow()
    res = trained_pipeline.process_flow(flow)

    assert "prediction_prob" in res
    assert "is_anomaly" in res
    assert "risk_score" in res
    assert "risk_level" in res
    assert res["feature_profile"] == "full_multilevel"
    assert 0.0 <= res["prediction_prob"] <= 1.0
    assert isinstance(res["is_anomaly"], bool)


def test_e2e_stateful_feature_evolution_and_reset(trained_pipeline):
    """Verifies temporal/behavioral features evolve across flows and reset cleanly."""
    trained_pipeline.reset_state()
    
    flow1 = create_sample_flow(timestamp=10.0, dst_host="10.0.0.1")
    flow2 = create_sample_flow(timestamp=12.0, dst_host="10.0.0.2")

    res1 = trained_pipeline.process_flow(flow1)
    res2 = trained_pipeline.process_flow(flow2)

    # Destination diversity should increase with new destination host
    div1 = res1["raw_features"]["behavioral_dst_diversity"]
    div2 = res2["raw_features"]["behavioral_dst_diversity"]
    assert div2 >= div1

    # Reset state and process flow1 again
    trained_pipeline.reset_state()
    res1_after_reset = trained_pipeline.process_flow(flow1)
    assert res1_after_reset["raw_features"]["behavioral_dst_diversity"] == div1


def test_e2e_save_load_registry_roundtrip(trained_pipeline):
    """Verifies pipeline save -> load -> predictor roundtrip reproducibility."""
    flow = create_sample_flow()
    res_orig = trained_pipeline.process_flow(flow)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "pipeline_artifact"
        ModelRegistry.save_pipeline(trained_pipeline, save_path, metadata={"version": "1.0"})

        # Instantiate predictor from saved artifact
        predictor = IDSPredictor.from_artifact(save_path)
        predictor.reset_state()
        
        res_loaded = predictor.predict_flow(flow)

        assert np.isclose(res_orig["prediction_prob"], res_loaded["prediction_prob"])
        assert res_orig["is_anomaly"] == res_loaded["is_anomaly"]
        assert res_orig["risk_level"] == res_loaded["risk_level"]


@pytest.mark.parametrize(
    ("metadata_key", "artifact_name"),
    [
        ("has_aligner", "aligner.joblib"),
        ("has_calibrator", "calibrator.joblib"),
    ],
)
def test_registry_fails_if_metadata_requires_missing_component(
    trained_pipeline,
    metadata_key,
    artifact_name,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "pipeline_artifact"
        ModelRegistry.save_pipeline(trained_pipeline, save_path)
        metadata_path = save_path / "pipeline_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata[metadata_key] = True
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        with pytest.raises(FileNotFoundError, match=artifact_name):
            ModelRegistry.load_pipeline(save_path)


def test_e2e_batch_dataframe_inference(trained_pipeline):
    """Verifies batch inference over DataFrame input."""
    df_flows = pd.DataFrame([
        {"timestamp_start": 100.0, "timestamp_end": 101.0, "src_host": "192.168.1.1", "dst_host": "10.0.0.1", "src_port": 1234, "dst_port": 80, "protocol": "tcp", "bytes_src": 100, "bytes_dst": 100, "pkts_src": 2, "pkts_dst": 2, "syn_count_src": 1, "ack_count_src": 1, "rst_count_src": 0, "conn_state_code": 0, "label": 0, "attack_category": "Normal"},
        {"timestamp_start": 102.0, "timestamp_end": 103.0, "src_host": "192.168.1.2", "dst_host": "10.0.0.2", "src_port": 1235, "dst_port": 80, "protocol": "tcp", "bytes_src": 200, "bytes_dst": 200, "pkts_src": 3, "pkts_dst": 3, "syn_count_src": 1, "ack_count_src": 1, "rst_count_src": 0, "conn_state_code": 0, "label": 0, "attack_category": "Normal"},
    ])
    results = trained_pipeline.process_dataframe(df_flows)
    assert len(results) == 2
    assert results[0]["src_ip"] == "192.168.1.1"
    assert results[1]["src_ip"] == "192.168.1.2"


def test_single_flow_and_materialized_feature_batch_have_prediction_parity(
    trained_pipeline,
):
    flow = create_sample_flow(timestamp=300.0)
    trained_pipeline.reset_state()
    single = trained_pipeline.process_flow(flow)

    trained_pipeline.reset_state()
    feature_row = pd.DataFrame([single["raw_features"]])
    batch = trained_pipeline.process_dataframe(feature_row)[0]

    assert np.isclose(single["prediction_prob"], batch["prediction_prob"])
    assert single["decision_threshold"] == batch["decision_threshold"]
    assert single["is_anomaly"] == batch["is_anomaly"]


def test_e2e_deterministic_inference(trained_pipeline):
    """Verifies identical inputs yield deterministic probability and risk scores."""
    flow = create_sample_flow()
    
    trained_pipeline.reset_state()
    res1 = trained_pipeline.process_flow(flow)
    
    trained_pipeline.reset_state()
    res2 = trained_pipeline.process_flow(flow)

    assert res1["prediction_prob"] == res2["prediction_prob"]
    assert res1["risk_score"] == res2["risk_score"]
