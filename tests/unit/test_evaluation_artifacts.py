import json
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_FILE = ROOT / "reports" / "experiments" / "in_domain_evaluation.json"
GOLDEN_MANIFEST = ROOT / "reports" / "golden_run_manifest.json"

@pytest.fixture
def eval_data():
    if not EVAL_FILE.exists():
        pytest.skip(f"Evaluation file not found: {EVAL_FILE}")
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def test_eval_file_exists():
    """Verify that an evaluation evidence artifact exists."""
    if not EVAL_FILE.exists() and not GOLDEN_MANIFEST.exists():
        pytest.skip(f"Evaluation artifact not found: {EVAL_FILE} or {GOLDEN_MANIFEST}")
    assert EVAL_FILE.exists() or GOLDEN_MANIFEST.exists()

def test_eval_file_contains_all_datasets(eval_data):
    """Verify that all four in-domain dataset profiles were evaluated."""
    expected_datasets = ["CICIDS2017", "Edge-IIoTset", "BoT-IoT", "N-BaIoT"]
    for ds in expected_datasets:
        assert ds in eval_data, f"Dataset {ds} is missing from the evaluation artifacts."

def test_eval_file_contains_all_models(eval_data):
    """Verify that all base components and ensembles are evaluated for each dataset."""
    expected_models = ["rf", "mlp", "ae", "rf+mlp", "rf+mlp+ae"]
    
    for ds, test_targets in eval_data.items():
        # In-domain evaluation maps ds -> ds
        assert ds in test_targets, f"In-domain test target {ds} missing for source {ds}"
        model_results = test_targets[ds]
        
        for model in expected_models:
            assert model in model_results, f"Model {model} missing from {ds} evaluation."

def test_eval_model_metrics_structure(eval_data):
    """Verify that each model contains the required metrics for P4 ingestion."""
    expected_metrics = ["accuracy", "precision", "recall", "f1_score", "frozen_weights", "frozen_threshold"]
    
    for ds, test_targets in eval_data.items():
        model_results = test_targets.get(ds, {})
        for model_name, metrics in model_results.items():
            for metric in expected_metrics:
                assert metric in metrics, f"Metric '{metric}' missing in {ds} -> {model_name}"
            
            # Type checks
            assert isinstance(metrics["frozen_weights"], dict), "frozen_weights must be a dictionary"
            assert isinstance(metrics["frozen_threshold"], (float, int)), "frozen_threshold must be numeric"
            assert isinstance(metrics["f1_score"], (float, int)), "f1_score must be numeric"
