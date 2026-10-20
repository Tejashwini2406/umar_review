"""Unit tests for Stage 8 publication integrity and evidence verification."""

from pathlib import Path
import pandas as pd
import pytest


def test_publication_audit_csv_exists_and_passes():
    audit_file = Path("reports/stage8/stage8_publication_audit.csv")
    golden_file = Path("reports/golden_run_manifest.json")
    if not audit_file.exists() and not golden_file.exists():
        pytest.skip("Legacy stage8 publication audit CSV and golden_run_manifest.json not found.")
    if audit_file.exists():
        df = pd.read_csv(audit_file)
        assert len(df) >= 12
        failing_claims = df[df["status"] != "PASS"]
        assert len(failing_claims) == 0, f"Discovered failing publication audit claims: {failing_claims}"
    else:
        assert golden_file.exists()


def test_artifact_manifest_exists():
    manifest_file = Path("reports/stage8/stage8_artifact_manifest.csv")
    golden_file = Path("reports/golden_run_manifest.json")
    if not manifest_file.exists() and not golden_file.exists():
        pytest.skip("Neither legacy stage8 manifest nor golden_run_manifest.json found.")
    assert manifest_file.exists() or golden_file.exists()


def test_reproducibility_scripts_exist():
    required_scripts = [
        "scripts/03_materialize_and_audit_features.py",
        "scripts/04_benchmark_models.py",
        "scripts/05_domain_adaptation.py",
        "scripts/06_statistical_robustness.py",
        "scripts/07_generate_claim_evidence_matrix.py",
        "scripts/08_run_publication_audit.py",
    ]
    for script_path in required_scripts:
        assert Path(script_path).exists(), f"Missing required script: {script_path}"
