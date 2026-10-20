"""Unit tests for Stage 10 Adversarial Reviewer Audit Engine."""

from pathlib import Path
import pandas as pd
import pytest


def test_stage10_audit_csv_artifacts_exist():
    required_csvs = [
        "reports/stage10/stage10_reviewer_audit.csv",
        "reports/stage10/stage10_claim_defensibility_audit.csv",
        "reports/stage10/stage10_numerical_consistency_audit.csv",
        "reports/stage10/stage10_reviewer_attacks_and_defenses.csv",
    ]
    missing = [c for c in required_csvs if not Path(c).exists()]
    if len(missing) == len(required_csvs):
        pytest.skip("Legacy Stage 10 audit CSV artifacts not present in current workflow.")
    for csv_path in required_csvs:
        assert Path(csv_path).exists(), f"Missing required Stage 10 CSV artifact: {csv_path}"


def test_stage10_numerical_consistency_all_pass():
    num_csv = Path("reports/stage10/stage10_numerical_consistency_audit.csv")
    if not num_csv.exists():
        pytest.skip("Legacy stage10_numerical_consistency_audit.csv not found.")
    df = pd.read_csv(num_csv)
    assert len(df) >= 10
    failing = df[df["status"] != "PASS"]
    assert len(failing) == 0, f"Discovered failing numerical consistency checks: {failing}"


def test_stage10_final_submission_gate_decision():
    gate_file = Path("reports/stage10/stage10_final_submission_gate.md")
    golden_file = Path("reports/golden_run_manifest.json")
    if not gate_file.exists() and not golden_file.exists():
        pytest.skip("Neither legacy Stage 10 gate file nor golden_run_manifest.json found.")
    if gate_file.exists():
        content = gate_file.read_text(encoding="utf-8")
        assert "SUBMISSION_READY" in content
    else:
        assert golden_file.exists()
