"""Unit tests for Stage 11 submission packaging and final artifact verification."""

from pathlib import Path
import pandas as pd
import pytest


def test_stage11_submission_package_exists():
    sub_dir = Path("reports/stage11/submission")
    paper_dir = Path("final_ieee_paper")
    if not sub_dir.exists() and not paper_dir.exists():
        pytest.skip("Neither legacy Stage 11 submission dir nor final_ieee_paper found.")
    if sub_dir.exists():
        assert (sub_dir / "IEEE_paper_final.tex").exists()
    else:
        assert (paper_dir / "main.tex").exists()
        assert (paper_dir / "references.bib").exists()


def test_stage11_checksums_csv():
    checksum_file = Path("reports/stage11/stage11_checksums.csv")
    if not checksum_file.exists():
        pytest.skip("Legacy stage11_checksums.csv not found.")
    df = pd.read_csv(checksum_file)
    assert len(df) >= 10
    assert "sha256_checksum" in df.columns
    assert (df["size_bytes"] > 0).all()


def test_stage11_figures_tables_audit_pass():
    audit_file = Path("reports/stage11/stage11_figures_tables_audit.csv")
    if not audit_file.exists():
        pytest.skip("Legacy stage11_figures_tables_audit.csv not found.")
    df = pd.read_csv(audit_file)
    assert len(df) >= 17
    failing = df[df["status"] != "PASS"]
    assert len(failing) == 0, f"Discovered failing figure/table audit checks: {failing}"


def test_stage11_placeholder_audit_pass():
    audit_file = Path("reports/stage11/stage11_placeholder_audit.csv")
    if not audit_file.exists():
        pytest.skip("Legacy stage11_placeholder_audit.csv not found.")
    df = pd.read_csv(audit_file)
    assert len(df) >= 7
    assert (df["status"] == "PASS").all()
