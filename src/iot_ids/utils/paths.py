from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "raw"
REPORTS_ROOT = REPO_ROOT / "reports"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "p1"


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate

