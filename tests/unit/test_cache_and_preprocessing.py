from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from iot_ids.data.cache import cache_fingerprint
from iot_ids.data.materialize import load_valid_manifest
from iot_ids.preprocessing.pipeline import PreprocessingPipeline


def test_cache_fingerprint_is_deterministic(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    a.write_text("x\n1\n", encoding="utf-8")
    one = cache_fingerprint([a], "s1", "l1", "f1")
    two = cache_fingerprint([a], "s1", "l1", "f1")
    assert one == two
    assert one != cache_fingerprint([a], "s2", "l1", "f1")


def test_preprocessing_pipeline_fit_transform() -> None:
    train_df = pd.DataFrame({"a": [0.0, 10.0, 100.0], "b": [1.0, 2.0, 3.0]})
    test_df = pd.DataFrame({"a": [50.0, 200.0], "b": [2.0, 4.0]})
    
    prep = PreprocessingPipeline(numeric_cols=["a", "b"])
    X_tr = prep.fit_transform(train_df)
    X_tst = prep.transform(test_df)
    
    assert X_tr.shape == (3, 2)
    assert X_tst.shape == (2, 2)


def test_incomplete_cache_manifest_is_not_valid(tmp_path: Path) -> None:
    out = tmp_path / "cache"
    out.mkdir()
    manifest = out / "manifest.json"
    manifest.write_text('{"complete": false, "rows": 10, "partitions": 1}', encoding="utf-8")
    assert load_valid_manifest(manifest, out) is None
    (out / "part-0000-0000.parquet").write_text("placeholder", encoding="utf-8")
    manifest.write_text('{"complete": true, "rows": 10, "partitions": 2}', encoding="utf-8")
    assert load_valid_manifest(manifest, out) is None
