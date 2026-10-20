from __future__ import annotations

from pathlib import Path
import pandas as pd

from iot_ids.pipeline.system import IDSSystemPipeline
from iot_ids.utils.paths import REPO_ROOT


from iot_ids.runtime.engine import InferenceEngine

def test_system_pipeline_end_to_end() -> None:
    for ds in ["Edge-IIoTset", "ToN-IoT"]:
        engine = InferenceEngine(profile="standardized_21", dataset=ds)
        test_df = pd.read_parquet(REPO_ROOT / "data" / "processed" / "stage3" / ds / "test.parquet").head(10)

        cols = [c for c in engine.feature_names if c in test_df.columns]
        res = engine.predict(test_df[cols])
        assert len(res) == 10
        assert all("probability" in r for r in res)
        assert all(0.0 <= r["probability"] <= 1.0 for r in res)
