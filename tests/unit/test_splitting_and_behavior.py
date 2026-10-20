from __future__ import annotations

import pandas as pd
from iot_ids.data.splitting.strategies import SPLIT_PLANS, stratified_split_indices
from iot_ids.features.behavioral.network import add_historical_destination_diversity, entropy


def test_split_plans_exist_for_active_datasets() -> None:
    assert set(SPLIT_PLANS) == {"Edge-IIoTset", "ToN-IoT"}


def test_stratified_split_indices_proportions() -> None:
    df = pd.DataFrame({
        "label": [0] * 60 + [1] * 40,
        "attack_category": ["Normal"] * 60 + ["DoS"] * 20 + ["Scanning"] * 20
    })
    tr, val, tst = stratified_split_indices(df, label_col="label", category_col="attack_category")
    assert len(tr) == 60
    assert len(val) == 20
    assert len(tst) == 20
    assert len(set(tr).intersection(set(val))) == 0
    assert len(set(tr).intersection(set(tst))) == 0
    assert len(set(val).intersection(set(tst))) == 0


def test_behavioral_destination_diversity_is_historical() -> None:
    frame = pd.DataFrame({"src": ["a", "a", "a"], "dst": ["x", "y", "y"]})
    out_df, _ = add_historical_destination_diversity(frame, "src", "dst", "div", window=10)
    assert out_df["div"].tolist() == [0.0, 1.0, 2.0]
    assert entropy(pd.Series(["x", "x", "y"])) > 0
