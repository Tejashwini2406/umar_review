"""Stage 2 Empirical Feature Quality & Diagnostics Script.

Streams canonical flows from ToN-IoT, Edge-IIoTset, NF-ToN-IoT-v2, and CICIoT2023 through
CanonicalFlowBuilder, collecting empirical distribution statistics, zero rates, state initialization
counts, and sliding window eviction counts for all candidate multi-level features.
"""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd

from iot_ids.data.adapters import (
    ToNIoTAdapter,
    EdgeIIoTsetAggregatorAdapter,
    NFToNIoTAdapter,
    CICIoT2023Adapter,
)
from iot_ids.features.canonical.flow_builder import CanonicalFlowBuilder, CANONICAL_18_FEATURE_NAMES


def audit_dataset_features(name: str, adapter, sample_limit: int = 10000) -> pd.DataFrame:
    print(f"--- Running Stage 2 Feature Diagnostics for: {name} ---")
    t0 = time.time()

    builder = CanonicalFlowBuilder(alpha=0.1, window_seconds=30.0)
    builder.reset_state()

    flow_stream = adapter.stream_canonical_flows(chunksize=25000)
    feature_vectors = []

    for i, flow in enumerate(flow_stream):
        vec = builder.build_vector(flow)
        feature_vectors.append(vec)
        if len(feature_vectors) >= sample_limit:
            break

    t1 = time.time()
    matrix = np.array(feature_vectors, dtype=np.float64)
    print(f"Ingested {len(matrix)} flows in {t1 - t0:.2f}s")
    print(f"  - First Event State Initializations: {builder.temporal_extractor.first_event_count}")
    print(f"  - Total State Updates: {builder.temporal_extractor.state_update_count}")
    print(f"  - Total 30s Window Evictions: {builder.behavioral_extractor.total_evictions}")

    # Compute statistical summary
    stats_list = []
    for idx, f_name in enumerate(CANONICAL_18_FEATURE_NAMES):
        col = matrix[:, idx]
        non_nan = col[~np.isnan(col)]
        zero_rate = np.mean(col == 0.0) if len(col) > 0 else 0.0
        missing_rate = np.mean(np.isnan(col)) if len(col) > 0 else 0.0

        stats_list.append(
            {
                "Dataset": name,
                "Feature": f_name,
                "Mean": float(np.mean(non_nan)) if len(non_nan) > 0 else 0.0,
                "Std": float(np.std(non_nan)) if len(non_nan) > 0 else 0.0,
                "Min": float(np.min(non_nan)) if len(non_nan) > 0 else 0.0,
                "Median": float(np.median(non_nan)) if len(non_nan) > 0 else 0.0,
                "Max": float(np.max(non_nan)) if len(non_nan) > 0 else 0.0,
                "Q25": float(np.percentile(non_nan, 25)) if len(non_nan) > 0 else 0.0,
                "Q75": float(np.percentile(non_nan, 75)) if len(non_nan) > 0 else 0.0,
                "Q95": float(np.percentile(non_nan, 95)) if len(non_nan) > 0 else 0.0,
                "ZeroRate": float(zero_rate),
                "MissingRate": float(missing_rate),
            }
        )

    return pd.DataFrame(stats_list)


def main():
    raw_dir = Path("data/raw")

    datasets = [
        ("ToN-IoT", ToNIoTAdapter(raw_dir / "ToN-IoT")),
        ("Edge-IIoTset", EdgeIIoTsetAggregatorAdapter(raw_dir / "Edge-IIoTset")),
        ("NF-ToN-IoT-v2", NFToNIoTAdapter(raw_dir / "NF-ToN-IoT-v2")),
        ("CICIoT2023", CICIoT2023Adapter(raw_dir / "CICIOT23")),
    ]

    all_dfs = []
    for name, adapter in datasets:
        df_stats = audit_dataset_features(name, adapter, sample_limit=10000)
        all_dfs.append(df_stats)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save detailed CSV summary
    output_csv = Path("reports/stage2_feature_diagnostics.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"\nSaved empirical feature diagnostics matrix to {output_csv}")


if __name__ == "__main__":
    main()
