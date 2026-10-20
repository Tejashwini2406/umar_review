import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from iot_ids.data.materialize import materialize_dataset
from iot_ids.utils.paths import REPO_ROOT

def main():
    print("============================================================")
    print("=== STAGE 01: FINAL DATASET MATERIALIZATION ===")
    print("============================================================\n")

    data_root = REPO_ROOT / "data" / "raw"
    output_root = REPO_ROOT / "data" / "processed" / "final"

    datasets = ["Edge-IIoTset", "ToN-IoT"]
    manifests = {}

    for ds in datasets:
        print(f"Materializing {ds} into {output_root / ds}...")
        manifest = materialize_dataset(
            dataset=ds,
            data_root=data_root,
            output_root=output_root,
            chunksize=250_000,
            force=False,
        )
        manifests[ds] = manifest
        print(f"  -> Done: {manifest['rows']:,} rows, {manifest['partitions']} partitions, fingerprint: {manifest['fingerprint']}")
        print()

    print("Stage 01 Complete: All active datasets materialized successfully.")

if __name__ == "__main__":
    main()
