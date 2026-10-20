import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from iot_ids.data.splitting.strategies import stratified_split_indices
from iot_ids.utils.paths import REPO_ROOT

def main():
    print("============================================================")
    print("=== STAGE 03: STRUCTURAL SPLIT GENERATION ===")
    print("============================================================\n")

    base_dir = REPO_ROOT / "data" / "processed" / "final"
    datasets = ["Edge-IIoTset", "ToN-IoT"]

    for ds in datasets:
        ds_dir = base_dir / ds
        fingerprint_dirs = [d for d in ds_dir.iterdir() if d.is_dir()]
        target_dir = fingerprint_dirs[0]
        
        splits_dir = target_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        parquet_files = sorted(target_dir.glob("part-*.parquet"))
        print(f"Loading partitions for {ds} split generation...")
        df_list = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(df_list, ignore_index=True)
        print(f"Total Rows Loaded: {len(df):,}")

        train_idx, val_idx, test_idx = stratified_split_indices(
            df,
            label_col="label",
            category_col="attack_category",
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            random_state=42,
        )

        print(f"  -> Train Rows: {len(train_idx):,} ({len(train_idx)/len(df)*100:.1f}%)")
        print(f"  -> Val Rows:   {len(val_idx):,} ({len(val_idx)/len(df)*100:.1f}%)")
        print(f"  -> Test Rows:  {len(test_idx):,} ({len(test_idx)/len(df)*100:.1f}%)")

        df.iloc[train_idx].to_parquet(splits_dir / "train.parquet", index=False)
        df.iloc[val_idx].to_parquet(splits_dir / "val.parquet", index=False)
        df.iloc[test_idx].to_parquet(splits_dir / "test.parquet", index=False)

        manifest = {
            "dataset": ds,
            "total_rows": len(df),
            "train_rows": len(train_idx),
            "val_rows": len(val_idx),
            "test_rows": len(test_idx),
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "complete": True,
        }
        (splits_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Splits saved to {splits_dir}\n")

    print("Stage 03 Complete: All structural splits generated successfully.")

if __name__ == "__main__":
    main()
