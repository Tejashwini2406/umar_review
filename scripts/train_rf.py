"""Milestone 3B — Random Forest Training.

Trains a Random Forest classifier on each dataset using the splits generated in Milestone 3A.
For large datasets like BoT-IoT, uses uniform random sampling to fit into memory,
ensuring a robust globally representative model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import joblib
import hashlib
import os
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(ROOT))
DATASETS = ["CICIDS2017", "Edge-IIoTset", "N-BaIoT", "BoT-IoT"]
if os.environ.get("SMOKE_TEST") == "1":
    DATASETS = [os.environ.get("SMOKE_DATASET", "N-BaIoT")]

# Downsample large datasets to avoid OOM while preserving global representation
MAX_TRAIN_ROWS = 2_000_000
if os.environ.get("SMOKE_TEST") == "1":
    MAX_TRAIN_ROWS = 1000
MAX_TEST_ROWS = 1_000_000

# Training profile configuration
PROFILE = os.environ.get("PROFILE", "in_domain")
_ablation = os.environ.get("ABLATION_LEVELS")
ABLATION_LEVELS = _ablation.split(",") if _ablation else None

SPLIT_VERSION = os.environ.get("SPLIT_VERSION", "split-v1")
MODEL_VERSION = os.environ.get("MODEL_VERSION", PROFILE)

def log(msg: str) -> None:
    print(f"[3B RF] {time.strftime('%H:%M:%S')} {msg}", flush=True)

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

def get_parquet_files(dataset: str, split_manifest: dict) -> list[Path]:
    fp = split_manifest["fingerprint"]
    for cache_name in ("v3_cache", "v2_cache", "research"):
        cache_dir = ROOT / "data" / "processed" / cache_name / dataset / fp
        if cache_dir.exists():
            return sorted(cache_dir.glob("part-*.parquet"))
    return sorted((ROOT / "data" / "processed" / "research" / dataset / fp).glob("part-*.parquet"))

def get_drop_indices(dataset: str) -> dict[str, dict[str, list[int]]]:
    drop_path = ROOT / "data" / "processed" / "splits" / SPLIT_VERSION / dataset / "decontamination_drop_indices.json"
    if not drop_path.exists():
        return {}
    return read_json(drop_path)

def load_split(dataset: str, split_name: str, split_manifest: dict, drop_indices: dict, max_attack_rows: int = None, is_train: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Loads a specific split, applying decontamination and attack-family stratified downsampling."""
    parquet_files = get_parquet_files(dataset, split_manifest)
    
    from run_milestone3a import split_assigner, hash_to_split
    split_drops = drop_indices.get(split_name, {})
    
    # Pass 1: Compute attack family distribution to determine stratified sampling fractions
    family_counts = {}
    total_attacks = 0
    if is_train and max_attack_rows:
        log(f"    [{dataset}] Profiling attack families for stratified sampling...")
        for file_idx, parquet in enumerate(parquet_files):
            df = pd.read_parquet(parquet)
            col_to_idx = {c: i for i, c in enumerate(df.columns)}
            file_drop_set = set(split_drops.get(str(file_idx), []))
            
            for row_idx, row in enumerate(df.itertuples(index=False, name=None)):
                if row_idx in file_drop_set: continue
                
                if dataset == "Edge-IIoTset":
                    row_text = "\x1f".join("" if pd.isna(row[i]) else str(row[i]) for i in range(len(row)))
                    split = hash_to_split(hashlib.sha256(row_text.encode("utf-8")).hexdigest())
                else:
                    split = split_assigner(dataset, row, file_idx, parquet.name, col_to_idx)
                
                if split == split_name and row[col_to_idx["canonical_label"]] != "BENIGN":
                    family = row[col_to_idx.get("attack_family", col_to_idx["canonical_label"])]
                    family_counts[family] = family_counts.get(family, 0) + 1
                    total_attacks += 1

    keep_fractions = {f: 1.0 for f in family_counts}
    if max_attack_rows and total_attacks > max_attack_rows:
        scale = max_attack_rows / total_attacks
        keep_fractions = {f: scale for f in family_counts}
        log(f"    [{dataset}] Stratifying {total_attacks} attacks to {max_attack_rows} (Scale: {scale:.4f})")

    frames = []
    for file_idx, parquet in enumerate(parquet_files):
        df = pd.read_parquet(parquet)
        col_to_idx = {c: i for i, c in enumerate(df.columns)}
        file_drop_set = set(split_drops.get(str(file_idx), []))
        keep_mask = np.zeros(len(df), dtype=bool)
        
        for row_idx, row in enumerate(df.itertuples(index=False, name=None)):
            if dataset == "Edge-IIoTset":
                row_text = "\x1f".join("" if pd.isna(row[i]) else str(row[i]) for i in range(len(row)))
                split = hash_to_split(hashlib.sha256(row_text.encode("utf-8")).hexdigest())
            else:
                split = split_assigner(dataset, row, file_idx, parquet.name, col_to_idx)
                
            if split == split_name and row_idx not in file_drop_set:
                canonical = row[col_to_idx["canonical_label"]]
                if is_train and canonical != "BENIGN":
                    family = row[col_to_idx.get("attack_family", col_to_idx["canonical_label"])]
                    fraction = keep_fractions.get(family, 1.0)
                    sample_hash = int(hashlib.md5(f"{file_idx}_{row_idx}_42".encode()).hexdigest()[:8], 16) / 0xffffffff
                    if sample_hash < fraction:
                        keep_mask[row_idx] = True
                else:
                    keep_mask[row_idx] = True
                    
        if keep_mask.any():
            frames.append(df[keep_mask])
            
    if not frames: return pd.DataFrame(), pd.Series()
    full_df = pd.concat(frames, ignore_index=True)
    y = (full_df["canonical_label"] != "BENIGN").astype(int)
    return full_df, y

def train_rf(dataset: str) -> None:
    if (ROOT / "models" / MODEL_VERSION / "random_forest" / f"{dataset}_rf.joblib").exists():
        log(f"Skipping {dataset}: already trained for profile {PROFILE}.")
        return

    manifest_path = ROOT / "data" / "processed" / "splits" / SPLIT_VERSION / dataset / "split_manifest.json"
    if not manifest_path.exists():
        log(f"Skipping {dataset}: split manifest not found. Run Milestone 3A first.")
        return
        
    manifest = read_json(manifest_path)
    drop_indices = get_drop_indices(dataset)
    
    log(f"[{dataset}] Loading Train Split (Class-Aware Sampled)...")
    t0 = time.time()
    X_train_raw, y_train = load_split(dataset, "train", manifest, drop_indices, max_attack_rows=MAX_TRAIN_ROWS, is_train=True)
    log(f"[{dataset}] Train Loaded: {len(X_train_raw)} rows in {time.time()-t0:.1f}s")
    
    from src.iot_ids.preprocessing.pipeline import get_model_feature_columns, fit_preprocessor, FittedPreprocessor
    base_cols = get_model_feature_columns(dataset, profile=PROFILE, levels=ABLATION_LEVELS)
    feature_cols = []
    for c in base_cols:
        if c.endswith("*"):
            prefix = c[:-1]
            feature_cols.extend([col for col in X_train_raw.columns if col.startswith(prefix)])
        else:
            if c in X_train_raw.columns:
                feature_cols.append(c)
                
    log(f"[{dataset}] [{PROFILE}] Fitting preprocessor on {len(feature_cols)} features...")
    preprocessor = fit_preprocessor(X_train_raw, feature_cols)
    prep_dir = ROOT / "models" / PROFILE / "preprocessing"
    prep_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save(prep_dir / f"{dataset}_preprocessor.joblib")
    
    X_train = preprocessor.transform(X_train_raw)
    
    log(f"[{dataset}] Loading Validation Split (Full)...")
    X_val_raw, y_val = load_split(dataset, "validation", manifest, drop_indices, is_train=False)
    X_val = preprocessor.transform(X_val_raw)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    
    log(f"[{dataset}] Training Random Forest...")
    t0 = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - t0
    log(f"[{dataset}] Training completed in {train_time:.1f}s")
    
    log(f"[{dataset}] Evaluating on Validation Split...")
    t0 = time.time()
    y_pred = rf.predict(X_val)
    eval_time = time.time() - t0
    
    acc = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)
    
    log(f"[{dataset}] Validation Accuracy: {acc:.4f} | F1: {f1:.4f}")
    
    # Save Model
    model_dir = ROOT / "models" / MODEL_VERSION / "random_forest"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{dataset}_rf.joblib"
    joblib.dump(rf, model_path)
    
    # Save Metrics
    metrics = {
        "dataset": dataset,
        "model": "RandomForest",
        "profile": PROFILE,
        "ablation_levels": ABLATION_LEVELS,
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "train_rows_used": len(X_train),
        "val_rows_used": len(X_val),
        "train_time_sec": round(train_time, 2),
        "inference_time_sec": round(eval_time, 2),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        },
        "hyperparameters": rf.get_params()
    }
    write_json(model_dir / f"{dataset}_rf_metrics.json", metrics)

def run():
    for ds in DATASETS:
        train_rf(ds)
        print("-" * 60)

if __name__ == "__main__":
    run()
