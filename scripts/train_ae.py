"""Milestone 4B — Autoencoder Training.

Trains an Autoencoder for anomaly detection on each dataset using the splits generated in Milestone 3A.
Uses partial_fit with MLPRegressor to train out-of-core on Benign data only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve
import joblib
import hashlib
import os
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(ROOT))
DATASETS = ["CICIDS2017", "Edge-IIoTset", "N-BaIoT", "BoT-IoT"]
if os.environ.get("SMOKE_TEST") == "1":
    DATASETS = [os.environ.get("SMOKE_DATASET", "N-BaIoT")]

# Training profile configuration
PROFILE = os.environ.get("PROFILE", "in_domain")
_ablation = os.environ.get("ABLATION_LEVELS")
ABLATION_LEVELS = _ablation.split(",") if _ablation else None

SPLIT_VERSION = os.environ.get("SPLIT_VERSION", "split-v1")
MODEL_VERSION = os.environ.get("MODEL_VERSION", PROFILE)

def log(msg: str) -> None:
    print(f"[4B AE] {time.strftime('%H:%M:%S')} {msg}", flush=True)

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

def load_benign_train(dataset: str, split_manifest: dict, drop_indices: dict):
    parquet_files = get_parquet_files(dataset, split_manifest)
    split_drops = drop_indices.get("train", {})
    from run_milestone3a import split_assigner, hash_to_split
    from src.iot_ids.preprocessing.pipeline import FittedPreprocessor
    import torch
    from torch.utils.data import TensorDataset
    preprocessor = FittedPreprocessor.load(ROOT / "models" / PROFILE / "preprocessing" / f"{dataset}_preprocessor.joblib")
    
    for file_idx, parquet in enumerate(parquet_files):
        df = pd.read_parquet(parquet)
        file_drop_set = set(split_drops.get(str(file_idx), []))
        keep_mask = np.zeros(len(df), dtype=bool)
        col_to_idx = {c: i for i, c in enumerate(df.columns)}
        
        for row_idx, row in enumerate(df.itertuples(index=False, name=None)):
            if row_idx in file_drop_set: continue
            
            if dataset == "Edge-IIoTset":
                row_text = "\x1f".join("" if pd.isna(row[i]) else str(row[i]) for i in range(len(row)))
                split = hash_to_split(hashlib.sha256(row_text.encode("utf-8")).hexdigest())
            else:
                split = split_assigner(dataset, row, file_idx, parquet.name, col_to_idx)
                
            if split == "train" and row[col_to_idx["canonical_label"]] == "BENIGN":
                keep_mask[row_idx] = True
                
        if keep_mask.any():
            chunk = df[keep_mask].copy()
            X_np = preprocessor.transform(chunk)
            y_np = (chunk["canonical_label"] != "BENIGN").astype(int).values
            yield torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

def load_full_split(dataset: str, split_name: str, split_manifest: dict, drop_indices: dict):
    from torch.utils.data import TensorDataset
    import torch
    import hashlib
    parquet_files = get_parquet_files(dataset, split_manifest)
    split_drops = drop_indices.get(split_name, {})
    from run_milestone3a import split_assigner, hash_to_split
    from src.iot_ids.preprocessing.pipeline import FittedPreprocessor
    preprocessor = FittedPreprocessor.load(ROOT / "models" / PROFILE / "preprocessing" / f"{dataset}_preprocessor.joblib")
    
    frames = []
    for file_idx, parquet in enumerate(parquet_files):
        df = pd.read_parquet(parquet)
        file_drop_set = set(split_drops.get(str(file_idx), []))
        keep_mask = np.zeros(len(df), dtype=bool)
        col_to_idx = {c: i for i, c in enumerate(df.columns)}
        
        for row_idx, row in enumerate(df.itertuples(index=False, name=None)):
            if row_idx in file_drop_set: continue
            if dataset == "Edge-IIoTset":
                row_text = "\x1f".join("" if pd.isna(row[i]) else str(row[i]) for i in range(len(row)))
                split = hash_to_split(hashlib.sha256(row_text.encode("utf-8")).hexdigest())
            else:
                split = split_assigner(dataset, row, file_idx, parquet.name, col_to_idx)
            if split == split_name:
                keep_mask[row_idx] = True
                
        if keep_mask.any():
            frames.append(df[keep_mask])
            
    if not frames:
        return TensorDataset(torch.empty(0), torch.empty(0))
        
    full_df = pd.concat(frames, ignore_index=True)
    X_np = preprocessor.transform(full_df)
    y_np = (full_df["canonical_label"] != "BENIGN").astype(int).values
    
    return TensorDataset(torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32).unsqueeze(1))

def train_ae(dataset: str) -> None:
    if (ROOT / "models" / MODEL_VERSION / "autoencoder" / f"{dataset}_ae.joblib").exists():
        log(f"Skipping {dataset}: already trained for profile {PROFILE}.")
        return

    manifest_path = ROOT / "data" / "processed" / "splits" / SPLIT_VERSION / dataset / "split_manifest.json"
    if not manifest_path.exists():
        log(f"Skipping {dataset}: split manifest not found.")
        return
        
    manifest = read_json(manifest_path)
    drop_indices = get_drop_indices(dataset)
    
    # MLPRegressor works as an Autoencoder when X = Y
    ae = MLPRegressor(
        hidden_layer_sizes=(32, 16, 32),
        activation='relu',
        solver='adam',
        max_iter=1,  # partial_fit
        random_state=42
    )
    
    log(f"[{dataset}] Training Autoencoder on BENIGN traffic (incremental)...")
    t0 = time.time()
    train_rows = 0
    
    for i, (X_chunk, _) in enumerate(load_benign_train(dataset, manifest, drop_indices)):
        X_np = X_chunk.numpy()
        ae.partial_fit(X_np, X_np)  # Autoencoder reconstructs inputs
        train_rows += len(X_chunk)
        if (i + 1) % 10 == 0:
            log(f"    [{dataset}] Processed {i+1} chunks (benign rows: {train_rows})")
            
    train_time = time.time() - t0
    log(f"[{dataset}] Training completed in {train_time:.1f}s. Total train benign rows: {train_rows}")
    
    log(f"[{dataset}] Evaluating on Validation Split to find threshold...")
    t0 = time.time()
    
    val_dataset = load_full_split(dataset, "validation", manifest, drop_indices)
    if len(val_dataset) == 0: return
    X_val = val_dataset.tensors[0].numpy()
    y_true_all = val_dataset.tensors[1].numpy().flatten()
    
    X_reconstructed = ae.predict(X_val)
    errors_all = np.mean(np.power(X_val - X_reconstructed, 2), axis=1)
    
    eval_time = time.time() - t0
    
    # Calculate optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_true_all, errors_all)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    y_pred_all = (errors_all > best_threshold).astype(int)
    
    acc = accuracy_score(y_true_all, y_pred_all)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average="binary", zero_division=0)
    
    log(f"[{dataset}] Validation Accuracy: {acc:.4f} | F1: {f1:.4f} | Threshold: {best_threshold:.4f}")
    
    # Save Model
    model_dir = ROOT / "models" / MODEL_VERSION / "autoencoder"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{dataset}_ae.joblib"
    joblib.dump(ae, model_path)
    
    # Save Metrics
    metrics = {
        "dataset": dataset,
        "model": "Autoencoder",
        "profile": PROFILE,
        "ablation_levels": ABLATION_LEVELS,
        "train_benign_rows_used": train_rows,
        "val_rows_used": len(y_true_all),
        "train_time_sec": round(train_time, 2),
        "inference_time_sec": round(eval_time, 2),
        "best_threshold": float(best_threshold),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        },
        "hyperparameters": ae.get_params()
    }
    write_json(model_dir / f"{dataset}_ae_metrics.json", metrics)

def run():
    for ds in DATASETS:
        train_ae(ds)
        print("-" * 60)

if __name__ == "__main__":
    run()
