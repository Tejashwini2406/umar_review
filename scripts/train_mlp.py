"""Milestone 4A — Multi-Layer Perceptron (MLP) Training.

Trains a genuine configurable tabular PyTorch MLP on each dataset.
Uses an IterableDataset to stream training data out-of-core, supporting all 52M rows,
while optionally applying epoch sampling/limits to avoid redundant compute.
Includes AdamW, early stopping, best-checkpoint restoration, and class-aware loss.
"""
from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import os
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(ROOT))
DATASETS = ["CICIDS2017", "Edge-IIoTset", "N-BaIoT", "BoT-IoT"]
if os.environ.get("SMOKE_TEST") == "1":
    DATASETS = [os.environ.get("SMOKE_DATASET", "N-BaIoT")]

# Training scale policy
MAX_TRAIN_ROWS_PER_EPOCH = 5_000_000
if os.environ.get("SMOKE_TEST") == "1":
    MAX_TRAIN_ROWS_PER_EPOCH = 1000

# Training profile configuration
PROFILE = os.environ.get("PROFILE", "in_domain")
_ablation = os.environ.get("ABLATION_LEVELS")
ABLATION_LEVELS = _ablation.split(",") if _ablation else None

SPLIT_VERSION = os.environ.get("SPLIT_VERSION", "split-v1")
MODEL_VERSION = os.environ.get("MODEL_VERSION", PROFILE)

def log(msg: str) -> None:
    print(f"[4A MLP] {time.strftime('%H:%M:%S')} {msg}", flush=True)

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

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class IDSStreamDataset(IterableDataset):
    def __init__(self, dataset: str, split_name: str, split_manifest: dict, drop_indices: dict, max_attack_rows: int = None, is_train: bool = False):
        self.dataset = dataset
        self.split_name = split_name
        self.split_manifest = split_manifest
        self.drop_indices = drop_indices.get(split_name, {})
        self.max_attack_rows = max_attack_rows
        self.is_train = is_train
        self.parquet_files = get_parquet_files(dataset, split_manifest)
        self.epoch = 0
        from src.iot_ids.preprocessing.pipeline import FittedPreprocessor
        self.preprocessor = FittedPreprocessor.load(ROOT / "models" / PROFILE / "preprocessing" / f"{dataset}_preprocessor.joblib")
        
        # Profile attack family distribution just like RF for downsampling
        self.keep_fractions = {}
        
        # --- EPOCH CACHE BYPASS ---
        # If a pre-materialized epoch cache exists for this dataset/split, we skip profiling entirely!
        cache_root = ROOT / "data" / "processed" / "v2_cache" / "sampling" / self.dataset
        self.use_cache = self.is_train and self.max_attack_rows and (cache_root / "manifest.json").exists()
        
        if self.is_train and self.max_attack_rows and not self.use_cache:
            from run_milestone3a import split_assigner, hash_to_split
            family_counts = {}
            total_attacks = 0
            for file_idx, parquet in enumerate(self.parquet_files):
                df = pd.read_parquet(parquet)
                col_to_idx = {c: i for i, c in enumerate(df.columns)}
                file_drop_set = set(self.drop_indices.get(str(file_idx), []))
                
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

            self.keep_fractions = {f: 1.0 for f in family_counts}
            if total_attacks > max_attack_rows:
                scale = max_attack_rows / total_attacks
                self.keep_fractions = {f: scale for f in family_counts}

    def __iter__(self):
        from run_milestone3a import split_assigner, hash_to_split
        
        if getattr(self, "use_cache", False):
            # FAST PATH: Stream pre-materialized exact cache chunks
            cache_root = ROOT / "data" / "processed" / "v2_cache" / "sampling" / self.dataset
            epoch_dir = cache_root / f"epoch_{self.epoch}"
            if not epoch_dir.exists():
                raise FileNotFoundError(f"Missing epoch {self.epoch} cache at {epoch_dir}")
                
            cache_parquets = sorted(epoch_dir.glob("part-*.parquet"))
            for p in cache_parquets:
                df = pd.read_parquet(p)
                X_np = self.preprocessor.transform(df)
                y_np = (df["canonical_label"] != "BENIGN").astype(int).values
                for i in range(len(X_np)):
                    yield torch.tensor(X_np[i], dtype=torch.float32), torch.tensor([y_np[i]], dtype=torch.float32)
            return

        # SLOW PATH: Original filtering logic
        for file_idx, parquet in enumerate(self.parquet_files):
            df = pd.read_parquet(parquet)
            file_drop_set = set(self.drop_indices.get(str(file_idx), []))
            keep_mask = np.zeros(len(df), dtype=bool)
            col_to_idx = {c: i for i, c in enumerate(df.columns)}
            
            for row_idx, row in enumerate(df.itertuples(index=False, name=None)):
                if row_idx in file_drop_set: continue
                if self.dataset == "Edge-IIoTset":
                    row_text = "\x1f".join("" if pd.isna(row[i]) else str(row[i]) for i in range(len(row)))
                    split = hash_to_split(hashlib.sha256(row_text.encode("utf-8")).hexdigest())
                else:
                    split = split_assigner(self.dataset, row, file_idx, parquet.name, col_to_idx)
                    
                if split == self.split_name:
                    canonical = row[col_to_idx["canonical_label"]]
                    if self.is_train and canonical != "BENIGN":
                        family = row[col_to_idx.get("attack_family", col_to_idx["canonical_label"])]
                        fraction = self.keep_fractions.get(family, 1.0)
                        sample_hash = int(hashlib.md5(f"{file_idx}_{row_idx}_{self.epoch}".encode()).hexdigest()[:8], 16) / 0xffffffff
                        if sample_hash < fraction:
                            keep_mask[row_idx] = True
                    else:
                        keep_mask[row_idx] = True
            
            if keep_mask.any():
                chunk_df = df[keep_mask].copy()
                X_np = self.preprocessor.transform(chunk_df)
                y_np = (chunk_df["canonical_label"] != "BENIGN").astype(int).values
                
                for i in range(len(X_np)):
                    yield torch.tensor(X_np[i], dtype=torch.float32), torch.tensor([y_np[i]], dtype=torch.float32)

class MLPModel(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int] = [128, 64], dropout: float = 0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def load_full_split(dataset: str, split_name: str, split_manifest: dict, drop_indices: dict) -> TensorDataset:
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

def train_mlp(dataset: str) -> None:
    if (ROOT / "models" / MODEL_VERSION / "neural_network" / f"{dataset}_mlp.pt").exists():
        log(f"Skipping {dataset}: already trained for profile {PROFILE}.")
        return

    manifest_path = ROOT / "data" / "processed" / "splits" / SPLIT_VERSION / dataset / "split_manifest.json"
    if not manifest_path.exists():
        log(f"Skipping {dataset}: split manifest not found.")
        return
        
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[{dataset}] Using device: {device}")
    
    manifest = read_json(manifest_path)
    drop_indices = get_drop_indices(dataset)
    
    # Validation dataset (in-memory)
    # Validation dataset (in-memory)
    log(f"[{dataset}] Loading Validation Split...")
    val_dataset = load_full_split(dataset, "validation", manifest, drop_indices)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)
    
    log(f"[{dataset}] Initializing Train Split Stream (Max {MAX_TRAIN_ROWS_PER_EPOCH} attacks/epoch)...")
    train_dataset = IDSStreamDataset(dataset, "train", manifest, drop_indices, max_attack_rows=MAX_TRAIN_ROWS_PER_EPOCH, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=1024)
    
    # Determine input dimension by peeking at first row
    first_X, _ = next(iter(train_loader))
    input_dim = first_X.shape[1]
    
    model = MLPModel(input_dim=input_dim).to(device)
    
    # Calculate class weights for imbalance
    stats = manifest["splits"]["train"]
    total_benign = stats.get("benign", 1)
    total_attack = stats.get("attack", 1)
    pos_weight_val = total_benign / total_attack if total_attack > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 10
    if os.environ.get("SMOKE_TEST") == "1":
        epochs = 1
    patience = 3
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    log(f"[{dataset}] Starting training...")
    t0_train = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        # Reseed attack sampling for broader coverage
        train_dataset.epoch = epoch
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_steps += 1
            
        avg_train_loss = train_loss / max(1, train_steps)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        y_true_val, y_pred_val = [], []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_steps += 1
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                y_true_val.extend(batch_y.cpu().numpy())
                y_pred_val.extend((probs > 0.5).astype(int))
                
        avg_val_loss = val_loss / max(1, val_steps)
        val_acc = accuracy_score(y_true_val, y_pred_val)
        
        log(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(f"  Early stopping at epoch {epoch+1}")
                break
                
    total_train_time = time.time() - t0_train
    
    # Restore best checkpoint
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    # Final Validation Metrics with best model
    model.eval()
    y_true_final, y_pred_final = [], []
    t0_eval = time.time()
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_true_final.extend(batch_y.numpy())
            y_pred_final.extend((probs > 0.5).astype(int))
            
    eval_time = time.time() - t0_eval
    
    acc = accuracy_score(y_true_final, y_pred_final)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_final, y_pred_final, average="binary", zero_division=0)
    log(f"[{dataset}] Final Val Accuracy: {acc:.4f} | F1: {f1:.4f}")
    
    # Save Model
    model_dir = ROOT / "models" / MODEL_VERSION / "neural_network"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # We save the entire model (architecture + weights) or state_dict. Let's save state_dict for best practices.
    # We also need input_dim for loading
    torch.save({"input_dim": input_dim, "state_dict": model.state_dict()}, model_dir / f"{dataset}_mlp.pt")
    
    # Save Metrics
    metrics = {
        "dataset": dataset,
        "model": "MLP (PyTorch)",
        "profile": PROFILE,
        "ablation_levels": ABLATION_LEVELS,
        "train_rows_per_epoch": train_steps * 1024,
        "val_rows_used": len(y_true_final),
        "train_time_sec": round(total_train_time, 2),
        "inference_time_sec": round(eval_time, 2),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        },
        "hyperparameters": {
            "hidden_layers": [128, 64],
            "dropout": 0.2,
            "optimizer": "AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 1024
        }
    }
    write_json(model_dir / f"{dataset}_mlp_metrics.json", metrics)

def run():
    for ds in DATASETS:
        train_mlp(ds)
        print("-" * 60)

if __name__ == "__main__":
    run()
