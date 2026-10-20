"""Milestone 5 — Ensemble and Cross-Domain Evaluation.

Optimizes ensemble weights and decision thresholds using the SOURCE validation set
via transparent weighted fusion. Applies frozen weights and thresholds to evaluate
compatible cross-domain targets. Also performs required ablations.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
import torch

import os
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(ROOT))
DATASETS = ["CICIDS2017", "Edge-IIoTset", "BoT-IoT", "N-BaIoT"]
if os.environ.get("SMOKE_TEST") == "1":
    DATASETS = [os.environ.get("SMOKE_DATASET", "N-BaIoT")]

# Training profile configuration
PROFILE = os.environ.get("PROFILE", "in_domain")
_ablation = os.environ.get("ABLATION_LEVELS")
ABLATION_LEVELS = _ablation.split(",") if _ablation else None

def log(msg: str) -> None:
    print(f"[5 EVAL] {time.strftime('%H:%M:%S')} {msg}", flush=True)

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

def get_parquet_files(dataset: str, split_manifest: dict) -> list[Path]:
    cache_dir = ROOT / "data" / "processed" / "research" / dataset / split_manifest["fingerprint"]
    return sorted(cache_dir.glob("part-*.parquet"))

def get_drop_indices(dataset: str) -> dict[str, dict[str, list[int]]]:
    drop_path = ROOT / "data" / "processed" / "splits" / "split-v1" / dataset / "decontamination_drop_indices.json"
    if not drop_path.exists():
        return {}
    return read_json(drop_path)

def load_split(dataset: str, split_name: str) -> tuple[pd.DataFrame, pd.Series]:
    manifest_path = ROOT / "data" / "processed" / "splits" / "split-v1" / dataset / "split_manifest.json"
    manifest = read_json(manifest_path)
    drop_indices = get_drop_indices(dataset)
    parquet_files = get_parquet_files(dataset, manifest)
    split_drops = drop_indices.get(split_name, {})
    from run_milestone3a import split_assigner, hash_to_split
    
    frames = []
    MAX_ROWS = 1_000_000
    total_rows = manifest["splits"][split_name]["rows"]
    fraction = MAX_ROWS / total_rows if total_rows > MAX_ROWS else 1.0
    np.random.seed(42)  # Deterministic subsampling for reproducibility
    
    for file_idx, parquet in enumerate(parquet_files):
        if dataset == "BoT-IoT":
            from run_milestone3a import BOT_TRAIN_END, BOT_VAL_END
            if split_name == "train" and file_idx >= BOT_TRAIN_END: continue
            if split_name == "validation" and (file_idx < BOT_TRAIN_END or file_idx >= BOT_VAL_END): continue
            if split_name == "test" and file_idx < BOT_VAL_END: continue
            
        df = pd.read_parquet(parquet)
        col_to_idx = {c: i for i, c in enumerate(df.columns)}
        hash_indices = [col_to_idx[c] for c in df.columns if c != "source_file"]
        file_drop_set = set(split_drops.get(str(file_idx), []))
        keep_mask = np.zeros(len(df), dtype=bool)
        
        if dataset == "BoT-IoT":
            keep_mask[:] = True
            if file_drop_set:
                for idx in file_drop_set:
                    keep_mask[idx] = False
                    
            num_valid = keep_mask.sum()
            if num_valid > 0 and fraction < 1.0:
                rand_vals = np.random.rand(num_valid)
                keep_valid = rand_vals < fraction
                keep_mask[keep_mask] = keep_valid
        elif dataset == "N-BaIoT":
            from run_milestone3a import NBAIOT_DEVICE_SPLIT
            device_col = df["device_id"].astype(str)
            splits = device_col.map(NBAIOT_DEVICE_SPLIT)
            keep_mask = np.array((splits == split_name).values, dtype=bool)
            if file_drop_set:
                for idx in file_drop_set:
                    keep_mask[idx] = False
            
            num_valid = keep_mask.sum()
            if num_valid > 0 and fraction < 1.0:
                rand_vals = np.random.rand(num_valid)
                keep_valid = rand_vals < fraction
                keep_mask[keep_mask] = keep_valid
        else:
            for row_idx, row in enumerate(df.itertuples(index=False, name=None)):
                if dataset == "Edge-IIoTset":
                    text = "\x1f".join("" if pd.isna(row[i]) else str(row[i]) for i in hash_indices)
                    import hashlib
                    split = hash_to_split(hashlib.sha256(text.encode("utf-8")).hexdigest())
                else:
                    split = split_assigner(dataset, row, file_idx, parquet.name, col_to_idx)
                    
                if split == split_name and row_idx not in file_drop_set:
                    if fraction < 1.0:
                        if np.random.rand() < fraction:
                            keep_mask[row_idx] = True
                    else:
                        keep_mask[row_idx] = True
                        
        if keep_mask.any():
            frames.append(df[keep_mask])
            
    if not frames:
        return np.array([]), np.array([])
        
    full_df = pd.concat(frames, ignore_index=True)
    from src.iot_ids.preprocessing.pipeline import FittedPreprocessor
    preprocessor = FittedPreprocessor.load(ROOT / "models" / PROFILE / "preprocessing" / f"{dataset}_preprocessor.joblib")
    
    X_np = preprocessor.transform(full_df)
    y = (full_df["canonical_label"] != "BENIGN").astype(int).values
    
    return X_np, y

def get_component_predictions(models: dict, metrics: dict, X: np.ndarray) -> dict[str, np.ndarray]:
    """Returns a dict of 1D numpy arrays with probabilities/scores."""
    preds = {}
    
    if "rf" in models:
        log("    [EVAL] Running RF inference...")
        preds["rf"] = np.concatenate([models["rf"].predict_proba(X[i:i+50000])[:, 1] for i in range(0, len(X), 50000)])
        
    if "mlp" in models:
        log("    [EVAL] Running MLP inference...")
        from train_mlp import MLPModel
        import torch
        model_dict = models["mlp"]
        mlp_net = MLPModel(input_dim=model_dict["input_dim"])
        mlp_net.load_state_dict(model_dict["state_dict"])
        mlp_net.eval()
        
        mlp_preds = []
        with torch.no_grad():
            for i in range(0, len(X), 50000):
                X_tensor = torch.tensor(X[i:i+50000], dtype=torch.float32)
                outputs = mlp_net(X_tensor)
                mlp_preds.append(torch.sigmoid(outputs).squeeze().numpy())
        preds["mlp"] = np.concatenate(mlp_preds)
            
    if "ae" in models:
        log("    [EVAL] Running AE inference...")
        threshold = metrics["ae"]["best_threshold"]
        # Guard: if threshold is infinite or NaN, the AE is undertrained
        # and cannot produce meaningful anomaly scores. Exclude it.
        if not np.isfinite(threshold):
            log(f"  [WARN] AE threshold is {threshold} — excluding AE from ensemble.")
        else:
            ae_pred = np.concatenate([models["ae"].predict(X[i:i+50000]) for i in range(0, len(X), 50000)])
            mse = np.mean(np.power(X - ae_pred, 2), axis=1)
            # Normalize strictly to [0,1] to mix with probabilities cleanly:
            # Logistic sigmoid centered at threshold:
            preds["ae"] = 1 / (1 + np.exp(-(mse - threshold) / (threshold + 1e-6)))
        
    return preds

def optimize_weights_and_threshold(preds_dict: dict, y_true: np.ndarray, combinations: list[tuple]) -> tuple[dict, float, float]:
    """Grid search over valid weight simplex and thresholds to maximize F1 on validation."""
    best_f1 = -1
    best_weights = {}
    best_thresh = 0.5
    
    thresholds = np.linspace(0.1, 0.9, 17)
    
    for w_tuple in combinations:
        # Generate weighted probability
        fused_proba = np.zeros(len(y_true))
        weights_dict = {}
        for (model_name, weight) in w_tuple:
            fused_proba += weight * preds_dict[model_name]
            weights_dict[model_name] = weight
            
        for thresh in thresholds:
            y_pred = (fused_proba >= thresh).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_weights = weights_dict
                best_thresh = thresh
                
    return best_weights, best_thresh, best_f1

def get_simplex_combinations(models_to_include: list[str]) -> list[tuple]:
    """Generates weight combinations sum(weights) == 1."""
    step = 0.05
    combinations = []
    
    if len(models_to_include) == 1:
        return [[(models_to_include[0], 1.0)]]
        
    if len(models_to_include) == 2:
        for w1 in np.arange(0.0, 1.01, step):
            w2 = 1.0 - w1
            combinations.append([(models_to_include[0], w1), (models_to_include[1], w2)])
            
    if len(models_to_include) == 3:
        for w1 in np.arange(0.0, 1.01, step):
            for w2 in np.arange(0.0, 1.01 - w1 + 1e-6, step):
                w3 = 1.0 - w1 - w2
                combinations.append([(models_to_include[0], w1), (models_to_include[1], w2), (models_to_include[2], w3)])
                
    return combinations

def run():
    cross_config = read_json(ROOT / "configs" / "experiments" / "cross_dataset_compatibility.json")
    results = {}
    
    loaded_models = {}
    loaded_metrics = {}
    test_sets = {}
    
    ablations = [
        ["rf"],
        ["mlp"],
        ["ae"],
        ["rf", "mlp"],
        ["rf", "mlp", "ae"]
    ]
    
    source_configs = {}
    
    # 1. Optimize Weights & Thresholds per source dataset on Validation Set
    for ds in DATASETS:
        log(f"[{ds}] Loading models...")
        loaded_models[ds] = {}
        loaded_metrics[ds] = {}
        
        rf_path = ROOT / "models" / PROFILE / "random_forest" / f"{ds}_rf.joblib"
        rf_met = ROOT / "models" / PROFILE / "random_forest" / f"{ds}_rf_metrics.json"
        if rf_path.exists():
            loaded_models[ds]["rf"] = joblib.load(rf_path)
            loaded_metrics[ds]["rf"] = read_json(rf_met)
            
        mlp_path = ROOT / "models" / PROFILE / "neural_network" / f"{ds}_mlp.pt"
        mlp_met = ROOT / "models" / PROFILE / "neural_network" / f"{ds}_mlp_metrics.json"
        if mlp_path.exists():
            loaded_models[ds]["mlp"] = torch.load(mlp_path)
            loaded_metrics[ds]["mlp"] = read_json(mlp_met)
            
        ae_path = ROOT / "models" / PROFILE / "autoencoder" / f"{ds}_ae.joblib"
        ae_met = ROOT / "models" / PROFILE / "autoencoder" / f"{ds}_ae_metrics.json"
        if ae_path.exists():
            loaded_models[ds]["ae"] = joblib.load(ae_path)
            loaded_metrics[ds]["ae"] = read_json(ae_met)
            
        if not loaded_models[ds]:
            continue
            
        log(f"[{ds}] Optimizing transparent fusions on Validation set...")
        X_val, y_val = load_split(ds, "validation")
        
        source_configs[ds] = {}
        
        if len(y_val) > 0:
            val_preds = get_component_predictions(loaded_models[ds], loaded_metrics[ds], X_val)
            
            for ab in ablations:
                valid_models = [m for m in ab if m in val_preds]
                if len(valid_models) != len(ab):
                    continue
                    
                ab_name = "+".join(valid_models)
                combos = get_simplex_combinations(valid_models)
                best_w, best_t, val_f1 = optimize_weights_and_threshold(val_preds, y_val, combos)
                
                source_configs[ds][ab_name] = {
                    "weights": best_w,
                    "threshold": best_t,
                    "validation_f1": val_f1
                }
                log(f"  [{ds}] {ab_name} -> F1: {val_f1:.4f} | Weights: {best_w} | Thresh: {best_t:.2f}")
                
        log(f"[{ds}] Loading Test set...")
        test_sets[ds] = load_split(ds, "test")

    # 2. Evaluate Matrix using Frozen Weights & Thresholds
    for cell_key, cell_info in cross_config["cells"].items():
        if cell_info["status"] != "compatible":
            continue
            
        train_ds, test_ds = cell_key.split("->")
        
        # If in-domain profile, strictly enforce same-dataset evaluation
        if PROFILE == "in_domain" and train_ds != test_ds:
            continue
            
        log(f"Evaluating {train_ds} -> {test_ds}")
        
        if test_ds not in test_sets or train_ds not in source_configs:
            log(f"  Skipping: missing test set or models.")
            continue
            
        X_test, y_test = test_sets[test_ds]
        if len(y_test) == 0:
            continue
            
        test_preds = get_component_predictions(loaded_models[train_ds], loaded_metrics[train_ds], X_test)
        
        if train_ds not in results:
            results[train_ds] = {}
            
        results[train_ds][test_ds] = {}
        
        for ab_name, config in source_configs[train_ds].items():
            fused_proba = np.zeros(len(y_test))
            for m_name, weight in config["weights"].items():
                fused_proba += weight * test_preds[m_name]
                
            y_pred = (fused_proba >= config["threshold"]).astype(int)
            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
            
            results[train_ds][test_ds][ab_name] = {
                "accuracy": round(acc, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "frozen_weights": config["weights"],
                "frozen_threshold": config["threshold"]
            }
            if ab_name == "rf+mlp+ae":
                log(f"  {train_ds}->{test_ds} [{ab_name}] | Acc: {acc:.4f} | F1: {f1:.4f}")
        
    eval_file = ROOT / "reports" / "experiments" / f"{PROFILE}_evaluation.json"
    write_json(eval_file, results)
    log(f"Evaluation complete. Results saved to {eval_file.name}")

if __name__ == "__main__":
    run()
