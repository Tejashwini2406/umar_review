import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, str(Path("C:/Users/umari/Documents/P1_task_Implementation/src")))
from iot_ids.adversarial.adversarial_training import train_mlp_adversarial

class MLPModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(21, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x): return self.net(x)

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data(ds):
    stage3 = Path("C:/Users/umari/Documents/P1_task_Implementation/data/processed/stage3") / ds
    tr = pd.read_parquet(stage3 / "train.parquet")
    te = pd.read_parquet(stage3 / "test.parquet")
    cols = ["flow_duration", "flow_bytes_per_sec", "flow_pkts_per_sec", "mean_pkt_size", "payload_byte_ratio", "pkt_count_ratio", "tcp_syn_ratio", "proto_tcp", "proto_udp", "proto_icmp", "proto_other", "temporal_iat_mean", "temporal_iat_cv", "temporal_flow_rate_ewma", "temporal_byte_rate_ewma", "temporal_syn_rate_ewma", "behavioral_dst_diversity", "behavioral_port_entropy", "behavioral_fanout_ratio", "behavioral_unanswered_ratio", "behavioral_src_activity_ewma"]
    return tr[cols].values, tr["label"].values, te[cols].values, te["label"].values

def nes_gradient_estimator(func, x, num_samples, sigma, c_mask):
    half_samples = num_samples // 2
    noise = np.random.randn(half_samples, x.shape[0]) * c_mask
    
    x_plus = x + sigma * noise
    x_minus = x - sigma * noise
    
    eval_pts = np.vstack([x_plus, x_minus])
    scores = func(eval_pts)
    
    scores_plus = scores[:half_samples]
    scores_minus = scores[half_samples:]
    
    grad = np.mean(((scores_plus - scores_minus) / (2 * sigma))[:, None] * noise, axis=0)
    return grad

def blackbox_nes_attack(model_func, X_att, max_queries, epsilon, alpha, c_mask, x_min, x_max, sigma=0.01, samples_per_iter=20):
    adv = X_att.copy()
    num_pts = adv.shape[0]
    
    success = np.zeros(num_pts, dtype=bool)
    queries_used = np.zeros(num_pts, dtype=int)
    
    # 1. Initial query
    p_init = model_func(adv)
    queries_used += 1
    
    success = (p_init < 0.5)
    active = ~success
    
    max_iters = max_queries // samples_per_iter
    
    for i in range(max_iters):
        if not np.any(active):
            break
            
        active_idx = np.where(active)[0]
        
        for idx in active_idx:
            if queries_used[idx] + samples_per_iter + 1 > max_queries:
                active[idx] = False
                continue
                
            x_curr = adv[idx]
            
            grad = nes_gradient_estimator(model_func, x_curr, samples_per_iter, sigma, c_mask)
            queries_used[idx] += samples_per_iter
            
            # gradient descent: we want to minimize probability
            x_new = x_curr - alpha * np.sign(grad)
            
            x_new = np.clip(x_new, X_att[idx] - epsilon, X_att[idx] + epsilon)
            x_new = x_curr * (1 - c_mask) + x_new * c_mask
            x_new = np.clip(x_new, x_min, x_max)
            
            adv[idx] = x_new
            
            p_new = model_func(np.array([adv[idx]]))[0]
            queries_used[idx] += 1
            
            if p_new < 0.5:
                success[idx] = True
                active[idx] = False
                
    return success, queries_used

def run():
    raise SystemExit(
        "This legacy runner uses another user's absolute paths and retrains on legacy splits. "
        "Use scripts/25_evaluate_black_box_ensemble_attack.py from the repository root."
    )
    datasets = ["Edge-IIoTset", "NF-ToN-IoT-v2", "ToN-IoT", "CICIoT2023"]
    budgets = [100, 250, 500]
    
    final_report = {}
    
    for ds in datasets:
        print(f"--- Dataset: {ds} ---")
        X_tr, y_tr, X_te, y_te = get_data(ds)
        
        s = RobustScaler()
        X_tr_sc = s.fit_transform(X_tr)
        X_te_sc = s.transform(X_te)
        
        set_seeds(42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_tr_sc, y_tr)
        
        m = MLPModule()
        cols = ["f"]*21
        train_mlp_adversarial(m, X_tr_sc, y_tr, cols, epsilon=0.10, pgd_steps=7, adv_ratio=0.5, epochs=15, batch_size=256, lr=0.001, verbose=False)
        m.eval()
        
        def option_c_predictor(X):
            with torch.no_grad():
                p_mlp = torch.sigmoid(m(torch.tensor(X, dtype=torch.float32))).squeeze().numpy()
            if p_mlp.ndim == 0:
                p_mlp = np.array([p_mlp])
            p_rf = rf.predict_proba(X)[:, 1]
            return 0.7 * p_rf + 0.3 * p_mlp
            
        mask_te = (y_te == 1)
        X_att_te = X_te_sc[mask_te]
        
        x_min = np.min(X_tr_sc, axis=0)
        x_max = np.max(X_tr_sc, axis=0)
        c_mask = np.ones(21)
        c_mask[7:11] = 0.0
        
        ds_results = {}
        for b in budgets:
            print(f"  Running budget {b}")
            set_seeds(42)
            t0 = time.time()
            success, queries = blackbox_nes_attack(
                option_c_predictor, X_att_te, max_queries=b, 
                epsilon=0.10, alpha=0.025, c_mask=c_mask, x_min=x_min, x_max=x_max
            )
            elapsed = time.time() - t0
            
            num_attacked = len(success)
            num_success = int(np.sum(success))
            num_fail = num_attacked - num_success
            asr = num_success / num_attacked if num_attacked > 0 else 0
            
            ds_results[f"budget_{b}"] = {
                "budget": b,
                "attacked_samples": num_attacked,
                "successful_samples": num_success,
                "failed_samples": num_fail,
                "asr": asr,
                "mean_queries": float(np.mean(queries)),
                "median_queries": float(np.median(queries)),
                "max_queries": int(np.max(queries)),
                "time_sec": elapsed
            }
            print(f"    ASR: {asr*100:.2f}%, Mean Q: {float(np.mean(queries)):.1f}")
            
        final_report[ds] = ds_results
        
    with open("C:/Users/umari/Documents/P1_task_Implementation/reports/claim_upgrade/05_blackbox_attack.json", "w") as f:
        json.dump(final_report, f, indent=2)

if __name__ == "__main__":
    run()
