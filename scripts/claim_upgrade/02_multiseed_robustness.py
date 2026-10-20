import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, str(Path("C:/Users/umari/Documents/P1_task_Implementation/src")))
from iot_ids.adversarial.attacks import pgd_attack
from iot_ids.adversarial.adversarial_training import train_mlp_adversarial

class MLPModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(21, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x): return self.net(x)

class EnsModule(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    def forward(self, x):
        return torch.mean(torch.stack([m(x) for m in self.models]), dim=0)

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

def get_data(ds):
    stage3 = Path("C:/Users/umari/Documents/P1_task_Implementation/data/processed/stage3") / ds
    tr = pd.read_parquet(stage3 / "train.parquet")
    val = pd.read_parquet(stage3 / "val.parquet")
    te = pd.read_parquet(stage3 / "test.parquet")
    cols = ["flow_duration", "flow_bytes_per_sec", "flow_pkts_per_sec", "mean_pkt_size", "payload_byte_ratio", "pkt_count_ratio", "tcp_syn_ratio", "proto_tcp", "proto_udp", "proto_icmp", "proto_other", "temporal_iat_mean", "temporal_iat_cv", "temporal_flow_rate_ewma", "temporal_byte_rate_ewma", "temporal_syn_rate_ewma", "behavioral_dst_diversity", "behavioral_port_entropy", "behavioral_fanout_ratio", "behavioral_unanswered_ratio", "behavioral_src_activity_ewma"]
    return tr[cols].values, tr["label"].values, val[cols].values, val["label"].values, te[cols].values, te["label"].values

def evaluate_preds(y_true, p_pred):
    return float(roc_auc_score(y_true, p_pred)), float(f1_score(y_true, (p_pred >= 0.5).astype(int), average="macro"))

def attack_and_eval(model, rf, X_att, y_att, c_mask, x_min, x_max):
    adv = pgd_attack(model, X_att, y_att, epsilon=0.10, alpha=0.025, steps=10, continuous_mask=c_mask, x_min=x_min, x_max=x_max).numpy()
    with torch.no_grad():
        p_rob_adv = torch.sigmoid(model(torch.tensor(adv, dtype=torch.float32))).squeeze().numpy()
    p_rf_adv = rf.predict_proba(adv)[:, 1]
    p_opt = 0.7 * p_rf_adv + 0.3 * p_rob_adv
    return float(np.mean(p_opt < 0.5))

def run():
    datasets = ["Edge-IIoTset", "NF-ToN-IoT-v2", "ToN-IoT", "CICIoT2023"]
    seeds = list(range(42, 52))
    
    final_report = {}
    out_dir = Path("C:/Users/umari/Documents/P1_task_Implementation/reports/claim_upgrade")
    
    for ds in datasets:
        t0 = time.time()
        print(f"--- Dataset: {ds} ---")
        X_tr, y_tr, X_val, y_val, X_te, y_te = get_data(ds)
        
        s = RobustScaler()
        X_tr_sc = s.fit_transform(X_tr)
        X_val_sc = s.transform(X_val)
        X_te_sc = s.transform(X_te)
        
        set_seeds(42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_tr_sc, y_tr)
        
        p_rf_val = rf.predict_proba(X_val_sc)[:, 1]
        p_rf_te = rf.predict_proba(X_te_sc)[:, 1]
        
        mask_val = (y_val == 1)
        X_att_val_t = torch.tensor(X_val_sc[mask_val], dtype=torch.float32)
        y_att_val_t = torch.tensor(y_val[mask_val], dtype=torch.float32)
        
        mask_te = (y_te == 1)
        X_att_te_t = torch.tensor(X_te_sc[mask_te], dtype=torch.float32)
        y_att_te_t = torch.tensor(y_te[mask_te], dtype=torch.float32)
        
        x_min = torch.tensor(np.min(X_tr_sc, axis=0), dtype=torch.float32)
        x_max = torch.tensor(np.max(X_tr_sc, axis=0), dtype=torch.float32)
        c_mask = torch.ones(21, dtype=torch.float32)
        c_mask[7:11] = 0.0
        
        models = []
        seed_results = []
        
        for sd in seeds:
            print(f"  Training seed {sd}")
            set_seeds(sd)
            m = MLPModule()
            cols = ["f"]*21
            train_mlp_adversarial(m, X_tr_sc, y_tr, cols, epsilon=0.10, pgd_steps=7, adv_ratio=0.5, epochs=15, batch_size=256, lr=0.001, verbose=False)
            m.eval()
            models.append(m)
            
            with torch.no_grad():
                p_rob_val = torch.sigmoid(m(torch.tensor(X_val_sc, dtype=torch.float32))).squeeze().numpy()
                p_rob_te = torch.sigmoid(m(torch.tensor(X_te_sc, dtype=torch.float32))).squeeze().numpy()
            
            p_opt_val = 0.7 * p_rf_val + 0.3 * p_rob_val
            val_auc, val_f1 = evaluate_preds(y_val, p_opt_val)
            
            set_seeds(42) # fixed attack eval seed
            val_asr = attack_and_eval(m, rf, X_att_val_t, y_att_val_t, c_mask, x_min, x_max)
            score = val_auc - val_asr  # Score = Validation AUC - 1.0 * Validation PGD ASR
            
            p_opt_te = 0.7 * p_rf_te + 0.3 * p_rob_te
            te_auc, te_f1 = evaluate_preds(y_te, p_opt_te)
            
            set_seeds(42)
            te_asr = attack_and_eval(m, rf, X_att_te_t, y_att_te_t, c_mask, x_min, x_max)
            
            seed_results.append({
                "seed": sd,
                "val_auc": val_auc, "val_asr": val_asr, "selection_score": score,
                "te_auc": te_auc, "te_f1": te_f1, "te_asr": te_asr
            })
            
        best_run = max(seed_results, key=lambda x: x["selection_score"])
        print(f"  Best selected seed: {best_run['seed']} with score {best_run['selection_score']:.4f}")
        
        print("  Evaluating ensemble...")
        with torch.no_grad():
            preds = [torch.sigmoid(m(torch.tensor(X_te_sc, dtype=torch.float32))).squeeze().numpy() for m in models]
        p_rob_ens = np.mean(preds, axis=0)
        p_opt_ens = 0.7 * p_rf_te + 0.3 * p_rob_ens
        ens_auc, ens_f1 = evaluate_preds(y_te, p_opt_ens)
        
        ens_m = EnsModule(models)
        ens_m.eval()
        set_seeds(42)
        ens_t0 = time.time()
        ens_asr = attack_and_eval(ens_m, rf, X_att_te_t, y_att_te_t, c_mask, x_min, x_max)
        ens_time = time.time() - ens_t0
        print(f"  Ensemble eval time: {ens_time:.2f}s")
        
        final_report[ds] = {
            "baseline_seeds": seed_results,
            "selected_checkpoint": best_run,
            "ensemble": {
                "te_auc": ens_auc, "te_f1": ens_f1, "te_asr": ens_asr, "overhead_multiplier": 10.0
            }
        }
        print(f"  Dataset time: {time.time() - t0:.2f}s")
    
    with open(out_dir / "02_multiseed_robustness.json", "w") as f:
        json.dump(final_report, f, indent=2)

if __name__ == "__main__":
    run()
