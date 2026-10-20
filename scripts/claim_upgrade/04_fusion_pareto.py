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
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path("C:/Users/umari/Documents/P1_task_Implementation/src")))
from iot_ids.adversarial.attacks import pgd_attack
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

def evaluate_preds(y_true, p_pred):
    auc = float(roc_auc_score(y_true, p_pred))
    pred_bin = (p_pred >= 0.5).astype(int)
    f1 = float(f1_score(y_true, pred_bin, average="macro"))
    tn, fp, fn, tp = confusion_matrix(y_true, pred_bin).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return auc, f1, fpr

def compute_pareto_front(results):
    front = []
    for r1 in results:
        dominated = False
        for r2 in results:
            if r1['w_rf'] == r2['w_rf']:
                continue
            # we want HIGHER auc and LOWER asr
            if r2['te_auc'] >= r1['te_auc'] and r2['te_asr'] <= r1['te_asr']:
                if r2['te_auc'] > r1['te_auc'] or r2['te_asr'] < r1['te_asr']:
                    dominated = True
                    break
        if not dominated:
            front.append(r1['w_rf'])
    return front

def run():
    datasets = ["Edge-IIoTset", "NF-ToN-IoT-v2", "ToN-IoT", "CICIoT2023"]
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    multi_seed_weights = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    seeds = list(range(42, 52))
    
    with open("C:/Users/umari/Documents/P1_task_Implementation/reports/claim_upgrade/02_multiseed_robustness.json", "r") as f:
        p2_data = json.load(f)
        
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
        
        p_rf_te = rf.predict_proba(X_te_sc)[:, 1]
        
        mask_te = (y_te == 1)
        X_att_te_t = torch.tensor(X_te_sc[mask_te], dtype=torch.float32)
        y_att_te_t = torch.tensor(y_te[mask_te], dtype=torch.float32)
        
        x_min = torch.tensor(np.min(X_tr_sc, axis=0), dtype=torch.float32)
        x_max = torch.tensor(np.max(X_tr_sc, axis=0), dtype=torch.float32)
        c_mask = torch.ones(21, dtype=torch.float32)
        c_mask[7:11] = 0.0
        
        models_data = {}
        for sd in seeds:
            print(f"  Training/attacking seed {sd}")
            set_seeds(sd)
            m = MLPModule()
            cols = ["f"]*21
            train_mlp_adversarial(m, X_tr_sc, y_tr, cols, epsilon=0.10, pgd_steps=7, adv_ratio=0.5, epochs=15, batch_size=256, lr=0.001, verbose=False)
            m.eval()
            
            with torch.no_grad():
                p_rob_te = torch.sigmoid(m(torch.tensor(X_te_sc, dtype=torch.float32))).squeeze().numpy()
            
            set_seeds(42) # Attack seed
            adv = pgd_attack(m, X_att_te_t, y_att_te_t, epsilon=0.10, alpha=0.025, steps=10, continuous_mask=c_mask, x_min=x_min, x_max=x_max).numpy()
            with torch.no_grad():
                p_rob_adv = torch.sigmoid(m(torch.tensor(adv, dtype=torch.float32))).squeeze().numpy()
            p_rf_adv = rf.predict_proba(adv)[:, 1]
            
            models_data[sd] = {
                "p_rob_te": p_rob_te,
                "p_rob_adv": p_rob_adv,
                "p_rf_adv": p_rf_adv
            }
            
        selected_seed = p2_data[ds]["selected_checkpoint"]["seed"]
        
        def evaluate_sweep(target_seed):
            data = models_data[target_seed]
            res = []
            
            # Precompute pure RF and pure MLP for relative diffs
            pure_rf_auc, _, _ = evaluate_preds(y_te, p_rf_te)
            pure_rf_asr = float(np.mean(data["p_rf_adv"] < 0.5))
            
            pure_mlp_auc, _, _ = evaluate_preds(y_te, data["p_rob_te"])
            pure_mlp_asr = float(np.mean(data["p_rob_adv"] < 0.5))
            
            for w in weights:
                p_opt_te = w * p_rf_te + (1-w) * data["p_rob_te"]
                auc, f1, fpr = evaluate_preds(y_te, p_opt_te)
                
                p_opt_adv = w * data["p_rf_adv"] + (1-w) * data["p_rob_adv"]
                asr = float(np.mean(p_opt_adv < 0.5))
                
                res.append({
                    "w_rf": w,
                    "te_auc": auc,
                    "te_f1": f1,
                    "te_fpr": fpr,
                    "te_asr": asr,
                    "n_test_samples": len(y_te),
                    "rel_auc_vs_rf": auc - pure_rf_auc,
                    "rel_asr_vs_rf": asr - pure_rf_asr,
                    "rel_asr_vs_mlp": asr - pure_mlp_asr
                })
            
            frontier = compute_pareto_front(res)
            return res, frontier
            
        cond_A_res, cond_A_front = evaluate_sweep(42) # default canonical seed
        cond_B_res, cond_B_front = evaluate_sweep(selected_seed)
        
        multi_seed_results = {}
        for w in multi_seed_weights:
            asrs = []
            for sd in seeds:
                d = models_data[sd]
                p_opt_adv = w * d["p_rf_adv"] + (1-w) * d["p_rob_adv"]
                asrs.append(float(np.mean(p_opt_adv < 0.5)))
            multi_seed_results[w] = {
                "mean": float(np.mean(asrs)),
                "median": float(np.median(asrs)),
                "std": float(np.std(asrs, ddof=1)),
                "raw_asrs": asrs
            }
            
        final_report[ds] = {
            "cond_A_original_seed42": {
                "results": cond_A_res,
                "pareto_frontier": cond_A_front,
                "is_07_pareto": 0.7 in cond_A_front
            },
            "cond_B_selected_seed": {
                "seed": selected_seed,
                "results": cond_B_res,
                "pareto_frontier": cond_B_front,
                "is_07_pareto": 0.7 in cond_B_front
            },
            "multi_seed_robustness": multi_seed_results
        }
    
    with open("C:/Users/umari/Documents/P1_task_Implementation/reports/claim_upgrade/04_fusion_pareto.json", "w") as f:
        json.dump(final_report, f, indent=2)

if __name__ == "__main__":
    run()
