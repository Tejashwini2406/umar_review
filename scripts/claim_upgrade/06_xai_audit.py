import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

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
    cols = ["flow_duration", "flow_bytes_per_sec", "flow_pkts_per_sec", "mean_pkt_size", "payload_byte_ratio", "pkt_count_ratio", "tcp_syn_ratio", "proto_tcp", "proto_udp", "proto_icmp", "proto_other", "temporal_iat_mean", "temporal_iat_cv", "temporal_flow_rate_ewma", "temporal_byte_rate_ewma", "temporal_syn_rate_ewma", "behavioral_dst_diversity", "behavioral_port_entropy", "behavioral_fanout_ratio", "behavioral_unanswered_ratio", "behavioral_src_activity_ewma"]
    return tr[cols].values, tr["label"].values, cols

def run():
    ds = "Edge-IIoTset"
    X_tr, y_tr, feature_names = get_data(ds)
    
    s = RobustScaler()
    X_tr_sc = s.fit_transform(X_tr)
    
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
        if p_mlp.ndim == 0: p_mlp = np.array([p_mlp])
        p_rf = rf.predict_proba(X)[:, 1]
        return 0.7 * p_rf + 0.3 * p_mlp
        
    def mlp_predictor(X):
        with torch.no_grad():
            p_mlp = torch.sigmoid(m(torch.tensor(X, dtype=torch.float32))).squeeze().numpy()
        if p_mlp.ndim == 0: p_mlp = np.array([p_mlp])
        return p_mlp
        
    # 1. Background set
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_tr_sc), size=100, replace=False)
    bg_data = X_tr_sc[bg_idx]
    
    # 2. Sample set to explain
    explain_idx = rng.choice(len(X_tr_sc), size=10, replace=False)
    X_explain = X_tr_sc[explain_idx]
    
    # 3. Component-wise exact/approximate SHAP
    tree_explainer = shap.TreeExplainer(rf)
    sv_rf = tree_explainer.shap_values(X_explain)
    if isinstance(sv_rf, list) and len(sv_rf) == 2: sv_rf = sv_rf[1]
    elif sv_rf.ndim == 3: sv_rf = sv_rf[:, :, 1]
    
    bg_tensor = torch.tensor(bg_data, dtype=torch.float32)
    X_exp_tensor = torch.tensor(X_explain, dtype=torch.float32)
    grad_explainer = shap.GradientExplainer(m, bg_tensor)
    sv_mlp = grad_explainer.shap_values(X_exp_tensor)
    if isinstance(sv_mlp, list): sv_mlp = sv_mlp[0]
    if sv_mlp.ndim == 3: sv_mlp = sv_mlp[:, :, 0]
    
    # Base values
    rf_expected = tree_explainer.expected_value
    if isinstance(rf_expected, np.ndarray): rf_expected = rf_expected[1]
    
    # GradientExplainer doesn't always have expected_value, estimate it:
    mlp_expected = float(np.mean(mlp_predictor(bg_data)))
    
    linear_shap = 0.7 * sv_rf + 0.3 * sv_mlp
    linear_expected = 0.7 * rf_expected + 0.3 * mlp_expected
    
    # 4. Benchmarking true exact SHAP
    t0 = time.time()
    exact_samples = 2**21 # 2,097,152
    print(f"Benchmarking computational cost of evaluating exact SHAP enumerations ({exact_samples} queries)...")
    dummy_input = np.random.randn(exact_samples, 21).astype(np.float32)
    batch_size = 100000
    for i in range(0, exact_samples, batch_size):
        _ = option_c_predictor(dummy_input[i:i+batch_size])
    cost_per_sample = time.time() - t0
    print(f"Time to evaluate {exact_samples} coalitions (1 sample): {cost_per_sample:.2f} seconds")
    
    # 5. Fused Direct Wrapper KernelSHAP
    fused_explainer = shap.KernelExplainer(option_c_predictor, bg_data)
    
    t_start = time.time()
    fused_sv = fused_explainer.shap_values(X_explain, nsamples=10000)
    t_fused = time.time() - t_start
    fused_expected = fused_explainer.expected_value
    
    # Comparisons
    per_feature_abs_error = np.abs(linear_shap - fused_sv)
    mae = np.mean(per_feature_abs_error)
    max_error = np.max(per_feature_abs_error)
    
    spearman_rhos = []
    for i in range(len(X_explain)):
        rho, _ = spearmanr(linear_shap[i], fused_sv[i])
        spearman_rhos.append(rho)
    
    actual_outputs = option_c_predictor(X_explain)
    linear_additivity = (linear_shap.sum(axis=1) + linear_expected) - actual_outputs
    fused_additivity = (fused_sv.sum(axis=1) + fused_expected) - actual_outputs
    
    linear_additivity_mae = np.mean(np.abs(linear_additivity))
    fused_additivity_mae = np.mean(np.abs(fused_additivity))
    
    report = {
        "xai_audit_info": {
            "attribution_game": "Global SHAP feature importance approximation (Linear combination of RF TreeExplainer and MLP GradientExplainer)",
            "output_space": "Prediction probabilities",
            "background_dataset": "100 samples from Stage3 train split",
            "explainer": "TreeExplainer for RF, GradientExplainer for MLP, KernelExplainer for Fused",
            "normalization": "None for local SHAP; Global SHAP currently uses normalized L1 mean absolute SHAP",
            "feature_ordering": "Canonical 21 features",
            "coalitions_for_exact_shap": 2**21
        },
        "benchmarks": {
            "time_to_evaluate_exact_coalitions_per_sample": cost_per_sample,
            "kernel_shap_10000_samples_time": t_fused
        },
        "comparison_metrics": {
            "mae_linear_vs_fused": float(mae),
            "max_error_linear_vs_fused": float(max_error),
            "spearman_rho_mean": float(np.mean(spearman_rhos)),
            "spearman_rho_min": float(np.min(spearman_rhos)),
            "spearman_rho_max": float(np.max(spearman_rhos)),
            "linear_additivity_residual_mae": float(linear_additivity_mae),
            "fused_additivity_residual_mae": float(fused_additivity_mae)
        },
        "sample_level_data": []
    }
    
    for i in range(len(X_explain)):
        report["sample_level_data"].append({
            "sample_index": i,
            "actual_output": float(actual_outputs[i]),
            "linear_expected_value": float(linear_expected),
            "fused_expected_value": float(fused_expected),
            "linear_shap": [float(x) for x in linear_shap[i]],
            "fused_shap": [float(x) for x in fused_sv[i]],
            "spearman_rho": float(spearman_rhos[i]),
            "linear_additivity_residual": float(linear_additivity[i]),
            "fused_additivity_residual": float(fused_additivity[i])
        })
        
    with open("C:/Users/umari/Documents/P1_task_Implementation/reports/claim_upgrade/06_xai_audit.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    run()
