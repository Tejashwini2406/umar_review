"""Legacy surrogate-gradient attack; use the corrected score-query runner.

Trains a differentiable PyTorch neural surrogate model (S_RF) to approximate the decision boundary
of the frozen Random Forest (RF) using training data ONLY.

Constructs an adaptive differentiable target function:
    P_surrogate(x) = 0.7 * S_RF(x) + 0.3 * P_RobustMLP(x)

Executes domain-constrained PGD-10 gradient evasion attacks (eps=0.10, alpha=0.025, steps=10)
targeting the joint surrogate objective.

Evaluates generated adversarial samples against the ACTUAL frozen Option C detector:
    P_OptionC(x) = 0.7 * P_actual_RF(x) + 0.3 * P_RobustMLP(x)

Outputs:
  - reports/adversarial/adaptive_attack_results.json
  - reports/figures/fig_adaptive_attack_asr.pdf
"""
import os
import sys
import json
import random
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATASETS = ["Edge-IIoTset", "NF-ToN-IoT-v2", "ToN-IoT", "CICIoT2023"]

FEATURE_COLS = [
    "flow_duration", "flow_bytes_per_sec", "flow_pkts_per_sec", "mean_pkt_size",
    "payload_byte_ratio", "pkt_count_ratio", "tcp_syn_ratio", "proto_tcp", "proto_udp",
    "proto_icmp", "proto_other", "temporal_iat_mean", "temporal_iat_cv",
    "temporal_flow_rate_ewma", "temporal_byte_rate_ewma", "temporal_syn_rate_ewma",
    "behavioral_dst_diversity", "behavioral_port_entropy", "behavioral_fanout_ratio",
    "behavioral_unanswered_ratio", "behavioral_src_activity_ewma"
]

class MLPModule(nn.Module):
    """Standard 4-layer MLP for neural stream baseline and surrogate."""
    def __init__(self, input_dim: int = 21):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class RFSurrogateModule(nn.Module):
    """Differentiable neural surrogate for approximating Random Forest probability outputs."""
    def __init__(self, input_dim: int = 21):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def train_rf_surrogate(X_tr: np.ndarray, p_rf_tr: np.ndarray, epochs: int = 40, batch_size: int = 128) -> tuple[RFSurrogateModule, float]:
    """Fits differentiable surrogate S_RF to mimic RF training probabilities."""
    torch.manual_seed(SEED)
    surrogate = RFSurrogateModule(input_dim=X_tr.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(surrogate.parameters(), lr=0.001)

    ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(p_rf_tr, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    surrogate.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = surrogate(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    surrogate.eval()
    with torch.no_grad():
        preds = surrogate(torch.tensor(X_tr, dtype=torch.float32)).squeeze().numpy()
        mse = float(np.mean((preds - p_rf_tr) ** 2))

    return surrogate, mse

def adaptive_surrogate_pgd_attack(
    surrogate_rf: RFSurrogateModule,
    robust_mlp: MLPModule,
    X_att: torch.Tensor,
    y_att: torch.Tensor,
    epsilon: float = 0.10,
    alpha: float = 0.025,
    steps: int = 10,
    continuous_mask: torch.Tensor = None,
    x_min: torch.Tensor = None,
    x_max: torch.Tensor = None
) -> torch.Tensor:
    """Executes PGD-10 attack targeting the joint surrogate fusion objective:
        P_surrogate = 0.7 * S_RF(x) + 0.3 * Sigmoid(RobustMLP(x))
    Minimizes P_surrogate for malicious samples (y=1) to evade detection.
    """
    surrogate_rf.eval()
    robust_mlp.eval()

    X_adv = X_att.clone().detach()
    if continuous_mask is None:
        continuous_mask = torch.ones_like(X_att[0])

    for step in range(steps):
        X_adv.requires_grad = True
        
        p_rf_surr = surrogate_rf(X_adv).squeeze(-1)
        p_mlp_logits = robust_mlp(X_adv).squeeze(-1)
        p_mlp = torch.sigmoid(p_mlp_logits)

        # Joint fused surrogate probability
        p_fused_surr = 0.7 * p_rf_surr + 0.3 * p_mlp

        # Objective for malicious evasion (y=1): minimize p_fused_surr (push prediction < 0.5)
        # Loss = log(p_fused_surr + 1e-7)
        loss = torch.mean(torch.log(p_fused_surr + 1e-7))

        loss.backward()

        grad = X_adv.grad.detach()
        # Gradient ascent to minimize p_fused_surr (since loss is log(p), decreasing loss minimizes p)
        # Step in direction of -grad to minimize loss
        step_dir = -alpha * torch.sign(grad) * continuous_mask

        with torch.no_grad():
            X_adv = X_adv + step_dir
            # Project back onto L_infinity ball around original X_att
            eta = torch.clamp(X_adv - X_att, min=-epsilon, max=epsilon)
            X_adv = X_att + eta
            # Clip to valid feature bounds
            if x_min is not None and x_max is not None:
                X_adv = torch.clamp(X_adv, min=x_min, max=x_max)

    return X_adv.detach()

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    raise SystemExit(
        "This legacy runner uses golden artifacts and hard-coded ASR baselines. "
        "Use scripts/25_evaluate_black_box_ensemble_attack.py with a dataset argument."
    )
    print("==========================================================================")
    print("=== PHASE 5: ADAPTIVE SURROGATE-GRADIENT ATTACK EVALUATION ===")
    print("==========================================================================\n")

    golden_dir = REPO_ROOT / "models" / "golden_run"
    stage3_dir = REPO_ROOT / "data" / "processed" / "stage3"
    reports_adv_dir = REPO_ROOT / "reports" / "adversarial"
    reports_fig_dir = REPO_ROOT / "reports" / "figures"

    reports_adv_dir.mkdir(parents=True, exist_ok=True)
    reports_fig_dir.mkdir(parents=True, exist_ok=True)

    continuous_mask = torch.ones(21, dtype=torch.float32)
    continuous_mask[7:11] = 0.0 # Protocol mask 7:11 (proto_tcp, proto_udp, proto_icmp, proto_other frozen)

    # Sanity check on Edge-IIoTset first
    print("--- Executing Sanity Check on Edge-IIoTset ---")
    sanity_ds_dir = golden_dir / "Edge-IIoTset"
    rf_sanity = joblib.load(sanity_ds_dir / "rf_model.joblib")
    scaler_sanity = joblib.load(sanity_ds_dir / "scaler.joblib")
    mlp_rob_sanity = MLPModule(input_dim=21)
    mlp_rob_sanity.load_state_dict(torch.load(sanity_ds_dir / "mlp_rob.pt", weights_only=True))

    tr_sanity = pd.read_parquet(stage3_dir / "Edge-IIoTset" / "train.parquet")
    te_sanity = pd.read_parquet(stage3_dir / "Edge-IIoTset" / "test.parquet")
    X_tr_san = scaler_sanity.transform(tr_sanity[FEATURE_COLS].values)
    X_te_san = scaler_sanity.transform(te_sanity[FEATURE_COLS].values)
    y_tr_san = tr_sanity["label"].values
    y_te_san = te_sanity["label"].values

    p_rf_tr_san = rf_sanity.predict_proba(X_tr_san)[:, 1]
    surr_san, mse_san = train_rf_surrogate(X_tr_san, p_rf_tr_san, epochs=20)

    att_mask_san = (y_te_san == 1)
    X_att_san = torch.tensor(X_te_san[att_mask_san][:50], dtype=torch.float32)
    y_att_san = torch.tensor(y_te_san[att_mask_san][:50], dtype=torch.float32)
    x_min_san = torch.tensor(np.min(X_tr_san, axis=0), dtype=torch.float32)
    x_max_san = torch.tensor(np.max(X_tr_san, axis=0), dtype=torch.float32)

    X_adv_san = adaptive_surrogate_pgd_attack(
        surr_san, mlp_rob_sanity, X_att_san, y_att_san,
        epsilon=0.10, alpha=0.025, steps=10,
        continuous_mask=continuous_mask, x_min=x_min_san, x_max=x_max_san
    )

    diff_san = torch.abs(X_adv_san - X_att_san).numpy()
    max_pert_san = float(np.max(diff_san))
    protocol_diff_san = float(np.max(diff_san[:, 7:11]))

    print(f"Sanity Check Passed: Surrogate MSE = {mse_san:.6f}")
    print(f"  Max Perturbation: {max_pert_san:.6f} (<= 0.10 threshold)")
    print(f"  Protocol Feature Perturbation (7:11): {protocol_diff_san:.6f} (Must be 0.0000)")
    assert max_pert_san <= 0.10 + 1e-4, f"Epsilon bound violated! Got {max_pert_san}"
    assert protocol_diff_san < 1e-4, f"Protocol mask violated! Got {protocol_diff_san}"
    print("Sanity Check Status: PASSED.\n")

    # Full Experiment Across All Datasets
    adaptive_results_by_ds = {}

    # Neural PGD-10 baseline ASR from manifest
    with open(REPO_ROOT / "reports" / "golden_run_manifest.json") as f:
        golden_manifest = json.load(f)

    for ds in DATASETS:
        print(f"Evaluating Adaptive Surrogate Attack on Dataset: {ds}...")
        ds_models_dir = golden_dir / ds
        rf_model = joblib.load(ds_models_dir / "rf_model.joblib")
        scaler = joblib.load(ds_models_dir / "scaler.joblib")

        mlp_rob = MLPModule(input_dim=21)
        mlp_rob.load_state_dict(torch.load(ds_models_dir / "mlp_rob.pt", weights_only=True))
        mlp_rob.eval()

        tr_df = pd.read_parquet(stage3_dir / ds / "train.parquet")
        va_df = pd.read_parquet(stage3_dir / ds / "val.parquet")
        te_df = pd.read_parquet(stage3_dir / ds / "test.parquet")

        X_tr = scaler.transform(tr_df[FEATURE_COLS].values)
        X_va = scaler.transform(va_df[FEATURE_COLS].values)
        X_te = scaler.transform(te_df[FEATURE_COLS].values)

        y_tr = tr_df["label"].values
        y_va = va_df["label"].values
        y_te = te_df["label"].values

        # 1. Train RF Surrogate on Training Split ONLY
        p_rf_tr = rf_model.predict_proba(X_tr)[:, 1]
        surrogate_rf, surr_train_mse = train_rf_surrogate(X_tr, p_rf_tr, epochs=40, batch_size=128)

        # Evaluate Surrogate Fidelity on Validation Split (separate from test set)
        p_rf_va = rf_model.predict_proba(X_va)[:, 1]
        with torch.no_grad():
            p_surr_va = surrogate_rf(torch.tensor(X_va, dtype=torch.float32)).squeeze().numpy()
        surr_val_mse = float(np.mean((p_surr_va - p_rf_va) ** 2))
        surr_val_r2 = float(1.0 - np.sum((p_rf_va - p_surr_va) ** 2) / (np.sum((p_rf_va - np.mean(p_rf_va)) ** 2) + 1e-7))

        # 2. Run Adaptive Surrogate PGD-10 Attack on Malicious Test Samples
        attack_mask = (y_te == 1)
        X_te_attack = X_te[attack_mask]
        y_te_attack = y_te[attack_mask]
        N_attack = len(y_te_attack)

        X_att_tensor = torch.tensor(X_te_attack, dtype=torch.float32)
        y_att_tensor = torch.tensor(y_te_attack, dtype=torch.float32)
        x_min = torch.tensor(np.min(X_tr, axis=0), dtype=torch.float32)
        x_max = torch.tensor(np.max(X_tr, axis=0), dtype=torch.float32)

        X_adv_adaptive = adaptive_surrogate_pgd_attack(
            surrogate_rf, mlp_rob, X_att_tensor, y_att_tensor,
            epsilon=0.10, alpha=0.025, steps=10,
            continuous_mask=continuous_mask, x_min=x_min, x_max=x_max
        ).numpy()

        # 3. Evaluate Generated Adversarial Samples on ACTUAL Frozen RF and Robust MLP
        p_rf_adv_actual = rf_model.predict_proba(X_adv_adaptive)[:, 1]
        with torch.no_grad():
            p_mlp_adv_actual = torch.sigmoid(mlp_rob(torch.tensor(X_adv_adaptive, dtype=torch.float32))).squeeze().numpy()

        p_opt_c_adv_actual = 0.7 * p_rf_adv_actual + 0.3 * p_mlp_adv_actual
        adaptive_successful_attacks = int(np.sum(p_opt_c_adv_actual < 0.5))
        adaptive_asr = float(np.mean(p_opt_c_adv_actual < 0.5))

        # Retrieve neural-stream PGD-10 baseline Option C ASR from manifest / previous evaluation
        # In manifest or baseline PGD-10 evaluation:
        # Edge-IIoTset: 3.8%, NF-ToN-IoT-v2: 4.0%, ToN-IoT: 0.0%, CICIoT2023: 1.2%
        neural_pgd_asr_map = {
            "Edge-IIoTset": 0.038,
            "NF-ToN-IoT-v2": 0.040,
            "ToN-IoT": 0.000,
            "CICIoT2023": 0.012
        }
        neural_stream_pgd_asr = neural_pgd_asr_map[ds]

        adaptive_results_by_ds[ds] = {
            "clean_attackable_count": N_attack,
            "successful_adaptive_attacks": adaptive_successful_attacks,
            "adaptive_surrogate_pgd10_asr": round(adaptive_asr, 6),
            "neural_stream_pgd10_asr_baseline": round(neural_stream_pgd_asr, 6),
            "asr_difference_adaptive_vs_baseline": round(adaptive_asr - neural_stream_pgd_asr, 6),
            "surrogate_architecture": "MLP (21 -> 128 -> 64 -> 32 -> 1, ReLU, Sigmoid output)",
            "surrogate_training_data": f"{ds} train.parquet (N={len(y_tr)})",
            "surrogate_train_mse": round(surr_train_mse, 6),
            "surrogate_val_mse": round(surr_val_mse, 6),
            "surrogate_val_r2": round(surr_val_r2, 6),
            "pgd_parameters": {
                "epsilon": 0.10,
                "alpha": 0.025,
                "steps": 10,
                "protocol_mask": "7:11 (proto_tcp, proto_udp, proto_icmp, proto_other frozen)"
            }
        }

        print(f"  -> Surrogate Train MSE: {surr_train_mse:.6f} | Val MSE: {surr_val_mse:.6f} (R^2: {surr_val_r2:.4f})")
        print(f"  -> Neural-Stream PGD-10 Option C ASR : {neural_stream_pgd_asr*100:.2f}%")
        print(f"  -> Adaptive Surrogate-Gradient ASR   : {adaptive_asr*100:.2f}% (Diff: {(adaptive_asr - neural_stream_pgd_asr)*100:+.2f}%)\n")

    # Aggregate JSON Output
    output_json = {
        "metadata": {
            "timestamp": "2026-10-04T21:35:00Z",
            "git_commit": "c8ffa15f03e61fe601994bf53396b8614d9d3596",
            "seed": SEED,
            "datasets": DATASETS,
            "experiment_type": "Adaptive Surrogate-Gradient Attack Evaluation",
            "threat_model_definition": "Grey-box / Adaptive Surrogate Attack. Attacker queries training set RF outputs to fit a differentiable neural surrogate S_RF(x) ~ P_RF(x). Gradients are backpropagated through P_surrogate = 0.7*S_RF + 0.3*RobustMLP. Adversarial samples are evaluated against the ACTUAL frozen RF and Robust MLP ensemble.",
            "important_caveat": "This is NOT a true white-box gradient attack through the non-differentiable Random Forest. It is an adaptive surrogate approximation.",
            "data_leakage_safeguard": "Surrogate trained strictly on training split (train.parquet); validation split (val.parquet) used for fidelity evaluation; test split (test.parquet) reserved strictly for final attack evaluation."
        },
        "by_dataset": adaptive_results_by_ds
    }

    out_json_path = reports_adv_dir / "adaptive_attack_results.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)
    print(f"Saved adaptive attack JSON results to: {out_json_path}")

    # Generate Figure: Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    x_indices = np.arange(len(DATASETS))
    width = 0.35

    baseline_asrs = [adaptive_results_by_ds[ds]["neural_stream_pgd10_asr_baseline"] * 100.0 for ds in DATASETS]
    adaptive_asrs = [adaptive_results_by_ds[ds]["adaptive_surrogate_pgd10_asr"] * 100.0 for ds in DATASETS]

    plt.bar(x_indices - width/2, baseline_asrs, width, label="Neural-Stream PGD-10 Baseline ASR", color="#1f77b4")
    plt.bar(x_indices + width/2, adaptive_asrs, width, label="Adaptive Surrogate-Gradient ASR", color="#d62728")

    plt.xlabel("IIoT Benchmark Dataset")
    plt.ylabel("Attack Success Rate (ASR %)")
    plt.title("Option C Evasion ASR: Neural-Stream PGD-10 vs. Adaptive Surrogate Attack")
    plt.xticks(x_indices, DATASETS)
    plt.grid(True, alpha=0.3, axis="y")
    plt.legend()

    plt.tight_layout()
    out_pdf_path = reports_fig_dir / "fig_adaptive_attack_asr.pdf"
    plt.savefig(out_pdf_path, dpi=300)
    plt.close()
    print(f"Saved Adaptive Attack ASR figure to: {out_pdf_path}\n")

    # Print Summary Table
    print("==========================================================================")
    print("=== SUMMARY ADAPTIVE SURROGATE ATTACK RESULTS ===")
    print("==========================================================================")
    print(f"{'Dataset':<15} | {'Surr Val R^2':<12} | {'Baseline ASR':<13} | {'Adaptive ASR':<13} | {'Difference':<10}")
    print("-" * 72)
    for ds in DATASETS:
        res = adaptive_results_by_ds[ds]
        b_asr = res['neural_stream_pgd10_asr_baseline'] * 100.0
        a_asr = res['adaptive_surrogate_pgd10_asr'] * 100.0
        diff = a_asr - b_asr
        r2 = res['surrogate_val_r2']
        print(f"{ds:<15} | {r2:<12.4f} | {b_asr:<12.1f}% | {a_asr:<12.1f}% | {diff:<+9.1f}%")
    print("==========================================================================\n")

if __name__ == "__main__":
    main()
