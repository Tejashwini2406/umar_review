"""Phase 2: Empirical Fusion-Weight Ablation Study.

Evaluates the dual-stream probability fusion equation:
    P_w = w * P_RF + (1 - w) * P_RobustMLP
across w in [0.0, 0.1, ..., 1.0] for all four benchmark datasets:
  - Edge-IIoTset
  - NF-ToN-IoT-v2
  - ToN-IoT
  - CICIoT2023

Calculates Clean ROC-AUC, Clean Macro F1, and PGD-10 ASR without modifying frozen models,
dataset splits, scalers, or existing benchmark outputs.

Generates:
  - reports/tables/fusion_ablation_results.json
  - reports/figures/fig_fusion_ablation_pareto.pdf
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

from iot_ids.adversarial.attacks import pgd_attack

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATASETS = ["Edge-IIoTset", "NF-ToN-IoT-v2", "ToN-IoT", "CICIoT2023"]
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

FEATURE_COLS = [
    "flow_duration", "flow_bytes_per_sec", "flow_pkts_per_sec", "mean_pkt_size",
    "payload_byte_ratio", "pkt_count_ratio", "tcp_syn_ratio", "proto_tcp", "proto_udp",
    "proto_icmp", "proto_other", "temporal_iat_mean", "temporal_iat_cv",
    "temporal_flow_rate_ewma", "temporal_byte_rate_ewma", "temporal_syn_rate_ewma",
    "behavioral_dst_diversity", "behavioral_port_entropy", "behavioral_fanout_ratio",
    "behavioral_unanswered_ratio", "behavioral_src_activity_ewma"
]

class MLPModule(nn.Module):
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

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("==========================================================================")
    print("=== PHASE 2: EMPIRICAL FUSION-WEIGHT ABLATION STUDY ===")
    print("==========================================================================\n")

    golden_dir = REPO_ROOT / "models" / "golden_run"
    stage3_dir = REPO_ROOT / "data" / "processed" / "stage3"
    reports_tables_dir = REPO_ROOT / "reports" / "tables"
    reports_fig_dir = REPO_ROOT / "reports" / "figures"

    reports_tables_dir.mkdir(parents=True, exist_ok=True)
    reports_fig_dir.mkdir(parents=True, exist_ok=True)

    continuous_mask = torch.ones(21, dtype=torch.float32)
    continuous_mask[7:11] = 0.0

    ablation_by_dataset = {}
    checkpoint_hashes = {}

    for ds in DATASETS:
        print(f"Auditing Dataset: {ds}...")
        ds_models_dir = golden_dir / ds
        rf_path = ds_models_dir / "rf_model.joblib"
        mlp_path = ds_models_dir / "mlp_rob.pt"
        scaler_path = ds_models_dir / "scaler.joblib"

        checkpoint_hashes[ds] = {
            "rf_model.joblib": file_sha256(rf_path),
            "mlp_rob.pt": file_sha256(mlp_path),
            "scaler.joblib": file_sha256(scaler_path),
        }

        rf_model = joblib.load(rf_path)
        scaler = joblib.load(scaler_path)

        mlp_rob = MLPModule(input_dim=21)
        mlp_rob.load_state_dict(torch.load(mlp_path, weights_only=True))
        mlp_rob.eval()

        tr_df = pd.read_parquet(stage3_dir / ds / "train.parquet")
        te_df = pd.read_parquet(stage3_dir / ds / "test.parquet")

        X_tr = scaler.transform(tr_df[FEATURE_COLS].values)
        X_te = scaler.transform(te_df[FEATURE_COLS].values)
        y_te = te_df["label"].values

        # Clean Probabilities
        p_rf_clean = rf_model.predict_proba(X_te)[:, 1]
        with torch.no_grad():
            p_mlp_clean = torch.sigmoid(mlp_rob(torch.tensor(X_te, dtype=torch.float32))).squeeze().numpy()

        # Adversarial Probabilities (PGD-10 attack on malicious test samples)
        attack_mask = (y_te == 1)
        X_te_attack = X_te[attack_mask]
        y_te_attack = y_te[attack_mask]

        X_att_tensor = torch.tensor(X_te_attack, dtype=torch.float32)
        y_att_tensor = torch.tensor(y_te_attack, dtype=torch.float32)
        x_min = torch.tensor(np.min(X_tr, axis=0), dtype=torch.float32)
        x_max = torch.tensor(np.max(X_tr, axis=0), dtype=torch.float32)

        X_adv_rob = pgd_attack(
            mlp_rob, X_att_tensor, y_att_tensor,
            epsilon=0.10, alpha=0.025, steps=10,
            continuous_mask=continuous_mask, x_min=x_min, x_max=x_max
        ).numpy()

        p_rf_adv = rf_model.predict_proba(X_adv_rob)[:, 1]
        with torch.no_grad():
            p_mlp_adv = torch.sigmoid(mlp_rob(torch.tensor(X_adv_rob, dtype=torch.float32))).squeeze().numpy()

        results_list = []
        for w in WEIGHTS:
            p_fusion_clean = w * p_rf_clean + (1.0 - w) * p_mlp_clean
            clean_auc = float(roc_auc_score(y_te, p_fusion_clean))
            clean_f1 = float(f1_score(y_te, (p_fusion_clean >= 0.5).astype(int), average="macro"))

            p_fusion_adv = w * p_rf_adv + (1.0 - w) * p_mlp_adv
            pgd_asr = float(np.mean(p_fusion_adv < 0.5))

            results_list.append({
                "weight_rf": round(w, 2),
                "weight_mlp": round(1.0 - w, 2),
                "clean_roc_auc": round(clean_auc, 6),
                "clean_macro_f1": round(clean_f1, 6),
                "pgd10_asr": round(pgd_asr, 6),
            })

        ablation_by_dataset[ds] = results_list

    # Calculate Mean Performance across Datasets per Weight
    mean_by_weight = []
    for idx, w in enumerate(WEIGHTS):
        clean_aucs = [ablation_by_dataset[ds][idx]["clean_roc_auc"] for ds in DATASETS]
        clean_f1s = [ablation_by_dataset[ds][idx]["clean_macro_f1"] for ds in DATASETS]
        pgd_asrs = [ablation_by_dataset[ds][idx]["pgd10_asr"] for ds in DATASETS]

        mean_by_weight.append({
            "weight_rf": round(w, 2),
            "weight_mlp": round(1.0 - w, 2),
            "mean_clean_roc_auc": round(float(np.mean(clean_aucs)), 6),
            "mean_clean_macro_f1": round(float(np.mean(clean_f1s)), 6),
            "mean_pgd10_asr": round(float(np.mean(pgd_asrs)), 6),
        })

    output_json = {
        "metadata": {
            "timestamp": "2026-10-04T21:05:00Z",
            "git_commit": "c8ffa15f03e61fe601994bf53396b8614d9d3596",
            "seed": SEED,
            "datasets": DATASETS,
            "weights_evaluated": WEIGHTS,
            "checkpoint_hashes": checkpoint_hashes,
            "weight_selection_policy": "Predetermined 0.7/0.3 Option C baseline evaluated empirically against full grid",
            "metric_definitions": {
                "clean_roc_auc": "Receiver Operating Characteristic AUC on held-out clean test split",
                "clean_macro_f1": "Macro-averaged F1 score at decision threshold 0.5 on clean test split",
                "pgd10_asr": "Attack Success Rate (fraction of malicious test flows misclassified as benign under PGD-10 eps=0.10, alpha=0.025 attack on neural stream)"
            },
            "adversarial_sample_generation": "Regenerated deterministically using seed=42 and frozen PGD-10 pipeline parameters"
        },
        "by_dataset": ablation_by_dataset,
        "mean_across_datasets": mean_by_weight
    }

    out_json_path = reports_tables_dir / "fusion_ablation_results.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)
    print(f"\nSaved ablation JSON results to: {out_json_path}")

    # Generate Figure: Pareto Trade-off Plot
    plt.figure(figsize=(12, 5))

    # Subplot 1: Clean ROC-AUC vs Weight
    plt.subplot(1, 2, 1)
    for ds in DATASETS:
        aucs = [res["clean_roc_auc"] for res in ablation_by_dataset[ds]]
        plt.plot(WEIGHTS, aucs, marker="o", label=ds)
    mean_aucs = [res["mean_clean_roc_auc"] for res in mean_by_weight]
    plt.plot(WEIGHTS, mean_aucs, marker="s", color="black", linewidth=2.5, linestyle="--", label="Mean Across Datasets")
    plt.axvline(x=0.7, color="red", linestyle=":", label="Option C (w=0.7)")
    plt.xlabel("RF Fusion Weight (w)")
    plt.ylabel("Clean ROC-AUC")
    plt.title("Clean ROC-AUC vs. RF Fusion Weight")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    # Subplot 2: PGD-10 ASR vs Weight
    plt.subplot(1, 2, 2)
    for ds in DATASETS:
        asrs = [res["pgd10_asr"] * 100.0 for res in ablation_by_dataset[ds]]
        plt.plot(WEIGHTS, asrs, marker="o", label=ds)
    mean_asrs = [res["mean_pgd10_asr"] * 100.0 for res in mean_by_weight]
    plt.plot(WEIGHTS, mean_asrs, marker="s", color="black", linewidth=2.5, linestyle="--", label="Mean Across Datasets")
    plt.axvline(x=0.7, color="red", linestyle=":", label="Option C (w=0.7)")
    plt.xlabel("RF Fusion Weight (w)")
    plt.ylabel("PGD-10 Attack Success Rate (ASR %)")
    plt.title("Adversarial PGD-10 ASR vs. RF Fusion Weight")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    plt.tight_layout()
    out_pdf_path = reports_fig_dir / "fig_fusion_ablation_pareto.pdf"
    plt.savefig(out_pdf_path, dpi=300)
    plt.close()
    print(f"Saved Pareto figure to: {out_pdf_path}\n")

    # Print Summary Results Table
    print("==========================================================================")
    print("=== SUMMARY FUSION-WEIGHT ABLATION RESULTS ===")
    print("==========================================================================")
    header = f"{'w_RF':<6} {'w_MLP':<6} | " + " | ".join([f"{ds[:10]:<10} AUC/ASR" for ds in DATASETS]) + " | Mean AUC | Mean ASR"
    print(header)
    print("-" * len(header))

    for idx, w in enumerate(WEIGHTS):
        row_str = f"{w:<6.1f} {1.0-w:<6.1f} | "
        ds_parts = []
        for ds in DATASETS:
            res = ablation_by_dataset[ds][idx]
            ds_parts.append(f"{res['clean_roc_auc']:.4f}/{res['pgd10_asr']*100:.1f}%")
        row_str += " | ".join([f"{p:<17}" for p in ds_parts])
        m_auc = mean_by_weight[idx]["mean_clean_roc_auc"]
        m_asr = mean_by_weight[idx]["mean_pgd10_asr"] * 100.0
        row_str += f" | {m_auc:.4f}   | {m_asr:.1f}%"
        print(row_str)

    print("==========================================================================\n")

if __name__ == "__main__":
    main()
