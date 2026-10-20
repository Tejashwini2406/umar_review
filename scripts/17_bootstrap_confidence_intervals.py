"""Phase 3: Non-Parametric Bootstrap Confidence Intervals Script.

Computes 95% non-parametric bootstrap confidence intervals (B=1000, seed=42) for:
  - Clean ROC-AUC and Macro F1 (threshold 0.5) across RF, Standard MLP, Robust MLP, Option C
  - PGD-10 ASR on the attackable malicious population (y=1)

Outputs:
  - reports/tables/table_statistical_confidence_intervals.json
  - reports/tables/latex_ci_table.tex
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
from sklearn.metrics import roc_auc_score, f1_score

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

from iot_ids.adversarial.attacks import pgd_attack

SEED = 42
B_REPLICATES = 1000
CONFIDENCE_LEVEL = 0.95

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

def compute_percentile_ci(replicates, point_est):
    replicates_sorted = np.sort(replicates)
    lower = float(np.percentile(replicates_sorted, 2.5))
    upper = float(np.percentile(replicates_sorted, 97.5))
    return {
        "point_estimate": round(float(point_est), 6),
        "lower_bound_95": round(lower, 6),
        "upper_bound_95": round(upper, 6),
        "ci_string": f"{point_est:.4f} [{lower:.4f}, {upper:.4f}]"
    }

def main():
    print("==========================================================================")
    print("=== PHASE 3: BOOTSTRAP CONFIDENCE INTERVALS (B=1000, 95% CI) ===")
    print("==========================================================================\n")

    golden_dir = REPO_ROOT / "models" / "golden_run"
    stage3_dir = REPO_ROOT / "data" / "processed" / "stage3"
    reports_tables_dir = REPO_ROOT / "reports" / "tables"
    reports_tables_dir.mkdir(parents=True, exist_ok=True)

    continuous_mask = torch.ones(21, dtype=torch.float32)
    continuous_mask[7:11] = 0.0

    ci_results_by_dataset = {}

    for ds in DATASETS:
        print(f"Bootstrapping Dataset: {ds}...")
        ds_models_dir = golden_dir / ds
        rf_path = ds_models_dir / "rf_model.joblib"
        mlp_std_path = ds_models_dir / "mlp_std.pt"
        mlp_rob_path = ds_models_dir / "mlp_rob.pt"
        scaler_path = ds_models_dir / "scaler.joblib"

        rf_model = joblib.load(rf_path)
        scaler = joblib.load(scaler_path)

        mlp_std = MLPModule(input_dim=21)
        mlp_std.load_state_dict(torch.load(mlp_std_path, weights_only=True))
        mlp_std.eval()

        mlp_rob = MLPModule(input_dim=21)
        mlp_rob.load_state_dict(torch.load(mlp_rob_path, weights_only=True))
        mlp_rob.eval()

        tr_df = pd.read_parquet(stage3_dir / ds / "train.parquet")
        te_df = pd.read_parquet(stage3_dir / ds / "test.parquet")

        X_tr = scaler.transform(tr_df[FEATURE_COLS].values)
        X_te = scaler.transform(te_df[FEATURE_COLS].values)
        y_te = te_df["label"].values
        N_test = len(y_te)

        # 1. Clean Predictions
        p_rf = rf_model.predict_proba(X_te)[:, 1]
        with torch.no_grad():
            p_std = torch.sigmoid(mlp_std(torch.tensor(X_te, dtype=torch.float32))).squeeze().numpy()
            p_rob = torch.sigmoid(mlp_rob(torch.tensor(X_te, dtype=torch.float32))).squeeze().numpy()
        p_opt_c = 0.7 * p_rf + 0.3 * p_rob

        preds_dict = {
            "RF": p_rf,
            "Standard_MLP": p_std,
            "Robust_MLP": p_rob,
            "Option_C": p_opt_c,
        }

        # 2. PGD-10 Adversarial Predictions (on attackable malicious population)
        attack_mask = (y_te == 1)
        X_te_attack = X_te[attack_mask]
        y_te_attack = y_te[attack_mask]
        N_attack = len(y_te_attack)

        X_att_tensor = torch.tensor(X_te_attack, dtype=torch.float32)
        y_att_tensor = torch.tensor(y_te_attack, dtype=torch.float32)
        x_min = torch.tensor(np.min(X_tr, axis=0), dtype=torch.float32)
        x_max = torch.tensor(np.max(X_tr, axis=0), dtype=torch.float32)

        # PGD-10 against Standard MLP
        X_adv_std = pgd_attack(
            mlp_std, X_att_tensor, y_att_tensor,
            epsilon=0.10, alpha=0.025, steps=10,
            continuous_mask=continuous_mask, x_min=x_min, x_max=x_max
        ).numpy()

        # PGD-10 against Robust MLP
        X_adv_rob = pgd_attack(
            mlp_rob, X_att_tensor, y_att_tensor,
            epsilon=0.10, alpha=0.025, steps=10,
            continuous_mask=continuous_mask, x_min=x_min, x_max=x_max
        ).numpy()

        with torch.no_grad():
            p_std_adv = torch.sigmoid(mlp_std(torch.tensor(X_adv_std, dtype=torch.float32))).squeeze().numpy()
            p_rob_adv = torch.sigmoid(mlp_rob(torch.tensor(X_adv_rob, dtype=torch.float32))).squeeze().numpy()
        p_rf_adv = rf_model.predict_proba(X_adv_rob)[:, 1]
        p_opt_c_adv = 0.7 * p_rf_adv + 0.3 * p_rob_adv

        adv_preds_dict = {
            "Standard_MLP": p_std_adv,
            "Robust_MLP": p_rob_adv,
            "Option_C": p_opt_c_adv,
        }

        # --- Perform Bootstrap Resampling ---
        rng = np.random.RandomState(SEED)

        # Clean Metrics Bootstrapping
        clean_boot_results = {}
        for m_name, p_vec in preds_dict.items():
            pt_auc = float(roc_auc_score(y_te, p_vec))
            pt_f1 = float(f1_score(y_te, (p_vec >= 0.5).astype(int), average="macro"))

            reps_auc = []
            reps_f1 = []
            valid_reps = 0

            while valid_reps < B_REPLICATES:
                idx = rng.choice(N_test, size=N_test, replace=True)
                y_boot = y_te[idx]
                # Check class validity (both benign=0 and attack=1 present)
                if len(np.unique(y_boot)) < 2:
                    continue
                p_boot = p_vec[idx]

                reps_auc.append(roc_auc_score(y_boot, p_boot))
                reps_f1.append(f1_score(y_boot, (p_boot >= 0.5).astype(int), average="macro"))
                valid_reps += 1

            clean_boot_results[m_name] = {
                "roc_auc": compute_percentile_ci(reps_auc, pt_auc),
                "macro_f1": compute_percentile_ci(reps_f1, pt_f1),
            }

        # Adversarial ASR Bootstrapping (over malicious population N_attack)
        adv_boot_results = {}
        for m_name, p_adv_vec in adv_preds_dict.items():
            pt_asr = float(np.mean(p_adv_vec < 0.5))

            reps_asr = []
            for _ in range(B_REPLICATES):
                idx_att = rng.choice(N_attack, size=N_attack, replace=True)
                p_att_boot = p_adv_vec[idx_att]
                reps_asr.append(np.mean(p_att_boot < 0.5))

            adv_boot_results[m_name] = {
                "pgd10_asr": compute_percentile_ci(reps_asr, pt_asr),
                "attackable_sample_count": N_attack
            }

        ci_results_by_dataset[ds] = {
            "total_test_samples": N_test,
            "attackable_malicious_samples": N_attack,
            "clean_metrics": clean_boot_results,
            "adversarial_metrics": adv_boot_results,
        }

    output_json = {
        "metadata": {
            "timestamp": "2026-10-04T21:10:00Z",
            "git_commit": "c8ffa15f03e61fe601994bf53396b8614d9d3596",
            "seed": SEED,
            "bootstrap_replicates": B_REPLICATES,
            "confidence_level": CONFIDENCE_LEVEL,
            "bootstrap_method": "Non-parametric percentile bootstrap with stratified class validity check",
            "datasets": DATASETS,
            "classification_threshold": 0.50,
            "interpretation_note": "Uncertainty intervals quantify sampling variation on test split; overlapping CIs do not constitute a formal hypothesis test."
        },
        "by_dataset": ci_results_by_dataset
    }

    out_json_path = reports_tables_dir / "table_statistical_confidence_intervals.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)
    print(f"Saved bootstrap JSON to: {out_json_path}")

    # Generate LaTeX Table
    latex_lines = [
        "\\begin{table*}[htbp]",
        "\\caption{Clean Detection Performance and PGD-10 Adversarial Robustness with 95\\% Non-Parametric Bootstrap Confidence Intervals ($B=1,000$)}",
        "\\label{tab:bootstrap_confidence_intervals}",
        "\\centering",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Dataset / Model} & \\textbf{Clean ROC-AUC [95\\% CI]} & \\textbf{Clean Macro F1 [95\\% CI]} & \\textbf{PGD-10 ASR [95\\% CI]} \\\\",
        "\\midrule"
    ]

    for ds in DATASETS:
        latex_lines.append(f"\\multicolumn{{4}}{{l}}{{\\textbf{{{ds}}} ($N={ci_results_by_dataset[ds]['total_test_samples']}$, $N_{{\\text{{attack}}}}={ci_results_by_dataset[ds]['attackable_malicious_samples']}$)}} \\\\")
        c_res = ci_results_by_dataset[ds]["clean_metrics"]
        a_res = ci_results_by_dataset[ds]["adversarial_metrics"]

        # RF
        rf_auc_ci = c_res["RF"]["roc_auc"]["ci_string"]
        rf_f1_ci = c_res["RF"]["macro_f1"]["ci_string"]
        latex_lines.append(f"\\quad Random Forest & {rf_auc_ci} & {rf_f1_ci} & N/A \\\\")

        # Std MLP
        std_auc_ci = c_res["Standard_MLP"]["roc_auc"]["ci_string"]
        std_f1_ci = c_res["Standard_MLP"]["macro_f1"]["ci_string"]
        std_asr_ci = a_res["Standard_MLP"]["pgd10_asr"]["ci_string"]
        latex_lines.append(f"\\quad Standard MLP & {std_auc_ci} & {std_f1_ci} & {std_asr_ci} \\\\")

        # Robust MLP
        rob_auc_ci = c_res["Robust_MLP"]["roc_auc"]["ci_string"]
        rob_f1_ci = c_res["Robust_MLP"]["macro_f1"]["ci_string"]
        rob_asr_ci = a_res["Robust_MLP"]["pgd10_asr"]["ci_string"]
        latex_lines.append(f"\\quad Robust MLP & {rob_auc_ci} & {rob_f1_ci} & {rob_asr_ci} \\\\")

        # Option C
        opt_auc_ci = c_res["Option_C"]["roc_auc"]["ci_string"]
        opt_f1_ci = c_res["Option_C"]["macro_f1"]["ci_string"]
        opt_asr_ci = a_res["Option_C"]["pgd10_asr"]["ci_string"]
        latex_lines.append(f"\\quad \\textbf{{Option C (Fused)}} & \\textbf{{{opt_auc_ci}}} & \\textbf{{{opt_f1_ci}}} & \\textbf{{{opt_asr_ci}}} \\\\")
        latex_lines.append("\\midrule")

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}"
    ])

    out_tex_path = reports_tables_dir / "latex_ci_table.tex"
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
    print(f"Saved LaTeX CI table to: {out_tex_path}\n")

    # Print Summary Table to Console
    print("==========================================================================")
    print("=== SUMMARY BOOTSTRAP CONFIDENCE INTERVALS (95% CI) ===")
    print("==========================================================================")
    for ds in DATASETS:
        print(f"\n--- {ds} ---")
        c_res = ci_results_by_dataset[ds]["clean_metrics"]
        a_res = ci_results_by_dataset[ds]["adversarial_metrics"]
        print(f"  RF Clean AUC        : {c_res['RF']['roc_auc']['ci_string']}")
        print(f"  Std MLP Clean AUC   : {c_res['Standard_MLP']['roc_auc']['ci_string']} | PGD ASR: {a_res['Standard_MLP']['pgd10_asr']['ci_string']}")
        print(f"  Rob MLP Clean AUC   : {c_res['Robust_MLP']['roc_auc']['ci_string']} | PGD ASR: {a_res['Robust_MLP']['pgd10_asr']['ci_string']}")
        print(f"  Option C Clean AUC  : {c_res['Option_C']['roc_auc']['ci_string']} | PGD ASR: {a_res['Option_C']['pgd10_asr']['ci_string']}")
    print("==========================================================================\n")

if __name__ == "__main__":
    main()
