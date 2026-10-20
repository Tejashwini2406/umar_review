import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix,
)
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from iot_ids.models.ensemble.risk_layer import RiskLayer
from iot_ids.utils.paths import REPO_ROOT


# Inline model definitions matching training scripts
class MLPModule(nn.Module):
    def __init__(self, input_dim: int):
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

    def forward(self, x):
        return self.net(x)


class AutoencoderModule(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        code = self.encoder(x)
        return self.decoder(code)


def compute_metrics(y_true, y_probs, threshold=0.5):
    """Compute standard classification metrics."""
    y_pred = (y_probs >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y_true, y_probs))
    except ValueError:
        auc = 0.0
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = float(fp / max(fp + tn, 1))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
        "fpr": fpr,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 14: Ensemble & Model Comparison Experiment Runner")
    parser.add_argument(
        "--profile",
        type=str,
        choices=["historical_13", "standardized_21"],
        default="historical_13",
        help="Feature profile selection: 'historical_13' (default) or 'standardized_21'",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"=== ENSEMBLE & MODEL COMPARISON EXPERIMENT RUNNER [PROFILE: {args.profile}] ===")
    print("=" * 60 + "\n")

    if args.profile == "historical_13":
        base_dir = REPO_ROOT / "data" / "processed" / "final"
        models_base = REPO_ROOT / "models" / "final"
        output_dir = REPO_ROOT / "reports" / "ensemble"
        output_dir.mkdir(parents=True, exist_ok=True)
        datasets = ["Edge-IIoTset", "ToN-IoT"]
        prep_filename = "prep_indomain.joblib"
        json_filename = "ensemble_experiment_results.json"
        md_filename = "ENSEMBLE_EVIDENCE_REPORT.md"
    elif args.profile == "standardized_21":
        base_dir = REPO_ROOT / "data" / "processed" / "stage3"
        models_base = REPO_ROOT / "models" / "standardized"
        output_dir = REPO_ROOT / "reports" / "standardized"
        output_dir.mkdir(parents=True, exist_ok=True)
        datasets = ["ToN-IoT", "Edge-IIoTset", "NF-ToN-IoT-v2", "CICIoT2023"]
        prep_filename = "prep_standardized.joblib"
        json_filename = "standardized_ensemble_results.json"
        md_filename = "STANDARDIZED_ENSEMBLE_EVIDENCE.md"

    all_results = {}

    for ds in datasets:
        print(f"\n{'='*50}")
        print(f"  Dataset: {ds} (profile={args.profile})")
        print(f"{'='*50}")

        ds_dir = base_dir / ds
        if not ds_dir.exists():
            print(f"Warning: Dataset directory {ds_dir} does not exist. Skipping.")
            continue

        if args.profile == "historical_13":
            fingerprint_dirs = [d for d in ds_dir.iterdir() if d.is_dir()]
            if not fingerprint_dirs:
                print(f"Warning: No fingerprint directory in {ds_dir}. Skipping.")
                continue
            target_dir = fingerprint_dirs[0]
            test_df = pd.read_parquet(target_dir / "splits" / "test.parquet")
            val_df = pd.read_parquet(target_dir / "splits" / "val.parquet")
        else:
            test_df = pd.read_parquet(ds_dir / "test.parquet")
            val_df = pd.read_parquet(ds_dir / "val.parquet")

        m_dir = models_base / ds
        prep_file = m_dir / prep_filename
        rf_file = m_dir / "rf_model.joblib"
        mlp_file = m_dir / "mlp_model.pt"
        ae_file = m_dir / "ae_model.pt"
        risk_file = m_dir / "risk_layer.json"

        for f in [prep_file, rf_file, mlp_file, ae_file, risk_file]:
            if not f.exists():
                raise RuntimeError(f"Dataset '{ds}': Missing required model artifact {f}")

        prep = joblib.load(prep_file)
        rf = joblib.load(rf_file)

        X_test = prep.transform(test_df)
        y_test = test_df["label"].values
        X_val = prep.transform(val_df)
        y_val = val_df["label"].values
        input_dim = X_test.shape[1]

        if args.profile == "standardized_21" and input_dim != 21:
            raise RuntimeError(f"Dataset '{ds}': Transformed X_test dimension is {input_dim}, expected 21")

        # Load MLP
        mlp = MLPModule(input_dim=input_dim)
        mlp.load_state_dict(torch.load(mlp_file, weights_only=True))
        mlp.eval()

        # Load AE
        ae = AutoencoderModule(input_dim=input_dim, bottleneck_dim=16)
        ae.load_state_dict(torch.load(ae_file, weights_only=True))
        ae.eval()

        # Load RiskLayer
        risk_layer = RiskLayer.load(risk_file)

        ds_results = {}

        # ----- E1: Individual Model Baselines -----
        print("\n  [E1] Individual Model Baselines")

        # RF
        p_rf = rf.predict_proba(X_test)[:, 1]
        rf_metrics = compute_metrics(y_test, p_rf)
        ds_results["rf_baseline"] = rf_metrics
        print(f"    RF:  AUC={rf_metrics['auc']:.4f}, F1={rf_metrics['f1']:.4f}, FPR={rf_metrics['fpr']:.4f}")

        # MLP
        with torch.no_grad():
            mlp_logits = mlp(torch.tensor(X_test, dtype=torch.float32)).squeeze(-1)
            p_mlp = torch.sigmoid(mlp_logits).numpy()
        mlp_metrics = compute_metrics(y_test, p_mlp)
        ds_results["mlp_baseline"] = mlp_metrics
        print(f"    MLP: AUC={mlp_metrics['auc']:.4f}, F1={mlp_metrics['f1']:.4f}, FPR={mlp_metrics['fpr']:.4f}")

        # ----- E3: AE Anomaly Detection -----
        print("\n  [E3] AE Anomaly Detection Evaluation")
        with torch.no_grad():
            recon = ae(torch.tensor(X_test, dtype=torch.float32)).numpy()
        mse_per_sample = np.mean((X_test - recon) ** 2, axis=1)

        mse_benign = mse_per_sample[y_test == 0]
        mse_attack = mse_per_sample[y_test == 1]

        # Mann-Whitney U test for distribution separation
        stat, p_value = mannwhitneyu(mse_attack, mse_benign, alternative="greater")
        # AE AUC: can MSE alone separate benign from attack?
        try:
            ae_auc = float(roc_auc_score(y_test, mse_per_sample))
        except ValueError:
            ae_auc = 0.0

        ds_results["ae_anomaly"] = {
            "benign_mse_mean": float(np.mean(mse_benign)),
            "benign_mse_std": float(np.std(mse_benign)),
            "attack_mse_mean": float(np.mean(mse_attack)),
            "attack_mse_std": float(np.std(mse_attack)),
            "mse_ratio": float(np.mean(mse_attack) / max(np.mean(mse_benign), 1e-9)),
            "ae_auc_mse_only": ae_auc,
            "mannwhitney_stat": float(stat),
            "mannwhitney_pvalue": float(p_value),
        }
        print(f"    Benign MSE: {np.mean(mse_benign):.4f} ± {np.std(mse_benign):.4f}")
        print(f"    Attack MSE: {np.mean(mse_attack):.4f} ± {np.std(mse_attack):.4f}")
        print(f"    AE AUC (MSE-only): {ae_auc:.4f}")
        print(f"    Mann-Whitney p-value: {p_value:.4e}")

        # ----- E4: RF + MLP Ensemble Weight Sweep -----
        print("\n  [E4] RF + MLP Ensemble (Weight Sweep)")
        ensemble_results = {}
        for alpha in [0.3, 0.5, 0.7]:
            p_ens = alpha * p_rf + (1 - alpha) * p_mlp
            ens_metrics = compute_metrics(y_test, p_ens)
            ensemble_results[f"alpha_{alpha:.1f}"] = ens_metrics
            print(f"    a={alpha:.1f}: AUC={ens_metrics['auc']:.4f}, F1={ens_metrics['f1']:.4f}, FPR={ens_metrics['fpr']:.4f}")
        ds_results["ensemble_rf_mlp"] = ensemble_results

        # Best single vs best ensemble
        best_single = max(rf_metrics["auc"], mlp_metrics["auc"])
        best_ens = max(r["auc"] for r in ensemble_results.values())
        ds_results["ensemble_vs_single"] = {
            "best_single_auc": best_single,
            "best_ensemble_auc": best_ens,
            "ensemble_improvement": best_ens - best_single,
        }

        # ----- E5: RiskLayer Full Fusion -----
        print("\n  [E5] RiskLayer Full Fusion (RF + MLP + AE)")
        decisions = risk_layer.predict(p_rf, p_mlp, mse_per_sample)
        states = np.array([d.risk_state for d in decisions])

        risk_counts = {
            "HIGH CONFIDENCE ATTACK": int((states == "HIGH CONFIDENCE ATTACK").sum()),
            "SUSPICIOUS / ANOMALOUS": int((states == "SUSPICIOUS / ANOMALOUS").sum()),
            "BENIGN": int((states == "BENIGN").sum()),
        }

        # For detection metrics: HIGH CONFIDENCE + SUSPICIOUS = detected
        detected = (states != "BENIGN").astype(int)
        risk_metrics = compute_metrics(y_test, detected.astype(float), threshold=0.5)

        ds_results["risk_layer"] = {
            "state_counts": risk_counts,
            "detection_metrics": risk_metrics,
            "tau_sup": risk_layer.tau_sup,
            "tau_ae": risk_layer.tau_ae,
        }
        print(f"    HIGH CONFIDENCE ATTACK: {risk_counts['HIGH CONFIDENCE ATTACK']}")
        print(f"    SUSPICIOUS / ANOMALOUS: {risk_counts['SUSPICIOUS / ANOMALOUS']}")
        print(f"    BENIGN: {risk_counts['BENIGN']}")
        print(f"    Combined Detection AUC: {risk_metrics['auc']:.4f}, F1: {risk_metrics['f1']:.4f}")

        all_results[ds] = ds_results

    # Save results
    results_path = output_dir / json_filename
    results_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved: {results_path}")

    # Generate evidence report
    md = f"""# ENSEMBLE & MODEL COMPARISON EVIDENCE REPORT [PROFILE: {args.profile}]

> [!IMPORTANT]
> **Multi-model comparison and ensemble evaluation using frozen model checkpoints.**
> RF, MLP, AE evaluated individually and in ensemble configurations. Profile: `{args.profile}`.

---

## Results Summary

"""
    for ds, ds_res in all_results.items():
        md += f"### {ds}\n\n"
        md += "#### Individual Baselines\n\n"
        md += "| Model | AUC | F1 | FPR | Precision | Recall |\n"
        md += "|:---|:---:|:---:|:---:|:---:|:---:|\n"
        rf_m = ds_res["rf_baseline"]
        mlp_m = ds_res["mlp_baseline"]
        md += f"| RF | {rf_m['auc']:.4f} | {rf_m['f1']:.4f} | {rf_m['fpr']:.4f} | {rf_m['precision']:.4f} | {rf_m['recall']:.4f} |\n"
        md += f"| MLP | {mlp_m['auc']:.4f} | {mlp_m['f1']:.4f} | {mlp_m['fpr']:.4f} | {mlp_m['precision']:.4f} | {mlp_m['recall']:.4f} |\n\n"

        ae_m = ds_res["ae_anomaly"]
        md += f"#### AE Anomaly Detection\n\n"
        md += f"- Benign MSE: {ae_m['benign_mse_mean']:.4f} ± {ae_m['benign_mse_std']:.4f}\n"
        md += f"- Attack MSE: {ae_m['attack_mse_mean']:.4f} ± {ae_m['attack_mse_std']:.4f}\n"
        md += f"- AE-only AUC: {ae_m['ae_auc_mse_only']:.4f}\n"
        md += f"- Mann-Whitney p-value: {ae_m['mannwhitney_pvalue']:.4e}\n\n"

        md += "#### RF + MLP Ensemble\n\n"
        md += "| α (RF weight) | AUC | F1 | FPR |\n"
        md += "|:---:|:---:|:---:|:---:|\n"
        for alpha_key, ens_m in ds_res["ensemble_rf_mlp"].items():
            md += f"| {alpha_key} | {ens_m['auc']:.4f} | {ens_m['f1']:.4f} | {ens_m['fpr']:.4f} |\n"

        evs = ds_res["ensemble_vs_single"]
        md += f"\nBest single AUC: {evs['best_single_auc']:.4f}, Best ensemble AUC: {evs['best_ensemble_auc']:.4f}, Δ: {evs['ensemble_improvement']:+.4f}\n\n"

        rl = ds_res["risk_layer"]
        md += "#### RiskLayer Full Fusion\n\n"
        md += f"| Risk State | Count |\n|:---|:---:|\n"
        for state, count in rl["state_counts"].items():
            md += f"| {state} | {count} |\n"
        dm = rl["detection_metrics"]
        md += f"\nCombined Detection: AUC={dm['auc']:.4f}, F1={dm['f1']:.4f}, FPR={dm['fpr']:.4f}\n\n"
        md += "---\n\n"

    md_path = output_dir / md_filename
    md_path.write_text(md, encoding="utf-8")
    print(f"Evidence report: {md_path}")
    print(f"\nEnsemble experiments for profile '{args.profile}' complete.")


if __name__ == "__main__":
    main()

