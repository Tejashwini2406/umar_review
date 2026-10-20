"""SHAP Explainability Experiment Runner.

Runs the complete XAI experimental protocol using SHAP values:
  X1: SHAP TreeExplainer for RF (global feature importance)
  X2: SHAP GradientExplainer for MLP (global feature importance)
  X3: RF vs MLP SHAP ranking consensus (Spearman rho)
  X4: AE per-feature reconstruction error on attacks vs benign
  X5: Domain-specific SHAP comparison (cross-domain stability)

Outputs:
  - reports/xai/shap_results.json
  - reports/xai/SHAP_EVIDENCE_REPORT.md
"""
import argparse
import json
import sys
from pathlib import Path

joblib = __import__("joblib")
np = __import__("numpy")
pd = __import__("pandas")
torch = __import__("torch")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from iot_ids.xai.shap_explainer import explain_rf_shap, explain_mlp_shap, compare_shap_rankings
from iot_ids.xai.ae_explainer import explain_ae_reconstruction
from iot_ids.utils.paths import REPO_ROOT


# Inline model definitions matching training scripts
class MLPModule(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class AutoencoderModule(torch.nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int = 16):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, bottleneck_dim),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(bottleneck_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, input_dim),
        )

    def forward(self, x):
        code = self.encoder(x)
        return self.decoder(code)


def main():
    parser = argparse.ArgumentParser(description="Stage 12: SHAP Explainability Experiments")
    parser.add_argument(
        "--profile",
        choices=["historical_13", "standardized_21"],
        default="historical_13",
        help="Feature profile selection: 'historical_13' (default) or 'standardized_21'",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"=== SHAP EXPLAINABILITY EXPERIMENT RUNNER [PROFILE: {args.profile}] ===")
    print("=" * 60 + "\n")

    if args.profile == "historical_13":
        base_dir = REPO_ROOT / "data" / "processed" / "final"
        models_base = REPO_ROOT / "models" / "final"
        output_dir = REPO_ROOT / "reports" / "xai"
        datasets = ["Edge-IIoTset", "ToN-IoT"]
        prep_filename = "prep_indomain.joblib"
    else:
        base_dir = REPO_ROOT / "data" / "processed" / "stage3"
        models_base = REPO_ROOT / "models" / "standardized"
        output_dir = REPO_ROOT / "reports" / "standardized" / "xai"
        datasets = ["ToN-IoT", "Edge-IIoTset", "NF-ToN-IoT-v2", "CICIoT2023"]
        prep_filename = "prep_standardized.joblib"

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    csv_rows = []

    for ds in datasets:
        print(f"\n--- Processing {ds} ({args.profile}) ---")
        ds_dir = base_dir / ds
        if not ds_dir.exists():
            print(f"  SKIPPED: {ds_dir} not found")
            continue

        if args.profile == "historical_13":
            fingerprint_dirs = [d for d in ds_dir.iterdir() if d.is_dir()]
            if not fingerprint_dirs:
                print(f"  SKIPPED: No fingerprint dir in {ds_dir}")
                continue
            test_path = fingerprint_dirs[0] / "splits" / "test.parquet"
        else:
            test_path = ds_dir / "test.parquet"

        if not test_path.exists():
            print(f"  SKIPPED: {test_path} not found")
            continue

        test_df = pd.read_parquet(test_path)
        test_sub = test_df.sample(n=min(1000, len(test_df)), random_state=42)

        m_dir = models_base / ds
        prep = joblib.load(m_dir / prep_filename)
        rf = joblib.load(m_dir / "rf_model.joblib")

        X = prep.transform(test_sub)
        y = test_sub["label"].values

        if args.profile == "historical_13":
            feature_names = prep.numeric_cols
        else:
            feature_names = prep.feature_names

        input_dim = X.shape[1]
        expected_dim = 13 if args.profile == "historical_13" else 21
        assert input_dim == expected_dim, f"Dimension mismatch: expected {expected_dim}, got {input_dim}"

        mlp = MLPModule(input_dim=input_dim)
        mlp.load_state_dict(torch.load(m_dir / "mlp_model.pt", weights_only=True))
        mlp.eval()

        ae = AutoencoderModule(input_dim=input_dim, bottleneck_dim=16)
        ae.load_state_dict(torch.load(m_dir / "ae_model.pt", weights_only=True))
        ae.eval()

        # ----- X1: SHAP TreeExplainer for RF -----
        print("  [X1] Computing SHAP TreeExplainer for RF...")
        try:
            rf_shap = explain_rf_shap(rf, X, feature_names, max_samples=500)
            print(f"       Top features: {list(rf_shap['mean_abs_shap'].keys())[:5]}")
        except Exception as e:
            print(f"       ERROR: {e}")
            rf_shap = None

        # ----- X2: SHAP for MLP -----
        print("  [X2] Computing SHAP for MLP...")
        try:
            mlp_shap = explain_mlp_shap(mlp, X, feature_names, background_samples=100, max_samples=300)
            print(f"       Top features: {list(mlp_shap['mean_abs_shap'].keys())[:5]}")
        except Exception as e:
            print(f"       ERROR: {e}")
            mlp_shap = None

        # ----- X3: RF vs MLP SHAP consensus -----
        consensus = None
        if rf_shap and mlp_shap:
            print("  [X3] Computing RF <-> MLP SHAP consensus...")
            consensus = compare_shap_rankings(rf_shap, mlp_shap, "RF_SHAP", "MLP_SHAP")
            print(f"       Spearman rho = {consensus['spearman_rho']:.4f} (p = {consensus['p_value']:.4e})")

        # ----- X4: AE reconstruction error breakdown -----
        print("  [X4] Computing AE reconstruction error decomposition...")
        ae_benign = explain_ae_reconstruction(ae, X[y == 0], feature_names)
        ae_attack = explain_ae_reconstruction(ae, X[y == 1], feature_names)

        benign_mse_mean = float(np.mean(ae_benign["total_mse"]))
        attack_mse_mean = float(np.mean(ae_attack["total_mse"]))
        mse_ratio = attack_mse_mean / max(benign_mse_mean, 1e-9)
        print(f"       Benign MSE: {benign_mse_mean:.4f}, Attack MSE: {attack_mse_mean:.4f}, Ratio: {mse_ratio:.2f}")

        # Build feature rankings CSV rows
        if rf_shap:
            for rank, (fname, imp_val) in enumerate(rf_shap["mean_abs_shap"].items(), 1):
                mlp_imp = mlp_shap["mean_abs_shap"].get(fname, 0.0) if mlp_shap else 0.0
                csv_rows.append({
                    "dataset": ds,
                    "profile": args.profile,
                    "rank": rank,
                    "feature": fname,
                    "rf_shap_importance": imp_val,
                    "mlp_shap_importance": mlp_imp,
                })

        ds_results = {
            "dataset": ds,
            "profile": args.profile,
            "n_samples": len(X),
            "feature_dim": input_dim,
            "features": feature_names,
        }
        if rf_shap:
            ds_results["rf_shap_importance"] = rf_shap["mean_abs_shap"]
            ds_results["rf_shap_stats"] = rf_shap["feature_stats"]
        if mlp_shap:
            ds_results["mlp_shap_importance"] = mlp_shap["mean_abs_shap"]
            ds_results["mlp_shap_stats"] = mlp_shap["feature_stats"]
        if consensus:
            ds_results["consensus"] = {
                "spearman_rho": consensus["spearman_rho"],
                "p_value": consensus["p_value"],
                "comparison": consensus.get("comparison", []),
            }
        ds_results["ae_analysis"] = {
            "benign_mean_mse": benign_mse_mean,
            "attack_mean_mse": attack_mse_mean,
            "mse_ratio": mse_ratio,
            "benign_top_features": dict(list(ae_benign["mean_mse_per_feature"].items())[:5]),
            "attack_top_features": dict(list(ae_attack["mean_mse_per_feature"].items())[:5]),
        }

        all_results[ds] = ds_results

    # ----- X5: Cross-Domain SHAP Stability -----
    ds_names = [d for d in datasets if d in all_results]
    if len(ds_names) >= 2 and "rf_shap_importance" in all_results[ds_names[0]] and "rf_shap_importance" in all_results[ds_names[1]]:
        print("\n--- [X5] Cross-Domain SHAP Stability ---")
        feats_0 = set(all_results[ds_names[0]]["rf_shap_importance"].keys())
        feats_1 = set(all_results[ds_names[1]]["rf_shap_importance"].keys())
        common = feats_0 & feats_1
        if len(common) >= 3:
            from scipy.stats import spearmanr
            vec_0 = [all_results[ds_names[0]]["rf_shap_importance"][f] for f in common]
            vec_1 = [all_results[ds_names[1]]["rf_shap_importance"][f] for f in common]
            res = spearmanr(vec_0, vec_1)
            rho = float(res.statistic) if not np.isnan(res.statistic) else 0.0
            p_val = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0
            all_results["cross_domain_shap_stability"] = {
                "domain_pair": f"{ds_names[0]} <-> {ds_names[1]}",
                "common_features": list(common),
                "spearman_rho": rho,
                "p_value": p_val,
            }
            print(f"  {ds_names[0]} <-> {ds_names[1]} RF SHAP stability: rho = {rho:.4f} (p = {p_val:.4e})")

    # Save JSON results
    json_path = output_dir / "shap_results.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nSHAP results saved: {json_path}")

    # Save feature_rankings.csv
    if csv_rows:
        df_csv = pd.DataFrame(csv_rows)
        csv_path = output_dir / "feature_rankings.csv"
        df_csv.to_csv(csv_path, index=False)
        print(f"Feature rankings saved: {csv_path}")

    # Generate evidence report
    md_filename = "SHAP_EVIDENCE.md" if args.profile == "standardized_21" else "SHAP_EVIDENCE_REPORT.md"
    md_content = f"""# SHAP EXPLAINABILITY EVIDENCE REPORT [PROFILE: {args.profile}]

> [!IMPORTANT]
> **SHAP (SHapley Additive exPlanations) analysis using TreeExplainer for RF
> and GradientExplainer/KernelExplainer for MLP on frozen model checkpoints.**
> Profile: `{args.profile}` ({expected_dim} features).

---

## 1. Experimental Protocol

| Parameter | Value |
|:---|:---|
| Feature Profile | `{args.profile}` ({expected_dim} features) |
| RF Explainer | SHAP TreeExplainer (exact Shapley values) |
| MLP Explainer | SHAP GradientExplainer (expected gradients) / KernelExplainer fallback |
| Samples Explained | Up to 500 (RF) / 300 (MLP) per dataset |
| Background Samples | 100 (for MLP explainer) |

---

## 2. Per-Dataset SHAP Results

"""
    for ds_name, ds_res in all_results.items():
        if ds_name.startswith("cross_domain"):
            continue
        md_content += f"### {ds_name}\n\n"
        if "rf_shap_importance" in ds_res:
            md_content += "**RF SHAP Feature Importance (Top 5):**\n\n"
            md_content += "| Feature | Mean |SHAP| |\n|:---|:---:|\n"
            for f, v in list(ds_res["rf_shap_importance"].items())[:5]:
                md_content += f"| `{f}` | {v:.4f} |\n"
            md_content += "\n"
        if "mlp_shap_importance" in ds_res:
            md_content += "**MLP SHAP Feature Importance (Top 5):**\n\n"
            md_content += "| Feature | Mean |SHAP| |\n|:---|:---:|\n"
            for f, v in list(ds_res["mlp_shap_importance"].items())[:5]:
                md_content += f"| `{f}` | {v:.4f} |\n"
            md_content += "\n"
        if "consensus" in ds_res:
            c = ds_res["consensus"]
            md_content += f"**RF <-> MLP Consensus:** Spearman ρ = **{c['spearman_rho']:.4f}** (p = {c['p_value']:.4e})\n\n"
        if "ae_analysis" in ds_res:
            ae = ds_res["ae_analysis"]
            md_content += f"**AE Analysis:** Benign MSE = {ae['benign_mean_mse']:.4f}, Attack MSE = {ae['attack_mean_mse']:.4f}, Ratio = {ae['mse_ratio']:.2f}\n\n"
        md_content += "---\n\n"

    if "cross_domain_shap_stability" in all_results:
        cds = all_results["cross_domain_shap_stability"]
        md_content += f"""## 3. Cross-Domain SHAP Stability

| Domain Pair | Spearman ρ | p-value |
|:---|:---:|:---:|
| {cds['domain_pair']} | **{cds['spearman_rho']:.4f}** | {cds['p_value']:.4e} |

"""

    md_path = output_dir / md_filename
    md_path.write_text(md_content, encoding="utf-8")
    print(f"Evidence report: {md_path}")
    print(f"\nSHAP Explainability experiments complete for profile '{args.profile}'.")


if __name__ == "__main__":
    main()

