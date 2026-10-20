"""Golden Experiment Pipeline for IoT IDS Option C Framework.

Executes end-to-end scientific validation and recomputation across four real benchmark datasets:
  - Edge-IIoTset
  - NF-ToN-IoT-v2
  - ToN-IoT
  - CICIoT2023

Pipeline Steps:
  1. Train & Evaluate Detection Baseline (RF, Standard MLP, Robust MLP, Option C)
  2. Perform Domain-Constrained PGD-10 Adversarial Evaluation
  3. Perform Global & Local XAI Audit on Real Test Samples
  4. Perform Differential Privacy (DP-SGD) & Data Minimization Audit
  5. Run Execution-Flow Runtime Benchmarks (N=1, 32, 1024)
  6. Generate Golden Experiment Manifest & Final Evidence Summaries
"""
import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure src/ is in python path
REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

from iot_ids.adversarial.adversarial_training import train_mlp_adversarial
from iot_ids.adversarial.attacks import pgd_attack
from iot_ids.models.autoencoder import Autoencoder
from iot_ids.privacy.data_minimization import DataMinimizationPolicy
from iot_ids.privacy.dp_sgd import DPConfig, train_robust_mlp_dp
from iot_ids.runtime.engine import InferenceEngine
from iot_ids.xai.global_xai import analyze_dataset_global_xai, compute_cross_dataset_stability

# Set reproducible seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATASETS = ["Edge-IIoTset", "NF-ToN-IoT-v2", "ToN-IoT", "CICIoT2023"]
STAGE3_DIR = REPO_ROOT / "data" / "processed" / "stage3_corrected"

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

def train_standard_mlp(X_tr: np.ndarray, y_tr: np.ndarray, epochs: int = 15, batch_size: int = 256) -> MLPModule:
    torch.manual_seed(SEED)
    model = MLPModule(input_dim=X_tr.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    return model

def train_robust_mlp(X_tr: np.ndarray, y_tr: np.ndarray, epochs: int = 15, batch_size: int = 256) -> MLPModule:
    torch.manual_seed(SEED)
    model = MLPModule(input_dim=X_tr.shape[1])
    train_mlp_adversarial(
        model=model,
        X_train=X_tr,
        y_train=y_tr,
        feature_names=FEATURE_COLS,
        epsilon=0.10,
        pgd_steps=7,
        adv_ratio=0.5,
        epochs=epochs,
        batch_size=batch_size,
        lr=0.001,
        verbose=False,
    )
    return model

def train_autoencoder(X_benign: np.ndarray, epochs: int = 15, batch_size: int = 256) -> Autoencoder:
    model = Autoencoder(input_dim=X_benign.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = TensorDataset(torch.tensor(X_benign, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()

    model.eval()
    return model

def main():
    print("==========================================================================")
    print("=== STARTING GOLDEN EXPERIMENT PIPELINE (REAL DATASET PROVENANCE) ===")
    print("==========================================================================\n")

    t_start = time.time()
    manifest_info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "datasets": DATASETS,
        "feature_count": len(FEATURE_COLS),
        "results": {}
    }

    models_dir = REPO_ROOT / "models" / "corrected_run"
    models_dir.mkdir(parents=True, exist_ok=True)
    run_reports_dir = REPO_ROOT / "reports" / "corrected_run"
    reports_xai_dir = run_reports_dir / "xai"
    reports_privacy_dir = run_reports_dir / "privacy"
    reports_runtime_dir = run_reports_dir / "runtime"
    run_reports_dir.mkdir(parents=True, exist_ok=True)
    reports_xai_dir.mkdir(parents=True, exist_ok=True)
    reports_privacy_dir.mkdir(parents=True, exist_ok=True)
    reports_runtime_dir.mkdir(parents=True, exist_ok=True)

    detection_results = {}
    adversarial_results = {}
    xai_dataset_results = {}
    dp_results = {}

    # Continuous mask: 1 for continuous, 0 for protocol one-hot indicators (idx 7-10)
    continuous_mask = torch.ones(21, dtype=torch.float32)
    continuous_mask[7:11] = 0.0

    print("--------------------------------------------------------------------------")
    print("PHASE 1: DETECTION BASELINE & ADVERSARIAL EVALUATION (4 REAL DATASETS)")
    print("--------------------------------------------------------------------------")

    for ds in DATASETS:
        print(f"\nProcessing Dataset: {ds}...")
        ds_dir = STAGE3_DIR / ds
        split_manifest_path = ds_dir / "split_manifest.json"
        if not split_manifest_path.is_file():
            raise RuntimeError(
                f"Missing corrected split manifest at {split_manifest_path}; "
                "run scripts/03_materialize_and_audit_features.py first"
            )
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        expected_strategy = (
            "official_source_partitions"
            if ds == "CICIoT2023"
            else "seeded_stratified_random"
        )
        if (
            split_manifest.get("sampling_method") != "uniform_reservoir"
            or split_manifest.get("split_strategy") != expected_strategy
        ):
            raise RuntimeError(
                f"{ds} split manifest does not match the corrected protocol"
            )
        tr_df = pd.read_parquet(ds_dir / "train.parquet")
        va_df = pd.read_parquet(ds_dir / "val.parquet")
        te_df = pd.read_parquet(ds_dir / "test.parquet")
        for split_name, split_df in (
            ("train", tr_df),
            ("validation", va_df),
            ("test", te_df),
        ):
            expected_counts = split_manifest["label_counts"][split_name]
            actual_counts = {
                "benign": int((split_df["label"] == 0).sum()),
                "attack": int((split_df["label"] == 1).sum()),
            }
            if actual_counts != expected_counts:
                raise RuntimeError(
                    f"{ds} {split_name} labels do not match the split manifest"
                )

        # Enforce Data Minimization Audit
        DataMinimizationPolicy.verify_schema_minimization(FEATURE_COLS)

        scaler = RobustScaler()
        X_tr = scaler.fit_transform(tr_df[FEATURE_COLS].values)
        X_te = scaler.transform(te_df[FEATURE_COLS].values)

        y_tr = tr_df["label"].values
        y_te = te_df["label"].values

        # 1. Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=SEED, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        p_rf = rf.predict_proba(X_te)[:, 1]
        rf_auc = float(roc_auc_score(y_te, p_rf))

        # 2. Standard MLP
        mlp_std = train_standard_mlp(X_tr, y_tr, epochs=15)
        mlp_std.eval()
        with torch.no_grad():
            p_mlp_std = torch.sigmoid(mlp_std(torch.tensor(X_te, dtype=torch.float32))).squeeze().numpy()
        mlp_std_auc = float(roc_auc_score(y_te, p_mlp_std))

        # 3. Robust MLP (PGD-7)
        mlp_rob = train_robust_mlp(X_tr, y_tr, epochs=15)
        mlp_rob.eval()
        with torch.no_grad():
            p_mlp_rob = torch.sigmoid(mlp_rob(torch.tensor(X_te, dtype=torch.float32))).squeeze().numpy()
        mlp_rob_auc = float(roc_auc_score(y_te, p_mlp_rob))

        # 4. Option C (0.7 * RF + 0.3 * Robust MLP)
        p_opt_c = 0.7 * p_rf + 0.3 * p_mlp_rob
        opt_c_auc = float(roc_auc_score(y_te, p_opt_c))

        detection_results[ds] = {
            "n_train": len(y_tr),
            "n_test": len(y_te),
            "rf_roc_auc": rf_auc,
            "std_mlp_roc_auc": mlp_std_auc,
            "robust_mlp_roc_auc": mlp_rob_auc,
            "option_c_roc_auc": opt_c_auc,
        }

        print(f"  -> RF ROC-AUC:         {rf_auc:.6f}")
        print(f"  -> Std MLP ROC-AUC:    {mlp_std_auc:.6f}")
        print(f"  -> Robust MLP ROC-AUC: {mlp_rob_auc:.6f}")
        print(f"  -> Option C ROC-AUC:   {opt_c_auc:.6f}")

        # Save model artifacts
        out_ds_dir = models_dir / ds
        out_ds_dir.mkdir(parents=True, exist_ok=True)
        (out_ds_dir / "training_split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2), encoding="utf-8"
        )
        scaler.feature_names = FEATURE_COLS
        joblib.dump(scaler, out_ds_dir / "scaler.joblib")
        joblib.dump(scaler, out_ds_dir / "prep_standardized.joblib")
        joblib.dump(rf, out_ds_dir / "rf_model.joblib")
        torch.save(mlp_std.state_dict(), out_ds_dir / "mlp_std.pt")
        torch.save(mlp_rob.state_dict(), out_ds_dir / "mlp_rob.pt")
        torch.save(mlp_rob.state_dict(), out_ds_dir / "mlp_adversarial.pt")

        # --- ADVERSARIAL EVALUATION (PGD-10, eps=0.10, alpha=0.025) ---
        attack_mask = (y_te == 1)
        X_te_attack = X_te[attack_mask]
        y_te_attack = y_te[attack_mask]

        if len(X_te_attack) > 0:
            X_att_tensor = torch.tensor(X_te_attack, dtype=torch.float32)
            y_att_tensor = torch.tensor(y_te_attack, dtype=torch.float32)
            x_min = torch.tensor(np.min(X_tr, axis=0), dtype=torch.float32)
            x_max = torch.tensor(np.max(X_tr, axis=0), dtype=torch.float32)

            # Attack Standard MLP
            X_adv_std = pgd_attack(
                mlp_std, X_att_tensor, y_att_tensor,
                epsilon=0.10, alpha=0.025, steps=10,
                continuous_mask=continuous_mask, x_min=x_min, x_max=x_max
            ).numpy()

            # Attack Robust MLP
            X_adv_rob = pgd_attack(
                mlp_rob, X_att_tensor, y_att_tensor,
                epsilon=0.10, alpha=0.025, steps=10,
                continuous_mask=continuous_mask, x_min=x_min, x_max=x_max
            ).numpy()

            # Predictions on adversarial samples
            # Std MLP ASR
            with torch.no_grad():
                p_std_adv = torch.sigmoid(mlp_std(torch.tensor(X_adv_std, dtype=torch.float32))).squeeze().numpy()
            asr_std = float(np.mean(p_std_adv < 0.5))

            # Robust MLP ASR (under attack on Robust MLP)
            with torch.no_grad():
                p_rob_adv = torch.sigmoid(mlp_rob(torch.tensor(X_adv_rob, dtype=torch.float32))).squeeze().numpy()
            asr_rob = float(np.mean(p_rob_adv < 0.5))

            # Option C ASR under attack on Robust MLP stream
            p_rf_att = rf.predict_proba(X_adv_rob)[:, 1]
            p_opt_c_adv = 0.7 * p_rf_att + 0.3 * p_rob_adv
            asr_opt_c = float(np.mean(p_opt_c_adv < 0.5))

            adversarial_results[ds] = {
                "std_mlp_pgd10_asr": asr_std,
                "robust_mlp_pgd10_asr": asr_rob,
                "option_c_pgd10_asr": asr_opt_c,
            }

            print(f"  [PGD-10 eps=0.10] Std MLP ASR: {asr_std*100:.2f}% | Robust MLP ASR: {asr_rob*100:.2f}% | Option C ASR: {asr_opt_c*100:.2f}%")

        # --- XAI AUDIT ---
        xai_res = analyze_dataset_global_xai(
            rf_model=rf,
            mlp_model=mlp_rob,
            X=X_te[:300],
            y=y_te[:300],
            feature_names=FEATURE_COLS,
            dataset_name=ds,
            max_shap_samples=200,
            background_samples=50
        )
        xai_dataset_results[ds] = xai_res

        # --- DIFFERENTIAL PRIVACY AUDIT ---
        dp_results[ds] = []
        for noise_mult in [0.0, 0.5, 1.0, 2.0]:
            dp_cfg = DPConfig(
                max_grad_norm=1.0,
                noise_multiplier=noise_mult,
                target_delta=1e-5,
                epochs=10,
                batch_size=64,
            )
            model_dp = MLPModule(input_dim=21)
            dp_out = train_robust_mlp_dp(model_dp, X_tr, y_tr, dp_config=dp_cfg, adv_eps=0.10, adv_alpha=0.025, adv_steps=7)
            
            model_dp.eval()
            with torch.no_grad():
                p_dp = torch.sigmoid(model_dp(torch.tensor(X_te, dtype=torch.float32))).squeeze().numpy()
            dp_auc = float(roc_auc_score(y_te, p_dp))

            # ASR under PGD-10
            if len(X_te_attack) > 0:
                X_adv_dp = pgd_attack(
                    model_dp, X_att_tensor, y_att_tensor,
                    epsilon=0.10, alpha=0.025, steps=10,
                    continuous_mask=continuous_mask, x_min=x_min, x_max=x_max
                ).numpy()
                with torch.no_grad():
                    p_dp_adv = torch.sigmoid(model_dp(torch.tensor(X_adv_dp, dtype=torch.float32))).squeeze().numpy()
                dp_asr = float(np.mean(p_dp_adv < 0.5))
            else:
                dp_asr = 0.0

            dp_results[ds].append({
                "noise_multiplier": noise_mult,
                "epsilon": dp_out["privacy_epsilon"],
                "delta": dp_out["privacy_delta"],
                "roc_auc": dp_auc,
                "pgd10_asr": dp_asr,
                "accounted_steps": dp_out["accounted_steps"],
                "optimizer_updates": dp_out["optimizer_updates"],
                "sampling_rate": dp_out["sampling_rate"],
            })
            eps_str = f"{dp_out['privacy_epsilon']:.2f}" if dp_out['privacy_epsilon'] != float('inf') else "inf"
            print(f"  [DP-SGD noise={noise_mult:.1f}] eps={eps_str} | ROC-AUC: {dp_auc:.6f} | ASR: {dp_asr*100:.2f}% | Accounted steps: {dp_out['accounted_steps']} | Updates: {dp_out['optimizer_updates']}")

        print("  Training BCL autoencoder on benign training flows...")
        ae = train_autoencoder(X_tr[y_tr == 0], epochs=15)
        torch.save(ae.state_dict(), out_ds_dir / "ae_model.pt")

    print("\n--------------------------------------------------------------------------")
    print("PHASE 2: XAI CROSS-DATASET STABILITY & EVIDENCE SUMMARY")
    print("--------------------------------------------------------------------------")
    xai_stability = compute_cross_dataset_stability(xai_dataset_results, FEATURE_COLS)
    
    xai_summary = {
        "datasets": xai_dataset_results,
        "stability": xai_stability,
        "methodology": "Option C Weighted Attribution (0.7 RF SHAP + 0.3 Robust MLP SHAP)",
        "provenance": "100% Real Benchmark Test Data (No Synthetic Fallbacks)"
    }
    with open(reports_xai_dir / "xai_evidence_summary.json", "w") as f:
        json.dump(xai_summary, f, indent=2)
    print(f"Saved {reports_xai_dir / 'xai_evidence_summary.json'}")

    print("\n--------------------------------------------------------------------------")
    print("PHASE 3: PRIVACY SUMMARY & EVIDENCE SAVE")
    print("--------------------------------------------------------------------------")
    privacy_summary = {
        "data_minimization": DataMinimizationPolicy.verify_schema_minimization(FEATURE_COLS),
        "dp_experiments": dp_results,
        "methodology": "Joint DP-SGD + PGD-7 Adversarial Training on Neural Stream",
        "provenance": "100% Real Benchmark Data (No Synthetic Fallbacks)"
    }
    with open(reports_privacy_dir / "privacy_evidence_summary.json", "w") as f:
        json.dump(privacy_summary, f, indent=2)
    print(f"Saved {reports_privacy_dir / 'privacy_evidence_summary.json'}")

    print("\n--------------------------------------------------------------------------")
    print("PHASE 4: RUNTIME OVERHEAD BENCHMARKING")
    print("--------------------------------------------------------------------------")
    sample_feat_dict = {feat: 1.0 for feat in FEATURE_COLS}
    engine = InferenceEngine(profile="standardized_21", dataset="ToN-IoT", models_base_dir=models_dir)
    
    t0 = time.perf_counter()
    _ = engine.predict(sample_feat_dict, explain=False)
    cold_ms = (time.perf_counter() - t0) * 1000.0

    batch_sizes = [1, 32, 64, 128, 256, 512, 1024]
    detection_benchmarks = {}
    for b_size in batch_sizes:
        batch_in = [sample_feat_dict] * b_size
        latencies = []
        for _ in range(15):
            t_s = time.perf_counter()
            _ = engine.predict(batch_in, explain=False)
            t_e = time.perf_counter()
            latencies.append((t_e - t_s) * 1000.0)
        mean_ms = float(np.mean(latencies))
        per_sample_ms = mean_ms / b_size
        tp = float(b_size / (mean_ms / 1000.0))
        detection_benchmarks[str(b_size)] = {
            "batch_size": b_size,
            "mean_batch_latency_ms": mean_ms,
            "per_sample_latency_ms": per_sample_ms,
            "throughput_samples_per_sec": tp,
        }

    runtime_res = {
        "cold_start_latency_ms": cold_ms,
        "pure_detection_benchmarks": detection_benchmarks,
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }
    with open(reports_runtime_dir / "runtime_overhead_summary.json", "w") as f:
        json.dump(runtime_res, f, indent=2)
    print(f"Saved {reports_runtime_dir / 'runtime_overhead_summary.json'}")

    print("\n--------------------------------------------------------------------------")
    print("PHASE 5: COMPUTE OVERALL DETECTION SUMMARY & MANIFEST")
    print("--------------------------------------------------------------------------")
    rf_aucs = [v["rf_roc_auc"] for v in detection_results.values()]
    std_aucs = [v["std_mlp_roc_auc"] for v in detection_results.values()]
    rob_aucs = [v["robust_mlp_roc_auc"] for v in detection_results.values()]
    opt_c_aucs = [v["option_c_roc_auc"] for v in detection_results.values()]

    rf_mean, rf_var = float(np.mean(rf_aucs)), float(np.var(rf_aucs))
    std_mean, std_var = float(np.mean(std_aucs)), float(np.var(std_aucs))
    rob_mean, rob_var = float(np.mean(rob_aucs)), float(np.var(rob_aucs))
    opt_c_mean, opt_c_var = float(np.mean(opt_c_aucs)), float(np.var(opt_c_aucs))

    manifest_info["detection_summary"] = {
        "by_dataset": detection_results,
        "aggregate": {
            "RF": {"mean_roc_auc": rf_mean, "variance": rf_var},
            "Standard_MLP": {"mean_roc_auc": std_mean, "variance": std_var},
            "Robust_MLP": {"mean_roc_auc": rob_mean, "variance": rob_var},
            "Option_C": {"mean_roc_auc": opt_c_mean, "variance": opt_c_var},
        }
    }
    manifest_info["adversarial_summary"] = adversarial_results
    manifest_info["total_duration_seconds"] = time.time() - t_start

    manifest_path = run_reports_dir / "golden_run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_info, f, indent=2)

    print("\n==========================================================================")
    print(f"GOLDEN PIPELINE COMPLETE in {manifest_info['total_duration_seconds']:.2f} seconds!")
    print(f"Manifest written to: {manifest_path}")
    print("==========================================================================")
    print(f"Option C Mean ROC-AUC: {opt_c_mean:.6f} | Variance: {opt_c_var:.8f}")
    print(f"RF Mean ROC-AUC:       {rf_mean:.6f} | Variance: {rf_var:.8f}")
    print("==========================================================================\n")

if __name__ == "__main__":
    main()
