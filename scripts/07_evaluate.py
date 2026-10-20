import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from iot_ids.models.ensemble.risk_layer import RiskLayer
from iot_ids.preprocessing.pipeline import PreprocessingPipeline
from iot_ids.utils.paths import REPO_ROOT


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


def eval_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }


def main():
    print("============================================================")
    print("=== STAGE 07: IN-DOMAIN & CROSS-DOMAIN EVALUATION ===")
    print("============================================================\n")

    base_dir = REPO_ROOT / "data" / "processed" / "final"
    models_base = REPO_ROOT / "models" / "final"
    results = {"in_domain": {}, "cross_domain": {}, "ablation": {}}

    datasets = ["Edge-IIoTset", "ToN-IoT"]

    # 1. IN-DOMAIN EVALUATIONS
    for ds in datasets:
        print(f"Executing In-Domain Evaluation for {ds}...")
        ds_dir = base_dir / ds
        target_dir = [d for d in ds_dir.iterdir() if d.is_dir()][0]
        test_df = pd.read_parquet(target_dir / "splits" / "test.parquet")
        
        m_dir = models_base / ds
        prep_indomain = joblib.load(m_dir / "prep_indomain.joblib")
        rf = joblib.load(m_dir / "rf_model.joblib")

        X_test = prep_indomain.transform(test_df)
        y_test = test_df["label"].values

        mlp = MLPModule(input_dim=X_test.shape[1])
        mlp.load_state_dict(torch.load(m_dir / "mlp_model.pt"))
        mlp.eval()

        ae = AutoencoderModule(input_dim=X_test.shape[1], bottleneck_dim=16)
        ae.load_state_dict(torch.load(m_dir / "ae_model.pt"))
        ae.eval()

        risk_layer = RiskLayer.load(m_dir / "risk_layer.json")

        p_rf = rf.predict_proba(X_test)[:, 1]

        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            p_mlp = torch.sigmoid(mlp(X_tensor)).squeeze().numpy()
            recon = ae(X_tensor).numpy()
            ae_mse = np.mean((X_test - recon) ** 2, axis=1)

        p_sup = 0.5 * p_rf + 0.5 * p_mlp
        decisions = risk_layer.predict(p_rf, p_mlp, ae_mse)

        res_sup = eval_metrics(y_test, p_sup)
        results["in_domain"][ds] = {
            "test_rows": len(test_df),
            "supervised_metrics": res_sup,
            "risk_layer_counts": pd.Series([d.risk_state for d in decisions]).value_counts().to_dict()
        }
        print(f"  -> {ds} In-Domain Test F1: {res_sup['f1']*100:.2f}%, Accuracy: {res_sup['accuracy']*100:.2f}%\n")

    # 2. CROSS-DOMAIN EVALUATIONS (Clean 6-Feature F_common ONLY)
    print("--- Executing Cross-Domain Evaluations (Clean 6-Feature F_common ONLY) ---")
    common_cols = [
        "duration", "src_bytes", "proto_tcp", "proto_udp", "proto_icmp", "is_well_known_port"
    ]

    for src_ds, tgt_ds in [("Edge-IIoTset", "ToN-IoT"), ("ToN-IoT", "Edge-IIoTset")]:
        print(f"Cross-Domain: {src_ds} -> {tgt_ds} (Clean 6-Feature F_common model)...")
        
        src_target = [d for d in (base_dir / src_ds).iterdir() if d.is_dir()][0]
        tgt_target = [d for d in (base_dir / tgt_ds).iterdir() if d.is_dir()][0]

        tr_src = pd.read_parquet(src_target / "splits" / "train.parquet")
        tst_tgt = pd.read_parquet(tgt_target / "splits" / "test.parquet")

        prep_common = PreprocessingPipeline(numeric_cols=common_cols)
        X_tr_c = prep_common.fit_transform(tr_src)
        y_tr_src = tr_src["label"].values

        rf_c = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf_c.fit(X_tr_c, y_tr_src)

        mlp_c = MLPModule(input_dim=len(common_cols))
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(mlp_c.parameters(), lr=0.001)

        ds_tr = torch.utils.data.TensorDataset(torch.tensor(X_tr_c, dtype=torch.float32), torch.tensor(y_tr_src, dtype=torch.float32).unsqueeze(1))
        loader = torch.utils.data.DataLoader(ds_tr, batch_size=256, shuffle=True)
        mlp_c.train()
        for epoch in range(10):
            for bx, by in loader:
                optimizer.zero_grad()
                out = mlp_c(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
        mlp_c.eval()

        X_tst_c = prep_common.transform(tst_tgt)
        y_tst_tgt = tst_tgt["label"].values

        p_rf_c = rf_c.predict_proba(X_tst_c)[:, 1]
        with torch.no_grad():
            p_mlp_c = torch.sigmoid(mlp_c(torch.tensor(X_tst_c, dtype=torch.float32))).squeeze().numpy()

        p_sup_c = 0.5 * p_rf_c + 0.5 * p_mlp_c
        cd_res = eval_metrics(y_tst_tgt, p_sup_c)
        results["cross_domain"][f"{src_ds}_to_{tgt_ds}"] = cd_res
        print(f"  -> {src_ds} -> {tgt_ds} F1: {cd_res['f1']*100:.2f}%, Accuracy: {cd_res['accuracy']*100:.2f}%\n")

    # 3. ABLATION STUDY
    print("--- Executing Feature Level Ablation Study ---")
    for ds in datasets:
        ds_target = [d for d in (base_dir / ds).iterdir() if d.is_dir()][0]
        tr_df = pd.read_parquet(ds_target / "splits" / "train.parquet")
        tst_df = pd.read_parquet(ds_target / "splits" / "test.parquet")
        y_tr = tr_df["label"].values
        y_tst = tst_df["label"].values

        ablation_subsets = {
            "A_Static_F_common": common_cols,
            "B_Static_plus_Temporal": common_cols + ["temporal_causal_count", "temporal_causal_rate", "temporal_iat_mean"],
            "C_Static_Temporal_Behavioral": common_cols + ["temporal_causal_count", "temporal_causal_rate", "temporal_iat_mean", "behavioral_dest_diversity", "behavioral_src_activity"],
        }

        results["ablation"][ds] = {}
        for subset_name, cols in ablation_subsets.items():
            prep = PreprocessingPipeline(numeric_cols=cols)
            X_tr = prep.fit_transform(tr_df)
            X_tst = prep.transform(tst_df)

            rf_abl = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
            rf_abl.fit(X_tr, y_tr)
            p_abl = rf_abl.predict_proba(X_tst)[:, 1]

            results["ablation"][ds][subset_name] = eval_metrics(y_tst, p_abl)

    out_file = REPO_ROOT / "reports" / "tables" / "final_evaluation_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nEvaluation Results Saved to: {out_file}")
    print("Stage 07 Complete: In-domain and cross-domain evaluations finished.")

if __name__ == "__main__":
    main()
