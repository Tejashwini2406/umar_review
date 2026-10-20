import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from iot_ids.preprocessing.pipeline import PreprocessingPipeline
from iot_ids.experiments.preprocessing import FeaturePreprocessor
from iot_ids.features.canonical.flow_builder import CANONICAL_18_FEATURE_NAMES
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


def train_mlp(X: np.ndarray, y: np.ndarray, epochs: int = 15, batch_size: int = 256) -> MLPModule:
    model = MLPModule(input_dim=X.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    return model


def train_autoencoder(X_benign: np.ndarray, epochs: int = 15, batch_size: int = 256) -> AutoencoderModule:
    model = AutoencoderModule(input_dim=X_benign.shape[1], bottleneck_dim=16)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    ds = TensorDataset(torch.tensor(X_benign, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for (bx,) in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, bx)
            loss.backward()
            optimizer.step()

    return model


def main():
    parser = argparse.ArgumentParser(description="Stage 05: Model Training (RF, MLP, Autoencoder)")
    parser.add_argument(
        "--profile",
        type=str,
        choices=["historical_13", "standardized_21"],
        default="historical_13",
        help="Feature profile selection: 'historical_13' (default) or 'standardized_21'",
    )
    args = parser.parse_args()

    print("============================================================")
    print(f"=== STAGE 05: MODEL TRAINING [PROFILE: {args.profile}] ===")
    print("============================================================\n")

    if args.profile == "historical_13":
        base_dir = REPO_ROOT / "data" / "processed" / "final"
        models_base = REPO_ROOT / "models" / "final"
        datasets = ["Edge-IIoTset", "ToN-IoT"]

        for ds in datasets:
            print(f"--- Training Component Models for {ds} (historical_13) ---")
            ds_dir = base_dir / ds
            fingerprint_dirs = [d for d in ds_dir.iterdir() if d.is_dir()]
            target_dir = fingerprint_dirs[0]
            splits_dir = target_dir / "splits"

            train_df = pd.read_parquet(splits_dir / "train.parquet")
            out_model_dir = models_base / ds
            out_model_dir.mkdir(parents=True, exist_ok=True)

            common_cols = [
                "duration", "src_bytes", "proto_tcp", "proto_udp", "proto_icmp", "is_well_known_port"
            ]
            
            if ds == "Edge-IIoTset":
                indomain_cols = common_cols + [
                    "temporal_causal_count", "temporal_causal_rate", "temporal_iat_mean",
                    "behavioral_dest_diversity", "behavioral_src_activity",
                    "mqtt_msgtype", "mbtcp_unit_id"
                ]
            else:
                indomain_cols = common_cols + [
                    "temporal_causal_count", "temporal_causal_rate", "temporal_iat_mean",
                    "behavioral_dest_diversity", "behavioral_src_activity",
                    "conn_state_encoded", "http_method_encoded"
                ]

            prep_common = PreprocessingPipeline(numeric_cols=common_cols)
            X_tr_common = prep_common.fit_transform(train_df)

            prep_indomain = PreprocessingPipeline(numeric_cols=indomain_cols)
            X_tr_indomain = prep_indomain.fit_transform(train_df)
            y_train = train_df["label"].values

            joblib.dump(prep_common, out_model_dir / "prep_common.joblib")
            joblib.dump(prep_indomain, out_model_dir / "prep_indomain.joblib")

            print("  Training Random Forest (100 trees, max depth 15)...")
            rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
            rf.fit(X_tr_indomain, y_train)
            joblib.dump(rf, out_model_dir / "rf_model.joblib")

            print("  Training MLP Classifier...")
            mlp = train_mlp(X_tr_indomain, y_train, epochs=15, batch_size=256)
            torch.save(mlp.state_dict(), out_model_dir / "mlp_model.pt")

            print("  Training Autoencoder on Benign Samples...")
            ben_mask = (y_train == 0)
            X_tr_benign = X_tr_indomain[ben_mask]
            
            ae = train_autoencoder(X_tr_benign, epochs=15, batch_size=256)
            torch.save(ae.state_dict(), out_model_dir / "ae_model.pt")

            print(f"Component models saved to {out_model_dir}\n")

    elif args.profile == "standardized_21":
        base_dir = REPO_ROOT / "data" / "processed" / "stage3"
        models_base = REPO_ROOT / "models" / "standardized"
        datasets = ["ToN-IoT", "Edge-IIoTset", "NF-ToN-IoT-v2", "CICIoT2023"]

        feature_names = list(CANONICAL_18_FEATURE_NAMES)
        
        # Mandatory Validation Check 1: Feature count
        if len(feature_names) != 21:
            raise RuntimeError(f"Expected 21 canonical features, but got {len(feature_names)}")

        # Mandatory Validation Check 2: No duplicate features
        if len(feature_names) != len(set(feature_names)):
            raise RuntimeError(f"Duplicate feature names detected in canonical feature list: {feature_names}")

        for ds in datasets:
            print(f"--- Training Component Models for {ds} (standardized_21) ---")
            ds_dir = base_dir / ds
            train_path = ds_dir / "train.parquet"
            val_path = ds_dir / "val.parquet"
            test_path = ds_dir / "test.parquet"

            if not train_path.exists():
                raise RuntimeError(f"Dataset '{ds}': Missing train.parquet at {train_path}")

            train_df = pd.read_parquet(train_path)
            val_df = pd.read_parquet(val_path) if val_path.exists() else None
            test_df = pd.read_parquet(test_path) if test_path.exists() else None

            # Mandatory Validation Check 3: Missing columns check
            missing_cols = [c for c in feature_names if c not in train_df.columns]
            if missing_cols:
                raise RuntimeError(f"Dataset '{ds}': Missing required 21 feature columns: {missing_cols}")

            if "label" not in train_df.columns:
                raise RuntimeError(f"Dataset '{ds}': Missing required 'label' column in train.parquet")

            out_model_dir = models_base / ds
            out_model_dir.mkdir(parents=True, exist_ok=True)

            prep = FeaturePreprocessor(feature_names=feature_names)
            X_tr = prep.fit_transform(train_df)
            y_train = train_df["label"].values

            # Mandatory Validation Check 4: Transformed dimension checks
            if X_tr.shape[1] != 21:
                raise RuntimeError(f"Dataset '{ds}': Transformed X_train dimension is {X_tr.shape[1]}, expected 21")

            if val_df is not None:
                X_val = prep.transform(val_df)
                if X_val.shape[1] != 21:
                    raise RuntimeError(f"Dataset '{ds}': Transformed X_val dimension is {X_val.shape[1]}, expected 21")

            if test_df is not None:
                X_test = prep.transform(test_df)
                if X_test.shape[1] != 21:
                    raise RuntimeError(f"Dataset '{ds}': Transformed X_test dimension is {X_test.shape[1]}, expected 21")

            joblib.dump(prep, out_model_dir / "prep_standardized.joblib")

            print("  Training Random Forest (100 trees, max depth 15)...")
            rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_train)
            joblib.dump(rf, out_model_dir / "rf_model.joblib")

            print("  Training MLP Classifier (input_dim=21)...")
            mlp = train_mlp(X_tr, y_train, epochs=15, batch_size=256)
            torch.save(mlp.state_dict(), out_model_dir / "mlp_model.pt")

            print("  Training Autoencoder on Benign Samples (input_dim=21)...")
            ben_mask = (y_train == 0)
            X_tr_benign = X_tr[ben_mask]
            
            ae = train_autoencoder(X_tr_benign, epochs=15, batch_size=256)
            torch.save(ae.state_dict(), out_model_dir / "ae_model.pt")

            print(f"Standardized component models saved to {out_model_dir}\n")

    print(f"Stage 05 Complete: All component models for profile '{args.profile}' trained successfully.")


if __name__ == "__main__":
    main()

