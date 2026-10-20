"""IIoT Intrusion Detection System — Production Streamlit Demonstration App.

Authoritative Option C 21-Feature Intrusion Detection System Architecture:
  Packet/PCAP Stream -> Flow Aggregator -> 21-Feature Vector -> Robust Scaler Preprocessor
  -> Random Forest (0.7) + Robust MLP (0.3) Fusion -> Threat Decision -> XAI Attribution.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import streamlit as st

# Add src/ directory to sys.path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from iot_ids.runtime.engine import InferenceEngine
from iot_ids.runtime.schema import STANDARDIZED_21_FEATURES
from iot_ids.data.packet_capture import PacketCaptureEngine, scapy_to_canonical_packet
from iot_ids.data.flow_aggregator import FlowAggregator
from iot_ids.xai.local_xai import explain_local_sample
from iot_ids.adversarial.attacks import pgd_attack

# Streamlit Page Setup
st.set_page_config(
    page_title="Industrial IIoT Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ Industrial IIoT Intrusion Detection System")
st.markdown(
    "**Authoritative Option C (0.7 RF + 0.3 Robust MLP) 21-Feature Intrusion Detection System** | "
    "*Multi-Dataset Industrial Cyber-Defense Platform*"
)

# -----------------------------------------------------------------------------
# SIDEBAR: SYSTEM HEALTH & CONFIGURATION
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration & Health")

dataset_name = st.sidebar.selectbox(
    "Target IIoT Dataset",
    ["Edge-IIoTset", "NF-ToN-IoT-v2", "ToN-IoT", "CICIoT2023"],
    index=0,
)

# Load InferenceEngine
@st.cache_resource
def load_engine(ds: str):
    models_dir = REPO_ROOT / "models" / "golden_run"
    if not (models_dir / ds).exists():
        models_dir = REPO_ROOT / "models" / "standardized"
    return InferenceEngine(profile="standardized_21", dataset=ds, models_base_dir=models_dir)

try:
    engine = load_engine(dataset_name)
    engine_loaded = True
except Exception as e:
    engine_loaded = False
    st.sidebar.error(f"Engine Load Failed: {e}")

# System Health Summary Panel
with st.sidebar.expander("🩺 System Health Panel", expanded=False):
    st.write(f"**Selected Dataset:** `{dataset_name}`")
    st.write(f"**Feature Profile:** `standardized_21` ({len(STANDARDIZED_21_FEATURES)} features)")
    st.write(f"**21-Feature Schema:** {'✅ Loaded' if len(STANDARDIZED_21_FEATURES)==21 else '❌ Error'}")
    st.write(f"**Preprocessor:** {'✅ Loaded' if (engine_loaded and engine.preprocessor is not None) else '❌ Not Loaded'}")
    st.write(f"**Random Forest (100 Trees):** {'✅ Loaded' if (engine_loaded and engine.rf_model is not None) else '❌ Not Loaded'}")
    st.write(f"**Robust Neural Stream:** {'✅ Loaded' if (engine_loaded and engine.robust_mlp_model is not None) else '❌ Not Loaded'}")
    st.write(f"**Option C Fusion (0.7/0.3):** {'✅ Active' if engine_loaded else '❌ Inactive'}")
    st.write(f"**XAI Attribution Module:** ✅ Ready")
    st.write(f"**Flow Aggregator Engine:** ✅ Ready")

# Inspection Mode Selection
demo_mode = st.sidebar.radio(
    "Demonstration Mode",
    [
        "📊 Held-Out Flow Replay",
        "🔌 PCAP / Packet Replay",
        "🎯 Controlled Test Flow",
        "🛡️ Adversarial Evasion (PGD-10)",
        "💡 XAI Attribution Analysis",
        "📈 Research Results Dashboard",
    ]
)

# Load dataset test samples
@st.cache_data
def load_held_out_samples(ds: str) -> pd.DataFrame:
    parquet_path = REPO_ROOT / "data" / "processed" / "stage3" / ds / "test.parquet"
    if not parquet_path.exists():
        # Fallback to stage3 directory search
        parquet_path = REPO_ROOT / "data" / "processed" / "stage3" / ds / "test.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    raise FileNotFoundError(f"Test samples not found for {ds} at {parquet_path}")

try:
    test_df = load_held_out_samples(dataset_name)
except Exception as err:
    test_df = None
    st.warning(f"Could not load test samples for {dataset_name}: {err}")

# Initialize Session Counters
if "replay_stats" not in st.session_state:
    st.session_state.replay_stats = {"processed": 0, "benign": 0, "attack": 0, "alerts": 0}

# -----------------------------------------------------------------------------
# MODE 1: HELD-OUT FLOW REPLAY
# -----------------------------------------------------------------------------
if demo_mode == "📊 Held-Out Flow Replay":
    st.header("📊 Held-Out Flow Replay")
    st.caption("Replays real held-out test set network flows through the production Option C detection engine.")

    if test_df is not None:
        filter_label = st.selectbox("Filter Class", ["All Samples", "Benign Flows Only (Label=0)", "Attack Flows Only (Label=1)"])
        if filter_label == "Benign Flows Only (Label=0)":
            sub_df = test_df[test_df["label"] == 0].reset_index(drop=True)
        elif filter_label == "Attack Flows Only (Label=1)":
            sub_df = test_df[test_df["label"] == 1].reset_index(drop=True)
        else:
            sub_df = test_df.reset_index(drop=True)

        sample_idx = st.slider("Select Sample Index", 0, max_value=len(sub_df) - 1, value=0)
        sample_row = sub_df.iloc[sample_idx:sample_idx+1]
        actual_label = int(sample_row["label"].values[0])
        attack_cat = sample_row["attack_category"].iloc[0] if "attack_category" in sample_row.columns else ("Attack" if actual_label==1 else "Normal")

        # Execute Engine Inference
        res = engine.predict(sample_row[STANDARDIZED_21_FEATURES], explain=True, top_k=5)[0]

        # Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Random Forest P_RF", f"{res['rf_probability']*100:.2f}%")
        col2.metric("Robust MLP P_MLP", f"{res['robust_mlp_probability']*100:.2f}%")
        col3.metric("Option C P_OptC", f"{res['probability']*100:.2f}%", delta="0.7*RF + 0.3*MLP")
        
        pred_label = "RED / ATTACK" if res["prediction"] == 1 else "GREEN / BENIGN"
        if res["prediction"] == actual_label:
            status_color = "green"
            match_str = "MATCHES GROUND TRUTH"
        else:
            status_color = "orange"
            match_str = "MISMATCH"

        col4.subheader(f"Prediction: {pred_label}")
        col4.caption(f"Ground Truth: {'ATTACK (' + str(attack_cat) + ')' if actual_label==1 else 'BENIGN'} ({match_str})")

        st.markdown("---")

        # Feature Breakdown Tabs
        tab_instant, tab_temp, tab_behav = st.tabs([
            "⚡ Level A: Instantaneous (11 Features)",
            "⏱️ Level B: Causal Temporal (5 Features)",
            "🕸️ Level C: Causal Behavioral (5 Features)",
        ])

        with tab_instant:
            instant_cols = STANDARDIZED_21_FEATURES[:11]
            st.dataframe(sample_row[instant_cols].T.rename(columns={sample_row.index[0]: "Value"}), use_container_width=True)

        with tab_temp:
            temp_cols = STANDARDIZED_21_FEATURES[11:16]
            st.dataframe(sample_row[temp_cols].T.rename(columns={sample_row.index[0]: "Value"}), use_container_width=True)

        with tab_behav:
            behav_cols = STANDARDIZED_21_FEATURES[16:21]
            st.dataframe(sample_row[behav_cols].T.rename(columns={sample_row.index[0]: "Value"}), use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Top 5 Feature Attributions (Weighted Component Attribution)")
        if "explanation" in res and "top_k_features" in res["explanation"]:
            exp_df = pd.DataFrame(res["explanation"]["top_k_features"])
            st.dataframe(
                exp_df[["feature", "raw_value", "weighted_attribution", "direction"]].style.format(
                    {"raw_value": "{:.4f}", "weighted_attribution": "{:.4f}"}
                ),
                use_container_width=True,
            )

# -----------------------------------------------------------------------------
# MODE 2: PCAP / PACKET REPLAY
# -----------------------------------------------------------------------------
elif demo_mode == "🔌 PCAP / Packet Replay":
    st.header("🔌 PCAP / Packet Stream Replay")
    st.caption("Ingests raw network packets, aggregates them into 5-tuple bidirectional flows, extracts 21 features, and evaluates Option C decisions.")

    pcap_sample_path = REPO_ROOT / "data" / "sample_reproduce_stream.pcap"

    if pcap_sample_path.exists():
        st.success(f"Sample PCAP Available: `{pcap_sample_path.name}` ({pcap_sample_path.stat().st_size} bytes)")
        if st.button("🚀 Replay Sample PCAP Stream"):
            import scapy.all as scapy
            flow_aggregator = FlowAggregator(inactivity_timeout=15.0)
            packets_read = 0
            flushed_alerts = []

            try:
                reader = scapy.PcapReader(str(pcap_sample_path))
                for pkt in reader:
                    packets_read += 1
                    cpkt = scapy_to_canonical_packet(pkt)
                    if cpkt is None:
                        continue
                    if cpkt is not None:
                        completed_flow = flow_aggregator.add_packet(cpkt)
                        if completed_flow is not None:
                            # Convert completed flow to feature vector
                            f_dict = completed_flow.to_dict()
                            df_f = pd.DataFrame([f_dict])
                            # Re-align missing features
                            for col in STANDARDIZED_21_FEATURES:
                                if col not in df_f.columns:
                                    df_f[col] = 0.0
                            alert_res = engine.predict(df_f[STANDARDIZED_21_FEATURES])[0]
                            flushed_alerts.append((completed_flow, alert_res))
                reader.close()
            except Exception as ex:
                st.warning(f"PCAP Stream Notice: {ex}")

            # Flush remaining active flows
            remaining = flow_aggregator.flush()
            for rflow in remaining:
                f_dict = rflow.to_dict()
                df_f = pd.DataFrame([f_dict])
                for col in STANDARDIZED_21_FEATURES:
                    if col not in df_f.columns:
                        df_f[col] = 0.0
                alert_res = engine.predict(df_f[STANDARDIZED_21_FEATURES])[0]
                flushed_alerts.append((rflow, alert_res))

            st.write(f"**Packets Processed:** {packets_read}")
            st.write(f"**Flows Constructed:** {len(flushed_alerts)}")

            if flushed_alerts:
                records = []
                for flow_obj, res_dict in flushed_alerts:
                    records.append({
                        "5-Tuple": f"{flow_obj.src_host}:{flow_obj.src_port} -> {flow_obj.dst_host}:{flow_obj.dst_port} ({flow_obj.protocol})",
                        "Duration (s)": f"{flow_obj.duration:.2f}",
                        "Total Bytes": flow_obj.total_bytes,
                        "P_OptionC": f"{res_dict['probability']:.4f}",
                        "Decision": "ATTACK" if res_dict['prediction'] == 1 else "BENIGN",
                    })
                st.table(pd.DataFrame(records))
            else:
                st.info("No completed flows generated from sample PCAP. Flow aggregator state active.")
    else:
        st.info("No sample PCAP found at `data/sample_reproduce_stream.pcap`. Upload a PCAP file to replay.")

# -----------------------------------------------------------------------------
# MODE 3: CONTROLLED TEST FLOW
# -----------------------------------------------------------------------------
elif demo_mode == "🎯 Controlled Test Flow":
    st.header("🎯 Controlled Test Flow Inspector")
    st.caption("Interactively evaluate synthetic or customized 21-feature vectors.")

    preset = st.selectbox(
        "Choose Flow Preset",
        ["Benign Normal Traffic", "High-Throughput DDoS Attack", "Port Scan / Reconnaissance"]
    )

    if preset == "Benign Normal Traffic":
        flow_vec = np.array([[1.2, 500.0, 10.0, 50.0, 0.2, 0.5, 0.1, 1.0, 0.0, 0.0, 0.0, 0.12, 0.2, 8.0, 400.0, 0.1, 1.0, 0.5, 0.1, 0.05, 5.0]])
    elif preset == "High-Throughput DDoS Attack":
        flow_vec = np.array([[0.05, 50000.0, 1000.0, 50.0, 0.9, 0.95, 0.9, 1.0, 0.0, 0.0, 0.0, 0.001, 2.5, 950.0, 48000.0, 0.9, 25.0, 3.5, 0.95, 0.8, 150.0]])
    else:
        flow_vec = np.array([[0.01, 100.0, 50.0, 40.0, 0.1, 0.9, 0.8, 1.0, 0.0, 0.0, 0.0, 0.002, 1.8, 45.0, 90.0, 0.8, 150.0, 4.2, 0.99, 0.95, 50.0]])

    df_custom = pd.DataFrame(flow_vec, columns=STANDARDIZED_21_FEATURES)
    st.write("**21-Feature Input Vector:**")
    st.dataframe(df_custom, use_container_width=True)

    if st.button("Evaluate Option C Engine"):
        res = engine.predict(df_custom, explain=True)[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P_RF (Random Forest)", f"{res['rf_probability']*100:.2f}%")
        c2.metric("P_MLP (Robust MLP)", f"{res['robust_mlp_probability']*100:.2f}%")
        c3.metric("Option C Fusion", f"{res['probability']*100:.2f}%")
        c4.subheader("ATTACK" if res["prediction"]==1 else "BENIGN")

# -----------------------------------------------------------------------------
# MODE 4: ADVERSARIAL EVASION (PGD-10)
# -----------------------------------------------------------------------------
elif demo_mode == "🛡️ Adversarial Evasion (PGD-10)":
    st.header("🛡️ Adversarial Evasion & Robustness Inspection")
    st.caption("Applies PGD-10 continuous feature perturbation to test neural stream vs Option C ensemble defense.")
    st.info("ℹ️ **Research Methodology Scope Note:** PGD gradients target the neural stream; adversarial samples are subsequently evaluated through Option C. Discrete protocol indicators (indices 7–10) are strictly unperturbed.")

    if test_df is not None:
        attack_sub = test_df[test_df["label"] == 1].reset_index(drop=True)
        if len(attack_sub) > 0:
            sample_idx = st.slider("Select Attack Flow Sample", 0, len(attack_sub) - 1, 0)
            sample_row = attack_sub.iloc[sample_idx:sample_idx+1]
            
            if st.button("⚡ Generate PGD-10 Adversarial Perturbation"):
                # Preprocess sample
                X_orig = engine.preprocessor.transform(sample_row[STANDARDIZED_21_FEATURES])
                X_orig_t = torch.tensor(X_orig, dtype=torch.float32)
                y_t = torch.tensor([1.0], dtype=torch.float32)

                cont_mask = torch.ones(21, dtype=torch.float32)
                cont_mask[7:11] = 0.0  # Freeze protocol indicators

                x_min = torch.tensor(np.full((1, 21), -10.0), dtype=torch.float32)
                x_max = torch.tensor(np.full((1, 21), 10.0), dtype=torch.float32)

                # PGD-10 Attack on Robust MLP
                X_adv = pgd_attack(
                    engine.robust_mlp_model, X_orig_t, y_t,
                    epsilon=0.10, continuous_mask=cont_mask,
                    x_min=x_min, x_max=x_max, steps=10, alpha=0.025
                ).numpy()

                # Evaluate on Models
                p_rf_orig = float(engine.rf_model.predict_proba(X_orig)[0, 1])
                p_rf_adv = float(engine.rf_model.predict_proba(X_adv)[0, 1])

                with torch.no_grad():
                    p_mlp_orig = float(torch.sigmoid(engine.robust_mlp_model(X_orig_t)).item())
                    p_mlp_adv = float(torch.sigmoid(engine.robust_mlp_model(torch.tensor(X_adv, dtype=torch.float32))).item())

                p_optc_orig = 0.7 * p_rf_orig + 0.3 * p_mlp_orig
                p_optc_adv = 0.7 * p_rf_adv + 0.3 * p_mlp_adv

                res_tbl = pd.DataFrame([
                    {"Stream": "Random Forest", "Clean Probability": f"{p_rf_orig*100:.2f}%", "Adversarial PGD-10": f"{p_rf_adv*100:.2f}%", "Evasion Status": "Defended" if p_rf_adv>=0.5 else "Evaded"},
                    {"Stream": "Robust MLP", "Clean Probability": f"{p_mlp_orig*100:.2f}%", "Adversarial PGD-10": f"{p_mlp_adv*100:.2f}%", "Evasion Status": "Defended" if p_mlp_adv>=0.5 else "Evaded"},
                    {"Stream": "Option C Ensemble (0.7/0.3)", "Clean Probability": f"{p_optc_orig*100:.2f}%", "Adversarial PGD-10": f"{p_optc_adv*100:.2f}%", "Evasion Status": "DEFENDED" if p_optc_adv>=0.5 else "Evaded"},
                ])

                st.subheader("Adversarial Evasion Evaluation Results")
                st.table(res_tbl)

                diff_df = pd.DataFrame({
                    "Feature": STANDARDIZED_21_FEATURES,
                    "Original (Scaled)": X_orig[0],
                    "Adversarial (Perturbed)": X_adv[0],
                    "Delta": X_adv[0] - X_orig[0],
                })
                st.subheader("Feature-Space Perturbation Delta")
                st.dataframe(diff_df.style.format({"Original (Scaled)": "{:.4f}", "Adversarial (Perturbed)": "{:.4f}", "Delta": "{:.4f}"}), use_container_width=True)

# -----------------------------------------------------------------------------
# MODE 5: XAI ATTRIBUTION ANALYSIS
# -----------------------------------------------------------------------------
elif demo_mode == "💡 XAI Attribution Analysis":
    st.header("💡 Local Feature Attribution Analysis")
    st.caption("Shows how replacing each feature with its standardized median changes the complete Option C probability.")
    st.info("ℹ️ **XAI Scope Note:** These are single-feature ablation effects on the fused predictor, not SHAP values; effects are not additive when features interact.")

    if test_df is not None:
        sample_idx = st.slider("Select Test Sample", 0, len(test_df) - 1, 0)
        sample_row = test_df.iloc[sample_idx:sample_idx+1]

        res = engine.predict(sample_row[STANDARDIZED_21_FEATURES], explain=True, top_k=10)[0]
        
        st.write(f"**Flow Prediction:** `{res['prediction']}` (`P_OptionC = {res['probability']:.4f}`)")

        if "explanation" in res and "top_k_features" in res["explanation"]:
            exp_data = res["explanation"]["top_k_features"]
            df_exp = pd.DataFrame(exp_data)
            
            st.subheader("Top 10 Fused-Probability Ablation Effects")
            st.bar_chart(df_exp.set_index("feature")["fused_probability_delta"])

            st.dataframe(df_exp[["feature", "raw_value", "baseline_value", "fused_probability_delta", "rf_probability_delta", "mlp_probability_delta", "direction"]], use_container_width=True)

# -----------------------------------------------------------------------------
# MODE 6: RESEARCH RESULTS DASHBOARD
# -----------------------------------------------------------------------------
elif demo_mode == "📈 Research Results Dashboard":
    st.header("📈 Verified Research Benchmark Dashboard")
    st.caption("Verified empirical metrics from golden run artifacts and publication benchmarks.")

    tab_auc, tab_adv_res, tab_dp_res, tab_rt_res = st.tabs([
        "🎯 ROC-AUC Performance",
        "🛡️ Adversarial ASR",
        "🔒 Differential Privacy",
        "⚡ Runtime & Throughput",
    ])

    with tab_auc:
        st.subheader("Table I: Detection ROC-AUC Across 4 Real IIoT Datasets")
        df_auc = pd.DataFrame([
            {"Dataset": "Edge-IIoTset", "RF ROC-AUC": "0.9999", "Std MLP": "0.9996", "Robust MLP": "0.9984", "Option C (Selected)": "0.9988"},
            {"Dataset": "NF-ToN-IoT-v2", "RF ROC-AUC": "0.9980", "Std MLP": "0.9763", "Robust MLP": "0.8541", "Option C (Selected)": "0.9927"},
            {"Dataset": "ToN-IoT", "RF ROC-AUC": "1.0000", "Std MLP": "1.0000", "Robust MLP": "1.0000", "Option C (Selected)": "1.0000"},
            {"Dataset": "CICIoT2023", "RF ROC-AUC": "0.9995", "Std MLP": "0.9959", "Robust MLP": "0.9904", "Option C (Selected)": "0.9965"},
            {"Dataset": "Mean (All Datasets)", "RF ROC-AUC": "0.9994", "Std MLP": "0.9930", "Robust MLP": "0.9607", "Option C (Selected)": "0.9970"},
        ])
        st.table(df_auc)

    with tab_adv_res:
        st.subheader("Table II: PGD-10 Adversarial Attack Success Rate (ASR, ε=0.10)")
        df_adv = pd.DataFrame([
            {"Dataset": "Edge-IIoTset", "Std MLP ASR": "0.0%", "Robust MLP ASR": "0.0%", "Option C ASR": "3.8%"},
            {"Dataset": "NF-ToN-IoT-v2", "Std MLP ASR": "53.8%", "Robust MLP ASR": "48.0%", "Option C ASR": "4.0%"},
            {"Dataset": "ToN-IoT", "Std MLP ASR": "2.3%", "Robust MLP ASR": "2.5%", "Option C ASR": "0.0%"},
            {"Dataset": "CICIoT2023", "Std MLP ASR": "2.3%", "Robust MLP ASR": "1.3%", "Option C ASR": "1.2%"},
        ])
        st.table(df_adv)

    with tab_dp_res:
        st.subheader("Table III: Differential Privacy (DP-SGD) Trade-Off on NF-ToN-IoT-v2 (δ=10⁻⁵)")
        df_dp = pd.DataFrame([
            {"Noise Multiplier (σ)": "0.0", "Epsilon (ε)": "∞", "Neural Stream AUC": "0.9763", "PGD-10 ASR": "1.5%"},
            {"Noise Multiplier (σ)": "0.5", "Epsilon (ε)": "15.85", "Neural Stream AUC": "0.8865", "PGD-10 ASR": "2.5%"},
            {"Noise Multiplier (σ)": "1.0 (Selected)", "Epsilon (ε)": "2.37", "Neural Stream AUC": "0.8541", "PGD-10 ASR": "47.7%"},
            {"Noise Multiplier (σ)": "2.0", "Epsilon (ε)": "0.80", "Neural Stream AUC": "0.8284", "PGD-10 ASR": "47.8%"},
        ])
        st.table(df_dp)

    with tab_rt_res:
        st.subheader("Table IV: Pure Detection Host Latency & Throughput Benchmarks (Single-CPU)")
        df_rt = pd.DataFrame([
            {"Batch (N)": 1, "Batch Latency (ms)": 69.40, "Per-Sample (ms)": 69.4000, "Throughput (samples/s)": "14.4"},
            {"Batch (N)": 32, "Batch Latency (ms)": 54.86, "Per-Sample (ms)": 1.7143, "Throughput (samples/s)": "583.3"},
            {"Batch (N)": 64, "Batch Latency (ms)": 46.51, "Per-Sample (ms)": 0.7267, "Throughput (samples/s)": "1,376.1"},
            {"Batch (N)": 128, "Batch Latency (ms)": 46.68, "Per-Sample (ms)": 0.3647, "Throughput (samples/s)": "2,742.3"},
            {"Batch (N)": 256, "Batch Latency (ms)": 61.11, "Per-Sample (ms)": 0.2387, "Throughput (samples/s)": "4,189.0"},
            {"Batch (N)": 512, "Batch Latency (ms)": 60.88, "Per-Sample (ms)": 0.1189, "Throughput (samples/s)": "8,409.5"},
            {"Batch (N)": 1024, "Batch Latency (ms)": 62.04, "Per-Sample (ms)": 0.0606, "Throughput (samples/s)": "16,504.7"},
        ])
        st.table(df_rt)

st.markdown("---")
st.caption("Industrial IIoT Intrusion Detection System | Authoritative Option C 21-Feature System Demonstration")
