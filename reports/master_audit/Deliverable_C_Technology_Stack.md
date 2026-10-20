# Deliverable C: Technology Stack Inventory

| Technology | Verifiable Version | Purpose | Actual Usage Location | Status |
|---|---|---|---|---|
| Python | 3.10.11 | Core Runtime | Entire project | Implemented |
| PyTorch | 2.13.0+cpu | Neural Network training & inference | `iot_ids/models/mlp.py` | Implemented |
| Scikit-Learn | >= 1.0 | Random Forest, RobustScaler | `iot_ids/models/rf.py` | Implemented |
| Pandas | Verified | Dataframe manipulation, Parquet reading | `iot_ids/data/` | Implemented |
| NumPy | Verified | Array manipulation, NES perturbation | `scripts/` | Implemented |
| Opacus | Verified | DP-SGD privacy accounting | `iot_ids/models/mlp.py` | Implemented |
| SHAP | Verified | Component-wise feature explanation | `iot_ids/xai/local_xai.py` | Implemented |
| Scapy | MISSING | PCAP parsing, live packet capture | `iot_ids/data/packet_capture.py` | Scaffolded (Tests skipped) |
| Pytest | 9.1.1, 8.4.2 | Testing framework | `tests/` | Implemented |
