# CURRENT SYSTEM ARCHITECTURE

## A. Frontend
- **Framework**: Not applicable / Prototype only (demo/app.py).
- **Major pages**: Basic Gradio demo.
- **Components**: Basic interface for uploading PCAPs.
- **User Workflows**: Upload PCAP -> View JSON alerts.
- **Data Displayed**: Probability scores from Option C.
- **Status**: EXPERIMENTAL / NOT FULLY INTEGRATED. The frontend is a local script, not connected to a scalable backend API.

## B. Backend/API
- **Framework**: Python CLI (src/iot_ids/cli.py). No true REST API (no FastAPI/Django).
- **Flow**: Command line execution only.
- **Authentication**: None.
- **Database**: None. Data is loaded from local Parquet files.
- **Deployment**: Dockerfile and docker-compose exist but simply wrap the CLI daemon.

## Simple Architecture Explanation
User -> CLI/Gradio Demo -> Packet Capture (Scapy) -> Flow Aggregator -> IDSPredictor -> RF + Robust MLP -> Fusion -> JSONL/Console Alerts
