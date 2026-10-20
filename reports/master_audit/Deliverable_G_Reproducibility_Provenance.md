# Deliverable G: Reproducibility and Provenance Report

## 1. Traceability
- **Data Provenance**: The Parquet dataset files correspond to the reported 21-feature schema. Chronological splits (Stage 3) are deterministic and mathematically sound.
- **Model Checkpoints**: The `models/golden_run/` checkpoints correspond exactly to the reported AUCs. Seed 42 was consistently used.
- **Adversarial Results**: P5 black-box NES attacks generate reproducible random seeds and strictly track budgets, successfully exposing vulnerabilities.

## 2. Irreproducibility and Gaps
- **End-to-End Latency**: Cannot be reproduced end-to-end because Scapy is absent and the benchmark script explicitly skips the network ingestion phase.
- **Legacy Reports**: Old markdown reports (e.g., uncorrected P4/P5 transfer ASRs) contain arithmetic denominator errors (failing to exclude initially misclassified samples). These were superseded by the `claim_upgrade` audits.
