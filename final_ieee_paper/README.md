# Final IEEE Publication Manuscript Package

This directory contains the camera-ready manuscript source, bibliography, and figures for publication:

- **`main.tex`**: Full publication manuscript formatted in IEEE Transactions on Dependable and Secure Computing (TDSC) style:
  *"An Adversarially Robust and Privacy-Preserving AI Framework for Explainable Intrusion Detection in IoT Environments: Empirical Characterization of Neural Seed Sensitivity, Ensemble Threat Models, and Operational Realities"*
- **`references.bib`**: Complete BibTeX bibliography with full citations.
- **`figures/`**: Publication-quality figures:
  - `system_overview.png` / `system_overview.pdf`: Overall hybrid framework architecture.
  - `multiseed_robustness.png` / `multiseed_robustness.pdf`: Robust MLP seed sensitivity and checkpoint selection across 10 seeds.
  - `blackbox_query_attack.png` / `blackbox_query_attack.pdf`: Neural-gradient transfer vs. black-box Natural Evolution Strategies (NES) query attacks.
  - `fusion_weight_tradeoff.png` / `fusion_weight_tradeoff.pdf`: Clean ROC-AUC vs. transfer PGD-10 ASR across fusion weights $w_{\text{RF}} \in [0.0, 1.0]$.
  - `detection_performance.png`: Detection ROC-AUC across component models and Option C.
  - `adversarial_asr.png`: PGD-10 Attack Success Rate comparison.
  - `runtime_throughput.png`: Host classifier-only throughput scaling across batch sizes $N \in [1, 1024]$.

### Verified Empirical Foundation
All reported numerical results are derived directly from the audited, reproducible ledgers:
- Mean Clean ROC-AUC: **0.9970** across Edge-IIoTset, NF-ToN-IoT-v2, ToN-IoT, and CICIoT2023.
- Multi-seed ASR: NF-ToN-IoT-v2 across seeds 42--51 ranges from **0.7% to 47.5%** (mean 12.36%, std 17.91%).
- Checkpoint selection: Heuristic selects seed 43 (**1.20% test ASR**) on NF-ToN-IoT-v2, but does not generalize to Edge-IIoTset (**10.70% selected vs. 8.39% baseline mean**).
- Black-box query attack: Direct NES score-query evasion reaches **36.07% strict ASR** and **83.90% operational ASR** ($\tau=0.71$) under 500 queries on NF-ToN-IoT-v2.
- Differential Privacy: Audited $(\varepsilon=2.37, \delta=10^{-5})$ strictly scoped to the neural branch weights.
- Explainability: Weighted Component Attribution Aggregation ($0.7\,\text{RF} + 0.3\,\text{MLP}$) validated as a low-latency surrogate against exact Shapley evaluation ($3.53$\,s/sample).
- Throughput: Classifier CPU inference achieves **16,504.7 samples/sec** ($0.0606$\,ms/sample at $N=1024$).
- Operational software: 174 passing automated tests; Scapy packet capture, bidirectional five-tuple flow aggregation, daemon, and Docker support.
