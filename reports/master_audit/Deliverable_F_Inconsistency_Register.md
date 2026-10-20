# Deliverable F: Comprehensive Inconsistency & Difficulty Register

| ID | Title | Severity | Category | What Evidence Shows | Required Action / Closure Criteria |
|---|---|---|---|---|---|
| 001 | Missing PCAP/Scapy Dependencies | P0 (Blocking for E2E) | Deployment | `test_pcap_and_live_ingress_e2e.py` skips entirely due to `ImportError: No module named 'scapy'`. | Install Scapy. Run live E2E benchmark. |
| 002 | End-to-End Throughput Misrepresentation | P1 (High) | Performance | 16,505 samples/s measures ONLY offline dictionary-to-probability inference, bypassing all stateful flow aggregation. | Develop a physical live 10Gbps hardware testbed benchmark from raw PCAP to alerts. |
| 003 | Black-box Evasion Vulnerability | P1 (High) | Adversarial Eval | 500-query NES blackbox attack achieves 36.1% strict evasion on Option C (NF-ToN). The PGD-10 transfer attack substantially understated risk. | Develop exact joint gradient attacks on tree-neural boundary; explore fused adversarial training. |
| 004 | XAI Surrogate Additivity Violation | P2 (Medium) | XAI | The linear combination of RF and MLP SHAP values deviates from true fused Shapley values with MAE ~0.835. | Explicitly label XAI as a surrogate approximation. |
| 005 | DP Scope Limitation | P2 (Medium) | Privacy | RF dominates fusion (70%) but is not differentially private. | Extend DP tree building (e.g., DP-RF) to the non-differentiable stream. |
