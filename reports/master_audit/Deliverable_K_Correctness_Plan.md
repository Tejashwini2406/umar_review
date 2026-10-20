# Deliverable K: Correctness-First Next-Step Plan

1. **Phase 1: Environment Stabilization**
   - Install `scapy` and network capture dependencies.
   - Run and pass `test_pcap_and_live_ingress_e2e.py`.
2. **Phase 2: True End-to-End Benchmarking**
   - Develop a physical PCAP replay script that times ingestion from disk to Scapy to `FlowAggregator` to Inference to Alert Sink.
   - Profile memory usage and identify bottlenecks in the EWMA temporal tracker.
3. **Phase 3: Adversarial Defenses**
   - Develop a joint adversarial training loop that mathematically accounts for the RF decision boundaries during neural gradient generation, mitigating the 36.1% black-box evasion vulnerability.
4. **Phase 4: Differential Privacy for Trees**
   - Replace the standard `RandomForestClassifier` with a Differentially Private Random Forest to protect the remaining 70% of the fusion decision weight.
