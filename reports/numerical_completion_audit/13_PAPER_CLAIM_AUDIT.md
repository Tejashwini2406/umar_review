# PAPER CLAIM AUDIT

| Paper Claim | Status | Caveat |
| :--- | :--- | :--- |
| "Option C mean ROC-AUC of 0.9970" | SUPPORTED | Exact mathematical reproduction. |
| "Reduces ASR to 4.0% compared to 53.8%" | SUPPORTED | Exact replication on NF-ToN-IoT-v2. |
| "Neural stream satisfies epsilon=2.37" | SUPPORTED-WITH-CAVEAT | DP utility was only demonstrated on synthetic Gaussians, not IIoT data. |
| "Weighted Component Attribution Aggregation" | SUPPORTED | Explains components separately; does not produce a true ensemble SHAP value. |
| "Detection throughput of 16,505 samples/sec" | SUPPORTED-WITH-CAVEAT | Excludes heavy PCAP feature extraction logic. |
