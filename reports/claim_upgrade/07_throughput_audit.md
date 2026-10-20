# END-TO-END THROUGHPUT AUDIT (P7)

## 1. Executive Summary
This audit investigated the claim that the Option C IDS achieves an end-to-end throughput of **16,505 samples/sec** (0.0606 ms/sample). The investigation reveals that this benchmark strictly measures **batched, in-memory, classifier-only inference**. It completely excludes all network ingestion, packet parsing, stateful flow aggregation, and feature extraction stages. 

**Verdict: FAIL** - The manuscript incorrectly presents isolated model inference throughput as end-to-end IDS pipeline throughput.

## 2. Benchmark Script Analysis
**Target File**: `scripts/benchmark_runtime_xai_privacy.py`

### Included Stages (Measured)
- Dictionary schema validation
- `RobustScaler` transformation
- `RandomForestClassifier.predict_proba`
- PyTorch `RobustMLP` forward pass
- Output probability fusion ($0.7 P_{RF} + 0.3 P_{MLP}$)

### Excluded Stages (Bypassed)
- Network Interface / PCAP I/O
- Scapy packet parsing (`scapy_to_canonical_packet`)
- Stateful Flow Aggregation (`FlowAggregator`)
- Dynamic Feature Extraction (e.g., inter-arrival times, EWMA rates)
- Alert routing / sink delivery

### Reproduced Numbers
From the raw artifact `reports/runtime/runtime_overhead_summary.json` at Batch Size 1024:
- **Per-Sample Latency**: 0.0606 ms
- **Throughput**: 16504.7 samples/sec

## 3. PCAP Smoke Test Inspection
**Target File**: `reports/pcap_alerts.json`

The raw artifacts from the PCAP smoke test contain solely the finalized alert objects, yielding:
- **Total Alerts Logged**: 6
- **Unique Flows Alerted**: 1
- **Packet Counts Recorded?**: False
- **Latency/Throughput Recorded?**: False
- **Errors Recorded?**: False

The smoke test merely verifies the integration of the API endpoints for 2 mock flows; it does not measure or constitute evidence of enterprise-scale production capacity.

## 4. Conclusion & Required Corrections
The throughput of 16,505 samples/sec is valid for **offline batch inference latency**. It must be explicitly re-labeled as "classifier-only inference throughput" in the manuscript. Any claims implying this represents live network packets processed per second must be removed until a true end-to-end network benchmark is conducted.
