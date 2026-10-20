import json
import os
import ast
from pathlib import Path

def audit_throughput():
    repo_root = Path("C:/Users/umari/Documents/P1_task_Implementation")
    
    # 1. Inspect existing benchmark runtime result
    benchmark_file = repo_root / "reports" / "runtime" / "runtime_overhead_summary.json"
    with open(benchmark_file, "r") as f:
        bench_data = json.load(f)
        
    b1024 = bench_data["pure_detection_benchmarks"]["1024"]
    reported_throughput = b1024["throughput_samples_per_sec"]
    reported_latency_ms = b1024["per_sample_latency_ms"]
    
    # 2. Trace the exact path of the benchmark script
    benchmark_script = repo_root / "scripts" / "benchmark_runtime_xai_privacy.py"
    with open(benchmark_script, "r") as f:
        script_content = f.read()
        
    # Analyze the script
    is_capture_excluded = "PacketCaptureEngine" not in script_content
    is_flow_agg_excluded = "FlowAggregator" not in script_content
    
    # 3. PCAP Smoke Test Inspection
    pcap_alerts_file = repo_root / "reports" / "pcap_alerts.json"
    pcap_alerts = []
    if pcap_alerts_file.exists():
        with open(pcap_alerts_file, "r") as f:
            for line in f:
                if line.strip():
                    pcap_alerts.append(json.loads(line))
                    
    num_alerts = len(pcap_alerts)
    flow_ids = set(a["flow_id"] for a in pcap_alerts)
    num_unique_flows = len(flow_ids)
    
    # Are latency statistics recorded in pcap_alerts.json?
    has_latency = any("latency" in a for a in pcap_alerts)
    has_packet_count = any("packet_count" in a for a in pcap_alerts)
    has_errors = any("error" in a for a in pcap_alerts)
    
    # Build JSON output
    audit_results = {
        "original_benchmark_claims": {
            "throughput_samples_sec": reported_throughput,
            "latency_ms_per_sample": reported_latency_ms,
            "batch_size_used": b1024["batch_size"]
        },
        "path_analysis": {
            "evaluated_component": "InferenceEngine.predict()",
            "excluded_stages": [
                "PCAP/Network I/O reading",
                "Scapy packet parsing",
                "Flow aggregation and state tracking",
                "Feature extraction (inter-arrival times, stats)",
                "Alert delivery to sinks (Elasticsearch, webhooks)"
            ],
            "included_stages": [
                "Dictionary schema validation",
                "RobustScaler transform",
                "RandomForest predict_proba",
                "RobustMLP PyTorch forward pass",
                "Probability fusion (0.7 RF + 0.3 MLP)",
                "Alert dict construction"
            ],
            "input_type": "Pre-constructed in-memory Python dictionary of 21 canonical features"
        },
        "pcap_smoke_test_artifacts": {
            "artifact_file": "reports/pcap_alerts.json",
            "total_alerts_recorded": num_alerts,
            "unique_flows_alerted": num_unique_flows,
            "contains_packet_counts": has_packet_count,
            "contains_latency_stats": has_latency,
            "contains_errors": has_errors
        },
        "verdict": "FAIL",
        "justification": "The reported 16,505 samples/sec throughput strictly measures batched, in-memory, classifier-only offline inference. It explicitly bypasses network ingestion, packet parsing, stateful flow aggregation, and feature extraction. Presenting this as end-to-end IDS throughput is mathematically and architecturally incorrect."
    }
    
    # Write JSON report
    out_dir = repo_root / "reports" / "claim_upgrade"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "07_throughput_audit.json", "w") as f:
        json.dump(audit_results, f, indent=2)
        
    # Write Markdown report
    md = f"""# END-TO-END THROUGHPUT AUDIT (P7)

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
- Output probability fusion ($0.7 P_{{RF}} + 0.3 P_{{MLP}}$)

### Excluded Stages (Bypassed)
- Network Interface / PCAP I/O
- Scapy packet parsing (`scapy_to_canonical_packet`)
- Stateful Flow Aggregation (`FlowAggregator`)
- Dynamic Feature Extraction (e.g., inter-arrival times, EWMA rates)
- Alert routing / sink delivery

### Reproduced Numbers
From the raw artifact `reports/runtime/runtime_overhead_summary.json` at Batch Size 1024:
- **Per-Sample Latency**: {reported_latency_ms:.4f} ms
- **Throughput**: {reported_throughput:.1f} samples/sec

## 3. PCAP Smoke Test Inspection
**Target File**: `reports/pcap_alerts.json`

The raw artifacts from the PCAP smoke test contain solely the finalized alert objects, yielding:
- **Total Alerts Logged**: {num_alerts}
- **Unique Flows Alerted**: {num_unique_flows}
- **Packet Counts Recorded?**: {has_packet_count}
- **Latency/Throughput Recorded?**: {has_latency}
- **Errors Recorded?**: {has_errors}

The smoke test merely verifies the integration of the API endpoints for 2 mock flows; it does not measure or constitute evidence of enterprise-scale production capacity.

## 4. Conclusion & Required Corrections
The throughput of 16,505 samples/sec is valid for **offline batch inference latency**. It must be explicitly re-labeled as "classifier-only inference throughput" in the manuscript. Any claims implying this represents live network packets processed per second must be removed until a true end-to-end network benchmark is conducted.
"""

    with open(out_dir / "07_throughput_audit.md", "w") as f:
        f.write(md)
        
    print("Audit P7 complete. Reports generated.")

if __name__ == "__main__":
    audit_throughput()
