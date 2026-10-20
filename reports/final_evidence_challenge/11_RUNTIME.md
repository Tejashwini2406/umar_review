# RUNTIME MEASUREMENT SCOPE

The `scripts/benchmark_runtime_xai_privacy.py` script measures throughput using `time.perf_counter()`.

### INCLUDED IN MEASUREMENT
- Scikit-learn `.predict()` execution (RF)
- PyTorch `.forward()` execution (MLP)
- Simple weighted fusion of probabilities.

### EXCLUDED FROM MEASUREMENT
- PCAP parsing
- Flow session tracking & state management
- Temporal/Behavioral EWMA feature extraction
- Type casting

*(Status: SOURCE-VERIFIED)*

### CONCLUSION
The reported latency (ms) exclusively measures offline ML inference. Claims of line-rate throughput in Gbit/s cannot be extrapolated from these numbers.
