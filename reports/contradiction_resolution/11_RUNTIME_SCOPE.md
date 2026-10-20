# RUNTIME SCOPE

- **Source**: `scripts/benchmark_runtime_xai_privacy.py:53`
- **Trace**: Measures exactly: `time.perf_counter() -> engine.predict(batch_inputs) -> time.perf_counter()`.
- **Inputs**: The `batch_inputs` is a pre-constructed Python dictionary of 21 extracted float features.
- **What is NOT measured**: packet capture (pcap), packet parsing, session tracking, timeout eviction, and continuous feature EWMA state updates.
- **Verdict**: The reported latencies (e.g., ~1-5ms) reflect pure in-memory ML inference time on pre-parsed features. "Line-rate" network claims are unsupported.
