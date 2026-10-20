# RUNTIME REPRODUCTION

- **Scope**: `benchmark_runtime_xai_privacy.py` exclusively times pure `.predict()` and `.forward()` iterations on pre-scaled float arrays in memory.
- **Exclusions**: PCAP capture, network state tracking, flow expiration, and temporal EWMA extractions are not timed.
- **Verdict**: The 16,505 samples/sec throughput is an inference capacity ceiling, not an end-to-end NIDS throughput metric.
