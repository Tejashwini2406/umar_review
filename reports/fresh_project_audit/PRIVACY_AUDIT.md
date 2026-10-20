# PRIVACY AUDIT

## Implementation changes

The DP-SGD trainer now computes per-record gradients, clips each record's global L2 gradient norm across all trainable parameters, adds Gaussian noise from the operating-system cryptographic random source, and uses independent Poisson sampling without a forced-nonempty fallback. Empty samples still receive a noise-only update, matching the accounted mechanism. Its epsilon is an integer-order RDP upper bound for the accounted Poisson-sampled Gaussian steps. Unnoised training losses are not emitted.

The schema audit now requires the exact approved 21 corrected Stage 3 features; unknown or duplicate columns are rejected. The legacy `scripts/run_privacy_experiments.py` runner has been retired because it used synthetic Gaussian data and mislabeled those outcomes as dataset experiments.

## Scope and evidence limits

- DP applies only to neural-model training. The Random Forest, data collection, preprocessing fitted on private data, BCL calibration, and the fused ensemble are not thereby made differentially private.
- The former synthetic privacy-utility results are withdrawn. Re-run `scripts/run_golden_pipeline.py` on verified corrected Stage 3 data before making utility claims; those results remain exploratory, not confirmatory research evidence.
- The accounting assumes record-level add/remove adjacency, the documented Poisson sampler, and Gaussian noise at the configured multiplier. Any change to sampling, batching, clipping, noise, or optimizer data access requires a fresh privacy analysis.
