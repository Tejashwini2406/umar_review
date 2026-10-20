# HISTORICAL CHECKPOINT COMPARISON

| Dataset | Model | Historical Golden ASR | Multi-Seed Mean | Multi-Seed Median | Seed Range |
|---|---|---:|---:|---:|---:|
| Edge-IIoTset | Option C | 3.80% | 8.39% | 9.20% | 0.30% - 15.20% |
| NF-ToN-IoT-v2 | Option C | 4.00% | 12.36% | 4.50% | 0.70% - 47.50% |

**Analysis**: The historical golden checkpoints are NOT extreme outliers. For NF-ToN-IoT-v2, 4.0% is remarkably close to the empirical multi-seed median (4.50%). The mean is heavily skewed by the ~20% of seeds that experience catastrophic robustness failure (~45%).
