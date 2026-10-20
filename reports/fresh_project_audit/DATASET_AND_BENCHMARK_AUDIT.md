# DATASET AND BENCHMARK AUDIT

## Evaluated Datasets
The datasets used in `stage3` are Edge-IIoTset, NF-ToN-IoT-v2, ToN-IoT, and CICIoT2023.
For each dataset, the training size is exactly 4,200 samples, and the test size is 1,400 samples.

## Limitations
1. **Size**: 7,000 samples per dataset (4200 train / 1400 val / 1400 test) is very small for modern Deep Learning. This brings generalization claims into question.
2. **Subsetting**: The data has been heavily subsetted from the original millions of rows to exactly 7,000 per dataset.
3. **No Database Integration**: The data is loaded via Parquet files in-memory, completely bypassing realistic data ingestion infrastructure.
