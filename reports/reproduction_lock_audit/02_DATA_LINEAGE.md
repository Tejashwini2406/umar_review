# DATA LINEAGE RECONSTRUCTION
- **Raw Data**: E.g. Edge-IIoTset initially contains millions of flows.
- **Filtering/Sampling**: The pipeline (`run_milestone3a.py`) severely subsets the dataset. 
- **Final Distribution**: Precisely 7,000 flows (4200 train, 1400 val, 1400 test).
- **Why 7,000?**: Downsampling to 7,000 ensures equal representation across datasets for simplified, controlled multi-domain experiments. However, it abandons massive portions of data, severely weakening generalization claims.
