# DATASET SAMPLING REPRODUCTION

### THE 7,000 SAMPLE CLAIM
`scripts/03_materialize_and_audit_features.py` stops parsing the PCAP flow stream exactly when `len(benign) == 2000` and `len(attack) == 5000`.

- **Does it stop ingestion?** YES. It breaks the iteration loop.
- **Is it deterministic?** YES, assuming the raw flow parsing order is deterministic.
- **Is it chronologically selected?** YES. It takes the very first 2,000 benign and 5,000 attack flows encountered.
- **Is class balancing occurring before or after collection?** BEFORE collection finishes (it halts collection independently per class).
- **Are train/val/test splits chronological?** YES. They are split chronologically within each class array.

*(Status: SOURCE-VERIFIED)*

### CONCLUSION
The 7,000 limit is a hardcoded engineering control limit to ensure rapid experimentation. It is NOT statistically justified. It chronologically truncates the dataset.
