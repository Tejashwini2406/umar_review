# 7,000-SAMPLE DATASET CONSTRUCTION

- **Source Code**: `scripts/03_materialize_and_audit_features.py:66-81`
- **Logic**: The stream adapter loops over raw flows. It stops aggregating exactly when `len(benign) == 2000` and `len(attack) == 5000`.
- **Splitting**: It then chronologically partitions each class separately into 60% Train, 20% Val, 20% Test.
- **Why exactly 7,000?**: It is an arbitrary engineering control limit explicitly hardcoded (`target_benign = 2000`, `target_attack = 5000`). It is NOT statistically justified. It chronologically drops millions of flows after the first 7,000 are seen.
