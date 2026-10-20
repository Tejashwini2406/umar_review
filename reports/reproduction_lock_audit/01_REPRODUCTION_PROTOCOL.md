# REPRODUCTION PROTOCOL
1. **Rule**: Absolute Anti-Contamination. No previous golden manifests, paper tables, or previously cached results were loaded or trusted.
2. **Execution**: A completely independent Python script (`scratch/execute_lock_audit.py`) was constructed to load raw `stage3` parquet files, instantiate new model classes (Random Forest, MLP_Std, MLP_Rob), fit them on the training split, and run predictions on the test split.
3. **Artifacts**: New models were saved exclusively to `reports/reproduction_lock_audit/models/`.
4. **Conclusion**: Everything in this directory represents fresh, independently verified computations.
