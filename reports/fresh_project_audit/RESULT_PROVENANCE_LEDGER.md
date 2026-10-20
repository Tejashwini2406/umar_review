# RESULT PROVENANCE LEDGER

| Metric | Dataset | Value | Source code | Freshly recomputed? | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Clean RF AUC | Edge-IIoTset | 0.9994 | `scratch/fresh_audit_generator.py` | YES | FRESHLY REPRODUCED |
| Clean OptC AUC | Edge-IIoTset | 0.9987 | `scratch/fresh_audit_generator.py` | YES | CONFLICTING (Lower than RF) |
| Privacy AUC/Eps | All | N/A | `scripts/run_privacy_experiments.py` | NO | NOT VERIFIED (Fake Data) |
