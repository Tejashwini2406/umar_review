# Deliverable E: Experimental Results Ledger

| Experiment | Target Model | Dataset | Metric | Corrected/Verified Result | Original Reported Result (Historical) | Verification Status |
|---|---|---|---|---|---|---|
| Clean Detection | Option C | Edge-IIoTset | ROC-AUC | 0.9988 | 0.9988 | VERIFIED |
| Clean Detection | Option C | NF-ToN-IoT-v2 | ROC-AUC | 0.9927 | 0.9927 | VERIFIED |
| Neural-Gradient Transfer ASR | Option C | NF-ToN-IoT-v2 | Strict ASR (<0.50) | 6.51% (Denominator fixed) | 4.0% | CORRECTED (P4/P5) |
| Query-based Blackbox NES ASR | Option C | NF-ToN-IoT-v2 | Strict ASR (<0.50) | 36.1% (500 queries) | N/A | VERIFIED (P5) |
| Differential Privacy | Neural Stream | All | Epsilon ($\epsilon$) | 2.37 | 2.37 | VERIFIED (Applies only to neural stream) |
| Multi-Seed Robustness | Option C | Edge-IIoTset | Test ASR Variance | Seeds mitigate catastrophic failure | "Guarantees < 5% ASR" | DOWNGRADED (P2) |
| Offline Classifier Throughput | Option C | - | Samples/sec (Batch 1024) | 16,504.7 | 16,505 | VERIFIED (Scope corrected in P7) |
| Exact SHAP Additivity | Option C | - | Mean Absolute Error | 0.835 | N/A | FAILED (Surrogate is inexact - P6) |
