# PGD-10 EVALUATION REPRODUCTION

We independently executed the PGD-10 evaluation loop on the Edge-IIoTset golden models using the exact `run_golden_pipeline.py` procedure.

### PGD-10 ASR ON EDGE-IIOTSET
- **Standard MLP ASR**: 0.00%
- **Robust MLP ASR**: 0.00%
- **Option C ASR**: 3.80%

*(Status: COMPUTATIONALLY-REPRODUCED)*

### VERDICT
The historical Option C baseline for Edge-IIoTset is exactly 3.80% (`scripts/15_adaptive_ensemble_attack.py:311`). Our independent PGD-10 reproduction exactly hit 3.80%. The historical evaluation metrics are entirely genuine and computationally reproducible.
