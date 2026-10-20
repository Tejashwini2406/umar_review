# ARUN KUMAR PROJECT CRITIQUE

| Reviewer concern | What it means technically | Does the CURRENT implementation address it? |
| :--- | :--- | :--- |
| Modest dataset size | 7000 flows is small | NO. Hardcoded to 7000 in `data/processed/stage3/`. |
| Surrogate Attack | Needed true ensemble evaluation | YES. `15_adaptive_ensemble_attack.py` implemented. |
| Privacy Tradeoff | Lack of real DP evaluation | NO. `run_privacy_experiments.py` uses fake Gaussian data. |
