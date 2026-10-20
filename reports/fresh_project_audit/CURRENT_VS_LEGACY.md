# CURRENT VS LEGACY IMPLEMENTATIONS

| Component | Current implementation | Legacy implementation | Which is used for final evaluation? |
| :--- | :--- | :--- | :--- |
| Models | `models/golden_run/` | `models/final/` | `models/golden_run/` |
| Adv Attack | `15_adaptive_ensemble_attack.py` | `10_run_adversarial_evaluation.py` | `15_adaptive_ensemble_attack.py` |
| Evaluation | `scratch/fresh_audit_generator.py` | `scripts/07_evaluate.py` | Fresh Script |
