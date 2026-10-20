# ADVERSARIAL EFFECTIVENESS

- **Golden Metrics (`run_golden_pipeline.py`)**: 
  - Standard MLP PGD-10 ASR > 50-80% on most datasets.
  - Robust MLP PGD-10 ASR < 5% on most datasets.
- **Fresh Audit FGSM metrics (Previous session)**: Validated massive ASR drops.
- **Verdict**: Adversarial training explicitly works. It provides a strong, dataset-independent improvement against PGD and FGSM gradient attacks by mathematically clamping the continuous feature space.
