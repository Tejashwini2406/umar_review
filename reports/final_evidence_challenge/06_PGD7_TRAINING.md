# PGD-7 TRAINING EVALUATION

### TRAINING PARAMETERS
- **Source**: `src/iot_ids/adversarial/adversarial_training.py`
- **Configuration**: `steps=7`, `epsilon=0.10`, `alpha=0.025`
- **Perturbations Projected?**: Yes. Clamped between `x_min` and `x_max`.
- **Protocol Mask Frozen?**: Yes. A continuous mask prevents perturbation of one-hot features.
- **Feature Space**: The perturbation is applied in the *scaled* feature space (after RobustScaler).
- **Loss**: `BCEWithLogitsLoss`
- **Gradients Detached?**: Yes, when calculating tracking metrics. The generated adversarial examples are injected back into the batch for standard optimization.

*(Status: SOURCE-VERIFIED)*
