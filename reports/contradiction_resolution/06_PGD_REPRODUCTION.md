# PGD-7 TRAINING VS PGD-10 EVALUATION

- **Training**: `train_mlp_adversarial` in `src/iot_ids/adversarial/adversarial_training.py` defaults to `pgd_steps=7`. `scripts/run_golden_pipeline.py:115` explicitly sets `pgd_steps=7`.
- **Evaluation**: `scripts/run_golden_pipeline.py:243` explicitly calls `pgd_attack(steps=10)` for standard and robust MLP.
- **Verdict**: There is no contradiction. Training legitimately uses 7 steps (for speed) and evaluation legitimately uses 10 steps (for rigor). Both use eps=0.10, alpha=0.025.
