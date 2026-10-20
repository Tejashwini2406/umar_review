# FUSION-WEIGHT ABLATION

Fresh retraining sweep across `w_RF` (0.0 to 1.0).

**Findings:**
- `w=0.0` (Pure MLP) yields lowest clean accuracy.
- `w=1.0` (Pure RF) yields highest clean accuracy across all datasets.
- The `w=0.7` point selected by the paper sacrifices a marginal fraction of clean accuracy (e.g., 0.9995 -> 0.9988 on Edge-IIoTset) to incorporate the adversarial robustness of the neural stream.
- The claim that 0.7/0.3 is "optimal" is subjective and trade-off dependent.
