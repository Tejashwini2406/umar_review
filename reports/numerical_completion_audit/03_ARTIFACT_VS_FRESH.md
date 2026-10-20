# FROZEN ARTIFACT VS FRESH TRAINING

| Dataset | Model | Artifact Metric (AUC) | Fresh Retraining Metric (AUC) | Difference |
| :--- | :--- | :--- | :--- | :--- |
| Edge-IIoTset | RF | 0.9995 | 0.9995 | 0.0000 |
| Edge-IIoTset | Std MLP | 0.8443 | 0.8443 | 0.0000 |
| Edge-IIoTset | Rob MLP | 0.8002 | 0.8028 | 0.0026 |

**Observation**: Robust MLP uses PGD-7 adversarial batch generation during training, introducing stochastic variations. Fresh retraining does not achieve exact bitwise replication, but remains firmly within tight statistical tolerance.
