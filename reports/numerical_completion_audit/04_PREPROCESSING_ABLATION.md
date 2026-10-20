# PREPROCESSING ABLATION (EDGE-IIOTSET)

A controlled experiment isolated the impact of removing `log1p` on MLP performance.

| Dataset | Preprocessing | Model | AUC | Macro F1 |
| :--- | :--- | :--- | :--- | :--- |
| Edge-IIoTset | Raw RobustScaler | Std MLP | 0.8443 | 0.4167 |
| Edge-IIoTset | log1p + RobustScaler | Std MLP | 0.9820 | 0.4032 |

**Conclusion**: The degradation to 0.8443 AUC was causally isolated to the omission of `log1p` in the `run_golden_pipeline.py` shortcut.
