# ADVERSARIAL EVALUATION

The adversarial evaluation (FGSM/PGD) is a **white-box neural attack**. It targets the differentiable MLP models by freezing protocol features (features 7-10).
The Random Forest is non-differentiable, so it is evaluated using an **adaptive surrogate attack** (where a differentiable S_RF approximates the RF).

## FGSM Evasion on Continuous Features (Epsilon = 0.10)
- **Edge-IIoTset**: Standard MLP ASR: 0.0%, Robust MLP ASR: 0.0%
- **NF-ToN-IoT-v2**: Standard MLP ASR: 76.6%, Robust MLP ASR: 53.8%
- **ToN-IoT**: Standard MLP ASR: 2.3%, Robust MLP ASR: 2.5%
- **CICIoT2023**: Standard MLP ASR: 2.6%, Robust MLP ASR: 1.4%
