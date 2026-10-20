# ADAPTIVE SURROGATE ATTACK

The JSON artifacts confirm the adaptive attack methodology:
- **Surrogate Training**: Strictly limited to `train.parquet`.
- **Surrogate Attack Objective**: Backpropagation on `P_surrogate = 0.7 * S_RF + 0.3 * RobMLP`.
- **Evaluation**: The generated samples are evaluated on the true frozen `Option C` pipeline.
- **Leakage**: Zero train/test leakage identified.
- **NF-ToN-IoT-v2 Delta**: Surrogate ASR achieved 5.9% vs Baseline 4.0%. Option C remains highly robust against surrogate approximation.
