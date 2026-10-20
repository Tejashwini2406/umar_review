# DIFFERENTIAL PRIVACY SCOPE

- **Correctness**: Opacus is mathematically configured properly (PRV accountant, $\epsilon=2.37$).
- **Limitation**: The experiment evaluates exclusively on synthetic Gaussian tensors (`rng.normal`), entirely bypassing the real `stage3` IIoT flow data. 
- **Status**: The DP-utility tradeoff curve is a theoretical demonstration.
