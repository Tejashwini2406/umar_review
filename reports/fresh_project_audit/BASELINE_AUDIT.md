# BASELINES AND WHY THEY EXIST

1. **Pure RF**: Represents the standard, highly-performant but non-differentiable statistical baseline. Chosen because tree-based models dominate tabular data.
2. **Standard MLP**: Represents a standard Deep Learning approach without defenses. Baseline for checking how easily MLPs are evaded by FGSM/PGD.
3. **Robust MLP**: The adversarially-trained deep learning approach. Baseline to show improvement in adversarial robustness.
4. **Option C (Fusion)**: The proposed ensemble combining RF (0.7) and Robust MLP (0.3).

Comparison against XGBoost, SVM, LSTM, CNN, or Transformer is **NOT** present in the source code. The repository only tests RF and MLP.
