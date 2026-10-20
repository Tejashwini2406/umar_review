# ADAPTIVE SURROGATE ATTACK
- **Method**: Train differentiable PyTorch neural surrogate to approximate non-differentiable RF output probabilities.
- **Gradients**: Computed entirely through the surrogate objective function.
- **Evaluation**: The generated adversarial examples are sent to the ACTUAL frozen RF and Robust MLP.
- **Verdict**: Verified from source. Valid evaluation methodology for non-differentiable trees.
