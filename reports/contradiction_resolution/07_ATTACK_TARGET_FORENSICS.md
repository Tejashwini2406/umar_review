# DIRECT MLP ATTACK VS ENSEMBLE ATTACK

- **Source**: `scripts/15_adaptive_ensemble_attack.py`
- **Attack Type**: `2. RF surrogate-based adaptive attack`.
- **Target Objective**: The gradients are backpropagated through a joint differentiable surrogate: `P_surrogate = 0.7 * S_RF(x) + 0.3 * P_RobustMLP(x)`.
- **Evaluation**: The generated adversarial examples are successfully evaluated against the ACTUAL frozen RF (`rf_model.predict_proba`) combined with the ACTUAL RobustMLP.
- **Verdict**: Methodologically sound. It is an adaptive surrogate attack, correctly bypassing the non-differentiability of the RF.
