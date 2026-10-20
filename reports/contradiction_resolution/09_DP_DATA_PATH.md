# DIFFERENTIAL PRIVACY DATA PATH

- **Source**: `scripts/run_privacy_experiments.py:84-100`
- **Code Trace**: `load_evaluation_samples()` literally defines `X_tr_b = rng.normal(loc=0.0, scale=1.0, size=(n_train_b, 21))` and completely ignores `data/processed/stage3/` parquet files.
- **Evaluation**: DP-SGD trains the Robust MLP on these synthetic Gaussian tensors, computes Opacus epsilon, and tests ROC-AUC on more synthetic Gaussian tensors.
- **Verdict**: The DP experiment is a privacy mechanism demonstration, NOT an evaluation of the IIoT datasets. The privacy-utility tradeoff curve is synthesized.
