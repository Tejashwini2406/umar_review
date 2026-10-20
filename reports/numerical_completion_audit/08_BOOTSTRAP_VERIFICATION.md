# BOOTSTRAP CONFIDENCE INTERVALS

`scripts/17_bootstrap_confidence_intervals.py` utilizes:
- **B**: 1000 replicates.
- **Method**: Non-parametric percentile (2.5 to 97.5).
- **Unit**: Test set sample replacement.
- **Model Training**: The models are frozen; predictions are resampled.
- **Verdict**: The CIs quantify test set sampling variance, NOT training initialization stochasticity.
