# REPOSITORY FORENSICS

## Current Architecture
The current IIoT IDS architecture uses a fusion approach (Option C):
- **Features**: 21 canonical flow numerical features (protocol features frozen).
- **Preprocessing**: Raw `RobustScaler` (without `log1p`).
- **Standard MLP**: 4-layer architecture (128 -> 64 -> 32 -> 1) with BatchNorm, ReLU, Dropout (0.2). Trained with Adam, BCEWithLogitsLoss, batch size 256, 15 epochs.
- **Robust MLP**: Identical architecture, trained via PGD-7 (epsilon=0.10, alpha=0.025, 7 iterations).
- **Random Forest**: `n_estimators=100`, `max_depth=15`.
- **Option C**: $P_{OptionC} = 0.7 \times P_{RF} + 0.3 \times P_{RobustMLP}$.

## Relevant Files
- **Core Library**: `src/iot_ids/`
  - Models: `src/iot_ids/models/`
  - Adversarial: `src/iot_ids/adversarial/attacks.py`, `src/iot_ids/adversarial/adversarial_training.py`
  - Runtime: `src/iot_ids/runtime/`, `src/iot_ids/cli.py`
- **Active Audits**: `scratch/run_numerical_audit.py`, `scratch/run_multiseed_robustness.py`, `scratch/run_asr_contradiction_resolution.py`
- **Unit/Integration Tests**: `tests/unit/`, `tests/integration/`

## Active vs Legacy Scripts
- **Active Training/Evaluation**: `run_golden_pipeline.py`, `run_privacy_experiments.py`, `benchmark_runtime_xai_privacy.py`. The canonical scripts are heavily consolidated into the pipeline registry or recent scratch audit scripts.
- **Legacy/Orphaned Scripts**: `scripts/train_rf.py`, `scripts/train_mlp.py`, `scripts/train_ae.py` are orphaned and disconnected from the `stage3` architecture. Do not use.

## Model Artifacts
- **Golden Artifacts**: `models/golden_run/` (Contains the frozen checkpoints responsible for the historical 4.0% ASR on NF-ToN-IoT-v2).
- **Reproducibility Artifacts**: `models/reproducibility_study/` (Contains the 10 independent random seed runs).

## Dataset Artifacts
- **Canonical Splits**: `data/processed/stage3/` for `Edge-IIoTset`, `NF-ToN-IoT-v2`, `ToN-IoT`, and `CICIoT2023`.
- **Split Sizes**: 4,200 Train / 1,400 Val / 1,400 Test.

## Current Reproducibility State
- **Clean Metrics**: Completely reproducible. ROC-AUC converges consistently to ~0.9970 across seeds.
- **Adversarial Metrics**: HIGHLY VOLATILE. PGD-10 ASR varies significantly based purely on PyTorch initialization and data shuffling seeds (mean: 12.3%, median: 4.5%, std dev: 17.9%). The historical results are mathematically present in the frozen artifacts but cannot be trivially replicated on fresh seeds without stochastic variance.

## Potential Leakage & Limitation Risks
1. **Adversarial Leaks**: The adaptive attack was historically tested exclusively on Option C, but the surrogate model trained strictly on `train.parquet`. The leakage was confirmed nonexistent in recent audits.
2. **DP Leaks**: The DP evaluation in `run_privacy_experiments.py` is simulated on synthetic Gaussian tensors (`rng.normal`) rather than real network flows.
3. **XAI Leaks**: Current XAI is computed via `TreeExplainer` and `GradientExplainer` separately on the two sub-models. True end-to-end SHAP on the fused non-linear sum is absent.
4. **Runtime Leaks**: `16,505 samples/sec` applies to offline memory `.predict()` strictly, excluding Scapy packet capture, flow extraction, and temporal EWMA generation.

## Code Paths That Must Not Be Modified
- `src/iot_ids/preprocessing/` canonical scaling pipelines must remain untouched to avoid inducing the preprocessing drift discovered in earlier audits.
- Model definitions and dataset shapes must strictly match the historical frozen `golden_run` models to ensure fair comparisons.

## Frontend/Backend Implementation
**Verdict: NOT IMPLEMENTED.**
A thorough scan of the repository reveals zero active frontend (React) or backend API (FastAPI) applications. The system exists entirely as a Python package (`iot-ids`) with a CLI tool (`cli.py`), test suite, and operational daemon engine. Any claim referencing a web dashboard is unsupported.
