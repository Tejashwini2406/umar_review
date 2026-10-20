# TRAINING CONFIGURATION
- **Data**: Canonical `stage3` partitions (60/20/20). 7000 samples per dataset.
- **Preprocessing**: Raw `RobustScaler`.
- **Architectures**: 4-layer MLP (128->64->32->1) and RF (max_depth=15).
- **Adversarial Training**: PGD-7, eps=0.1, alpha=0.025.
- **Seeds**: 10 independent seeds (42 through 51), plus a duplicate run of 42 (`42_runB`) for hardware non-determinism control.
