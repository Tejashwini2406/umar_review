# MODEL TRAINING REPRODUCTION
- **RF**: Re-trained from scratch. `n_estimators=100, max_depth=20, class_weight='balanced'`.
- **Standard MLP**: 4-layer (128->64->32->1) with BatchNorm and Dropout(0.2). Trained with Adam (lr=1e-3).
- **Robust MLP**: Same architecture. Adversarially trained with FGSM/PGD perturbations on continuous features.
- **Status**: ALL MODELS SUCCESSFULLY RE-TRAINED FROM SCRATCH AND SAVED to `reports/reproduction_lock_audit/models/`.
