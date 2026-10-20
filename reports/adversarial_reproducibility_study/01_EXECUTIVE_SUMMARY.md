# EXECUTIVE SUMMARY

The Multi-Seed Adversarial Robustness Reproducibility Study confirms that the adversarial robustness of the neural stream (and its fused Option C counterpart) is highly sensitive to training initialization. 

**Key Finding on Robustness Stability:**
The historical 4.0% Option C ASR on NF-ToN-IoT-v2 is **NOT** a fabricated or impossibly lucky outlier. In fact, 8 out of 10 fresh random seeds produced ASRs between 0.7% and 6.9%. However, 2 out of 10 seeds (including the default fresh `42` seed) experienced catastrophic robustness failure, yielding ~45% ASR. 

**Conclusion**: The architecture exhibits **HIGH ROBUSTNESS VARIABILITY**. The historical checkpoint is highly representative of the *median* behavior, but the architecture lacks training stability, meaning robustness cannot be guaranteed purely by the configuration without validating the resulting checkpoint.
