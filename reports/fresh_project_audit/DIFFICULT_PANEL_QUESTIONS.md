# DIFFICULT PANEL QUESTIONS

**1. If RF has AUC 0.9994 and Option C has 0.9987, why call fusion useful?**
Admit that fusion DOES NOT improve clean performance. Its only theoretical benefit is providing differentiable gradients for adversarial robustness analysis and blending anomaly detection.

**2. Is the entire system Differentially Private?**
No. The Random Forest is completely non-private.

**3. Is your attack really white-box?**
For the MLP, yes. For the RF, no, it is a surrogate grey-box attack.
