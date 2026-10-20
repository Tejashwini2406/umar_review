# FINAL INTERPRETATION

**C. HIGH ROBUSTNESS VARIABILITY**

Different seeds produce substantially different PGD-10 ASR, making the robustness claim highly checkpoint-dependent. While the historical 4.0% NF-ToN-IoT-v2 claim is mathematically genuine and representative of the median outcome, the training process exhibits a ~20% catastrophic failure rate where adversarial robustness collapses, yielding ~45% ASR. 

**Paper Implications:**
1. The paper **cannot** claim the PGD-10 robustness as a guaranteed property of the architecture alone; it is a property of the successfully trained checkpoints.
2. The paper **should** report the historical 4.0% and 3.8% values, as they are not anomalies, but they MUST be accompanied by a disclaimer regarding training instability.
3. The safest scientific wording is to report the performance of the evaluated checkpoints while explicitly acknowledging the high variance (e.g., reporting the multi-seed mean and standard deviation) in the limitations section.
