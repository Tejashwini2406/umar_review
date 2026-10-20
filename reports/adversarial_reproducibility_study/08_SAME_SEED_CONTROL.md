# SAME-SEED REPRODUCIBILITY CONTROL

To isolate hardware/environment nondeterminism from seed-based training initialization variability, `seed 42` was executed twice.

| Dataset | Metric | Seed 42 | Seed 42 (Run B) | Difference |
|---|---|---|---|---|
| Edge-IIoTset | Rob AUC | 0.8028 | 0.8028 | 0.0000 |
| Edge-IIoTset | Opt C ASR | 0.0800 | 0.0800 | 0.0000 |
| NF-ToN-IoT-v2 | Rob AUC | 0.8396 | 0.8396 | 0.0000 |
| NF-ToN-IoT-v2 | Opt C ASR | 0.4480 | 0.4480 | 0.0000 |

**Verdict**: Execution is perfectly stable within the same environment. The massive variability observed across seeds is purely a function of the stochastic initializations and batch orderings interacting with the PGD-7 adversarial min-max optimization.
