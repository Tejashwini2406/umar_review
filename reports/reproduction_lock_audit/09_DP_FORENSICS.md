# DIFFERENTIAL PRIVACY FORENSICS
- **Critical Failure**: The `scripts/run_privacy_experiments.py` explicitly utilizes `np.random.normal()` to generate fake, synthetic data to demonstrate DP-SGD utility/privacy tradeoffs. 
- **Verdict**: NOT VERIFIED on real IIoT data. The DP experiment is a synthetic demonstration only. Furthermore, Random Forest is non-private, so claiming the entire Option C is DP is false.
