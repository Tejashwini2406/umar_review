# FINAL LEDGER RECONCILIATION (P8)

## 1. Executive Summary
The numerical completion audit of the Option C IIoT IDS framework is now finished. All material claims in the IEEE manuscript have been reconciled against the raw execution evidence from P1 through P7. Unverified or overstated claims regarding exact SHAP calculations, end-to-end network throughput, and worst-case adversarial robustness guarantees have been formally downgraded or corrected in the manuscript.

## 2. Claim-by-Claim Ledger

### Claim 1: Data Leakage & Canonical Pipeline
- **Original Status**: Unverified chronological splits.
- **Audit Finding**: Chronological sorting and proper train/test splits were implemented correctly (P1).
- **Final Verdict**: PASS WITH SCOPE.

### Claim 2: Differential Privacy (DP)
- **Original Status**: DP claimed for the ensemble.
- **Audit Finding**: Opacus DP-SGD ($arepsilon=2.37$) strictly applies to the neural stream only. RF remains non-private (P3).
- **Final Verdict**: PASS WITH SCOPE (Corrected in manuscript).

### Claim 3: Multi-Seed Robustness
- **Original Status**: Guaranteed <5% ASR.
- **Audit Finding**: Selection over 10 seeds avoids catastrophic >99% ASR failures, but does not constitute a universal mathematical guarantee (P2).
- **Final Verdict**: PASS WITH SCOPE (Language downgraded).

### Claim 4: Fusion-Weight Pareto Optimality
- **Original Status**: 0.7/0.3 is strictly optimal.
- **Audit Finding**: 0.7/0.3 exists on the empirical Pareto frontier but is dominated by 0.8/0.2 in some conditions (P4).
- **Final Verdict**: PASS WITH SCOPE.

### Claim 5: Adversarial Evasion Rate
- **Original Status**: Reduces ASR to 4.0%.
- **Audit Finding**: The 4.0% claim relied on a flawed neural-gradient transfer attack. Corrected transfer ASR is 6.5%. However, under a rigorous query-based NES black-box attack (P5), the operational evasion rate reaches **36.1%**.
- **Final Verdict**: FAIL (Original threat model understated risk; paper updated to disclose 36.1% vulnerability).

### Claim 6: Exact Explainable AI (XAI)
- **Original Status**: XAI method is exact SHAP for the fused predictor.
- **Audit Finding**: Exact Shapley evaluation over the fused probability surface requires $2^{21}$ queries (3.53s per sample), making it intractable. The current linear combination is a surrogate that violates the exact SHAP additivity axiom (MAE ~0.835) (P6).
- **Final Verdict**: FAIL (Paper updated to explicitly acknowledge surrogate approximation).

### Claim 7: End-to-End Throughput
- **Original Status**: Pure detection throughput of 16,505 samples/sec.
- **Audit Finding**: The benchmark bypasses network packet ingestion, Scapy parsing, and stateful flow aggregation. It strictly measures in-memory offline classifier inference (P7).
- **Final Verdict**: FAIL (Paper updated to "offline classifier-only throughput").

## 3. Manuscript Reconciliation
The file `final_ieee_paper/main.tex` was updated to:
1. Re-label 16,505 samples/sec as *classifier-only throughput*.
2. Explicitly disclose the 36.1% query-based black-box evasion rate, correcting the 4.0% neural-transfer claim.
3. Disclose the surrogate nature of the XAI linear combination.
4. Add corresponding limitations to Table 4 for DP scope, XAI additivity, end-to-end throughput, and black-box optimization.

All raw computational artifacts and `main.tex` are preserved for submission.
