# Deliverable I: Development / Debugging Chronology

1. **Initial Architecture**: Development of RF and Std MLP on the 21-feature tabular dataset. Achieved high AUC.
2. **Adversarial Realization**: Discovered neural streams are highly vulnerable to PGD. Implemented DP-SGD + PGD-7 on a Robust MLP.
3. **Option C Creation**: Ensembled 0.7 RF + 0.3 Robust MLP to balance clean AUC (driven by RF) and adversarial robustness (driven by Rob MLP).
4. **Historical Miscalculations**: Initial adversarial transfer experiments failed to exclude naturally misclassified baseline errors, overstating "robustness". Throughput was calculated offline and mislabeled as "End-to-End".
5. **Master Audits (P1-P8)**: 
   - Fixed data leakage validation (P1).
   - Removed mathematical guarantees of robustness (P2).
   - Clarified DP scope to neural stream only (P3).
   - Exposed 36.1% operational evasion via NES blackbox queries (P5).
   - Disproved exact XAI claims (P6).
   - Downgraded throughput to "classifier-only" (P7).
   - Final IEEE manuscript corrected (P8).
