# Paper-to-Implementation Review
## "An Adversarially Robust and Privacy-Preserving AI Framework for Explainable Intrusion Detection in IoT Environments"

**Date:** August 26, 2026
**Review Scope:** P1 Foundation Implementation Status
**Paper Datasets:** Edge-IIoTset, ToN-IoT Network
**P1 Datasets:** CICIDS2017, Edge-IIoTset, BoT-IoT, N-BaIoT

---

## Executive Summary

The P1 implementation is building the **foundational ML subsystem** described in the paper's Section III.B (System Components). As of Milestone 2.5, the representation layer is frozen and validated. However, **model training and ensemble evaluation have not yet been executed**—these tasks remain for P2 and P3 milestones.

The paper reports specific results (94.29% and 97.73% Attack F1 for in-domain evaluation, 86.57%/86.96% for cross-domain), but **P1's scope explicitly excludes model training and adversarial testing**. Those will be validated in downstream phases.

---

## 1. Paper Architecture vs. P1 Implementation

### Paper's System Components (Section III.B):

The paper describes a 5-layer system:

| Layer | Paper Description | P1 Implementation Status |
|-------|-------------------|--------------------------|
| **Data Preprocessing** | Network traffic cleaning, encoding, normalization, feature representation | ✅ **COMPLETE** - Milestone 2.5 frozen |
| **Supervised Detection** | RF + MLP with combined probabilities (Eq. 1) | ⏳ **STAGED** - Code scaffolding done, training deferred |
| **Anomaly Detection** | Benign-trained AE with reconstruction error ECDF | ⏳ **STAGED** - Autoencoder defined, training deferred |
| **Risk Layer** | Combining supervised + anomaly scores for three outcomes | ⏳ **STAGED** - API design sketched, implementation deferred |
| **Evaluation & Analysis** | In-domain, cross-domain, XAI, adversarial testing | ❌ **DEFERRED** - Scope explicitly excludes P4/XAI/adversarial |

---

## 2. Feature Representation Alignment

### Paper's Feature Hierarchy (Section IV.B):

The paper uses **four-level hierarchy** for robust, privacy-preserving design:

```
Fcommon (6 features):     duration, src_bytes, proto_tcp/udp/icmp, is_well_known_port
  ├─ Ftemporal (3 features):  for in-domain only
  ├─ Fbehavioral (2 features): for in-domain only
  └─ Fds (2 features):         dataset-specific, for in-domain only
```

### P1's Representation Freeze (Milestone 2.5):

**✅ CLOSE ALIGNMENT** with deliberate scope expansion:

| Feature Level | Paper Scope | P1 Implementation | Status |
|---|---|---|---|
| **Common Profile** | 6 features for cross-domain | `FLOW_COMPATIBLE_C_E_B`: 4 features | ⚠️ **REDUCED** |
| **Instant Features** | Implicit | Documented: 17 total | ✅ **Extended** |
| **Temporal Features** | 3 implied | Documented: 4 total | ✅ **Extended** |
| **Behavioral Features** | 2 implied | Documented: 5 total | ✅ **Extended** |

#### Critical Finding: Feature Reduction in Common Profile

The paper specifies `Fcommon` includes `syn_flag_count`, `ack_flag_count`, `rst_flag_count`, but **P1's Milestone 2.5 audit reduced this to 4 features**:

```json
{
  "FLOW_COMPATIBLE_C_E_B": [
    "duration_seconds",
    "dst_port",
    "total_bytes",
    "packet_length_mean"
  ]
}
```

**Reason:** BoT-IoT does not expose decomposed SYN/ACK/RST flags like CICIDS2017 and Edge-IIoTset. This is a **documented scientific limitation, not an implementation gap**.

**Impact:** Cross-domain transfer in P1 will use **4 features instead of paper's 7**. This may explain why P1 prioritizes in-domain evaluation and cross-dataset compatibility audits. The paper's 6-feature `Fcommon` assumes all four datasets share that representation, which is not universally true.

---

## 3. Dataset Coverage Comparison

### Paper's Datasets:
- **Edge-IIoTset**: ✅ Included in P1
- **ToN-IoT Network**: ❌ **NOT in P1 scope**

### P1's Datasets:
- **CICIDS2017**: Extended evaluation (4 in paper's references, full dataset in P1)
- **Edge-IIoTset**: ✅ Matches paper
- **BoT-IoT**: Extended evaluation (cited in paper [4] but not evaluation dataset)
- **N-BaIoT**: Extended evaluation (source-aggregate profile, incompatible with flow-based)

**Implication:** P1's 4-dataset matrix is more comprehensive than the paper's binary Edge-IIoTset ↔ ToN-IoT transfer. However, **ToN-IoT is absent**, which means P1's cross-domain results will not directly replicate the paper's 86.57%/86.96% Attack F1 scores.

**Scientific Note:** Paper's ToN-IoT is replaced by P1's BoT-IoT + N-BaIoT for richer domain-shift analysis. This is defensible but represents a **scope deviation** from the paper's experimental plan.

---

## 4. Model Architecture Alignment

### Paper's Model Specifications (Section IV.C):

| Model | Paper Config | P1 Scaffold |
|-------|--------------|-------------|
| **Random Forest** | 100 trees, max_depth=15 | ✅ Specified in config |
| **MLP** | D_in → 128 → 64 → 32 → 1, ReLU, BatchNorm, dropout(0.2) | ✅ Specified in config |
| **Autoencoder** | D_in → 64 → 16 → 64 → D_in | ✅ Specified in config |
| **Ensemble Fusion** | P_sup = 0.5 · P_rf + 0.5 · P_mlp (Eq. 1) | ⏳ Scaffolded, weights not tuned |
| **AE Scoring** | Reconstruction error → ECDF normalization → [0,1] score | ⏳ Scaffolded |

**Status:** Model hyperparameters are **defined but not yet trained**. The ensemble weights are **hardcoded to 0.5/0.5**, whereas the paper mentions "validation-only weight selection" (Section IV.D roadmap). This remains a P2/P3 task.

---

## 5. Evaluation Methodology: Paper vs. P1

### Paper's Evaluation (Section V):

| Evaluation Mode | Metric | Paper Result | P1 Planned |
|---|---|---|---|
| **In-Domain** | Attack F1 | Edge: 94.29%, ToN: 97.73% | ⏳ Pending model training |
| **Zero-Adaptation** | Attack F1 | Edge→ToN: 86.57%, ToN→Edge: 86.96% | ⏳ Pending; using 4 features not 6 |
| **Explainability** | Spearman ρ (RF vs MLP) | Edge: 0.8632, ToN: 0.6252 | ⏳ Deferred to P4 |
| **Adversarial** | ASR + AE Catch Rate | FGSM/PGD-10 with ε={0.05..0.3} | ❌ Out of P1 scope; P4 task |

**P1's Evaluation Plan:** Based on `configs/experiments/splits.json` and roadmap:

1. ✅ **Representation & Splitting** (Milestone 2.5): Frozen
2. ⏳ **Same-Dataset In-Domain** (P2): Train on all 4 datasets
3. ⏳ **Ablation Studies** (P2): Feature-level impact analysis
4. ⏳ **Compatible 4×4 Cross-Domain Matrix** (P2): All pairwise transfers
5. ❌ **Adversarial/XAI** (P4): Out of P1 scope

**Finding:** P1's evaluation scope is **broader** (4 datasets vs. 2) but **narrower in advanced testing** (no adversarial, no XAI attribution analysis).

---

## 6. Privacy and Leakage Control

### Paper's Privacy Strategy (Section IV.B):

> "The framework uses derived flow-level features rather than raw packet payloads, reducing the need to exchange sensitive network content."

### P1's Leakage Control (Milestone 2):

✅ **STRONG ALIGNMENT**. P1 documents leakage roles explicitly:

**Excluded from Features (Leakage):**
- Flow IDs, source/destination IPs
- Raw timestamps (when used directly as predictive values)
- Device IDs, host identifiers
- Label-derived category/type fields

**Permitted Metadata Uses:**
- Grouped splitting (by device, flow, partition, day)
- Domain analysis and diagnostics
- Source-file/partition tracking

**Result:** `leakage_metadata.json` provides machine-readable guidance for preventing information leakage. P1's approach is **more rigorous than the paper**, explicitly auditing every feature's role.

---

## 7. Reproducibility & Artifact Management

### Paper's Reproducibility Claims (Section VI.E):

> "Frozen checkpoints, evaluation artifacts, and independent retraining verification."

The paper reports ±0.06 percentage points variance under retraining—**a useful benchmark**.

### P1's Artifact Strategy:

✅ **EXCEEDS PAPER STANDARDS**:

- Machine-readable config schemas: `configs/datasets/`, `configs/features/`, `configs/experiments/`
- Split manifests: `splits.json` with seed, method, stratification evidence
- Feature inventory: `feature_availability.json` (field-by-field audit)
- Label taxonomy: `label_taxonomy.json` (attack class semantics)
- Materialization fingerprints: Per-partition hashes for cache validation
- Serialized preprocessors: Fit on train splits only, versioned

**Advantage:** P1 can reproduce not just models but the **entire preprocessing pipeline**—a step beyond the paper's checkpoint approach.

---

## 8. Known Limitations & Deviations

### Limitations Inherited from Paper (Section VII):

1. ✅ **Cross-domain false positives**: Edge→ToN edge case (43.44% Macro F1). P1 will experience this with BoT-IoT as source.
2. ✅ **Feature richness trade-off**: 4 features in P1 vs. 6 in paper for common profile.
3. ✅ **AE dataset-dependency**: Reconstruction behavior varies; P1 documents this per-dataset.
4. ✅ **Static splits not live testbeds**: P1 uses offline datasets as paper does.
5. ✅ **XAI attribution ≠ causality**: P1 defers XAI to P4.

### P1-Specific Deviations:

1. **Dataset Substitution**: ToN-IoT (paper) → BoT-IoT + N-BaIoT (P1)
   - **Rationale**: Richer domain-shift analysis; N-BaIoT source-aggregate profile for behavior-based anomaly research
   - **Trade-off**: Cannot directly replicate paper's 86.57%/86.96% transfer scores

2. **Common Feature Reduction**: 6 → 4 features
   - **Rationale**: BoT-IoT incompatibility with SYN/ACK/RST decomposition
   - **Mitigation**: Explicit audit + compatibility matrix; N-BaIoT marked `N/A`

3. **Adversarial Testing Out of Scope**:
   - **Rationale**: P1 = ML foundation; P4 = adversarial research
   - **Handoff**: P4 receives component-level outputs (RF probabilities, MLP logits, AE scores) for downstream attack injection

4. **Ensemble Weight Tuning Deferred**:
   - **Current**: Hardcoded 0.5/0.5 (Eq. 1 from paper)
   - **Plan**: P2/P3 validation-only weight selection per dataset
   - **Impact**: In-domain performance may differ slightly from paper's 94.29%/97.73%

---

## 9. P1 Completion Roadmap vs. Paper Results

### Current State (Milestone 2.5):
- ✅ Representation frozen (4 feature levels, 26 total fields across in-domain)
- ✅ Dataset audit complete (4 datasets, 10M+ rows total)
- ✅ Splitting policy defined and validated
- ✅ Leakage audit complete
- ✅ Model scaffolding in place

### P2 / P3 Tasks (To Match Paper):

| Task | Paper Section | Timeline | Expected Output |
|------|------|----------|---|
| Train RF, MLP, AE on all 4 datasets | IV.C, VI.A | P2 | Trained models, metrics per dataset |
| Same-dataset in-domain evaluation | VI.A | P2 | Attack F1 scores (target: ~94-97%) |
| Tune ensemble weights | IV.C, IV.D | P2/P3 | Per-dataset weight selection |
| Cross-domain (4×4 matrix) evaluation | VI.B | P3 | Cross-transfer Attack F1 (target: ~85-87%) |
| Generate reproducibility artifacts | VI.E | P3 | Frozen checkpoints, retraining variance |

### P4 Tasks (Out of P1 Scope):

| Task | Paper Section | Responsible | Depends On |
|------|------|---|---|
| XAI feature attribution (Spearman ρ) | VI.C | P4 | P1 trained models |
| Constrained FGSM/PGD-10 adversarial | VI.D | P4 | P1 component outputs |
| AE catch rate on perturbed samples | VI.D | P4 | P1 anomaly scores |
| Risk layer decision logic (3-way) | IV.C | P4 | P1 ensemble + anomaly |

---

## 10. Recommendations for P2/P3 Alignment

### To Maximize Paper Alignment:

1. **Priority 1: Match Paper's Ensemble Weights**
   - Use validation-driven weight selection (currently hardcoded 0.5/0.5)
   - Target: Within ±1% of paper's reported Attack F1

2. **Priority 2: Document Cross-Domain Asymmetry**
   - Paper reports Edge→ToN (high recall, low precision) vs. ToN→Edge (balanced)
   - P1 will observe analogous asymmetry (e.g., CICIDS2017→BoT-IoT vs. reverse)
   - Create comparative table in P2 results

3. **Priority 3: Validate Feature Reduction Impact**
   - Train one dataset (e.g., Edge-IIoTset) with paper's 6-feature `Fcommon` vs. P1's 4-feature profile
   - Measure Attack F1 delta
   - Document in ablation study

4. **Priority 4: Prepare P4 Handoff**
   - Export component-level outputs (RF confidences, MLP logits, AE reconstruction errors)
   - Version model artifacts with metadata
   - Provide API for P4 to inject adversarial perturbations on labeled test sets

### To Strengthen Beyond Paper:

1. **N-BaIoT Behavior Analysis**
   - Explore why source-aggregate temporal features excel on N-BaIoT in-domain (if observed)
   - Compare AE reconstruction error distribution (paper vs. P1)

2. **Materialization Efficiency**
   - P1 has measured throughput (4K–45K rows/sec depending on dataset)
   - Document as operational metric for production deployment discussions

3. **Splitting Robustness**
   - Stratification by attack category / source device
   - Report variance across multiple random seeds (paper reports ±0.06 pp)

---

## 11. Critical Questions for Implementation

1. **ToN-IoT Absence**
   - Is ToN-IoT data unavailable, or deliberately substituted?
   - If substitution, should paper be updated to cite BoT-IoT evaluation instead?

2. **Feature Reduction Justification**
   - Will P1 attempt to engineer SYN/ACK/RST equivalents from BoT-IoT raw logs?
   - Or accept 4-feature limit as dataset-incompatibility boundary?

3. **Ensemble Weight Selection**
   - Validation-driven tuning: per dataset, or global?
   - Should P1 freeze weights after P2, or re-tune in P3 per cross-domain pair?

4. **AE Anomaly Threshold**
   - Paper uses 0.80 for AE catch rate; is this also used for risk-layer decisions?
   - P1 should document threshold selection methodology (ECDF quantile? validation curve?).

5. **Model Reproducibility Variance**
   - Paper reports ±0.06 pp retraining variance
   - P1 should target same tolerance; document seed / hardware differences

---

## 12. Conclusion

**P1 is on-track to deliver the foundational ML subsystem described in the paper, with necessary scope adjustments for 4-dataset evaluation.**

| Aspect | Status | Confidence |
|--------|--------|-----------|
| **Representation & Preprocessing** | ✅ Complete | High |
| **Model Scaffolding** | ✅ Complete | High |
| **In-Domain Training & Evaluation** | ⏳ P2 | High |
| **Cross-Domain Transfer** | ⏳ P2/P3 | Medium |
| **Reproducibility Artifacts** | ⏳ P3 | High |
| **Adversarial Testing & XAI** | ❌ Out of Scope | N/A |

**To align with paper's reported results (94.29% / 97.73% Attack F1, 86.57% / 86.96% cross-domain):**
- Expect ±3-5% variance due to dataset substitution (BoT-IoT vs. ToN-IoT) and feature reduction (4 vs. 6)
- Validate ensemble weights and AE thresholds via validation-driven selection
- Document asymmetric cross-domain performance in comparative tables

**Paper's innovations directly implemented in P1:**
- ✅ Multi-model detection (RF + MLP + AE)
- ✅ Feature harmonization via explicit level hierarchy
- ✅ Privacy-preserving flow-level features
- ✅ Reproducibility-first artifact management
- ⏳ XAI attribution agreement (P4 responsibility)
- ⏳ Adversarial robustness evaluation (P4 responsibility)

---

**Document Owner:** P1 Implementation Team
**Next Review:** Post-P2 Model Training
**Related Docs:** IMPLEMENTATION_PLAN.md, MILESTONE_2_5_VALIDATION.md, P1_IMPLEMENTATION_SUMMARY.md
