# Conference Review: "An Adversarially Robust and Privacy-Preserving AI Framework for Explainable Intrusion Detection in IoT Environments"

**Review Date:** August 26, 2026
**Submission Type:** Full Conference Paper
**Estimated Length:** 6 pages (IEEE format)
**Review Format:** Structured academic review with scoring

---

## 1. Summary

This paper proposes a multi-model IoT intrusion detection system (IDS) combining a Random Forest, feed-forward MLP, and benign-trained autoencoder. The supervised models form a probability-based ensemble while the autoencoder provides anomaly-based detection. The framework is evaluated on Edge-IIoTset and ToN-IoT Network datasets using 6 common features for zero-adaptation cross-domain transfer. Reported results show 94.29%/97.73% Attack F1 in-domain and 86.57%/86.96% cross-domain using only 6 semantically harmonized features.

---

## 2. Strengths

### 2.1 Well-Motivated Problem (Strong)
- **Clear articulation of gaps:** Section II.E systematically identifies four recurring limitations in existing IDS research (domain confinement, fragmented defenses, post-hoc explainability, privacy/cross-domain dependence)
- **Table I effectively frames contributions:** Maps research gaps to proposed solutions
- **Practical relevance:** IoT security remains an urgent problem; adversarial robustness + cross-domain + explainability are simultaneously important

### 2.2 Comprehensive Experimental Design (Strong)
- **Multi-faceted evaluation:** Not just classification accuracy but cross-domain transfer, explainability, and adversarial robustness in a single framework
- **Strict zero-adaptation protocol:** No target labels, retraining, or fine-tuning—a realistic and challenging test
- **Bidirectional transfer:** Reports both Edge→ToN and ToN→Edge, revealing asymmetry (precision 76.33% vs. 87.00%)
- **Reproducibility artifacts:** "Frozen checkpoints, evaluation artifacts, and independent retraining verification" (Section VI.E)

### 2.3 Privacy-Aware Design (Moderate)
- **Payload abstraction:** Uses derived flow-level features, not raw packets—reduces sensitive data exposure
- **Feature harmonization:** 6 common features enable cross-domain evaluation without raw data exchange
- Credible step toward federated/distributed learning (cited in Section II.D)

### 2.4 Honest Limitations Discussion (Strong)
- Section VII explicitly acknowledges: cross-domain false positives (43.44% Macro F1 on Edge→ToN), feature richness trade-off, AE dataset-dependency, no live testbeds, XAI attribution ≠ causality
- Acknowledges ±0.06 pp retraining variance—good transparency
- Discusses why AE effectiveness is dataset-dependent (reconstruction error distributions)

### 2.5 Clear Presentation (Moderate)
- Figure 1 provides good system overview
- Tables are well-structured (I, II, III, IV, V, VI, VII)
- Methodology sections are generally clear

---

## 3. Weaknesses

### 3.1 Limited Novelty (Moderate Concern)
- **Individual components not novel:**
  - Random Forest + MLP ensembles are standard practice
  - Benign-trained autoencoders for anomaly detection are well-established (cited work includes [24])
  - Reconstruction-error ECDF normalization is standard
- **Contribution is primarily "intelligent combination" and empirical validation**, not methodological innovation
- **Claim of novelty:** The paper frames this as "multi-model detection + feature harmonization + zero-adaptation transfer + XAI analysis + adversarial evaluation," but this is an engineering combination of established techniques rather than a novel method
- **Missing:** Any algorithmic or theoretical contribution; no new loss function, training strategy, or model variant

### 3.2 Experimental Scope Limitations (Significant Concern)

#### Adversarial Evaluation
- **Limited threat model:** Only normalized feature-space perturbations (ε ∈ {0.05, 0.1, 0.2, 0.3}), not packet-level attacks
- **MLP-only targeting:** FGSM/PGD evaluated only against MLP, not RF or AE independently
- **No adaptive attacks:** Assumes threat actor doesn't know ensemble + AE structure; real adversaries may craft perturbations that fool all components
- **Small scale:** ε = 0.30 is relatively small for normalized features; unclear what this represents in raw feature space
- **Missing comparison:** No baseline defense (adversarial training, certified defenses) to contextualize robustness claims

#### Cross-Domain Transfer Issues
- **Asymmetric results unexplained:** Edge→ToN achieves 99.98% attack recall but only 76.33% precision. ToN→Edge is 87%/87%. Why?
  - Likely due to class imbalance (Edge: 84.6% attack, ToN: 76.3% attack) and distribution mismatch
  - Paper does not investigate root cause or propose mitigation
- **Macro F1 collapse:** Edge→ToN's 43.44% Macro F1 is problematic—indicates severe benign-class misprediction
  - Not suitable for deployment without additional tuning
  - Paper acknowledges but doesn't propose solutions (e.g., cost-sensitive learning, threshold tuning, domain adaptation)

#### Feature Reduction Justification
- Paper claims **6 common features** but Section IV.B specifies: duration, src_bytes, TCP, UDP, ICMP, well-known-port indicator
- **Issue:** TCP, UDP, ICMP are protocol flags (3 binary features), not 3 independent features
- **Counting ambiguity:** Are these 6 or 10 features? Table III is titled "FOUR-LEVEL FEATURE HIERARCHY" with 13 total for in-domain, but the exact composition of F_common is not machine-readable in the paper
- **Packet-count exclusion:** Paper states "excluded from F_common because record granularity differs between datasets" but provides no evidence or alternative
- **Missing ablation:** No feature-importance analysis showing which of these 6 are actually predictive

### 3.3 Model Tuning & Hyperparameter Selection (Moderate Concern)
- **RF & MLP hyperparameters appear fixed:** 100 trees, max_depth=15, MLP 128→64→32→1, dropout=0.2
  - **No justification:** Why depth 15? Why dropout 0.2? Why not 0.1 or 0.3?
  - **No ablation:** No table showing sensitivity to these choices
  - **Concern:** Could these be overfit to in-domain performance?
- **Ensemble weights hardcoded at 0.5/0.5 (Eq. 1):**
  - Paper mentions "validation-only weight selection" but reports no results
  - Likely suboptimal; RF may outperform MLP on these datasets
  - **Missing ablation:** No 0.6/0.4, 0.7/0.3, etc. comparisons
- **AE threshold at 0.80 for catch rate (Section V.E):**
  - **Arbitrary?** How was 0.80 chosen? Validation curve not shown
  - **Unclear:** Is 0.80 also used in risk-layer decisions, or only for evaluation?

### 3.4 Statistical Rigor & Significance Testing (Moderate Concern)
- **No confidence intervals:** Attack F1 scores lack error bars or ±CI
- **Single train/val/test split:** No k-fold or cross-validation reported
  - Retraining variance (±0.06 pp) is mentioned but not demonstrated on the reported models
- **No significance tests:** E.g., is 94.29% significantly different from 97.73%? From other baselines?
- **Missing statistical context:**
  - No comparison with published IDS baselines on same datasets
  - Table IV/V report "authoritative benchmark" but relative to what?

### 3.5 Explainability Analysis Underdeveloped (Significant Concern)
- **Spearman ρ agreement alone insufficient:**
  - ρ = 0.8632 on Edge-IIoTset is good, but ρ = 0.6252 on ToN-IoT (p=0.0223) is moderate with borderline significance
  - **No interpretation:** Does 0.6252 mean models agree on top features? Or is agreement weak?
  - **No visualization:** Feature importance rankings not shown; hard to assess substantive alignment
- **Missing XAI depth:**
  - No SHAP, LIME, or saliency maps (mentioned in related work but not used)
  - No per-sample explanations for misclassified cases
  - No "why did this attack evade the supervised model but get caught by AE?"
- **Cross-domain XAI absent:** Do feature importances shift under domain transfer? Not explored

### 3.6 Autoencoder Coverage Inconsistency (Significant Concern)
- **Highly dataset-dependent:**
  - Edge-IIoTset: 100% AE catch rate on PGD perturbations (Table VII, complete evasion = 0%)
  - ToN-IoT: 19.4% catch rate on PGD at ε=0.30 (complete evasion = 8.42%)
  - **Root cause analysis weak:** Paper states benign reconstruction error is higher on ToN-IoT (0.168 vs. 8.90×10⁻⁵) but doesn't propose solutions
- **Question:** If AE fails on ToN-IoT adversarial samples, can this framework be deployed there reliably?
- **Missing mitigation:** No adaptive thresholding, no domain-specific AE training, no discussion of this trade-off

### 3.7 Reproducibility Gaps (Moderate Concern)
- **Code availability not mentioned:** Paper does not declare code release (GitHub, supplementary materials, etc.)
- **Random seed not specified:** How are weight initializations seeded? Is reproducibility guaranteed?
- **Dataset availability unclear:** Edge-IIoTset and ToN-IoT are public, but no DOI/version specified
- **Missing details:**
  - Exact PyTorch/scikit-learn versions (paper says "PyTorch 2.2, scikit-learn 1.4" but no requirements.txt reference)
  - Hardware details for training time / inference latency
  - Whether experiments run on CPU or GPU

### 3.8 Writing & Presentation Issues (Minor-Moderate)

**Clarity issues:**
- **Abstract overloads claims:** "adversarially aware IDS framework" + "explainable" + "privacy-preserving" + "zero-adaptation transfer" all in 150 words
  - Leads to underdeveloped treatment of each claim in the paper
- **Notation inconsistencies:**
  - Datasets denoted D_A, D_B but then X_A, X_B for feature spaces without clear definition
  - F_common introduced but never formalized as a set; Table III suggests 6 or 10 features with ambiguity
- **Missing related work depth:**
  - Cites recent ensemble IDS papers ([16], [17]) but doesn't compare results or explain why this ensemble is better
  - Section II.D mentions SHAP, LIME, attribution methods but paper only uses Spearman ρ

**Figure & table issues:**
- **Figure 1:** Shows architecture well but lacks flow diagram for risk-layer decision logic (attack vs. suspicious vs. benign)
- **Figure 3 (adversarial results):** 4 subplots with different scales make cross-dataset comparison hard
  - Why is ToN-IoT PGD-10 so much worse than Edge-IIoTset?
  - Speculation without evidence

---

## 4. Technical Soundness

### Issues Found:

1. **Zero-adaptation protocol is sound** but asymmetric cross-domain results (Section VI.B) suggest data leakage or dataset-specific tuning. Investigation needed.

2. **FGSM/PGD implementation:**
   - Bounded to feature space—reasonable for privacy-preserving features but limits real-world applicability
   - No mention of gradient computation; assume TensorFlow/PyTorch autograd, but should be stated

3. **AE ECDF normalization (Section IV.C):**
   - Fitted on benign validation data—correct
   - But what if test set contains novel benign patterns with high reconstruction error?
   - Paper doesn't discuss false-alarm rate vs. catch rate trade-off

4. **Ensemble probability fusion (Eq. 1):**
   - Assumes RF and MLP output well-calibrated probabilities
   - No calibration step (Platt scaling, temperature scaling) mentioned
   - MLP with softmax? Binary cross-entropy? Not specified

---

## 5. Results & Claims Assessment

### Claimed Results:
| Claim | Evidence | Assessment |
|---|---|---|
| "94.29% and 97.73% Attack F1 in-domain" | Table IV | ✓ Clear but single split; no CI |
| "86.57%/86.96% under zero-adaptation" | Table V | ✓ Claimed but asymmetric; unexplained |
| "Strong RF–MLP attribution agreement (ρ=0.8632)" | Table VI | ✓ Edge-IIoTset strong; ToN-IoT moderate (ρ=0.6252) |
| "AE provides consistent anomaly coverage" | Table VII | ⚠️ Only on Edge-IIoTset; fails on ToN-IoT (19.4% catch) |
| "Adversarially robust" | Section VI.D | ⚠️ Low supervised ASR (≤10.45%) but limited threat model |

### Verdict:
- **In-domain results are solid** but not exceptional (comparable to cited ensembles)
- **Cross-domain results are interesting but incomplete** (asymmetry needs explanation)
- **Adversarial robustness overclaimed** (limited to bounded perturbations, not adaptive attacks)
- **Explainability and privacy are aspirational** rather than deeply demonstrated

---

## 6. Significance & Impact

### For the Community:
- ✓ **Addresses real problem** (cross-domain IoT IDS)
- ✓ **Multi-faceted evaluation** is commendable
- ✗ **Limited novelty** in methods; primarily engineering validation
- ✗ **Not likely to change practice** (existing ensemble IDS are comparable)
- ? **Code release unknown** (critical for adoption)

### Venue Suitability:
- **Appropriate venues:** IEEE/ACM IoT conferences, security workshops, machine learning for security tracks
- **Likely not suitable for:** Top-tier ML/AI conferences (NeurIPS, ICML) or top security venues (CCS, USENIX Security)—novelty too limited
- **Best fit:** IEEE Internet of Things Journal, IEEE Access, or IoT security-focused venues

---

## 7. Missing Experiments / Analysis

1. **Ablation studies:**
   - Remove AE; measure performance (RF+MLP only)
   - Vary ensemble weights (0.4/0.6, 0.6/0.4, etc.)
   - Feature importance ranking by dataset
   - Impact of F_common feature reduction (6 vs. 13)

2. **Baseline comparisons:**
   - Compare to published IDS results on same datasets
   - Compare to single-model (RF alone, MLP alone) performance
   - Compare to other multi-model approaches

3. **Adversarial defenses:**
   - Adversarial training on MLP
   - Certified defenses (randomized smoothing, etc.)
   - Adaptive attacks (TRADES, AutoAttack)

4. **Explainability depth:**
   - SHAP values or LIME for individual predictions
   - Feature importance distribution plots
   - Case studies of correctly vs. incorrectly classified attacks

5. **Cross-domain root-cause analysis:**
   - Why Edge→ToN achieves 99.98% recall but 76.33% precision?
   - Distribution shift visualization (t-SNE, domain adaptation metrics)
   - Cost-sensitive tuning for Macro F1 improvement

6. **Statistical validation:**
   - k-fold or stratified cross-validation
   - Confidence intervals on all metrics
   - Retraining variance on the final models (claimed ±0.06 pp but not shown for these results)

---

## 8. Detailed Comments

### Comment 1: Feature Harmonization
The paper claims **6 common features** (Section IV.B) are semantic and cross-dataset compatible, but doesn't validate this claim empirically. Did you test whether:
- Duration field has same semantics across CICIDS2017/Edge-IIoTset/ToN-IoT (e.g., flow-level? packet-level?)?
- Source bytes includes retransmissions uniformly across datasets?
- Protocol flags (TCP/UDP/ICMP) are mutually exclusive and comprehensively labeled?

**Recommendation:** Add validation table showing feature statistics (mean, std, min, max, missing %) for each dataset on each of the 6 features. This proves semantic compatibility.

### Comment 2: Cross-Domain Asymmetry
Edge→ToN achieves 76.33% precision (99.98% recall) while ToN→Edge achieves 87% precision (86.91% recall). This 11 pp gap is huge and suggests:
1. Benign traffic in ToN-IoT is fundamentally different from Edge-IIoTset
2. Attack traffic distributions are poorly aligned
3. Class imbalance (Edge 84.6% attack vs. ToN 76.3%) causes training bias

**Recommendation:** Investigate via:
- Distribution plots (e.g., kernel density estimation for each feature in attack/benign/domain)
- DANN/CORAL domain adaptation baseline
- Cost-sensitive learning to address imbalance

### Comment 3: AE Anomaly Threshold
Equation relating reconstruction error to S_ae ∈ [0,1] is not shown. Is it:
- $S_{ae} = \text{quantile}(\text{MSE}_{\text{benign}}, p=\text{MSE})$? (Empirical ECDF)
- $S_{ae} = 1 - \exp(-\text{MSE})$? (Some sigmoid)
- $S_{ae} = \min(\text{MSE} / \text{threshold}, 1)$?

Also, is catch-rate threshold (0.80) same as risk-layer threshold? Clarify.

### Comment 4: XAI Correlation ≠ Explanation
Table VI reports Spearman ρ but this only measures **ranking agreement**, not **agreement on interpretation**. For example:
- RF: `[mqtt_msgtype=0.4, duration=0.3, dst_port=0.2, ...]`
- MLP: `[duration=0.4, mqtt_msgtype=0.35, dst_port=0.15, ...]`
- Spearman ρ ≈ 0.95 (high agreement), but do models actually rely on same features for the same reasons?

**Recommendation:** Add SHAP or saliency map examples showing side-by-side RF vs. MLP decisions on 5-10 representative samples.

### Comment 5: No Baseline Comparisons
The paper does not cite baseline IDS results on Edge-IIoTset or ToN-IoT from related work. For instance:
- How does 94.29% Attack F1 compare to CNN-based approaches in [4]?
- How does it compare to single RF or MLP?
- Is 86.57% cross-domain F1 state-of-the-art for zero-adaptation?

**Recommendation:** Add Table VIII comparing this framework to 2-3 cited ensemble baselines with published results.

---

## 9. Minor Issues

- **Typo/grammar:** "constrained FGSM and PGD-10 experiments evaluate the framework's resistance to feature-space evasion" (abstract) should clarify "bounded perturbation budget evasion"
- **Citation format inconsistency:** Some [1]–[3], some [4], [5]; use consistent format
- **Table II:** "ToN-IoT Network" is redundant (Network twice); use "ToN-IoT" for brevity
- **Figure 3:** Subplots labeled by dataset + attack but not by ε value clearly; hard to correlate with Table VII

---

## 10. Overall Assessment

### Strengths Summary:
1. Well-motivated problem with clear research gaps
2. Comprehensive multi-faceted evaluation (in-domain, cross-domain, XAI, adversarial)
3. Honest discussion of limitations
4. Privacy-aware design with feature harmonization
5. Clear presentation overall

### Weaknesses Summary:
1. **Limited novelty** — engineering combination of standard techniques (RF + MLP + AE)
2. **Incomplete adversarial evaluation** — bounded feature-space only, no adaptive attacks
3. **Asymmetric cross-domain results unexplained** — raises questions about data leakage
4. **Underdeveloped XAI** — Spearman ρ alone insufficient; no SHAP/LIME
5. **No baselines** — impossible to assess if 94.29% is state-of-the-art or baseline
6. **Hyperparameter tuning not justified** — why depth=15, dropout=0.2, weights=0.5/0.5?
7. **Statistical rigor** — no CI, no k-fold, single split results
8. **AE coverage inconsistent** — works perfectly on Edge-IIoTset, poorly on ToN-IoT

---

## 11. Recommendation to Conference Committee

**Borderline Accept / Major Revisions Needed**

### If to Marginal Acceptance Tier:
This paper makes a solid engineering contribution by carefully combining established techniques for IoT IDS evaluation across domains. Strengths include honest limitations discussion, reproducibility focus, and multi-faceted evaluation. However, limited methodological novelty, incomplete experimental analysis (missing baselines, ablations, statistical rigor), and underdeveloped claims (explainability, adversarial robustness) prevent strong acceptance.

### Decision:
- **Reject** for top-tier ML/AI venues (NeurIPS, ICML, ICLR) — insufficient novelty
- **Marginal Accept / Major Revisions** for IEEE IoT / security venues
  - Requires: (1) baseline comparisons, (2) cross-domain root-cause analysis, (3) ablation studies, (4) SHAP/saliency maps, (5) k-fold validation, (6) statistical testing, (7) code release commitment

### Reviewer Recommendation: **REVISIONS NEEDED — Resubmit with above fixes**

---

## 12. Questions for Authors

1. Why does Edge→ToN achieve 99.98% recall but 76.33% precision? Root cause of asymmetry?
2. How were hyperparameters (depth=15, dropout=0.2, weights=0.5/0.5) selected? Ablation studies?
3. Have you compared to baseline ensemble approaches or single-model IDS on the same datasets?
4. Why is AE anomaly detection so much more effective on Edge-IIoTset (100% catch) vs. ToN-IoT (19.4% catch)? Mitigation strategy?
5. Can you provide SHAP/saliency visualizations to validate that RF and MLP rely on same features for same reasons (not just rank correlation)?
6. What is the inference latency per packet? Memory footprint? Deployability on edge devices?
7. Will code be released? If so, on what timeline and platform?

---

## 13. Detailed Scoring (If Using a 1-5 Scale)

| Criterion | Score | Justification |
|---|---|---|
| **Novelty** | 2/5 | Engineering contribution; no new methods |
| **Technical Quality** | 3/5 | Sound overall but incomplete ablations, hyperparameter justification weak, statistical rigor limited |
| **Experimental Rigor** | 3/5 | Good breadth but missing baselines, ablations, statistical tests; single-split results |
| **Clarity & Presentation** | 4/5 | Generally clear; good figures/tables; minor notation/writing issues |
| **Significance & Impact** | 2/5 | Addresses real problem but unlikely to change practice; limited novelty reduces impact |
| **Reproducibility** | 3/5 | Mentions frozen checkpoints but no code release; hyperparameter details scattered |
| **Limitations Discussion** | 5/5 | Excellent—Section VII is honest and comprehensive |
| **Overall Accept Decision** | **BORDERLINE** | For IoT/security venues: REVISIONS NEEDED. For top-tier ML: REJECT. |

---

## Final Verdict

**Suitable for:** IEEE IoT venues, security workshops, applied ML conferences
**Not suitable for:** NeurIPS, ICML, CCS, USENIX Security, top-tier ML conferences
**Recommendation:** Major revisions addressing baselines, ablations, XAI depth, and cross-domain root-cause analysis. With fixes, could achieve acceptance at IoT-focused venues.

---

**Reviewer:** AI / ML Systems Researcher
**Review Confidence:** High (comprehensive evaluation methodology)
**Submitted:** August 26, 2026
