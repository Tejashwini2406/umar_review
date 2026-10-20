# Transformation Guide: Paper Submission to Top-Tier Venues
## NeurIPS, ICML, ICLR, CCS, USENIX Security

**Last Updated:** August 26, 2026
**Current Status:** Borderline / Needs Major Work
**Target Venues:** All 5 listed (with venue-specific strategies)
**Estimated Timeline:** 6-12 months of additional research

## Current Target Update — October 18, 2026

The active target is **USENIX Security '27, Cycle 2**, not the multi-venue pathways below. The official CFP lists mandatory registration on January 19, 2027, paper submission on January 26, 2027, and artifact submission on January 29, 2027. Submissions allow at most 13 body pages and require an Open Science Appendix; an Ethics appendix is strongly encouraged. See the [official CFP](https://www.usenix.org/conference/usenixsecurity27/call-for-papers).

**Readiness: not ready and not yet a defensible top-tier submission.** The current selected research direction is the Behavioral Consistency Layer (BCL), but its local results are exploratory, the ToN-IoT test split informed a scoring change, BCL detection is weak on two datasets, Edge-IIoTset is order-sensitive, and adaptive evasion has not been evaluated. All four current test sets have already been inspected; they cannot serve as a fresh confirmatory holdout. The IEEE draft is withdrawn and is not a suitable USENIX submission draft.

The remaining work is gated on an independent holdout/dataset, a pre-specified security threat model, adaptive evaluation of the full IDS+BCL path, and evidence that the proposed layer provides useful security value over detector-only baselines. If those gates fail, change or stop the BCL claim rather than filling in expected results. The recommendations below are historical generic venue guidance and are superseded by this target update.

---

## Executive Summary: The Challenge

**Current Assessment:** Engineering contribution (RF + MLP + AE ensemble) with limited novelty.

**Verdict by Venue:**
- ❌ **NeurIPS/ICML/ICLR:** REJECT — Insufficient ML/AI methodological novelty
- ❌ **CCS/USENIX Security:** REJECT — Weak adversarial guarantees; limited security proofs

**To Become Acceptable, You Must Add ONE of:**
1. **Novel ML method** (new ensemble strategy, theoretical guarantees) → NeurIPS/ICML/ICLR
2. **Novel security contribution** (certified defenses, formal security analysis) → CCS/USENIX Security
3. **Both** (hard but highest impact) → All five venues

---

## PHASE 1: CHOOSE YOUR PATH (Week 1-2)

### Option A: ML-First Path (→ NeurIPS/ICML/ICLR)
**If you want methodological novelty for machine learning venues**

**Core Insight Needed:** Something new about *how* the ensemble works or *why* it transfers across domains.

**Examples of ML-novelty you could add:**
- Domain-aware ensemble weights (learned per-dataset via adversarial domain adaptation)
- Novel cross-domain loss function (e.g., domain-invariant feature representation)
- Theoretical analysis of ensemble generalization under domain shift
- New feature harmonization algorithm (instead of manual 6-feature selection)

### Option B: Security-First Path (→ CCS/USENIX Security)
**If you want formal security contributions for security venues**

**Core Insight Needed:** Something new about *guarantees* against adversarial attacks.

**Examples of security-novelty you could add:**
- Certified defenses (provable robustness radius ε for which model is robust)
- Formal security analysis of ensemble voting vs. single model
- Privacy proof (differential privacy guarantees for cross-domain evaluation)
- Threat model formalization + hardness results

### Option C: Hybrid Path (→ All Five, Hardest)
**Combine ML innovation + security guarantees**

**Example:** "Domain-Adaptive Ensemble with Certified Robustness for IoT IDS"
- ML contribution: Domain-aware weight learning
- Security contribution: Certified robustness bounds per domain

---

## PHASE 2: ML-FIRST PATH IN DETAIL

### If You Choose: Novelty in Ensemble Learning

#### Step 1: Identify the Novel Contribution (Week 2-3)

**Current Problem:** Why does cross-domain transfer fail?
- Edge→ToN: 99.98% recall, 76.33% precision (asymmetric)
- ToN→Edge: 87% precision, 86.91% recall (balanced)

**Your Innovation:** Design an ensemble that **learns what weights work best per domain**.

**Proposed Contribution: "Domain-Adaptive Ensemble Weighting via Validation-Set Domain Shift"**

Instead of hardcoded 0.5/0.5, learn:
$$w_1(D), w_2(D), w_3(D) = \text{argmax}_{w} F1(\text{val}_D | w_1 P_{RF} + w_2 P_{MLP} + w_3 S_{AE})$$

Where $D$ is the domain/dataset, and weights are learned *per-domain* on validation data.

**Why this is novel:**
- Standard ensemble methods use fixed weights (e.g., majority voting)
- This paper proposes **domain-adaptive weights** — weights vary by dataset
- Theoretical claim: Improves cross-domain transfer by up to X% (measure this)

**Papers to cite:**
- [Stacking, Wolpert 1992] — foundational multi-level learning
- [Domain adaptation via importance weighting, Cortes et al. 2010]
- [Adaptive ensemble methods, Zhou 2012]

---

#### Step 2: Implement & Validate the Innovation (Week 4-8)

**Experiment 1: Per-Dataset Weight Tuning**
```
For each dataset D in {Edge, ToN, BoT-IoT, N-BaIoT}:
  Train RF, MLP, AE on train(D)
  For weights w in grid {0.1, 0.2, ..., 0.9}:
    Evaluate F1, Precision, Recall on val(D)
  Select w* = argmax F1(val(D))
  Report metrics on test(D)
```

**Expected Result:** Per-domain weights improve in-domain Macro F1 by 2-5 pp.

**Experiment 2: Cross-Domain Transfer with Domain-Adaptive Weights**

```
For each transfer direction (D_source → D_target):
  Use weights learned on D_source
  Apply to D_target test set
  Measure: Attack F1, Precision, Recall, Macro F1
  Compare to:
    - Fixed 0.5/0.5 weights (current paper)
    - Retuned weights on D_target (oracle baseline, unrealistic)
```

**Expected Result:** Domain-adaptive weights improve cross-domain Macro F1 by 5-10 pp (reduces asymmetry).

**Deliverable:** Table IX showing per-domain weights + ablation.

---

#### Step 3: Add Theoretical Analysis (Week 9-10)

**Theorem You Should Prove:**

**Claim:** "If RF and MLP have complementary error patterns (low correlation), domain-adaptive weighting achieves near-optimal ensemble on validation set."

**Proof Sketch:**
- Assume RF error rate ε₁, MLP error rate ε₂, correlation ρ
- Ensemble error ≤ f(ε₁, ε₂, ρ, w₁, w₂)
- Domain-adaptive weights can achieve ≈ f(ε₁, ε₂, ρ, w₁*, w₂*)
- Cross-domain: generalization bound using domain divergence (e.g., H-divergence)

**Cite:**
- [Ensembles: Foundations and Algorithms, Zhou 2012]
- [A theory of learning from different domains, Ben-David et al. 2010]

**Not Required but Nice:**
- Convergence rate (e.g., O(1/√n) in number of samples)
- Sample complexity for weight learning
- Bounds on transfer loss as function of domain distance

---

#### Step 4: Position as ML Contribution (Week 11-12)

**Rewrite Abstract:**

❌ OLD: "presents an adversarially aware IDS framework combining RF, MLP, and AE"

✅ NEW: "proposes domain-adaptive ensemble weighting via validation-set domain shift. We show that learned per-domain weights improve cross-domain IoT IDS transfer by up to 8.2% Attack F1 while maintaining in-domain performance. Theoretical analysis using H-divergence bounds proves generalization under dataset mismatch."

**Restructure Paper:**

| Section | Focus | Length |
|---------|-------|--------|
| I. Intro | Problem: ensemble weights are fixed; cross-domain transfer fails | 1 page |
| II. Related Work | Ensemble learning + domain adaptation methods | 1 page |
| III. Method | **NEW:** Domain-adaptive weighting algorithm (2 pages) | 2 pages |
| IV. Theory | **NEW:** Generalization bounds and sample complexity | 1 page |
| V. Experiments | (1) Per-domain weight tuning, (2) Cross-domain transfer with adaptive weights, (3) Ablations, (4) Baselines | 2 pages |
| VI. Discussion | When/why adaptive weights help; failure modes | 0.5 page |

**Total: 7-8 pages (IEEE format)**

---

#### Step 5: Benchmark Against Baselines (Week 12-13)

**Required Baselines:**

1. **Fixed weights (current paper):** 0.5/0.5 RF + MLP, plus AE

2. **Majority voting:** Predict attack if ≥2/3 models vote attack

3. **Stacking:** Train meta-learner (logistic regression) on held-out validation set using RF, MLP, AE scores as features

4. **DANN (Domain-Adversarial Neural Networks):** Adversarial loss to learn domain-invariant features; use single MLP on transformed features

5. **TrAdaBoost:** Transfer boosting method (cited in domain adaptation literature)

**Expected Table X: Cross-Domain Transfer Comparison**

```
Transfer      Majority   Stacking   DANN   TrAdaBoost   Adaptive-Weights   Oracle (Retrain)
Edge→ToN      78.1       82.3       79.5   84.1         86.8              94.2
ToN→Edge      85.2       86.1       84.7   85.9         87.3              97.7
BoT-IoT→...   [fill]     [fill]     [fill]  [fill]      [fill]            [fill]
```

**Interpretation:**
- Adaptive weights outperform fixed weights and Majority voting
- Competitive with or better than DANN/TrAdaBoost (if tuned well)
- Still far from oracle (retraining) — honest limitation

---

### Submission Strategy: ML-First Path

**Target Venue Priority:**
1. **NeurIPS** (top priority; broad audience, novelty valued)
2. **ICML** (secondary; strong if you have theoretical guarantees)
3. **ICLR** (tertiary; if you emphasize representational insights)

**CCS/USENIX Security:** Won't accept without security contribution; skip or combine with Path B.

**Key Claims for Review Committee:**
- "First work to study domain-adaptive ensemble weighting for cross-domain IoT IDS"
- "Theoretical bounds on cross-domain generalization using H-divergence"
- "8.2 pp Attack F1 improvement over fixed-weight ensembles under domain shift"

---

## PHASE 3: SECURITY-FIRST PATH IN DETAIL

### If You Choose: Novelty in Formal Security

#### Step 1: Formalize Security Guarantees (Week 2-3)

**Current Problem:** Paper claims "adversarially robust" but:
- Only bounded feature-space perturbations (ε ∈ {0.05, 0.1, 0.2, 0.3})
- No formal guarantee (e.g., "for all perturbations with ||δ|| ≤ ε, model is robust")
- AE anomaly detection fails on ToN-IoT (19.4% catch rate)

**Your Innovation: "Certified Ensemble Defenses for IoT Intrusion Detection"**

**Core Contribution:** Prove that the ensemble model is **certified robust** to bounded perturbations.

**Example Formal Result:**

**Theorem:** "For any input $x$ and perturbation $\delta$ with $||\delta||_\infty \leq \epsilon$, the adaptive ensemble with learned weights $w^*$ predicts correctly with probability ≥ $1 - \alpha$, where $\alpha = O(\epsilon^{\gamma})$ for some $\gamma > 0$."

**How to Achieve This:**

**Option A: Randomized Smoothing**
- Mechanism: Add Gaussian noise $N(0, \sigma^2)$ to input at inference time
- Theory: Neyman-Pearson lemma gives certified robustness
- Paper: Cohen et al. "Certified Adversarial Robustness via Randomized Smoothing" (ICML 2019)
- Advantage: Works on any classifier (plug-and-play)
- Disadvantage: May reduce accuracy; inference cost (500-5000 forward passes per sample)

**Option B: Interval Bound Propagation (IBP)**
- Mechanism: Track min/max bounds on activations through network
- Theory: Lipschitz continuity + interval arithmetic
- Paper: Gowal et al. "Achieving Verified Robustness to Symbol Substitutions via Interval Bound Propagation" (USENIX Security 2021)
- Advantage: Single forward pass; efficient
- Disadvantage: Weaker bounds than randomized smoothing; works mainly on small ε

**Option C: Certified Defenses for Ensembles (Novel)**
- Mechanism: Combine voting rules with individual model robustness certificates
- Theory: Prove that if k/n models are robust, ensemble voting is robust
- Result: Novel contribution specific to this paper
- Advantage: Highest novelty; tailored to your ensemble
- Disadvantage: Hardest to prove; requires careful analysis

---

#### Step 2: Implement Certified Robustness (Week 4-8)

**Experiment 1: Randomized Smoothing Certification**

```python
def certify_robust(model, x, sigma, n_samples=1000):
    """
    Returns: certified robustness radius r such that model is robust
             to L2 perturbations ||δ||_2 ≤ r with high confidence.
    """
    noise_predictions = []
    for _ in range(n_samples):
        x_noisy = x + N(0, sigma^2)
        y_pred = model.predict(x_noisy)
        noise_predictions.append(y_pred)

    # Count votes for top-2 classes
    top2_counts = sorted(count_votes(noise_predictions))[-2:]
    confidence = (top2_counts[1] - top2_counts[0]) / n_samples

    # Certified radius using Neyman-Pearson lemma
    # r = (sigma / 2) * (Phi^{-1}(confidence) - Phi^{-1}(alpha))
    # where Phi is Gaussian CDF, alpha is confidence level (e.g., 0.001)

    return certified_radius(sigma, confidence)
```

**Results Expected:**

```
Dataset         Accuracy   Certified Radius (ε)   Trade-off
Edge-IIoTset    92.1%      0.12                   -2 pp accuracy
ToN-IoT         95.3%      0.08                   -2.4 pp accuracy
Cross-domain    84.2%      0.06                   -2.5 pp accuracy
```

**Experiment 2: Adaptive Attack vs. Certified Defense**

```
Attack Method           Against Fixed Ensemble   Against Certified Ensemble
Standard PGD (ε=0.1)    8.55% ASR                 0% (robust by certification)
Adaptive PGD            12-15% ASR                0% (robust within certified ε)
AutoAttack (ε=0.1)      ~15% ASR                  0% (robust to certified ε)
```

**Key Claim:** "Our ensemble with certified defenses provably resists attacks in the certified region, unlike the original ensemble which has no formal guarantees."

---

#### Step 3: Prove Ensemble Voting Robustness (Week 9-11)

**Theorem to Prove (Novel Contribution):**

**Claim:** "If each of k classifiers in an ensemble is certified robust to ε-bounded perturbations with majority voting, then the ensemble voting mechanism is also certified robust."

**Proof Sketch:**
1. Let each classifier $h_i$ be certified robust: $\forall ||δ|| \leq ε, h_i(x) = h_i(x+δ)$ with prob ≥ $1-\alpha_i$
2. Ensemble voting: output = majority class among $\{h_1(x), ..., h_n(x)\}$
3. If ≥ $\lceil n/2 \rceil$ classifiers are robust, majority vote is robust
4. Union bound: $P(\text{voting fails}) ≤ \sum_i \alpha_i$
5. Result: Ensemble robustness = union over individual model robustness

**Cite:**
- [Randomized Smoothing, Cohen et al. ICML 2019]
- [Ensemble robustness, Barreno et al. 2010]

**Advanced Contribution (if you want to be even more novel):**
- Analyze **weighted majority voting** (your adaptive weights!)
- Prove robustness under non-uniform weights
- Show that optimal weights w* balance accuracy vs. robustness (Pareto frontier)

---

#### Step 4: Position as Security Contribution (Week 12)

**Rewrite Abstract:**

❌ OLD: "evaluates framework under FGSM and PGD-10 experiments"

✅ NEW: "proposes the first certified ensemble defense for cross-domain IoT IDS. Using randomized smoothing and weighted majority voting, we prove that our ensemble is robust to ε-bounded feature-space perturbations with ε ≥ 0.12 for in-domain and ε ≥ 0.06 for cross-domain evaluation. We establish that certified robustness of individual models transfers to voting ensembles via formal analysis."

**Restructure Paper for Security Venues:**

| Section | Focus | Length |
|---------|-------|--------|
| I. Intro | Problem: IoT IDS vulnerable to adversarial attacks; need formal guarantees | 1 page |
| II. Threat Model | Formal specification of adversary (bounded perturbations, knowledge assumptions) | 0.5 page |
| III. Related Work | Certified defenses, ensemble security, adversarial IDS | 1 page |
| IV. Method | **NEW:** Randomized smoothing + certified voting (2 pages) | 2 pages |
| V. Theorems | **NEW:** Formal security theorems with proofs | 1.5 page |
| VI. Experiments | (1) Certification curves, (2) Adaptive attacks, (3) Trade-off analysis, (4) Cross-domain certification | 2 pages |
| VII. Limitations & Defenses | Open problems; when defenses fail | 0.5 page |

**Total: 8-9 pages**

---

#### Step 5: Experimental Validation (Week 11-13)

**Experiment A: Certification vs. Attack Success Rate**

```
Perturbation Budget ε   Attack Success Rate (undefended)   Certified Robust?   ASR (certified)
0.05                    3-5%                               ✓ Yes               0%
0.10                    8-12%                              ✓ Yes               0%
0.15                    15-20%                             ✗ No (outside ε)    varied
```

**Experiment B: Cost of Defense**

```
Defense Method              Certified ε   Accuracy Drop   Inference Time (ms)   Deployable?
Undefended baseline         N/A           93% (Edge)      2.1                   Yes
Randomized smoothing (σ=0.1) 0.12         91% (-2pp)      1050 (500 samples)   Maybe (slow)
IBP + Fine-tuning           0.08          92% (-1pp)      2.3                   Yes
Adaptive weights (no cert)   N/A           94% (+1pp)      2.1                   Yes (but no guarantee)
```

**Experiment C: Attack Adaptation**

Attacker knows:
- Your certified defense parameters (σ, voting rule, etc.)
- Your model weights
- Certified radius ε

Question: Can adaptive attacker still fool you?

Answer (with proper analysis): "No, by definition of certified robustness."

But show empirically: "Adaptive attacks fail against certified radius."

---

### Submission Strategy: Security-First Path

**Target Venue Priority:**
1. **CCS** (top priority; strong security emphasis)
2. **USENIX Security** (secondary; formal methods valued)
3. **IEEE S&P** (tertiary; if you have novel threat model)

**NeurIPS/ICML/ICLR:** Won't accept without ML novelty; skip or combine with Path A.

**Key Claims for Review Committee:**
- "First certified ensemble defense for cross-domain IoT IDS"
- "Formal theorems proving voting-ensemble robustness"
- "Demonstrates certified robustness transfers across domains"

---

## PHASE 4: HYBRID PATH (HARDEST, HIGHEST REWARD)

### If You Choose: Both ML + Security Novelty

**Unified Contribution:**
1. **ML Novelty:** Domain-adaptive ensemble weights
2. **Security Novelty:** Certified robustness for adaptive weights

**Combined Theorem:**

"Domain-adaptive ensemble weights learned via validation-set tuning maintain certified robustness: if weights $w_1, w_2, w_3$ satisfy $||w|| ≤ 1$, and each base classifier is certified robust to ε, then weighted voting remains certified robust to ε."

**Advanced Result (if proven):**

"Optimal domain-adaptive weights $w^*$ form a Pareto frontier between in-domain accuracy and cross-domain robustness. Weights closer to $(0.5, 0.5, 0)$ maximize robustness; weights closer to oracle $(w_1^*, w_2^*, w_3^*)$ maximize accuracy."

**Papers This Would Rival:**
- Robustness + fairness (e.g., "Fair and Robust Learning" papers)
- Domain adaptation + certified defenses (very rare; high novelty)

**Submission Venues:**
- ✅ **NeurIPS** (strong novelty, both ML + security)
- ✅ **ICML** (if heavy on ML theory)
- ✅ **CCS** (if heavy on security formalization)
- ✅ **ICLR** (if emphasizing representation learning under robustness constraints)
- ⚠️ **USENIX Security** (possible if security analysis is deep)

**Estimated Effort:** 12-18 months (very ambitious)

---

## PHASE 5: EXPERIMENTAL EXCELLENCE (ALL PATHS)

### Critical Experiments for All Paths

#### 1. **Rigorous Cross-Domain Analysis**

**Problem:** Why is Edge→ToN asymmetric (99.98% recall, 76.33% precision)?

**Solution: Conduct Root-Cause Analysis**

```
Step 1: Visualize Feature Distributions
  - For each of 6 common features, plot:
    * Edge-IIoTset (attack vs. benign) KDE
    * ToN-IoT (attack vs. benign) KDE
    * Overlay on same figure
  - Measure: Jensen-Shannon divergence, Wasserstein distance

Step 2: Class Imbalance Analysis
  - Edge: 84.6% attack, 15.4% benign
  - ToN: 76.3% attack, 23.7% benign
  - Hypothesis: Model trained on Edge overpredicts "attack" on ToN
  - Test: Apply cost-sensitive reweighting; measure Macro F1 improvement

Step 3: Domain Divergence Quantification
  - Compute H-divergence (Cortes & Mohri 2014)
  - Compute Maximum Mean Discrepancy (MMD)
  - Correlate with cross-domain performance drop

Step 4: Error Analysis
  - For Edge→ToN: manually inspect 20 false positives (benign ToN predicted as attack)
  - For ToN→Edge: manually inspect 20 false negatives (attack Edge predicted as benign)
  - Are there patterns? Dataset-specific artifacts?
```

**Deliverable: Table + Figure showing root causes**

---

#### 2. **Comprehensive Ablation Studies**

**Critical Ablations:**

| Ablation | Purpose | Expected Finding |
|----------|---------|------------------|
| RF only (no MLP, no AE) | Establish RF baseline | Attack F1 ~ 85-90% |
| MLP only | Establish MLP baseline | Attack F1 ~ 80-88% |
| AE anomaly only (no RF/MLP) | Establish AE anomaly baseline | Low precision (high FP) |
| RF + MLP (no AE) | Value of anomaly signal | Macro F1 ~ 3-5 pp lower than ensemble |
| RF + AE (no MLP) | Value of MLP | Macro F1 ~ 2-3 pp lower |
| Fixed 0.5/0.5 vs. learned weights | Value of weight tuning | Cross-domain F1 ~ 2-4 pp better with learned |
| 13 features vs. 6 common features | Value of feature reduction | In-domain F1 ~ 1-2 pp lower with 6; cross-domain ~same or better |
| Benign AE vs. attacked-trained AE | Benign-only strategy | Benign AE better for cross-domain (lower false alarms) |

**Deliverable: Ablation Table (Table VIII)**

---

#### 3. **Statistical Rigor**

**For All Submissions:**

```
Step 1: K-Fold Cross-Validation (k=5)
  - Split each dataset into 5 stratified folds
  - Train/test on each fold; report mean ± std for all metrics
  - Report per-fold results as well as aggregate

Step 2: Confidence Intervals
  - Use bootstrap (1000 iterations) to compute 95% CI for Attack F1, Precision, Recall
  - Report: "Attack F1: 94.29% ± 1.2% (95% CI)"

Step 3: Significance Testing
  - For each pair of methods (e.g., fixed vs. learned weights):
    * Paired t-test on fold-level metrics (p-value, effect size)
    * Report: "Fixed 0.5/0.5 (89.1%) vs. Learned weights (91.2%), p < 0.05, Cohen's d = 0.6"

Step 4: Multiple Comparison Correction
  - If comparing 5 methods, use Bonferroni correction (α → α/10)
  - Report: "After Bonferroni correction, methods X and Y remain significantly different"

Step 5: Report Retraining Variance
  - Train 10 times with different random seeds
  - Report mean ± std across runs
  - Include in limitations: "Model variability is ±0.06 pp (as per Paper Section VI.E)"
```

**Deliverable: Results Table with ± error bars + significance test table**

---

#### 4. **Baseline Comparisons**

**For ML-First Path (NeurIPS/ICML/ICLR):**

| Baseline Method | Type | Reference | Expected vs. Paper |
|---|---|---|---|
| Single RF (100 trees, depth 15) | Supervised only | Scikit-learn default | ~5-8 pp lower F1 |
| Single MLP (128→64→32→1) | Supervised only | PyTorch standard | ~6-10 pp lower F1 |
| Majority voting (RF + MLP) | Simple ensemble | Zhou 2012 | ~1-2 pp lower F1 |
| Stacking (meta-learner on RF+MLP+AE) | Meta-ensemble | Wolpert 1992 | ~2-3 pp lower F1 |
| DANN (domain-invariant features + MLP) | Domain adaptation | Ganin et al. 2015 | ~3-5 pp lower cross-domain F1 |
| TrAdaBoost (transfer boosting) | Transfer learning | Dai et al. 2007 | ~2-4 pp lower cross-domain F1 |

**For Security-First Path (CCS/USENIX):**

| Defense Method | Type | Reference | Certified ε | Accuracy |
|---|---|---|---|---|
| Undefended ensemble | Baseline | Paper | N/A | 94.3% |
| Randomized smoothing | Certified | Cohen et al. 2019 | 0.12 | 92.1% |
| IBP + fine-tuning | Certified | Gowal et al. 2021 | 0.08 | 93.2% |
| Adversarial training | Heuristic | Madry et al. 2018 | ~0.1 (uncertified) | 91.8% |
| Ensemble without voting | Baseline | N/A | N/A | 88.9% |

---

#### 5. **Hyperparameter Sensitivity Analysis**

**Experiment: Grid Search + Ablation**

```
Hyperparameters to tune:
  - RF: n_trees ∈ {50, 100, 200}, max_depth ∈ {10, 15, 20}
  - MLP: layer sizes ∈ {[64,32,1], [128,64,32,1], [256,128,64,1]}, dropout ∈ {0.0, 0.2, 0.5}
  - AE: bottleneck ∈ {8, 16, 32}, learning rate ∈ {0.0001, 0.001, 0.01}
  - Ensemble: weights ∈ grid of 100 combinations (not just 0.5/0.5)
  - Anomaly threshold: ∈ {0.70, 0.75, 0.80, 0.85}

Methodology:
  - Use grid search on validation set (not test set!)
  - Report best hyperparameters for each dataset
  - Report sensitivity: "Varying MLP dropout from 0.0 to 0.5 changes Attack F1 by ±1.2 pp"
  - Justify choices: "We select dropout=0.2 based on validation F1 curves"
```

**Deliverable: Hyperparameter tuning curves + sensitivity table**

---

### Critical Experiments Summary

**Create These 5 Tables/Figures:**

1. **Table VIII:** Ablation study (each component's contribution)
2. **Table IX:** Baseline comparisons (your method vs. prior art)
3. **Table X:** Cross-domain root-cause analysis (feature distributions, class imbalance, domain divergence)
4. **Figure IV:** Statistical rigor (k-fold cross-validation with error bars)
5. **Figure V:** Hyperparameter sensitivity (heatmaps or curves)

---

## PHASE 6: WRITING & POSITIONING (ALL PATHS)

### For NeurIPS/ICML/ICLR (ML Venues)

**Structure:**

1. **Abstract (150 words):**
   - Lead with ML contribution (domain-adaptive weighting OR certified robustness)
   - Quantify novelty ("X% improvement over fixed-weight baseline")
   - Mention theory ("theoretical bounds using H-divergence")

2. **Introduction (1 page):**
   - Problem: Why does cross-domain IDS transfer fail?
   - Gap: Existing ensembles use fixed weights; no formal guarantees
   - Solution: (YOUR CONTRIBUTION)
   - Contributions (numbered list, 3-4 items)

3. **Related Work (1 page):**
   - Ensemble learning
   - Domain adaptation
   - Adversarial robustness (brief)
   - Distinguish YOUR work from prior art

4. **Method (2 pages):**
   - Clear algorithm + pseudocode
   - Formal definitions
   - Example walkthrough

5. **Theory (1 page):**
   - Theorem statement + proof (or proof sketch)
   - Intuition for why it works
   - Convergence/sample complexity (optional)

6. **Experiments (2 pages):**
   - Ablations
   - Baselines
   - Main results (tables + figures)
   - Statistical significance

7. **Discussion (0.5 page):**
   - Insights from results
   - When/why method works
   - Failure modes

8. **Conclusion (0.25 page):**
   - Summary
   - Future work

**Tone:** Formal, technical, emphasize novelty.

---

### For CCS/USENIX Security (Security Venues)

**Structure:**

1. **Abstract (150 words):**
   - Lead with security guarantee ("provably robust to ε-bounded perturbations")
   - Quantify improvement ("8.5× better certified radius than baseline")
   - Mention formal analysis

2. **Introduction + Threat Model (1 page):**
   - Threat: adversarial evasion attacks on IoT IDS
   - Adversary knowledge: white-box, black-box, adaptive
   - Threat scope: feature-space vs. packet-level perturbations
   - Contribution: formal security guarantees for ensemble

3. **Related Work (1 page):**
   - Adversarial ML attacks
   - Certified defenses
   - IoT security (brief)
   - Ensemble security (brief)

4. **Threat Model Formalization (0.5 page):**
   - Formal definitions of adversary capabilities
   - Perturbation constraints (e.g., $||\delta||_\infty \leq \epsilon$)
   - Security goal: "For all δ, certified ensemble outputs correct prediction"

5. **Method (2 pages):**
   - Randomized smoothing / IBP algorithm + pseudocode
   - Certified voting mechanism
   - Formal definitions (certificates, robustness radius)

6. **Security Theorems (1-1.5 pages):**
   - Theorem 1: Individual model robustness (cite Cohen et al.)
   - Theorem 2 (NEW): Voting ensemble robustness (your contribution)
   - Theorem 3 (OPTIONAL): Weighted voting robustness
   - Proofs (or proof sketches)

7. **Experiments (2 pages):**
   - Certification curves (certified ε vs. accuracy)
   - Adaptive attack results
   - Cost of defense (inference time, accuracy loss)
   - Comparison to baselines

8. **Limitations (0.5 page):**
   - When certificates don't hold
   - Efficiency trade-offs
   - Future defenses

9. **Conclusion (0.25 page):**
   - Summary
   - Impact on IoT security

**Tone:** Formal, security-centric, emphasize guarantees.

---

## PHASE 7: PRACTICAL SUBMISSION CHECKLIST

### Before Submitting to Any Venue:

#### Code & Reproducibility
- [ ] Code released on GitHub (with README, requirements.txt, setup instructions)
- [ ] Random seeds fixed; reproducibility ±0.02 pp variance
- [ ] Datasets linked (DOI or download script)
- [ ] Docker image or conda environment file provided
- [ ] Benchmarking script: run this to reproduce all figures

#### Paper Quality
- [ ] Grammar/spell-check (use LanguageTool or Grammarly)
- [ ] All figures/tables have captions and are referenced in text
- [ ] All claims have citations or references to figures/tables
- [ ] No orphaned sections; paper flows logically
- [ ] Margins, fonts, format comply with venue (IEEE, ACM, etc.)

#### Experimental Rigor
- [ ] K-fold cross-validation with error bars reported
- [ ] Significance tests (t-tests, p-values) for key claims
- [ ] Hyperparameter justification or sensitivity analysis
- [ ] Baseline comparisons (at least 2-3 prior methods)
- [ ] Ablation studies (remove each component; report impact)
- [ ] Reproducibility: retraining variance documented (±X pp)

#### Novelty Claims
- [ ] Clearly distinguish YOUR contribution from related work
- [ ] Claim is specific and measurable ("8% improvement" vs. "better")
- [ ] No overclaiming (e.g., don't claim "adversarially robust" without formal proof)
- [ ] Honest limitations section (Section VI/VII)

#### Venue-Specific
- [ ] **NeurIPS:** ML novelty front-and-center; theory section strong; experimental section comprehensive
- [ ] **ICML:** Similar to NeurIPS; strong baselines; convergence/sample complexity analysis appreciated
- [ ] **ICLR:** Emphasize representational or learning insights; may accept if emphasizing domain adaptation
- [ ] **CCS:** Security formal; threat model explicit; theorems with proofs; adaptive attack evaluation
- [ ] **USENIX Security:** Similar to CCS; practical evaluation of defenses; deployability discussion

---

## Timeline Estimate by Path

### ML-First Path (NeurIPS/ICML/ICLR)
- Weeks 1-2: Choose contribution + literature review
- Weeks 3-8: Implement domain-adaptive weighting
- Weeks 9-10: Theoretical analysis + proofs
- Weeks 11-13: Experiments + baselines + ablations
- Weeks 14-16: Writing + revision
- **Total: 4 months to submission-ready**

### Security-First Path (CCS/USENIX Security)
- Weeks 1-2: Choose defense + threat model formalization
- Weeks 3-8: Implement certified defense (randomized smoothing or IBP)
- Weeks 9-11: Theorems + proofs + formal analysis
- Weeks 12-14: Experiments + adaptive attacks + baselines
- Weeks 15-17: Writing + revision
- **Total: 4.25 months to submission-ready**

### Hybrid Path (All Five Venues)
- Weeks 1-2: Plan ML + Security contributions
- Weeks 3-10: Implement both (parallel tracks)
- Weeks 11-14: Theoretical analysis (both ML + Security)
- Weeks 15-18: Comprehensive experiments
- Weeks 19-24: Writing + multiple revisions
- **Total: 6 months to submission-ready; 12 months if aiming for acceptance**

---

## Recommended Path Selection

**Choose ML-First If:**
- You're more interested in improving ensemble learning
- You have ML background and less security background
- NeurIPS/ICML/ICLR accept rate is better for you than CCS
- You believe domain-adaptive weights is genuinely novel

**Choose Security-First If:**
- You're more interested in formal guarantees
- You have security/formal methods background
- CCS/USENIX acceptance is higher priority
- You believe certified ensemble defenses haven't been studied for IoT

**Choose Hybrid If:**
- You have 6-12 months
- You want to submit to all 5 venues
- Your team has both ML and security expertise
- High-risk, high-reward strategy

**My Recommendation:** Start with **ML-First Path** (domain-adaptive weights):
- Easier to implement (4 months to submission-ready)
- More natural contribution from your current ensemble setup
- Can pivot to hybrid later if results are strong
- NeurIPS/ICML are prestigious and broad

---

## Success Criteria by Venue

### NeurIPS (Acceptance Rate ~26%)
**Your paper should:**
- [ ] Have clear algorithmic contribution (domain-adaptive weights)
- [ ] Include theoretical analysis (generalization bounds or sample complexity)
- [ ] Show >5% improvement over baselines on standard benchmarks
- [ ] Provide extensive ablations and analysis
- [ ] Be written for ML audience (not security-specific jargon)

### ICML (Acceptance Rate ~28%)
**Your paper should:**
- [ ] Similar to NeurIPS + emphasis on optimization/learning
- [ ] Convergence analysis or sample efficiency results
- [ ] Baselines on publicly available datasets (CICIDS2017, Edge-IIoTset)

### ICLR (Acceptance Rate ~32%)
**Your paper should:**
- [ ] Representations / domain adaptation insights
- [ ] Clear learning dynamics under domain shift
- [ ] Interpretability or analysis of learned weights per domain

### CCS (Acceptance Rate ~18%)
**Your paper should:**
- [ ] Formal security theorems with proofs
- [ ] Certified robustness with quantified ε and accuracy trade-off
- [ ] Adaptive attack evaluation
- [ ] Threat model explicitly stated and justified

### USENIX Security (Acceptance Rate ~17%)
**Your paper should:**
- [ ] Similar to CCS
- [ ] Practical evaluation (inference time, memory, deployability)
- [ ] Discussion of real-world deployment challenges

---

## Final Recommendation

**Next Step (This Week):**

1. **Hold team meeting:** Decide Path A, B, or C
2. **Sketch contribution:** Write 1-page summary of novel idea
3. **Literature review:** Read 5 key papers in chosen direction
4. **Create roadmap:** Assign ownership for implementation, experiments, writing

**Then proceed to Phase 2/3/4 based on path.**

---

## Questions to Answer Before Starting

1. **Do you have time?** ML/Security papers take 4-6 months minimum to publication-ready
2. **Do you have expertise?** ML-First easier for ML-focused teams; Security-First for formal methods teams
3. **What's the priority?** If reputation matters most → aim for NeurIPS; if security impact matters most → aim for CCS
4. **Can you release code?** Venues increasingly require GitHub repos; if not, avoid top-tier
5. **Do you want to co-author with security experts?** Hybrid path benefits from interdisciplinary team

---

**Document created:** August 26, 2026
**Next review:** After Path selection (Week 1-2)
**Success metric:** "Paper accepted to ≥1 of {NeurIPS, ICML, ICLR, CCS, USENIX Security}"
