# CLAIM / EVIDENCE MATRIX

### 1. Option C beats RF on clean AUC
- **Current Status**: Unsupported. (Historical clean ROC-AUC for RF $\approx$ 0.9976, Option C $\approx$ 0.9970).
- **Why unsupported**: RF slightly outperforms Option C on clean detection. Option C sacrifices marginal clean accuracy to gain substantial adversarial robustness.
- **Required Evidence**: Option C must statistically outperform RF on clean detection across a predefined evaluation protocol.
- **Experiment Required**: Evaluate and statistically compare RF vs Option C clean ROC-AUC across all four datasets using predetermined seeds.
- **Implementation Required**: Multi-seed clean evaluation with paired statistical tests or confidence intervals.
- **Success Criterion**: Option C ROC-AUC > RF ROC-AUC at $p < 0.05$ significance.
- **Failure Criterion**: Option C ROC-AUC $\le$ RF ROC-AUC or not statistically significant.
- **Potential Valid Wording**: "Option C statistically outperforms RF on clean detection."
- **Potential Invalid Wording**: "Option C provides a robustness/accuracy trade-off, sacrificing marginal clean detection for adversarial security."

### 2. 0.7/0.3 is optimal
- **Current Status**: Unsupported.
- **Why unsupported**: The weights were chosen based on frozen checkpoints. Multi-seed testing reveals high robustness variability; there is no formal proof that 0.7/0.3 is the globally optimal Pareto point.
- **Required Evidence**: 0.7/0.3 must lie on the empirical Pareto frontier of accuracy vs. ASR, dominating or equating other combinations.
- **Experiment Required**: Grid sweep fusion weights $w \in [0.0, 1.0]$ across multiple seeds and plot empirical Pareto frontiers (Clean AUC vs PGD-10 ASR).
- **Implementation Required**: Iterative weight ablation on the existing robust neural streams.
- **Success Criterion**: The 0.7 point lies cleanly on the non-dominated empirical accuracy/robustness Pareto frontier.
- **Failure Criterion**: The 0.7 point is strictly dominated by another weight configuration (i.e., another configuration yields better clean accuracy AND better robustness).
- **Potential Valid Wording**: "The selected 0.7/0.3 fusion lies on the empirical accuracy-robustness Pareto frontier."
- **Potential Invalid Wording**: "0.7/0.3 is the optimal fusion weighting."

### 3. The architecture is adversarially robust independent of training seed
- **Current Status**: Unsupported.
- **Why unsupported**: Multi-seed analysis found massive seed instability (NF-ToN median 4.5% ASR, but 20% of seeds suffered catastrophic failure at ~45% ASR).
- **Required Evidence**: Consistent low-variance adversarial robustness across random seeds.
- **Experiment Required**: Train robust MLP ensembles or introduce robustness-aware checkpoint selection.
- **Implementation Required**: Validation-set PGD filtering algorithm, or runtime neural ensembling.
- **Success Criterion**: Mean ASR < 10% and maximum ASR < 15% across 10 random seeds.
- **Failure Criterion**: Multi-seed evaluations still exhibit > 20% standard deviation or catastrophic >40% failure modes.
- **Potential Valid Wording**: "Through robustness-aware checkpoint selection / ensembling, the architecture consistently guarantees < 10% ASR across random initializations."
- **Potential Invalid Wording**: "The PGD-7 architecture is inherently adversarially robust."

### 4. The adaptive attack is a true white-box/ensemble-aware attack
- **Current Status**: Unsupported.
- **Why unsupported**: The RF is non-differentiable. The current adaptive attack trains a differentiable surrogate surrogate model, approximating the RF. It is a gray-box surrogate transfer attack.
- **Required Evidence**: A genuine query-based black-box gradient estimator (SPSA, NES) acting against the actual deployed ensemble predictor function.
- **Experiment Required**: Execute NES/SPSA against the true Option C pipeline.
- **Implementation Required**: Implement query-based gradient estimation algorithms in the attack generation logic.
- **Success Criterion**: The attack computes true adversarial perturbation bounds without relying on a differentiable proxy architecture.
- **Failure Criterion**: The true query-based attack fails to evade the system, proving the proxy attack underestimated Option C's security.
- **Potential Valid Wording**: "Option C was rigorously evaluated against a true black-box query-based ensemble attack."
- **Potential Invalid Wording**: "We executed a true white-box gradient attack on the non-differentiable ensemble."

### 5. The IDS has end-to-end differential privacy
- **Current Status**: Unsupported.
- **Why unsupported**: Only the neural MLP stream is trained with DP-SGD; the RF is unprotected. Furthermore, current evaluation operates strictly on synthetic Gaussians, not real IIoT data.
- **Required Evidence**: The neural network must be trained via DP-SGD on actual `stage3` IIoT records and preserve meaningful detection capabilities.
- **Experiment Required**: Train the DP-SGD MLP on `Edge-IIoTset` and `NF-ToN-IoT-v2`, calculate $\varepsilon, \delta$, and measure test AUC.
- **Implementation Required**: Replace synthetic `rng.normal` simulation with true `train.parquet` batching using Opacus.
- **Success Criterion**: The neural branch achieves $\varepsilon \le 3.0$ while preserving ROC-AUC > 0.80 on real IIoT test data.
- **Failure Criterion**: The neural branch fails to converge, destroying clean detection utility under the DP gradient clipping budget.
- **Potential Valid Wording**: "The neural detection branch was trained under real-data differential privacy."
- **Potential Invalid Wording**: "The complete IDS offers end-to-end differential privacy."

### 6. The XAI method is exact SHAP for the fused predictor
- **Current Status**: Unsupported.
- **Why unsupported**: SHAP is currently computed on the RF (`TreeExplainer`) and MLP (`GradientExplainer`) separately, then linearly averaged. This violates the non-linear interaction rules of exact unified game-theoretic SHAP.
- **Required Evidence**: A model-agnostic `KernelExplainer` or `ExactExplainer` executed against the combined $f(x) = 0.7 RF(x) + 0.3 MLP(x)$ output.
- **Experiment Required**: Execute KernelSHAP against the Option C prediction wrapper. Compare the resulting base values and feature importances against the component-wise approximation.
- **Implementation Required**: Wrapper class mapping inputs to Option C probability outputs, fed into `shap.KernelExplainer`.
- **Success Criterion**: Exact Option C SHAP can be successfully computed and analyzed within feasible time constraints.
- **Failure Criterion**: Exact SHAP is computationally intractable for the dataset size, necessitating the approximation.
- **Potential Valid Wording**: "Model explainability is delivered via exact game-theoretic SHAP computed directly against the fused predictor."
- **Potential Invalid Wording**: "Weighted component attribution perfectly mirrors true ensemble SHAP."

### 7. The runtime result represents end-to-end throughput
- **Current Status**: Unsupported.
- **Why unsupported**: The 16,505 samples/sec metric times only the offline `model.predict()` function on pre-scaled arrays, entirely ignoring packet parsing, flow construction, and network I/O.
- **Required Evidence**: Active flow capture benchmark measuring packets-in to alerts-out.
- **Experiment Required**: Replay PCAPs or generate high-volume live traffic and time the total system latency.
- **Implementation Required**: End-to-end Python timing hooks within the operational `FlowAggregator` and `Predictor` daemons.
- **Success Criterion**: The full packet-to-alert pipeline successfully measures its own true wall-clock processing limit.
- **Failure Criterion**: The live system cannot be reliably benchmarked due to thread-blocking I/O limits.
- **Potential Valid Wording**: "The complete packet-to-alert pipeline sustained X flows/sec under the evaluated testbed configuration."
- **Potential Invalid Wording**: "The IDS operates at line rate."

### 8. The system generalizes to unseen domains
- **Current Status**: Unsupported.
- **Why unsupported**: The existing audits confirm extreme over-fitting if datasets aren't homogeneously structured.
- **Required Evidence**: Positive classification performance (AUC > 0.80) when trained on 3 datasets and evaluated on a 1 held-out dataset.
- **Experiment Required**: Leave-one-domain-out (LODO) experiments across all four canonical datasets.
- **Implementation Required**: Training script that merges 3 `train.parquet` datasets and evaluates on the 4th `test.parquet`.
- **Success Criterion**: LODO ROC-AUC > 0.85.
- **Failure Criterion**: LODO ROC-AUC approaches 0.50 (random guessing), indicating the features are domain-specific.
- **Potential Valid Wording**: "Cross-domain generalization was explicitly evaluated under leave-one-domain-out testing."
- **Potential Invalid Wording**: "The system generalizes universally to unseen operational environments."

### 9. React/FastAPI dashboard is implemented
- **Current Status**: Unsupported.
- **Why unsupported**: Repository forensics confirms absolutely zero frontend or web backend code exists. The implementation is purely a Python package + CLI.
- **Required Evidence**: A complete React web app and FastAPI service passing integration tests and handling active database connections.
- **Experiment Required**: None.
- **Implementation Required**: Do not pursue unless explicitly prioritized by research scope.
- **Success Criterion**: Full integration test coverage passing for the HTTP endpoints and frontend builds.
- **Failure Criterion**: The dashboard is merely a mocked wireframe or non-functional skeleton.
- **Potential Valid Wording**: "The project includes a complete operational dashboard (React/FastAPI)."
- **Potential Invalid Wording**: (Will remain "not pursued" and removed from paper claims if unbuilt).

### 10. The PCAP demonstration represents meaningful deployment-scale performance
- **Current Status**: Unsupported.
- **Why unsupported**: The current smoke test evaluates 4 packets and 2 flows merely to verify syntax and API integration.
- **Required Evidence**: Multi-million flow PCAP tests or sustained traffic generation validating bounded memory and dropped flow limits.
- **Experiment Required**: Stress-test the operational daemon with 10k, 100k, 1M flows.
- **Implementation Required**: Replay large-scale benchmark PCAPs through the `iot-ids` live ingress engine.
- **Success Criterion**: The daemon processes 1M flows without crashing, memory leaking, or deadlocking.
- **Failure Criterion**: The FlowAggregator suffers OOM crashes or catastrophic flow dropping at scale.
- **Potential Valid Wording**: "Operational load-testing validated the system's stability handling sustained high-volume traffic bursts."
- **Potential Invalid Wording**: "The system is enterprise production ready."
