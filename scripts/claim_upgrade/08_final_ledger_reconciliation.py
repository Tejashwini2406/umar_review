import re
from pathlib import Path
import json

def reconcile_paper():
    repo_root = Path("C:/Users/umari/Documents/P1_task_Implementation")
    tex_file = repo_root / "final_ieee_paper" / "main.tex"
    
    with open(tex_file, "r") as f:
        content = f.read()
        
    original_content = content
    
    # 1. ABSTRACT
    content = content.replace(
        "reduces the Attack Success Rate (ASR) to 4.0\\%, compared to 53.8\\% for a standard neural stream.",
        "reduces the neural-gradient transfer Attack Success Rate (ASR) to 6.5\\%, compared to 53.8\\% for a standard neural stream. However, under a 500-query black-box NES attack, empirical evasion rises to 36.1\\%."
    )
    content = content.replace(
        "Model explainability is delivered via Weighted Component Attribution Aggregation ($0.7 \\cdot \\text{RF} + 0.3 \\cdot \\text{MLP}$).",
        "Model explainability is delivered via a computationally efficient surrogate Weighted Component Attribution Aggregation ($0.7 \\cdot \\text{RF} + 0.3 \\cdot \\text{MLP}$), as exact fused Shapley calculation is intractable."
    )
    content = content.replace(
        "establish a pure detection throughput of 16,505 samples/sec (0.0606 ms/sample) at batch size $N=1024$.",
        "establish an offline classifier-only throughput of 16,505 samples/sec at batch size $N=1024$, explicitly excluding packet ingestion and flow aggregation."
    )
    
    # 2. XAI Section
    content = content.replace(
        "This approach avoids computing expensive joint game-theoretic Shapley values across fused non-differentiable model boundaries while providing consistent component-level feature attributions.",
        "This approach serves as a computationally tractable component-wise surrogate approximation. An audit of exact game-theoretic Shapley enumeration confirmed that evaluating the $2^{21}$ coalitions per sample across the fused boundaries is mathematically intractable for live inference and that the linear surrogate violates the strict SHAP additivity axiom."
    )
    
    # 3. Adversarial Section
    content = content.replace(
        "reducing the final ensemble ASR to \\textbf{4.0\\%} on NF-ToN-IoT-v2.",
        "reducing the final ensemble transfer ASR to \\textbf{6.5\\%} on NF-ToN-IoT-v2. However, under a strict 500-query black-box Natural Evolution Strategies (NES) attack optimizing directly against the fused probability output, the operational evasion rate reaches \\textbf{36.1\\%}. This demonstrates that neural-gradient transfer substantially understates the empirical vulnerability of the ensemble to direct query-based optimization."
    )
    # Also update table 2 if possible, or just leave text. The text correction is sufficient.
    content = content.replace("Option C ASR \\\\\n\\midrule\nEdge-IIoTset   & 0.000 & 0.000 & \\textbf{0.038}", "Option C ASR \\\\\n\\midrule\nEdge-IIoTset   & 0.000 & 0.000 & \\textbf{0.054}")
    content = content.replace("NF-ToN-IoT-v2  & 0.538 & 0.480 & \\textbf{0.040}", "NF-ToN-IoT-v2  & 0.538 & 0.480 & \\textbf{0.065}")
    
    # 4. Throughput Section
    content = content.replace(
        "Inference latency was benchmarked on a single-CPU host across batch sizes $N \\in \\{1, \\dots, 1024\\}$ for pure Option C detection prediction.",
        "Offline classifier-only inference latency was benchmarked on a single-CPU host across batch sizes $N \\in \\{1, \\dots, 1024\\}$ for Option C detection prediction on pre-aggregated tabular features."
    )
    content = content.replace(
        "At batch size $N=1024$, the inference engine achieved approximately 16,505 samples/sec (per-sample latency of 0.0606 ms) on a single CPU host. Single-sample inference latency ($N=1$) is 69.40 ms.",
        "At batch size $N=1024$, the classification engine achieved approximately 16,505 samples/sec (per-sample latency of 0.0606 ms) on a single CPU host. Crucially, this metric excludes upstream network packet capture, Scapy payload parsing, and stateful flow aggregation overheads."
    )
    content = content.replace(
        "Table~\\ref{tab:runtime_benchmarks}\n\\centering\n\\begin{tabular}{rccc}\n\\toprule\n\\textbf{Batch ($N$)} & \\textbf{Batch Latency (ms)} & \\textbf{Per-Sample (ms)} & \\textbf{Throughput (samples/s)} \\\\\n\\midrule",
        "Table~\\ref{tab:runtime_benchmarks}\n\\centering\n\\begin{tabular}{rccc}\n\\toprule\n\\textbf{Batch ($N$)} & \\textbf{Batch Latency (ms)} & \\textbf{Per-Sample (ms)} & \\textbf{Classifier Throughput (samples/s)} \\\\\n\\midrule"
    )
    
    # 5. Limitations Section
    limitation_updates = """\\textbf{Limitation / Gap} & \\textbf{Current Evidence Scope} & \\textbf{Future Direction} \\\\
\\midrule
XAI Additivity & Component-wise SHAP surrogate violates exact Shapley additivity. & Develop exact SHAP estimators for hybrid ensembles. \\\\
End-to-End Throughput & Benchmark excludes live packet capture and flow aggregation. & Validate end-to-end throughput on physical testbed. \\\\
DP Scope Limit & Guaranteed on neural stream only ($\\varepsilon=2.37$). & Extend DP tree building to Random Forest stream. \\\\
Black-box Optimization & Query-based attack (36.1\\% ASR) bypassed neural-gradient transfer. & Develop exact joint gradient attacks on tree-neural boundary. \\\\
Federated Training & Multi-client aggregation code implemented. & Conduct multi-node federated training trials. \\\\"""
    
    # regex replace the table body
    import re
    content = re.sub(
        r"\\textbf\{Limitation / Gap\}.*?\\bottomrule",
        limitation_updates.replace("\\", "\\\\") + "\n\\\\bottomrule",
        content,
        flags=re.DOTALL
    )
    
    # 6. Conclusion
    content = content.replace(
        "reduces PGD-10 evasion ASR to 4.0\\%",
        "reduces PGD-10 neural-gradient transfer ASR to 6.5\\% (though remaining vulnerable to 36.1\\% evasion under 500-query black-box optimization)"
    )
    content = content.replace(
        "provides transparent feature attributions",
        "provides computationally tractable surrogate feature attributions"
    )
    content = content.replace(
        "sustains a single-CPU host processing throughput of 16,505 samples/sec",
        "sustains a single-CPU offline classifier throughput of 16,505 samples/sec"
    )
    
    with open(tex_file, "w") as f:
        f.write(content)
        
    print(f"Changes applied to main.tex: {len(content) != len(original_content)}")
    print("Writing markdown report...")
    
    md_report = """# FINAL LEDGER RECONCILIATION (P8)

## 1. Executive Summary
The numerical completion audit of the Option C IIoT IDS framework is now finished. All material claims in the IEEE manuscript have been reconciled against the raw execution evidence from P1 through P7. Unverified or overstated claims regarding exact SHAP calculations, end-to-end network throughput, and worst-case adversarial robustness guarantees have been formally downgraded or corrected in the manuscript.

## 2. Claim-by-Claim Ledger

### Claim 1: Data Leakage & Canonical Pipeline
- **Original Status**: Unverified chronological splits.
- **Audit Finding**: Chronological sorting and proper train/test splits were implemented correctly (P1).
- **Final Verdict**: PASS WITH SCOPE.

### Claim 2: Differential Privacy (DP)
- **Original Status**: DP claimed for the ensemble.
- **Audit Finding**: Opacus DP-SGD ($\varepsilon=2.37$) strictly applies to the neural stream only. RF remains non-private (P3).
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
"""
    with open(repo_root / "reports" / "claim_upgrade" / "08_final_ledger_reconciliation.md", "w") as f:
        f.write(md_report)

    print("Reconciliation complete.")

if __name__ == "__main__":
    reconcile_paper()
