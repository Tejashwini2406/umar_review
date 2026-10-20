"""Stage 7 Claim-Evidence Matrix Generator.

Maps every paper headline claim to its source CSV artifact, metric, exact numerical value,
statistical support, and documented scientific limitations.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def generate_claim_evidence_matrix():
    output_dir = Path("reports/stage7")
    output_dir.mkdir(parents=True, exist_ok=True)

    claims = [
        {
            "claim_id": "CLM-001",
            "paper_section": "Sec. V.A / Table V",
            "claim": "Within-domain intrusion detection performance improves monotonically when Level B (Temporal) and Level C (Behavioral) features are added to Level A features.",
            "metric": "Macro F1 / FPR",
            "value": "Macro F1 = 0.986, FPR = 1.6%",
            "dataset_scope": "All 4 Datasets",
            "transfer_scope": "Within-Domain",
            "source_artifact": "reports/stage4/stage4_within_domain_results.csv",
            "source_column": "macro_f1, fpr",
            "statistical_support": "Random Forest Macro F1 increases from 0.923 (baseline) to 0.986 (instant_behavioral), FPR drops from 15.8% to 1.6%.",
            "limitations": "Evaluated on chronological in-domain test splits."
        },
        {
            "claim_id": "CLM-002",
            "paper_section": "Sec. VI.A / Table VI",
            "claim": "Under un-adapted zero-shot transfer, stateless Level A features (instant_only) achieve the highest transfer ROC-AUC.",
            "metric": "Cross-Domain ROC-AUC",
            "value": "0.567 ROC-AUC",
            "dataset_scope": "All 4 Datasets",
            "transfer_scope": "12 Transfer Directions",
            "source_artifact": "reports/stage5/stage5_adaptation_summary.csv",
            "source_column": "roc_auc.mean",
            "statistical_support": "instant_only Logistic Regression ROC-AUC = 0.567 vs 0.487 for baseline_common.",
            "limitations": "Un-adapted tree models suffer timing distribution shift."
        },
        {
            "claim_id": "CLM-003",
            "paper_section": "Sec. VI.B / Table VI",
            "claim": "Unsupervised feature scaling alignment recovers a substantial portion of cross-domain timing shift without using target labels.",
            "metric": "Cross-Domain ROC-AUC",
            "value": "0.607 ROC-AUC (+28.9% gain)",
            "dataset_scope": "All 4 Datasets",
            "transfer_scope": "12 Transfer Directions",
            "source_artifact": "reports/stage6/stage6_confidence_intervals.csv",
            "source_column": "mean, ci_lower_95, ci_upper_95",
            "statistical_support": "Random Forest full_multilevel ROC-AUC increases from 0.318 (zero-shot) to 0.607 (alignment), 95% CI [0.507, 0.717].",
            "limitations": "Unsupervised alignment alone does not achieve full label-calibrated performance."
        },
        {
            "claim_id": "CLM-004",
            "paper_section": "Sec. VI.C / Table VI",
            "claim": "Minimal target domain adaptation (1% budget) completely recovers the superiority of the multi-level representation.",
            "metric": "Cross-Domain ROC-AUC",
            "value": "0.936 ROC-AUC (1% budget)",
            "dataset_scope": "All 4 Datasets",
            "transfer_scope": "12 Transfer Directions",
            "source_artifact": "reports/stage6/stage6_confidence_intervals.csv",
            "source_column": "mean, ci_lower_95, ci_upper_95",
            "statistical_support": "Random Forest full_multilevel ROC-AUC jumps from 0.318 to 0.936 with ~42 target labeled samples (95% CI [0.889, 0.977]).",
            "limitations": "Requires ~42 labeled target flows."
        },
        {
            "claim_id": "CLM-005",
            "paper_section": "Sec. VI.C / Table VI",
            "claim": "Under 5% target budget adaptation, full_multilevel achieves near-perfect cross-domain generalization.",
            "metric": "Cross-Domain ROC-AUC",
            "value": "0.992 ROC-AUC (5% budget)",
            "dataset_scope": "All 4 Datasets",
            "transfer_scope": "12 Transfer Directions",
            "source_artifact": "reports/stage6/stage6_confidence_intervals.csv",
            "source_column": "mean, ci_lower_95, ci_upper_95",
            "statistical_support": "Random Forest full_multilevel ROC-AUC = 0.992 (95% CI [0.989, 0.996]) vs 0.814 for instant_only.",
            "limitations": "Requires ~210 labeled target flows."
        },
        {
            "claim_id": "CLM-006",
            "paper_section": "Sec. VI.D / Table VI",
            "claim": "Causal Behavioral topology features (Level C) provide the single strongest protection against false alarm explosion under cross-domain deployment.",
            "metric": "Cross-Domain FPR (%)",
            "value": "44.6% FPR (zero-shot) / 28.7% FPR (10% budget)",
            "dataset_scope": "All 4 Datasets",
            "transfer_scope": "12 Transfer Directions",
            "source_artifact": "reports/stage5/stage5_fpr_comparison.csv",
            "source_column": "fpr",
            "statistical_support": "instant_behavioral slashes Logistic Regression zero-shot FPR from 91.8% (baseline) to 44.6% (-47.2% absolute FPR reduction).",
            "limitations": "False alarms remain higher than in-domain deployment."
        },
        {
            "claim_id": "CLM-007",
            "paper_section": "Sec. VII / Table VII",
            "claim": "Target domain adaptation yields large statistical effect sizes over zero-shot control across all transfer directions.",
            "metric": "Cohen's dz / Hedges' g",
            "value": "Cohen's dz = 1.134, Hedges' g = 1.055, p = 0.0024",
            "dataset_scope": "All 4 Datasets",
            "transfer_scope": "12 Paired Transfer Units",
            "source_artifact": "reports/stage6/stage6_effect_sizes.csv",
            "source_column": "cohens_dz, hedges_g, ttest_pvalue",
            "statistical_support": "Paired t-test t = 3.927, p = 0.0024 across N = 12 paired transfer directions.",
            "limitations": "N = 12 transfer directions."
        },
        {
            "claim_id": "CLM-008",
            "paper_section": "Sec. VIII.I / Table VIII",
            "claim": "Cross-domain transfer performance does NOT depend on dataset-identifying shortcut features (protocol indicators or raw rate metrics).",
            "metric": "Adapted ROC-AUC Retention (%)",
            "value": "97.2% Performance Retention",
            "dataset_scope": "All 4 Datasets",
            "transfer_scope": "12 Transfer Directions",
            "source_artifact": "reports/stage6/stage6_shortcut_ablation.csv",
            "source_column": "roc_auc",
            "statistical_support": "semantic_features_only (15 features) achieves 0.869 ROC-AUC vs 0.894 for full_multilevel (21 features).",
            "limitations": "Evaluated under 5% budget adaptation."
        }
    ]

    df_claims = pd.DataFrame(claims)
    df_claims.to_csv(output_dir / "stage7_claim_evidence_matrix.csv", index=False)
    print(f"Exported Claim-Evidence Matrix to {output_dir / 'stage7_claim_evidence_matrix.csv'}")


if __name__ == "__main__":
    generate_claim_evidence_matrix()
