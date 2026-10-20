"""
Generate publication-quality LaTeX tables from frozen JSON artifacts.
Reads: reports/tables/final_evaluation_results.json
       reports/tables/xai_results.json
       results/adversarial/adversarial_results.json
       configs/features/canonical_schema.json
Outputs: paper/tables/tab_*.tex
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "reports" / "tables" / "final_evaluation_results.json"
XAI = REPO / "reports" / "tables" / "xai_results.json"
ADV = REPO / "results" / "adversarial" / "adversarial_results.json"
SCHEMA = REPO / "configs" / "features" / "canonical_schema.json"
OUT_DIR = REPO / "paper" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS) as f:
    data = json.load(f)
with open(XAI) as f:
    xai = json.load(f)
with open(ADV) as f:
    adv = json.load(f)
with open(SCHEMA) as f:
    schema = json.load(f)


def pct(v):
    return f"{v*100:.2f}"

def auc(v):
    return f"{v:.4f}"


def tab_indomain():
    """Table 3: In-domain detection performance."""
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{In-Domain Detection Performance on Test Sets}",
        r"\label{tab:indomain}",
        r"\centering",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Acc.} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} & \textbf{Macro F1} & \textbf{ROC} & \textbf{PR} \\",
        r" & (\%) & (\%) & (\%) & (\%) & (\%) & AUC & AUC \\",
        r"\midrule",
    ]
    for ds in ['Edge-IIoTset', 'ToN-IoT']:
        m = data['in_domain'][ds]['supervised_metrics']
        n = data['in_domain'][ds]['test_rows']
        label = ds.replace('-', '{-}')
        lines.append(
            f"{label} & {pct(m['accuracy'])} & {pct(m['precision'])} & "
            f"{pct(m['recall'])} & \\textbf{{{pct(m['f1'])}}} & {pct(m['macro_f1'])} & "
            f"{auc(m['roc_auc'])} & {auc(m['pr_auc'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    text = "\n".join(lines)
    (OUT_DIR / "tab_indomain.tex").write_text(text)
    print("Generated: tab_indomain.tex")


def tab_crossdomain():
    """Table 4: Cross-domain zero-adaptation transfer."""
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Cross-Domain Zero-Adaptation Transfer Performance (6 $F_{\text{common}}$ Features)}",
        r"\label{tab:crossdomain}",
        r"\centering",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"\textbf{Direction} & \textbf{Acc.} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} & \textbf{Macro F1} & \textbf{ROC} & \textbf{PR} \\",
        r" & (\%) & (\%) & (\%) & (\%) & (\%) & AUC & AUC \\",
        r"\midrule",
    ]
    for key, label in [('Edge-IIoTset_to_ToN-IoT', r'Edge $\rightarrow$ ToN'),
                       ('ToN-IoT_to_Edge-IIoTset', r'ToN $\rightarrow$ Edge')]:
        m = data['cross_domain'][key]
        lines.append(
            f"{label} & {pct(m['accuracy'])} & {pct(m['precision'])} & "
            f"{pct(m['recall'])} & \\textbf{{{pct(m['f1'])}}} & {pct(m['macro_f1'])} & "
            f"{auc(m['roc_auc'])} & {auc(m['pr_auc'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    text = "\n".join(lines)
    (OUT_DIR / "tab_crossdomain.tex").write_text(text)
    print("Generated: tab_crossdomain.tex")


def tab_adversarial():
    """Table 5: Adversarial evaluation summary."""
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Adversarial Evaluation: Supervised Evasion and AE Anomaly Coverage}",
        r"\label{tab:adversarial}",
        r"\centering",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Attack} & $\varepsilon$ & \textbf{ASR (\%)} & \textbf{Evaded} & \textbf{AE Catch} \\",
        r"\midrule",
    ]
    for r in adv:
        ds = r['dataset'].replace('-', '{-}')
        lines.append(
            f"{ds} & {r['attack']} & {r['epsilon']} & "
            f"{r['attack_success_rate']*100:.2f} & {r['evaded_supervised']} & "
            f"{r['ae_catch_rate']*100:.1f}\\% \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    text = "\n".join(lines)
    (OUT_DIR / "tab_adversarial.tex").write_text(text)
    print("Generated: tab_adversarial.tex")


def tab_xai():
    """Table 6: XAI consensus correlation."""
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{XAI Model Consensus: RF--MLP Feature Importance Rank Correlation}",
        r"\label{tab:xai}",
        r"\centering",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Spearman $\rho$} & \textbf{$p$-value} \\",
        r"\midrule",
    ]
    for ds in ['Edge-IIoTset', 'ToN-IoT']:
        label = ds.replace('-', '{-}')
        rho = xai[ds]['consensus_spearman_rho']
        pval = xai[ds]['consensus_p_value']
        lines.append(f"{label} & {rho:.4f} & {pval:.2e} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    text = "\n".join(lines)
    (OUT_DIR / "tab_xai.tex").write_text(text)
    print("Generated: tab_xai.tex")


def tab_features():
    """Table 2: Feature representations."""
    lines = [
        r"\begin{table*}[htbp]",
        r"\caption{Multi-Level Feature Representation}",
        r"\label{tab:features}",
        r"\centering",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"\textbf{Feature} & \textbf{Level} & \textbf{Semantic Definition} & \textbf{Valid For} \\",
        r"\midrule",
    ]
    for feat in schema['features']:
        name = feat['name'].replace('_', r'\_')
        level = feat['level'].capitalize()
        sem = feat['semantic_definition'][:60]
        valid = feat.get('valid_for', 'both').replace('_', ' ')
        lines.append(f"{name} & {level} & {sem} & {valid} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    text = "\n".join(lines)
    (OUT_DIR / "tab_features.tex").write_text(text)
    print("Generated: tab_features.tex")


def tab_datasets():
    """Table 1: Dataset characteristics."""
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Dataset Characteristics}",
        r"\label{tab:datasets}",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Test Rows} & \textbf{In-Domain} & \textbf{Cross-Domain} & \textbf{Attack Ratio} \\",
        r" & & \textbf{Features} & \textbf{Features} & \textbf{(Test)} \\",
        r"\midrule",
    ]
    for ds in ['Edge-IIoTset', 'ToN-IoT']:
        label = ds.replace('-', '{-}')
        m = data['in_domain'][ds]
        n_test = m['test_rows']
        tp = m['supervised_metrics']['tp']
        fn = m['supervised_metrics']['fn']
        fp = m['supervised_metrics']['fp']
        tn = m['supervised_metrics']['tn']
        atk_ratio = (tp + fn) / n_test * 100
        lines.append(
            f"{label} & {n_test:,} & 13 & 6 & {atk_ratio:.1f}\\% \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    text = "\n".join(lines)
    (OUT_DIR / "tab_datasets.tex").write_text(text)
    print("Generated: tab_datasets.tex")


if __name__ == '__main__':
    tab_datasets()
    tab_features()
    tab_indomain()
    tab_crossdomain()
    tab_adversarial()
    tab_xai()
    print("\nAll LaTeX tables generated.")
