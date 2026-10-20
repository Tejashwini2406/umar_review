import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_md(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

in_domain = read_json(ROOT / "reports" / "experiments" / "in_domain_evaluation.json")
cross_domain = read_json(ROOT / "reports" / "experiments" / "cross_domain_evaluation.json")

datasets = ["CICIDS2017", "Edge-IIoTset", "BoT-IoT", "N-BaIoT"]

# 1. Final in-domain metrics table & RF vs MLP vs AE comparison & RF+MLP+AE ensemble table
in_domain_md = "# Primary In-Domain Performance\n\n"
in_domain_md += "| Dataset | Model | Accuracy | Precision | Recall | F1 Score |\n"
in_domain_md += "|---|---|---|---|---|---|\n"

for ds in datasets:
    if ds in in_domain and ds in in_domain[ds]:
        models = in_domain[ds][ds]
        for m in ["rf", "mlp", "ae", "rf+mlp+ae"]:
            if m in models:
                metrics = models[m]
                acc = metrics.get("accuracy", 0)
                prec = metrics.get("precision", 0)
                rec = metrics.get("recall", 0)
                f1 = metrics.get("f1_score", 0)
                in_domain_md += f"| **{ds}** | {m.upper()} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} |\n"

write_md(ROOT / "reports" / "tables" / "in_domain_metrics.md", in_domain_md)

# 2. Cross-domain F1 matrix
cross_domain_md = "# Cross-Domain Transferability (F1 Score - Universal 4-Feature Profile)\n\n"
ceb_datasets = ["CICIDS2017", "Edge-IIoTset", "BoT-IoT"]
cross_domain_md += "| Source Train Dataset | -> CICIDS2017 Test | -> Edge-IIoTset Test | -> BoT-IoT Test |\n"
cross_domain_md += "|---|---|---|---|\n"

for src in ceb_datasets:
    row = f"| **{src}** |"
    for dst in ceb_datasets:
        val = "0.0000"
        if src in cross_domain and dst in cross_domain[src]:
            res = cross_domain[src][dst].get("rf+mlp+ae", {})
            val = f"{res.get('f1_score', 0):.4f}"
        row += f" {val} |"
    cross_domain_md += row + "\n"

write_md(ROOT / "reports" / "tables" / "cross_domain_matrix.md", cross_domain_md)

# 3. Feature-profile table
profile_md = "# Feature Representation Profiles\n\n"
profile_md += "| Dataset | Semantic Profile | Feature Count | Target Track |\n"
profile_md += "|---|---|---|---|\n"
profile_md += "| CICIDS2017 | IN_DOMAIN_CICIDS2017 | 18 | Primary |\n"
profile_md += "| Edge-IIoTset | IN_DOMAIN_EDGE_IIOT | 7 | Primary |\n"
profile_md += "| BoT-IoT | IN_DOMAIN_BOT_IOT | 18 | Primary |\n"
profile_md += "| N-BaIoT | NBAIOT_SOURCE_AGGREGATE | 115 | Primary |\n"
profile_md += "| C/E/B Intersection | FLOW_COMPATIBLE_C_E_B | 4 | Control (Transfer) |\n"
write_md(ROOT / "reports" / "tables" / "feature_profiles.md", profile_md)

print("Tables generated in reports/tables/")
