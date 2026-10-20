import os
import shutil
import hashlib
import zipfile

BASE_DIR = "reports/final_overleaf_submission"
os.makedirs(os.path.join(BASE_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "verification"), exist_ok=True)

# 1. Copy references.bib
shutil.copy("reports/paper_redesign/manuscript/references.bib", os.path.join(BASE_DIR, "references.bib"))

# 2. Copy figures
fig_src_dir = "reports/paper_redesign/figures"
fig_dst_dir = os.path.join(BASE_DIR, "figures")

figures_list = [
    "framework_architecture",
    "feature_representation",
    "adversarial_pipeline",
    "runtime_architecture",
    "option_ensemble_comparison",
    "auc_comparison",
    "adversarial_asr_comparison",
    "shap_rank_agreement",
    "ensemble_stability"
]

for fig in figures_list:
    for ext in [".pdf", ".png"]:
        src_file = os.path.join(fig_src_dir, fig + ext)
        dst_file = os.path.join(fig_dst_dir, fig + ext)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            print(f"Copied figure: {fig}{ext}")

# Also copy numbered aliases figure_01_*, etc.
alias_map = {
    "framework_architecture": "figure_01_framework_architecture",
    "feature_representation": "figure_02_feature_representation",
    "adversarial_pipeline": "figure_03_adversarial_pipeline",
    "runtime_architecture": "figure_04_runtime_architecture",
    "option_ensemble_comparison": "figure_05_option_ensemble_comparison",
    "auc_comparison": "plot_01_auc_comparison",
    "adversarial_asr_comparison": "plot_02_adversarial_asr_comparison",
    "shap_rank_agreement": "plot_03_shap_rank_agreement",
    "ensemble_stability": "plot_04_ensemble_stability"
}

for src_name, alias_name in alias_map.items():
    for ext in [".pdf", ".png"]:
        src_file = os.path.join(fig_src_dir, src_name + ext)
        alias_file = os.path.join(fig_dst_dir, alias_name + ext)
        if os.path.exists(src_file):
            shutil.copy(src_file, alias_file)

# 3. Copy verification reports
verif_files = [
    "FINAL_VERIFICATION_SUMMARY.md",
    "FINAL_MANUSCRIPT_CORRECTION_REPORT.md",
    "LEVEL2_DISCREPANCY_RESOLUTION.md",
    "LEVEL2_OPTION_C_LINEAGE_CHECK.md"
]

for vf in verif_files:
    src_v = os.path.join("reports/final_verification", vf)
    dst_v = os.path.join(BASE_DIR, "verification", vf)
    if os.path.exists(src_v):
        shutil.copy(src_v, dst_v)
        print(f"Copied verification report: {vf}")

print("Assets successfully copied to submission directory!")
