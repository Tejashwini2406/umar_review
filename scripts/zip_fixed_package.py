import os
import zipfile

FIXED_DIR = "reports/fixed_overleaf_submission"
ZIP_PATH = os.path.join(FIXED_DIR, "Adversarially_Robust_IoT_IDS_IEEE_Overleaf_Fixed.zip")
ZIP_ALIAS_PATH = os.path.join(FIXED_DIR, "Adversarially_Robust_IoT_IDS_IEEE_Overleaf.zip")

zip_files_map = {
    os.path.join(FIXED_DIR, "IEEE_paper_final.tex"): "IEEE_paper_final.tex",
    os.path.join(FIXED_DIR, "references.bib"): "references.bib",
    os.path.join(FIXED_DIR, "README.md"): "README.md",
    os.path.join(FIXED_DIR, "COMPILATION_FIX_REPORT.md"): "COMPILATION_FIX_REPORT.md",
}

fig_dir = os.path.join(FIXED_DIR, "figures")
if os.path.exists(fig_dir):
    for fname in os.listdir(fig_dir):
        fpath = os.path.join(fig_dir, fname)
        if os.path.isfile(fpath):
            zip_files_map[fpath] = os.path.join("figures", fname)

for z_target in [ZIP_PATH, ZIP_ALIAS_PATH]:
    print(f"Creating ZIP archive: {z_target}")
    with zipfile.ZipFile(z_target, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for src_path, arc_name in zip_files_map.items():
            if os.path.exists(src_path):
                zipf.write(src_path, arc_name)
                print(f"  Added: {arc_name}")

print("ZIP archives successfully created!")
