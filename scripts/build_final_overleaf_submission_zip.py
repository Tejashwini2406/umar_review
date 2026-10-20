import os
import zipfile

FINAL_DIR = "final_paper"
ZIP_PATHS = [
    "Adversarially_Robust_IoT_IDS_IEEE_Submission.zip",
    os.path.join(FINAL_DIR, "Adversarially_Robust_IoT_IDS_IEEE_Submission.zip")
]

zip_files_map = {
    os.path.join(FINAL_DIR, "IEEE_paper_final.tex"): "IEEE_paper_final.tex",
    os.path.join(FINAL_DIR, "references.bib"): "references.bib",
    os.path.join(FINAL_DIR, "README.md"): "README.md",
    os.path.join(FINAL_DIR, "PAPER_STRUCTURE_MAPPING.md"): "PAPER_STRUCTURE_MAPPING.md",
}

fig_dir = os.path.join(FINAL_DIR, "figures")
if os.path.exists(fig_dir):
    for fname in os.listdir(fig_dir):
        fpath = os.path.join(fig_dir, fname)
        if os.path.isfile(fpath):
            zip_files_map[fpath] = os.path.join("figures", fname)

for z_target in ZIP_PATHS:
    print(f"Creating Overleaf ZIP submission package at: {z_target}")
    with zipfile.ZipFile(z_target, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for src_path, arc_name in zip_files_map.items():
            if os.path.exists(src_path):
                zipf.write(src_path, arc_name)
                print(f"  Added: {arc_name}")

print("Master submission ZIP packages successfully created!")
