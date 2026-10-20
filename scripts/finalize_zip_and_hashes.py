import os
import hashlib
import zipfile

SUBMISSION_DIR = "reports/final_overleaf_submission"
ZIP_PATH = os.path.join(SUBMISSION_DIR, "Adversarially_Robust_IoT_IDS_IEEE_Overleaf.zip")
HASH_FILE = os.path.join(SUBMISSION_DIR, "FILE_HASHES.sha256")

# 1. Create ZIP archive with files at root level
zip_files_map = {
    os.path.join(SUBMISSION_DIR, "IEEE_paper_final.tex"): "IEEE_paper_final.tex",
    os.path.join(SUBMISSION_DIR, "references.bib"): "references.bib",
    os.path.join(SUBMISSION_DIR, "README.md"): "README.md",
}

fig_dir = os.path.join(SUBMISSION_DIR, "figures")
for fname in os.listdir(fig_dir):
    fpath = os.path.join(fig_dir, fname)
    if os.path.isfile(fpath):
        zip_files_map[fpath] = os.path.join("figures", fname)

print("Building Overleaf ZIP Archive...")
with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for src_path, arc_name in zip_files_map.items():
        zipf.write(src_path, arc_name)
        print(f"Added to ZIP: {arc_name}")

print(f"Zip file created successfully: {ZIP_PATH}")

# 2. Generate FILE_HASHES.sha256
hashes = []
for root, dirs, files in os.walk(SUBMISSION_DIR):
    for fname in sorted(files):
        if fname == "FILE_HASHES.sha256":
            continue
        fpath = os.path.join(root, fname)
        rel_path = os.path.relpath(fpath, SUBMISSION_DIR).replace("\\", "/")
        
        sha256 = hashlib.sha256()
        with open(fpath, 'rb') as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        
        file_hash = sha256.hexdigest()
        hashes.append(f"{file_hash}  {rel_path}")

with open(HASH_FILE, 'w') as f:
    f.write("\n".join(hashes) + "\n")

print(f"Hashes successfully written to: {HASH_FILE}")
