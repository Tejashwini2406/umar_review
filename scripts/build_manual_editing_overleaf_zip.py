import os
import zipfile

SOURCE_DIR = "ieee_paper_draft"
OUTPUT_ZIP = "IoT_IDS_Final_Overleaf_Manual_Editing.zip"

# Exclude unnecessary files/directories
EXCLUDE_DIRS = {"notes", "scratch", "__pycache__", ".git", ".idea", ".vscode"}
EXCLUDE_EXTS = {".zip", ".pyc", ".tmp", ".log"}

zip_entries = []

with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(SOURCE_DIR):
        # Filter directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in sorted(files):
            _, ext = os.path.splitext(file)
            if ext in EXCLUDE_EXTS:
                continue
            
            full_path = os.path.join(root, file)
            # Relative path within ieee_paper_draft
            arc_name = os.path.relpath(full_path, SOURCE_DIR).replace("\\", "/")
            zipf.write(full_path, arc_name)
            zip_entries.append(arc_name)

print(f"Successfully created ZIP archive at: {os.path.abspath(OUTPUT_ZIP)}")
print("\n=== ZIP FILE TREE ===")
for entry in sorted(zip_entries):
    print(f"  {entry}")
