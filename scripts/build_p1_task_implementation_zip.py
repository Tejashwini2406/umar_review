import os
import zipfile
import time

REPO_ROOT = os.path.abspath(".")
OUTPUT_ZIP_NAME = "p1_task_implementation.zip"
OUTPUT_ZIP_PATH = os.path.join(REPO_ROOT, OUTPUT_ZIP_NAME)

# Directories to strictly exclude
EXCLUDE_DIRS = {
    "data",
    ".venv",
    ".venv-1",
    "venv",
    "env",
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".ipynb_checkpoints"
}

# File extensions / files to exclude
EXCLUDE_EXTS = {".pyc", ".pyo", ".pyd"}
EXCLUDE_FILES = {OUTPUT_ZIP_NAME}

print(f"Starting creation of repository ZIP archive: {OUTPUT_ZIP_PATH}")
start_time = time.time()

file_count = 0
total_uncompressed_bytes = 0

with zipfile.ZipFile(OUTPUT_ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(REPO_ROOT):
        # Filter directories in-place to avoid traversing excluded subtrees
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith("__pycache__")]
        
        for file in sorted(files):
            if file in EXCLUDE_FILES or file == OUTPUT_ZIP_NAME:
                continue
            
            _, ext = os.path.splitext(file)
            if ext.lower() in EXCLUDE_EXTS:
                continue
            
            full_path = os.path.join(root, file)
            
            # Double check we are not zipping the output zip file
            if os.path.abspath(full_path) == os.path.abspath(OUTPUT_ZIP_PATH):
                continue
            
            rel_path = os.path.relpath(full_path, REPO_ROOT).replace("\\", "/")
            
            # Skip any data/ paths just in case
            if rel_path.startswith("data/") or rel_path == "data":
                continue
            
            sz = os.path.getsize(full_path)
            zipf.write(full_path, rel_path)
            file_count += 1
            total_uncompressed_bytes += sz

zip_sz_bytes = os.path.getsize(OUTPUT_ZIP_PATH)
elapsed = time.time() - start_time

print("\n=== ZIP CREATION COMPLETE ===")
print(f"Archive Path          : {OUTPUT_ZIP_PATH}")
print(f"Files Packaged        : {file_count}")
print(f"Uncompressed Size     : {total_uncompressed_bytes / (1024*1024):.2f} MB")
print(f"Compressed ZIP Size   : {zip_sz_bytes / (1024*1024):.2f} MB")
print(f"Time Taken            : {elapsed:.2f} seconds")
