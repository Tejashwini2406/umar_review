import os
import hashlib
import base64
import subprocess

protected_files = [
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\abstract.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\introduction.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\related_work.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\tables\research_gap.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\experimental_setup.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\results_and_discussion.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\limitations.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\conclusion.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\main.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\references.bib",
]

methodology_path = r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\methodology.tex"

# Read raw bytes of methodology.tex
with open(methodology_path, 'rb') as f:
    meth_bytes = f.read()

meth_text = meth_bytes.decode('utf-8')

print("\n=== FORENSIC FILE INTEGRITY (methodology.tex) ===")
print("SIZE:", len(meth_bytes))
print("SHA256:", hashlib.sha256(meth_bytes).hexdigest())

forbidden = {
    r"\textbf{**": meth_text.count(r"\textbf{**"),
    r"**}": meth_text.count(r"**}"),
    r"*et al.*": meth_text.count(r"*et al.*"),
    r"P\_{": meth_text.count(r"P\_{"),
    r"\\\_": meth_text.count(r"\\\_"),
    r"\\%": meth_text.count(r"\\%"),
    r"**": meth_text.count(r"**"),
}

print("\n=== FORBIDDEN PATTERN AUDIT ===")
for p, cnt in forbidden.items():
    print(f"  {p}: {cnt}")

print("\n=== BASE64 ENCODING (methodology.tex) ===")
print("BASE64_START")
print(base64.b64encode(meth_bytes).decode("ascii"))
print("BASE64_END")

