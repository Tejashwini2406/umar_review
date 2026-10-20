import os

rw_path = r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\related_work.tex"
tbl_path = r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\tables\research_gap.tex"

with open(rw_path, 'r', encoding='utf-8') as f:
    rw_text = f.read()

with open(tbl_path, 'r', encoding='utf-8') as f:
    tbl_text = f.read()

# Task 7: Forensic Pattern Audit
forbidden = {
    r"\textbf{**": rw_text.count(r"\textbf{**") + tbl_text.count(r"\textbf{**"),
    r"**}": rw_text.count(r"**}") + tbl_text.count(r"**}"),
    r"*et al.*": rw_text.count(r"*et al.*") + tbl_text.count(r"*et al.*"),
    r"P\_{": rw_text.count(r"P\_{") + tbl_text.count(r"P\_{"),
    r"\\\_": rw_text.count(r"\\\_") + tbl_text.count(r"\\\_"),
    r"\\%": rw_text.count(r"\\%") + tbl_text.count(r"\\%"),
    r"**": rw_text.count(r"**") + tbl_text.count(r"**"),
    r"Option C": rw_text.count("Option C") + tbl_text.count("Option C")
}

print("=== FORENSIC PATTERN AUDIT ===")
for p, cnt in forbidden.items():
    print(f"  {p}: related_work={rw_text.count(p)}, research_gap={tbl_text.count(p)}, Total={cnt}")

linewidth_standalone = r"\centering" + "\n\n" + r"\linewidth" in tbl_text or "\n\\linewidth\n" in tbl_text
print(f"  Standalone \\linewidth in table: {'FOUND (FAIL)' if linewidth_standalone else 'NONE (PASS)'}")

print("\n=== LATEX-SAFE SYMBOLS AUDIT ===")
print("  $\\checkmark$ in table:", r"$\checkmark$" in tbl_text)
print("  $\\times$ in table:", r"$\times$" in tbl_text)
print("  Unicode tick count:", tbl_text.count("\u2713"))
print("  Unicode cross count:", tbl_text.count("\u2717") + tbl_text.count("\u2718") + tbl_text.count("\u274c") + tbl_text.count("✗"))
print("  Unicode em-dash count:", tbl_text.count("\u2014") + tbl_text.count("—"))
