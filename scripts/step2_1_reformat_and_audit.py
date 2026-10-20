import os
import subprocess

ab_path = r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\abstract.tex"
intro_path = r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\introduction.tex"

with open(ab_path, 'r', encoding='utf-8') as f:
    ab_read = f.read()

with open(intro_path, 'r', encoding='utf-8') as f:
    intro_read = f.read()

# Forbidden pattern counts (literal raw file checks)
ab_db_pct = ab_read.count("\\\\%")
intro_db_pct = intro_read.count("\\\\%")

ab_tb_star = ab_read.count("\\textbf{**")
intro_tb_star = intro_read.count("\\textbf{**")

ab_p_esc_sub = ab_read.count("P\\_{")
intro_p_esc_sub = intro_read.count("P\\_{")

ab_dbl_esc_u = ab_read.count("\\\\\\_")
intro_dbl_esc_u = intro_read.count("\\\\\\_")

ab_stars = ab_read.count("**")
intro_stars = intro_read.count("**")

print("=== FORBIDDEN PATTERN TABLE AUDIT ===")
print(f"\\\\%        | abstract: {ab_db_pct} | intro: {intro_db_pct} | Total: {ab_db_pct + intro_db_pct}")
print(f"\\textbf{{** | abstract: {ab_tb_star} | intro: {intro_tb_star} | Total: {ab_tb_star + intro_tb_star}")
print(f"P\\_{{       | abstract: {ab_p_esc_sub} | intro: {intro_p_esc_sub} | Total: {ab_p_esc_sub + intro_p_esc_sub}")
print(f"\\\\\\_       | abstract: {ab_dbl_esc_u} | intro: {intro_dbl_esc_u} | Total: {ab_dbl_esc_u + intro_dbl_esc_u}")
print(f"**        | abstract: {ab_stars} | intro: {intro_stars} | Total: {ab_stars + intro_stars}")

required_patterns = [
    r"$86.67\%$",
    r"$73.13\%$",
    r"$0.00\%$",
    r"\textbf{Cross-Dataset Performance Variation:}",
    r"\textbf{Adversarial Evasion Sensitivity:}",
    r"\textbf{Standardized Feature Representation:}",
    r"\textbf{Dual-Stream Probability Ensemble:}",
    r"\textbf{Domain-Constrained Adversarial Evaluation:}",
    r"\textbf{Cross-Dataset Empirical Assessment:}",
    r"$P_{\text{RF}}$",
    r"$P_{\text{MLP\_Adv}}$",
    r"$0.7P_{\text{RF}} + 0.3P_{\text{MLP\_Adv}}$"
]

print("\n=== REQUIRED PATTERN AUDIT ===")
combined = ab_read + "\n" + intro_read
for pat in required_patterns:
    exists = pat in combined
    status = "EXISTS" if exists else "MISSING"
    print(f"  {pat} -> {status}")

print("\n=== RAW FILE PROOF - ABSTRACT ===")
for line in ab_read.splitlines():
    if "86.67" in line:
        print(line)

print("\n=== RAW FILE PROOF - INTRODUCTION HEADINGS ===")
for line in intro_read.splitlines():
    if "\\textbf{" in line:
        print(line.strip())

print("\n=== RAW FILE PROOF - MODEL NOTATION ===")
for line in intro_read.splitlines():
    if "weighted soft-voting" in line:
        print(line.strip())
