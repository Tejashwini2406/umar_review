import base64
import hashlib
from pathlib import Path

ab_path = Path("ieee_paper_draft/sections/abstract.tex")
intro_path = Path("ieee_paper_draft/sections/introduction.tex")

ab_raw = ab_path.read_bytes()
intro_raw = intro_path.read_bytes()

ab_text = ab_raw.decode("utf-8")
intro_text = intro_raw.decode("utf-8")

print("=" * 80)
print("FILE:", ab_path.as_posix())
print("SIZE:", len(ab_raw))
print("SHA256:", hashlib.sha256(ab_raw).hexdigest())

print("\nBASE64_START")
print(base64.b64encode(ab_raw).decode("ascii"))
print("BASE64_END")

print("\nFORBIDDEN PATTERN AUDIT - ABSTRACT")
patterns = [
    r"\\%",
    r"\textbf{**",
    r"P\_",
    r"\\\_",
    "**"
]
for p in patterns:
    # Use exact literal string count
    cnt = ab_text.count(p)
    print(f"  {repr(p)}: {cnt}")

print("\nPOSITIVE PATTERN AUDIT - ABSTRACT")
ab_positives = [
    r"$86.67\%$",
    r"$73.13\%$",
    r"$0.00\%$"
]
for p in ab_positives:
    print(f"  {repr(p)}: {'EXISTS' if p in ab_text else 'MISSING'}")

print("=" * 80)
print("FILE:", intro_path.as_posix())
print("SIZE:", len(intro_raw))
print("SHA256:", hashlib.sha256(intro_raw).hexdigest())

print("\nBASE64_START")
print(base64.b64encode(intro_raw).decode("ascii"))
print("BASE64_END")

print("\nFORBIDDEN PATTERN AUDIT - INTRODUCTION")
# In Python text string:
# 1. "\\%" matches double backslash %
# 2. "\textbf{**" matches \textbf{**
# 3. "P\_{" matches escaped subscript between P and {
# 4. "\\\_" matches double escaped underscore \\_
# 5. "**" matches markdown stars
for p in [r"\\%", r"\textbf{**", r"P\_{", r"\\\_", "**"]:
    cnt = intro_text.count(p)
    print(f"  {repr(p)}: {cnt}")

print("\nPOSITIVE PATTERN AUDIT - INTRODUCTION")
intro_positives = [
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
for p in intro_positives:
    print(f"  {repr(p)}: {'EXISTS' if p in intro_text else 'MISSING'}")
