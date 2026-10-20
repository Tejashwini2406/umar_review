import os
import hashlib
import re

manuscript_files = [
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\main.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\abstract.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\introduction.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\related_work.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\methodology.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\experimental_setup.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\results_and_discussion.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\limitations.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\conclusion.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\tables\research_gap.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\tables\dataset_summary.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\tables\experimental_results.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\tables\adversarial_results.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\references.bib",
]

expected_hashes = {
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\main.tex": "e65ae797ddbb5ef8dcfb9f5fda6b9b25ce8eb43cdc5befa7ddd9bd810657d65f",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\abstract.tex": "14178ee957628f787d681fa58031693084fb38d10d60c2dd1408ac6db392cbaf",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\introduction.tex": "c41c672932257ec3cf6a8166a22ed077da9ba643af47a925d152b3bc1c443405",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\related_work.tex": "d707087966472e36fe75d9e56d75214f6d17c143d88a0167ee7ad42ba261e604",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\methodology.tex": "a05e93ff3c10f37005050b3e9c1ed07e79ea77592b1188bfecccf7a3a2321e26",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\experimental_setup.tex": "b2a0c9247c0048fc10d106c1669958fda7963055f0f4d24d7395783ef8ec51e1",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\results_and_discussion.tex": "1f146d963c3ed6fe1bf313050606b031b6416d9980cd54b0fc57dd2d028721fe",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\limitations.tex": "c08a9f56bb2fd1fef83ab64ada28a37d1e0f3314dcb4a377afac3d7d5b6ab36b",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\conclusion.tex": "4db862c21262fba50071c374db8fff12cb9411a2bad046906fc48e74d3992912",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\tables\research_gap.tex": "7b9029f776e8aef02210323024c7389e24d6788f19dc996488336b2fb4110e18",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\tables\dataset_summary.tex": "3e66080fb1a1fd5e28f89eeda77a640fc8ac2f29614b1f121ddb70bb42d5667a",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\tables\experimental_results.tex": "1e3cea793b226650c745ca9b5682de51e295c83589a304b698542c8bdd226dd8",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\tables\adversarial_results.tex": "226fae9b289e20337cd213176deb7b0566435656a68f20dd33616a0c4e798cbc",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\references.bib": "da997958a0bd6f1a3bd75034629afbc87c437ffb7c232cb657de1af6650cba5c",
}

print("==========================================================================")
print("=== STEP 8 FORENSIC CROSS-SECTION CONSISTENCY AUDIT ===")
print("==========================================================================\n")

print("--- 1. MANUSCRIPT HASH INTEGRITY AUDIT ---")
modified_count = 0
for path in manuscript_files:
    data = open(path, 'rb').read()
    curr_hash = hashlib.sha256(data).hexdigest()
    if curr_hash != expected_hashes[path]:
        print(f"[FAIL] Modified: {os.path.basename(path)}")
        modified_count += 1
    else:
        print(f"[PASS] Unmodified: {os.path.basename(path)}")

print(f"\nTotal Manuscript Files Modified: {modified_count}")
print(f"VERDICT: {'ZERO MANUSCRIPT FILES WERE MODIFIED' if modified_count == 0 else 'MANUSCRIPT FILES MODIFIED (FAIL)'}\n")

print("--- 2. LATENCY & THROUGHPUT PHRASING AUDIT ---")
latency_keywords = ["latency", "ms", "3.5", "throughput", "samples per second"]
for path in manuscript_files:
    txt = open(path, 'r', encoding='utf-8').read()
    for line_no, line in enumerate(txt.splitlines(), 1):
        if any(kw in line.lower() for kw in ["latency", "3.5", "throughput"]):
            fname = os.path.basename(path)
            print(f"[{fname}:L{line_no}] {line.strip()}")

print("\n--- 3. FEATURE REPRESENTATION PHRASING AUDIT ---")
for path in manuscript_files:
    txt = open(path, 'r', encoding='utf-8').read()
    for line_no, line in enumerate(txt.splitlines(), 1):
        if any(kw in line.lower() for kw in ["21-dimensional", "21-column", "21 feature", "18 feature"]):
            fname = os.path.basename(path)
            print(f"[{fname}:L{line_no}] {line.strip()}")

print("\n--- 4. OPTION C ENSEMBLE FORMULATION AUDIT ---")
for path in manuscript_files:
    txt = open(path, 'r', encoding='utf-8').read()
    for line_no, line in enumerate(txt.splitlines(), 1):
        if "option c" in line.lower() or "0.7" in line or "soft-voting" in line.lower():
            fname = os.path.basename(path)
            print(f"[{fname}:L{line_no}] {line.strip()}")

print("\n--- 5. FORBIDDEN PATTERNS AUDIT ---")
forbidden = [r"\textbf{**", r"**}", r"*et al.*", r"P\_{", r"\\\_", r"\\%", r"**"]
total_forbidden = 0
for path in manuscript_files:
    txt = open(path, 'r', encoding='utf-8').read()
    for p in forbidden:
        cnt = txt.count(p)
        if cnt > 0:
            print(f"[FAIL] {os.path.basename(path)} contains '{p}': {cnt}")
            total_forbidden += cnt

if total_forbidden == 0:
    print("All forbidden patterns count across entire workspace: 0 (PASS)")
