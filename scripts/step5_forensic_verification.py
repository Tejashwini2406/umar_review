import os
import hashlib
import base64

protected_files = [
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\abstract.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\introduction.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\related_work.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\tables\research_gap.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\methodology.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\results_and_discussion.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\limitations.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\conclusion.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\main.tex",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\references.bib",
]

expected_protected_hashes = {
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\abstract.tex": "14178ee957628f787d681fa58031693084fb38d10d60c2dd1408ac6db392cbaf",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\introduction.tex": "c41c672932257ec3cf6a8166a22ed077da9ba643af47a925d152b3bc1c443405",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\related_work.tex": "d707087966472e36fe75d9e56d75214f6d17c143d88a0167ee7ad42ba261e604",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\tables\research_gap.tex": "7b9029f776e8aef02210323024c7389e24d6788f19dc996488336b2fb4110e18",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\methodology.tex": "a05e93ff3c10f37005050b3e9c1ed07e79ea77592b1188bfecccf7a3a2321e26",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\results_and_discussion.tex": "1f146d963c3ed6fe1bf313050606b031b6416d9980cd54b0fc57dd2d028721fe",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\limitations.tex": "c08a9f56bb2fd1fef83ab64ada28a37d1e0f3314dcb4a377afac3d7d5b6ab36b",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\conclusion.tex": "4db862c21262fba50071c374db8fff12cb9411a2bad046906fc48e74d3992912",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\main.tex": "e65ae797ddbb5ef8dcfb9f5fda6b9b25ce8eb43cdc5befa7ddd9bd810657d65f",
    r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\references.bib": "da997958a0bd6f1a3bd75034629afbc87c437ffb7c232cb657de1af6650cba5c",
}

print("=== PROTECTED MANUSCRIPT FILES HASH VERIFICATION ===")
protected_changed = False
for pf in protected_files:
    if os.path.exists(pf):
        with open(pf, 'rb') as f:
            curr_hash = hashlib.sha256(f.read()).hexdigest()
            if curr_hash != expected_protected_hashes[pf]:
                print(f"[FAIL] Protected file modified: {pf}")
                protected_changed = True

print(f"Protected manuscript files modified: {'YES (FAIL)' if protected_changed else 'NONE (PASS)'}")

target_files = {
    "experimental_setup.tex": r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\sections\experimental_setup.tex",
    "dataset_summary.tex": r"c:\Users\umari\Documents\P1_task_Implementation\ieee_paper_draft\tables\dataset_summary.tex",
}

for name, path in target_files.items():
    with open(path, 'rb') as f:
        content_bytes = f.read()
    content_text = content_bytes.decode('utf-8')

    print(f"\n=== FORENSIC FILE INTEGRITY ({name}) ===")
    print("SIZE:", len(content_bytes))
    print("SHA256:", hashlib.sha256(content_bytes).hexdigest())

    forbidden = {
        r"\textbf{**": content_text.count(r"\textbf{**"),
        r"**}": content_text.count(r"**}"),
        r"*et al.*": content_text.count(r"*et al.*"),
        r"P\_{": content_text.count(r"P\_{"),
        r"\\\_": content_text.count(r"\\\_"),
        r"\\%": content_text.count(r"\\%"),
        r"**": content_text.count(r"**"),
    }

    print(f"=== FORBIDDEN PATTERN AUDIT ({name}) ===")
    for p, cnt in forbidden.items():
        print(f"  {p}: {cnt}")

    print(f"=== BASE64 ENCODING ({name}) ===")
    print("BASE64_START")
    print(base64.b64encode(content_bytes).decode("ascii"))
    print("BASE64_END")
