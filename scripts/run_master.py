"""P1 Autonomous Master Execution Script.

Orchestrates the model training and evaluation phase of the P1 IDS implementation.
Performs a fast smoke test before full execution to ensure code integrity,
as per Execution Budget Policy.
"""
import time
import subprocess
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "reports" / "experiments" / "master_registry.json"

def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {}

def save_registry(registry: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

def run_script(script_name: str, smoke_test: bool = False, profile: str = "in_domain"):
    stage_id = f"{script_name}_{profile}_smoke" if smoke_test else f"{script_name}_{profile}"
    print(f"\n{'='*60}\nLaunching {script_name} (Smoke Test: {smoke_test})\n{'='*60}")
    
    registry = load_registry()
    
    # We define a fingerprint that changes if we update methodology.
    # We updated methodology to centralized preprocessing (v2)
    fingerprint = "centralized_preprocessing_v2"
    
    stage_info = registry.get(stage_id, {})
    if stage_info.get("status") == "COMPLETED" and stage_info.get("fingerprint") == fingerprint:
        print(f"[{script_name}] Skipping. Stage is already COMPLETED with valid fingerprint.")
        return
        
    registry[stage_id] = {"status": "RUNNING", "fingerprint": fingerprint}
    save_registry(registry)
    
    import os
    env = os.environ.copy()
    env["PROFILE"] = profile
    if smoke_test:
        env["SMOKE_TEST"] = "1"
        
    t0 = time.time()
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script_name)], env=env)
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with exit code {result.returncode}")
        registry[stage_id]["status"] = "FAILED"
        save_registry(registry)
        sys.exit(1)
        
    print(f"[{script_name}] Completed in {elapsed:.1f}s")
    registry[stage_id]["status"] = "COMPLETED"
    save_registry(registry)

if __name__ == "__main__":
    import os
    # Default to SMOKE_TEST if explicitly requested from command line
    force_smoke = os.environ.get("SMOKE_TEST") == "1"
    
    scripts = [
        "train_rf.py",
        "train_mlp.py",
        "train_ae.py",
        "evaluate_cross_domain.py"
    ]
    
    profiles = ["in_domain", "cross_domain"]
    
    if force_smoke:
        print(">>> FORCING SMOKE TEST PHASE ONLY <<<")
        for profile in profiles:
            for script in scripts:
                run_script(script, smoke_test=True, profile=profile)
    else:
        print(">>> BEGINNING SMOKE TEST PHASE <<<")
        for profile in profiles:
            for script in scripts:
                run_script(script, smoke_test=True, profile=profile)
        
        print("\n>>> BEGINNING FULL RESEARCH EXECUTION <<<")
        for profile in profiles:
            for script in scripts:
                run_script(script, smoke_test=False, profile=profile)
