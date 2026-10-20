# Teammate Handoff

## Current state

This repository contains the IoT IDS code, the Behavioral Consistency Layer (BCL), its BoT-IoT adapter and locked source-file split, tests, and experiment provenance/configuration. The research is **not submission-ready**. Current BCL results are exploratory, the BoT-IoT files have not been evaluated, and adaptive evasion has not been tested.

Read these first:

1. [Implementation and paper-readiness status](FINAL_IMPLEMENTATION_STATUS.md) for verified work, results, and caveats.
2. [Publication remaining-work checklist](PUBLICATION_REMAINING_WORK.md) for the go/no-go criteria and next experiments.
3. [Project README](README.md) for architecture, installation, and pipeline commands.

## Repository contents and data handling

The public repository intentionally excludes raw datasets, generated processed splits, binary model weights, and the sample packet capture. They are not needed to run the unit tests, and the raw datasets must not be redistributed unless their terms explicitly allow it. No credentials or local environment files belong in Git. Obtain dataset access through each dataset's authorized source and keep files under the ignored `data/raw/` directory.

The committed JSON manifests, calibration/evaluation summaries, and configuration files are provenance and exploratory result records; they are not model weights or confirmatory evidence. Models need to be regenerated from authorized local data before inference or evaluation.

## Set up and verify

Use Python 3.10 or newer. From PowerShell:

```powershell
git clone https://github.com/Tejashwini2406/majorproject.git
Set-Location majorproject
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,models]"
pytest -q
ruff check .
```

The latest verified regression result is **174 passed, 8 skipped, 0 failed**. The BoT-IoT path is unit-tested with generated fixtures, not with the official dataset. Expect existing warnings noted in `FINAL_IMPLEMENTATION_STATUS.md`; do not present a test pass as evidence of research validity.

## Continue the BoT-IoT evaluation

1. Obtain the official BoT-IoT 5% full-feature CSV subset using authorized access. Keep the four files in `data/raw/BoT-IoT/`; do not commit them.
2. Read the locked split and data checks in `scripts/03_materialize_and_audit_features.py` and `src/iot_ids/data/adapters/bot_iot.py` before running anything. The frozen assignment is files 1-2 for training, file 3 for validation/calibration, and file 4 for test.
3. Run materialization, then confirm the file hashes, source order, schema, class coverage, and at least 100 benign validation rows. Do not inspect file-4 outcomes while making protocol or method decisions.
4. If the protocol passes, run:

   ```powershell
   python scripts/16_train_bcl_baselines.py --dataset BoT-IoT
   python scripts/15_evaluate_bcl.py --dataset BoT-IoT
   ```

5. If any frozen split fails a requirement, preserve and report the failure. Do not reassign files, tune thresholds, or revise features based on test results.

## Research priorities

- Pre-register the threat model, adaptive attack budgets, baselines, calibration procedure, and primary metrics before opening any new holdout.
- Evaluate adaptive evasion against the complete IDS+BCL runtime and compare at matched false-positive rates.
- Investigate weak BCL utility on NF-ToN-IoT-v2/CICIoT2023 and stream-order sensitivity on Edge-IIoTset using training/validation data only.
- Establish whether there is a defensible security contribution. Preserve negative results and stop or change venue if the predeclared utility/security gates fail.
- Write a new anonymized manuscript from validated evidence. The existing IEEE draft is withdrawn and must not be submitted.

See `PUBLICATION_REMAINING_WORK.md` for the full checklist, commands, and submission gate.
