# Multi-Level IoT Intrusion Detection System (IoT-IDS)

This repository contains an operational IoT intrusion-detection prototype and experimental research code. Its current evaluation is incomplete; historical research claims have been withdrawn pending the corrected data protocol below.

> **Research-status warning (October 19, 2026):** Historical metrics in this README and the current paper draft are withdrawn. Corrected split protocol, models, and BCL evaluations exist for the four locally available datasets, but results remain exploratory and not submission-ready. ToN-IoT scoring was revised after test inspection; Edge-IIoTset is order-sensitive; and BCL detection is weak on NF-ToN-IoT-v2 and CICIoT2023. A direct score-query Option C attack now runs against hash-verified corrected checkpoints, but its bounded-sample results are checkpoint-specific and do not optimize the advisory BCL alert path. See [implementation and paper-readiness status](FINAL_IMPLEMENTATION_STATUS.md).
> **Teammate handoff:** Start with [TEAMMATE_HANDOFF.md](TEAMMATE_HANDOFF.md) for setup, verification, and the safe continuation sequence.

---

## 1. Project Overview
The repository includes legacy experiments for structured flow features and domain adaptation. Their historical outputs do not establish improved cross-domain generalization and are not evidence for the approved BCL conference paper.

---

## 2. Research Methodology
The approved paper direction is a focused Behavioral Consistency Layer (BCL) security contribution. The existing multi-level/domain-adaptation research code and its historical outputs are not current evidence for that paper; see [implementation and paper-readiness status](FINAL_IMPLEMENTATION_STATUS.md).

---

## 3. System Architecture

```text
                               DEPLOYABLE ARTIFACT
                    (model.joblib, preprocessor.joblib, aligner.joblib, calibrator.joblib)
                                       │
                                       ▼
Raw Telemetry ──► Packet Capture ──► Flow Aggregator ──► Multi-Level Builder ──► IDSPredictor ──► Alerts & Metrics
(PCAP / Ingress)    (Scapy Engine)     (5-Tuple LRU)       (18 Features)        (RF Model)     (JSONL & Console)
```

---

## 4. Dataset Preparation
The Stage 3 materializer writes bounded samples under `data/processed/stage3_corrected/`, leaving historical outputs untouched. The current corrected run contains 10,000 flows per dataset except CICIoT2023, which contains 50,000 flows split 30,000/10,000/10,000 across its official train/validation/test partitions. ToN-IoT, Edge-IIoTset, and NF-ToN-IoT-v2 use seeded stratified row-level random splits; these are not chronological, host-disjoint, or campaign-disjoint evaluations. Temporal and behavioral history features are enabled only for Edge-IIoTset because the other adapters do not provide the required event-time and host provenance. Historical outputs that predate this protocol remain withdrawn, and current BCL results are exploratory:
- **ToN-IoT** (Ethernet & Wi-Fi IoT telemetry)
- **Edge-IIoTset** (Industrial IoT protocol telemetry)
- **NF-ToN-IoT-v2** (NetFlow v9 feature representation of ToN-IoT)
- **CICIoT2023** (33-device IoT attack capture)
- **BoT-IoT** is the planned independent holdout; the adapter is available, but no local raw files, corrected split, trained model, or BCL evaluation exists for it yet.

BoT-IoT is the planned independent evaluation dataset. A streaming CSV adapter and corrected materialization path are registered, but the official SharePoint file-download endpoint requires authentication and returned HTTP 403 here; no files or BoT-IoT results are included in the corrected runs. Download the official 5% subset through authorized access and extract its four full-feature CSV files under `data/raw/BoT-IoT/`; `data_names.csv` is treated as metadata. The source-file split is frozen as files 1-2 for train, file 3 for validation/calibration, and file 4 for test. Materialization checks exact filenames, hashes all source CSVs, and fails if any resulting split lacks benign or attack samples. The adapter preserves source order and rejects decreasing timestamps within each file. The `flgs` field only supports flag-presence indicators, not packet-level SYN/ACK/RST counts, so feature semantics must be audited before making cross-dataset claims. See the [official UNSW BoT-IoT page](https://research.unsw.edu.au/projects/bot-iot-dataset).

---

## 5. Feature Architecture
The system defines **18 conceptual semantic features** expanding to **21 numerical model matrix columns**:
- **Level A (Instantaneous)**: 8 conceptual features / 11 numerical columns (flow duration, rates, payload ratio, packet ratio, SYN ratio, 4-way protocol one-hot indicators).
- **Level B (Causal Temporal)**: 5 conceptual/numerical features (EWMA inter-arrival timing mean, CV, flow rate EWMA, byte rate EWMA, SYN rate EWMA).
- **Level C (Causal Behavioral)**: 5 conceptual/numerical features (destination IP diversity, destination port entropy, fanout ratio, unanswered ratio, activity EWMA).

For the current Stage 3 policy, Level B and C are populated only for Edge-IIoTset; these feature columns are zeroed for other datasets rather than derived from synthetic timestamps or identities.

---

## 6. Experimental Methodology
The repository contains an operational IDS, a BCL runtime, corrected sample data, model artifacts, and BCL evaluation outputs. Research validation is still incomplete: the results are exploratory, bounded samples do not establish generalization, and the current BCL results are not consistently useful across datasets. The BoT-IoT independent evaluation path is implemented but cannot run until the official files are available locally. See [implementation and paper-readiness status](FINAL_IMPLEMENTATION_STATUS.md) and the [publication remaining-work checklist](PUBLICATION_REMAINING_WORK.md).

---

## 7. Historical Research Results (Withdrawn Pending Re-evaluation)
- **Within-Domain Benchmark**: `full_multilevel` Random Forest reaches **0.986 Macro F1** and slashes FPR to **1.6%**.
- **Minimal Target Adaptation**: 5% target label adaptation (42--210 samples) recovers cross-domain transfer to **0.992 ROC-AUC** (95% CI [0.989, 0.996], Cohen's $d_z = 1.134, p = 0.0024$).
- **Shortcut Independence**: Stripping protocol indicators and raw rates retains **97.2% of performance**.

---

## 8. Installation

```bash
# Clone repository & install package in editable production mode
git clone https://github.com/Tejashwini2406/majorproject.git
cd majorproject
python -m venv .venv
# PowerShell:
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,models]"
```

---

## 9. Training

```bash
iot-ids train --source-domain ToN-IoT --target-domain Edge-IIoTset --model-family RandomForest --artifact-dir models/final/deployable_artifact
```

---

## 10. Validation

```bash
iot-ids validate --artifact-dir models/final/deployable_artifact --test-domain ToN-IoT
```

---

## 11. Corrected BCL Evaluation (Exploratory; Not Submission-Ready)

The current corrected sample run has been materialized, trained, and evaluated for the four datasets with local raw data. These commands reproduce it from local raw data; they leave historical Stage 3 outputs untouched but overwrite the corrected sample artifacts and model/evaluation outputs.

```bash
python scripts/03_materialize_and_audit_features.py
python scripts/03_materialize_and_audit_features.py --dataset CICIoT2023 --sample-limit 50000
python scripts/16_train_bcl_baselines.py --dataset ToN-IoT
python scripts/15_evaluate_bcl.py --dataset ToN-IoT
python scripts/16_train_bcl_baselines.py --dataset Edge-IIoTset
python scripts/15_evaluate_bcl.py --dataset Edge-IIoTset
python scripts/16_train_bcl_baselines.py --dataset NF-ToN-IoT-v2
python scripts/15_evaluate_bcl.py --dataset NF-ToN-IoT-v2
python scripts/16_train_bcl_baselines.py --dataset CICIoT2023
python scripts/15_evaluate_bcl.py --dataset CICIoT2023
```

The BCL-specific trainer avoids the unrelated privacy/XAI experiment workload and writes detector, autoencoder, and calibration/evaluation artifacts under `models/corrected_run/`. BCL compares each signal with the benign validation distribution and scores unusually high **or low** values; its alert threshold is calibrated on benign validation rows only. The ToN-IoT scoring rule was revised after an earlier look at that test split, so its results cannot confirm the current method. Edge-IIoTset also has substantial stream-order sensitivity, while NF-ToN-IoT-v2 and CICIoT2023 have low BCL attack detection at the calibrated operating point. Outputs record these caveats and must not be presented as confirmatory paper evidence. The row-level random splits for three datasets do not support time-, host-, or campaign-disjoint generalization claims.

## Differential Privacy (Exploratory)

The DP-SGD utility sweep is run as part of the real-data golden pipeline, not by the retired synthetic-data runner:

```powershell
python scripts/run_golden_pipeline.py
```

The DP guarantee applies only to neural-model training under the reported Poisson sampling, clipping, noise, and RDP accountant. It does not cover the Random Forest, preprocessing, BCL calibration, or fused ensemble. Results produced by the corrected mechanism have not yet been recomputed and remain exploratory. `scripts/run_privacy_experiments.py` exits with a warning because its former Gaussian data was synthetic.

## Packet-limit behavior

For `replay-pcap`, `predict-live`, `run-daemon`, and `predict-batch`, omitting `--limit` leaves processing uncapped. `--limit 0` processes no records/packets; negative limits are rejected.

The 50,000-row CICIoT2023 sample provides 204 benign validation flows, enough to execute the 1% false-positive calibration guard; the smaller 10,000-row sample had only 52. Neither sample size is a substitute for a fresh confirmatory holdout.

BoT-IoT has not yet been materialized or evaluated: the raw data is not present. After placing the official four CSV files under `data/raw/BoT-IoT/`, run the locked source-disjoint protocol and BCL pipeline:

```bash
python scripts/03_materialize_and_audit_features.py --dataset BoT-IoT --sample-limit 50000
python scripts/16_train_bcl_baselines.py --dataset BoT-IoT
python scripts/15_evaluate_bcl.py --dataset BoT-IoT
```

The evaluator marks results exploratory until the independent holdout's provenance and non-inspection are documented. If any locked split lacks both labels or the source files fail the chronology/schema checks, do not alter the split after viewing outcomes; record the protocol failure and obtain a suitable holdout.

---

## 12. Query-Based Option C Attack (Exploratory)

The following command runs a score-query SPSA attack directly against the frozen `0.7 * RF + 0.3 * RobustMLP` predictor. It checks corrected split/model provenance before scoring. The default attack uses up to 128 correctly detected test attacks, 40 SPSA steps, an `L-infinity` budget of `0.10`, and 80 optimization score queries per selected flow; clean-test filtering and adversarial verification queries are counted separately in the output. Results are checkpoint-specific and do not establish physical realizability, seed-wide robustness, or evasion of the separate advisory BCL alert path.

```powershell
python scripts/25_evaluate_black_box_ensemble_attack.py --dataset NF-ToN-IoT-v2
```

Each run writes a provenance-bound JSON result to `reports/adversarial/<dataset>_black_box_attack.json`. Use validation data to pre-register attack settings before any confirmatory test evaluation.

---

## 13. Batch Inference

```bash
iot-ids predict-batch --artifact-dir models/final/deployable_artifact --input-file data/processed/stage3/ToN-IoT/test.parquet --limit 10 --output-json reports/batch_alerts.json
```

---

## 14. Streaming Inference

```bash
iot-ids predict-stream --artifact-dir models/final/deployable_artifact --input-json data/sample_stream_packets.json
```

---

## 15. PCAP Replay

```bash
iot-ids replay-pcap --artifact-dir models/final/deployable_artifact --pcap-file data/sample_reproduce_stream.pcap
```

---

## 16. Live Capture

```bash
iot-ids predict-live --artifact-dir models/final/deployable_artifact --interface eth0 --bpf-filter "ip"
```

---

## 17. Runtime Daemon

```bash
iot-ids run-daemon --artifact-dir models/final/deployable_artifact --pcap-file data/sample_reproduce_stream.pcap --log-file reports/daemon_alerts.jsonl
```

---

## 18. Docker Deployment

```bash
docker-compose up --build
```

---

## 19. Testing

```bash
python -m pytest -q
```
Latest full-suite run (October 19, 2026): **174 passed, 8 skipped, 0 failed** in 36.44 seconds. It reported 151 warnings, primarily sklearn artifact-version mismatches and SHAP deprecations; skips cover optional or legacy artifact-dependent checks.

---

## 20. Reproducibility
See [reproducibility_checklist.md](reports/final/reproducibility_checklist.md) for full step-by-step reproduction instructions.

---

## 21. Limitations
1. Live network sniffing (`predict-live`) requires elevated network interface privileges (`root` / `CAP_NET_RAW` on Linux, Administrator/Npcap on Windows).
2. Enterprise SIEM alert exporters (Kafka, Syslog) are not implemented; alerts persist locally via `JSONLFileSink` and `ConsoleAlertSink`.

---

## 22. Repository Structure
- `src/iot_ids/`: Core operational package (`config`, `data`, `features`, `experiments`, `adaptation`, `pipeline`, `inference`, `registry`, `runtime`, `utils`)
- `scripts/`: Materialization, benchmarking, publication audit, and operational scripts
- `reports/`: Research evidence CSVs, LaTeX publication source (`reports/stage9/IEEE_paper_final.tex`), and final audit matrices (`reports/final/`)
- `tests/`: automated unit and integration tests.

---

## 23. Citation

There is no publication citation for this project yet. Do not cite the withdrawn IEEE draft or use the historical BibTeX entry from earlier project versions as a published paper.
