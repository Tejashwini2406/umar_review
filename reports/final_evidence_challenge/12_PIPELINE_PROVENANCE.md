# PIPELINE PROVENANCE

The repository contains competing pipelines. 

1. `train_rf.py` & `train_mlp.py` (Orphaned / Legacy)
2. `10_run_adversarial_evaluation.py` (Legacy direct MLP attack script)
3. `run_golden_pipeline.py` (Master script orchestrating exact publication numbers)
4. `run_privacy_experiments.py` (Isolated synthetic demonstration)

### RECOMMENDED CANONICAL PIPELINE
`run_golden_pipeline.py` is the RECOMMENDED CANONICAL PIPELINE. It directly references the exact artifacts deposited in `models/golden_run/` and computes the exact numbers hardcoded in the downstream attack scripts. 
