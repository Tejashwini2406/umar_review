# PREPROCESSING PIPELINE INCONSISTENCY

### ARCHITECTURAL INCONSISTENCY
The project contains two preprocessing definitions:
1. `src/iot_ids/preprocessing/pipeline.py`: Uses `log1p` -> `RobustScaler` -> `clip`.
2. `scripts/run_golden_pipeline.py`: Instantiates `RobustScaler` directly, bypassing the official class entirely.

*(Status: SOURCE-VERIFIED)*

### IS THIS INTENTIONAL?
There is no documentation explicitly justifying bypassing the official class. The use of raw `RobustScaler` in `run_golden_pipeline.py` appears to be an oversight/shortcut when unifying the training scripts into a single master pipeline. 

*(Status: SOURCE-INFERRED - IMPLEMENTATION DRIFT)*

The final reported paper numbers and golden artifacts were definitely generated using the raw `RobustScaler` bypass.
