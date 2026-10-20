Execution started at: 2026-09-26 17:36:17
PyTorch Version: 2.13.0+cpu | CPU Execution
Random Seed: 42

--- Processing Dataset: Edge-IIoTset ---
Dataset Edge-IIoTset: Train rows=4200, Val rows=1400, Test rows=1400
  RF ROC-AUC:         Fresh=0.999478 | Golden=0.999478
  Std MLP ROC-AUC:    Fresh=0.844327 | Golden=0.844327
  Rob MLP ROC-AUC:    Fresh=0.800177 | Golden=0.800177
  Option C ROC-AUC:   Fresh=0.998752 | Golden=0.998752
  PGD-10 ASR: Std=0.0000 | Rob=0.0000 | Option C=0.0380

--- Running Fresh XAI Global Audit on Edge-IIoTset ---

--- Processing Dataset: NF-ToN-IoT-v2 ---
Dataset NF-ToN-IoT-v2: Train rows=4200, Val rows=1400, Test rows=1400
  RF ROC-AUC:         Fresh=0.993996 | Golden=0.993996
  Std MLP ROC-AUC:    Fresh=0.839791 | Golden=0.839791
  Rob MLP ROC-AUC:    Fresh=0.829925 | Golden=0.829925
  Option C ROC-AUC:   Fresh=0.992671 | Golden=0.992671
  PGD-10 ASR: Std=0.5380 | Rob=0.4800 | Option C=0.0400

--- Processing Dataset: ToN-IoT ---
Dataset ToN-IoT: Train rows=4200, Val rows=1400, Test rows=1400
  RF ROC-AUC:         Fresh=1.000000 | Golden=1.000000
  Std MLP ROC-AUC:    Fresh=0.774510 | Golden=0.774510
  Rob MLP ROC-AUC:    Fresh=0.806290 | Golden=0.806290
  Option C ROC-AUC:   Fresh=1.000000 | Golden=1.000000
  PGD-10 ASR: Std=0.0230 | Rob=0.0250 | Option C=0.0000

--- Processing Dataset: CICIoT2023 ---
Dataset CICIoT2023: Train rows=4200, Val rows=1400, Test rows=1400
  RF ROC-AUC:         Fresh=0.996800 | Golden=0.996800
  Std MLP ROC-AUC:    Fresh=0.977570 | Golden=0.977570
  Rob MLP ROC-AUC:    Fresh=0.988443 | Golden=0.988443
  Option C ROC-AUC:   Fresh=0.996502 | Golden=0.996502
  PGD-10 ASR: Std=0.0230 | Rob=0.0130 | Option C=0.0120

--- Running Fresh DP Retraining on NF-ToN-IoT-v2 ---
  DP noise=0.0 | eps=inf | AUC=0.976334 | PGD-10 ASR=1.50%
  DP noise=0.5 | eps=15.85 | AUC=0.886465 | PGD-10 ASR=2.50%
  DP noise=1.0 | eps=2.37 | AUC=0.854060 | PGD-10 ASR=47.70%
  DP noise=2.0 | eps=0.80 | AUC=0.828417 | PGD-10 ASR=47.80%

--- Running Fresh Host Inference Runtime Benchmark ---
  Batch    1: Batch=92.94ms | Per-sample=92.9390ms | Throughput=    10.8 samples/s
  Batch   32: Batch=86.45ms | Per-sample=2.7017ms | Throughput=   370.1 samples/s
  Batch   64: Batch=86.25ms | Per-sample=1.3476ms | Throughput=   742.0 samples/s
  Batch  128: Batch=106.11ms | Per-sample=0.8290ms | Throughput=  1206.3 samples/s
  Batch  256: Batch=100.25ms | Per-sample=0.3916ms | Throughput=  2553.5 samples/s
  Batch  512: Batch=93.49ms | Per-sample=0.1826ms | Throughput=  5476.3 samples/s
  Batch 1024: Batch=98.78ms | Per-sample=0.0965ms | Throughput= 10366.4 samples/s

Fresh Reproduction Pipeline Completed in 1056.86 seconds!