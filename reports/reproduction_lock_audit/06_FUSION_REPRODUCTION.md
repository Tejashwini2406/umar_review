# FUSION REPRODUCTION
Fusion sweep from scratch (w * RF + (1-w) * Rob MLP):
## Edge-IIoTset
- Weight rf_0.0: AUC = 0.9777
- Weight rf_0.1: AUC = 0.9797
- Weight rf_0.2: AUC = 0.9803
- Weight rf_0.3: AUC = 0.9814
- Weight rf_0.4: AUC = 0.9842
- Weight rf_0.5: AUC = 0.9884
- Weight rf_0.6: AUC = 0.9922
- Weight rf_0.7: AUC = 0.9936
- Weight rf_0.8: AUC = 0.9977
- Weight rf_0.9: AUC = 0.9989
- Weight rf_1.0: AUC = 0.9995
## NF-ToN-IoT-v2
- Weight rf_0.0: AUC = 0.8327
- Weight rf_0.1: AUC = 0.8464
- Weight rf_0.2: AUC = 0.8758
- Weight rf_0.3: AUC = 0.9485
- Weight rf_0.4: AUC = 0.9822
- Weight rf_0.5: AUC = 0.9860
- Weight rf_0.6: AUC = 0.9901
- Weight rf_0.7: AUC = 0.9908
- Weight rf_0.8: AUC = 0.9932
- Weight rf_0.9: AUC = 0.9934
- Weight rf_1.0: AUC = 0.9935
## ToN-IoT
- Weight rf_0.0: AUC = 1.0000
- Weight rf_0.1: AUC = 1.0000
- Weight rf_0.2: AUC = 1.0000
- Weight rf_0.3: AUC = 1.0000
- Weight rf_0.4: AUC = 1.0000
- Weight rf_0.5: AUC = 1.0000
- Weight rf_0.6: AUC = 1.0000
- Weight rf_0.7: AUC = 1.0000
- Weight rf_0.8: AUC = 1.0000
- Weight rf_0.9: AUC = 1.0000
- Weight rf_1.0: AUC = 1.0000
## CICIoT2023
- Weight rf_0.0: AUC = 0.9840
- Weight rf_0.1: AUC = 0.9924
- Weight rf_0.2: AUC = 0.9946
- Weight rf_0.3: AUC = 0.9955
- Weight rf_0.4: AUC = 0.9959
- Weight rf_0.5: AUC = 0.9963
- Weight rf_0.6: AUC = 0.9965
- Weight rf_0.7: AUC = 0.9967
- Weight rf_0.8: AUC = 0.9969
- Weight rf_0.9: AUC = 0.9968
- Weight rf_1.0: AUC = 0.9964

**Conclusion**: 0.7/0.3 is an arbitrary heuristic. For many datasets, 1.0 (pure RF) is mathematically superior for clean performance.