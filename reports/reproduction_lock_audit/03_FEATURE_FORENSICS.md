# 21-FEATURE PIPELINE RECONSTRUCTION
- **Continuous Features (17)**: e.g., flow_duration, flow_bytes_per_sec, mean_pkt_size. Allowed to change during PGD.
- **Categorical/Protocol Features (4)**: proto_tcp, proto_udp, proto_icmp, proto_other. Frozen during PGD (mask=0.0).
- **Scaling**: StandardScaler is fitted on train, transformed on test.
- **Leakage Risk**: Low due to strict split logic, but subsetting to 7,000 is arbitrary.
