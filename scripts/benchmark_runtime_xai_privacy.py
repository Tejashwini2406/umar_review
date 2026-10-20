"""Runtime Latency Benchmark Script with XAI and DP Overhead Measurement.

Measures single-CPU host inference latency across batch sizes N in {1, 32, 64, 128, 256, 512, 1024} for:
  1. Standard Detection Pipeline (Option C probability fusion)
  2. Local Explanation Generation Pathway (Option C weighted attribution aggregation)

Separates pure detection latency from explanation generation latency.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from iot_ids.runtime.engine import InferenceEngine
from iot_ids.runtime.schema import STANDARDIZED_21_FEATURES


def main():
    print("==========================================================================")
    print("=== RUNNING RUNTIME LATENCY & OVERHEAD BENCHMARKS ===")
    print("==========================================================================\n")

    engine = InferenceEngine(profile="standardized_21", dataset="ToN-IoT")

    batch_sizes = [1, 32, 64, 128, 256, 512, 1024]
    dataset = {feat: 1.0 for feat in STANDARDIZED_21_FEATURES}

    detection_benchmarks = {}
    explanation_benchmarks = {}

    print("--> 1. Benchmarking Pure Detection Pipeline Latency...")
    # Cold start N=1
    t0 = time.perf_counter()
    _ = engine.predict(dataset, explain=False)
    cold_latency_ms = (time.perf_counter() - t0) * 1000.0
    print(f"    Cold Start N=1 Latency : {cold_latency_ms:.2f} ms")

    for b_size in batch_sizes:
        batch_inputs = [dataset] * b_size
        
        # Warmup
        _ = engine.predict(batch_inputs[:min(4, b_size)], explain=False)

        latencies = []
        for _ in range(20):
            t_start = time.perf_counter()
            _ = engine.predict(batch_inputs, explain=False)
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0)

        mean_batch_ms = float(np.mean(latencies))
        per_sample_ms = mean_batch_ms / b_size
        throughput = float(b_size / (mean_batch_ms / 1000.0))

        detection_benchmarks[str(b_size)] = {
            "batch_size": b_size,
            "mean_batch_latency_ms": mean_batch_ms,
            "per_sample_latency_ms": per_sample_ms,
            "throughput_samples_per_sec": throughput,
        }

        print(f"    Batch N={b_size:4d} | Per-Sample: {per_sample_ms:7.4f} ms | Throughput: {throughput:8.1f} samples/sec")

    print("\n--> 2. Benchmarking Local Explanation Generation Overhead...")
    ex_batch_sizes = [1, 8, 16, 32]
    for b_size in ex_batch_sizes:
        batch_inputs = [dataset] * b_size

        latencies = []
        for _ in range(5):
            t_start = time.perf_counter()
            _ = engine.predict(batch_inputs, explain=True, top_k=5)
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0)

        mean_batch_ms = float(np.mean(latencies))
        per_sample_ms = mean_batch_ms / b_size

        explanation_benchmarks[str(b_size)] = {
            "batch_size": b_size,
            "mean_explanation_latency_ms": mean_batch_ms,
            "per_sample_explanation_ms": per_sample_ms,
        }

        print(f"    Batch N={b_size:2d} | Per-Sample Explanation: {per_sample_ms:7.2f} ms")

    reports_dir = REPO_ROOT / "reports" / "runtime"
    reports_dir.mkdir(parents=True, exist_ok=True)

    benchmark_summary = {
        "cold_start_latency_ms": cold_latency_ms,
        "pure_detection_benchmarks": detection_benchmarks,
        "explanation_benchmarks": explanation_benchmarks,
    }

    with open(reports_dir / "runtime_overhead_summary.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print(f"\nSuccessfully generated Runtime Benchmark Summary in {reports_dir}")


if __name__ == "__main__":
    main()
