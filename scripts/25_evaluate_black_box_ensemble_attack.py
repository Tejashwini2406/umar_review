"""Evaluate a score-query SPSA attack against the frozen Option C ensemble.

The attack queries the actual RF+MLP predictor; it does not differentiate an RF
surrogate. Metrics characterize the selected checkpoint and are not robustness
guarantees for other training seeds or physically realizable flow mutations.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from iot_ids.adversarial.black_box import spsa_attack
from iot_ids.experiments.provenance import read_verified_splits, sha256_file
from iot_ids.runtime.engine import InferenceEngine
from iot_ids.runtime.schema import STANDARDIZED_21_FEATURES
from iot_ids.utils.paths import REPO_ROOT

PROTOCOL_FEATURES = {"proto_tcp", "proto_udp", "proto_icmp", "proto_other"}


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total < 1:
        raise ValueError("Confidence intervals require at least one sample")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _predict_option_c(engine: InferenceEngine, inputs: np.ndarray) -> np.ndarray:
    if engine.rf_model is None or engine.robust_mlp_model is None:
        raise RuntimeError("The standardized Option C model artifacts were not loaded")
    rf_probability = engine.rf_model.predict_proba(inputs)[:, 1]
    with torch.no_grad():
        mlp_logits = engine.robust_mlp_model(
            torch.as_tensor(inputs, dtype=torch.float32)
        ).squeeze(-1)
        mlp_probability = torch.sigmoid(mlp_logits).cpu().numpy()
    scores = 0.7 * rf_probability + 0.3 * mlp_probability
    if not np.isfinite(scores).all():
        raise ValueError("Option C returned non-finite probabilities")
    return scores


def evaluate_black_box_attack(
    dataset: str,
    data_dir: Path,
    models_dir: Path,
    output_dir: Path,
    *,
    sample_limit: int = 128,
    epsilon: float = 0.10,
    alpha: float = 0.025,
    steps: int = 40,
    probe_scale: float = 0.01,
    seed: int = 42,
) -> dict[str, Any]:
    if sample_limit < 1:
        raise ValueError("sample_limit must be at least 1")

    model_dir = models_dir / dataset
    validation, test = read_verified_splits(dataset, data_dir, model_dir)
    del validation

    dataset_dir = data_dir / dataset
    train_path = dataset_dir / "train.parquet"
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing training split required for feature bounds: {train_path}")
    train = pd.read_parquet(train_path)
    missing = [
        name for name in (*STANDARDIZED_21_FEATURES, "label") if name not in train.columns
    ]
    if missing:
        raise ValueError(f"{dataset} training split is missing columns: {missing}")
    manifest = json.loads(
        (dataset_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
    expected_train_counts = manifest["label_counts"]["train"]
    actual_train_counts = {
        "benign": int((train["label"] == 0).sum()),
        "attack": int((train["label"] == 1).sum()),
    }
    if expected_train_counts != actual_train_counts:
        raise ValueError("Training labels do not match the corrected split manifest")

    engine = InferenceEngine(
        profile="standardized_21",
        dataset=dataset,
        models_base_dir=models_dir,
    )
    train_inputs = engine._transform_preprocessor_frozen(
        train.loc[:, STANDARDIZED_21_FEATURES]
    )
    test_inputs = engine._transform_preprocessor_frozen(
        test.loc[:, STANDARDIZED_21_FEATURES]
    )

    clean_scores = _predict_option_c(engine, test_inputs)
    labels = test["label"].to_numpy(dtype=np.int8)
    eligible = np.flatnonzero((labels == 1) & (clean_scores >= engine.decision_threshold))
    if len(eligible) == 0:
        raise ValueError(f"{dataset} test split has no correctly detected attack flows")

    rng = np.random.default_rng(seed)
    selected = np.sort(
        rng.choice(eligible, size=min(sample_limit, len(eligible)), replace=False)
    )
    selected_inputs = test_inputs[selected]
    feature_mask = np.asarray(
        [name not in PROTOCOL_FEATURES for name in STANDARDIZED_21_FEATURES],
        dtype=bool,
    )

    result = spsa_attack(
        lambda candidate: _predict_option_c(engine, candidate),
        selected_inputs,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        probe_scale=probe_scale,
        feature_mask=feature_mask,
        x_min=np.min(train_inputs, axis=0),
        x_max=np.max(train_inputs, axis=0),
        seed=seed,
    )
    adversarial_scores = _predict_option_c(engine, result.adversarial)
    evaded = adversarial_scores < engine.decision_threshold
    output_path = output_dir / f"{dataset}_black_box_attack.json"
    output = {
        "dataset": dataset,
        "evaluation_status": "exploratory_checkpoint_specific",
        "experiment": "score_query_spsa_against_actual_option_c",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split_strategy": manifest["split_strategy"],
        "provenance": {
            "split_manifest_sha256": sha256_file(dataset_dir / "split_manifest.json"),
            "test_split_sha256": sha256_file(dataset_dir / "test.parquet"),
            "training_artifact_sha256": json.loads(
                (model_dir / "training_config.json").read_text(encoding="utf-8")
            )["artifact_sha256"],
        },
        "threat_model": {
            "access": "score-query access to the deployed Option C probability",
            "target": "0.7 * actual RF probability + 0.3 * actual robust-MLP probability",
            "budget_space": "RobustScaler model-input coordinates",
            "epsilon_linf": epsilon,
            "mutable_features": [
                name for name, mutable in zip(STANDARDIZED_21_FEATURES, feature_mask) if mutable
            ],
            "frozen_features": sorted(PROTOCOL_FEATURES),
            "physical_realizability": (
                "Not established; independent feature perturbations in scaled space "
                "may not correspond to valid network flows."
            ),
            "bcl_scope": (
                "The BCL is advisory and does not change Option C's ATTACK/BENIGN "
                "decision. Its alert path is not optimized by this per-flow attack."
            ),
        },
        "attack": {
            "method": "SPSA finite-difference sign updates against actual Option C scores",
            "seed": seed,
            "steps": steps,
            "alpha": alpha,
            "probe_scale": probe_scale,
            "optimization_queries_per_sample": result.queries_per_sample,
            "optimization_query_inputs": result.total_queries,
            "clean_test_score_query_inputs": len(test),
            "adversarial_evaluation_query_inputs": len(selected),
            "total_score_query_inputs": (
                len(test) + result.total_queries + len(selected)
            ),
        },
        "evaluation": {
            "test_rows": len(test),
            "correctly_detected_test_attacks": len(eligible),
            "attacks_selected_for_query_attack": len(selected),
            "successful_evasions": int(evaded.sum()),
            "attack_success_rate": float(evaded.mean()),
            "attack_success_rate_95ci_wilson": _wilson_interval(
                int(evaded.sum()), len(evaded)
            ),
            "attack_success_rate_denominator": (
                "selected test attacks correctly detected by the clean Option C model"
            ),
            "clean_probability_mean": float(clean_scores[selected].mean()),
            "adversarial_probability_mean": float(adversarial_scores.mean()),
            "decision_threshold": engine.decision_threshold,
        },
        "limitations": [
            "Results describe one frozen checkpoint, not the distribution across training seeds.",
            "This test-set attack evaluation is exploratory and does not support a universal robustness claim.",
            "Feature-space perturbations have not been checked for physical/network validity.",
        ],
        "result_file": str(output_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("Edge-IIoTset", "NF-ToN-IoT-v2", "ToN-IoT", "CICIoT2023", "BoT-IoT"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "stage3_corrected",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=REPO_ROOT / "models" / "corrected_run"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "reports" / "adversarial"
    )
    parser.add_argument("--sample-limit", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--probe-scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = evaluate_black_box_attack(
        args.dataset,
        args.data_dir,
        args.models_dir,
        args.output_dir,
        sample_limit=args.sample_limit,
        epsilon=args.epsilon,
        alpha=args.alpha,
        steps=args.steps,
        probe_scale=args.probe_scale,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
