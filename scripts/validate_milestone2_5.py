from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from iot_ids.data.materialize import load_valid_manifest, materialize_dataset


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def semantic_candidates() -> list[dict[str, str]]:
    return [
        {
            "canonical_candidate": "duration_seconds",
            "CICIDS2017_source": "Flow Duration",
            "Edge-IIoTset_source": "udp.time_delta",
            "BoT-IoT_source": "dur",
            "units": "seconds after normalization",
            "semantic_equivalence": "Partial but defensible as current-record elapsed duration/delta.",
            "decision": "ACCEPTED",
            "reason": "All three expose a current flow/record duration-like measurement; units are normalizable.",
        },
        {
            "canonical_candidate": "protocol_family",
            "CICIDS2017_source": "not present",
            "Edge-IIoTset_source": "tcp/udp/icmp field presence",
            "BoT-IoT_source": "proto",
            "units": "categorical",
            "semantic_equivalence": "Not universally available.",
            "decision": "REJECTED",
            "reason": "CICIDS MachineLearningCSV schema lacks protocol field.",
        },
        {
            "canonical_candidate": "src_port",
            "CICIDS2017_source": "not present",
            "Edge-IIoTset_source": "tcp.srcport",
            "BoT-IoT_source": "sport",
            "units": "port number",
            "semantic_equivalence": "Equivalent where present.",
            "decision": "REJECTED",
            "reason": "CICIDS schema lacks source port.",
        },
        {
            "canonical_candidate": "dst_port",
            "CICIDS2017_source": "Destination Port",
            "Edge-IIoTset_source": "tcp.dstport / udp.port",
            "BoT-IoT_source": "dport",
            "units": "port number",
            "semantic_equivalence": "Defensible service/destination-port semantics.",
            "decision": "ACCEPTED",
            "reason": "All three expose destination/service port. Conditional leakage risk is documented.",
        },
        {
            "canonical_candidate": "total_packets",
            "CICIDS2017_source": "Total Fwd Packets + Total Backward Packets",
            "Edge-IIoTset_source": "not present in prepared ML CSV",
            "BoT-IoT_source": "pkts",
            "units": "packets",
            "semantic_equivalence": "Equivalent for CICIDS/BoT only.",
            "decision": "REJECTED",
            "reason": "Edge prepared ML CSV lacks packet count.",
        },
        {
            "canonical_candidate": "directional_packet_counts",
            "CICIDS2017_source": "Total Fwd Packets / Total Backward Packets",
            "Edge-IIoTset_source": "not present",
            "BoT-IoT_source": "spkts / dpkts",
            "units": "packets",
            "semantic_equivalence": "Equivalent for CICIDS/BoT only.",
            "decision": "REJECTED",
            "reason": "Edge prepared ML CSV lacks directional packet counts.",
        },
        {
            "canonical_candidate": "total_bytes",
            "CICIDS2017_source": "Total Length of Fwd Packets + Total Length of Bwd Packets",
            "Edge-IIoTset_source": "tcp.len + mqtt.len + http.content_length",
            "BoT-IoT_source": "bytes",
            "units": "bytes",
            "semantic_equivalence": "Partial but defensible as record/flow byte-length magnitude.",
            "decision": "ACCEPTED",
            "reason": "All three expose byte/length magnitude; Edge is protocol-field composite and documented as weaker.",
        },
        {
            "canonical_candidate": "directional_byte_counts",
            "CICIDS2017_source": "Total Length of Fwd/Bwd Packets",
            "Edge-IIoTset_source": "not present",
            "BoT-IoT_source": "sbytes / dbytes",
            "units": "bytes",
            "semantic_equivalence": "Equivalent for CICIDS/BoT only.",
            "decision": "REJECTED",
            "reason": "Edge prepared ML CSV lacks directional byte counts.",
        },
        {
            "canonical_candidate": "packet_length_mean",
            "CICIDS2017_source": "Packet Length Mean",
            "Edge-IIoTset_source": "tcp.len / mqtt.len / http.content_length proxy",
            "BoT-IoT_source": "mean",
            "units": "bytes",
            "semantic_equivalence": "Partial; Edge has current protocol payload length, not flow mean.",
            "decision": "ACCEPTED_CONDITIONAL",
            "reason": "Kept as a weak current-size statistic; documented as conditional rather than exact mean equivalence.",
        },
        {
            "canonical_candidate": "packet_length_std_min_max",
            "CICIDS2017_source": "Packet Length Std/Min/Max",
            "Edge-IIoTset_source": "not present",
            "BoT-IoT_source": "stddev/min/max",
            "units": "bytes",
            "semantic_equivalence": "Equivalent for CICIDS/BoT only.",
            "decision": "REJECTED",
            "reason": "Edge prepared ML CSV lacks packet-length distribution statistics.",
        },
        {
            "canonical_candidate": "rates",
            "CICIDS2017_source": "Flow Bytes/s, Flow Packets/s",
            "Edge-IIoTset_source": "not present as flow rate",
            "BoT-IoT_source": "rate/srate/drate",
            "units": "bytes/sec or packets/sec depending source",
            "semantic_equivalence": "Not universal and units differ.",
            "decision": "REJECTED",
            "reason": "Edge lacks equivalent flow-rate fields; BoT rate semantics are not safely interchangeable with all CICIDS rate fields.",
        },
        {
            "canonical_candidate": "tcp_syn_ack_rst",
            "CICIDS2017_source": "SYN/ACK/RST Flag Count",
            "Edge-IIoTset_source": "tcp.connection.syn / tcp.flags.ack / tcp.connection.rst",
            "BoT-IoT_source": "flgs/state only",
            "units": "count/indicator",
            "semantic_equivalence": "Not audited as equivalent for BoT-IoT.",
            "decision": "REJECTED",
            "reason": "BoT-IoT exposes compact flags/state, not decomposed per-record SYN/ACK/RST counts. Removed from C/E/B profile.",
        },
        {
            "canonical_candidate": "inter_arrival_statistics",
            "CICIDS2017_source": "Flow IAT Mean/Std",
            "Edge-IIoTset_source": "udp.time_delta only",
            "BoT-IoT_source": "stime/ltime/dur but no IAT distribution",
            "units": "time",
            "semantic_equivalence": "Not universal.",
            "decision": "REJECTED",
            "reason": "The fields represent different timing concepts.",
        },
    ]


def split_sanity(audit: dict) -> dict[str, object]:
    datasets = {ds["name"]: ds for ds in audit["datasets"]}
    return {
        "CICIDS2017": {
            "split_unit": "source daily CSV file",
            "grouping_key": "source_file",
            "chronology": "Preserve day/file boundaries; exact chronological order available from filenames.",
            "proportions": "Target 70/15/15 by files/rows, adjusted for binary class coverage.",
            "class_coverage": "All selected evaluation splits must include BENIGN and ATTACK; attack families remain campaign/file concentrated.",
            "duplicate_contamination": "Requires row-hash checks after split; direct random-row splitting avoided.",
            "limitation": "Some attack families occur in only one file, so family coverage cannot be guaranteed in every split.",
            "files": [
                {"file": Path(f["path"]).name, "rows": f["row_count"], "labels": f.get("label_distribution_sample", {}).get("Label", {})}
                for f in datasets["CICIDS2017"]["files"]
            ],
        },
        "Edge-IIoTset": {
            "split_unit": "rows ordered by frame.time within prepared ML CSV when timestamp parsing succeeds",
            "grouping_key": "source_file plus optional source/destination host metadata for leakage checks",
            "chronology": "Single selected CSV means source-file holdout is impossible; use time-aware split if frame.time is usable, otherwise stratified row split with leakage warning.",
            "proportions": "70/15/15 target.",
            "class_coverage": datasets["Edge-IIoTset"].get("label_distribution", {}),
            "duplicate_contamination": "Payload/URI leakage fields excluded; duplicate row-hash checks required in materialized splits.",
            "limitation": "Prepared ML CSV may already be shuffled/processed; source PCAPs are not re-extracted in P1.",
        },
        "BoT-IoT": {
            "split_unit": "CSV partition and stime ordering",
            "grouping_key": "source_file",
            "chronology": "Use stime for chronological validation where feasible; never load all 74 partitions at once.",
            "proportions": "70/15/15 target by streamed rows/partitions.",
            "class_coverage": datasets["BoT-IoT"].get("label_distribution", {}),
            "duplicate_contamination": "Partition-aware row-hash checks required; pkSeqID excluded as leakage-risk sequence identifier.",
            "limitation": "Normal traffic is extremely rare (~9.5K of 73.37M rows), so leakage-resistant splits may have weak benign coverage unless explicitly guarded.",
        },
        "N-BaIoT": {
            "split_unit": "device/file group",
            "grouping_key": "device_id",
            "chronology": "No raw timestamp/order field in traffic CSVs; use device/file grouped evaluation.",
            "proportions": "Device-aware split target, not random row split.",
            "class_coverage": datasets["N-BaIoT"].get("label_distribution", {}),
            "duplicate_contamination": "Device identity excluded as direct input; duplicate row-hash checks required within materialized partitions.",
            "limitation": "Some devices lack Mirai files; grouped split must verify binary coverage before use.",
        },
    }


def compatibility_matrix() -> dict[str, object]:
    return read_json(ROOT / "configs" / "experiments" / "cross_dataset_compatibility.json")


def run_benchmark() -> dict[str, object]:
    out = ROOT / "data" / "cache" / "milestone2_5_benchmark"
    configs = {
        "CICIDS2017": 20_000,
        "Edge-IIoTset": 157_800,
        "BoT-IoT": 15_000,
        "N-BaIoT": 8_000,
    }
    manifests = []
    for dataset, rows_per_file in configs.items():
        manifests.append(
            materialize_dataset(
                dataset,
                ROOT / "data" / "raw",
                out,
                chunksize=50_000,
                debug_rows_per_file=rows_per_file,
                force=True,
            )
        )
    cache_bytes = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    total_rows = sum(int(m["rows"]) for m in manifests)
    return {
        "mode": "bounded_production_path",
        "note": "Uses production materialization code path with bounded rows per file; not full-data materialization and not model performance.",
        "cache_root": str(out),
        "cache_bytes": cache_bytes,
        "total_rows": total_rows,
        "datasets": manifests,
    }


def validate_cache_restart() -> dict[str, object]:
    out = ROOT / "data" / "cache" / "milestone2_5_cache_validation"
    first = materialize_dataset("Edge-IIoTset", ROOT / "data" / "raw", out, chunksize=10_000, debug_rows_per_file=1_000, force=True)
    second = materialize_dataset("Edge-IIoTset", ROOT / "data" / "raw", out, chunksize=10_000, debug_rows_per_file=1_000)
    changed = materialize_dataset(
        "Edge-IIoTset",
        ROOT / "data" / "raw",
        out,
        chunksize=10_000,
        debug_rows_per_file=1_000,
        feature_version="canonical-features-v2-test",
    )
    incomplete_dir = out / "incomplete-case"
    incomplete_dir.mkdir(parents=True, exist_ok=True)
    manifest = incomplete_dir / "manifest.json"
    manifest.write_text(json.dumps({"complete": False, "rows": 10, "partitions": 1}), encoding="utf-8")
    incomplete_detected = load_valid_manifest(manifest, incomplete_dir) is None
    shutil.rmtree(incomplete_dir)
    return {
        "first_status": first["cache_status"],
        "second_status": second["cache_status"],
        "same_fingerprint_reused": first["fingerprint"] == second["fingerprint"] and second["cache_status"] == "reused",
        "changed_feature_version_invalidated": changed["fingerprint"] != first["fingerprint"],
        "incomplete_cache_rejected": incomplete_detected,
        "raw_data_untouched_policy": "Materializer writes only under data/cache output_root and reads data/raw.",
    }


def feature_classification(schema: dict) -> list[dict[str, object]]:
    return [
        {
            "name": f["name"],
            "level": f["level"],
            "source_datasets": [dataset for dataset, sources in f["availability"].items() if sources],
            "raw_or_derived": "derived" if f["name"] in {"traffic_asymmetry", "packet_direction_ratio"} else "raw/source-provided",
            "definition": f["semantic_definition"],
            "causality_leakage": f"{f['leakage_role']}; {f['missing_behavior']}",
        }
        for f in schema["features"]
    ]


def main() -> None:
    audit = read_json(ROOT / "reports" / "eda" / "dataset_audit.json")
    schema = read_json(ROOT / "configs" / "features" / "canonical_schema.json")
    out_dir = ROOT / "reports" / "eda"
    write_json(out_dir / "semantic_profile_audit.json", semantic_candidates())
    write_json(out_dir / "nbaiot_compatibility.json", compatibility_matrix())
    write_json(out_dir / "feature_level_audit.json", feature_classification(schema))
    write_json(out_dir / "split_sanity.json", split_sanity(audit))
    benchmark = run_benchmark()
    write_json(out_dir / "materialization_benchmark.json", benchmark)
    write_json(out_dir / "cache_restart_validation.json", validate_cache_restart())
    rows = semantic_candidates()
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]).replace("|", "/") for h in headers) + " |")
    (out_dir / "semantic_profile_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"benchmark_rows": benchmark["total_rows"], "benchmark_cache_bytes": benchmark["cache_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
