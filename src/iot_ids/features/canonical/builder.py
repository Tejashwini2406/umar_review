"""[DEPRECATED] Legacy Feature Builder Module.

DEPRECATION NOTICE: This module is retained for backwards compatibility with pre-Stage 1 exploratory scripts.
For production multi-level 18-feature extraction, use `iot_ids.features.canonical.flow_builder.CanonicalFlowBuilder`.
"""
from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from iot_ids.data.labels import nbaiot_label_from_filename

warnings.warn(
    "iot_ids.features.canonical.builder is deprecated. Use CanonicalFlowBuilder in iot_ids.features.canonical.flow_builder.",
    DeprecationWarning,
    stacklevel=2,
)


def _get_series(frame: pd.DataFrame, col: str, default: float | str = 0.0) -> pd.Series:
    if col in frame.columns:
        return frame[col]
    return pd.Series(default, index=frame.index)


def safe_num(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def build_canonical_features(dataset: str, frame: pd.DataFrame, source_path: str | Path | None = None) -> pd.DataFrame:
    dataset_key = dataset.lower().replace("_", "-")
    if dataset_key == "edge-iiotset":
        out = _edge_features(frame)
    elif dataset_key in ("ton-iot", "ton_iot"):
        out = _ton_features(frame)
    elif dataset_key == "cicids2017":
        out = _cicids_features(frame)
    elif dataset_key == "bot-iot":
        out = _bot_features(frame)
    elif dataset_key == "n-baiot":
        if source_path is None:
            raise ValueError("N-BaIoT canonicalization requires source_path for filename labels.")
        out = _nbaiot_features(frame, source_path)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if source_path is not None:
        out["source_file"] = str(source_path)
    return out


def _edge_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)

    # 1. F_common (8 Features)
    out["duration"] = safe_num(_get_series(frame, "udp.time_delta", 0.0))
    out["src_bytes"] = safe_num(_get_series(frame, "tcp.len", 0.0))
    out["src_pkts"] = 1.0  # 1 frame per row
    out["dst_pkts"] = (safe_num(_get_series(frame, "tcp.flags.ack", 0.0)) > 0).astype(float)

    srcport = safe_num(_get_series(frame, "tcp.srcport", 0.0))
    dstport = safe_num(_get_series(frame, "tcp.dstport", 0.0))
    udpport = safe_num(_get_series(frame, "udp.port", 0.0))
    icmpcksum = safe_num(_get_series(frame, "icmp.checksum", 0.0))

    out["proto_tcp"] = ((srcport > 0) | (dstport > 0)).astype(float)
    out["proto_udp"] = (udpport > 0).astype(float)
    out["proto_icmp"] = (icmpcksum > 0).astype(float)
    out["is_well_known_port"] = ((dstport < 1024) & (dstport > 0)).astype(float)

    # 2. In-Domain Specifics (Edge)
    out["mqtt_msgtype"] = safe_num(_get_series(frame, "mqtt.msgtype", 0.0))
    out["mbtcp_unit_id"] = safe_num(_get_series(frame, "mbtcp.unit_id", 0.0))

    # 3. Metadata / State Keys
    out["source_host"] = _get_series(frame, "ip.src_host", "0.0.0.0").astype(str)
    out["destination_host"] = _get_series(frame, "ip.dst_host", "0.0.0.0").astype(str)
    out["timestamp"] = _get_series(frame, "frame.time", "").astype(str)

    # 4. Target Labels
    raw_label = safe_num(_get_series(frame, "Attack_label", 0))
    out["label"] = (raw_label > 0).astype(int)
    out["attack_category"] = _get_series(frame, "Attack_type", "Normal").astype(str)

    return out


def _ton_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)

    # 1. F_common (8 Features)
    out["duration"] = safe_num(_get_series(frame, "duration", 0.0))
    out["src_bytes"] = safe_num(_get_series(frame, "src_bytes", 0.0))
    out["src_pkts"] = safe_num(_get_series(frame, "src_pkts", 0.0))
    out["dst_pkts"] = safe_num(_get_series(frame, "dst_pkts", 0.0))

    proto = _get_series(frame, "proto", "").astype(str).str.lower()
    out["proto_tcp"] = (proto == "tcp").astype(float)
    out["proto_udp"] = (proto == "udp").astype(float)
    out["proto_icmp"] = (proto == "icmp").astype(float)

    dst_port = safe_num(_get_series(frame, "dst_port", 0.0))
    out["is_well_known_port"] = ((dst_port < 1024) & (dst_port > 0)).astype(float)

    # 2. In-Domain Specifics (ToN)
    conn_state = _get_series(frame, "conn_state", "OTH").astype(str)
    out["conn_state_encoded"] = (conn_state == "SF").astype(float)
    http_method = _get_series(frame, "http_method", "-").astype(str)
    out["http_method_encoded"] = (http_method != "-").astype(float)

    # 3. Metadata / State Keys
    out["source_host"] = _get_series(frame, "src_ip", "0.0.0.0").astype(str)
    out["destination_host"] = _get_series(frame, "dst_ip", "0.0.0.0").astype(str)
    out["timestamp"] = frame.index.astype(str)

    # 4. Target Labels
    raw_label = safe_num(_get_series(frame, "label", 0))
    out["label"] = (raw_label > 0).astype(int)
    out["attack_category"] = _get_series(frame, "type", "normal").astype(str)

    return out


# Legacy functions preserved for historical reproducibility
def _cicids_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["duration"] = safe_num(_get_series(frame, "Flow Duration", 0)) / 1e6
    out["label"] = np.where(_get_series(frame, "Label", "").astype(str).str.upper().eq("BENIGN"), 0, 1)
    out["attack_category"] = _get_series(frame, "Label", "").astype(str)
    return out


def _bot_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["duration"] = safe_num(_get_series(frame, "dur", 0))
    out["label"] = np.where(safe_num(_get_series(frame, "attack", 0)).eq(0), 0, 1)
    out["attack_category"] = _get_series(frame, "category", "").astype(str)
    return out


def _nbaiot_features(frame: pd.DataFrame, source_path: str | Path) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    label = nbaiot_label_from_filename(Path(source_path).name)
    out["label"] = 0 if label == "benign" else 1
    out["attack_category"] = label
    return out
