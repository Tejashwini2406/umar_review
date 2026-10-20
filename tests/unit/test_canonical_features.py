from __future__ import annotations

import pandas as pd
from iot_ids.features.canonical.builder import build_canonical_features


def test_edge_canonical_features() -> None:
    frame = pd.DataFrame(
        {
            "udp.time_delta": [0.5],
            "tcp.len": [100],
            "tcp.flags.ack": [1],
            "tcp.srcport": [4444],
            "tcp.dstport": [80],
            "Attack_label": [0],
            "Attack_type": ["Normal"],
            "ip.src_host": ["192.168.0.10"],
            "ip.dst_host": ["192.168.0.1"],
            "frame.time": ["2021 11:44:10"],
        }
    )
    out = build_canonical_features("Edge-IIoTset", frame)
    assert out.loc[0, "duration"] == 0.5
    assert out.loc[0, "src_bytes"] == 100
    assert out.loc[0, "is_well_known_port"] == 1.0
    assert out.loc[0, "label"] == 0


def test_ton_canonical_features() -> None:
    frame = pd.DataFrame(
        {
            "duration": [2.5],
            "src_bytes": [1000],
            "src_pkts": [10],
            "dst_pkts": [8],
            "proto": ["tcp"],
            "dst_port": [443],
            "label": [1],
            "type": ["backdoor"],
            "src_ip": ["192.168.1.37"],
            "dst_ip": ["192.168.1.193"],
        }
    )
    out = build_canonical_features("ToN-IoT", frame)
    assert out.loc[0, "duration"] == 2.5
    assert out.loc[0, "src_bytes"] == 1000
    assert out.loc[0, "proto_tcp"] == 1.0
    assert out.loc[0, "is_well_known_port"] == 1.0
    assert out.loc[0, "label"] == 1
