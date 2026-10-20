from __future__ import annotations

from pathlib import Path

import pytest

from iot_ids.data.adapters import BoTIoTAdapter, build_default_adapters
from iot_ids.data.flow_object import CanonicalFlow


def _drain_with_report(stream):
    flows = []
    while True:
        try:
            flows.append(next(stream))
        except StopIteration as stop:
            return flows, stop.value


def test_bot_iot_discovers_csv_and_excludes_metadata(tmp_path: Path):
    source_dir = tmp_path / "BoT-IoT"
    (source_dir / "partitions").mkdir(parents=True)
    (source_dir / "partitions" / "part-01.csv").write_text("attack\n0\n")
    (source_dir / "data_names.csv").write_text("name\n")
    (source_dir / "notes.txt").write_text("metadata")

    adapter = BoTIoTAdapter(source_dir)

    assert adapter.get_source_files() == [source_dir / "partitions" / "part-01.csv"]
    assert adapter.discover_files() == adapter.get_source_files()
    assert adapter.audit().status == "AVAILABLE"


def test_bot_iot_streams_canonical_flows_and_reports_invalid_rows(tmp_path: Path):
    source = tmp_path / "flows.csv"
    source.write_text(
        "stime,dur,proto,sport,dport,saddr,daddr,spkts,dpkts,sbytes,dbytes,"
        "attack,category,flgs,state\n"
        "1700000000,0.5,tcp,1234,80,10.0.0.1,10.0.0.2,3,2,120,240,"
        "1,DoS,e s a r,RST\n"
        "1700000001,0.0,udp,53,53000,10.0.0.2,10.0.0.1,1,1,20,40,"
        "0,Normal,e,CON\n"
        "1700000002,0.2,tcp,1234,80,,10.0.0.2,1,1,20,40,"
        "1,DoS,e s,CON\n",
        encoding="utf-8",
    )
    adapter = BoTIoTAdapter(source)

    flows, report = _drain_with_report(adapter.stream_canonical_flows(chunksize=2))

    assert len(flows) == 2
    assert all(isinstance(flow, CanonicalFlow) for flow in flows)
    assert flows[0].timestamp_start == 1_700_000_000
    assert flows[0].timestamp_end == 1_700_000_000.5
    assert (flows[0].src_host, flows[0].dst_host) == ("10.0.0.1", "10.0.0.2")
    assert (flows[0].bytes_src, flows[0].bytes_dst) == (120, 240)
    assert (flows[0].syn_count_src, flows[0].ack_count_src, flows[0].rst_count_src) == (
        1,
        1,
        1,
    )
    assert flows[0].conn_state_code == 2
    assert (flows[0].label, flows[0].attack_category) == (1, "DoS")
    assert (flows[1].protocol, flows[1].label, flows[1].attack_category) == (
        "udp",
        0,
        "Normal",
    )
    assert report.raw_records_read == 3
    assert report.canonical_flows_emitted == 2
    assert report.invalid_records_dropped == 1
    assert len(report.warnings) == 1


def test_bot_iot_rejects_missing_required_columns(tmp_path: Path):
    source = tmp_path / "incomplete.csv"
    source.write_text("stime,dur,attack\n1700000000,0.1,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        next(BoTIoTAdapter(source).stream_canonical_flows())


def test_bot_iot_rejects_out_of_order_records(tmp_path: Path):
    source = tmp_path / "out-of-order.csv"
    source.write_text(
        "stime,dur,proto,sport,dport,saddr,daddr,spkts,dpkts,sbytes,dbytes,"
        "attack,category,flgs,state\n"
        "1700000001,0.1,tcp,1,80,10.0.0.1,10.0.0.2,1,1,10,20,0,Normal,e,CON\n"
        "1700000000,0.1,tcp,1,80,10.0.0.1,10.0.0.2,1,1,10,20,0,Normal,e,CON\n",
        encoding="utf-8",
    )
    stream = BoTIoTAdapter(source).stream_canonical_flows()

    next(stream)
    with pytest.raises(ValueError, match="not in chronological order"):
        next(stream)


def test_bot_iot_is_in_default_adapter_registry(tmp_path: Path):
    adapters = build_default_adapters(tmp_path)

    assert any(isinstance(adapter, BoTIoTAdapter) for adapter in adapters)
