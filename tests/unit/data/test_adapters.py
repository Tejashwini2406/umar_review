"""Unit tests for DatasetAdapters."""

from pathlib import Path
import pytest

from iot_ids.data.adapters.ton_iot import ToNIoTAdapter
from iot_ids.data.adapters.edge_iiotset import EdgeIIoTsetAggregatorAdapter
from iot_ids.data.adapters.nf_ton_iot import NFToNIoTAdapter
from iot_ids.data.adapters.ciciot2023 import CICIoT2023Adapter
from iot_ids.data.flow_object import CanonicalFlow


def test_adapters_instantiation():
    raw_path = Path("data/raw")

    ton = ToNIoTAdapter(raw_path / "ToN-IoT")
    edge = EdgeIIoTsetAggregatorAdapter(raw_path / "Edge-IIoTset")
    nfton = NFToNIoTAdapter(raw_path / "NF-ToN-IoT-v2")
    ciciot = CICIoT2023Adapter(raw_path / "CICIOT23")

    assert ton.name == "ToN-IoT"
    assert edge.name == "Edge-IIoTset"
    assert nfton.name == "NF-ToN-IoT-v2"
    assert ciciot.name == "CICIoT2023"


def test_ton_adapter_streaming_sample():
    raw_path = Path("data/raw/ToN-IoT")
    if not (raw_path / "train_test_network.csv").exists():
        pytest.skip("ToN-IoT dataset file not found locally")

    adapter = ToNIoTAdapter(raw_path)
    stream = adapter.stream_canonical_flows(chunksize=100)
    flows = [next(stream) for _ in range(10)]

    assert len(flows) == 10
    for f in flows:
        assert isinstance(f, CanonicalFlow)
        assert f.duration >= 0.0
        assert f.label in (0, 1)
