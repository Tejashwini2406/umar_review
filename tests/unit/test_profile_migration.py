import pytest
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.iot_ids.preprocessing.pipeline import get_model_feature_columns

def test_cross_domain_profile_preservation():
    """Verify F_common (6 features) is correctly returned for active datasets."""
    for ds in ["Edge-IIoTset", "ToN-IoT"]:
        cols = get_model_feature_columns(ds, profile="cross_domain")
        assert len(cols) == 6
        assert "duration" in cols
        assert "src_bytes" in cols
        assert "proto_tcp" in cols
        assert "is_well_known_port" in cols
        assert "dst_bytes" not in cols

def test_in_domain_profile_richness():
    """Verify in-domain profiles contain the multi-level features."""
    edge = get_model_feature_columns("Edge-IIoTset", profile="in_domain")
    assert len(edge) >= 13
    assert "temporal_causal_count" in edge
    assert "behavioral_dest_diversity" in edge
    assert "mqtt_msgtype" in edge

    ton = get_model_feature_columns("ToN-IoT", profile="in_domain")
    assert len(ton) >= 13
    assert "temporal_causal_count" in ton
    assert "behavioral_dest_diversity" in ton
    assert "conn_state_encoded" in ton

def test_invalid_profile_rejection():
    """The resolver must strictly reject typos and unknown profiles."""
    with pytest.raises(ValueError, match="Invalid profile requested"):
        get_model_feature_columns("Edge-IIoTset", profile="invalid_profile_name")
