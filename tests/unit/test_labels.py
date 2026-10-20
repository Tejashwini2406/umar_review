from __future__ import annotations

from iot_ids.data.labels import nbaiot_label_from_filename, normalize_binary_label


def test_binary_label_mapping() -> None:
    assert normalize_binary_label("CICIDS2017", "BENIGN").binary_label == "BENIGN"
    assert normalize_binary_label("CICIDS2017", "DoS Hulk").binary_label == "ATTACK"
    assert normalize_binary_label("Edge-IIoTset", "0").binary_label == "BENIGN"
    assert normalize_binary_label("BoT-IoT", "1").binary_label == "ATTACK"


def test_nbaiot_filename_labels() -> None:
    assert nbaiot_label_from_filename("1.benign.csv") == "benign"
    assert nbaiot_label_from_filename("2.gafgyt.combo.csv") == "gafgyt.combo"
    record = normalize_binary_label("N-BaIoT", "", source_path="9.mirai.udpplain.csv")
    assert record.binary_label == "ATTACK"
    assert record.attack_family == "mirai"

