from __future__ import annotations

import json
from pathlib import Path

from preprocessing.orthrus_alert_case import OrthrusAlertCase
from preprocessing.orthrus_mapping import SidecarMappingProvider
from preprocessing.orthrus_sidecar_generator import OrthrusSidecarGenerator


class FakeTemporalData:
    def __init__(self, src, dst, t, operation=None):
        self.src = list(src)
        self.dst = list(dst)
        self.t = list(t)
        if operation is not None:
            self.operation = list(operation)


def test_generate_sidecar_from_temporal_data_and_rows_and_load(tmp_path: Path):
    td_payload = {
        "src": [1, 2],
        "dst": [2, 3],
        "t": [100, 200],
        "operation": ["EVENT_EXECUTE", "EVENT_READ"],
    }
    td_path = tmp_path / "temporal_data.json"
    td_path.write_text(json.dumps(td_payload), encoding="utf-8")

    rows = [
        {
            "src_index_id": 1,
            "dst_index_id": 2,
            "timestamp_rec": 100,
            "operation": "EVENT_EXECUTE",
            "event_uuid": "uuid-0",
            "raw_metadata": {"cmd": "bash"},
        },
        {
            "src_index_id": 2,
            "dst_index_id": 3,
            "timestamp_rec": 200,
            "operation": "EVENT_READ",
            "event_uuid": "uuid-1",
            "raw_metadata": {"file": "/etc/passwd"},
        },
    ]

    sidecar_path = tmp_path / "sidecar.json"

    gen = OrthrusSidecarGenerator()
    gen.generate_from_temporal_data_and_rows(str(td_path), rows, str(sidecar_path))

    # Basic structure
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["mapping_mode"] == "sidecar-assisted"
    assert sidecar["join_strategy"] == "src_dst_t_operation"
    assert sidecar["num_edges"] == 2
    assert sidecar["matched"] == 2
    assert sidecar["unmatched"] == 0
    assert set(sidecar["edges"].keys()) == {"0", "1"}

    # Validate against the temporal_data JSON
    report = gen.validate_sidecar(str(sidecar_path), temporal_data_path=str(td_path))
    assert report["missing_edge_keys"] == []
    assert report["extra_edge_keys"] == []

    # Load via SidecarMappingProvider
    case = OrthrusAlertCase(temporal_data=FakeTemporalData(**td_payload))
    provider = SidecarMappingProvider(str(sidecar_path))
    provider.enrich_case(case)

    assert case.edge_to_event_uuid[0] == "uuid-0"
    assert case.edge_to_event_uuid[1] == "uuid-1"
    assert case.mapping_quality["matched"] == 2


def test_validate_sidecar_missing_and_extra_edges(tmp_path: Path):
    td_payload = {"src": [1, 2, 3], "dst": [2, 3, 4], "t": [100, 200, 300]}
    td_path = tmp_path / "temporal_data.json"
    td_path.write_text(json.dumps(td_payload), encoding="utf-8")

    sidecar = {
        "mapping_mode": "sidecar-assisted",
        "join_strategy": "edge_index",
        "num_edges": 3,
        "matched": 1,
        "unmatched": 2,
        "warnings": [],
        "edges": {
            "0": {"event_uuid": "uuid-0", "src_index_id": 1, "dst_index_id": 2, "timestamp_rec": 100, "operation": "EVENT_EXECUTE", "raw_metadata": {}},
            "5": {"event_uuid": "uuid-5", "src_index_id": 9, "dst_index_id": 10, "timestamp_rec": 999, "operation": "EVENT_OPEN", "raw_metadata": {}},
        },
    }
    sidecar_path = tmp_path / "sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    gen = OrthrusSidecarGenerator()
    report = gen.validate_sidecar(str(sidecar_path), temporal_data_path=str(td_path))

    assert report["missing_edge_keys"] == [1, 2]
    assert report["extra_edge_keys"] == [5]


def test_generate_sidecar_fallback_join_without_operation(tmp_path: Path):
    # TemporalData without operation -> generator must fallback to (src,dst,t)
    td_payload = {"src": [1, 2], "dst": [2, 3], "t": [100, 200]}
    td_path = tmp_path / "temporal_data.json"
    td_path.write_text(json.dumps(td_payload), encoding="utf-8")

    rows = [
        {"src_index_id": 1, "dst_index_id": 2, "timestamp_rec": 100, "operation": "EVENT_EXECUTE", "event_uuid": "uuid-0"},
        {"src_index_id": 2, "dst_index_id": 3, "timestamp_rec": 200, "operation": "EVENT_READ", "event_uuid": "uuid-1"},
    ]

    sidecar_path = tmp_path / "sidecar.json"

    gen = OrthrusSidecarGenerator(prefer_operation=True)
    sidecar = gen.generate_from_temporal_data_and_rows(str(td_path), rows, str(sidecar_path))

    assert sidecar["join_strategy"] == "src_dst_t"
    assert sidecar["matched"] == 2
    assert any("Operation not available" in w for w in sidecar["warnings"])
