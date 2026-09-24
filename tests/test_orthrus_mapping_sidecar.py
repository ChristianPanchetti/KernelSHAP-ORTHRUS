from __future__ import annotations

import json
from pathlib import Path

from preprocessing.orthrus_alert_case import OrthrusAlertCase
from preprocessing.orthrus_mapping import SidecarMappingProvider, DbAssistedMappingProvider, describe_component
from preprocessing.orthrus_join_keys import extract_edge_join_keys
from perturbation.orthrus_interpretable_builder import OrthrusInterpretableBuilder


class FakeTemporalData:
    def __init__(self, src, dst, t, edge_type=None):
        self.src = list(src)
        self.dst = list(dst)
        self.t = list(t)
        if edge_type is not None:
            self.edge_type = list(edge_type)


def test_sidecar_mapping_provider_enriches_case_and_reports(tmp_path: Path):
    td = FakeTemporalData(src=[1, 2], dst=[2, 3], t=[100, 200], edge_type=[0, 1])
    case = OrthrusAlertCase(temporal_data=td)

    sidecar = {
        "mapping_mode": "sidecar-assisted",
        "join_strategy": "edge_index",
        "edges": {
            "0": {"event_uuid": "uuid-0", "src_index_id": 1, "dst_index_id": 2, "timestamp_rec": 100},
            "1": {"event_uuid": "uuid-1", "src_index_id": 2, "dst_index_id": 3, "timestamp_rec": 200},
        },
    }

    p = tmp_path / "sidecar.json"
    p.write_text(json.dumps(sidecar))

    provider = SidecarMappingProvider(str(p))
    provider.enrich_case(case)

    assert case.mapping_mode == "sidecar-assisted"
    assert case.edge_to_event_uuid[0] == "uuid-0"
    assert case.edge_to_event_uuid[1] == "uuid-1"
    assert case.mapping_quality["matched"] == 2


def test_sidecar_incomplete_and_extra_edges(tmp_path: Path):
    td = FakeTemporalData(src=[1, 2, 3], dst=[2, 3, 4], t=[100, 200, 300], edge_type=[0, 1, 2])
    case = OrthrusAlertCase(temporal_data=td)

    sidecar = {
        "mapping_mode": "sidecar-assisted",
        "join_strategy": "edge_index",
        "edges": {
            "0": {"event_uuid": "uuid-0", "src_index_id": 1, "dst_index_id": 2, "timestamp_rec": 100},
            "5": {"event_uuid": "uuid-5", "src_index_id": 9, "dst_index_id": 10, "timestamp_rec": 999},
        },
    }

    p = tmp_path / "sidecar.json"
    p.write_text(json.dumps(sidecar))

    provider = SidecarMappingProvider(str(p))
    provider.enrich_case(case)

    assert case.mapping_mode == "sidecar-assisted"
    # only edge 0 matched
    assert case.mapping_quality["matched"] == 1
    assert case.mapping_quality["unmatched"] == 2
    assert 5 in case.mapping_quality["extra_sidecar_edges"]


def test_db_assisted_match_edges_to_rows_synthetic():
    td = FakeTemporalData(src=[1, 2], dst=[2, 3], t=[100, 200], edge_type=["EVENT_EXECUTE", "EVENT_READ"])
    case = OrthrusAlertCase(temporal_data=td)

    provider = DbAssistedMappingProvider(db_config=None)

    edge_keys = extract_edge_join_keys(case)
    rows = [
        {"src_index_id": 1, "dst_index_id": 2, "timestamp_rec": 100, "operation": "EVENT_EXECUTE", "event_uuid": "uuid-0"},
        {"src_index_id": 2, "dst_index_id": 3, "timestamp_rec": 200, "operation": "EVENT_READ", "event_uuid": "uuid-1"},
    ]

    mapping = provider._match_edges_to_rows(edge_keys, rows)
    # mapping should contain both edges
    assert 0 in mapping and 1 in mapping
    assert mapping[0][0]["event_uuid"] == "uuid-0"


def test_describe_component_sidecar(tmp_path: Path):
    td = FakeTemporalData(src=[1, 1, 2], dst=[2, 3, 3], t=[100, 120, 180], edge_type=[0, 1, 1])
    case = OrthrusAlertCase(temporal_data=td)

    sidecar = {
        "mapping_mode": "sidecar-assisted",
        "edges": {
            "1": {"event_uuid": "uuid-1", "raw_metadata": {"cmd": "ls"}},
        },
    }
    p = tmp_path / "sidecar.json"
    p.write_text(json.dumps(sidecar))

    provider = SidecarMappingProvider(str(p))
    provider.enrich_case(case)

    OrthrusInterpretableBuilder(grouping_mode="edge_type").build(case)
    text = describe_component(case, "edge_type:1", max_edges=2)
    assert "sidecar-assisted" in case.mapping_quality["mode"]
    assert "uuid-1" in str(case.edge_to_event_uuid.values())
    assert "component=edge_type:1" in text
