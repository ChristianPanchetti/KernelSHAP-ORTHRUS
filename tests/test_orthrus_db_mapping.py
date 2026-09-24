from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from perturbation.orthrus_interpretable_builder import OrthrusInterpretableBuilder
from perturbation.orthrus_perturbation_manager import OrthrusPerturbationManager
from preprocessing.orthrus_alert_case import OrthrusAlertCase
from preprocessing.orthrus_join_keys import decode_operation_from_temporal_data
from preprocessing.orthrus_mapping import DbAssistedMappingProvider, SidecarMappingProvider, describe_component
from preprocessing.orthrus_sidecar_generator import OrthrusSidecarGenerator


RELATIONS = {1: "EVENT_READ", "EVENT_READ": 1, 2: "EVENT_WRITE", "EVENT_WRITE": 2}


def make_case():
    td = SimpleNamespace(src=[1, 1, 3], dst=[2, 4, 2], t=[100, 101, 102],
                         edge_type=["EVENT_READ", "EVENT_WRITE", "EVENT_READ"],
                         x_src=[[1.0], [1.0], [3.0]], x_dst=[[2.0], [4.0], [2.0]])
    case = OrthrusAlertCase(td, full_data=object(), metadata={"dataset": "THEIA_E5"})
    OrthrusInterpretableBuilder(grouping_mode="node").build(case)
    return case


def event(src=1, dst=2, t=100, op="EVENT_READ", uuid="event-1", row_id=1):
    return dict(src_index_id=src, dst_index_id=dst, timestamp_rec=t,
                operation=op, event_uuid=uuid, _id=row_id)


def test_nodes_unique_ambiguous_missing_and_partial_metadata_without_mutating_inputs():
    case = make_case()
    before = deepcopy(vars(case.temporal_data))
    full_data = case.full_data
    rows = [{"index_id": 1, "node_uuid": "process", "node_type": "subject", "path": "/bin/sh", "cmd": None},
            {"index_id": 2, "node_uuid": "file", "node_type": "file", "path": "/a"},
            {"index_id": 2, "node_uuid": "other", "node_type": "subject", "path": "/b"}]
    DbAssistedMappingProvider().enrich_case(case, rows=[], node_rows=rows, component_ids=["node:1"])
    nodes = case.metadata["node_mapping"]
    assert set(nodes) == {1, 2, 4}  # C->B is affected, but C is not a component endpoint.
    assert nodes[1]["status"] == "resolved" and nodes[1]["missing_fields"] == ["cmd"]
    assert nodes[2]["status"] == "ambiguous" and "node_uuid" not in nodes[2]
    assert nodes[4]["status"] == "missing"
    assert vars(case.temporal_data) == before and case.full_data is full_data


def test_network_values_are_withheld_until_columns_are_explicitly_verified():
    rows = [{"index_id": 2, "node_uuid": "network", "node_type": "netflow",
             "src_addr": "127.0.0.1", "src_port": "80", "dst_addr": "broken", "dst_port": "x"}]
    case = make_case()
    DbAssistedMappingProvider().enrich_case(case, rows=[], node_rows=rows)
    assert "src_addr" not in case.metadata["node_mapping"][2]
    assert "broken" not in describe_component(case, "node:1")
    DbAssistedMappingProvider(verified_network_fields=["src_addr", "src_port"]).enrich_case(case, rows=[], node_rows=rows)
    node = case.metadata["node_mapping"][2]
    assert node["src_addr"] == "127.0.0.1" and "dst_addr" not in node


def test_onehot_decoding_uses_supplied_one_based_vocabulary_and_rejects_invalid_values():
    td = SimpleNamespace(edge_type=np.array([[1, 0], [0, 1], [0, 0], [1, 1]]))
    assert decode_operation_from_temporal_data(td, RELATIONS) == ["EVENT_READ", "EVENT_WRITE", None, None]
    assert decode_operation_from_temporal_data(td) == [None] * 4
    assert decode_operation_from_temporal_data(SimpleNamespace(edge_type=[0, 1, 2]), RELATIONS) == [None, "EVENT_READ", "EVENT_WRITE"]
    assert decode_operation_from_temporal_data(SimpleNamespace(msg=[[99, 0, 1, 99]]), RELATIONS, (1, 3)) == ["EVENT_WRITE"]


def test_torch_onehot_decoding_when_torch_is_available():
    torch = pytest.importorskip("torch")
    td = SimpleNamespace(edge_type=torch.tensor([[0., 1.], [1., 0.]]))
    assert decode_operation_from_temporal_data(td, RELATIONS) == ["EVENT_WRITE", "EVENT_READ"]


def test_events_resolved_ambiguous_missing_and_no_arbitrary_uuid():
    case = make_case()
    rows = [event(), event(1, 4, 101, "EVENT_WRITE", "candidate-a", 2),
            event(1, 4, 101, "EVENT_WRITE", "candidate-b", 3)]
    DbAssistedMappingProvider().enrich_case(case, rows=iter(rows))
    assert case.edge_to_event_uuid == {0: "event-1"}
    assert [case.edge_to_original_metadata[i]["status"] for i in range(3)] == ["resolved", "ambiguous", "missing"]
    assert case.mapping_quality["events"] == {"resolved": 1, "ambiguous": 1, "missing": 1}


def test_unknown_operation_and_missing_timestamp_never_match_incomplete_keys():
    case = make_case()
    case.temporal_data.edge_type = [0, 0, 0]
    case.temporal_data.t[1] = None
    case.edge_to_event_uuid = {0: "unverified-old-uuid"}
    DbAssistedMappingProvider().enrich_case(case, rows=[event(op="0")])
    assert case.edge_to_event_uuid == {}
    assert case.mapping_quality["events"] == {"unresolvable": 3}


def test_a_previous_database_join_does_not_hide_new_ambiguous_candidates():
    case = make_case()
    provider = DbAssistedMappingProvider()
    provider.enrich_case(case, rows=[event()])
    assert case.edge_to_event_uuid == {0: "event-1"}
    provider.enrich_case(case, rows=[event(), event(uuid="another", row_id=2)])
    assert case.edge_to_event_uuid == {}
    assert case.edge_to_original_metadata[0]["status"] == "ambiguous"


def test_read_direction_is_already_reversed_in_database_and_is_not_reversed_again():
    case = make_case()
    DbAssistedMappingProvider().enrich_case(case, rows=[event(2, 1)])
    assert case.edge_to_event_uuid == {}
    DbAssistedMappingProvider().enrich_case(case, rows=[event(1, 2)])
    assert case.edge_to_event_uuid == {0: "event-1"}


def test_source_component_and_other_describe_actual_repeated_role_policy():
    case = make_case()
    DbAssistedMappingProvider().enrich_case(case, rows=[], node_rows=[])
    roles = case.metadata["component_mapping"]["node:1"]
    assert roles["assigned_edges"] == [0, 1]
    assert roles["source_rows_if_disabled"] == [0, 1]
    assert roles["destination_rows_if_disabled"] == [0, 1, 2]
    perturbed = OrthrusPerturbationManager(case, mode="neutralize_edges").apply_mask([0, 1]).dataset
    assert perturbed.metadata["neutralized_source_rows"] == roles["source_rows_if_disabled"]
    assert perturbed.metadata["neutralized_destination_rows"] == roles["destination_rows_if_disabled"]
    text = describe_component(perturbed, "node:1")
    assert "edge assegnati=[0, 1]" in text
    assert "Righe effettivamente neutralizzate" in text
    assert "righe x_dst se disattivata=[0, 1, 2]" in text
    case.component_to_edges = {"OTHER": [0, 2], "edge:1": [1]}
    DbAssistedMappingProvider().enrich_case(case, rows=[], component_ids=["OTHER"])
    assert case.metadata["component_mapping"]["OTHER"]["source_node_ids"] == [1, 3]
    assert "nodi source=[1, 3]" in describe_component(case, "OTHER")


def make_sidecar(tmp_path):
    payload = {"src": [1, 3], "dst": [2, 4], "t": [100, 200],
               "operation": ["EVENT_READ", "EVENT_WRITE"]}
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    sidecar = tmp_path / "sidecar.json"
    OrthrusSidecarGenerator().generate_from_temporal_data_and_rows(
        str(artifact), [event(), event(3, 4, 200, "EVENT_WRITE", "event-2", 2)], str(sidecar))
    batch = OrthrusAlertCase(SimpleNamespace(src=[3], dst=[4], t=[200], edge_type=["EVENT_WRITE"]),
                             metadata={"batch_index": 1, "global_edge_offset": 900})
    return artifact, sidecar, batch


def test_verified_sidecar_remaps_graph_index_to_batch_and_db_reuses_uuid(tmp_path):
    artifact, sidecar, batch = make_sidecar(tmp_path)
    SidecarMappingProvider(str(sidecar), temporal_data_path=artifact, graph_edge_offset=1).enrich_case(batch)
    assert batch.edge_to_event_uuid == {0: "event-2"}
    assert batch.edge_to_original_metadata[0]["graph_edge_index"] == 1
    DbAssistedMappingProvider().enrich_case(batch, rows=[event(3, 4, 200, "EVENT_WRITE", "event-2", 2),
                                                       event(3, 4, 200, "EVENT_WRITE", "another", 3)])
    assert batch.edge_to_event_uuid == {0: "event-2"}
    assert batch.edge_to_original_metadata[0]["database_row_id"] == 2
    assert batch.metadata["global_edge_offset"] == 900


@pytest.mark.parametrize("wrong", ["offset", "artifact", "batch"])
def test_sidecar_wrong_provenance_never_assigns_uuid(tmp_path, wrong):
    artifact, sidecar, batch = make_sidecar(tmp_path)
    offset = 0 if wrong == "offset" else 1
    if wrong == "artifact":
        artifact.write_text(artifact.read_text() + " ", encoding="utf-8")
    if wrong == "batch":
        batch.temporal_data.t = [201]
    SidecarMappingProvider(str(sidecar), temporal_data_path=artifact, graph_edge_offset=offset).enrich_case(batch)
    assert batch.edge_to_event_uuid == {}
    assert not batch.mapping_quality["artifact_and_batch_verified"]


def test_legacy_sidecar_and_stale_verified_identity_are_not_reused_by_db(tmp_path):
    artifact, sidecar, batch = make_sidecar(tmp_path)
    SidecarMappingProvider(str(sidecar)).enrich_case(batch)
    DbAssistedMappingProvider().enrich_case(batch, rows=[])
    assert batch.edge_to_event_uuid == {}
    SidecarMappingProvider(str(sidecar), temporal_data_path=artifact, graph_edge_offset=1).enrich_case(batch)
    batch.metadata["batch_index"] = 2
    DbAssistedMappingProvider().enrich_case(batch, rows=[])
    assert batch.edge_to_event_uuid == {}


def test_generator_does_not_choose_first_candidate_or_ignore_conflicting_operation(tmp_path):
    artifact, sidecar, _ = make_sidecar(tmp_path)
    rows = [event(), event(uuid="collision", row_id=3), event(3, 4, 200, "EVENT_READ", "wrong-operation", 4)]
    result = OrthrusSidecarGenerator().generate_from_temporal_data_and_rows(str(artifact), iter(rows), str(sidecar))
    assert result["edges"] == {} and result["matched"] == 0


def test_graph_sidecar_checks_full_order_and_rejects_ambiguous_identical_edges(tmp_path, monkeypatch):
    from preprocessing import orthrus_sidecar_generator as generator_module

    artifact, sidecar, batch = make_sidecar(tmp_path)
    graph_edges = [(1, 2, 0, {"time": 100, "label": "EVENT_READ", "event_uuid": "from-graph-1"}),
                   (3, 4, 0, {"time": 200, "label": "EVENT_WRITE", "event_uuid": "from-graph-2"})]
    graph = SimpleNamespace(edges=lambda **kwargs: graph_edges)
    monkeypatch.setattr(generator_module, "_load_networkx_graph", lambda path: graph)
    generator = OrthrusSidecarGenerator()
    generator.generate_from_graph("trusted-graph", str(sidecar), temporal_data_path=str(artifact))
    SidecarMappingProvider(str(sidecar), temporal_data_path=artifact, graph_edge_offset=1).enrich_case(batch)
    assert batch.edge_to_event_uuid == {0: "from-graph-2"}
    graph_edges.reverse()
    with pytest.raises(ValueError, match="Graph order"):
        generator.generate_from_graph("trusted-graph", str(sidecar), temporal_data_path=str(artifact))

    payload = {"src": [3, 3], "dst": [4, 4], "t": [200, 200], "operation": ["EVENT_WRITE"] * 2}
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    graph_edges[:] = [(3, 4, i, {"time": 200, "label": "EVENT_WRITE", "event_uuid": f"uuid-{i}"}) for i in range(2)]
    generator.generate_from_graph("trusted-graph", str(sidecar), temporal_data_path=str(artifact))
    SidecarMappingProvider(str(sidecar), temporal_data_path=artifact, graph_edge_offset=1).enrich_case(batch)
    assert batch.edge_to_event_uuid == {}
    assert batch.edge_to_original_metadata[0]["status"] == "ambiguous"


def test_verified_uuid_still_rejects_database_duplicates_and_conflicting_endpoints(tmp_path):
    artifact, sidecar, batch = make_sidecar(tmp_path)
    loader = SidecarMappingProvider(str(sidecar), temporal_data_path=artifact, graph_edge_offset=1)
    loader.enrich_case(batch)
    DbAssistedMappingProvider().enrich_case(batch, rows=[event(3, 4, 200, "EVENT_WRITE", "event-2", 2),
                                                       event(3, 4, 200, "EVENT_WRITE", "event-2", 3)])
    assert batch.edge_to_event_uuid == {}
    assert batch.edge_to_original_metadata[0]["status"] == "ambiguous"
    loader.enrich_case(batch)
    DbAssistedMappingProvider().enrich_case(batch, rows=[event(4, 3, 200, "EVENT_WRITE", "event-2", 2)])
    assert batch.edge_to_event_uuid == {}
    assert batch.edge_to_original_metadata[0]["status"] == "conflict"


def test_postgresql_fetch_is_readonly_selective_parameterized_and_once_per_node(monkeypatch):
    calls = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, query, params):
            calls.append((query, params))

        def fetchall(self):
            if "subject_node_table" in calls[-1][0]:
                return [(1, "subject-1", "/bin/sh", "sh")]
            if "event_table" in calls[-1][0]:
                return [(1, 2, 100, "EVENT_READ", "event-1", 7)]
            return []

    class Connection:
        def set_session(self, **kwargs):
            assert kwargs == {"readonly": True, "autocommit": False}

        def cursor(self):
            return Cursor()

        def close(self):
            calls.append(("closed", None))

    monkeypatch.setitem(sys.modules, "psycopg2", SimpleNamespace(connect=lambda **kw: Connection()))
    case = make_case()
    DbAssistedMappingProvider(db_config={}).enrich_case(case, component_ids=["node:1"])
    node_queries = [(q, p) for q, p in calls if "_node_table" in q]
    assert len(node_queries) == 3
    assert all(params == ([1, 2, 4],) for _, params in node_queries)
    query, params = next((q, p) for q, p in calls if "event_table" in q)
    assert "SELECT *" not in query and "WHERE" in query and "%s" in query
    assert set(params) == {"1", "2", "4", 100, 101, "EVENT_READ", "EVENT_WRITE"}
    assert case.edge_to_event_uuid == {0: "event-1"}
    assert calls[-1][0] == "closed"


def test_database_errors_do_not_expose_connection_details(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("password=secret host=private")
    monkeypatch.setitem(sys.modules, "psycopg2", SimpleNamespace(connect=fail))
    with pytest.raises(RuntimeError) as error:
        DbAssistedMappingProvider(db_config={}).enrich_case(make_case())
    assert "secret" not in str(error.value) and "private" not in str(error.value)
    assert error.value.__suppress_context__
