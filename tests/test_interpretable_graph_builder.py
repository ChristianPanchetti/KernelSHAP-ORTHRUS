from __future__ import annotations

from preprocessing.schema import LogDataset, LogRecord
from perturbation.interpretable_graph_builder import InterpretableGraphBuilder

import pytest


def _ds(*records: LogRecord) -> LogDataset:
    return LogDataset(records=list(records), source_path=None, meta={})


def test_grouping_record_mode_assigns_each_record():
    r0 = LogRecord.from_dict({"_uid": "u0"})
    r1 = LogRecord.from_dict({"_uid": "u1"})
    r2 = LogRecord.from_dict({"_uid": "u2"})

    space = InterpretableGraphBuilder(grouping_mode="record").build(_ds(r0, r1, r2))

    assert space.num_components == 3
    assert space.record_to_component == (0, 1, 2)
    assert [c.component_id for c in space.components] == ["record:0", "record:1", "record:2"]


def test_grouping_exec_path_groups_when_available_and_falls_back():
    r0 = LogRecord.from_dict({"_uid": "u0", "executionBinary": {"path": "/bin/a"}})
    r1 = LogRecord.from_dict({"_uid": "u1", "executionBinary": {"path": "/bin/a"}})
    r2 = LogRecord.from_dict({"_uid": "u2"})

    space = InterpretableGraphBuilder(grouping_mode="exec_path").build(_ds(r0, r1, r2))

    ids = [c.component_id for c in space.components]
    assert "exec:/bin/a" in ids
    assert "record:2" in ids

    exec_idx = ids.index("exec:/bin/a")
    rec2_idx = ids.index("record:2")
    assert space.record_to_component[0] == exec_idx
    assert space.record_to_component[1] == exec_idx
    assert space.record_to_component[2] == rec2_idx


def test_grouping_entity_id_groups_by_entity_id_when_available():
    r0 = LogRecord.from_dict({"_uid": "u0", "entityId": 1})
    r1 = LogRecord.from_dict({"_uid": "u1", "entityId": 1})
    r2 = LogRecord.from_dict({"_uid": "u2", "entityId": 2})
    r3 = LogRecord.from_dict({"_uid": "u3"})

    space = InterpretableGraphBuilder(grouping_mode="entity_id").build(_ds(r0, r1, r2, r3))

    ids = [c.component_id for c in space.components]
    assert "entity:1" in ids
    assert "entity:2" in ids
    assert "record:3" in ids

    ent1_idx = ids.index("entity:1")
    ent2_idx = ids.index("entity:2")
    rec3_idx = ids.index("record:3")
    assert space.record_to_component[0] == ent1_idx
    assert space.record_to_component[1] == ent1_idx
    assert space.record_to_component[2] == ent2_idx
    assert space.record_to_component[3] == rec3_idx


def test_grouping_parent_child_groups_edges_when_available():
    r0 = LogRecord.from_dict({"_uid": "u0", "parentEntityId": 10, "entityId": 11})
    r1 = LogRecord.from_dict({"_uid": "u1", "parentEntityId": 10, "entityId": 11})
    r2 = LogRecord.from_dict({"_uid": "u2", "parentEntityId": 10, "entityId": 12})
    r3 = LogRecord.from_dict({"_uid": "u3", "entityId": 13})

    space = InterpretableGraphBuilder(grouping_mode="parent_child").build(_ds(r0, r1, r2, r3))

    ids = [c.component_id for c in space.components]
    assert "parent_child:10->11" in ids
    assert "parent_child:10->12" in ids
    assert "record:3" in ids

    e1 = ids.index("parent_child:10->11")
    e2 = ids.index("parent_child:10->12")
    rec3 = ids.index("record:3")
    assert space.record_to_component[0] == e1
    assert space.record_to_component[1] == e1
    assert space.record_to_component[2] == e2
    assert space.record_to_component[3] == rec3


def test_max_components_merges_remaining_into_other():
    records = [
        LogRecord.from_dict({"_uid": f"u{i}", "executionBinary": {"path": f"/bin/{i}"}})
        for i in range(6)
    ]
    space = InterpretableGraphBuilder(grouping_mode="exec_path", max_components=3).build(_ds(*records))

    assert space.num_components == 3
    assert space.components[-1].component_id == "OTHER"

    all_indices = []
    for c in space.components:
        all_indices.extend(list(c.record_indices))

    assert sorted(all_indices) == list(range(6))


def test_time_window_requires_parseable_timestamps():
    r0 = LogRecord.from_dict({"_uid": "u0"})
    r1 = LogRecord.from_dict({"_uid": "u1"})

    builder = InterpretableGraphBuilder(grouping_mode="time_window", time_window_seconds=60)
    with pytest.raises(ValueError, match=r"time_window.*requires valid timestamps"):
        builder.build(_ds(r0, r1))


def test_time_window_groups_by_fixed_windows():
    r0 = LogRecord.from_dict({"_uid": "u0", "startTime": "2026-01-01T00:00:00.000000000Z"})
    r1 = LogRecord.from_dict({"_uid": "u1", "startTime": "2026-01-01T00:00:30Z"})
    r2 = LogRecord.from_dict({"_uid": "u2", "startTime": "2026-01-01T00:01:00Z"})

    space = InterpretableGraphBuilder(
        grouping_mode="time_window",
        time_window_seconds=60,
    ).build(_ds(r0, r1, r2))

    ids = [c.component_id for c in space.components]
    assert "time_window:2026-01-01T00:00:00Z" in ids
    assert "time_window:2026-01-01T00:01:00Z" in ids

    w0 = ids.index("time_window:2026-01-01T00:00:00Z")
    w1 = ids.index("time_window:2026-01-01T00:01:00Z")
    assert space.record_to_component[0] == w0
    assert space.record_to_component[1] == w0
    assert space.record_to_component[2] == w1


def test_event_type_errors_if_missing_everywhere():
    r0 = LogRecord.from_dict({"_uid": "u0"})
    r1 = LogRecord.from_dict({"_uid": "u1"})

    builder = InterpretableGraphBuilder(grouping_mode="event_type")
    with pytest.raises(ValueError, match=r"event_type.*requires an event type"):
        builder.build(_ds(r0, r1))


def test_event_type_groups_when_present_and_falls_back_for_missing():
    r0 = LogRecord.from_dict({"_uid": "u0", "eventType": "PROC_START"})
    r1 = LogRecord.from_dict({"_uid": "u1", "eventType": "PROC_START"})
    r2 = LogRecord.from_dict({"_uid": "u2"})

    space = InterpretableGraphBuilder(grouping_mode="event_type").build(_ds(r0, r1, r2))
    ids = [c.component_id for c in space.components]

    assert "event_type:PROC_START" in ids
    assert "record:2" in ids


def test_local_subgraph_is_not_implemented():
    r0 = LogRecord.from_dict({"_uid": "u0"})
    with pytest.raises(NotImplementedError):
        InterpretableGraphBuilder(grouping_mode="local_subgraph").build(_ds(r0))
