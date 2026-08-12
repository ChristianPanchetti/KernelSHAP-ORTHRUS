from __future__ import annotations

import pytest

from preprocessing.schema import LogDataset, LogRecord
from perturbation.interpretable_graph_builder import InterpretableComponent, InterpretableSpace
from perturbation.perturbation_manager import PerturbationManager


def _make_space_for_n_records(n: int, grouping_mode: str = "record") -> InterpretableSpace:
    components = []
    for i in range(n):
        components.append(
            InterpretableComponent(
                component_id=f"record:{i}",
                name=f"record:{i}",
                kind=grouping_mode,
                record_indices=(i,),
                description="",
            )
        )
    return InterpretableSpace(
        components=components,
        record_to_component=tuple(range(n)),
        build_info={"grouping_mode": grouping_mode},
    )


def test_mask_length_mismatch_raises_clear_error():
    r0 = LogRecord.from_dict({"_uid": "u0"})
    ds = LogDataset(records=[r0], source_path=None, meta={})
    space = _make_space_for_n_records(1)
    pm = PerturbationManager(original=ds, space=space, mode="drop_records")

    with pytest.raises(ValueError, match=r"Mask length mismatch"):
        pm.apply_mask([1, 0])


def test_drop_records_all_zero_all_one():
    r0 = LogRecord.from_dict({"_uid": "u0"})
    r1 = LogRecord.from_dict({"_uid": "u1"})
    ds = LogDataset(records=[r0, r1], source_path=None, meta={})
    space = _make_space_for_n_records(2)
    pm = PerturbationManager(original=ds, space=space, mode="drop_records")

    res0 = pm.apply_mask([0, 0])
    assert len(res0.dataset.records) == 0

    res1 = pm.apply_mask([1, 1])
    assert len(res1.dataset.records) == 2


def test_neutralize_command_keeps_records_but_sanitizes_fields():
    r0 = LogRecord.from_dict(
        {
            "_uid": "u0",
            "executionBinary": {"path": "/bin/bash"},
            "cmdlineArgs": ["-c", "echo ok"],
        }
    )
    r1 = LogRecord.from_dict(
        {
            "_uid": "u1",
            "executionBinary": {"path": "/usr/bin/python3"},
            "cmdlineArgs": ["-c", "import os"],
        }
    )
    ds = LogDataset(records=[r0, r1], source_path=None, meta={})
    space = _make_space_for_n_records(2)
    pm = PerturbationManager(original=ds, space=space, mode="neutralize_command")

    res = pm.apply_mask([1, 0])
    assert len(res.dataset.records) == 2

    kept = res.dataset.records[0]
    neut = res.dataset.records[1]

    assert kept.execution_path == "/bin/bash"
    assert kept.cmdline_args == ["-c", "echo ok"]

    assert neut.execution_path is None
    assert neut.cmdline_args is None

    # Ensure raw is sanitized and the original record raw was not mutated.
    assert neut.raw.get("executionBinary", {}).get("path") == ""
    assert neut.raw.get("cmdlineArgs") is None
    assert r1.raw.get("executionBinary", {}).get("path") == "/usr/bin/python3"


def test_neutralize_command_all_zero_all_one_masks():
    r0 = LogRecord.from_dict({"_uid": "u0", "executionBinary": {"path": "/bin/bash"}})
    r1 = LogRecord.from_dict({"_uid": "u1", "executionBinary": {"path": "/usr/bin/python3"}})
    ds = LogDataset(records=[r0, r1], source_path=None, meta={})
    space = _make_space_for_n_records(2)
    pm = PerturbationManager(original=ds, space=space, mode="neutralize_command")

    res0 = pm.apply_mask([0, 0])
    assert len(res0.dataset.records) == 2
    assert all(r.execution_path is None for r in res0.dataset.records)

    res1 = pm.apply_mask([1, 1])
    assert [r.execution_path for r in res1.dataset.records] == ["/bin/bash", "/usr/bin/python3"]


def test_not_implemented_modes_raise_clear_errors():
    r0 = LogRecord.from_dict({"_uid": "u0"})
    ds = LogDataset(records=[r0], source_path=None, meta={})
    space = _make_space_for_n_records(1)

    for mode in ["mask_features", "drop_subgraph", "remove_node_with_incident_edges"]:
        pm = PerturbationManager(original=ds, space=space, mode=mode)
        with pytest.raises(NotImplementedError, match=mode):
            pm.apply_mask([1])


def test_drop_time_window_requires_time_window_space():
    r0 = LogRecord.from_dict({"_uid": "u0"})
    ds = LogDataset(records=[r0], source_path=None, meta={})
    space = _make_space_for_n_records(1, grouping_mode="record")
    pm = PerturbationManager(original=ds, space=space, mode="drop_time_window")

    with pytest.raises(ValueError, match=r"requires.*grouping_mode='time_window'"):
        pm.apply_mask([1])


def test_drop_time_window_behaves_like_drop_records_when_enabled():
    r0 = LogRecord.from_dict({"_uid": "u0"})
    r1 = LogRecord.from_dict({"_uid": "u1"})
    ds = LogDataset(records=[r0, r1], source_path=None, meta={})
    space = _make_space_for_n_records(2, grouping_mode="time_window")
    pm = PerturbationManager(original=ds, space=space, mode="drop_time_window")

    res = pm.apply_mask([1, 0])
    assert len(res.dataset.records) == 1
    assert res.dataset.records[0].raw.get("_uid") == "u0"
