from __future__ import annotations

from preprocessing.schema import LogDataset, LogRecord
from perturbation.interpretable_graph_builder import InterpretableComponent, InterpretableSpace
from perturbation.perturbation_manager import PerturbationManager


def make_simple_space():
    # Two components: record 0 -> comp 0, record 1 -> comp 1
    comp0 = InterpretableComponent(
        component_id="c0",
        name="c0",
        kind="execution_path",
        record_indices=(0,),
        description="",
    )
    comp1 = InterpretableComponent(
        component_id="c1",
        name="c1",
        kind="execution_path",
        record_indices=(1,),
        description="",
    )
    space = InterpretableSpace(components=[comp0, comp1], record_to_component=(0, 1), build_info={"grouping_mode": "exec_path"})
    return space


def test_apply_mask_behaviour():
    # Prepare two minimal records
    r0 = LogRecord.from_dict({"_uid": "u0"})
    r1 = LogRecord.from_dict({"_uid": "u1"})
    ds = LogDataset(records=[r0, r1], source_path=None, meta={})

    space = make_simple_space()
    pm = PerturbationManager(original=ds, space=space, mode="drop_records")

    # Keep only component 0 -> should keep record 0 only
    res = pm.apply_mask([1, 0])
    assert isinstance(res.dataset, LogDataset)
    assert len(res.dataset.records) == 1
    assert res.dataset.records[0].raw.get("_uid") == "u0"

    # Keep none -> empty records
    res2 = pm.apply_mask([0, 0])
    assert len(res2.dataset.records) == 0

    # Keep both -> both records present
    res3 = pm.apply_mask([1, 1])
    assert len(res3.dataset.records) == 2
