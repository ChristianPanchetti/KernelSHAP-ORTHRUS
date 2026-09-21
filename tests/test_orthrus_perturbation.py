from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from adapters.orthrus_ano_adapter import DummyOrthrusAnoAdapter
from perturbation.orthrus_interpretable_builder import OrthrusInterpretableBuilder
from perturbation.orthrus_perturbation_manager import OrthrusPerturbationManager
from preprocessing.log_input_loader import LogInputLoader
from preprocessing.orthrus_alert_case import OrthrusAlertCase


class FakeTemporalData:
    def __init__(self, src, dst, t, edge_type=None, msg=None, x_src=None, x_dst=None, edge_feats=None):
        self.src = list(src)
        self.dst = list(dst)
        self.t = list(t)
        if edge_type is not None:
            self.edge_type = list(edge_type)
        if msg is not None:
            self.msg = list(msg)
        if x_src is not None:
            self.x_src = list(x_src)
        if x_dst is not None:
            self.x_dst = list(x_dst)
        if edge_feats is not None:
            self.edge_feats = list(edge_feats)


def test_drop_edges_by_edge_type():
    td = FakeTemporalData(
        src=[1, 1, 2, 2],
        dst=[2, 3, 3, 4],
        t=[100, 120, 180, 200],
        edge_type=[0, 1, 1, 0],
        msg=[10, 20, 30, 40],
        x_src=[[1], [2], [3], [4]],
        x_dst=[[5], [6], [7], [8]],
        edge_feats=[[0.1], [0.2], [0.3], [0.4]],
    )
    case = OrthrusAlertCase(temporal_data=td)
    OrthrusInterpretableBuilder(grouping_mode="edge_type").build(case)
    # Drop edge_type=1
    comp_ids = list(case.component_to_edges.keys())
    mask = [0 if "edge_type:1" in cid else 1 for cid in comp_ids]
    pert = OrthrusPerturbationManager(case).apply_mask(mask).dataset
    # Only edges with edge_type=0 remain (indices 0, 3)
    assert pert.temporal_data.src == [1, 2]
    assert pert.temporal_data.dst == [2, 4]
    assert pert.temporal_data.t == [100, 200]
    assert pert.temporal_data.msg == [10, 40]
    assert pert.temporal_data.x_src == [[1], [4]]
    assert pert.temporal_data.x_dst == [[5], [8]]
    assert pert.temporal_data.edge_feats == [[0.1], [0.4]]
    assert pert.temporal_data.edge_type == [0, 0]
    # edge_index is reconstructed
    assert hasattr(pert.temporal_data, "edge_index")
    np.testing.assert_array_equal(pert.temporal_data.edge_index, np.stack([[1, 2], [2, 4]], axis=0))
    # All edge-aligned fields have same length
    n = len(pert.temporal_data.src)
    for field in ["dst", "t", "msg", "x_src", "x_dst", "edge_feats", "edge_type"]:
        if hasattr(pert.temporal_data, field):
            assert len(getattr(pert.temporal_data, field)) == n
    # Metadata
    assert pert.metadata["perturbation_mode"] == "drop_edges"
    assert pert.metadata["original_num_edges"] == 4
    assert pert.metadata["perturbed_num_edges"] == 2
    assert "edge_type:1" in pert.metadata["dropped_components"]
    assert "edge_type:0" in pert.metadata["active_components"]


def test_drop_edges_by_time_chunk():
    td = FakeTemporalData(
        src=[1, 1, 2, 2],
        dst=[2, 3, 3, 4],
        t=[100, 120, 180, 200],
        edge_type=[0, 1, 1, 0],
    )
    case = OrthrusAlertCase(temporal_data=td)
    OrthrusInterpretableBuilder(grouping_mode="time_chunk", time_chunk_seconds=1).build(case)
    # Drop first chunk
    comp_ids = list(case.component_to_edges.keys())
    mask = [0 if ":0" in cid else 1 for cid in comp_ids]
    pert = OrthrusPerturbationManager(case).apply_mask(mask).dataset
    # Only edges with t > min(t) remain
    kept_t = [td.t[i] for i in range(4) if td.t[i] != min(td.t)]
    assert pert.temporal_data.t == kept_t
    n = len(pert.temporal_data.t)
    for field in ["src", "dst", "edge_type"]:
        assert len(getattr(pert.temporal_data, field)) == n


def test_neutralize_edges_preserves_topology_and_zeroes_features():
    td = FakeTemporalData(
        src=[1, 1, 2, 2],
        dst=[2, 3, 3, 4],
        t=[100, 120, 180, 200],
        edge_type=[0, 1, 1, 0],
        msg=[[10, 11], [20, 21], [30, 31], [40, 41]],
        x_src=[[1], [2], [3], [4]],
        x_dst=[[5], [6], [7], [8]],
        edge_feats=[[0.1], [0.2], [0.3], [0.4]],
    )
    td.edge_index = np.stack([[1, 1, 2, 2], [2, 3, 3, 4]], axis=0)

    full_data = {"edge_type": [[1, 0], [0, 1]]}
    case = OrthrusAlertCase(temporal_data=td, full_data=full_data)
    OrthrusInterpretableBuilder(grouping_mode="edge_type").build(case)
    comp_ids = list(case.component_to_edges.keys())
    mask = [0 if "edge_type:1" in cid else 1 for cid in comp_ids]

    pert = OrthrusPerturbationManager(case, mode="neutralize_edges").apply_mask(mask).dataset

    assert pert.num_edges == 4
    assert pert.temporal_data.src == td.src
    assert pert.temporal_data.dst == td.dst
    assert pert.temporal_data.t == td.t
    assert pert.temporal_data.edge_type == td.edge_type
    np.testing.assert_array_equal(pert.temporal_data.edge_index, td.edge_index)

    assert pert.temporal_data.msg == td.msg
    assert pert.temporal_data.x_src == [[0], [0], [0], [0]]
    assert pert.temporal_data.x_dst == [[5], [0], [0], [8]]
    assert pert.temporal_data.edge_feats == td.edge_feats
    assert td.x_src == [[1], [2], [3], [4]]
    assert td.x_dst == [[5], [6], [7], [8]]
    assert pert.full_data is full_data
    assert full_data == {"edge_type": [[1, 0], [0, 1]]}
    assert case.metadata == {}

    assert pert.metadata["perturbation_mode"] == "neutralize_edges"
    assert pert.metadata["original_num_edges"] == 4
    assert pert.metadata["perturbed_num_edges"] == 4
    assert pert.metadata["neutralized_edges"] == [1, 2]
    assert pert.metadata["neutralized_source_rows"] == [0, 1, 2, 3]
    assert pert.metadata["neutralized_destination_rows"] == [1, 2]
    assert "edge_type:1" in pert.metadata["inactive_components"]
    assert "edge_type:0" in pert.metadata["active_components"]


def test_neutralize_edges_preserves_torch_dtype_and_device():
    torch = pytest.importorskip("torch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td = FakeTemporalData(
        src=[1, 1, 2],
        dst=[2, 3, 3],
        t=[100, 120, 180],
        edge_type=[0, 1, 0],
    )
    td.src = torch.tensor(td.src, dtype=torch.long, device=device)
    td.dst = torch.tensor(td.dst, dtype=torch.long, device=device)
    td.t = torch.tensor(td.t, dtype=torch.long, device=device)
    td.edge_type = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32, device=device)
    td.msg = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32, device=device)
    td.x_src = torch.tensor([[1], [2], [3]], dtype=torch.int64, device=device)
    td.x_dst = torch.tensor([[4], [5], [6]], dtype=torch.int64, device=device)
    td.edge_feats = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float64, device=device)
    td.edge_index = torch.stack([td.src, td.dst])

    case = OrthrusAlertCase(temporal_data=td, component_to_edges={"z_keep": [0, 2], "a_mask": [1]})
    assert OrthrusInterpretableBuilder.suggested_component_ids(case) == ["z_keep", "a_mask"]
    originals = {field: getattr(td, field).clone() for field in ("x_src", "x_dst")}
    pert = OrthrusPerturbationManager(case, mode="mask_edge_features").apply_mask([1, 0]).dataset

    assert pert.num_edges == 3
    assert torch.equal(pert.temporal_data.src, td.src)
    assert torch.equal(pert.temporal_data.dst, td.dst)
    assert torch.equal(pert.temporal_data.t, td.t)
    assert torch.equal(pert.temporal_data.edge_index, td.edge_index)
    assert torch.equal(pert.temporal_data.edge_type, td.edge_type)

    assert pert.temporal_data.msg.dtype == td.msg.dtype
    assert pert.temporal_data.msg.device == td.msg.device
    assert pert.temporal_data.x_src.dtype == td.x_src.dtype
    assert pert.temporal_data.x_src.device == td.x_src.device
    assert pert.temporal_data.edge_feats.dtype == td.edge_feats.dtype
    assert pert.temporal_data.edge_feats.device == td.edge_feats.device

    assert torch.equal(pert.temporal_data.msg, td.msg)
    assert torch.equal(pert.temporal_data.edge_feats, td.edge_feats)
    assert pert.temporal_data.x_src.tolist() == [[0], [0], [3]]
    assert pert.temporal_data.x_dst.tolist() == [[4], [0], [0]]
    for field in ("x_src", "x_dst"):
        assert torch.equal(getattr(td, field), originals[field])
        assert getattr(pert.temporal_data, field).dtype == getattr(td, field).dtype
        assert getattr(pert.temporal_data, field).device == device
    assert pert.metadata["inactive_components"] == ["a_mask"]
    again = OrthrusPerturbationManager(case, mode="neutralize_edges").apply_mask([1, 1]).dataset
    assert torch.equal(again.temporal_data.x_src, td.x_src)
    assert torch.equal(again.temporal_data.x_dst, td.x_dst)


def test_dummy_mode_still_works():
    # Should not raise or break
    from preprocessing.log_input_loader import LogInputLoader
    from adapters.orthrus_ano_adapter import DummyOrthrusAnoAdapter
    import logging
    import pathlib
    dataset = LogInputLoader(logger=logging.getLogger("test")).load(pathlib.Path(__file__).parent.parent / "logs_input.json")
    score = DummyOrthrusAnoAdapter(seed=0, logger=logging.getLogger("test")).predict_anomaly_score(dataset)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_component_order_is_insertion_order_and_masks_start_from_original():
    td = FakeTemporalData(src=[1, 2], dst=[2, 1], t=[10, 20],
                          x_src=[[3], [4]], x_dst=[[5], [6]])
    case = OrthrusAlertCase(td, component_to_edges={"z_first": [0], "a_second": [1]})
    assert OrthrusInterpretableBuilder.suggested_component_ids(case) == ["z_first", "a_second"]
    manager = OrthrusPerturbationManager(case, mode="neutralize_edges")
    first = manager.apply_mask([0, 1]).dataset
    assert first.temporal_data.x_src == [[0], [4]]
    assert first.temporal_data.x_dst == [[0], [6]]
    assert first.metadata["inactive_components"] == ["z_first"]
    second = manager.apply_mask([1, 0]).dataset
    assert second.temporal_data.x_src == [[3], [0]]
    assert second.temporal_data.x_dst == [[5], [0]]
    assert manager.apply_mask([1, 1]).dataset.temporal_data.x_src == td.x_src
    with pytest.raises(ValueError, match="only 0 or 1"):
        manager.apply_mask([1, 2])
