from __future__ import annotations

import pytest

from adapters.orthrus_ano_adapter import RealOrthrusAnoAdapter
from preprocessing.orthrus_alert_case import OrthrusAlertCase


class FakeEdgeIndex:
    def __init__(self, num_edges: int):
        self.shape = (2, num_edges)


class MovableData:
    def __init__(self, num_edges: int):
        self.edge_index = FakeEdgeIndex(num_edges)
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class MovableFullData:
    def __init__(self):
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeModel:
    def __init__(self, losses):
        self.losses = losses
        self.eval_called = False
        self.calls = []

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, batch, full_data, *, inference):
        self.calls.append((batch, full_data, inference))
        return self.losses


def make_case(num_edges: int = 3):
    return OrthrusAlertCase(
        temporal_data=MovableData(num_edges),
        full_data=MovableFullData(),
    )


def test_real_adapter_calls_model_and_returns_mean():
    case = make_case(3)
    model = FakeModel([1.0, 2.0, 6.0])
    adapter = RealOrthrusAnoAdapter(model=model)

    score = adapter.predict_anomaly_score(case)

    assert score == pytest.approx(3.0)
    assert model.calls == [(case.temporal_data, case.full_data, True)]
    assert model.eval_called is True


def test_real_adapter_moves_batch_copy_and_leaves_history_unchanged():
    case = make_case(2)
    model = FakeModel([2, 4])
    adapter = RealOrthrusAnoAdapter(model=model, device="test-device")

    assert adapter.predict_anomaly_score(case) == pytest.approx(3.0)
    assert model.calls[0][0].moved_to == "test-device"
    assert case.temporal_data.moved_to is None
    assert model.calls[0][1] is case.full_data
    assert case.full_data.moved_to is None


def test_real_adapter_can_leave_model_mode_unchanged():
    model = FakeModel([1])
    adapter = RealOrthrusAnoAdapter(model=model, require_eval_mode=False)

    assert adapter.predict_anomaly_score(make_case(1)) == 1.0
    assert model.eval_called is False


def test_real_adapter_rejects_unexpected_case_type():
    adapter = RealOrthrusAnoAdapter(model=FakeModel([1]))
    with pytest.raises(TypeError, match="OrthrusAlertCase"):
        adapter.predict_anomaly_score(object())


def test_real_adapter_requires_temporal_data():
    adapter = RealOrthrusAnoAdapter(model=FakeModel([1]))
    case = OrthrusAlertCase(temporal_data=None, full_data=object())
    with pytest.raises(ValueError, match="temporal_data"):
        adapter.predict_anomaly_score(case)


def test_real_adapter_requires_full_data():
    adapter = RealOrthrusAnoAdapter(model=FakeModel([1]))
    case = OrthrusAlertCase(temporal_data=MovableData(1), full_data=None)
    with pytest.raises(ValueError, match="full_data"):
        adapter.predict_anomaly_score(case)


def test_real_adapter_rejects_empty_output():
    adapter = RealOrthrusAnoAdapter(model=FakeModel([]))
    with pytest.raises(ValueError, match="empty"):
        adapter.predict_anomaly_score(make_case(1))


def test_real_adapter_rejects_loss_length_mismatch():
    adapter = RealOrthrusAnoAdapter(model=FakeModel([1.0, 2.0]))
    with pytest.raises(ValueError, match="length mismatch"):
        adapter.predict_anomaly_score(make_case(3))


def test_real_adapter_rejects_non_numeric_output_and_unknown_reduction():
    with pytest.raises(ValueError, match="only 'mean'"):
        RealOrthrusAnoAdapter(model=FakeModel([1]), score_reduction="max")

    adapter = RealOrthrusAnoAdapter(model=FakeModel(["not-a-number"]))
    with pytest.raises(TypeError, match="not numeric"):
        adapter.predict_anomaly_score(make_case(1))


def test_real_adapter_counts_edges_from_src_and_msg_fallbacks():
    src_batch = type("SrcBatch", (), {"src": [1, 2]})()
    msg_batch = type("MsgBatch", (), {"msg": [[1], [2], [3]]})()

    assert RealOrthrusAnoAdapter(FakeModel([1, 3])).predict_anomaly_score(
        OrthrusAlertCase(src_batch, full_data=object())
    ) == pytest.approx(2.0)
    assert RealOrthrusAnoAdapter(FakeModel([1, 2, 6])).predict_anomaly_score(
        OrthrusAlertCase(msg_batch, full_data=object())
    ) == pytest.approx(3.0)


def test_real_adapter_accepts_torch_tensor_and_disables_gradients():
    torch = pytest.importorskip("torch")

    class TorchModel(FakeModel):
        def __call__(self, batch, full_data, *, inference):
            assert torch.is_grad_enabled() is False
            return super().__call__(batch, full_data, inference=inference)

    losses = torch.tensor([[1.0], [3.0], [5.0]], requires_grad=True)
    adapter = RealOrthrusAnoAdapter(model=TorchModel(losses))

    assert adapter.predict_anomaly_score(make_case(3)) == pytest.approx(3.0)


@pytest.mark.parametrize("empty_cache", [True, False])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_mask_scores_restore_temporal_state_and_caches_even_on_failure(empty_cache, backend):
    from types import SimpleNamespace
    from perturbation.orthrus_perturbation_manager import OrthrusPerturbationManager

    import numpy as np
    xp = pytest.importorskip("torch") if backend == "torch" else np
    tensor = xp.tensor if backend == "torch" else np.array
    loader = SimpleNamespace(
        cur_e_id=7, neighbors=tensor([[1, 2]]),
        e_id=tensor([[5, 6]]), _assoc=tensor([0, 1]),
    )
    reindexer = SimpleNamespace(
        x_src_cache=None if empty_cache else tensor([[2.0], [3.0]]),
        x_dst_cache=None if empty_cache else tensor([[4.0], [5.0]]),
        assoc=None,
    )

    class StatefulModel(FakeModel):
        fail = False

        def __init__(self):
            super().__init__([])
            self.encoder = SimpleNamespace(
                neighbor_loader=loader, graph_reindexer=reindexer, assoc=tensor([0, 1])
            )
            self.graph_reindexer = reindexer

        def __call__(self, batch, full_data, *, inference):
            assert inference and self.eval_called
            if backend == "torch":
                assert not xp.is_grad_enabled()
            score = float(loader.cur_e_id + batch.x_src.sum() + batch.x_dst.sum())
            if reindexer.x_src_cache is not None:
                score += float(reindexer.x_src_cache.sum() + reindexer.x_dst_cache.sum())
            loader.cur_e_id += len(batch.src)
            loader.neighbors[...] += 10
            loader.e_id[...] += 20
            loader._assoc[...] = 99
            self.encoder.assoc[...] = 88
            for field in ("x_src_cache", "x_dst_cache"):
                if getattr(reindexer, field) is None:
                    setattr(reindexer, field, xp.ones((2, 1)))
                else:
                    getattr(reindexer, field)[...] += 30
            reindexer.assoc = tensor([9])
            if self.fail:
                raise RuntimeError("inference failed after mutation")
            return [score] * len(batch.src)

    model = StatefulModel()
    batch = SimpleNamespace(
        src=tensor([0, 1]), dst=tensor([1, 0]), t=tensor([10, 20]),
        edge_index=tensor([[0, 1], [1, 0]]),
        x_src=tensor([[1.0], [2.0]]), x_dst=tensor([[3.0], [4.0]]),
        edge_type=xp.eye(2),
    )
    full_data = SimpleNamespace(edge_type=xp.eye(2))
    case = OrthrusAlertCase(batch, full_data=full_data, component_to_edges={"z": [0], "a": [1]})
    manager = OrthrusPerturbationManager(case, mode="neutralize_edges")
    adapter = RealOrthrusAnoAdapter(model, isolate_state=True)

    def assert_restored():
        assert loader.cur_e_id == 7
        assert loader.neighbors.tolist() == [[1, 2]]
        assert loader.e_id.tolist() == [[5, 6]]
        assert loader._assoc.tolist() == [0, 1]
        assert model.encoder.assoc.tolist() == [0, 1]
        assert reindexer.assoc is None
        if empty_cache:
            assert reindexer.x_src_cache is None and reindexer.x_dst_cache is None
        else:
            assert reindexer.x_src_cache.tolist() == [[2.0], [3.0]]
            assert reindexer.x_dst_cache.tolist() == [[4.0], [5.0]]
        assert full_data.edge_type.tolist() == [[1, 0], [0, 1]]
        assert batch.x_src.tolist() == [[1.0], [2.0]]

    a = adapter.score_mask(manager, [1, 1])
    assert_restored()
    b = adapter.score_mask(manager, [0, 1])
    assert a != b
    assert_restored()
    assert adapter.score_mask(manager, [1, 1]) == a
    assert_restored()
    model.fail = True
    with pytest.raises(RuntimeError, match="after mutation"):
        adapter.score_mask(manager, [0, 1])
    assert_restored()
    model.fail = False
    assert adapter.score_mask(manager, [1, 1]) == a


def test_mask_scoring_requires_explicit_isolation():
    adapter = RealOrthrusAnoAdapter(FakeModel([1]))
    with pytest.raises(ValueError, match="isolate_state=True"):
        adapter.score_mask(None, [1])
    with pytest.raises(ValueError, match="neighbor loader"):
        RealOrthrusAnoAdapter(FakeModel([1]), isolate_state=True)
