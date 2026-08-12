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


def test_real_adapter_moves_batch_and_full_data_to_device():
    case = make_case(2)
    model = FakeModel([2, 4])
    adapter = RealOrthrusAnoAdapter(model=model, device="test-device")

    assert adapter.predict_anomaly_score(case) == pytest.approx(3.0)
    assert case.temporal_data.moved_to == "test-device"
    assert case.full_data.moved_to == "test-device"


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
