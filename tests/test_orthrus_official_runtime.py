from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from adapters.orthrus_runtime import (
    OfficialOrthrusModules,
    OrthrusOfficialRuntimeConfig,
    OrthrusRuntimeError,
    run_official_orthrus_smoke_test,
)


class FakeEdgeIndex:
    def __init__(self, count):
        self.shape = (2, count)


class FakeGraph:
    def __init__(self, name, edges=3):
        self.name = name
        self.edge_index = FakeEdgeIndex(edges)
        self.moved_to = None

    def to(self, device=None, **kwargs):
        self.moved_to = device
        return self


class FakeModel:
    def __init__(self):
        self.graph_reindexer = object()
        self.calls = []
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, batch, full_data, *, inference):
        self.calls.append((batch, full_data, inference))
        return [1.0, 2.0, 6.0]


def make_config(tmp_path: Path, **changes):
    root = tmp_path / "orthrus"
    (root / "src").mkdir(parents=True)
    config_path = root / "config" / "orthrus.yml"
    config_path.parent.mkdir()
    config_path.write_text("test", encoding="utf-8")
    epoch = tmp_path / "model_epoch_1"
    epoch.mkdir()
    (epoch / "state_dict.pkl").write_bytes(b"test")
    (epoch / "neighbor_loader.pkl").write_bytes(b"test")
    values = dict(
        external_root=root,
        config_path=config_path,
        dataset_name="THEIA_E3",
        model_epoch_dir=epoch,
        split="test",
        graph_index=1,
        batch_index=1,
        device="cpu",
    )
    values.update(changes)
    return OrthrusOfficialRuntimeConfig(**values)


def make_modules(calls):
    cfg = SimpleNamespace(name="cfg")
    graphs = {
        "train": [FakeGraph("train-0")],
        "val": [FakeGraph("val-0")],
        "test": [FakeGraph("test-0"), FakeGraph("test-1")],
    }
    full_data = object()
    model = FakeModel()
    batches = [FakeGraph("batch-0"), FakeGraph("batch-1")]

    def get_args(*, args):
        calls.append(("cfg_args", args))
        return SimpleNamespace(parsed=args)

    def get_cfg(args):
        calls.append(("cfg_loader", args))
        return cfg

    def load_all(received_cfg):
        calls.append(("load_all_datasets", received_cfg))
        return graphs["train"], graphs["val"], graphs["test"], full_data, 99

    def build_model(**kwargs):
        calls.append(("build_model", kwargs))
        return model

    def load_model(received_model, path):
        calls.append(("load_model", received_model, path))
        return received_model

    def batch_factory(received_cfg, graph, reindexer):
        calls.append(("batch_loader_factory", received_cfg, graph, reindexer))
        return iter(batches)

    modules = OfficialOrthrusModules(
        config=SimpleNamespace(get_runtime_required_args=get_args, get_yml_cfg=get_cfg),
        data_utils=SimpleNamespace(load_all_datasets=load_all, load_model=load_model),
        factory=SimpleNamespace(build_model=build_model, batch_loader_factory=batch_factory),
    )
    return modules, cfg, graphs, full_data, model, batches


def test_official_runtime_follows_cfg_dataset_model_loader_and_adapter_contract(tmp_path: Path):
    calls = []
    config = make_config(tmp_path)
    modules, cfg, graphs, full_data, model, batches = make_modules(calls)

    result = run_official_orthrus_smoke_test(config, modules=modules)

    assert calls[0] == ("cfg_args", ["THEIA_E3", "--cpu"])
    assert calls[1][0] == "cfg_loader"
    assert calls[2] == ("load_all_datasets", cfg)
    build_call = calls[3][1]
    assert build_call == {
        "data_sample": graphs["test"][1],
        "device": "cpu",
        "cfg": cfg,
        "max_node_num": 99,
    }
    assert calls[4] == ("load_model", model, str(config.model_epoch_dir))
    assert calls[5] == (
        "batch_loader_factory",
        cfg,
        graphs["test"][1],
        model.graph_reindexer,
    )
    assert graphs["test"][1].moved_to == "cpu"
    assert model.calls == [(batches[1], full_data, True)]
    assert model.eval_called is True
    assert result.score == pytest.approx(3.0)
    assert result.num_edges == result.edge_loss_count == 3
    assert result.dataset == "THEIA_E3"
    assert result.split == "test"
    assert result.graph_index == result.batch_index == 1
    assert result.device == "cpu"


def test_official_runtime_rejects_unknown_split(tmp_path: Path):
    config = make_config(tmp_path, split="unknown")
    modules, *_ = make_modules([])

    with pytest.raises(OrthrusRuntimeError, match="Unknown ORTHRUS split"):
        run_official_orthrus_smoke_test(config, modules=modules)


def test_official_runtime_rejects_out_of_range_batch_index(tmp_path: Path):
    config = make_config(tmp_path, batch_index=8)
    modules, *_ = make_modules([])

    with pytest.raises(OrthrusRuntimeError, match="batch_index 8 is out of range"):
        run_official_orthrus_smoke_test(config, modules=modules)


def test_official_runtime_requires_model_epoch_for_faithful_state(tmp_path: Path):
    config = make_config(tmp_path, model_epoch_dir=None)
    modules, *_ = make_modules([])

    with pytest.raises(OrthrusRuntimeError, match="neighbor_loader.pkl"):
        run_official_orthrus_smoke_test(config, modules=modules)


def test_runtime_module_has_no_mandatory_orthrus_or_torch_import():
    import adapters.orthrus_runtime as runtime

    assert callable(runtime.run_official_orthrus_smoke_test)


@pytest.mark.parametrize("split,batch_index,expected_prefix", [("val", 0, []), ("test", 1, ["val0", "val1", "test0"])])
@pytest.mark.parametrize("corrupt", [None, "counter", "neighbors"])
def test_temporal_preparation_checks_history_and_never_forwards_target(split, batch_index, expected_prefix, corrupt):
    import numpy as np
    from adapters.orthrus_runtime import _prepare_temporal_batch

    class Array(np.ndarray):
        def cpu(self):
            return self

        def to(self, device):
            return self

    def array(value):
        return np.asarray(value).view(Array)

    class Loader:
        def __init__(self, num_nodes, size, device):
            self.size = size
            self.cur_e_id = 0
            self.e_id = array(np.full((num_nodes, size), -1))
            self.neighbors = array(np.zeros((num_nodes, size), dtype=int))
            self._assoc = array(np.zeros(num_nodes, dtype=int))

        def insert(self, src, dst):
            for s, d in zip(src, dst):
                for node, neighbor in ((s, d), (d, s)):
                    self.e_id[node, 0] = self.cur_e_id
                    self.neighbors[node, 0] = neighbor
                self.cur_e_id += 1

    class Graph:
        def __init__(self, name, offset):
            self.name = name
            self.src, self.dst = array([0, 2]), array([1, 3])
            self.t = array([offset, offset + 1])
            self.msg = array([[offset], [offset + 1]])
            self.edge_type = array([[1, 0], [0, 1]])

        def to(self, device=None):
            return self

    train = Graph("train", 0)
    val0, val1 = Graph("val0", 2), Graph("val1", 4)
    test0, test1 = Graph("test0", 6), Graph("test1", 8)
    test_graph = SimpleNamespace(to=lambda device=None: None)
    splits = {"train": [train], "val": [val0, val1], "test": [test_graph]}
    all_batches = [train, val0, val1, test0, test1]
    history = SimpleNamespace(**{name: array(np.concatenate([getattr(g, name) for g in all_batches]))
                                 for name in ("t", "edge_type", "msg")})
    loader = Loader(4, 1, "cpu")
    loader.insert(train.src, train.dst)
    if corrupt == "counter":
        loader.cur_e_id += 1
    if corrupt == "neighbors":
        loader.neighbors[0, 0] = 3
    reindexer = SimpleNamespace(x_src_cache=None, x_dst_cache=None)

    class Model:
        encoder = SimpleNamespace(neighbor_loader=loader, graph_reindexer=reindexer)
        graph_reindexer = reindexer

        def __init__(self):
            self.calls = []

        def eval(self):
            return self

        def __call__(self, batch, full_data, inference):
            assert inference and full_data is history
            self.calls.append(batch.name)
            loader.insert(batch.src, batch.dst)
            reindexer.x_src_cache = batch.msg.copy()
            reindexer.x_dst_cache = batch.msg.copy()
            return [1.0, 2.0]

    model = Model()
    config = SimpleNamespace(split=split, graph_index=0, batch_index=batch_index, device="cpu", dataset_name="THEIA_E5")
    official = SimpleNamespace(torch=SimpleNamespace(equal=np.array_equal), factory=SimpleNamespace(
        batch_loader_factory=lambda cfg, graph, ri: iter([test0, test1] if graph is test_graph else [graph])))
    if corrupt:
        with pytest.raises(OrthrusRuntimeError, match="Checkpoint"):
            _prepare_temporal_batch(config, official, object(), model, splits, history)
        assert model.calls == []
    else:
        batch, info = _prepare_temporal_batch(config, official, object(), model, splits, history)
        assert model.calls == expected_prefix
        assert batch is (val0 if split == "val" else test1)
        assert info["global_edge_offset"] == loader.cur_e_id == (2 if split == "val" else 8)
        assert info["prefix_batches"] == len(expected_prefix)
        assert (reindexer.x_src_cache is None) == (split == "val")


def test_four_mask_validation_runs_exactly_four_isolated_scores():
    import numpy as np
    from adapters.orthrus_runtime import _validate_four_masks
    from preprocessing.orthrus_alert_case import OrthrusAlertCase

    class Array(np.ndarray):
        device = "cpu"

        def detach(self):
            return self

        def cpu(self):
            return self

        def numel(self):
            return self.size

        def contiguous(self):
            return np.ascontiguousarray(self).view(Array)

        def numpy(self):
            return np.asarray(self)

    def array(value):
        return np.asarray(value).view(Array)

    class History(SimpleNamespace):
        def __iter__(self):
            return iter(vars(self).items())

    class Losses(list):
        def numel(self):
            return len(self)

    class Model:
        def __init__(self):
            self.graph_reindexer = SimpleNamespace(x_src_cache=None, x_dst_cache=None, assoc=None)
            self.encoder = SimpleNamespace(graph_reindexer=self.graph_reindexer, assoc=array([0, 1]),
                neighbor_loader=SimpleNamespace(cur_e_id=10, neighbors=array([[1], [0]]),
                    e_id=array([[8], [9]]), _assoc=array([0, 1])))
            self.calls = []
            self.hook = None

        def eval(self):
            return self

        def register_forward_hook(self, hook):
            self.hook = hook
            return SimpleNamespace(remove=lambda: setattr(self, "hook", None))

        def __call__(self, batch, full_data, inference):
            score = float(batch.x_src.sum() + batch.x_dst.sum() + self.encoder.neighbor_loader.cur_e_id)
            self.calls.append(score)
            self.encoder.neighbor_loader.cur_e_id += len(batch.src)
            self.encoder.neighbor_loader.e_id[...] += 1024
            self.graph_reindexer.x_src_cache = batch.x_src.copy()
            self.graph_reindexer.x_dst_cache = batch.x_dst.copy()
            output = Losses([score] * len(batch.src))
            self.hook(self, (batch, full_data), output)
            return output

    batch = SimpleNamespace(src=array(np.arange(1024) % 2), dst=array((np.arange(1024) + 1) % 2),
                            t=array(np.arange(1024)), edge_type=array(np.ones((1024, 1))),
                            msg=array(np.ones((1024, 2))), x_src=array(np.ones((1024, 1))),
                            x_dst=array(np.ones((1024, 1))))
    batch.edge_index = array(np.stack([batch.src, batch.dst]))
    case = OrthrusAlertCase(batch, full_data=History(t=batch.t.copy(), edge_type=batch.edge_type.copy()))
    model = Model()
    result = _validate_four_masks(SimpleNamespace(device="cpu"), case, model, [])
    assert list(result["evaluations"]) == ["A1", "B", "A2", "Z"]
    assert len(model.calls) == 4 and model.calls[0] == model.calls[2]
    assert model.calls[0] != model.calls[1] != model.calls[3]
    assert all(row["edge_loss_count"] == 1024 for row in result["evaluations"].values())
    assert result["repeatable"] and result["invariants_verified"]
    assert model.encoder.neighbor_loader.cur_e_id == 10
    assert model.graph_reindexer.x_src_cache is None
    assert model.hook is None


def test_public_preparation_returns_before_phase9_evaluations(tmp_path, monkeypatch):
    import adapters.orthrus_runtime as runtime
    config = make_config(tmp_path)
    modules, cfg, graphs, full_data, model, batches = make_modules([])
    modules.config.rel2id = {"EVENT_READ": 1, 1: "EVENT_READ"}
    seen = []

    def prepare(received_config, official, received_cfg, received_model, splits, history):
        assert received_model is model and received_cfg is cfg and history is full_data
        seen.append("temporal_preparation")
        return batches[1], {"global_edge_offset": 123, "batch_index": 1}

    monkeypatch.setattr(runtime, "_prepare_temporal_batch", prepare)
    monkeypatch.setattr(runtime, "_validate_four_masks", lambda *a: pytest.fail("Must not execute A1/B/A2/Z"))
    cwd = Path.cwd()
    case, received_model, relations, warnings = runtime.prepare_official_orthrus_case(config, modules=modules)
    assert seen == ["temporal_preparation"] and model.calls == []
    assert received_model is model and case.temporal_data is batches[1] and case.full_data is full_data
    assert case.metadata["global_edge_offset"] == 123 and relations == modules.config.rel2id
    assert warnings == () and Path.cwd() == cwd
