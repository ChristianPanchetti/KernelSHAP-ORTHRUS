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

