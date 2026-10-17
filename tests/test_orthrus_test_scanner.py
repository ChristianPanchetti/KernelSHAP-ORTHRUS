import builtins
import csv
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from adapters.orthrus_ano_adapter import RealOrthrusAnoAdapter
from adapters.orthrus_runtime import _iter_official_temporal_batches, OrthrusRuntimeError
from preprocessing.orthrus_alert_case import OrthrusAlertCase
from xai.orthrus_test_scanner import scan_batches, load_ground_truth, suggested_cases, score_statistics


class Array(np.ndarray):
    def cpu(self):
        return self
    def to(self, device):
        return self


def array(values):
    return np.asarray(values).view(Array)


def synthetic():
    class Loader:
        def __init__(self, num_nodes, size, device):
            self.size, self.cur_e_id = size, 0
            self.e_id = array(np.full((num_nodes, size), -1))
            self.neighbors = array(np.zeros((num_nodes, size), dtype=int))
            self._assoc = array(np.zeros(num_nodes, dtype=int))
        def insert(self, src, dst):
            for s, d in zip(src, dst):
                for n, other in ((s, d), (d, s)):
                    self.e_id[n, 0] = self.cur_e_id
                    self.neighbors[n, 0] = other
                self.cur_e_id += 1

    class Graph:
        def __init__(self, offset, scores, nodes):
            self.offset, self.scores = offset, scores
            self.src, self.dst = array(nodes), array([3] * len(nodes))
            self.t = array(range(offset, offset + len(nodes)))
            self.edge_type = array([0] * len(nodes))
        def to(self, device=None):
            return self

    graphs = [Graph(0, [0], [0]), Graph(1, [99], [0]),
              Graph(2, [1., 3., 5., 7.], [1, 1, 2, 0]), Graph(6, [2., 4.], [0, 2])]
    history = SimpleNamespace(**{k: array(np.concatenate([getattr(g, k) for g in graphs])) for k in ("t", "edge_type")})
    loader = Loader(4, 1, "cpu")
    loader.insert(graphs[0].src, graphs[0].dst)
    reindexer = SimpleNamespace(x_src_cache=None, x_dst_cache=None)
    class Model:
        def __init__(self):
            self.calls = []
            self.graph_reindexer = reindexer
            self.encoder = SimpleNamespace(neighbor_loader=loader, graph_reindexer=reindexer)
        def eval(self):
            return self
        def __call__(self, batch, full_data, inference):
            assert loader.cur_e_id == batch.offset
            assert full_data is history and inference
            assert (reindexer.x_src_cache is None) == (len(self.calls) == 0)
            self.calls.append(batch.offset)
            loader.insert(batch.src, batch.dst)
            reindexer.x_src_cache = batch.offset
            return batch.scores
    model = Model()
    config = SimpleNamespace(device="cpu", dataset_name="THEIA_E5")
    official = SimpleNamespace(torch=SimpleNamespace(equal=np.array_equal), factory=SimpleNamespace(
        batch_loader_factory=lambda cfg, graph, ri: iter([graph])))
    splits = {"train": graphs[:1], "val": graphs[1:2], "test": graphs[2:]}
    return config, official, model, splits, history


def test_full_temporal_scan_statistics_node_metadata_and_cases(tmp_path):
    config, official, model, splits, history = synthetic()
    sequence = _iter_official_temporal_batches(config, official, object(), model, splits, history)
    path = tmp_path / "scan.csv"
    summary = scan_batches(sequence, RealOrthrusAnoAdapter(model), path, {1}, {"ground_truth_level": "node"})
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2
    assert model.calls == [1, 2, 6]  # all validation then each test once
    assert model.encoder.neighbor_loader.cur_e_id == 8
    assert model.graph_reindexer.x_src_cache == 6  # never restored
    assert [int(r["global_edge_offset"]) for r in rows] == [2, 6]
    assert [int(r["graph_index"]) for r in rows] == [0, 1]
    assert float(rows[0]["mean_anomaly_score"]) == 4
    assert float(rows[0]["min_edge_score"]) == 1
    assert float(rows[0]["median_edge_score"]) == 4
    assert float(rows[0]["p95_edge_score"]) == pytest.approx(6.7)
    assert float(rows[0]["max_edge_score"]) == 7
    assert int(rows[0]["malicious_node_count"]) == 1
    assert int(rows[0]["malicious_src_node_count"]) == 1
    assert int(rows[0]["malicious_dst_node_count"]) == 0
    assert "malicious_edges" not in rows[0]
    assert summary["total_test_edges_scanned"] == 6
    assert summary["batch_mean_scores"] == score_statistics([4, 3])
    assert summary["batch_mean_scores"]["percentiles"]["p95"] == pytest.approx(3.95)
    assert summary["batches_containing_malicious_nodes"] == 1
    assert summary["descriptive_groups"]["containing_malicious_nodes"]["batch_mean_scores"]["mean"] == 4
    assert summary["suggested_cases"]["highest_mean_score"]["graph_index"] == 0
    assert json.loads((tmp_path / "scan_summary.json").read_text()) == summary


def test_temporal_sequence_rejects_missing_forward():
    config, official, model, splits, history = synthetic()
    sequence = _iter_official_temporal_batches(config, official, object(), model, splits, history)
    next(sequence)
    with pytest.raises(OrthrusRuntimeError, match="exactly once"):
        next(sequence)


def test_official_export_validation_and_derived_metadata(tmp_path):
    path = tmp_path / "nodes.json"
    data = {"dataset": "THEIA_E5", "ground_truth_level": "node", "node_ids": [1, 1, 3],
            "official_label_function": "labelling.get_ground_truth", "ground_truth_source": "official CSV + official UUID lookup"}
    path.write_text(json.dumps(data))
    nodes, metadata = load_ground_truth(path, "THEIA_E5")
    assert nodes == {1, 3}
    assert metadata["ground_truth_level"] == "node"
    assert all("derived" in v for v in metadata["batch_fields"].values())
    assert "malicious_edges" not in metadata["batch_fields"]
    with pytest.raises(ValueError):
        load_ground_truth(path, "OTHER")
    data["ground_truth_level"] = "edge"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_ground_truth(path, "THEIA_E5")


def test_cases_missing_categories_ties_and_empty():
    row = dict(dataset="THEIA_E5", split="test", graph_index=0, batch_index=0, global_edge_offset=0,
               mean_anomaly_score=2, malicious_node_count=0, contains_malicious_node=False,
               malicious_src_node_count=0, malicious_dst_node_count=0)
    rows = [row, {**row, "batch_index": 1}]
    cases = suggested_cases(rows)
    assert cases == suggested_cases(rows)
    assert cases["median_score"]["batch_index"] == 0
    assert not cases["low_score_with_malicious_nodes"]["available"]
    assert all(not c["available"] for c in suggested_cases([]).values())
    assert score_statistics([]) is None


def test_partial_csv_no_success_summary_on_failure(tmp_path):
    config, official, model, splits, history = synthetic()
    sequence = _iter_official_temporal_batches(config, official, object(), model, splits, history)
    adapter = RealOrthrusAnoAdapter(model)
    real = adapter.predict_edge_losses
    def fail(case):
        if case.metadata["graph_index"] == 1:
            raise RuntimeError("interrupted")
        return real(case)
    adapter.predict_edge_losses = fail
    path = tmp_path / "scan.csv"
    summary_path = tmp_path / "scan_summary.json"
    summary_path.write_text("stale")
    with pytest.raises(RuntimeError, match="interrupted"):
        scan_batches(sequence, adapter, path, set(), {})
    assert len(list(csv.DictReader(path.open()))) == 1
    assert not summary_path.exists()


def test_scan_cli_does_not_import_shap_pipeline_or_mapping(monkeypatch, tmp_path):
    import main
    import xai.orthrus_test_scanner as scanner
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name not in ("pipeline", "shap", "psycopg2", "preprocessing.orthrus_mapping", "xai.kernel_shap_explainer")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    calls = []
    monkeypatch.setattr(scanner, "run_test_scan", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "orthrus", "--adapter", "real", "--orthrus-config", "runtime.json",
                                      "--scan-test", "--scan-ground-truth", "nodes.json", "--scan-output", str(tmp_path / "scan.csv")])
    assert main.main() == 0
    assert len(calls) == 1


def test_cli_requires_offline_official_labels(monkeypatch):
    from arg_parser import parse_args
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "orthrus", "--orthrus-config", "runtime.json", "--scan-test"])
    with pytest.raises(SystemExit):
        parse_args()


def test_server_entrypoint_reuses_loading_ignores_single_target_and_preserves_output_path(tmp_path, monkeypatch):
    from adapters import orthrus_runtime as runtime
    from xai.orthrus_test_scanner import run_test_scan
    from test_orthrus_official_runtime import make_config

    config, official, model, splits, history = synthetic()
    cfg = SimpleNamespace(_ground_truth_dir="ground_truth/darpa", _test_mode=False,
        dataset=SimpleNamespace(ground_truth_relative_path=["E5-THEIA/official.csv"]),
        detection=SimpleNamespace(evaluation=SimpleNamespace(ground_truth_version="darpa")))
    seen = []
    modules = runtime.OfficialOrthrusModules(
        config=SimpleNamespace(get_runtime_required_args=lambda **k: k, get_yml_cfg=lambda a: cfg),
        data_utils=SimpleNamespace(load_all_datasets=lambda c: (splits["train"], splits["val"], splits["test"], history, 4),
                                   load_model=lambda m, p: seen.append("checkpoint") or m),
        factory=SimpleNamespace(build_model=lambda **k: seen.append(k["data_sample"]) or model,
                                batch_loader_factory=official.factory.batch_loader_factory), torch=official.torch)
    real = runtime.run_official_orthrus_smoke_test
    monkeypatch.setattr(runtime, "run_official_orthrus_smoke_test", lambda c, **k: real(c, modules=modules, **k))
    settings = make_config(tmp_path, dataset_name="THEIA_E5", split="val", graph_index=999, batch_index=999)
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps({"external_root": str(settings.external_root), "config_path": str(settings.config_path),
        "model_epoch_dir": str(settings.model_epoch_dir), "dataset_name": "THEIA_E5",
        "split": "val", "graph_index": 999, "batch_index": 999}))
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"dataset": "THEIA_E5", "ground_truth_level": "node", "node_ids": [1],
        "official_label_function": "labelling.get_ground_truth", "ground_truth_source": "official export",
        "ground_truth_version": "darpa", "ground_truth_relative_path": ["E5-THEIA/official.csv"]}))
    monkeypatch.chdir(tmp_path)
    summary = run_test_scan(config_path, "results/scan.csv", labels)
    assert summary["total_test_batches"] == 2
    assert (tmp_path / "results/scan.csv").is_file()
    from pathlib import Path
    assert Path.cwd() == tmp_path
    assert model.calls == [1, 2, 6]
    assert seen == [splits["test"][0], "checkpoint"]
