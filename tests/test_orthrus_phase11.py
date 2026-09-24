import json
import logging
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import pipeline
from config import AppConfig, KernelSHAPConfig, PerturbationConfig
from preprocessing.orthrus_alert_case import OrthrusAlertCase
from preprocessing.orthrus_mapping import DbAssistedMappingProvider
from perturbation.orthrus_interpretable_builder import OrthrusInterpretableBuilder
from xai.kernel_shap_explainer import KernelSHAPExplainer


def prepared_case():
    batch = SimpleNamespace(src=np.array([20, 3]), dst=np.array([9, 9]), t=np.array([10, 11]),
                            x_src=np.ones((2, 1)), x_dst=np.ones((2, 1)),
                            edge_type=["EVENT_READ", "EVENT_WRITE"])
    history = SimpleNamespace(edge_type=np.eye(2))
    case = OrthrusAlertCase(batch, full_data=history, metadata={"dataset": "THEIA_E5", "split": "test",
        "graph_index": 0, "batch_index": 0, "global_edge_offset": 7})
    loader = SimpleNamespace(cur_e_id=7, neighbors=np.array([[1]]), e_id=np.array([[6]]))
    reindexer = SimpleNamespace(x_src_cache=None, x_dst_cache=None)

    class Model:
        def __init__(self):
            self.calls = []
            self.encoder = SimpleNamespace(neighbor_loader=loader, graph_reindexer=reindexer)

        def eval(self):
            return self

        def __call__(self, batch, full_data, inference):
            assert loader.cur_e_id == 7 and reindexer.x_src_cache is None
            assert full_data is history and inference
            # Repeated destination induces a real overlap between the two interventions.
            value = 5 + 2 * batch.x_src[0, 0] - 3 * batch.x_src[1, 0] + 4 * batch.x_dst[:, 0].min()
            self.calls.append(value)
            loader.cur_e_id += 2
            loader.e_id[:] = 99
            reindexer.x_src_cache = np.ones((2, 1))
            return [float(value)] * len(batch.src)

    return case, Model()


def configuration(tmp_path):
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps(dict(external_root="unused", config_path="unused", dataset_name="THEIA_E5",
                                      model_epoch_dir="unused", split="test", graph_index=0, batch_index=0)), encoding="utf-8")
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps({"events": [], "nodes": [
        {"index_id": 20, "node_uuid": "process-20", "node_type": "subject", "path": "/bin/shell", "cmd": "shell"},
        {"index_id": 3, "node_uuid": "file-3", "node_type": "file", "path": "/tmp/input"},
        {"index_id": 3, "node_uuid": "conflict-3", "node_type": "subject", "path": "/tmp/other"},
    ]}), encoding="utf-8")
    return AppConfig(input_path=tmp_path / "unused", output_json_path=tmp_path / "out.json",
        adapter_type="real", pipeline_mode="orthrus", orthrus_config_path=runtime,
        mapping_backend="offline", mapping_rows_path=rows,
        perturbation=PerturbationConfig(grouping_mode="node", perturbation_mode="neutralize_edges", max_components=8),
        shap=KernelSHAPConfig(num_samples=16, seed=4, top_k=2))


def test_pipeline_real_shap_simulated_model_overlap_mapping_and_export(tmp_path, monkeypatch):
    pytest.importorskip("shap")
    case, model = prepared_case()
    cfg = configuration(tmp_path)
    monkeypatch.setattr(pipeline, "prepare_official_orthrus_case", lambda config: (case, model, {}, ()))
    adapters, masks, enrichments = [], [], []
    real_adapter = pipeline.RealOrthrusAnoAdapter

    class TrackedAdapter(real_adapter):
        def __init__(self, *args, **kwargs):
            assert kwargs["isolate_state"] is True
            super().__init__(*args, **kwargs)
            adapters.append(self)

        def score_mask(self, manager, mask):
            assert not enrichments
            masks.append((id(self), id(manager), tuple(mask)))
            return super().score_mask(manager, mask)

    class TrackedProvider(DbAssistedMappingProvider):
        def enrich_case(self, received, **kwargs):
            assert len(model.calls) == 4
            enrichments.append(received)
            return super().enrich_case(received, **kwargs)

    monkeypatch.setattr(pipeline, "RealOrthrusAnoAdapter", TrackedAdapter)
    monkeypatch.setattr(pipeline, "DbAssistedMappingProvider", TrackedProvider)
    result = pipeline.run_kernel_shap_pipeline(cfg, logging.getLogger("test"))
    assert result.metadata["component_order"] == ["node:20", "node:3"]
    assert result.shap_values == pytest.approx([4, -1])
    assert result.baseline_score == 5 and result.original_score == 8
    assert result.metadata["expected_value"] == 5 and abs(result.metadata["reconstruction_residual"]) < 1e-8
    assert result.metadata["evaluated_unique_masks"] == 4
    assert len(adapters) == len(enrichments) == 1
    assert len({m[:2] for m in masks}) == 1
    assert {m[2] for m in masks} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert model.encoder.neighbor_loader.cur_e_id == 7
    assert model.encoder.neighbor_loader.e_id.tolist() == [[6]]
    assert model.encoder.graph_reindexer.x_src_cache is None
    assert np.all(case.temporal_data.x_src == 1) and np.all(case.temporal_data.x_dst == 1)
    assert np.array_equal(case.full_data.edge_type, np.eye(2))
    payload = json.loads(cfg.output_json_path.read_text(encoding="utf-8"))
    assert payload["components"][0]["component_id"] == "node:20"
    assert "/bin/shell" in payload["components"][0]["description"]
    assert "/bin/shell" in payload["explanation_it"] and "/bin/shell" in payload["explanation_en"]
    assert payload["metadata"]["mapping_quality"]["nodes"]["missing"] == 1
    assert payload["metadata"]["mapping_quality"]["nodes"]["ambiguous"] == 1
    assert "node_uuid" not in payload["metadata"]["case"]["node_mapping"]["3"]
    assert payload["components"][0]["sign"] == "positive" and payload["components"][1]["sign"] == "negative"
    assert payload["metadata"]["case"]["node_mapping"]["20"]["provenance"] == "offline_rows"
    assert payload["metadata"]["case"]["component_mapping"]["node:20"]["destination_rows_if_disabled"] == [0, 1]


def test_other_single_component_and_invalid_spaces():
    case, _ = prepared_case()
    case.temporal_data.src = np.array([20, 20])
    info = OrthrusInterpretableBuilder(grouping_mode="node", max_components=1).build(case)
    assert pipeline._orthrus_space(case, info, 1).num_components == 1
    case.component_to_edges = {"node:20": [0], "OTHER": [1]}
    space = pipeline._orthrus_space(case, info, 2)
    assert space.components[1].component_id == "OTHER" and space.components[1].kind == "node_group"
    with pytest.raises(ValueError, match="limit 1"):
        pipeline._orthrus_space(case, info, 1)
    case.component_to_edges = {"node:20": [0, 1], "empty": []}
    with pytest.raises(ValueError, match="nonempty"):
        pipeline._orthrus_space(case, info, 8)


def test_other_remains_aggregate_in_complete_export(tmp_path, monkeypatch):
    pytest.importorskip("shap")
    case, model = prepared_case()
    case.temporal_data.src = np.array([20, 3, 5])
    case.temporal_data.dst = np.array([9, 9, 9])
    case.temporal_data.t = np.array([10, 11, 12])
    case.temporal_data.edge_type = ["EVENT_READ"] * 3
    case.temporal_data.x_src = np.ones((3, 1))
    case.temporal_data.x_dst = np.ones((3, 1))
    cfg = configuration(tmp_path)
    cfg = replace(cfg, perturbation=replace(cfg.perturbation, max_components=2))
    monkeypatch.setattr(pipeline, "prepare_official_orthrus_case", lambda config: (case, model, {}, ()))
    result = pipeline.run_kernel_shap_pipeline(cfg, logging.getLogger("test"))
    assert result.metadata["component_order"] == ["node:20", "OTHER"]
    assert result.components[1].name == "OTHER (2 source nodes)"
    assert result.metadata["component_edges"]["OTHER"] == [1, 2]
    assert len(result.shap_values) == 2


def test_single_component_uses_same_scorer_and_nonzero_baseline():
    pytest.importorskip("shap")
    case, _ = prepared_case()
    case.temporal_data.src[:] = 20
    info = OrthrusInterpretableBuilder(grouping_mode="node").build(case)
    space = pipeline._orthrus_space(case, info, 1)
    calls = []
    def score(mask):
        calls.append(mask)
        return 10 - 2 * mask[0]
    result = KernelSHAPExplainer().explain(space, SimpleNamespace(mode="neutralize_edges"), None, mask_scorer=score)
    assert result.shap_values == pytest.approx([-2])
    assert result.baseline_score == 10 and result.original_score == 8 and len(calls) == 2


@pytest.mark.parametrize("values,expected,raises", [([float("nan"), 0], 5, True), ([3], 5, True),
                                                   ([3, 0], float("inf"), True), ([0, 0], 5, False)])
def test_invalid_backend_outputs_and_visible_residual(monkeypatch, values, expected, raises):
    class Backend:
        def __init__(self, model, background):
            self.model = model
            model(background)
            self.expected_value = expected
        def shap_values(self, x, **kwargs):
            self.model(x)
            return np.array([values])
    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(KernelExplainer=Backend))
    case, _ = prepared_case()
    info = OrthrusInterpretableBuilder(grouping_mode="node").build(case)
    space = pipeline._orthrus_space(case, info, 8)
    run = lambda: KernelSHAPExplainer().explain(space, SimpleNamespace(mode="neutralize_edges"), None,
                                               mask_scorer=lambda m: 5 + 3 * m[0])
    if raises:
        with pytest.raises(ValueError):
            run()
    else:
        result = run()
        assert result.metadata["reconstruction_residual"] == 3 and result.metadata["warnings"]


def test_nonfinite_scorer_is_rejected_before_backend():
    case, _ = prepared_case()
    info = OrthrusInterpretableBuilder(grouping_mode="node").build(case)
    with pytest.raises(ValueError, match="non-finite"):
        KernelSHAPExplainer().explain(pipeline._orthrus_space(case, info, 8), SimpleNamespace(mode="neutralize_edges"),
                                     None, mask_scorer=lambda m: float("nan"))


def test_required_mapping_failure_is_not_exported_as_complete(tmp_path, monkeypatch):
    pytest.importorskip("shap")
    case, model = prepared_case()
    cfg = configuration(tmp_path)
    cfg = replace(cfg, mapping_backend="postgresql", mapping_rows_path=None)
    monkeypatch.setattr(pipeline, "prepare_official_orthrus_case", lambda config: (case, model, {}, ()))
    class FailedProvider:
        def __init__(self, **kwargs):
            pass
        def enrich_case(self, *args, **kwargs):
            raise RuntimeError("Mapping unavailable")
    monkeypatch.setattr(pipeline, "DbAssistedMappingProvider", FailedProvider)
    with pytest.raises(RuntimeError, match="Mapping unavailable"):
        pipeline.run_kernel_shap_pipeline(cfg, logging.getLogger("test"))
    assert not cfg.output_json_path.exists()


def test_cli_mode_defaults_and_explicit_legacy_options(tmp_path, monkeypatch):
    from arg_parser import parse_args
    from main import build_config_from_args
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "orthrus", "--orthrus-config", "case.json"])
    cfg = build_config_from_args(parse_args())
    assert cfg.adapter_type == "real" and cfg.perturbation.grouping_mode == "node"
    assert cfg.perturbation.max_components == 8 and cfg.perturbation.perturbation_mode == "neutralize_edges"
    with pytest.raises(ValueError, match="node grouping"):
        pipeline._run_orthrus_pipeline(replace(cfg, perturbation=PerturbationConfig()), logging.getLogger("test"))
    monkeypatch.setattr(sys, "argv", ["main.py"])
    legacy = build_config_from_args(parse_args())
    assert legacy.adapter_type == "dummy" and legacy.perturbation.max_components == 50
    assert legacy.perturbation.grouping_mode == "exec_path"
