import copy
import json
import math
import sys

import pytest

from xai.result_visualizer import _data, build_figures, export_plots, export_plots_from_json


def sample():
    return {
        "baseline_score": 5., "original_anomaly_score": 1.25,
        "components": [
            {"component_id": "node:20", "shap_value": 2., "name": "UNTRUSTED"},
            {"component_id": "node:3", "shap_value": -7.},
            {"component_id": "OTHER", "shap_value": 1.},
        ],
        "metadata": {"case": {
            "component_mapping": {"node:20": {"source_node_ids": [20]}, "node:3": {"source_node_ids": [3]}},
            "node_mapping": {"20": {"status": "resolved", "provenance": "offline_rows", "path": "/bin/verified"},
                             "3": {"status": "ambiguous", "path": "/bin/UNTRUSTED"}},
        }},
    }


def test_order_values_labels_and_other():
    payload = sample()
    before = copy.deepcopy(payload)
    labels, values, baseline, original = _data(payload)
    assert values == [-7., 2., 1.]
    assert labels == ["node:3", "node:20\n/bin/verified", "OTHER (aggregata)"]
    assert baseline == 5. and original == 1.25
    assert payload == before
    class Result:
        def to_dict(self):
            return payload
    assert _data(Result()) == _data(payload)


@pytest.mark.parametrize("status", ["missing", "ambiguous", "conflict", "unverified", "unresolvable"])
def test_untrusted_identity_falls_back(status):
    payload = sample()
    payload["metadata"]["case"]["node_mapping"]["20"]["status"] = status
    assert _data(payload)[0][1] == "node:20"


def test_unknown_provenance_and_unverified_field():
    payload = sample()
    node = payload["metadata"]["case"]["node_mapping"]["20"]
    node["provenance"] = "unverified"
    assert _data(payload)[0][1] == "node:20"
    node["provenance"] = "postgresql"
    node["unverified_fields"] = ["path"]
    assert _data(payload)[0][1] == "node:20"
    payload["metadata"] = {}
    assert _data(payload)[0][1] == "node:20"


def test_many_components_and_long_labels():
    payload = sample()
    payload["metadata"]["case"]["node_mapping"]["20"]["path"] = "/" + "a" * 500 + "/tail"
    payload["components"] += [{"component_id": f"node:extra{i}", "shap_value": 0.} for i in range(40)]
    labels, values, _, _ = _data(payload)
    assert len(labels) == len(values) == 43
    assert "…" in labels[1] and "tail" in labels[1]
    assert max(map(len, labels[1].splitlines())) <= 40


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_reject_nonfinite(value):
    payload = sample()
    payload["components"][0]["shap_value"] = value
    with pytest.raises(ValueError, match="finite"):
        _data(payload)


def test_reject_empty_and_duplicates():
    payload = sample()
    payload["components"].append(payload["components"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        _data(payload)
    payload["components"] = []
    with pytest.raises(ValueError, match="empty"):
        _data(payload)


def test_figures_signed_bars_waterfall_and_layout():
    pytest.importorskip("matplotlib")
    bar, waterfall = build_figures(sample())
    ax = bar.axes[0]
    assert [p.get_width() for p in ax.patches] == [-7., 2., 1.]
    assert ax.patches[0].get_facecolor() != ax.patches[1].get_facecolor()
    axw = waterfall.axes[0]
    assert len(axw.patches) == 6
    assert [p.get_x() for p in axw.patches[1:4]] == [5., -2., 0.]
    assert axw.patches[-2].get_width() == 1.
    assert axw.patches[-1].get_width() == 1.25
    assert "+0.25" in axw.get_title()
    for fig in (bar, waterfall):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        assert not fig._supxlabel.get_window_extent(renderer).overlaps(
            fig.axes[0].xaxis.label.get_window_extent(renderer))
        for label in fig.axes[0].get_yticklabels():
            assert fig.bbox.contains(*label.get_window_extent(renderer).get_points()[0])
            assert fig.bbox.contains(*label.get_window_extent(renderer).get_points()[1])


def test_export_from_result_and_json_without_runtime(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    # Fail even an attempted import of the computation/database paths.
    for module in ("pipeline", "shap", "psycopg2", "adapters.orthrus_runtime", "xai.kernel_shap_explainer"):
        monkeypatch.setitem(sys.modules, module, None)
    path = tmp_path / "case.json"
    path.write_text(json.dumps(sample()), encoding="utf-8")
    before = path.read_bytes()
    paths = export_plots_from_json(path)
    assert len(paths) == 9
    for target in paths:
        if target.suffix == ".csv":
            assert "SHAP value" in target.read_text(encoding="utf-8")
            continue
        assert target.stat().st_size > 1000
        assert target.read_bytes().startswith(b"%PDF" if target.suffix == ".pdf" else b"\x89PNG")
    assert path.read_bytes() == before
    class Result:
        def to_dict(self):
            return sample()
    assert export_plots(Result(), path) == paths


def test_cli_json_branch_does_not_run_pipeline(tmp_path, monkeypatch):
    import main
    import xai.result_visualizer as visualizer
    called = []
    monkeypatch.setitem(sys.modules, "pipeline", None)
    monkeypatch.setattr(sys, "argv", ["main.py", "--plot-json", str(tmp_path / "case.json")])
    monkeypatch.setattr(visualizer, "export_plots_from_json", lambda path: called.append(path) or [])
    assert main.main() == 0
    assert called == [str(tmp_path / "case.json")]


def rich_sample():
    payload = sample()
    case = payload["metadata"]["case"]
    case["node_mapping"]["20"]["node_type"] = "file"
    case["node_mapping"]["99"] = {"status": "resolved", "provenance": "postgresql",
                                     "node_type": "subject", "path": "/usr/bin/exporter"}
    case["component_mapping"]["node:20"]["assigned_edges"] = [0, 1, 2]
    case["component_mapping"]["OTHER"] = {"source_node_ids": [7, 8], "assigned_edges": [3]}
    payload["metadata"]["edge_to_original_metadata"] = {
        str(i): {"src_index_id": src, "dst_index_id": dst, "operation": operation, "status": "resolved"}
        for i, src, dst, operation in [(0, 20, 99, "EVENT_READ"), (1, 20, 99, "EVENT_READ"),
                                        (2, 99, 20, "EVENT_OPEN"), (3, 7, 99, "EVENT_WRITE")]}
    return payload


def test_rich_summary_aggregate_and_direction(tmp_path):
    import csv
    from xai.provenance_visualizer import summarize, build_provenance_figure
    payload = rich_sample()
    before = copy.deepcopy(payload)
    records = summarize(payload)
    row = records[1]["row"]
    assert row["SHAP value"] == 2.
    assert row["Assigned batch edges"] == 3
    assert row["Main operations"] == "EVENT_READ (2), EVENT_OPEN (1)"
    assert "/usr/bin/exporter [subject; id=99] (3)" == row["Main counterpart entities"]
    assert records[1]["relations"][("99", "in")] == {"EVENT_OPEN": 1}
    aggregate = [r for r in records if r["component_id"] == "OTHER"]
    assert len(aggregate) == 1
    assert aggregate[0]["row"]["SHAP value"] == 1.
    assert aggregate[0]["row"]["Component / semantic name"] == "OTHER (2 source nodes)"
    assert len(records) == 3 and payload == before
    fig = build_provenance_figure(records)
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert sum("SHAP =" in label for label in labels) == 3
    assert not any("node:7 |" in label or "node:8 |" in label for label in labels)
    fig.clear()
    path = tmp_path / "rich.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert len(export_plots_from_json(path)) == 9
    with path.with_name("rich_summary_table.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3 and rows[1]["SHAP value"] == "2.0"


def test_summary_absolute_values_unsigned_and_shap_signed():
    from xai.provenance_visualizer import COLUMNS, build_summary_figure, summarize
    records = summarize(sample())
    fig = build_summary_figure(records)
    try:
        cells = fig.axes[0].tables[0].get_celld()
        signed_column = COLUMNS.index("SHAP value")
        absolute_column = COLUMNS.index("|SHAP|")
        assert [cells[row, signed_column].get_text().get_text() for row in (1, 2, 3)] == ["-7", "+2", "+1"]
        assert [cells[row, absolute_column].get_text().get_text() for row in (1, 2, 3)] == ["7", "2", "1"]
    finally:
        fig.clear()


def test_no_mapping_and_unverified_netflow():
    from xai.provenance_visualizer import summarize, build_provenance_figure
    payload = sample()
    payload["metadata"] = {}
    records = summarize(payload)
    assert len(records) == 3
    fig = build_provenance_figure(records)
    fig.canvas.draw()
    fig.clear()
    payload = rich_sample()
    payload["metadata"]["case"]["node_mapping"]["99"].update(node_type="netflow", path="UNVERIFIED_ENDPOINT")
    assert "UNVERIFIED_ENDPOINT" not in summarize(payload)[1]["row"]["Main counterpart entities"]


def test_actual_offline_cli_without_computation_imports(tmp_path, monkeypatch):
    import main
    path = tmp_path / "offline.json"
    path.write_text(json.dumps(rich_sample()), encoding="utf-8")
    for module in ("pipeline", "shap", "psycopg2", "torch", "adapters.orthrus_runtime",
                   "preprocessing.orthrus_mapping", "xai.kernel_shap_explainer"):
        monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.setattr(sys, "argv", ["main.py", "--plot-json", str(path)])
    assert main.main() == 0
    assert len(list(tmp_path.glob("offline_*.png"))) == 4
    assert len(list(tmp_path.glob("offline_*.pdf"))) == 4
    assert path.with_name("offline_summary_table.csv").exists()


@pytest.mark.parametrize("count", [1, 8, 30])
def test_render_negative_reconstruction_and_long_labels(count):
    pytest.importorskip("matplotlib")
    payload = sample()
    payload["baseline_score"] = -3.
    payload["components"] = [{"component_id": f"node:{i}", "shap_value": -1.} for i in range(count)]
    payload["metadata"] = {}
    bar, waterfall = build_figures(payload)
    assert len(bar.axes[0].patches) == count
    assert waterfall.axes[0].patches[-2].get_width() == -3. - count
    for fig in (bar, waterfall):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for label in fig.axes[0].get_yticklabels():
            bounds = label.get_window_extent(renderer)
            assert fig.bbox.contains(*bounds.get_points()[0])
            assert fig.bbox.contains(*bounds.get_points()[1])
