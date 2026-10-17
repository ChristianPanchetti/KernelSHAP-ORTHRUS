import builtins
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import export_scan_ground_truth as exporter
from xai.orthrus_test_scanner import load_ground_truth


@pytest.fixture
def official_setup(tmp_path, monkeypatch):
    root = tmp_path / "orthrus"
    (root / "src").mkdir(parents=True)
    for name in ("config.py", "labelling.py"):
        (root / "src" / name).write_text("# mocked official module\n")
    yaml = root / "orthrus.yml"
    yaml.write_text("mock")
    labels = root / "labels" / "E5-THEIA" / "official.csv"
    labels.parent.mkdir(parents=True)
    labels.write_text("uuid,label,extra\n")
    cfg = SimpleNamespace(dataset=SimpleNamespace(name="THEIA_E5",
        ground_truth_relative_path=["E5-THEIA/official.csv"]), _ground_truth_dir="labels",
        detection=SimpleNamespace(evaluation=SimpleNamespace(ground_truth_version="darpa")))
    calls = []
    official_config = SimpleNamespace(
        get_runtime_required_args=lambda **kwargs: calls.append(("args", kwargs)) or kwargs,
        get_yml_cfg=lambda args: cfg)
    def get_ground_truth(received):
        assert received is cfg and Path.cwd() == root
        calls.append("official_helper")
        return [9, np.int64(2), 9, 2**60 + 1], {}, {}
    labelling = SimpleNamespace(get_ground_truth=get_ground_truth)
    @contextmanager
    def modules(received_root):
        assert received_root == root
        yield official_config, labelling
    monkeypatch.setattr(exporter, "_official_label_modules", modules)
    settings = {"external_root": str(root), "config_path": str(yaml), "dataset_name": "THEIA_E5",
                "model_epoch_dir": "not-needed-for-export", "overrides": {"database.host": "server-db"}}
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps(settings))
    return SimpleNamespace(root=root, cfg=cfg, calls=calls, helper=labelling, labels=labels,
                           config_path=config_path, settings=settings)


def test_official_output_numeric_sorted_deduplicated_metadata_and_scanner_compatibility(official_setup, tmp_path, monkeypatch):
    setup = official_setup
    monkeypatch.chdir(tmp_path)
    payload = exporter.export_from_config(setup.config_path, "outputs/labels.json")
    path = tmp_path / "outputs/labels.json"
    assert json.loads(path.read_text()) == payload
    assert payload["node_ids"] == [2, 9, 2**60 + 1]
    assert all(type(n) is int for n in payload["node_ids"])
    assert payload["malicious_node_count"] == 3
    assert payload["dataset"] == "THEIA_E5"
    assert payload["ground_truth_level"] == "node"
    assert payload["official_label_function"] == "labelling.get_ground_truth"
    assert payload["ground_truth_files"] == [str(setup.labels)]
    assert payload["ground_truth_relative_path"] == setup.cfg.dataset.ground_truth_relative_path
    assert payload["ground_truth_version"] == "darpa"
    assert setup.calls == [("args", {"args": ["THEIA_E5", "--cpu", "--database.host=server-db"]}), "official_helper"]
    nodes, _ = load_ground_truth(path, "THEIA_E5")
    assert nodes == {2, 9, 2**60 + 1}
    assert Path.cwd() == tmp_path


@pytest.mark.parametrize("failure", ["db", "missing_file", "dataset", "loaded_dataset", "invalid_ids"])
def test_clear_failures_no_output_and_cwd_restored(official_setup, tmp_path, failure):
    setup = official_setup
    original = Path.cwd()
    if failure == "db":
        def fail(cfg):
            raise ConnectionError("secret-password")
        setup.helper.get_ground_truth = fail
        message = "PostgreSQL connectivity"
    elif failure == "missing_file":
        setup.labels.unlink()
        message = "ground-truth file is missing"
    elif failure == "dataset":
        setup.settings["dataset_name"] = "OTHER"
        setup.config_path.write_text(json.dumps(setup.settings))
        message = "dataset_name=THEIA_E5"
    elif failure == "loaded_dataset":
        setup.cfg.dataset.name = "OTHER"
        message = "not THEIA_E5"
    else:
        setup.helper.get_ground_truth = lambda cfg: ([1.5], {}, {})
        message = "invalid numeric"
    output = tmp_path / "output.json"
    with pytest.raises(exporter.GroundTruthExportError, match=message) as error:
        exporter.export_from_config(setup.config_path, output)
    assert "secret-password" not in str(error.value)
    assert not output.exists()
    assert Path.cwd() == original


def test_cli_export_has_no_inference_shap_or_mapping_imports(official_setup, tmp_path, monkeypatch, capsys):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name not in {"pipeline", "factory", "data_utils", "shap", "preprocessing.orthrus_mapping",
                            "xai.kernel_shap_explainer"}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    output = tmp_path / "export.json"
    assert exporter.main(["--orthrus-config", str(official_setup.config_path), "--output", str(output)]) == 0
    assert "Exported 3" in capsys.readouterr().out
    assert official_setup.calls.count("official_helper") == 1


def test_cli_failure_status(official_setup, tmp_path, capsys):
    official_setup.labels.unlink()
    assert exporter.main(["--orthrus-config", str(official_setup.config_path),
                          "--output", str(tmp_path / "export.json")]) == 1
    assert "ground-truth file is missing" in capsys.readouterr().err


def test_flat_official_imports_use_requested_root_and_restore_project_modules(tmp_path):
    root = tmp_path / "external"
    src = root / "src"
    src.mkdir(parents=True)
    (src / "config.py").write_text("DATASET = 'THEIA_E5'\n")
    (src / "provnet_utils.py").write_text("from config import DATASET\n")
    (src / "labelling.py").write_text("from provnet_utils import DATASET\ndef get_ground_truth(cfg):\n    return {7}, {}, {}\n")
    previous_path = sys.path[:]
    previous = {name: sys.modules.get(name) for name in ("config", "labelling", "provnet_utils")}
    with exporter._official_label_modules(root) as (config_module, labelling):
        assert config_module.DATASET == labelling.DATASET == "THEIA_E5"
        assert labelling.get_ground_truth(None)[0] == {7}
        assert Path(labelling.__file__).resolve() == src / "labelling.py"
    assert sys.path == previous_path
    assert all(sys.modules.get(name) is module for name, module in previous.items())
