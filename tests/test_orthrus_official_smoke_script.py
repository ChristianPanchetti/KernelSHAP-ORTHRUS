from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from adapters.orthrus_runtime import OrthrusOfficialRuntimeConfig
from scripts import run_orthrus_official_smoke as smoke_script


def valid_payload() -> dict:
    return {
        "external_root": "external/orthrus",
        "config_path": "external/orthrus/config/orthrus.yml",
        "dataset_name": "THEIA_E3",
        "model_epoch_dir": "artifacts/models/model_epoch_1",
        "split": "test",
        "graph_index": 2,
        "batch_index": 3,
        "device": "cpu",
        "from_weights_path": None,
        "overrides": {"detection.gnn_training.encoder.batch_size": 64},
    }


def test_json_is_read_and_official_config_is_built(tmp_path: Path):
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(valid_payload()), encoding="utf-8")

    config = smoke_script.build_config(smoke_script.load_config(path))

    assert isinstance(config, OrthrusOfficialRuntimeConfig)
    assert config.external_root == Path("external/orthrus")
    assert config.dataset_name == "THEIA_E3"
    assert config.graph_index == 2
    assert config.batch_index == 3


def test_runner_is_called_and_summary_is_printed(tmp_path: Path, monkeypatch, capsys):
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(valid_payload()), encoding="utf-8")
    received = []
    expected = SimpleNamespace(
        score=1.25,
        num_edges=4,
        edge_loss_count=4,
        dataset="THEIA_E3",
        split="test",
        batch_index=3,
        device="cpu",
        warnings=("warning di test",),
    )

    def fake_run(config):
        received.append(config)
        return expected

    monkeypatch.setattr("adapters.orthrus_runtime.run_official_orthrus_smoke_test", fake_run)

    result = smoke_script.run_from_config(path)

    output = capsys.readouterr().out
    assert result is expected
    assert len(received) == 1
    assert isinstance(received[0], OrthrusOfficialRuntimeConfig)
    assert "score: 1.25" in output
    assert "num_edges: 4" in output
    assert "dataset: THEIA_E3" in output
    assert "warnings: warning di test" in output


def test_missing_json_has_clear_error(tmp_path: Path):
    with pytest.raises(smoke_script.SmokeConfigError, match="non trovato"):
        smoke_script.load_config(tmp_path / "missing.json")


def test_invalid_json_has_clear_error(tmp_path: Path):
    path = tmp_path / "invalid.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(smoke_script.SmokeConfigError, match="JSON non valido"):
        smoke_script.load_config(path)
