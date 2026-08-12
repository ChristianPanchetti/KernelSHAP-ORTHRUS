from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from adapters.orthrus_ano_adapter import DummyOrthrusAnoAdapter, RealOrthrusAnoAdapter
from config import AppConfig, KernelSHAPConfig, LoggingConfig, PerturbationConfig
from pipeline import run_kernel_shap_pipeline
from preprocessing.orthrus_alert_case import OrthrusAlertCase


def test_orthrus_scaffolding_imports():
    # These imports should stay torch/pyg-free.
    from perturbation.orthrus_interpretable_builder import OrthrusInterpretableBuilder  # noqa: F401
    from perturbation.orthrus_perturbation_manager import OrthrusPerturbationManager  # noqa: F401


def test_real_adapter_requires_injected_model():
    with pytest.raises(ValueError, match="constructed model"):
        RealOrthrusAnoAdapter(model=None, logger=logging.getLogger("test"))


def test_dummy_adapter_rejects_orthrus_case():
    adapter = DummyOrthrusAnoAdapter(seed=0, logger=logging.getLogger("test"))
    case = OrthrusAlertCase(temporal_data=object())

    with pytest.raises(TypeError):
        adapter.predict_anomaly_score(case)


def test_pipeline_mode_adapter_mismatch_is_caught_early(tmp_path: Path, default_input_path: Path):
    cfg = AppConfig(
        input_path=default_input_path,
        output_json_path=tmp_path / "out.json",
        adapter_type="real",
        pipeline_mode="dummy",
        shap=KernelSHAPConfig(num_samples=10, seed=0, top_k=3),
        perturbation=PerturbationConfig(),
        logging=LoggingConfig(level="INFO", log_file=None),
    )

    with pytest.raises(ValueError, match="pipeline_mode"):
        run_kernel_shap_pipeline(cfg, logging.getLogger("test"))


def test_cli_orthrus_mode_runs_and_fails_with_stub(tmp_path: Path, project_root: Path, default_input_path: Path):
    out_json = tmp_path / "kernel_shap_results.json"

    cmd = [
        sys.executable,
        "main.py",
        "--mode",
        "orthrus",
        "--adapter",
        "real",
        "--input",
        str(default_input_path),
        "--output",
        str(out_json),
        "--num-samples",
        "10",
        "--seed",
        "0",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "ORTHRUS pipeline mode is scaffolded" in (proc.stdout + proc.stderr)
