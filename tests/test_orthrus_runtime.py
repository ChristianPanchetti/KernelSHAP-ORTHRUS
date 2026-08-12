from __future__ import annotations

from pathlib import Path

import pytest

from adapters.orthrus_runtime import (
    OrthrusRuntime,
    OrthrusRuntimeConfig,
    OrthrusRuntimeError,
    run_unperturbed_smoke_test,
)


class FakeEdgeIndex:
    def __init__(self, num_edges: int):
        self.shape = (2, num_edges)


class FakeTemporalData:
    def __init__(self, num_edges: int):
        self.edge_index = FakeEdgeIndex(num_edges)
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeFullData:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeModel:
    def __init__(self, losses):
        self.losses = losses
        self.calls = []

    def eval(self):
        return self

    def __call__(self, batch, full_data, *, inference):
        self.calls.append((batch, full_data, inference))
        return self.losses


def make_existing_config(tmp_path: Path, **overrides):
    checkpoint = tmp_path / "model.ckpt"
    temporal = tmp_path / "temporal.pt"
    full_data = tmp_path / "full_data.pt"
    for path in (checkpoint, temporal, full_data):
        path.write_bytes(b"test-placeholder")

    values = {
        "checkpoint_path": checkpoint,
        "temporal_data_path": temporal,
        "full_data_path": full_data,
        "external_root": tmp_path / "missing-orthrus",
        "orthrus_module": "orthrus_missing_for_test",
        "device": "cpu",
        "required_dependencies": (),
    }
    values.update(overrides)
    return OrthrusRuntimeConfig(**values)


def test_runtime_module_imports_and_reports_missing_external(tmp_path: Path):
    runtime = OrthrusRuntime(make_existing_config(tmp_path))

    availability = runtime.inspect_availability()

    assert availability.available is False
    assert availability.external_root_available is False
    assert any("External ORTHRUS" in warning for warning in availability.warnings)


def test_runtime_fails_clearly_when_external_orthrus_is_missing(tmp_path: Path):
    runtime = OrthrusRuntime(make_existing_config(tmp_path))

    with pytest.raises(OrthrusRuntimeError, match="External ORTHRUS code is unavailable"):
        runtime.prepare()


@pytest.mark.parametrize(
    ("missing_field", "expected"),
    [
        ("checkpoint_path", "checkpoint"),
        ("temporal_data_path", "TemporalData artifact"),
        ("full_data_path", "full_data artifact"),
    ],
)
def test_runtime_fails_clearly_when_required_path_is_missing(tmp_path: Path, missing_field: str, expected: str):
    config = make_existing_config(tmp_path, **{missing_field: tmp_path / "does-not-exist"})

    with pytest.raises(OrthrusRuntimeError, match=expected):
        OrthrusRuntime(config).prepare()


def test_unperturbed_smoke_test_uses_final_adapter_path(tmp_path: Path):
    config = make_existing_config(tmp_path, device="test-device")
    batch = FakeTemporalData(3)
    full_data = FakeFullData()
    model = FakeModel([1.0, 2.0, 6.0])
    loaded_paths = []

    def model_loader(checkpoint_path, device, model_kwargs, external_module):
        assert checkpoint_path == config.checkpoint_path
        assert device == "test-device"
        assert model_kwargs == {}
        assert external_module is None
        return model

    def artifact_loader(path, device):
        loaded_paths.append((path, device))
        if path == config.temporal_data_path:
            return batch
        if path == config.full_data_path:
            return full_data
        raise AssertionError(f"Unexpected artifact path: {path}")

    result = run_unperturbed_smoke_test(
        config,
        model_loader=model_loader,
        artifact_loader=artifact_loader,
    )

    assert result.score == pytest.approx(3.0)
    assert result.num_edges == 3
    assert result.edge_loss_count == result.num_edges
    assert result.device == "test-device"
    assert model.calls == [(batch, full_data, True)]
    assert batch.device == "test-device"
    assert full_data.device == "test-device"
    assert loaded_paths == [
        (config.temporal_data_path, "test-device"),
        (config.full_data_path, "test-device"),
    ]
    assert any("bypassed" in warning for warning in result.warnings)


def test_runtime_wraps_artifact_loader_error_with_context(tmp_path: Path):
    config = make_existing_config(tmp_path)

    def model_loader(checkpoint_path, device, model_kwargs, external_module):
        return FakeModel([1.0])

    def broken_loader(path, device):
        raise OSError("opaque loader failure")

    with pytest.raises(OrthrusRuntimeError, match="Failed to load TemporalData artifact"):
        run_unperturbed_smoke_test(
            config,
            model_loader=model_loader,
            artifact_loader=broken_loader,
        )


def test_runtime_rejects_none_from_model_loader(tmp_path: Path):
    config = make_existing_config(tmp_path)

    with pytest.raises(OrthrusRuntimeError, match="model loader returned None"):
        OrthrusRuntime(
            config,
            model_loader=lambda checkpoint, device, kwargs, module: None,
            artifact_loader=lambda path, device: object(),
        ).prepare()
