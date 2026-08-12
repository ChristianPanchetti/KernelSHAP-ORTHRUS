"""Runtime loading layer for one real, unperturbed ORTHRUS inference.

The module is import-safe without ORTHRUS, PyTorch, or PyG. Real dependencies
are inspected and imported only when the runtime is asked to prepare a model
and its artifacts.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from adapters.orthrus_ano_adapter import RealOrthrusAnoAdapter
from preprocessing.orthrus_alert_case import OrthrusAlertCase


class OrthrusRuntimeError(RuntimeError):
    """Raised when the external ORTHRUS runtime cannot be prepared clearly."""


@dataclass(frozen=True)
class OrthrusRuntimeConfig:
    """Paths and import contract required for real ORTHRUS inference."""

    checkpoint_path: Path
    temporal_data_path: Path
    full_data_path: Path
    external_root: Optional[Path] = None
    orthrus_module: Optional[str] = None
    model_factory: Optional[str] = None
    device: Any = "cpu"
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    required_dependencies: Sequence[str] = ("torch", "torch_geometric")


@dataclass(frozen=True)
class OrthrusRuntimeAvailability:
    external_root_available: bool
    missing_dependencies: tuple[str, ...]
    orthrus_module_available: bool
    warnings: tuple[str, ...]

    @property
    def available(self) -> bool:
        return (
            self.external_root_available
            and not self.missing_dependencies
            and self.orthrus_module_available
        )


@dataclass(frozen=True)
class PreparedOrthrusRuntime:
    model: Any
    temporal_data: Any
    full_data: Any
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class OrthrusSmokeTestResult:
    score: float
    num_edges: int
    edge_loss_count: int
    device: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "num_edges": self.num_edges,
            "edge_loss_count": self.edge_loss_count,
            "device": self.device,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class OrthrusOfficialRuntimeConfig:
    """Configuration for the cfg-driven runtime shipped by official ORTHRUS."""

    external_root: Path
    config_path: Path
    dataset_name: str
    model_epoch_dir: Optional[Path]
    split: str = "test"
    graph_index: int = 0
    batch_index: int = 0
    device: Any = "cpu"
    from_weights_path: Optional[Path] = None
    overrides: Mapping[str, Any] | Sequence[str] = field(default_factory=dict)
    require_model_epoch: bool = True


@dataclass(frozen=True)
class OfficialOrthrusModules:
    """Injectable references to the three official modules used by the runtime."""

    config: Any
    data_utils: Any
    factory: Any
    torch: Any = None


@dataclass(frozen=True)
class OrthrusOfficialSmokeTestResult:
    score: float
    num_edges: int
    edge_loss_count: int
    dataset: str
    split: str
    graph_index: int
    batch_index: int
    device: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "num_edges": self.num_edges,
            "edge_loss_count": self.edge_loss_count,
            "dataset": self.dataset,
            "split": self.split,
            "graph_index": self.graph_index,
            "batch_index": self.batch_index,
            "device": self.device,
            "warnings": list(self.warnings),
        }


ModelLoader = Callable[[Path, Any, Mapping[str, Any], Optional[Any]], Any]
ArtifactLoader = Callable[[Path, Any], Any]


class OrthrusRuntime:
    """Prepare model, local batch, and global data for real ORTHRUS inference.

    Injected loaders exist to test the final orchestration contract without
    shipping external code or artifacts. In real use, the default artifact
    loader uses ``torch.load`` and the configured model factory is imported
    dynamically from the external ORTHRUS checkout.
    """

    def __init__(
        self,
        config: OrthrusRuntimeConfig,
        *,
        model_loader: Optional[ModelLoader] = None,
        artifact_loader: Optional[ArtifactLoader] = None,
    ):
        self.config = config
        self._model_loader = model_loader
        self._artifact_loader = artifact_loader

    def inspect_availability(self) -> OrthrusRuntimeAvailability:
        warnings: list[str] = []
        missing_dependencies = tuple(
            name for name in self.config.required_dependencies if not _module_spec_available(name)
        )

        root = self.config.external_root
        root_available = bool(root is not None and Path(root).is_dir())
        if not root_available:
            warnings.append("External ORTHRUS root is not available")

        module_available = False
        if root_available and self.config.orthrus_module:
            _add_external_root_to_import_path(Path(root))
            module_available = _module_spec_available(self.config.orthrus_module)
            if not module_available:
                warnings.append(f"ORTHRUS module '{self.config.orthrus_module}' is not importable")
        elif not self.config.orthrus_module:
            warnings.append("No ORTHRUS module was configured")

        if missing_dependencies:
            warnings.append("Missing optional dependencies: " + ", ".join(missing_dependencies))

        return OrthrusRuntimeAvailability(
            external_root_available=root_available,
            missing_dependencies=missing_dependencies,
            orthrus_module_available=module_available,
            warnings=tuple(warnings),
        )

    def prepare(self) -> PreparedOrthrusRuntime:
        _validate_runtime_paths(self.config)
        warnings: list[str] = []

        if self._model_loader is None:
            availability = self.inspect_availability()
            if not availability.external_root_available:
                raise OrthrusRuntimeError(
                    "External ORTHRUS code is unavailable. Configure external_root to an existing ORTHRUS checkout."
                )
            if availability.missing_dependencies:
                raise OrthrusRuntimeError(
                    "Missing dependencies required by real ORTHRUS: "
                    + ", ".join(availability.missing_dependencies)
                )
            if not availability.orthrus_module_available:
                raise OrthrusRuntimeError(
                    f"Configured ORTHRUS module {self.config.orthrus_module!r} is not importable from "
                    f"{self.config.external_root}"
                )
            external_module = importlib.import_module(str(self.config.orthrus_module))
            model = _load_model_from_factory(self.config, external_module)
        else:
            external_module = None
            model = self._model_loader(
                Path(self.config.checkpoint_path),
                self.config.device,
                self.config.model_kwargs,
                external_module,
            )
            if self.config.external_root is None or not Path(self.config.external_root).is_dir():
                warnings.append("External ORTHRUS availability was bypassed by the injected model loader")

        if model is None:
            raise OrthrusRuntimeError("ORTHRUS model loader returned None")

        artifact_loader = self._artifact_loader or _load_torch_artifact
        temporal_data = _load_artifact_clearly(
            artifact_loader, Path(self.config.temporal_data_path), self.config.device, "TemporalData"
        )
        full_data = _load_artifact_clearly(
            artifact_loader, Path(self.config.full_data_path), self.config.device, "full_data"
        )

        return PreparedOrthrusRuntime(
            model=model,
            temporal_data=temporal_data,
            full_data=full_data,
            warnings=tuple(warnings),
        )


def run_unperturbed_smoke_test(
    config: OrthrusRuntimeConfig,
    *,
    model_loader: Optional[ModelLoader] = None,
    artifact_loader: Optional[ArtifactLoader] = None,
) -> OrthrusSmokeTestResult:
    """Run one final-path ORTHRUS inference without SHAP or perturbations."""

    prepared = OrthrusRuntime(
        config,
        model_loader=model_loader,
        artifact_loader=artifact_loader,
    ).prepare()
    case = OrthrusAlertCase(
        temporal_data=prepared.temporal_data,
        full_data=prepared.full_data,
        metadata={
            "checkpoint_path": str(config.checkpoint_path),
            "temporal_data_path": str(config.temporal_data_path),
            "full_data_path": str(config.full_data_path),
            "smoke_test": "unperturbed",
        },
    )

    adapter = RealOrthrusAnoAdapter(model=prepared.model, device=config.device)
    score = adapter.predict_anomaly_score(case)
    num_edges = _infer_num_edges_for_report(prepared.temporal_data)

    # A successful adapter call guarantees one validated loss per batch edge.
    return OrthrusSmokeTestResult(
        score=float(score),
        num_edges=num_edges,
        edge_loss_count=num_edges,
        device=str(config.device),
        warnings=prepared.warnings,
    )


def run_official_orthrus_smoke_test(
    config: OrthrusOfficialRuntimeConfig,
    *,
    modules: Optional[OfficialOrthrusModules] = None,
) -> OrthrusOfficialSmokeTestResult:
    """Run one unperturbed batch through the official cfg-driven ORTHRUS path."""

    normalized = replace(
        config,
        external_root=Path(config.external_root).resolve(),
        config_path=Path(config.config_path).resolve(),
        model_epoch_dir=(Path(config.model_epoch_dir).resolve() if config.model_epoch_dir else None),
        from_weights_path=(
            Path(config.from_weights_path).resolve() if config.from_weights_path else None
        ),
    )
    _validate_official_runtime_config(normalized)
    official = modules or _import_official_orthrus_modules(Path(normalized.external_root))
    previous_cwd = Path.cwd()
    try:
        # Official config.py defines ROOT_ARTIFACT_DIR as "./artifacts".
        os.chdir(normalized.external_root)
        return _run_official_orthrus_smoke_test(normalized, official)
    finally:
        os.chdir(previous_cwd)


def _run_official_orthrus_smoke_test(
    config: OrthrusOfficialRuntimeConfig,
    official: OfficialOrthrusModules,
) -> OrthrusOfficialSmokeTestResult:
    cfg = _load_official_cfg(config, official.config)

    try:
        loaded = official.data_utils.load_all_datasets(cfg)
    except Exception as exc:
        raise OrthrusRuntimeError("Official ORTHRUS load_all_datasets(cfg) failed") from exc
    if not isinstance(loaded, (tuple, list)) or len(loaded) != 5:
        raise OrthrusRuntimeError(
            "Official load_all_datasets(cfg) must return train, val, test, full_data, max_node_num"
        )
    train_data, val_data, test_data, full_data, max_node_num = loaded
    splits = {"train": train_data, "val": val_data, "test": test_data}
    if config.split not in splits:
        raise OrthrusRuntimeError(
            f"Unknown ORTHRUS split {config.split!r}; expected one of: train, val, test"
        )
    graph = _select_index(splits[config.split], config.graph_index, "graph_index")

    try:
        model = official.factory.build_model(
            data_sample=graph,
            device=config.device,
            cfg=cfg,
            max_node_num=max_node_num,
        )
    except Exception as exc:
        raise OrthrusRuntimeError("Official ORTHRUS build_model(...) failed") from exc
    if model is None:
        raise OrthrusRuntimeError("Official ORTHRUS build_model(...) returned None")

    warnings: list[str] = []
    if config.model_epoch_dir is not None:
        try:
            model = official.data_utils.load_model(model, str(config.model_epoch_dir))
        except Exception as exc:
            raise OrthrusRuntimeError(
                f"Official ORTHRUS load_model(...) failed for {config.model_epoch_dir}"
            ) from exc
    else:
        warnings.append(
            "No model_epoch_dir loaded: neighbor-loader state is fresh, so inference is not temporally faithful"
        )

    if config.from_weights_path is not None:
        torch_module = official.torch or _import_torch_clearly()
        try:
            state_dict = torch_module.load(str(config.from_weights_path), map_location=config.device)
            model.load_state_dict(state_dict)
        except Exception as exc:
            raise OrthrusRuntimeError(
                f"Failed to apply ORTHRUS pretrained weights from {config.from_weights_path}"
            ) from exc
        warnings.append(
            "Pretrained weights were applied after model_epoch_dir; they replace model weights but not neighbor-loader state"
        )

    if callable(getattr(graph, "to", None)):
        moved = graph.to(device=config.device)
        if moved is not None:
            graph = moved
    graph_reindexer = getattr(model, "graph_reindexer", None)
    if graph_reindexer is None:
        raise OrthrusRuntimeError("Official ORTHRUS model has no graph_reindexer")
    try:
        batch_loader = official.factory.batch_loader_factory(cfg, graph, graph_reindexer)
    except Exception as exc:
        raise OrthrusRuntimeError("Official ORTHRUS batch_loader_factory(...) failed") from exc
    batch = _select_index(batch_loader, config.batch_index, "batch_index")

    case = OrthrusAlertCase(
        temporal_data=batch,
        full_data=full_data,
        metadata={
            "dataset": config.dataset_name,
            "split": config.split,
            "graph_index": config.graph_index,
            "batch_index": config.batch_index,
            "smoke_test": "official-unperturbed",
        },
    )
    # Official ORTHRUS keeps full_data on CPU and indexes it with e_id.cpu().
    # The selected graph/batch is already moved above; avoid moving full_data in the adapter.
    adapter = RealOrthrusAnoAdapter(model=model, device=None)
    score = adapter.predict_anomaly_score(case)
    num_edges = _infer_num_edges_for_report(batch)
    return OrthrusOfficialSmokeTestResult(
        score=float(score),
        num_edges=num_edges,
        edge_loss_count=num_edges,
        dataset=config.dataset_name,
        split=config.split,
        graph_index=config.graph_index,
        batch_index=config.batch_index,
        device=str(config.device),
        warnings=tuple(warnings),
    )


def _validate_official_runtime_config(config: OrthrusOfficialRuntimeConfig) -> None:
    root = Path(config.external_root)
    if not root.is_dir() or not (root / "src").is_dir():
        raise OrthrusRuntimeError(f"Official ORTHRUS checkout/src is missing: {root}")
    official_config = (root / "config" / "orthrus.yml").resolve()
    requested_config = Path(config.config_path).resolve()
    if not requested_config.is_file():
        raise OrthrusRuntimeError(f"Official ORTHRUS config does not exist: {requested_config}")
    if requested_config != official_config:
        raise OrthrusRuntimeError(
            "Official get_yml_cfg() loads external_root/config/orthrus.yml; config_path must point to that file"
        )
    if not str(config.dataset_name).strip():
        raise OrthrusRuntimeError("dataset_name must not be empty")
    if config.graph_index < 0 or config.batch_index < 0:
        raise OrthrusRuntimeError("graph_index and batch_index must be non-negative")
    if config.require_model_epoch and config.model_epoch_dir is None:
        raise OrthrusRuntimeError(
            "model_epoch_dir is required for a faithful smoke test because it contains neighbor_loader.pkl"
        )
    if config.model_epoch_dir is not None:
        epoch_dir = Path(config.model_epoch_dir)
        if not epoch_dir.is_dir():
            raise OrthrusRuntimeError(f"model_epoch_dir does not exist or is not a directory: {epoch_dir}")
        for filename in ("state_dict.pkl", "neighbor_loader.pkl"):
            if not (epoch_dir / filename).is_file():
                raise OrthrusRuntimeError(f"model_epoch_dir is missing required {filename}: {epoch_dir}")
    if config.from_weights_path is not None and not Path(config.from_weights_path).is_file():
        raise OrthrusRuntimeError(f"from_weights_path does not exist: {config.from_weights_path}")


def _load_official_cfg(config: OrthrusOfficialRuntimeConfig, config_module: Any) -> Any:
    cli_args = [config.dataset_name]
    if str(config.device).lower().startswith("cpu"):
        cli_args.append("--cpu")
    if config.from_weights_path is not None:
        cli_args.append("--from_weights")
    if isinstance(config.overrides, Mapping):
        cli_args.extend(f"--{key}={value}" for key, value in config.overrides.items())
    else:
        cli_args.extend(str(value) for value in config.overrides)
    try:
        args = config_module.get_runtime_required_args(args=cli_args)
        return config_module.get_yml_cfg(args)
    except SystemExit as exc:
        raise OrthrusRuntimeError("Official ORTHRUS configuration argument parsing failed") from exc
    except Exception as exc:
        raise OrthrusRuntimeError("Official ORTHRUS cfg loading failed") from exc


def _select_index(values: Any, index: int, label: str) -> Any:
    try:
        for current, value in enumerate(values):
            if current == index:
                return value
    except TypeError as exc:
        raise OrthrusRuntimeError(f"ORTHRUS {label} source is not iterable") from exc
    raise OrthrusRuntimeError(f"ORTHRUS {label} {index} is out of range")


def _import_torch_clearly() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise OrthrusRuntimeError("PyTorch is required by the official ORTHRUS runtime") from exc


def _import_official_orthrus_modules(external_root: Path) -> OfficialOrthrusModules:
    src = (external_root / "src").resolve()
    _add_external_root_to_import_path(src)
    config_file = src / "config.py"
    try:
        spec = importlib.util.spec_from_file_location("_kshap_official_orthrus_config", config_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create a module spec for {config_file}")
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        previous_config = sys.modules.get("config")
        sys.modules["config"] = config_module
        try:
            data_utils_module = importlib.import_module("data_utils")
            factory_module = importlib.import_module("factory")
        finally:
            if previous_config is None:
                sys.modules.pop("config", None)
            else:
                sys.modules["config"] = previous_config
    except ModuleNotFoundError as exc:
        raise OrthrusRuntimeError(
            f"Missing dependency while importing official ORTHRUS from {src}: {exc.name}"
        ) from exc
    except Exception as exc:
        raise OrthrusRuntimeError(f"Could not import official ORTHRUS modules from {src}") from exc
    return OfficialOrthrusModules(
        config=config_module,
        data_utils=data_utils_module,
        factory=factory_module,
    )

def _validate_runtime_paths(config: OrthrusRuntimeConfig) -> None:
    for label, raw_path in (
        ("checkpoint", config.checkpoint_path),
        ("TemporalData artifact", config.temporal_data_path),
        ("full_data artifact", config.full_data_path),
    ):
        path = Path(raw_path)
        if not path.is_file():
            raise OrthrusRuntimeError(f"Required {label} does not exist or is not a file: {path}")


def _add_external_root_to_import_path(root: Path) -> None:
    root_text = str(root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _module_spec_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
        return False


def _load_model_from_factory(config: OrthrusRuntimeConfig, external_module: Any) -> Any:
    if not config.model_factory:
        raise OrthrusRuntimeError(
            "No model_factory configured. Use 'module.path:callable' for the real ORTHRUS model/checkpoint factory."
        )

    factory = _resolve_callable(config.model_factory, external_module)
    try:
        model = factory(
            checkpoint_path=Path(config.checkpoint_path),
            device=config.device,
            **dict(config.model_kwargs),
        )
    except Exception as exc:
        raise OrthrusRuntimeError(
            f"ORTHRUS model factory {config.model_factory!r} failed while loading checkpoint "
            f"{config.checkpoint_path}"
        ) from exc
    if model is None:
        raise OrthrusRuntimeError(f"ORTHRUS model factory {config.model_factory!r} returned None")
    return model


def _resolve_callable(spec: str, external_module: Any) -> Callable[..., Any]:
    if ":" not in spec:
        raise OrthrusRuntimeError("model_factory must use the form 'module.path:callable'")
    module_name, attribute_path = spec.split(":", 1)
    try:
        module = external_module if module_name in {"", "."} else importlib.import_module(module_name)
        value: Any = module
        for part in attribute_path.split("."):
            value = getattr(value, part)
    except (ImportError, AttributeError) as exc:
        raise OrthrusRuntimeError(f"Cannot resolve ORTHRUS model factory {spec!r}") from exc
    if not callable(value):
        raise OrthrusRuntimeError(f"Configured ORTHRUS model factory {spec!r} is not callable")
    return value


def _load_torch_artifact(path: Path, device: Any) -> Any:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise OrthrusRuntimeError(
            "PyTorch is required to load real ORTHRUS artifacts but is not installed"
        ) from exc
    return torch.load(str(path), map_location=device or "cpu")


def _load_artifact_clearly(loader: ArtifactLoader, path: Path, device: Any, label: str) -> Any:
    try:
        value = loader(path, device)
    except OrthrusRuntimeError:
        raise
    except Exception as exc:
        raise OrthrusRuntimeError(f"Failed to load {label} artifact from {path}") from exc
    if value is None:
        raise OrthrusRuntimeError(f"Loader returned None for {label} artifact: {path}")
    return value


def _infer_num_edges_for_report(batch: Any) -> int:
    edge_index = getattr(batch, "edge_index", None)
    shape = getattr(edge_index, "shape", None) if edge_index is not None else None
    if shape is not None and len(shape) == 2 and int(shape[0]) == 2:
        return int(shape[1])
    for field in ("src", "msg"):
        value = getattr(batch, field, None)
        if value is not None:
            try:
                return int(len(value))
            except TypeError:
                pass
    raise OrthrusRuntimeError("Cannot report edge count for the loaded TemporalData batch")
