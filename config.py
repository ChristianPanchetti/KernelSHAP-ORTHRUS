from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    log_file: Optional[Path] = None


@dataclass(frozen=True)
class KernelSHAPConfig:
    """Kernel SHAP configuration (official SHAP backend only).

    Notes:
    - `num_samples` is forwarded to `shap.KernelExplainer(...).shap_values(..., nsamples=...)`.
    """

    num_samples: int = 200
    seed: int = 0
    top_k: int = 10


@dataclass(frozen=True)
class PerturbationConfig:
    """Perturbation and interpretable-space configuration."""

    grouping_mode: str = "exec_path"  # exec_path | record | entity_id | parent_child | time_window | event_type | local_subgraph
    max_components: Optional[int] = 50
    time_window_seconds: int = 60  # used only when grouping_mode == 'time_window'
    perturbation_mode: str = "drop_records"  # drop_records | neutralize_command | drop_time_window | mask_features | drop_subgraph | remove_node_with_incident_edges


@dataclass(frozen=True)
class AppConfig:
    """Central configuration for the end-to-end Kernel SHAP pipeline."""

    input_path: Path
    output_json_path: Path

    # Adapter selection
    adapter_type: str = "dummy"  # dummy | placeholder | real

    # Pipeline mode selection
    pipeline_mode: str = "dummy"  # dummy | orthrus

    # Pipeline parameters
    device: str = "cpu"  # kept for future integration (e.g., torch model)

    shap: KernelSHAPConfig = field(default_factory=KernelSHAPConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Optional extra output
    output_ranking_csv_path: Optional[Path] = None

    def ensure_output_dirs(self) -> None:
        self.output_json_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_ranking_csv_path is not None:
            self.output_ranking_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.logging.log_file is not None:
            self.logging.log_file.parent.mkdir(parents=True, exist_ok=True)
