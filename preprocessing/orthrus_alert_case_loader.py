from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Union

from preprocessing.orthrus_alert_case import OrthrusAlertCase
from preprocessing.orthrus_mapping import OrthrusMappingProvider, get_mapping_provider


class OrthrusAlertCaseLoader:
    """Load ORTHRUS preprocessed artifacts into an `OrthrusAlertCase`.

    Parameters (per-task spec):
    - temporal_data_path: path to a saved TemporalData-like object
    - mapping_mode: 'artifact-only' (default), 'graph-assisted', 'db-assisted'
    - mapping_provider: optional explicit provider (overrides factory)

    Notes:
    - Real ORTHRUS artifacts are typically saved via `torch.save` and should be
      loaded via `torch.load`. To keep the dummy pipeline torch-free, we import
      torch lazily.
    - For tests (and for synthetic objects), we provide a pickle fallback when
      torch is not installed.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger(__name__)

    def load(
        self,
        temporal_data_path: Union[str, Path],
        mapping_mode: str = "artifact-only",
        mapping_provider: Optional[OrthrusMappingProvider] = None,
    ) -> OrthrusAlertCase:
        path = Path(temporal_data_path)
        if not path.exists():
            raise FileNotFoundError(f"TemporalData artifact not found: {path}")

        provider = mapping_provider or get_mapping_provider(mapping_mode)
        if mapping_provider is not None:
            # Defensive: avoid silent mismatches.
            if provider.mode != str(mapping_mode).strip().lower():
                raise ValueError(
                    f"mapping_mode mismatch: mapping_mode='{mapping_mode}' but provider.mode='{provider.mode}'"
                )

        temporal_data = _load_temporal_data_artifact(path, logger=self._logger)

        case = OrthrusAlertCase(
            temporal_data=temporal_data,
            metadata={
                "temporal_data_path": str(path),
            },
            mapping_mode=provider.mode,
        )

        # Enrich mapping metadata (artifact-only is implemented; others may raise).
        provider.enrich_case(case)
        return case


def _load_temporal_data_artifact(path: Path, logger: logging.Logger) -> object:
    """Load a TemporalData-like artifact.

    - Prefer torch.load (real ORTHRUS artifacts)
    - Fallback to pickle when torch is missing (tests/synthetic fixtures)
    """

    try:
        import torch  # type: ignore

        logger.debug("Loading temporal_data via torch.load: %s", path)
        try:
            return torch.load(str(path), map_location="cpu")
        except Exception as torch_err:  # noqa: BLE001
            # If the file is not a torch.save artifact (e.g., synthetic tests), try pickle.
            logger.debug("torch.load failed (%s); trying pickle.load", torch_err)
            try:
                with path.open("rb") as f:
                    return pickle.load(f)
            except Exception as pickle_err:  # noqa: BLE001
                raise RuntimeError(
                    "Failed to load TemporalData artifact via both torch.load and pickle.load. "
                    "If this is a real ORTHRUS artifact saved with torch.save, ensure PyTorch is installed and the file is valid."
                ) from pickle_err
    except ModuleNotFoundError:
        logger.debug("torch not installed; using pickle.load: %s", path)
        with path.open("rb") as f:
            return pickle.load(f)
