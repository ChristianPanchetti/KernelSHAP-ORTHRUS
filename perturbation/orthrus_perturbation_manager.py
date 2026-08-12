from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set

import numpy as np

from preprocessing.orthrus_alert_case import OrthrusAlertCase


@dataclass(frozen=True)
class OrthrusPerturbationResult:
    """Container for a perturbed ORTHRUS alert case.

    For compatibility with the current Kernel SHAP explainer (which expects
    `.dataset`), we intentionally expose the perturbed case via `dataset`.
    """

    mask: List[int]
    dataset: OrthrusAlertCase


class OrthrusPerturbationManager:
    """ORTHRUS-specific perturbation manager.

    This manager perturbs an `OrthrusAlertCase` by dropping interpretable
    components, which correspond to groups of edges.
    """

    def __init__(self, original: OrthrusAlertCase, mode: str = "drop_edges"):
        supported = {"drop_edges", "neutralize_edges", "mask_edge_features"}
        mode = str(mode).strip().lower()
        if mode == "mask_edge_features":
            mode = "neutralize_edges"
        if mode not in supported:
            raise ValueError(f"Unsupported ORTHRUS perturbation mode: {mode}. Supported: {sorted(supported)}")

        if not original.component_to_edges:
            raise ValueError(
                "Cannot apply perturbation: `component_to_edges` is not defined in the original alert case. "
                "Please run an `OrthrusInterpretableBuilder` first."
            )

        self.original = original
        self.mode = mode
        self.logger = logging.getLogger(self.__class__.__name__)

    def apply_mask(self, mask: Sequence[int]) -> OrthrusPerturbationResult:
        """Applies a binary mask to the interpretable components."""
        num_components = len(self.original.component_to_edges)
        if len(mask) != num_components:
            raise ValueError(f"Mask length mismatch: expected {num_components} components, got a mask of {len(mask)}")

        if self.mode == "drop_edges":
            perturbed_case = self._drop_edges(mask)
        elif self.mode == "neutralize_edges":
            perturbed_case = self._neutralize_edges(mask)
        else:
            # This is currently unreachable due to the check in __init__
            raise RuntimeError(f"Unreachable: unsupported mode '{self.mode}'")

        return OrthrusPerturbationResult(mask=list(mask), dataset=perturbed_case)

    def _drop_edges(self, component_mask: Sequence[int]) -> OrthrusAlertCase:
        """
        Filters the alert case by dropping edges based on a component mask.

        Decision (TASK 4): Edge Indexing Strategy
        -----------------------------------------
        We adopt Option B: Original edge indices are preserved.

        - The `temporal_data` object is modified by slicing all edge-aligned
          fields (src, dst, t, msg, etc.) to remove the dropped edges.
        - The `edge_index` is reconstructed from the sliced `src` and `dst`,
          but it still contains the original node IDs.
        - All mapping dictionaries (`edge_to_event_uuid`, `component_to_edges`, etc.)
          in the new `OrthrusAlertCase` are filtered to only contain entries
          related to the kept edges. They still use original edge indices.

        This is the safest approach for ORTHRUS inference, as the model receives
        a valid subgraph with original node identifiers. It also simplifies the
        explainer's logic, as the mapping from a component to its original edges
        remains stable.
        """
        original_case = self.original
        component_ids = list(original_case.component_to_edges.keys())

        # 1. Determine which edges and components to keep
        kept_edge_indices: Set[int] = set()
        active_components: List[str] = []
        dropped_components: List[str] = []

        for i, comp_id in enumerate(component_ids):
            if component_mask[i] == 1:
                kept_edge_indices.update(original_case.component_to_edges[comp_id])
                active_components.append(comp_id)
            else:
                dropped_components.append(comp_id)

        # Sort for deterministic slicing
        sorted_kept_indices = sorted(list(kept_edge_indices))
        # Determine original number of edges in a robust way
        try:
            orig_num_edges = int(original_case.num_edges)
        except Exception:
            # Best-effort fallback: infer from temporal_data
            td = original_case.temporal_data
            if td is None:
                orig_num_edges = 0
            elif hasattr(td, "src"):
                try:
                    orig_num_edges = int(len(getattr(td, "src")))
                except Exception:
                    orig_num_edges = 0
            elif hasattr(td, "edge_index"):
                try:
                    ei = getattr(td, "edge_index")
                    if hasattr(ei, "shape") and len(getattr(ei, "shape")) == 2:
                        orig_num_edges = int(ei.shape[1])
                    else:
                        orig_num_edges = int(len(ei[0]))
                except Exception:
                    orig_num_edges = 0
            else:
                orig_num_edges = 0

        self.logger.debug(f"Keeping {len(sorted_kept_indices)} out of {orig_num_edges} edges.")

        # 2. Create a new, filtered TemporalData object
        new_temporal_data = self._filter_temporal_data(original_case.temporal_data, sorted_kept_indices)

        # 3. Filter all edge-aligned mapping fields
        new_edge_to_event_uuid = {
            k: v for k, v in original_case.edge_to_event_uuid.items() if k in kept_edge_indices
        }
        new_edge_to_log_record = {
            k: v for k, v in original_case.edge_to_log_record.items() if k in kept_edge_indices
        }
        new_edge_to_original_metadata = {
            k: v for k, v in original_case.edge_to_original_metadata.items() if k in kept_edge_indices
        }

        # 4. Update metadata
        new_metadata = dict(original_case.metadata)
        new_metadata.update(
            {
                "perturbation_mode": "drop_edges",
                "original_num_edges": orig_num_edges,
                "perturbed_num_edges": len(sorted_kept_indices),
                "active_components": active_components,
                "dropped_components": dropped_components,
            }
        )

        # 5. Construct the new OrthrusAlertCase
        perturbed_case = OrthrusAlertCase(
            temporal_data=new_temporal_data,
            full_data=original_case.full_data,  # Remains unchanged
            metadata=new_metadata,
            mapping_mode=original_case.mapping_mode,
            # --- Filtered mapping fields ---
            edge_to_event_uuid=new_edge_to_event_uuid,
            edge_to_log_record=new_edge_to_log_record,
            edge_to_original_metadata=new_edge_to_original_metadata,
            # --- Unchanged or preserved fields ---
            component_to_edges=original_case.component_to_edges,
            mapping_quality=original_case.mapping_quality,
            log_dataset=original_case.log_dataset,  # Keep reference to original logs if present
        )

        return perturbed_case

    def _neutralize_edges(self, component_mask: Sequence[int]) -> OrthrusAlertCase:
        """Neutralize inactive component features while preserving edge order/count.

        This mode is intended for real ORTHRUS inference, where `LastNeighborLoader`
        derives global e_id values from chronological insertion order. Dropping edges
        shifts that internal counter; neutralizing features keeps the sequence intact.
        """
        original_case = self.original
        component_ids = list(original_case.component_to_edges.keys())

        active_components: List[str] = []
        inactive_components: List[str] = []
        neutralized_edges: Set[int] = set()

        for i, comp_id in enumerate(component_ids):
            if component_mask[i] == 1:
                active_components.append(comp_id)
            else:
                inactive_components.append(comp_id)
                neutralized_edges.update(original_case.component_to_edges[comp_id])

        try:
            orig_num_edges = int(original_case.num_edges)
        except Exception:
            orig_num_edges = 0

        sorted_neutralized_edges = sorted(int(i) for i in neutralized_edges)
        new_temporal_data = self._neutralize_temporal_data(original_case.temporal_data, sorted_neutralized_edges)

        new_metadata = dict(original_case.metadata)
        new_metadata.update(
            {
                "perturbation_mode": "neutralize_edges",
                "original_num_edges": orig_num_edges,
                "perturbed_num_edges": orig_num_edges,
                "active_components": active_components,
                "inactive_components": inactive_components,
                "neutralized_edges": sorted_neutralized_edges,
            }
        )

        return OrthrusAlertCase(
            temporal_data=new_temporal_data,
            full_data=original_case.full_data,
            metadata=new_metadata,
            edge_to_record=original_case.edge_to_record,
            node_to_records=original_case.node_to_records,
            component_to_edges=original_case.component_to_edges,
            mapping_mode=original_case.mapping_mode,
            edge_to_event_uuid=original_case.edge_to_event_uuid,
            edge_to_log_record=original_case.edge_to_log_record,
            edge_to_original_metadata=original_case.edge_to_original_metadata,
            mapping_quality=original_case.mapping_quality,
            log_dataset=original_case.log_dataset,
        )

    def _filter_temporal_data(self, temporal_data: Any, keep_indices: list[int]) -> Any:
        """
        Creates a new TemporalData-like object containing only the edges
        specified by `keep_indices`.
        """
        if temporal_data is None:
            return None

        # Use copy to avoid modifying the original object's non-edge attributes
        try:
            new_data = copy.copy(temporal_data)
        except Exception as e:
            raise TypeError(f"Could not shallow-copy temporal_data of type {type(temporal_data)}") from e

        # List of all potential edge-aligned fields in ORTHRUS
        edge_fields = ["src", "dst", "t", "msg", "edge_type", "edge_feats", "x_src", "x_dst"]

        for field in edge_fields:
            if hasattr(temporal_data, field):
                original_attr = getattr(temporal_data, field)
                if original_attr is not None:
                    try:
                        # Prefer direct list indexing for Python lists (more robust across envs)
                        if isinstance(original_attr, list):
                            sliced_attr = [original_attr[i] for i in keep_indices]
                            setattr(new_data, field, sliced_attr)
                        else:
                            # Use numpy for arrays/tensors or other indexable types
                            arr = np.array(original_attr)
                            sliced = arr[keep_indices]
                            # Convert to list when possible
                            try:
                                out = sliced.tolist()
                            except Exception:
                                out = sliced
                            setattr(new_data, field, out)
                    except Exception as e:
                        self.logger.warning(f"Could not slice attribute '{field}' (type: {type(original_attr)}): {e}")

        # Reconstruct edge_index from the new src and dst
        if hasattr(new_data, "src") and hasattr(new_data, "dst"):
            new_src = getattr(new_data, "src")
            new_dst = getattr(new_data, "dst")
            if new_src is not None and new_dst is not None:
                # Stack vertically and keep as numpy array, as ORTHRUS expects a tensor-like object
                new_edge_index = np.stack([new_src, new_dst], axis=0)
                setattr(new_data, "edge_index", new_edge_index)

        return new_data

    def _neutralize_temporal_data(self, temporal_data: Any, neutralize_indices: list[int]) -> Any:
        if temporal_data is None:
            return None

        try:
            new_data = copy.copy(temporal_data)
        except Exception as e:
            raise TypeError(f"Could not shallow-copy temporal_data of type {type(temporal_data)}") from e

        feature_fields = ["msg", "x_src", "x_dst", "edge_feats"]
        for field in feature_fields:
            if hasattr(temporal_data, field):
                original_attr = getattr(temporal_data, field)
                if original_attr is not None:
                    try:
                        setattr(new_data, field, _zero_edge_rows(original_attr, neutralize_indices))
                    except Exception as e:
                        self.logger.warning(f"Could not neutralize attribute '{field}' (type: {type(original_attr)}): {e}")

        return new_data


def _zero_edge_rows(value: Any, indices: list[int]) -> Any:
    if value is None or not indices:
        return value

    if _is_torch_tensor(value):
        out = value.clone()
        if out.ndim == 0:
            return out
        idx = _torch_index(indices, device=out.device)
        out[idx] = _torch_zeros_like(out[idx])
        return out

    if isinstance(value, np.ndarray):
        out = value.copy()
        out[indices] = np.zeros_like(out[indices])
        return out

    if isinstance(value, list):
        out = copy.deepcopy(value)
        for i in indices:
            if 0 <= i < len(out):
                out[i] = _zero_like_python_value(out[i])
        return out

    try:
        out = copy.copy(value)
        for i in indices:
            out[i] = _zero_like_python_value(out[i])
        return out
    except Exception:
        raise TypeError(f"Unsupported edge-aligned feature type: {type(value)}")


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and hasattr(value, "clone")


def _torch_index(indices: list[int], device: Any):
    import torch  # type: ignore

    return torch.tensor(indices, dtype=torch.long, device=device)


def _torch_zeros_like(value: Any):
    import torch  # type: ignore

    return torch.zeros_like(value)


def _zero_like_python_value(value: Any) -> Any:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, complex)):
        return type(value)(0)
    if isinstance(value, tuple):
        return tuple(_zero_like_python_value(v) for v in value)
    if isinstance(value, list):
        return [_zero_like_python_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return np.zeros_like(value)
    if _is_torch_tensor(value):
        return _torch_zeros_like(value)
    return 0
