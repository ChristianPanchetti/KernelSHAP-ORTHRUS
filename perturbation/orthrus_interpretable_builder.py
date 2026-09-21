from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from preprocessing.orthrus_alert_case import OrthrusAlertCase
from preprocessing.orthrus_mapping import infer_num_edges


@dataclass(frozen=True)
class OrthrusInterpretableBuildInfo:
    """Metadata emitted by the ORTHRUS interpretable builder."""

    grouping_mode: str
    max_components: str


class OrthrusInterpretableBuilder:
    """Build interpretable components from an ORTHRUS `TemporalData` case.

    This module is intentionally separate from `InterpretableGraphBuilder`:
    - The debug/dummy pipeline builds components from `LogDataset` records.
    - The ORTHRUS/DARPA pipeline builds components from TemporalData edges,
      source nodes, edge types, or temporal chunks.

    Implemented grouping modes:
    - `edge`: one component per edge/event in the TemporalData window
    - `node`: partition edges by source node (`src`)
    - `edge_type`: group by edge type class
    - `time_chunk`: group by time sub-window within the TemporalData window

    The resulting mapping is stored in `OrthrusAlertCase.component_to_edges`.
    Integration with the real ORTHRUS end-to-end pipeline remains future work,
    but this builder itself is implemented and covered by synthetic tests.
    """

    def __init__(
        self,
        grouping_mode: str = "edge",
        max_components: Optional[int] = None,
        time_chunk_seconds: int = 60,
    ):
        self.grouping_mode = str(grouping_mode)
        self.max_components = max_components
        self.time_chunk_seconds = int(time_chunk_seconds)

    def build(self, case: OrthrusAlertCase):
        self.validate_case_minimum(case)

        temporal_data = case.temporal_data
        num_edges = infer_num_edges(temporal_data)
        if num_edges is None or num_edges <= 0:
            raise ValueError("Cannot build ORTHRUS interpretable space: temporal_data has no edges")

        mode = self.grouping_mode.strip().lower()
        supported = {"edge", "node", "edge_type", "time_chunk"}
        if mode not in supported:
            raise ValueError(f"Unsupported grouping_mode: {self.grouping_mode}. Supported: {sorted(supported)}")

        groups: Dict[str, List[int]]

        if mode == "edge":
            groups = {f"edge:{i}": [i] for i in range(num_edges)}
        elif mode == "node":
            # Partition edges by src node (disjoint grouping).
            if not hasattr(temporal_data, "src"):
                raise ValueError("grouping_mode='node' requires temporal_data.src")
            groups = {}
            for i in range(num_edges):
                src = _safe_int(_edge_attr(temporal_data, "src", i))
                key = f"node:{src}" if src is not None else f"edge:{i}"
                groups.setdefault(key, []).append(i)
        elif mode == "edge_type":
            if not hasattr(temporal_data, "edge_type"):
                raise ValueError("grouping_mode='edge_type' requires temporal_data.edge_type")
            groups = {}
            for i in range(num_edges):
                et = _edge_type_label(_edge_attr(temporal_data, "edge_type", i))
                key = f"edge_type:{et}" if et is not None else f"edge:{i}"
                groups.setdefault(key, []).append(i)
        elif mode == "time_chunk":
            if not hasattr(temporal_data, "t"):
                raise ValueError("grouping_mode='time_chunk' requires temporal_data.t")
            if self.time_chunk_seconds <= 0:
                raise ValueError("time_chunk_seconds must be > 0")

            ts: List[int] = []
            for i in range(num_edges):
                t = _safe_int(_edge_attr(temporal_data, "t", i))
                if isinstance(t, int):
                    ts.append(t)
            if not ts:
                raise ValueError("grouping_mode='time_chunk' requires numeric temporal_data.t values")

            t0 = min(ts)
            # Interpret `time_chunk_seconds` in the same unit as `temporal_data.t` for tests.
            chunk_ns = int(self.time_chunk_seconds)
            groups = {}
            for i in range(num_edges):
                t = _safe_int(_edge_attr(temporal_data, "t", i))
                if not isinstance(t, int):
                    key = f"edge:{i}"
                else:
                    bucket = int((t - t0) // chunk_ns)
                    key = f"time_chunk:{bucket}"
                groups.setdefault(key, []).append(i)
        else:
            raise RuntimeError("Unreachable: unsupported grouping_mode")

        # Ensure deterministic component contents.
        for k in list(groups.keys()):
            groups[k] = sorted(set(int(x) for x in groups[k]))

        groups = _apply_max_components(groups, max_components=self.max_components)

        # Sanity: every edge must be assigned to exactly one component.
        seen = set()
        for edges in groups.values():
            for e in edges:
                if e in seen:
                    raise RuntimeError("Internal error: edge assigned to multiple components")
                seen.add(e)
        if len(seen) != num_edges:
            raise RuntimeError(f"Internal error: {num_edges - len(seen)} edge(s) not mapped to any component")

        case.component_to_edges = groups
        return OrthrusInterpretableBuildInfo(
            grouping_mode=mode,
            max_components=str(self.max_components) if self.max_components is not None else "None",
        )

    @staticmethod
    def suggested_component_ids(case: OrthrusAlertCase) -> List[str]:
        """Return the canonical component_to_edges insertion order.

        Mask producers must use this same order as OrthrusPerturbationManager;
        alphabetical sorting would silently assign masks to different components.
        """

        return list((case.component_to_edges or {}).keys())

    @staticmethod
    def validate_case_minimum(case: OrthrusAlertCase) -> None:
        if case.temporal_data is None:
            raise ValueError("OrthrusAlertCase.temporal_data is required")

        # We avoid inspecting PyG/Torch structures here on purpose.
        if not isinstance(case.metadata, dict):
            raise TypeError("OrthrusAlertCase.metadata must be a dict")


def _edge_attr(temporal_data, name: str, edge_index: int):
    if temporal_data is None or not hasattr(temporal_data, name):
        return None
    v = getattr(temporal_data, name)
    if v is None:
        return None
    try:
        return v[int(edge_index)]
    except Exception:
        return None


def _safe_int(value):
    if value is None:
        return None
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    try:
        s = str(value).strip()
        if s.isdigit():
            return int(s)
    except Exception:
        pass
    return value


def _edge_type_label(edge_type_value):
    if edge_type_value is None:
        return None
    if hasattr(edge_type_value, "item") and callable(getattr(edge_type_value, "item")):
        try:
            edge_type_value = edge_type_value.item()
        except Exception:
            pass
    if isinstance(edge_type_value, (int, str)):
        return edge_type_value
    if hasattr(edge_type_value, "tolist") and callable(getattr(edge_type_value, "tolist")):
        try:
            edge_type_value = edge_type_value.tolist()
        except Exception:
            pass
    if isinstance(edge_type_value, (list, tuple)) and edge_type_value:
        try:
            max_i = 0
            max_v = float(edge_type_value[0])
            for i in range(1, len(edge_type_value)):
                v = float(edge_type_value[i])
                if v > max_v:
                    max_v = v
                    max_i = i
            return int(max_i)
        except Exception:
            return str(edge_type_value)
    return str(edge_type_value)


def _apply_max_components(groups: Dict[str, List[int]], max_components: Optional[int]) -> Dict[str, List[int]]:
    if max_components is None or len(groups) <= max_components:
        return groups
    if max_components <= 0:
        raise ValueError("max_components must be > 0 or None")

    # Keep the biggest components, merge the rest into OTHER.
    keep = max(1, int(max_components) - 1)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    kept = ordered[:keep]
    rest = ordered[keep:]

    out: Dict[str, List[int]] = {k: v for k, v in kept}
    other_edges: List[int] = []
    for _, edges in rest:
        other_edges.extend(list(edges))
    out["OTHER"] = sorted(set(int(x) for x in other_edges))
    return out
