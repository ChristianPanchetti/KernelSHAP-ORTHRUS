from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from preprocessing.orthrus_alert_case import OrthrusAlertCase
from preprocessing.orthrus_join_keys import (
    build_join_key,
    decode_operation_from_temporal_data,
    extract_edge_join_keys,
)
import json
from pathlib import Path


SUPPORTED_MAPPING_MODES = {
    "artifact-only",
    "graph-assisted",
    "db-assisted",
    "sidecar-assisted",
}


class OrthrusMappingProvider(abc.ABC):
    """Populate mapping-aware fields on an `OrthrusAlertCase`.

    Design goal:
    - Keep SHAP/perturbation/builder logic independent from mapping.
    - Mapping is only used to enrich the case with optional metadata and to
      generate readable descriptions.

    Providers should:
    - set `case.mapping_mode`
    - (optionally) populate `edge_to_event_uuid`, `edge_to_log_record`,
      `edge_to_original_metadata`, `mapping_quality`

    NOTE:
    - This module intentionally avoids importing torch/torch_geometric at import time.
      Providers should work with simple Python objects exposing edge-aligned attributes.
    """

    mode: str

    def __init__(self, mode: str):
        mode = str(mode).strip().lower()
        if mode not in SUPPORTED_MAPPING_MODES:
            raise ValueError(f"Unsupported mapping_mode: {mode}. Supported: {sorted(SUPPORTED_MAPPING_MODES)}")
        self.mode = mode

    @abc.abstractmethod
    def enrich_case(self, case: OrthrusAlertCase) -> OrthrusAlertCase:
        """Populate mapping fields on `case`.

        Implementations may mutate `case` in-place and should return it.
        """

    def describe_edge(self, case: OrthrusAlertCase, edge_index: int) -> str:
        """Return a readable description for a single edge.

        Default implementation uses `edge_to_original_metadata` when present,
        otherwise falls back to best-effort extraction from `case.temporal_data`.
        """

        meta = (case.edge_to_original_metadata or {}).get(int(edge_index))
        if not meta:
            meta = _edge_metadata_from_temporal_data(case.temporal_data, int(edge_index))

        parts = [f"edge={edge_index}"]
        for k in ("src", "dst", "t", "t_iso", "edge_type"):
            if k in meta and meta[k] is not None:
                parts.append(f"{k}={meta[k]}")
        return ", ".join(parts)

    def describe_component(self, case: OrthrusAlertCase, component_id: str, max_edges: int = 5) -> str:
        """Return a readable description for a component (group of edges)."""

        component_id = str(component_id)
        edges = (case.component_to_edges or {}).get(component_id)
        if edges is None:
            return f"component={component_id} (not found)"

        n = len(edges)
        sample = list(edges[: max(0, int(max_edges))])
        sample_desc = "; ".join(self.describe_edge(case, i) for i in sample)
        if n > len(sample):
            sample_desc = sample_desc + f"; … (+{n - len(sample)} more)"

        return f"component={component_id} edges={n}: {sample_desc}".rstrip(": ")


class ArtifactOnlyMappingProvider(OrthrusMappingProvider):
    """Artifact-only mapping.

    Produces best-effort metadata solely from the TemporalData-like object:
    - edge index
    - src
    - dst
    - timestamp (t)
    - edge_type (if present)

    No event_uuid / DB join is attempted.
    """

    def __init__(self):
        super().__init__(mode="artifact-only")

    def enrich_case(self, case: OrthrusAlertCase) -> OrthrusAlertCase:
        case.mapping_mode = "artifact-only"

        temporal_data = case.temporal_data
        num_edges = infer_num_edges(temporal_data)
        if num_edges is None:
            raise ValueError(
                "artifact-only mapping requires temporal_data to expose an edge dimension (e.g., `.src` or `.edge_index`)."
            )

        # Populate best-effort per-edge metadata.
        edge_meta: Dict[int, Dict[str, Any]] = {}
        has_edge_type = hasattr(temporal_data, "edge_type")

        for i in range(num_edges):
            m = _edge_metadata_from_temporal_data(temporal_data, i)
            if not has_edge_type:
                m.pop("edge_type", None)
            edge_meta[i] = m

        case.edge_to_original_metadata = edge_meta

        # Quality/coverage summary.
        case.mapping_quality = {
            "mode": "artifact-only",
            "num_edges": int(num_edges),
            "has_src": hasattr(temporal_data, "src"),
            "has_dst": hasattr(temporal_data, "dst"),
            "has_t": hasattr(temporal_data, "t"),
            "has_edge_type": bool(has_edge_type),
            "note": "No event_uuid/DB join available in artifact-only mapping.",
        }

        return case


class DbAssistedMappingProvider(OrthrusMappingProvider):
    """DB-assisted mapping (stub).

    Intended future behavior:
    - join TemporalData edges to original events/logs stored in ORTHRUS Postgres
    - populate `edge_to_event_uuid` and/or `edge_to_log_record`

    This is intentionally NOT implemented yet.
    """

    def __init__(self):
        super().__init__(mode="db-assisted")
    def __init__(self, db_config: Optional[Dict] = None, join_strategy: str = "src_dst_t_operation",
                 rel2id: Optional[Dict[str, int]] = None, edge_type_slice: Optional[tuple[int, int]] = None,
                 fail_on_unmatched: bool = False):
        super().__init__(mode="db-assisted")
        self.db_config = db_config
        self.join_strategy = join_strategy
        self.rel2id = rel2id
        self.edge_type_slice = edge_type_slice
        self.fail_on_unmatched = bool(fail_on_unmatched)

    def enrich_case(self, case: OrthrusAlertCase, rows: Optional[Iterable[Dict]] = None) -> OrthrusAlertCase:
        """Attempt to enrich `case` by joining edges to provided `rows`.

        If `rows` is None and `db_config` is not set, raises NotImplementedError.
        If `rows` is provided (synthetic or loaded elsewhere), it will be used for matching.
        """
        case.mapping_mode = "db-assisted"

        edge_keys = self._build_edge_keys(case)
        if not edge_keys:
            case.mapping_quality = {"mode": "db-assisted", "note": "no edge keys"}
            return case

        if rows is None:
            if self.db_config is None:
                raise NotImplementedError("DbAssistedMappingProvider requires db_config for live DB fetch or rows param for synthetic tests.")
            rows = self._fetch_event_rows(min_t=None, max_t=None)

        # normalize rows into dict keyed by join key
        row_map = {}
        for r in rows:
            k = (int(r.get("src_index_id")) if r.get("src_index_id") is not None else None,
                 int(r.get("dst_index_id")) if r.get("dst_index_id") is not None else None,
                 int(r.get("timestamp_rec")) if r.get("timestamp_rec") is not None else None,
                 str(r.get("operation")) if r.get("operation") is not None else None)
            row_map.setdefault(k, []).append(r)

        matched = 0
        edge_to_uuid = {}
        edge_meta = {}
        for i, k in enumerate(edge_keys):
            hits = row_map.get(k) or []
            if len(hits) == 1:
                ev = hits[0].get("event_uuid")
                edge_to_uuid[i] = ev
                edge_meta[i] = hits[0]
                matched += 1
            elif len(hits) > 1:
                # collisions: pick first as representative
                ev = hits[0].get("event_uuid")
                edge_to_uuid[i] = ev
                edge_meta[i] = hits[0]
            else:
                # unmatched
                pass

        case.edge_to_event_uuid = edge_to_uuid
        case.edge_to_original_metadata = edge_meta
        num_edges = len(edge_keys)
        case.mapping_quality = {
            "mode": "db-assisted",
            "join_strategy": self.join_strategy,
            "num_edges": num_edges,
            "matched": matched,
            "unmatched": num_edges - matched,
            "match_rate": matched / num_edges if num_edges else 0.0,
        }

        if self.fail_on_unmatched and matched != num_edges:
            raise RuntimeError("Unmatched edges while fail_on_unmatched=True")

        return case

    def _build_edge_keys(self, case: OrthrusAlertCase):
        return extract_edge_join_keys(case, rel2id=self.rel2id, edge_type_slice=self.edge_type_slice)

    def _fetch_event_rows(self, min_t: Optional[int], max_t: Optional[int]):
        # Live DB fetching not implemented in tests. Real implementation would use psycopg2/asyncpg.
        raise NotImplementedError("DB fetching not implemented in offline provider. Pass `rows` to `enrich_case` for testing.")

    def _match_edges_to_rows(self, edge_keys, rows: Iterable[Dict]):
        # Helper to match keys to rows. Kept separate for unit testing.
        row_map = {}
        for r in rows:
            k = (int(r.get("src_index_id")) if r.get("src_index_id") is not None else None,
                 int(r.get("dst_index_id")) if r.get("dst_index_id") is not None else None,
                 int(r.get("timestamp_rec")) if r.get("timestamp_rec") is not None else None,
                 str(r.get("operation")) if r.get("operation") is not None else None)
            row_map.setdefault(k, []).append(r)

        mapping = {}
        for i, k in enumerate(edge_keys):
            hits = row_map.get(k) or []
            if hits:
                mapping[i] = hits
        return mapping


class GraphAssistedMappingProvider(OrthrusMappingProvider):
    """Graph-assisted mapping via pre-embedding graphs (stub).

    Intended future behavior:
    - load the pre-embedding NetworkX MultiDiGraph windows (saved during graph construction)
    - align TemporalData edge indices to those edges and recover event_uuid/labels

    This is intentionally NOT implemented yet.
    """

    def __init__(self):
        super().__init__(mode="graph-assisted")

    def enrich_case(self, case: OrthrusAlertCase) -> OrthrusAlertCase:
        raise NotImplementedError(
            "Graph-assisted mapping is not implemented yet. Required inputs typically include: "
            "(1) access to the pre-embedding graph artifacts (e.g., saved NetworkX MultiDiGraph windows), "
            "(2) a deterministic edge ordering/alignment strategy between those graphs and TemporalData edges, "
            "(3) the edge attributes needed for mapping (event_uuid/time/operation/labels). "
            "If alignment is not deterministic, consider writing an explicit edge_index->event_uuid sidecar during preprocessing."
        )


class SidecarMappingProvider(OrthrusMappingProvider):
    """Load mapping from a sidecar JSON/JSONL/CSV file keyed by edge index.

    Sidecar format (JSON) example:
    {
      "mapping_mode": "sidecar-assisted",
      "join_strategy": "edge_index",
      "edges": {
         "0": {"event_uuid": "...", "src_index_id": 1, ...}
      }
    }
    """

    def __init__(self, sidecar_path: str):
        super().__init__(mode="sidecar-assisted")
        self.sidecar_path = str(sidecar_path)

    def enrich_case(self, case: OrthrusAlertCase) -> OrthrusAlertCase:
        case.mapping_mode = "sidecar-assisted"

        p = Path(self.sidecar_path)
        if not p.exists():
            raise FileNotFoundError(f"Sidecar not found: {self.sidecar_path}")

        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        edges = data.get("edges") or {}

        edge_to_uuid = {}
        edge_to_meta = {}
        sidecar_edge_indices = set()
        for k, v in edges.items():
            try:
                idx = int(k)
            except Exception:
                continue
            sidecar_edge_indices.add(idx)
            if isinstance(v, dict):
                ev = v.get("event_uuid")
                if ev:
                    edge_to_uuid[idx] = ev
                edge_to_meta[idx] = v

        # Count matched vs unmatched
        num_edges = infer_num_edges(case.temporal_data) or 0
        matched = sum(1 for i in range(num_edges) if i in edge_to_uuid)
        unmatched = num_edges - matched

        case.edge_to_event_uuid = edge_to_uuid
        case.edge_to_original_metadata = edge_to_meta
        case.mapping_quality = {
            "mode": "sidecar-assisted",
            "join_strategy": data.get("join_strategy", "edge_index"),
            "num_edges": num_edges,
            "matched": matched,
            "unmatched": unmatched,
            "match_rate": matched / num_edges if num_edges else 0.0,
            "sidecar_path": self.sidecar_path,
            "warnings": [] if unmatched == 0 else [f"{unmatched} unmatched edges"],
            "extra_sidecar_edges": sorted([i for i in sidecar_edge_indices if i >= num_edges]),
        }

        return case


def get_mapping_provider(mapping_mode: str) -> OrthrusMappingProvider:
    """Factory for mapping providers."""

    mode = str(mapping_mode).strip().lower()
    if mode == "artifact-only":
        return ArtifactOnlyMappingProvider()
    if mode == "db-assisted":
        return DbAssistedMappingProvider()
    if mode == "graph-assisted":
        return GraphAssistedMappingProvider()
    if mode == "sidecar-assisted":
        # caller should instantiate SidecarMappingProvider with path; factory shortcut not used.
        raise ValueError("Use SidecarMappingProvider(sidecar_path) directly for sidecar-assisted mode.")
    raise ValueError(f"Unsupported mapping_mode: {mode}. Supported: {sorted(SUPPORTED_MAPPING_MODES)}")


def describe_component(case: OrthrusAlertCase, component_id: str, max_edges: int = 5) -> str:
    """Public helper to describe a component without exposing provider internals.

    This is the single function the SHAP/reporting layer should call.
    Adding new mapping modes should not require changing SHAP logic.
    """

    mode = str(case.mapping_mode).strip().lower()
    if mode == "artifact-only":
        provider = get_mapping_provider(mode)
        return provider.describe_component(case, component_id=component_id, max_edges=max_edges)

    if mode == "db-assisted":
        # Future: use `edge_to_event_uuid` / `edge_to_log_record` once populated.
        if (case.edge_to_event_uuid or {}) or (case.edge_to_log_record or {}):
            provider = get_mapping_provider("artifact-only")
            base = provider.describe_component(case, component_id=component_id, max_edges=max_edges)
            return f"{base} (db-assisted details available)"
        return (
            f"component={component_id}: db-assisted mapping not available yet. "
            "Populate OrthrusAlertCase.edge_to_event_uuid and/or edge_to_log_record via DbAssistedMappingProvider."
        )

    if mode == "sidecar-assisted":
        if (case.edge_to_event_uuid or {}) or (case.edge_to_original_metadata or {}):
            provider = get_mapping_provider("artifact-only")
            base = provider.describe_component(case, component_id=component_id, max_edges=max_edges)
            return f"{base} (sidecar-assisted details available)"
        return (
            f"component={component_id}: sidecar-assisted mapping not available. "
            "Provide a SidecarMappingProvider and a valid sidecar file to populate edge_to_event_uuid."
        )

    if mode == "graph-assisted":
        # Future: use pre-embedding graphs to recover event_uuid and richer semantics.
        if (case.edge_to_event_uuid or {}):
            provider = get_mapping_provider("artifact-only")
            base = provider.describe_component(case, component_id=component_id, max_edges=max_edges)
            return f"{base} (graph-assisted details available)"
        return (
            f"component={component_id}: graph-assisted mapping not available yet. "
            "Provide pre-embedding graph artifacts and an alignment strategy to populate edge_to_event_uuid."
        )

    # Unknown mode: fail safe to artifact-only description.
    provider = get_mapping_provider("artifact-only")
    return provider.describe_component(case, component_id=component_id, max_edges=max_edges)


def infer_num_edges(temporal_data: Any) -> Optional[int]:
    """Try to infer the number of edges/events in a TemporalData-like object."""

    if temporal_data is None:
        return None

    if hasattr(temporal_data, "src"):
        try:
            return int(len(getattr(temporal_data, "src")))
        except Exception:
            pass

    if hasattr(temporal_data, "edge_index"):
        ei = getattr(temporal_data, "edge_index")
        try:
            # edge_index is typically shape (2, E)
            if hasattr(ei, "shape") and len(getattr(ei, "shape")) == 2:
                return int(ei.shape[1])
            return int(len(ei[0]))
        except Exception:
            pass

    return None


def _edge_metadata_from_temporal_data(temporal_data: Any, edge_index: int) -> Dict[str, Any]:
    src = _safe_scalar(_get_edge_aligned_attr(temporal_data, "src", edge_index))
    dst = _safe_scalar(_get_edge_aligned_attr(temporal_data, "dst", edge_index))
    t = _safe_scalar(_get_edge_aligned_attr(temporal_data, "t", edge_index))

    meta: Dict[str, Any] = {
        "edge_index": int(edge_index),
        "src": _maybe_int(src),
        "dst": _maybe_int(dst),
        "t": _maybe_int(t) if t is not None else None,
    }

    # Best-effort timestamp formatting (do NOT assume units).
    if isinstance(meta.get("t"), int):
        t_iso = _try_format_timestamp_iso(meta["t"])
        if t_iso is not None:
            meta["t_iso"] = t_iso

    if hasattr(temporal_data, "edge_type"):
        et_raw = _get_edge_aligned_attr(temporal_data, "edge_type", edge_index)
        meta["edge_type"] = _edge_type_label(et_raw)

    return meta


def _get_edge_aligned_attr(obj: Any, name: str, edge_index: int) -> Any:
    if obj is None or not hasattr(obj, name):
        return None
    v = getattr(obj, name)
    if v is None:
        return None

    # Common case: list / numpy / torch indexing.
    try:
        return v[int(edge_index)]
    except Exception:
        pass

    # Some objects store edge_index-aligned attributes as dicts.
    if isinstance(v, dict):
        return v.get(int(edge_index))

    return None


def _safe_scalar(value: Any) -> Any:
    if value is None:
        return None

    # Torch scalar tensors / numpy scalars.
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass

    return value


def _maybe_int(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    try:
        # numpy int/float, or strings like "123"
        if str(value).strip().isdigit():
            return int(str(value).strip())
    except Exception:
        pass
    return value


def _edge_type_label(edge_type_value: Any) -> Any:
    """Return a stable edge_type label.

    Supports:
    - scalar int/str
    - one-hot / vector-like list/tuple
    - numpy/torch tensors (best-effort)
    """

    v = edge_type_value
    if v is None:
        return None

    v = _safe_scalar(v)

    if isinstance(v, (int, str)):
        return v

    # Try to handle 1D vector-like
    if isinstance(v, (list, tuple)) and v:
        try:
            # argmax
            max_i = 0
            max_val = float(v[0])
            for i in range(1, len(v)):
                fv = float(v[i])
                if fv > max_val:
                    max_val = fv
                    max_i = i
            return int(max_i)
        except Exception:
            # fallback to string repr
            return str(v)

    # numpy/torch 1D tensors: try `tolist()`.
    if hasattr(v, "tolist") and callable(getattr(v, "tolist")):
        try:
            as_list = v.tolist()
            if isinstance(as_list, list):
                return _edge_type_label(as_list)
        except Exception:
            pass

    return str(v)


def _try_format_timestamp_iso(t: int) -> Optional[str]:
    """Best-effort formatting for timestamps.

    Heuristic:
    - if t looks like nanoseconds (very large), convert using t / 1e9
    - if t looks like milliseconds, convert using t / 1e3
    - else assume seconds

    Returns None if formatting fails.
    """

    try:
        if t > 10**14:
            seconds = t / 1e9
        elif t > 10**11:
            seconds = t / 1e9
        elif t > 10**10:
            seconds = t / 1e3
        else:
            seconds = float(t)
        dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return None
