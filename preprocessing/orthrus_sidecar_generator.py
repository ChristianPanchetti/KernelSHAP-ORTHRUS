from __future__ import annotations

import json
import pickle
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from preprocessing.orthrus_join_keys import (
    build_join_key, extract_edge_join_keys,
    decode_operation_from_temporal_data, artifact_sha256,
)


class OrthrusSidecarGenerator:
    """Generate and validate ORTHRUS sidecar mapping files.

    The sidecar is a JSON file that connects a TemporalData edge index to the
    original ORTHRUS event metadata (most importantly: `event_uuid`).

    Design constraints:
    - Keep this module usable even without `torch` (tests in this repo are torch-free).
    - When `torch` is available, support loading ORTHRUS artifacts saved via `torch.save`.
    - Do not modify external ORTHRUS code.

    Security note:
    - Loading graph/TemporalData artifacts (pickle/torch.save) is inherently unsafe.
      Only load trusted files.
    """

    def __init__(self, *, prefer_operation: bool = True, rel2id=None, edge_type_slice=None):
        self.prefer_operation = bool(prefer_operation)
        self.rel2id = rel2id
        self.edge_type_slice = edge_type_slice

    def generate_from_graph(self, graph_path: str, output_path: str,
                            *, temporal_data_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate a sidecar from a pre-embedding NetworkX MultiDiGraph.

        Expected edge attributes (as produced by ORTHRUS graph construction):
        - event_uuid
        - time (timestamp_rec)
        - label (operation)

        If the graph was saved with `torch.save`, this method will use `torch.load`
        when available, otherwise it will try a best-effort zip+pickle fallback.
        """

        graph = _load_networkx_graph(graph_path)

        edges_out: Dict[str, Dict[str, Any]] = {}
        missing_uuid_examples: List[int] = []

        total_edges = 0
        graph_keys = []
        for i, (u, v, k, attr) in enumerate(graph.edges(data=True, keys=True)):
            total_edges += 1
            attr_dict = dict(attr) if isinstance(attr, dict) else {}

            graph_keys.append(build_join_key(u, v, attr_dict.get("time"), attr_dict.get("label")))
            event_uuid = attr_dict.get("event_uuid")
            if not event_uuid:
                if len(missing_uuid_examples) < 10:
                    missing_uuid_examples.append(int(i))
                continue

            raw_meta = {kk: vv for kk, vv in attr_dict.items() if kk not in {"event_uuid", "time", "label"}}
            raw_meta.setdefault("nx_key", k)

            edges_out[str(i)] = {
                "event_uuid": str(event_uuid),
                "src_index_id": _maybe_int(u),
                "dst_index_id": _maybe_int(v),
                "timestamp_rec": _maybe_int(attr_dict.get("time")),
                "operation": str(attr_dict.get("label")) if attr_dict.get("label") is not None else None,
                "raw_metadata": _jsonify(raw_meta),
            }

        num_edges = int(total_edges)
        matched = int(len(edges_out))
        unmatched = int(num_edges - matched)

        warnings: List[str] = []
        if unmatched:
            extra = "" if not missing_uuid_examples else f" (examples: {missing_uuid_examples})"
            warnings.append(f"{unmatched} edges missing event_uuid{extra}")

        sidecar = {
            "mapping_mode": "sidecar-assisted",
            "source": str(graph_path),
            "generated_at": _now_iso_utc(),
            "join_strategy": "edge_index",
            "num_edges": num_edges,
            "matched": matched,
            "unmatched": unmatched,
            "warnings": warnings,
            "edges": edges_out,
        }

        if temporal_data_path is not None:
            td = _load_temporal_data_like(temporal_data_path)
            keys = extract_edge_join_keys(SimpleNamespace(temporal_data=td), self.rel2id, self.edge_type_slice)
            if keys != graph_keys or any(None in key for key in keys):
                raise ValueError("Graph order/keys do not match the supplied TemporalData artifact")
            sidecar["provenance"] = {"method": "graph_order_verified", "edge_index_scope": "graph",
                                     "artifact_sha256": artifact_sha256(temporal_data_path)}
        _write_json(sidecar, output_path)
        return sidecar

    def generate_from_temporal_data_and_rows(
        self,
        temporal_data_path: str,
        rows: Iterable[Dict[str, Any]],
        output_path: str,
    ) -> Dict[str, Any]:
        """Generate a sidecar by joining TemporalData edges to `rows`.

        This method is intended to be DB-free and testable: pass `rows` as a list of dicts.

        Supported TemporalData inputs:
        - JSON file with keys: src, dst, t and optionally operation/edge_type
        - Pickle file of a python object exposing src/dst/t attributes
        - `torch.save` artifact when `torch` is installed

        Row format (dict keys expected):
        - src_index_id, dst_index_id, timestamp_rec
        - operation (optional; used if operations are available from TemporalData)
        - event_uuid
        """

        td = _load_temporal_data_like(temporal_data_path)
        src, dst, t = _extract_edge_arrays(td)
        num_edges = min(len(src), len(dst), len(t))

        ops = decode_operation_from_temporal_data(td, self.rel2id, self.edge_type_slice) if self.prefer_operation else []
        include_operation = len(ops) == num_edges and all(op is not None for op in ops)

        join_strategy = "src_dst_t_operation" if include_operation else "src_dst_t"

        rows = list(rows)
        row_map_full = _index_rows(rows, include_operation=True)
        row_map_partial = _index_rows(rows, include_operation=False)

        edges_out: Dict[str, Dict[str, Any]] = {}
        collisions = 0
        matched = 0

        for i in range(num_edges):
            op = ops[i] if include_operation else None
            key = build_join_key(src[i], dst[i], t[i], op if include_operation else None)
            hits = row_map_full.get(key, []) if include_operation else row_map_partial.get(key, [])

            chosen = hits[0] if len(hits) == 1 else None
            if hits and len(hits) > 1:
                collisions += 1

            event_uuid = None
            if chosen is not None:
                event_uuid = chosen.get("event_uuid")
                if event_uuid:
                    matched += 1

            operation_out = None
            if chosen is not None and chosen.get("operation") is not None:
                operation_out = str(chosen.get("operation"))
            elif op is not None:
                operation_out = str(op)

            raw_metadata = _extract_raw_metadata(chosen) if isinstance(chosen, dict) else {}

            if event_uuid:
                edges_out[str(i)] = {
                    "event_uuid": str(event_uuid),
                    "src_index_id": _maybe_int(src[i]),
                    "dst_index_id": _maybe_int(dst[i]),
                    "timestamp_rec": _maybe_int(t[i]),
                    "operation": operation_out,
                    "raw_metadata": _jsonify(raw_metadata),
                }

        unmatched = num_edges - matched
        warnings: List[str] = []
        if not include_operation:
            warnings.append(
                "Operation not available from TemporalData; join performed on (src,dst,t) only. "
                "This may be ambiguous if multiple events share the same (src,dst,t)."
            )
        if collisions:
            warnings.append(f"{collisions} join-key collisions; UUIDs left unresolved")
        if unmatched:
            warnings.append(f"{unmatched} edges unmatched")

        sidecar = {
            "mapping_mode": "sidecar-assisted",
            "source": str(temporal_data_path),
            "generated_at": _now_iso_utc(),
            "join_strategy": join_strategy,
            "num_edges": int(num_edges),
            "matched": int(matched),
            "unmatched": int(unmatched),
            "warnings": warnings,
            "edges": edges_out,
        }

        if include_operation:
            sidecar["provenance"] = {"method": "unique_complete_join", "edge_index_scope": "graph",
                                     "artifact_sha256": artifact_sha256(temporal_data_path)}
        _write_json(sidecar, output_path)
        return sidecar

    def validate_sidecar(self, sidecar_path: str, temporal_data_path: Optional[str] = None) -> Dict[str, Any]:
        """Validate a sidecar file.

        Returns a report dict. This method is intentionally non-throwing for
        "coverage" issues (missing/extra/unmapped edges), but it will throw on
        unreadable JSON or if the sidecar is structurally invalid.
        """

        p = Path(sidecar_path)
        if not p.exists():
            raise FileNotFoundError(f"Sidecar not found: {sidecar_path}")

        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        errors: List[str] = []
        warnings: List[str] = []

        mode = data.get("mapping_mode")
        if mode != "sidecar-assisted":
            warnings.append(f"Unexpected mapping_mode={mode!r} (expected 'sidecar-assisted')")

        edges = data.get("edges")
        if not isinstance(edges, dict):
            errors.append("'edges' must be a dict keyed by edge index")
            raise ValueError("; ".join(errors))

        parsed_indices: List[int] = []
        for k in edges.keys():
            try:
                parsed_indices.append(int(k))
            except Exception:
                errors.append(f"Invalid edge index key: {k!r}")

        if errors:
            raise ValueError("; ".join(errors))

        parsed_indices = sorted(set(parsed_indices))

        num_edges_td: Optional[int] = None
        if temporal_data_path is not None:
            td = _load_temporal_data_like(temporal_data_path)
            src, dst, t = _extract_edge_arrays(td)
            num_edges_td = min(len(src), len(dst), len(t))

        missing_keys: List[int] = []
        extra_keys: List[int] = []
        unmapped_present: List[int] = []

        if num_edges_td is not None:
            idx_set = set(parsed_indices)
            missing_keys = [i for i in range(num_edges_td) if i not in idx_set]
            extra_keys = [i for i in parsed_indices if i >= num_edges_td]

            # Present but event_uuid missing/empty
            for i in range(num_edges_td):
                v = edges.get(str(i))
                if isinstance(v, dict):
                    ev = v.get("event_uuid")
                    if not ev:
                        unmapped_present.append(i)

        report = {
            "valid": True,
            "mapping_mode": mode,
            "sidecar_path": str(sidecar_path),
            "num_edges_sidecar": len(parsed_indices),
            "num_edges_temporal": num_edges_td,
            "missing_edge_keys": missing_keys,
            "extra_edge_keys": extra_keys,
            "present_but_unmapped": unmapped_present,
            "warnings": warnings + list(data.get("warnings") or []),
        }

        # Cross-check declared num_edges if present.
        declared_num_edges = data.get("num_edges")
        if num_edges_td is not None and declared_num_edges is not None:
            try:
                if int(declared_num_edges) != int(num_edges_td):
                    report["warnings"].append(
                        f"Declared num_edges={declared_num_edges} differs from temporal_data num_edges={num_edges_td}"
                    )
            except Exception:
                report["warnings"].append(f"Non-integer num_edges field: {declared_num_edges!r}")

        return report


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(data: Dict[str, Any], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _load_networkx_graph(path: str) -> Any:
    """Best-effort loader for a NetworkX graph.

    Supports:
    - plain pickle (.pkl/.pickle)
    - ORTHRUS artifacts saved via `torch.save` (requires torch, or zip+pickle fallback)
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Graph not found: {path}")

    # 1) Plain pickle
    if p.suffix.lower() in {".pkl", ".pickle", ".gpickle"}:
        with p.open("rb") as f:
            return pickle.load(f)

    # 2) torch.load when available
    try:
        import torch  # type: ignore

        return torch.load(str(p), map_location="cpu")
    except ModuleNotFoundError:
        pass

    # 3) torch zipfile fallback (pure-python objects only)
    if zipfile.is_zipfile(p):
        try:
            with zipfile.ZipFile(p, "r") as zf:
                name = "data.pkl"
                if name not in zf.namelist():
                    name = next((n for n in zf.namelist() if n.endswith("data.pkl")), None)
                if not name:
                    raise RuntimeError("Torch zipfile does not contain data.pkl")
                payload = zf.read(name)
            return pickle.loads(payload)
        except Exception as e:
            raise RuntimeError(
                "Unable to load torch-saved graph without torch. Install torch or provide a pickled NetworkX graph. "
                f"Details: {e}"
            )

    raise RuntimeError("Unsupported graph format. Provide a .pkl/.pickle graph or install torch for torch.save artifacts.")


def _load_temporal_data_like(path: str) -> Any:
    """Load a TemporalData-like artifact.

    Supports:
    - JSON: {src: [...], dst: [...], t: [...], operation/edge_type: [...]}
    - pickle (.pkl/.pickle) of an object exposing src/dst/t attributes
    - torch.save artifact when torch is installed

    NOTE: If you pass a torch.save TemporalData file without torch installed,
    this will raise an error (TemporalData typically contains torch tensors).
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"TemporalData not found: {path}")

    if p.suffix.lower() in {".json"}:
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            raise ValueError("TemporalData JSON must be an object/dict")
        return SimpleNamespace(**d)

    if p.suffix.lower() in {".pkl", ".pickle"}:
        with p.open("rb") as f:
            return pickle.load(f)

    try:
        import torch  # type: ignore

        return torch.load(str(p), map_location="cpu")
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Torch is required to load this TemporalData artifact. "
            "Either install torch or pass a JSON/.pkl TemporalData-like file for sidecar generation."
        ) from e


def _extract_edge_arrays(td: Any) -> Tuple[List[Any], List[Any], List[Any]]:
    arrays = []
    for field in ("src", "dst", "t"):
        value = getattr(td, field, None)
        if value is None:
            raise ValueError(f"TemporalData mapping requires {field}")
        arrays.append(list(value))
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("TemporalData mapping requires aligned src/dst/t")
    return tuple(arrays)


def _index_rows(rows: Iterable[Dict[str, Any]], *, include_operation: bool) -> Dict[Tuple, List[Dict[str, Any]]]:
    m: Dict[Tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        key = build_join_key(
            r.get("src_index_id"),
            r.get("dst_index_id"),
            r.get("timestamp_rec"),
            r.get("operation") if include_operation else None,
        )
        m.setdefault(key, []).append(r)
    return m


def _maybe_int(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    try:
        s = str(v).strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
    except Exception:
        pass
    return v


def _jsonify(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable structures."""

    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}

    # Fall back to string representation
    return str(obj)


def _extract_raw_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the `raw_metadata` payload from a DB/event row.

    The sidecar schema reserves a small set of join-related keys at top-level.
    All other information is collected into this nested dict.
    """

    if not isinstance(row, dict):
        return {}

    known = {"event_uuid", "src_index_id", "dst_index_id", "timestamp_rec", "operation", "raw_metadata"}

    out: Dict[str, Any] = {}
    if isinstance(row.get("raw_metadata"), dict):
        out.update(row.get("raw_metadata") or {})

    for k, v in row.items():
        if k in known or str(k).startswith("_"):
            continue
        out.setdefault(str(k), v)

    return out
