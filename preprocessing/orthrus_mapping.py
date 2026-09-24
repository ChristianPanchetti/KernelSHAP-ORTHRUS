from __future__ import annotations

import abc
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from preprocessing.orthrus_alert_case import OrthrusAlertCase
from preprocessing.orthrus_join_keys import (
    build_join_key,
    extract_edge_join_keys,
    case_mapping_fingerprint,
    artifact_sha256,
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

        meta = {**_edge_metadata_from_temporal_data(case.temporal_data, int(edge_index)),
                **case.edge_to_original_metadata.get(int(edge_index), {})}
        parts = [f"edge={edge_index} (indice locale)"]
        for role, key in (("src", "src_index_id"), ("dst", "dst_index_id")):
            node_id = meta.get(key, meta.get(role))
            parts.append(f"{role}={node_id} [{_describe_node(case, node_id)}]")
        for key in ("edge_type", "operation", "event_uuid", "status", "graph_edge_index"):
            if meta.get(key) is not None:
                parts.append(f"{key}={meta[key]}")
        timestamp = meta.get("timestamp_rec", meta.get("t"))
        parts.append(f"timestamp={timestamp}")
        if "timestamp_rec" in meta:
            parts.append("unità timestamp=ns (THEIA_E5)")
        provenance = meta.get("provenance", {})
        source = provenance.get("source", case.mapping_mode) if isinstance(provenance, dict) else provenance
        parts.append(f"provenienza={source}")
        return ", ".join(parts)

    def describe_component(self, case: OrthrusAlertCase, component_id: str, max_edges: int = 5) -> str:
        component_id = str(component_id)
        edges = case.component_to_edges.get(component_id)
        if edges is None:
            return f"component={component_id} (not found)"
        sample = list(edges[:max(0, int(max_edges))])
        detail = "; ".join(self.describe_edge(case, i) for i in sample)
        if len(edges) > len(sample):
            detail += f"; … (+{len(edges) - len(sample)} more)"
        result = f"component={component_id} edges={len(edges)} mapping={case.mapping_mode}: {detail}"
        if hasattr(case.temporal_data, "src") and hasattr(case.temporal_data, "dst"):
            roles = _component_roles(case, component_id)
            result += (f"; edge assegnati={roles['assigned_edges']}; "
                       f"nodi source={roles['source_node_ids']}; nodi destination={roles['destination_node_ids']}; "
                       f"righe x_src se disattivata={roles['source_rows_if_disabled']}; "
                       f"righe x_dst se disattivata={roles['destination_rows_if_disabled']}. "
                       "Neutralizzazione delle feature per nodo e ruolo, senza eliminazione degli eventi.")
        if "neutralized_source_rows" in case.metadata:
            result += (f" Righe effettivamente neutralizzate dalla mask corrente: "
                       f"x_src={case.metadata['neutralized_source_rows']}, "
                       f"x_dst={case.metadata.get('neutralized_destination_rows', [])}.")
        return result


def _describe_node(case, node_id):
    nodes = case.metadata.get("node_mapping", {})
    node = nodes.get(node_id, nodes.get(str(node_id), {}))
    fields = [f"stato={node.get('status', 'not_enriched')}"]
    for key in ("node_uuid", "node_type", "path", "cmd", "src_addr", "src_port", "dst_addr", "dst_port",
                "missing_fields", "unverified_fields", "provenance"):
        if node.get(key) not in (None, [], ""):
            fields.append(f"{key}={node[key]}")
    return ", ".join(fields)


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
    """Read-only THEIA_E5 enrichment, invoked once outside SHAP score evaluation.

    Offline node_rows need a node_type: subject, file or netflow. Supplied rows
    must contain all candidates for the requested keys; truncated exports cannot
    establish uniqueness. db_config is passed only to a lazy psycopg2 connection.
    Network columns are withheld unless explicitly audited by the caller.
    """

    NODE_COLUMNS = {
        "subject": ("index_id", "node_uuid", "path", "cmd"),
        "file": ("index_id", "node_uuid", "path"),
        "netflow": ("index_id", "node_uuid", "src_addr", "src_port", "dst_addr", "dst_port"),
    }
    EVENT_COLUMNS = ("src_index_id", "dst_index_id", "timestamp_rec", "operation", "event_uuid", "_id")
    NETWORK_COLUMNS = ("src_addr", "src_port", "dst_addr", "dst_port")

    def __init__(self, db_config: Optional[Dict] = None, join_strategy: str = "src_dst_t_operation",
                 rel2id: Optional[Dict] = None, edge_type_slice: Optional[tuple[int, int]] = None,
                 fail_on_unmatched: bool = False, *, verified_network_fields=()):
        super().__init__(mode="db-assisted")
        if join_strategy != "src_dst_t_operation":
            raise ValueError("Only src_dst_t_operation is supported")
        if not set(verified_network_fields).issubset(self.NETWORK_COLUMNS):
            raise ValueError("Unknown network metadata field")
        self.db_config = db_config
        self.join_strategy = join_strategy
        self.rel2id = rel2id
        self.edge_type_slice = edge_type_slice
        self.fail_on_unmatched = bool(fail_on_unmatched)
        self.verified_network_fields = tuple(verified_network_fields)

    def enrich_case(self, case: OrthrusAlertCase, rows: Optional[Iterable[Dict]] = None,
                    *, node_rows: Optional[Iterable[Dict]] = None, component_ids=None) -> OrthrusAlertCase:
        """Enrich selected components (all case edges by default), without touching tensors.

        rows/node_rows, when supplied, replace the corresponding database fetch.
        Offline callers may supply only node_rows; event resolution then remains missing.
        Each call replaces the previous DB enrichment with its explicitly selected scope.
        """
        if rows is None and node_rows is None and self.db_config is None:
            raise NotImplementedError("Supply offline rows/node_rows or an optional PostgreSQL configuration")
        keys = self._build_edge_keys(case)
        if component_ids is None:
            selected = list(range(len(keys)))
            components = list(case.component_to_edges)
        else:
            components = list(dict.fromkeys(component_ids))
            if any(c not in case.component_to_edges for c in components):
                raise ValueError("Unknown component requested for mapping")
            selected = sorted({i for c in components for i in case.component_to_edges[c]})
        if any(not isinstance(i, int) or i < 0 or i >= len(keys) for i in selected):
            raise ValueError("Component edge index outside current batch")
        node_ids = sorted({n for i in selected for n in keys[i][:2] if n is not None})
        fingerprint = case_mapping_fingerprint(case, keys)
        verified = {}
        for i in selected:
            meta = case.edge_to_original_metadata.get(i, {})
            provenance = meta.get("provenance", {})
            if not isinstance(provenance, dict):
                continue
            identity_source = meta.get("identity_source", {})
            sidecar_identity = (provenance.get("source") == "verified_sidecar" or
                                (provenance.get("source") == "verified_identity"
                                 and identity_source.get("source") == "verified_sidecar"))
            uuid = case.edge_to_event_uuid.get(i)
            if (uuid and sidecar_identity and provenance.get("identity_verified") is True
                    and provenance.get("case_fingerprint") == fingerprint
                    and tuple(provenance.get("verified_key", ())) == keys[i]
                    and meta.get("event_uuid") == uuid):
                verified[i] = uuid

        event_source = "offline_rows" if rows is not None else "not_queried"
        node_source = "offline_rows" if node_rows is not None else "not_queried"
        if self.db_config is not None and (rows is None or node_rows is None) and selected:
            fetched_nodes, fetched_events = self._fetch_rows(
                node_ids if node_rows is None else [],
                sorted({keys[i] for i in selected if i not in verified and None not in keys[i]})
                if rows is None else [],
                sorted(set(verified.values())) if rows is None else [])
            if node_rows is None:
                node_rows, node_source = fetched_nodes, "postgresql"
            if rows is None:
                rows, event_source = fetched_events, "postgresql"
        rows = list(rows) if rows is not None else []
        node_rows = list(node_rows) if node_rows is not None else []

        nodes = self._resolve_nodes(node_ids, node_rows, node_source)
        matches = self._match_edges_to_rows(keys, rows)
        uuid_rows = {}
        for row in rows:
            if row.get("event_uuid"):
                uuid_rows.setdefault(str(row["event_uuid"]), []).append(row)
        edge_meta, edge_uuids = {}, {}
        for i in selected:
            key = keys[i]
            meta = {"edge_index": i, "src_index_id": key[0], "dst_index_id": key[1],
                    "timestamp_rec": key[2], "operation": key[3]}
            origin = event_source
            uuid = verified.get(i)
            hits = uuid_rows.get(uuid, []) if uuid else matches.get(i, [])
            status = "missing"
            if uuid:
                origin = "verified_identity"
                old = case.edge_to_original_metadata[i]
                meta["identity_source"] = dict(old.get("identity_source") or old.get("provenance", {}))
                meta["graph_edge_index"] = old.get("graph_edge_index")
                # UUID lookup must still agree with the graph's oriented edge.
                if any(_event_row_key(row) != key for row in hits):
                    status = "conflict"
                elif len(hits) > 1:
                    status = "ambiguous"
                else:
                    status = "resolved"
                    meta["database_status"] = ("resolved" if hits else
                                               "not_queried" if event_source == "not_queried" else "missing")
            elif None in key:
                status = "unresolvable"
            elif len(hits) > 1:
                status = "ambiguous"
            elif len(hits) == 1:
                uuid = hits[0].get("event_uuid")
                status = "resolved" if isinstance(uuid, str) and uuid.strip() else "unresolvable"
            meta["status"] = status
            meta["candidate_count"] = len(hits)
            if status == "unresolvable":
                meta["missing_fields"] = [name for name, value in zip(
                    ("src_index_id", "dst_index_id", "timestamp_rec", "operation"), key) if value is None]
                if len(hits) == 1 and not uuid:
                    meta["missing_fields"].append("event_uuid")
            meta["provenance"] = {"source": origin, "identity_verified": status == "resolved",
                                  "case_fingerprint": fingerprint, "verified_key": list(key)}
            if status == "resolved":
                meta["event_uuid"] = str(uuid)
                edge_uuids[i] = str(uuid)
                if hits:
                    meta["database_row_id"] = hits[0].get("_id")
            edge_meta[i] = meta

        event_counts = dict(Counter(m["status"] for m in edge_meta.values()))
        node_counts = dict(Counter(m["status"] for m in nodes.values()))
        case.mapping_mode = self.mode
        case.edge_to_event_uuid = edge_uuids
        case.edge_to_original_metadata = edge_meta
        case.edge_to_log_record = {}  # No stale records from a different enrichment scope.
        case.metadata = {**case.metadata, "node_mapping": nodes,
                         "component_mapping": {c: _component_roles(case, c) for c in components},
                         "mapping_scope": {"component_ids": components, "local_edge_indices": selected}}
        case.mapping_quality = {
            "mode": self.mode, "join_strategy": self.join_strategy,
            "num_edges": len(keys), "selected_edges": len(selected),
            "matched": len(edge_uuids), "unmatched": len(selected) - len(edge_uuids),
            "match_rate": len(edge_uuids) / len(selected) if selected else 0.0,
            "events": event_counts, "nodes": node_counts,
            "node_source": node_source, "event_source": event_source,
            "network_fields_verified_by_caller": list(self.verified_network_fields),
            "orientation": "ORTHRUS database direction; no second reversal",
        }
        if self.fail_on_unmatched and len(edge_uuids) != len(selected):
            raise RuntimeError("Unresolved events while fail_on_unmatched=True")
        return case

    def _resolve_nodes(self, node_ids, rows, source):
        candidates = {}
        for row in rows:
            node_id = build_join_key(row.get("index_id"), None, None, None)[0]
            candidates.setdefault(node_id, []).append(row)
        result = {}
        for node_id in node_ids:
            hits = candidates.get(node_id, [])
            meta = {"index_id": node_id, "status": "missing", "candidate_count": len(hits),
                    "provenance": source}
            if len(hits) > 1:
                meta["status"] = "ambiguous"
            elif len(hits) == 1:
                row = hits[0]
                kind = row.get("node_type")
                if kind not in self.NODE_COLUMNS or not row.get("node_uuid"):
                    meta["status"] = "unresolvable"
                    meta["missing_fields"] = [k for k in ("node_type", "node_uuid") if not row.get(k)]
                else:
                    meta.update(status="resolved", node_type=kind, node_uuid=str(row["node_uuid"]))
                    fields = self.NODE_COLUMNS[kind][2:]
                    meta["missing_fields"] = [f for f in fields if row.get(f) in (None, "")]
                    if kind == "netflow":
                        meta["unverified_fields"] = [f for f in fields if f not in self.verified_network_fields]
                        fields = self.verified_network_fields
                    for field in fields:
                        if row.get(field) not in (None, ""):
                            meta[field] = row[field]
            result[node_id] = meta
        return result

    def _build_edge_keys(self, case):
        return extract_edge_join_keys(case, self.rel2id, self.edge_type_slice)

    def _match_edges_to_rows(self, edge_keys, rows):
        row_map = {}
        for row in rows:
            key = _event_row_key(row)
            if None not in key:
                row_map.setdefault(key, []).append(row)
        return {i: row_map[key] for i, key in enumerate(edge_keys)
                if None not in key and key in row_map}

    def _fetch_rows(self, node_ids, edge_keys, event_uuids):
        """Only SELECT, bounded parameters; no schema changes or credentials in errors.

        The local DDL has no suitable lookup indexes. Query plans and indexes must
        be checked on the server; a selective WHERE alone cannot ensure an index scan.
        """
        if not (node_ids or edge_keys or event_uuids):
            return [], []
        try:
            import psycopg2
        except ImportError:
            raise RuntimeError("Optional psycopg2 driver is required for PostgreSQL mapping") from None
        connection = None
        try:
            connection = psycopg2.connect(**self.db_config)
            connection.set_session(readonly=True, autocommit=False)
            nodes, events = [], []
            with connection.cursor() as cursor:
                cursor.execute("SELECT set_config('statement_timeout', %s, true)", ("30000",))
                for kind, columns in self.NODE_COLUMNS.items():
                    for chunk in _chunks(node_ids):
                        cursor.execute(f"SELECT {', '.join(columns)} FROM {kind}_node_table WHERE index_id = ANY(%s)",
                                       (chunk,))
                        nodes.extend({**dict(zip(columns, row)), "node_type": kind} for row in cursor.fetchall())
                events = self._fetch_event_rows(cursor, edge_keys, event_uuids)
            return nodes, events
        except Exception:
            # Driver exceptions can contain DSNs, host names and credentials.
            raise RuntimeError("PostgreSQL mapping failed; verify connectivity, schema and query plan on the server") from None
        finally:
            if connection is not None:
                try:
                    connection.close()  # Roll back the read-only transaction.
                except Exception:
                    pass

    def _fetch_event_rows(self, cursor, edge_keys, event_uuids):
        rows = []
        columns = ', '.join(self.EVENT_COLUMNS)
        for chunk in _chunks(edge_keys):
            conditions = ' OR '.join('(src_index_id = %s AND dst_index_id = %s AND timestamp_rec = %s AND operation = %s)'
                                     for _ in chunk)
            params = tuple(value for s, d, t, op in chunk for value in (str(s), str(d), t, op))
            cursor.execute(f"SELECT {columns} FROM event_table WHERE {conditions}", params)
            rows.extend(dict(zip(self.EVENT_COLUMNS, row)) for row in cursor.fetchall())
        for chunk in _chunks(event_uuids):
            cursor.execute(f"SELECT {columns} FROM event_table WHERE event_uuid = ANY(%s)", (chunk,))
            rows.extend(dict(zip(self.EVENT_COLUMNS, row)) for row in cursor.fetchall())
        # A row can be fetched both by a key and by UUID; retain genuine DB duplicates.
        by_id = {}
        for row in rows:
            by_id[row["_id"]] = row
        return list(by_id.values())


def _chunks(values, size=200):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _event_row_key(row):
    operation = row.get("operation")
    if not isinstance(operation, str) or not operation.startswith("EVENT_"):
        operation = None
    return build_join_key(row.get("src_index_id"), row.get("dst_index_id"),
                          row.get("timestamp_rec"), operation)


def _component_roles(case, component_id):
    """Describe the existing neutralization policy; never invoke perturbation/model."""
    edges = list(case.component_to_edges[component_id])
    src = [_maybe_int(_safe_scalar(v)) for v in case.temporal_data.src]
    dst = [_maybe_int(_safe_scalar(v)) for v in case.temporal_data.dst]
    sources = sorted({src[i] for i in edges})
    destinations = sorted({dst[i] for i in edges})
    return {"assigned_edges": edges, "source_node_ids": sources, "destination_node_ids": destinations,
            "source_rows_if_disabled": [i for i, n in enumerate(src) if n in sources],
            "destination_rows_if_disabled": [i for i, n in enumerate(dst) if n in destinations],
            "neutralization_policy": "all_occurrences_per_node_role"}


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
    """Load a JSON sidecar; optional artifact verification binds graph rows to a batch.

    Sidecar format (JSON) example:
    {
      "mapping_mode": "sidecar-assisted",
      "join_strategy": "edge_index",
      "edges": {
         "0": {"event_uuid": "...", "src_index_id": 1, ...}
      }
    }
    """

    def __init__(self, sidecar_path: str, *, temporal_data_path=None,
                 graph_edge_offset=None, rel2id=None, edge_type_slice=None):
        super().__init__(mode="sidecar-assisted")
        self.sidecar_path = str(sidecar_path)
        self.temporal_data_path = temporal_data_path
        self.graph_edge_offset = graph_edge_offset
        self.rel2id = rel2id
        self.edge_type_slice = edge_type_slice

    def enrich_case(self, case: OrthrusAlertCase) -> OrthrusAlertCase:
        case.mapping_mode = "sidecar-assisted"

        p = Path(self.sidecar_path)
        if not p.exists():
            raise FileNotFoundError(f"Sidecar not found: {self.sidecar_path}")

        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if self.temporal_data_path is not None:
            return self._enrich_verified(case, data)
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
                edge_to_meta[idx] = {**v, "status": "unverified",
                                     "provenance": {"source": "legacy_sidecar", "identity_verified": False}}

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
            "identity_verified": False,
            "warnings": ["Legacy sidecar: artifact/batch identity not verified"] +
                        ([] if unmatched == 0 else [f"{unmatched} unmatched edges"]),
            "extra_sidecar_edges": sorted([i for i in sidecar_edge_indices if i >= num_edges]),
        }

        return case

    def _enrich_verified(self, case, data):
        from preprocessing.orthrus_sidecar_generator import _load_temporal_data_like
        from types import SimpleNamespace

        if not isinstance(self.graph_edge_offset, int) or self.graph_edge_offset < 0:
            raise ValueError("Verified sidecars require an explicit nonnegative graph_edge_offset")
        reference = _load_temporal_data_like(str(self.temporal_data_path))
        reference_keys = extract_edge_join_keys(SimpleNamespace(temporal_data=reference), self.rel2id, self.edge_type_slice)
        keys = extract_edge_join_keys(case, self.rel2id, self.edge_type_slice)
        offset = self.graph_edge_offset
        provenance = data.get("provenance", {})
        valid_artifact = (provenance.get("artifact_sha256") == artifact_sha256(self.temporal_data_path)
                          and provenance.get("method") in ("graph_order_verified", "unique_complete_join")
                          and provenance.get("edge_index_scope") == "graph"
                          and data.get("num_edges") == len(reference_keys)
                          and offset + len(keys) <= len(reference_keys)
                          and reference_keys[offset:offset + len(keys)] == keys)
        fingerprint = case_mapping_fingerprint(case, keys)
        counts = Counter(reference_keys)
        uuids, metadata = {}, {}
        for i, key in enumerate(keys):
            row = data.get("edges", {}).get(str(offset + i), {})
            verified = (valid_artifact and None not in key and counts[key] == 1
                        and _event_row_key(row) == key and bool(row.get("event_uuid"))
                        and row.get("status", "resolved") == "resolved")
            status = "resolved" if verified else "unverified"
            if valid_artifact and counts[key] > 1:
                status = "ambiguous"
            meta = {"edge_index": i, "graph_edge_index": offset + i,
                    "src_index_id": key[0], "dst_index_id": key[1],
                    "timestamp_rec": key[2], "operation": key[3], "status": status,
                    "provenance": {"source": "verified_sidecar", "identity_verified": verified,
                                   "artifact_sha256": provenance.get("artifact_sha256"),
                                   "case_fingerprint": fingerprint, "verified_key": list(key)}}
            if verified:
                uuids[i] = str(row["event_uuid"])
                meta["event_uuid"] = uuids[i]
            metadata[i] = meta
        case.edge_to_event_uuid = uuids
        case.edge_to_original_metadata = metadata
        case.edge_to_log_record = {}
        case.mapping_quality = {"mode": self.mode, "num_edges": len(keys), "matched": len(uuids),
                                "unmatched": len(keys) - len(uuids),
                                "artifact_and_batch_verified": bool(valid_artifact),
                                "graph_edge_offset": offset,
                                "events": dict(Counter(m["status"] for m in metadata.values()))}
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
    """Render only already available metadata: never fetch or evaluate scores."""
    return ArtifactOnlyMappingProvider().describe_component(case, component_id, max_edges)


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
