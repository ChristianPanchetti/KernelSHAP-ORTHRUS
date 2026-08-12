from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


def build_join_key(src: Any, dst: Any, t: Any, operation: Any) -> Tuple:
    return (int(src) if src is not None else None,
            int(dst) if dst is not None else None,
            int(t) if t is not None else None,
            str(operation) if operation is not None else None)


def decode_operation_from_temporal_data(temporal_data: Any, rel2id: Optional[Dict[str, int]] = None,
                                        edge_type_slice: Optional[Tuple[int, int]] = None) -> List[Any]:
    """Return a list of operation labels, one per edge.

    Strategy:
    - If `temporal_data` has `edge_type` attribute, use it (best-effort via _edge_type_label-like logic).
    - Otherwise, if `msg` is present and rel2id provided, attempt to decode operation via argmax
      on the edge-type slice inside `msg` (this requires knowing the slice boundaries).
    - If neither is possible, raise NotImplementedError with guidance.
    """
    ops = []
    if temporal_data is None:
        raise NotImplementedError("temporal_data is None; cannot decode operation")

    # Prefer explicit edge_type
    if hasattr(temporal_data, "edge_type"):
        et = getattr(temporal_data, "edge_type")
        for v in et:
            # simple scalar or one-hot list
            if isinstance(v, (int, str)):
                ops.append(v)
                continue
            try:
                # try to get argmax from list/tuple
                if isinstance(v, (list, tuple)) and v:
                    max_i = 0
                    max_val = float(v[0])
                    for i in range(1, len(v)):
                        fv = float(v[i])
                        if fv > max_val:
                            max_val = fv
                            max_i = i
                    ops.append(max_i)
                    continue
            except Exception:
                pass
            ops.append(str(v))
        return ops

    # Fallback: decode from msg using rel2id and slice
    if hasattr(temporal_data, "msg") and rel2id is not None and edge_type_slice is not None:
        msgs = getattr(temporal_data, "msg")
        s, e = edge_type_slice
        for m in msgs:
            seg = m[s:e]
            # assume seg is iterable of scores
            try:
                max_i = 0
                max_val = float(seg[0])
                for i in range(1, len(seg)):
                    fv = float(seg[i])
                    if fv > max_val:
                        max_val = fv
                        max_i = i
                # map index -> operation name if rel2id provided as name->id
                inv = {v: k for k, v in rel2id.items()} if rel2id else {}
                ops.append(inv.get(max_i, int(max_i)))
                continue
            except Exception:
                ops.append(str(seg))
        return ops

    raise NotImplementedError(
        "Cannot decode operation: temporal_data lacks `edge_type` and msg decode requires rel2id and edge_type_slice"
    )


def extract_edge_join_keys(case: Any, rel2id: Optional[Dict[str, int]] = None,
                           edge_type_slice: Optional[Tuple[int, int]] = None) -> List[Tuple]:
    td = case.temporal_data
    if td is None:
        return []

    n = 0
    if hasattr(td, "src"):
        try:
            n = len(getattr(td, "src"))
        except Exception:
            pass

    if n == 0 and hasattr(td, "edge_index"):
        ei = getattr(td, "edge_index")
        try:
            n = int(ei.shape[1])
        except Exception:
            try:
                n = len(ei[0])
            except Exception:
                n = 0

    if n == 0:
        return []

    ops = decode_operation_from_temporal_data(td, rel2id=rel2id, edge_type_slice=edge_type_slice)

    keys = []
    for i in range(n):
        s = getattr(td, "src")[i] if hasattr(td, "src") else None
        d = getattr(td, "dst")[i] if hasattr(td, "dst") else None
        t = getattr(td, "t")[i] if hasattr(td, "t") else None
        op = ops[i] if i < len(ops) else None
        keys.append(build_join_key(s, d, t, op))

    return keys
