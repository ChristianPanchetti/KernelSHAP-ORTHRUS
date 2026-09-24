from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple


def _plain(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _integer(value):
    value = _plain(value)
    if value is None or isinstance(value, (bool, list, tuple)):
        return None
    try:
        result = int(value)
        if isinstance(value, float) and value != result:
            return None
        return result
    except (TypeError, ValueError, OverflowError):
        return None


def build_join_key(src: Any, dst: Any, t: Any, operation: Any) -> Tuple:
    return (_integer(src), _integer(dst), _integer(t),
            str(operation) if operation is not None else None)


def decode_operation_from_temporal_data(temporal_data: Any, rel2id: Optional[Dict] = None,
                                        edge_type_slice: Optional[Tuple[int, int]] = None) -> List:
    """Decode official ORTHRUS one-hot vectors using supplied, 1-based rel2id.

    Scalar numbers are explicit relation IDs (not zero-based argmax indices).
    Unknown/malformed values return None. No vocabulary is guessed or imported.
    """
    if temporal_data is None:
        return []
    vocabulary = {}
    conflicts = set()
    for key, value in (rel2id or {}).items():
        if isinstance(key, str) and key.startswith("EVENT_"):
            number, label = _integer(value), key
        elif isinstance(value, str) and value.startswith("EVENT_"):
            number, label = _integer(key), value
        else:
            continue
        if number is None or number < 1:
            continue
        if number in vocabulary and vocabulary[number] != label:
            conflicts.add(number)
        vocabulary[number] = label
    for number in conflicts:
        vocabulary.pop(number, None)

    values = getattr(temporal_data, "edge_type", None)
    if values is None:
        values = getattr(temporal_data, "operation", None)
    if values is None:
        msg = getattr(temporal_data, "msg", None)
        if msg is None or edge_type_slice is None:
            return []
        start, stop = edge_type_slice
        if start < 0 or stop <= start:
            return []
        values = [row[start:stop] for row in msg]

    operations = []
    for value in values:
        value = _plain(value)
        if isinstance(value, str):
            operations.append(value if value.startswith("EVENT_") else None)
        elif isinstance(value, (list, tuple)):
            # Actual encoder uses one_hot(arange(N))[rel2id[label] - 1].
            valid = (bool(value) and all(v in (0, 1) for v in value)
                     and sum(value) == 1
                     and set(vocabulary) == set(range(1, len(value) + 1)))
            operations.append(vocabulary.get(value.index(1) + 1) if valid else None)
        else:
            operations.append(vocabulary.get(_integer(value)))
    return operations


def extract_edge_join_keys(case: Any, rel2id: Optional[Dict] = None,
                           edge_type_slice: Optional[Tuple[int, int]] = None) -> List[Tuple]:
    td = case.temporal_data
    if td is None:
        return []
    src, dst = getattr(td, "src", None), getattr(td, "dst", None)
    if src is None or dst is None:
        edge_index = getattr(td, "edge_index", None)
        if edge_index is None:
            return []
        src, dst = edge_index[0], edge_index[1]
    if len(src) != len(dst):
        raise ValueError("Mapping requires aligned src/dst arrays")
    timestamps = getattr(td, "t", None)
    if timestamps is not None and len(timestamps) != len(src):
        raise ValueError("Mapping requires aligned timestamps")
    ops = decode_operation_from_temporal_data(td, rel2id, edge_type_slice)
    if ops and len(ops) != len(src):
        raise ValueError("Mapping requires aligned operations")
    return [build_join_key(src[i], dst[i], timestamps[i] if timestamps is not None else None,
                           ops[i] if ops else None) for i in range(len(src))]


def case_mapping_fingerprint(case, keys):
    """Bind verified identities to ordered batch keys and available case coordinates."""
    coordinates = {k: case.metadata.get(k) for k in
                   ("dataset", "split", "graph_index", "batch_index", "global_edge_offset")}
    payload = json.dumps([keys, coordinates], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def artifact_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
