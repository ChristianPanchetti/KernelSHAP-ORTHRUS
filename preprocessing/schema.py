from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class LogRecord:
    """A normalized representation of a single process/log record.

    The goal is NOT to model the full ORTHRUS schema, but to provide a stable and
    minimal internal format for the XAI pipeline.

    Notes (IT/EN):
    - We keep the original JSON record in `raw` for forward compatibility.
    - We extract only a few fields that are useful to build interpretable
      components (e.g., execution path) and to run the black-box adapter.
    """

    entity_id: Optional[int]
    parent_entity_id: Optional[int]
    pid: Optional[int]
    ppid: Optional[int]
    tid: Optional[int]
    start_time: Optional[str]

    execution_path: Optional[str]
    cmdline_args: Optional[List[str]]

    anon_uid: Optional[str]
    raw: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LogRecord":
        execution_binary = d.get("executionBinary") or {}
        execution_path = execution_binary.get("path")
        if isinstance(execution_path, str):
            execution_path = execution_path.strip() or None
        else:
            execution_path = None

        cmdline_args = d.get("cmdlineArgs")
        if cmdline_args is not None and not isinstance(cmdline_args, list):
            # Defensive parsing: tolerate schema drift.
            cmdline_args = [str(cmdline_args)]

        return LogRecord(
            entity_id=_to_int_or_none(d.get("entityId")),
            parent_entity_id=_to_int_or_none(d.get("parentEntityId")),
            pid=_to_int_or_none(d.get("pid")),
            ppid=_to_int_or_none(d.get("ppid")),
            tid=_to_int_or_none(d.get("tid")),
            start_time=_to_str_or_none(d.get("startTime")),
            execution_path=execution_path,
            cmdline_args=[str(x) for x in cmdline_args] if cmdline_args else None,
            anon_uid=_to_str_or_none(d.get("_uid")),
            raw=dict(d),
        )


@dataclass(frozen=True)
class LogDataset:
    """A collection of `LogRecord` loaded from a JSON input."""

    records: List[LogRecord]
    source_path: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def _to_int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_str_or_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None
