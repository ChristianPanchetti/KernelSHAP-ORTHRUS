from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from preprocessing.schema import LogDataset


@dataclass(frozen=True)
class InterpretableComponent:
    """A single interpretable component for Kernel SHAP.

    In this first (simple) representation we use binary components: a component is
    either present (1) or absent (0) in a perturbation mask.

    IT/EN:
    - A component maps to one or more log records.
    - Perturbation is performed by masking/removing the records belonging to components set to 0.
    """

    component_id: str
    name: str
    kind: str
    record_indices: Tuple[int, ...]
    description: str


@dataclass(frozen=True)
class InterpretableSpace:
    """The interpretable space used by Kernel SHAP."""

    components: List[InterpretableComponent]
    record_to_component: Tuple[int, ...]
    build_info: Dict[str, str] = field(default_factory=dict)

    @property
    def num_components(self) -> int:
        return len(self.components)

    def component_names(self) -> List[str]:
        return [c.name for c in self.components]


class InterpretableGraphBuilder:
    """Build an interpretable space (binary components) from a `LogDataset`.

        The goal is to map each log record to exactly one *interpretable component*.
        Perturbation will then mask/remove all records of the components set to 0.

        Supported `grouping_mode` values:
        - `exec_path`: group by executable path (`LogRecord.execution_path`), with a per-record
            fallback when missing.
        - `record`: one component per record.
        - `entity_id`: one component per `LogRecord.entity_id` (per-record fallback when missing).
        - `parent_child`: one component per edge `parent_entity_id -> entity_id` (per-record fallback
            when missing).
        - `time_window`: one component per fixed time window computed from `LogRecord.start_time`.
            Records with missing/unparseable timestamps fall back to per-record components.
        - `event_type`: one component per event type, extracted from `LogRecord.raw` if present
            (keys: `eventType` or `event_type`). Records missing the field fall back to per-record.
        - `local_subgraph`: reserved for future work. Currently raises `NotImplementedError`.

        Notes (IT/EN):
        - We keep the legacy behavior of `exec_path` and `record` unchanged.
        - Fallback to per-record components is used to avoid merging unrelated events.
    """

    def __init__(
        self,
        grouping_mode: str = "exec_path",
        max_components: Optional[int] = 50,
        time_window_seconds: int = 60,
    ):
        supported = {
            "exec_path",
            "record",
            "entity_id",
            "parent_child",
            "time_window",
            "event_type",
            "local_subgraph",
        }
        if grouping_mode not in supported:
            raise ValueError(f"Unsupported grouping_mode: {grouping_mode}")
        if max_components is not None and max_components <= 0:
            raise ValueError("max_components must be > 0 or None")
        if int(time_window_seconds) <= 0:
            raise ValueError("time_window_seconds must be > 0")

        self.grouping_mode = grouping_mode
        self.max_components = max_components
        self.time_window_seconds = int(time_window_seconds)

    def build(self, dataset: LogDataset) -> InterpretableSpace:
        if not dataset.records:
            raise ValueError("Empty dataset: cannot build interpretable space")

        build_info = {
            "grouping_mode": self.grouping_mode,
            "max_components": str(self.max_components) if self.max_components is not None else "None",
        }

        if self.grouping_mode == "record":
            groups: Dict[str, List[int]] = {f"record:{i}": [i] for i in range(len(dataset.records))}
            kind = "record"
        elif self.grouping_mode == "exec_path":
            groups, kind = self._group_by_exec_path(dataset)
        elif self.grouping_mode == "entity_id":
            groups, kind = self._group_by_entity_id(dataset)
        elif self.grouping_mode == "parent_child":
            groups, kind = self._group_by_parent_child(dataset)
        elif self.grouping_mode == "time_window":
            build_info["time_window_seconds"] = str(self.time_window_seconds)
            groups, kind = self._group_by_time_window(dataset)
        elif self.grouping_mode == "event_type":
            groups, kind = self._group_by_event_type(dataset)
        elif self.grouping_mode == "local_subgraph":
            # Stub requested by spec: do not implement complex graph logic without explicit edges.
            raise NotImplementedError(
                "grouping_mode='local_subgraph' is reserved for future work. "
                "No explicit edges are available in the current dataset/schema."
            )
        else:
            # Defensive: should be impossible due to validation in __init__.
            raise RuntimeError(f"Unsupported grouping_mode: {self.grouping_mode}")

        components = self._groups_to_components(groups, kind)
        components = self._apply_max_components(components, len(dataset.records))

        record_to_component = [-1] * len(dataset.records)
        for comp_idx, comp in enumerate(components):
            for rec_idx in comp.record_indices:
                record_to_component[rec_idx] = comp_idx

        # Sanity: every record must be assigned to exactly one component
        if any(x < 0 for x in record_to_component):
            missing = sum(1 for x in record_to_component if x < 0)
            raise RuntimeError(f"Internal error: {missing} record(s) not mapped to any component")

        return InterpretableSpace(
            components=components,
            record_to_component=tuple(record_to_component),
            build_info=build_info,
        )

    @staticmethod
    def _group_by_exec_path(dataset: LogDataset) -> Tuple[Dict[str, List[int]], str]:
        groups: Dict[str, List[int]] = {}
        for idx, rec in enumerate(dataset.records):
            if rec.execution_path:
                key = f"exec:{rec.execution_path}"
            else:
                # IT/EN: fallback to a unique key for this record.
                key = f"record:{idx}"
            groups.setdefault(key, []).append(idx)
        return groups, "execution_path"

    @staticmethod
    def _group_by_entity_id(dataset: LogDataset) -> Tuple[Dict[str, List[int]], str]:
        groups: Dict[str, List[int]] = {}
        for idx, rec in enumerate(dataset.records):
            if rec.entity_id is not None:
                key = f"entity:{rec.entity_id}"
            else:
                # Missing entity_id -> fall back to per-record grouping.
                key = f"record:{idx}"
            groups.setdefault(key, []).append(idx)
        return groups, "entity_id"

    @staticmethod
    def _group_by_parent_child(dataset: LogDataset) -> Tuple[Dict[str, List[int]], str]:
        groups: Dict[str, List[int]] = {}
        for idx, rec in enumerate(dataset.records):
            if rec.parent_entity_id is not None and rec.entity_id is not None:
                key = f"parent_child:{rec.parent_entity_id}->{rec.entity_id}"
            else:
                # Missing either parent or child -> fall back to per-record grouping.
                key = f"record:{idx}"
            groups.setdefault(key, []).append(idx)
        return groups, "parent_child"

    @staticmethod
    def _get_event_type_from_raw(raw: dict) -> Optional[str]:
        # Keep it intentionally minimal and schema-driven: do not guess many keys.
        for k in ("eventType", "event_type"):
            if k in raw:
                v = raw.get(k)
                if v is None:
                    return None
                s = str(v).strip()
                return s or None
        return None

    def _group_by_event_type(self, dataset: LogDataset) -> Tuple[Dict[str, List[int]], str]:
        groups: Dict[str, List[int]] = {}
        found_any = False
        for idx, rec in enumerate(dataset.records):
            ev = self._get_event_type_from_raw(rec.raw)
            if ev:
                found_any = True
                key = f"event_type:{ev}"
            else:
                # If the field is missing for some records, avoid merging unknown events.
                key = f"record:{idx}"
            groups.setdefault(key, []).append(idx)

        if not found_any:
            raise ValueError(
                "grouping_mode='event_type' requires an event type field, but none was found. "
                "Expected LogRecord.raw to contain 'eventType' or 'event_type'."
            )

        return groups, "event_type"

    @staticmethod
    def _parse_start_time(value: str) -> datetime:
        """Parse `LogRecord.start_time` into a timezone-aware datetime.

        Expected formats include ISO-8601 strings like:
        - 2026-04-08T10:00:00.010000000Z
        - 2026-04-08T10:00:00Z

        SHAP tests/pipeline run on Python's stdlib only: we do not depend on dateutil.
        The ORTHRUS-like samples may contain nanosecond precision; we truncate to microseconds.
        """

        s = str(value).strip()
        if not s:
            raise ValueError("Empty start_time")

        if s.endswith("Z"):
            s = s[:-1] + "+00:00"

        # Split timezone (if any) from the time part.
        t_idx = s.find("T")
        tz_idx = -1
        if t_idx != -1:
            plus_idx = s.find("+", t_idx)
            minus_idx = s.find("-", t_idx + 1)
            candidates = [i for i in (plus_idx, minus_idx) if i != -1]
            if candidates:
                tz_idx = min(candidates)

        if tz_idx != -1:
            dt_part, tz_part = s[:tz_idx], s[tz_idx:]
        else:
            dt_part, tz_part = s, ""

        if "." in dt_part:
            base, frac = dt_part.split(".", 1)
            frac_digits = "".join(ch for ch in frac if ch.isdigit())
            if frac_digits:
                frac_digits = (frac_digits + "000000")[:6]
                dt_part = f"{base}.{frac_digits}"
            else:
                dt_part = base

        normalized = dt_part + tz_part
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _group_by_time_window(self, dataset: LogDataset) -> Tuple[Dict[str, List[int]], str]:
        groups: Dict[str, List[int]] = {}
        parsed_any = False

        for idx, rec in enumerate(dataset.records):
            dt: Optional[datetime] = None
            if rec.start_time:
                try:
                    dt = self._parse_start_time(rec.start_time)
                except ValueError:
                    dt = None

            if dt is None:
                # IT/EN: if timestamp is missing/unparseable, do a safe fallback.
                key = f"record:{idx}"
            else:
                parsed_any = True
                window_start_ts = int(dt.timestamp() // self.time_window_seconds) * self.time_window_seconds
                window_start_dt = datetime.fromtimestamp(window_start_ts, tz=timezone.utc)
                window_start_str = window_start_dt.isoformat().replace("+00:00", "Z")
                key = f"time_window:{window_start_str}"

            groups.setdefault(key, []).append(idx)

        if not parsed_any:
            raise ValueError(
                "grouping_mode='time_window' requires valid timestamps, but LogRecord.start_time is missing "
                "or not parseable for all records."
            )

        return groups, "time_window"

    @staticmethod
    def _groups_to_components(groups: Dict[str, List[int]], kind: str) -> List[InterpretableComponent]:
        comps: List[InterpretableComponent] = []
        # Sort by size (descending) for stable ranking and max_components truncation.
        for key, indices in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            name = key
            description = f"Binary component for {key} ({len(indices)} record(s))"
            comps.append(
                InterpretableComponent(
                    component_id=key,
                    name=name,
                    kind=kind,
                    record_indices=tuple(indices),
                    description=description,
                )
            )
        return comps

    def _apply_max_components(
        self,
        components: List[InterpretableComponent],
        num_records: int,
    ) -> List[InterpretableComponent]:
        if self.max_components is None or len(components) <= self.max_components:
            return components

        # Keep the most frequent components and merge the remainder into a single "OTHER".
        # IT/EN: this keeps the pipeline executable on large inputs without losing records.
        keep = max(1, self.max_components - 1)
        kept = components[:keep]
        rest = components[keep:]

        other_indices: List[int] = []
        for c in rest:
            other_indices.extend(list(c.record_indices))

        other = InterpretableComponent(
            component_id="OTHER",
            name="OTHER",
            kind="merged",
            record_indices=tuple(sorted(other_indices)),
            description=f"Merged component for the remaining {len(rest)} components ({len(other_indices)} record(s))",
        )
        return kept + [other]
