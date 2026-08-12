from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Sequence

from preprocessing.schema import LogDataset, LogRecord
from perturbation.interpretable_graph_builder import InterpretableSpace


@dataclass(frozen=True)
class PerturbationResult:
    """Container for a perturbed input."""

    mask: List[int]
    dataset: LogDataset


class PerturbationManager:
    """Generate a perturbed input from a binary mask z.

    Mask semantics:
    - z[i] == 1 => component i is kept
    - z[i] == 0 => component i is removed/masked

     Supported perturbation modes:

     1) `drop_records` (default)
         Remove all records mapped to components set to 0.

     2) `neutralize_command`
         Keep all records, but neutralize command-like textual fields for components
         set to 0. With the current schema this means:
         - `LogRecord.execution_path` -> None
         - `LogRecord.cmdline_args` -> None
         The corresponding keys in `LogRecord.raw` are also sanitized when present
         (e.g., `executionBinary.path`, `cmdlineArgs`, `envs`).

     3) `mask_features`
         Reserved for feature-level masking (not available with the current schema).
         Raises `NotImplementedError`.

     4) `drop_time_window`
         Like `drop_records`, but only allowed when the interpretable space was
         built with grouping_mode='time_window'. Raises a clear error otherwise.

     5) `drop_subgraph` / `remove_node_with_incident_edges`
         Reserved for future graph-based representations. Raises `NotImplementedError`.

    IT/EN:
    - This is intentionally simple and robust.
    - When integrating the real ORTHRUS-ano, you may need a different masking
      behavior (e.g., neutralize features instead of dropping records).
    """

    def __init__(
        self,
        original: LogDataset,
        space: InterpretableSpace,
        mode: str = "drop_records",
    ):
        supported = {
            "drop_records",
            "neutralize_command",
            "mask_features",
            "drop_time_window",
            "drop_subgraph",
            "remove_node_with_incident_edges",
        }
        if mode not in supported:
            raise ValueError(
                f"Unsupported perturbation mode: {mode}. Supported: {', '.join(sorted(supported))}"
            )

        self.original = original
        self.space = space
        self.mode = mode

    def apply_mask(self, mask: Sequence[int]) -> PerturbationResult:
        if len(mask) != self.space.num_components:
            raise ValueError(
                f"Mask length mismatch: expected {self.space.num_components}, got {len(mask)}"
            )

        keep = [1 if int(v) != 0 else 0 for v in mask]

        if self.mode == "drop_records":
            dataset = self._drop_records(keep)
        elif self.mode == "neutralize_command":
            dataset = self._neutralize_command(keep)
        elif self.mode == "mask_features":
            raise NotImplementedError(
                "perturbation mode 'mask_features' is not supported yet: the current LogRecord schema "
                "does not expose a feature vector suitable for partial masking."
            )
        elif self.mode == "drop_time_window":
            dataset = self._drop_time_window(keep)
        elif self.mode in {"drop_subgraph", "remove_node_with_incident_edges"}:
            raise NotImplementedError(
                f"perturbation mode '{self.mode}' requires an explicit graph-based representation in LogDataset "
                "(nodes + edges). The current schema only provides flat records, so this mode is intentionally stubbed."
            )
        else:
            raise RuntimeError("Unreachable: unsupported mode")

        return PerturbationResult(mask=keep, dataset=dataset)

    def _drop_records(self, keep: Sequence[int]) -> LogDataset:
        # Keep the legacy behavior exactly: remove records when their component is 0.
        records: List[LogRecord] = []
        for rec_idx, rec in enumerate(self.original.records):
            comp_idx = self.space.record_to_component[rec_idx]
            if keep[comp_idx] == 1:
                records.append(rec)

        return LogDataset(
            records=records,
            source_path=self.original.source_path,
            meta={
                **(self.original.meta or {}),
                "perturbation": {"mode": self.mode, "kept_components": int(sum(keep))},
            },
        )

    def _drop_time_window(self, keep: Sequence[int]) -> LogDataset:
        grouping_mode = (self.space.build_info or {}).get("grouping_mode")
        if grouping_mode != "time_window":
            raise ValueError(
                "perturbation mode 'drop_time_window' requires an interpretable space built with "
                "grouping_mode='time_window'."
            )
        # Semantically equivalent to drop_records, but guarded to avoid silent misuse.
        return self._drop_records(keep)

    @staticmethod
    def _neutralize_record_command_fields(rec: LogRecord) -> LogRecord:
        """Return a safe copy of `rec` with command-like fields neutralized.

        With the current normalized schema we can reliably neutralize:
        - `execution_path`
        - `cmdline_args`

        We also sanitize corresponding keys in `raw` when present, to avoid creating
        a dataset that is silently inconsistent between normalized and raw fields.
        """

        new_raw = dict(rec.raw or {})

        eb = new_raw.get("executionBinary")
        if isinstance(eb, dict):
            eb2 = dict(eb)
            # Use empty string because the loader normalizes empty/whitespace to None.
            eb2["path"] = ""
            new_raw["executionBinary"] = eb2

        if "cmdlineArgs" in new_raw:
            new_raw["cmdlineArgs"] = None

        # Optional (raw-only) fields; do not assume they exist.
        if "envs" in new_raw:
            new_raw["envs"] = None

        return replace(
            rec,
            execution_path=None,
            cmdline_args=None,
            raw=new_raw,
        )

    def _neutralize_command(self, keep: Sequence[int]) -> LogDataset:
        records: List[LogRecord] = []
        neutralized_components = 0

        # Deterministic: preserve original record order.
        for rec_idx, rec in enumerate(self.original.records):
            comp_idx = self.space.record_to_component[rec_idx]
            if keep[comp_idx] == 1:
                records.append(rec)
            else:
                records.append(self._neutralize_record_command_fields(rec))

        neutralized_components = int(len(keep) - sum(int(x) for x in keep))

        return LogDataset(
            records=records,
            source_path=self.original.source_path,
            meta={
                **(self.original.meta or {}),
                "perturbation": {
                    "mode": self.mode,
                    "kept_components": int(sum(keep)),
                    "neutralized_components": neutralized_components,
                },
            },
        )
