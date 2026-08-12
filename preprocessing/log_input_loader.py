from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from preprocessing.schema import LogDataset, LogRecord


class LogInputLoader:
    """Load `logs_input.json` and convert it into an internal representation."""

    def __init__(self, logger: logging.Logger | None = None):
        self._logger = logger or logging.getLogger(__name__)

    def load(self, path: Union[str, Path]) -> LogDataset:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        if p.suffix.lower() != ".json":
            raise ValueError(f"Expected a .json input, got: {p.name}")

        self._logger.info(f"Loading input logs from: {p}")
        raw = self._read_json(p)

        if isinstance(raw, dict):
            # Some datasets may wrap the list into a top-level object.
            # IT/EN: support both shapes without assuming a specific ORTHRUS schema.
            if "records" in raw and isinstance(raw["records"], list):
                raw_records = raw["records"]
                meta: Dict[str, Any] = {k: v for k, v in raw.items() if k != "records"}
            else:
                raise ValueError(
                    "Unsupported JSON object shape. Expected a list, or an object with a 'records' list."
                )
        elif isinstance(raw, list):
            raw_records = raw
            meta = {}
        else:
            raise ValueError("Unsupported JSON root. Expected list or object.")

        records: List[LogRecord] = []
        bad = 0
        for idx, item in enumerate(raw_records):
            if not isinstance(item, dict):
                bad += 1
                self._logger.warning(f"Skipping non-object record at index {idx}: {type(item)}")
                continue
            try:
                records.append(LogRecord.from_dict(item))
            except Exception as e:  # noqa: BLE001 - want robust parsing
                bad += 1
                self._logger.warning(f"Skipping malformed record at index {idx}: {e}")

        self._logger.info(f"Loaded {len(records)} record(s) (skipped {bad}).")
        return LogDataset(records=records, source_path=str(p), meta=meta)

    @staticmethod
    def _read_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
