from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional

from xai.kernel_shap_explainer import KernelSHAPResult


class ResultExporter:
    """Save explanation results to disk."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger(__name__)

    def export_json(self, result: KernelSHAPResult, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = result.to_dict()
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self._logger.info(f"Saved results JSON to: {output_path}")

    def export_ranking_csv(self, result: KernelSHAPResult, output_path: Path) -> None:
        """Export the ranking as a simple CSV."""

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "index", "name", "shap_value", "abs_shap"])
            for r, idx in enumerate(result.ranking, start=1):
                writer.writerow(
                    [
                        r,
                        idx,
                        result.components[idx].name,
                        float(result.shap_values[idx]),
                        float(abs(result.shap_values[idx])),
                    ]
                )

        self._logger.info(f"Saved ranking CSV to: {output_path}")
