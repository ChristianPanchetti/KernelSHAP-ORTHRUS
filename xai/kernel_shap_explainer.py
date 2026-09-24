from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from adapters.orthrus_ano_adapter import OrthrusAnoAdapter
from perturbation.interpretable_graph_builder import InterpretableComponent, InterpretableSpace
from perturbation.perturbation_manager import PerturbationManager


@dataclass(frozen=True)
class KernelSHAPResult:
    """Result of a Kernel SHAP explanation run."""

    baseline_score: float
    original_score: float

    components: List[InterpretableComponent]
    shap_values: List[float]

    ranking: List[int]
    explanation_it: str
    explanation_en: str

    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        comps = []
        for i, c in enumerate(self.components):
            comps.append(
                {
                    "index": i,
                    "component_id": c.component_id,
                    "name": c.name,
                    "kind": c.kind,
                    "description": c.description,
                    "num_records": len(c.record_indices),
                    "shap_value": float(self.shap_values[i]),
                    "sign": "positive" if self.shap_values[i] > 0 else "negative" if self.shap_values[i] < 0 else "zero",
                }
            )

        ranking = []
        for rank, idx in enumerate(self.ranking, start=1):
            ranking.append(
                {
                    "rank": rank,
                    "index": idx,
                    "name": self.components[idx].name,
                    "shap_value": float(self.shap_values[idx]),
                    "abs_shap": float(abs(self.shap_values[idx])),
                }
            )

        return {
            "original_anomaly_score": float(self.original_score),
            "baseline_score": float(self.baseline_score),
            "components": comps,
            "ranking": ranking,
            "explanation_it": self.explanation_it,
            "explanation_en": self.explanation_en,
            "metadata": self.metadata,
        }


class KernelSHAPExplainer:
    """Kernel SHAP explainer for a black-box anomaly detector.

    We explain the mapping:

        z (binary mask) -> perturbed_input(z) -> anomaly_score

    This implementation uses only SHAP's official `shap.KernelExplainer`.
    """

    def __init__(
        self,
        num_samples: int = 200,
        seed: int = 0,
        top_k: int = 10,
        logger: Optional[logging.Logger] = None,
    ):
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.top_k = int(top_k)
        self._logger = logger or logging.getLogger(__name__)

    def explain(
        self,
        space: InterpretableSpace,
        perturbation: PerturbationManager,
        adapter: OrthrusAnoAdapter,
        *, mask_scorer=None,
    ) -> KernelSHAPResult:
        m = space.num_components
        if m == 0:
            raise ValueError("Interpretable space has 0 components")

        self._logger.info(
            f"Explaining anomaly score with SHAP KernelExplainer (M={m}, nsamples={self.num_samples})"
        )

        z0 = [0] * m
        z1 = [1] * m

        cache: Dict[Tuple[int, ...], float] = {}

        def score(mask):
            raw = np.asarray(mask)
            if raw.shape != (m,) or not np.all((raw == 0) | (raw == 1)):
                raise ValueError("Scorer requires a binary mask aligned with components")
            key = tuple(int(v) for v in raw)
            if key not in cache:
                value = (mask_scorer(key) if mask_scorer is not None else
                         adapter.predict_anomaly_score(perturbation.apply_mask(key).dataset))
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("Mask scorer returned a non-finite anomaly score")
                cache[key] = value
            return cache[key]

        baseline_score, original_score = score(z0), score(z1)
        self._logger.info(f"Baseline score f(0): {baseline_score:.6f}")
        self._logger.info(f"Original score f(1): {original_score:.6f}")
        shap_values, diagnostics = self._explain_with_shap_library(
            space, perturbation, adapter, baseline_mask=z0, mask_scorer=score)
        if len(shap_values) != m or not all(math.isfinite(v) for v in shap_values):
            raise ValueError("SHAP values must be finite and match the component count")
        expected = diagnostics["expected_value"]
        if not math.isfinite(expected):
            raise ValueError("SHAP expected_value must be finite")
        shap_sum = math.fsum(shap_values)
        residual = original_score - baseline_score - shap_sum
        tolerance = 1e-8 + 1e-6 * max(abs(original_score), abs(baseline_score), abs(shap_sum))
        warnings = []
        if abs(residual) > tolerance:
            warnings.append("SHAP reconstruction residual exceeds tolerance; check scorer consistency and solver diagnostics")
        if abs(expected - baseline_score) > tolerance:
            warnings.append("SHAP expected_value differs from f(0)")
        for warning in warnings:
            self._logger.warning(warning)
        diagnostics.update(shap_sum=shap_sum, reconstruction_residual=residual,
                           numerical_tolerance=tolerance, warnings=warnings,
                           evaluated_unique_masks=len(cache),
                           component_order=[c.component_id for c in space.components])

        ranking = _rank_by_abs_value(shap_values)
        explanation_it, explanation_en = _format_explanation(
            components=space.components,
            shap_values=shap_values,
            baseline_score=baseline_score,
            original_score=original_score,
            top_k=self.top_k,
        )

        return KernelSHAPResult(
            baseline_score=float(baseline_score),
            original_score=float(original_score),
            components=list(space.components),
            shap_values=[float(x) for x in shap_values],
            ranking=ranking,
            explanation_it=explanation_it,
            explanation_en=explanation_en,
            metadata={
                **diagnostics,
                "num_components": m,
                "num_samples": self.num_samples,
                "seed": self.seed,
                "backend": "shap.KernelExplainer",
                "perturbation_mode": perturbation.mode,
                "grouping_mode": space.build_info.get("grouping_mode"),
            },
        )

    def _explain_with_shap_library(
        self,
        space: InterpretableSpace,
        perturbation: PerturbationManager,
        adapter: OrthrusAnoAdapter,
        baseline_mask: Sequence[int],
        *, mask_scorer,
    ) -> Tuple[List[float], Dict[str, Any]]:
        try:
            import shap  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "The 'shap' package is required but not installed. Install it (e.g., pip install shap)."
            ) from e

        m = space.num_components
        baseline = np.asarray([list(baseline_mask)], dtype=float)  # (1, M)
        x = np.asarray([[1.0] * m], dtype=float)  # explain full input (1, M)

        def model(z: np.ndarray) -> np.ndarray:
            # Serial calls, with the same cached scorer used for both endpoints.
            return np.asarray([mask_scorer(row) for row in z], dtype=float)

        # SHAP's KernelExplainer may sample coalitions using NumPy's global RNG.
        # Setting a seed here improves reproducibility across runs.
        np.random.seed(self.seed)

        explainer = shap.KernelExplainer(model, baseline)
        sv = explainer.shap_values(x, nsamples=self.num_samples)

        # KernelExplainer returns array for single-output; list for multi-output.
        if isinstance(sv, list):
            if len(sv) != 1:
                raise ValueError("Expected single-output SHAP values")
            sv = sv[0]
        sv = np.asarray(sv, dtype=float)
        if sv.ndim == 2 and sv.shape[0] == 1:
            sv = sv[0]
        if sv.shape != (m,):
            sv = sv.reshape(-1)

        expected = np.asarray(explainer.expected_value, dtype=float)
        if expected.size != 1:
            raise ValueError("Expected a scalar SHAP expected_value")
        return [float(v) for v in sv], {
            "expected_value": float(expected.reshape(-1)[0]),
            "shap_version": getattr(shap, "__version__", "unknown"),
            "effective_nsamples": getattr(explainer, "nsamplesAdded", None),
            "l1_reg": getattr(explainer, "l1_reg", None),
        }


def _rank_by_abs_value(values: Sequence[float]) -> List[int]:
    return [
        i
        for i, _ in sorted(
            enumerate(values),
            key=lambda iv: (-abs(float(iv[1])), iv[0]),
        )
    ]


def _format_explanation(
    components: Sequence[InterpretableComponent],
    shap_values: Sequence[float],
    baseline_score: float,
    original_score: float,
    top_k: int,
) -> Tuple[str, str]:
    ranking = _rank_by_abs_value(shap_values)[:top_k]

    def fmt_item(i: int) -> str:
        return f"{components[i].name} (SHAP={float(shap_values[i]):+.4f})"

    top_items = "; ".join(fmt_item(i) for i in ranking)

    it = (
        "Spiegazione Kernel SHAP (black-box ORTHRUS-ano): "
        f"baseline f(0)={baseline_score:.4f}, score originale f(1)={original_score:.4f}. "
        f"Componenti più influenti (|SHAP|): {top_items}. "
        "Valori SHAP positivi aumentano lo score di anomalia, negativi lo riducono."
    )

    en = (
        "Kernel SHAP explanation (ORTHRUS-ano as a black box): "
        f"baseline f(0)={baseline_score:.4f}, original score f(1)={original_score:.4f}. "
        f"Most influential components (|SHAP|): {top_items}. "
        "Positive SHAP values increase the anomaly score, negative values decrease it."
    )

    return it, en
