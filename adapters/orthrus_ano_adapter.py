from __future__ import annotations

import abc
from contextlib import nullcontext
import logging
import math
import numbers
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

from preprocessing.orthrus_alert_case import OrthrusAlertCase
from preprocessing.schema import LogDataset


class OrthrusAnoAdapter(abc.ABC):
    """Adapter interface for ORTHRUS-ano (black-box anomaly detector).

    IT:
    Questa classe deve incapsulare solo la parte ORTHRUS-ano:
    preprocessing compatibile, inference, reconstruction error e anomaly score.
    
    The project treats ORTHRUS-ano as a black box:

        input log / graph / window  ->  anomaly score

    This interface is the ONLY integration point.
    """

    @abc.abstractmethod
    def predict_anomaly_score(self, input_object: Any) -> float:
        """Return an anomaly score for the given input.

        The score is assumed to be a real-valued scalar.
        Higher means "more anomalous".
        """


class PlaceholderOrthrusAnoAdapter(OrthrusAnoAdapter):
    """Placeholder adapter that fails with a clear integration message."""

    def __init__(self, message: Optional[str] = None):
        self._message = message or (
            "ORTHRUS-ano is not integrated yet. "
            "Implement a concrete adapter by loading the real model/code and "
            "calling it inside predict_anomaly_score(...) -> float. "
            "For the real ORTHRUS/DARPA pipeline this will typically accept an OrthrusAlertCase."
        )

    def predict_anomaly_score(self, input_object: Any) -> float:
        raise RuntimeError(self._message)


class RealOrthrusAnoAdapter(OrthrusAnoAdapter):
    """Core adapter for an already constructed ORTHRUS-ano model.

    Model/checkpoint construction intentionally stays outside this class. The
    adapter validates an `OrthrusAlertCase`, invokes the injected model as
    `model(batch, full_data, inference=True)`, validates its per-edge losses,
    and reduces them to the scalar score required by Kernel SHAP.
    """

    def __init__(
        self,
        model: Any,
        device: Any = None,
        score_reduction: str = "mean",
        require_eval_mode: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        if model is None:
            raise ValueError("RealOrthrusAnoAdapter requires an already constructed model")
        if str(score_reduction).strip().lower() != "mean":
            raise ValueError("Unsupported score_reduction. RealOrthrusAnoAdapter currently supports only 'mean'")

        self.model = model
        self.device = device
        self.score_reduction = "mean"
        self.require_eval_mode = bool(require_eval_mode)
        self._logger = logger or logging.getLogger(__name__)

    def predict_anomaly_score(self, input_object: OrthrusAlertCase) -> float:  # type: ignore[override]
        if not isinstance(input_object, OrthrusAlertCase):
            raise TypeError(
                "RealOrthrusAnoAdapter expects an OrthrusAlertCase. "
                "Use the dummy adapter for LogDataset-based debug runs."
            )
        if input_object.temporal_data is None:
            raise ValueError("OrthrusAlertCase.temporal_data is required for real ORTHRUS inference")
        if input_object.full_data is None:
            raise ValueError("OrthrusAlertCase.full_data is required for real ORTHRUS inference")

        batch = _move_to_device_if_possible(input_object.temporal_data, self.device, name="temporal_data")
        full_data = _move_to_device_if_possible(input_object.full_data, self.device, name="full_data")
        num_edges = _infer_batch_num_edges(batch)

        if self.require_eval_mode and callable(getattr(self.model, "eval", None)):
            self.model.eval()

        with _no_grad_context():
            edge_losses = self.model(batch, full_data, inference=True)

        values = _edge_losses_as_floats(edge_losses)
        if len(values) != num_edges:
            raise ValueError(
                "ORTHRUS edge loss length mismatch: "
                f"model returned {len(values)} loss(es) for a batch with {num_edges} edge(s)"
            )
        return float(sum(values) / len(values))


def _move_to_device_if_possible(value: Any, device: Any, *, name: str) -> Any:
    if device is None or not callable(getattr(value, "to", None)):
        return value
    try:
        moved = value.to(device)
    except Exception as exc:
        raise RuntimeError(f"Could not move {name} to device {device!r}") from exc
    return value if moved is None else moved


def _infer_batch_num_edges(batch: Any) -> int:
    edge_index = getattr(batch, "edge_index", None)
    if edge_index is not None:
        shape = getattr(edge_index, "shape", None)
        if shape is not None and len(shape) == 2 and int(shape[0]) == 2:
            return int(shape[1])

    for field in ("src", "msg"):
        value = getattr(batch, field, None)
        if value is not None:
            try:
                return int(len(value))
            except TypeError:
                pass

    raise ValueError(
        "Cannot infer the number of batch edges: expected edge_index with shape [2, E], "
        "or an edge-aligned src/msg field"
    )


def _no_grad_context():
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return nullcontext()
    return torch.no_grad()


def _edge_losses_as_floats(edge_losses: Any) -> list[float]:
    if edge_losses is None:
        raise ValueError("ORTHRUS model returned no edge losses")

    if _is_torch_tensor(edge_losses):
        tensor = edge_losses.detach()
        if tensor.numel() == 0:
            raise ValueError("ORTHRUS model returned an empty edge loss tensor")
        if tensor.ndim > 1:
            tensor = tensor.squeeze()
        if tensor.ndim == 0 and tensor.numel() == 1:
            tensor = tensor.reshape(1)
        if tensor.ndim != 1:
            raise ValueError(
                "ORTHRUS edge losses must be one-dimensional or squeeze to a one-dimensional per-edge vector"
            )
        raw_values = tensor.cpu().tolist()
    else:
        if isinstance(edge_losses, (str, bytes)) or not isinstance(edge_losses, Sequence):
            raise TypeError("ORTHRUS model output must be a torch.Tensor or a numeric sequence of per-edge losses")
        if len(edge_losses) == 0:
            raise ValueError("ORTHRUS model returned an empty edge loss sequence")
        raw_values = list(edge_losses)

    values: list[float] = []
    for index, value in enumerate(raw_values):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"ORTHRUS edge loss at index {index} is not numeric: {value!r}")
        values.append(float(value))
    return values


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and all(
        hasattr(value, attr) for attr in ("detach", "numel", "squeeze")
    )


@dataclass
class DummyOrthrusAnoAdapter(OrthrusAnoAdapter):
    """A deterministic dummy adapter for end-to-end pipeline testing.

    IT/EN:
        - This does NOT represent the real ORTHRUS-ano behavior.
        - It exists only to make the explainability pipeline runnable and to enable
            technical tests/sanity checks for SHAP.

        Fake semantics (for testing only):
        - The score increases with the number of records and execution-path diversity.
        - The score also increases when "suspicious" keywords appear in:
            - `execution_path`
            - `cmdline_args`

        Suspicious keywords (case-insensitive):
        bash, sh, curl, wget, chmod, /tmp, python, nc, netcat, ssh, scp, sudo.
    """

    seed: int = 0
    logger: Optional[logging.Logger] = None

    def predict_anomaly_score(self, input_object: Any) -> float:
        if not isinstance(input_object, LogDataset):
            raise TypeError(
                "DummyOrthrusAnoAdapter expects a LogDataset. "
                "Use the real adapter for OrthrusAlertCase-based ORTHRUS/DARPA runs."
            )

        log = self.logger or logging.getLogger(__name__)

        n = len(input_object.records)
        if n == 0:
            # Baseline when everything is masked away.
            return 0.0

        paths = [r.execution_path for r in input_object.records if r.execution_path]
        unique_paths = set(paths)

        # Heuristic: more diversity and more events -> higher score.
        # The formula is intentionally simple, stable and deterministic.
        diversity = len(unique_paths)

        suspicious_total = 0.0
        suspicious_hits = set()
        suspicious_records = 0
        for r in input_object.records:
            s, hits = _suspiciousness(r.execution_path, r.cmdline_args)
            if s > 0:
                suspicious_records += 1
                suspicious_hits |= hits
            suspicious_total += s

        # Small deterministic "jitter" from seed + dataset size.
        jitter = (math.sin(self.seed + n * 0.1) + 1.0) * 0.01

        # Suspiciousness is intentionally weighted higher so that Kernel SHAP can
        # recover suspicious components as important in tests.
        raw_score = (
            0.15 * math.log1p(n)
            + 0.08 * math.log1p(diversity)
            + 0.30 * suspicious_total
            + 0.06 * math.log1p(suspicious_records)
            + 0.04 * math.log1p(len(suspicious_hits))
            + jitter
        )
        score = _sigmoid(raw_score)

        log.debug(
            "DummyOrthrusAnoAdapter: n=%d diversity=%d suspicious_total=%.3f suspicious_records=%d raw=%.4f score=%.4f",
            n,
            diversity,
            suspicious_total,
            suspicious_records,
            raw_score,
            score,
        )
        return float(score)


def build_adapter(
    adapter_type: str,
    seed: int,
    logger: logging.Logger,
    model: Any = None,
    device: Any = None,
) -> OrthrusAnoAdapter:
    """Factory to build an ORTHRUS-ano adapter."""

    adapter_type = adapter_type.lower().strip()
    if adapter_type == "dummy":
        return DummyOrthrusAnoAdapter(seed=seed, logger=logger)
    if adapter_type == "placeholder":
        return PlaceholderOrthrusAnoAdapter()
    if adapter_type == "real":
        return RealOrthrusAnoAdapter(model=model, device=device, logger=logger)

    raise ValueError(f"Unknown adapter_type: {adapter_type}")


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


_SUSPICIOUS_WEIGHTS = {
    "bash": 1.5,
    "sh": 1.0,
    "curl": 1.2,
    "wget": 1.2,
    "chmod": 1.1,
    "/tmp": 1.1,
    "python": 1.0,
    "nc": 1.0,
    "netcat": 1.2,
    "ssh": 1.0,
    "scp": 1.1,
    "sudo": 1.1,
}


def _suspiciousness(execution_path: Optional[str], cmdline_args: Optional[list[str]]):
    """Return (score, hits) for fake suspicious keyword matches.

    This is intentionally simple and deterministic. It is NOT intended to model
    real ORTHRUS-ano features.
    """

    parts = []
    if execution_path:
        parts.append(execution_path)
    if cmdline_args:
        parts.extend([str(x) for x in cmdline_args])

    text = " ".join(parts).lower()
    if not text:
        return 0.0, set()

    tokens = set(re.findall(r"[a-z0-9]+", text))

    hits = set()
    score = 0.0

    # Special-case: path pattern
    if "/tmp" in text:
        hits.add("/tmp")
        score += _SUSPICIOUS_WEIGHTS["/tmp"]

    for kw, w in _SUSPICIOUS_WEIGHTS.items():
        if kw == "/tmp":
            continue
        if kw == "python":
            if any(t.startswith("python") for t in tokens):
                hits.add("python")
                score += w
            continue

        if kw in tokens:
            hits.add(kw)
            score += w

    return score, hits
