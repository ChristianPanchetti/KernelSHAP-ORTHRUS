"""Render existing attribution results only; no inference or mapping dependencies."""
from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path


POSITIVE = "#c44e52"
NEGATIVE = "#2878b5"


def _label(component, metadata):
    cid = component["component_id"]
    if cid == "OTHER":
        return "OTHER (aggregata)"
    case = metadata.get("case", {})
    sources = case.get("component_mapping", {}).get(cid, {}).get("source_node_ids", [])
    nodes = case.get("node_mapping", {})
    if len(sources) == 1:
        node = nodes.get(str(sources[0]), nodes.get(sources[0], {}))
        if node.get("status") == "resolved" and node.get("provenance") in ("postgresql", "offline_rows"):
            for field in ("path", "cmd"):
                detail = node.get(field)
                if detail and field not in node.get("unverified_fields", []):
                    detail = " ".join(str(detail).split())
                    if len(detail) > 80:
                        detail = detail[:39] + "…" + detail[-40:]
                    return cid + "\n" + "\n".join(textwrap.wrap(detail, 40))
    return "\n".join(textwrap.wrap(cid, 40))


def _data(result):
    payload = result.to_dict() if hasattr(result, "to_dict") else result
    components = payload["components"]
    if not components:
        raise ValueError("Cannot plot an empty explanation")
    ids = [c["component_id"] for c in components]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate component_id")
    values = [float(c["shap_value"]) for c in components]
    baseline = float(payload["baseline_score"])
    original = float(payload["original_anomaly_score"])
    if not all(math.isfinite(v) for v in values + [baseline, original]):
        raise ValueError("Plot scores and SHAP values must be finite")
    order = sorted(range(len(values)), key=lambda i: -abs(values[i]))
    labels = [_label(components[i], payload.get("metadata", {})) for i in order]
    return labels, [values[i] for i in order], baseline, original


def build_figures(result):
    """Return contribution and waterfall Figures without recomputing SHAP.

    Full identities and descriptions remain in the source JSON. All components
    are displayed; the ordering is stable descending absolute contribution.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    labels, values, baseline, original = _data(result)
    height = max(5, 2.8 + sum(max(1, label.count("\n") + 1) * .24 + .22 for label in labels))
    figures = []
    for waterfall in (False, True):
        fig = Figure(figsize=(12, height + (1.6 if waterfall else 0)), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        ax.axvline(0, color="#555555", linewidth=.8)
        colors = [POSITIVE if v > 0 else NEGATIVE if v < 0 else "#777777" for v in values]
        if not waterfall:
            bars = ax.barh(range(len(values)), values, color=colors)
            ax.set_yticks(range(len(labels)), labels)
            ax.bar_label(bars, labels=[f"{v:+.6g}" for v in values], padding=5, fontsize=9)
            ax.set_title("Contributi Kernel SHAP — tutte le componenti")
            ax.set_xlabel("Valore SHAP (positivo: aumenta lo score; negativo: lo riduce)")
        else:
            reconstructed = baseline + math.fsum(values)
            residual = original - reconstructed
            cumulative = baseline
            ax.barh(0, baseline, color="#777777")
            ax.text(baseline, 0, f"  {baseline:.6g}", va="center", fontsize=9)
            for i, (value, color) in enumerate(zip(values, colors), 1):
                end = baseline + math.fsum(values[:i])
                ax.plot([cumulative, cumulative], [i-1+.35, i-.35], color="#aaaaaa", linewidth=.8)
                ax.barh(i, value, left=cumulative, color=color)
                ax.annotate(f"{value:+.6g}", (max(cumulative, end), i), xytext=(5, 0),
                            textcoords="offset points", va="center", fontsize=9)
                cumulative = end
            for row, score in ((len(values)+1, reconstructed), (len(values)+2, original)):
                ax.barh(row, score, color="#555555" if row == len(values)+1 else "#35966a")
                ax.annotate(f"{score:.6g}", (score, row), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
            ax.set_yticks(range(len(labels)+3), ["Baseline f(0)", *labels, "Score ricostruito", "Score originale f(1)"])
            ax.set_title(f"Waterfall Kernel SHAP\nResiduo f(1) − [f(0) + Σ SHAP] = {residual:+.9g}")
            ax.set_xlabel("Score cumulativo — residuo mostrato separatamente, non attribuito a componenti")
        ax.invert_yaxis()
        ax.use_sticky_edges = False
        ax.margins(x=.25)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        for text in ax.get_yticklabels():
            text.set_parse_math(False)
        fig.supxlabel("Attribuzioni numeriche sulle componenti; OTHER resta aggregata. Nessuna eliminazione di nodi o edge.",
                      fontsize=8)
        figures.append(fig)
    return tuple(figures)


def export_plots(result, output_json_path):
    """Save four sibling files named <JSON stem>_shap_{bar,waterfall}.{png,pdf}."""
    path = Path(output_json_path)
    figures = build_figures(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    try:
        for kind, fig in zip(("bar", "waterfall"), figures):
            for extension in ("png", "pdf"):
                target = path.with_name(f"{path.stem}_shap_{kind}.{extension}")
                fig.savefig(target, dpi=160, bbox_inches="tight", pad_inches=.2)
                outputs.append(target)
    finally:
        for fig in figures:
            fig.clear()
    return outputs


def export_plots_from_json(json_path):
    path = Path(json_path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return export_plots(payload, path)
