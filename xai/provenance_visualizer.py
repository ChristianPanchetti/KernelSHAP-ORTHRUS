"""Offline provenance context. Only components carry SHAP attribution."""
from __future__ import annotations

import csv
import math
import textwrap
from collections import Counter, defaultdict

NOTE = "SHAP values apply to component nodes only; edges provide provenance context."
COLUMNS = ["Rank", "Component / semantic name", "Node ID", "Node type", "SHAP value",
           "|SHAP|", "Assigned batch edges", "Main operations", "Main counterpart entities",
           "Mapping status"]


def _get(mapping, key):
    return mapping.get(str(key), mapping.get(key, {})) or {}


def _entity(nodes, node_id):
    node = _get(nodes, node_id)
    kind = node.get("node_type", "unknown")
    # Netflow endpoint fields are deliberately never promoted to identities.
    trusted = node.get("status") == "resolved" and node.get("provenance") in ("postgresql", "offline_rows")
    path = node.get("path") if trusted and "path" not in node.get("unverified_fields", []) and kind != "netflow" else None
    return str(path or f"node:{node_id}"), str(kind), node.get("status", "missing")


def _ranked(counter, limit=None):
    items = sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    return items if limit is None else items[:limit]


def _frequencies(counter, limit=3):
    items = _ranked(counter, limit)
    text = ", ".join(f"{key} ({count})" for key, count in items)
    if len(counter) > len(items):
        text += f", +{len(counter) - len(items)} more"
    return text or "unavailable"


def summarize(result):
    """Return all component rows and directional context without mutating JSON."""
    payload = result.to_dict() if hasattr(result, "to_dict") else result
    metadata = payload.get("metadata", {}) or {}
    case = metadata.get("case", {}) or {}
    nodes = case.get("node_mapping", {}) or {}
    mappings = case.get("component_mapping", {}) or {}
    events = metadata.get("edge_to_original_metadata", {}) or {}
    records = []
    for rank, component in enumerate(sorted(payload["components"], key=lambda c: -abs(float(c["shap_value"]))), 1):
        cid = component["component_id"]
        mapping = _get(mappings, cid)
        sources = [str(n) for n in (mapping.get("source_node_ids", []) or [])]
        if not sources and cid.startswith("node:"):
            sources = [cid.split(":", 1)[1]]
        aggregate = cid == "OTHER"
        name, kind, status = _entity(nodes, sources[0]) if len(sources) == 1 and not aggregate else (cid, "unknown", "missing")
        if aggregate:
            name = f"OTHER ({len(sources)} source nodes)" if sources else "OTHER (source count unavailable)"
            kind, status = "node group", "aggregate"
        edges = mapping.get("assigned_edges")
        count = len(edges) if edges is not None else component.get("num_records", "unavailable")
        operations, counterparts, statuses = Counter(), Counter(), Counter()
        relations = defaultdict(Counter)
        for edge in edges or []:
            event = _get(events, edge)
            statuses[event.get("status", "missing")] += 1
            operation = str(event.get("operation") or "unknown operation")
            operations[operation] += 1
            src, dst = event.get("src_index_id"), event.get("dst_index_id")
            if src is None or dst is None:
                continue
            src, dst = str(src), str(dst)
            if aggregate:
                # Assigned edges originate in the aggregate source group. Even a
                # destination that is a member remains neutral context here.
                if src in sources:
                    other, direction = dst, "out"
                elif dst in sources:
                    other, direction = src, "in"
                else:
                    continue
            elif src in sources:
                other, direction = dst, "out"
            elif dst in sources:
                other, direction = src, "in"
            else:
                continue
            counterparts[other] += 1
            relations[(other, direction)][operation] += 1
        entities = Counter()
        for other, frequency in counterparts.items():
            entity, other_kind, _ = _entity(nodes, other)
            entities[f"{entity} [{other_kind}; id={other}]"] += frequency
        value = float(component["shap_value"])
        row = dict(zip(COLUMNS, [rank, name, f"{len(sources)} source nodes" if aggregate and sources else
                                ", ".join(sources) or "unavailable", kind, value, abs(value), count,
                                _frequencies(operations), _frequencies(entities, 2),
                                f"{status}; events: {_frequencies(statuses, None)}"]))
        records.append({"component_id": cid, "row": row, "relations": dict(relations),
                        "counterparts": counterparts, "nodes": nodes})
    return records


def _short(value, width=38, maximum=115):
    text = " ".join(str(value).split())
    if len(text) > maximum:
        text = text[:maximum - 1] + "…"
    return "\n".join(textwrap.wrap(text, width))


def _figure(size):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=size)
    FigureCanvasAgg(fig)
    return fig


def build_provenance_figure(records, top_components=5, top_counterparts=2):
    """Deterministic component rows, with neutral context copies per row."""
    from matplotlib.patches import FancyArrowPatch
    selected = records[:top_components]
    fig = _figure((14, max(4, len(selected) * 2.6 + 1.5)))
    ax = fig.add_axes((.02, .10, .96, .82))
    ax.set(xlim=(0, 10), ylim=(-1, len(selected) * 3), title="SHAP-attributed provenance subgraph")
    ax.axis("off")
    for i, record in enumerate(selected):
        row = record["row"]
        y = (len(selected) - i - 1) * 3 + 1
        value = row["SHAP value"]
        color = "#c44e52" if value > 0 else "#2878b5" if value < 0 else "#777777"
        label = (f"{_short(row['Component / semantic name'])}\n"
                 f"{record['component_id']} | {row['Node type']}\n"
                 f"SHAP = {value:+.6g}\nAssigned batch edges = {row['Assigned batch edges']}")
        ax.text(1.5, y, label, ha="center", va="center", fontsize=10, parse_math=False,
                bbox=dict(boxstyle="round,pad=.6", facecolor="white", edgecolor=color, linewidth=2.5))
        chosen = _ranked(record["counterparts"], top_counterparts)
        if not chosen:
            ax.text(6.5, y, "Provenance context unavailable", ha="center", color="#666666")
        for j, (other, frequency) in enumerate(chosen):
            cy = y + (len(chosen) - 1) * .65 - j * 1.3
            entity, kind, _ = _entity(record["nodes"], other)
            ax.text(8.2, cy, f"Context only\n{_short(entity)}\n{kind} | id={other}", ha="center", va="center",
                    fontsize=9, parse_math=False,
                    bbox=dict(boxstyle="square,pad=.5", facecolor="#f0f0f0", edgecolor="#999999"))
            directions = [(direction, ops) for (node, direction), ops in record["relations"].items() if node == other]
            for k, (direction, ops) in enumerate(sorted(directions)):
                start, end = (3.0, y), (6.9, cy)
                if direction == "in":
                    start, end = end, start
                offset = (k - (len(directions) - 1) / 2) * .28
                start, end = (start[0], start[1] + offset), (end[0], end[1] + offset)
                ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                            linewidth=.8 + min(2, math.log1p(sum(ops.values())) / 3), color="#888888"))
                label = "\n".join(f"{op} × {n}" for op, n in _ranked(ops, 3))
                if len(ops) > 3:
                    label += f"\n+{len(ops) - 3} operations"
                ax.text(5, (y + cy) / 2 + offset + .12, label, ha="center", va="bottom", fontsize=8,
                        parse_math=False, bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
        omitted = sum(record["counterparts"].values()) - sum(n for _, n in chosen)
        if omitted:
            ax.text(5, y - 1.05, f"{omitted} events to other counterparts omitted", ha="center", fontsize=8, color="#666666")
    fig.text(.5, .045, NOTE + "\nRed: increases anomaly score; blue: decreases it, relative to the chosen baseline.",
             ha="center", fontsize=9)
    return fig


def build_summary_figure(records):
    fig = _figure((20, max(4, len(records) * 1.05 + 2)))
    ax = fig.add_axes((.02, .13, .96, .77))
    ax.axis("off")
    widths = [4, 30, 18, 12, 12, 12, 10, 28, 40, 25]
    cells = [[_short(record["row"][column], width, 150) if column not in ("SHAP value", "|SHAP|")
              else format(record["row"][column], "+.6g" if column == "SHAP value" else ".6g")
              for column, width in zip(COLUMNS, widths)] for record in records]
    table = ax.table(cellText=cells, colLabels=[_short(c, w) for c, w in zip(COLUMNS, widths)],
                     colWidths=[.035, .16, .09, .06, .065, .065, .055, .135, .19, .145], bbox=(0, 0, 1, 1), cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        cell.get_text().set_parse_math(False)
        cell.set_edgecolor("#dddddd")
        cell.set_facecolor("#e7edf3" if row == 0 else "white" if row % 2 else "#f5f5f5")
    fig.suptitle("Explanation summary — all SHAP components", fontsize=14)
    fig.text(.5, .045, NOTE + "\nMain operations/counterparts are frequency summaries; CSV preserves full entity labels and numeric precision.", ha="center", fontsize=10)
    return fig


def export_context(result, path):
    records = summarize(result)
    outputs = []
    target = path.with_name(f"{path.stem}_summary_table.csv")
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(record["row"] for record in records)
    outputs.append(target)
    for kind, builder in (("provenance_subgraph", build_provenance_figure), ("summary_table", build_summary_figure)):
        fig = builder(records)
        try:
            for extension in ("png", "pdf"):
                target = path.with_name(f"{path.stem}_{kind}.{extension}")
                fig.savefig(target, dpi=160, bbox_inches="tight", pad_inches=.2)
                outputs.append(target)
        finally:
            fig.clear()
    return outputs
