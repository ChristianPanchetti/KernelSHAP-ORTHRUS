"""Streaming ORTHRUS inference and descriptive case selection; no SHAP or DB."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from adapters.orthrus_ano_adapter import RealOrthrusAnoAdapter

OFFICIAL_LABEL_FUNCTION = "labelling.get_ground_truth"
COLUMNS = ["dataset", "split", "graph_index", "batch_index", "global_edge_offset",
           "num_edges", "mean_anomaly_score", "min_edge_score", "median_edge_score",
           "p95_edge_score", "max_edge_score", "malicious_node_count",
           "contains_malicious_node", "malicious_src_node_count", "malicious_dst_node_count"]
GT_FIELDS = COLUMNS[11:]


def load_ground_truth(path, dataset):
    """Read a pre-exported official node-ID set, never resolve UUIDs via DB."""
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw)
    if (data.get("dataset") != dataset or data.get("ground_truth_level") != "node"
            or data.get("official_label_function") != OFFICIAL_LABEL_FUNCTION
            or not isinstance(data.get("node_ids"), list)
            or not isinstance(data.get("ground_truth_source"), str)
            or not data["ground_truth_source"].strip()):
        raise ValueError("Ground truth requires dataset, node level, official_label_function="
                         "labelling.get_ground_truth, node_ids, and ground_truth_source")
    if any(type(n) is not int or n < 0 for n in data["node_ids"]):
        raise ValueError("Official node_ids must be nonnegative integers")
    metadata = {k: v for k, v in data.items() if k != "node_ids"}
    metadata.update(export_path=str(path.resolve()), export_sha256=hashlib.sha256(raw).hexdigest(),
                    batch_fields={k: "derived from membership in the official node-ID set" for k in GT_FIELDS},
                    official_fields=["node_ids (input export only)"],
                    note="Endpoint membership is descriptive; it is not an official batch label or detection metric")
    return set(data["node_ids"]), metadata


def score_statistics(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    return {"min": float(values.min()), "max": float(values.max()),
            "mean": float(sum(values.tolist()) / len(values)), "median": float(np.median(values)),
            "percentiles": {f"p{p}": float(np.percentile(values, p)) for p in (25, 50, 75, 90, 95, 99)}}


def suggested_cases(rows):
    """Stable ties use scan order; closest-to-median uses absolute distance."""
    positive = [r for r in rows if r["contains_malicious_node"]]
    negative = [r for r in rows if not r["contains_malicious_node"]]
    score = lambda r: r["mean_anomaly_score"]
    def median(group):
        middle = np.median([score(r) for r in group])
        return min(group, key=lambda r: abs(score(r) - middle))
    choices = [
        ("highest_mean_score", rows, lambda g: max(g, key=score)),
        ("high_score_with_malicious_nodes", positive, lambda g: max(g, key=score)),
        ("ordinary_score_without_malicious_nodes", negative, median),
        ("high_score_without_malicious_nodes", negative, lambda g: max(g, key=score)),
        ("low_score_with_malicious_nodes", positive, lambda g: min(g, key=score)),
        ("median_score", rows, median),
    ]
    result = {}
    for reason, group, select in choices:
        if not group:
            result[reason] = {"available": False, "selection_reason": "No qualifying batch"}
        else:
            row = select(group)
            result[reason] = {"available": True, "selection_reason": reason,
                              **{k: row[k] for k in ("dataset", "split", "graph_index", "batch_index",
                                  "global_edge_offset", "mean_anomaly_score", *GT_FIELDS)}}
    return result


def scan_batches(sequence, adapter, output, malicious_nodes, ground_truth_metadata, *, logger=None, warnings=()):
    """Consume val/test once. Only batch statistics remain in memory."""
    logger = logger or logging.getLogger(__name__)
    if adapter._state_snapshot is not None:
        raise ValueError("Temporal scanning requires a non-isolated adapter")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_name(output.stem + "_summary.json")
    # Prevent a previous successful summary from describing a partial new run.
    if summary_path.exists():
        summary_path.unlink()
    rows = []
    edges = 0
    try:
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            stream.flush()
            for case in sequence:
                info = case.metadata
                logger.info("Scanning %s/%s batch %s ...", info["split"], info["graph_index"], info["batch_index"])
                losses = adapter.predict_edge_losses(case)
                if not np.isfinite(losses).all():
                    raise ValueError("Non-finite ORTHRUS edge losses")
                if info["split"] == "val":
                    continue
                if info["split"] != "test":
                    raise ValueError("Expected official val/test sequence")
                batch = case.temporal_data
                src = set(int(n) for n in batch.src.cpu().tolist())
                dst = set(int(n) for n in batch.dst.cpu().tolist())
                bad_src, bad_dst = src & malicious_nodes, dst & malicious_nodes
                row = {k: info[k] for k in COLUMNS[:5]}
                row.update(num_edges=len(losses), mean_anomaly_score=float(sum(losses) / len(losses)),
                           min_edge_score=float(min(losses)), max_edge_score=float(max(losses)),
                           median_edge_score=float(np.median(losses)), p95_edge_score=float(np.percentile(losses, 95)),
                           malicious_node_count=len(bad_src | bad_dst),
                           contains_malicious_node=bool(bad_src | bad_dst),
                           malicious_src_node_count=len(bad_src), malicious_dst_node_count=len(bad_dst))
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                edges += len(losses)
                logger.info("batches_scanned=%s edges_scanned=%s mean_score=%.7f", len(rows), edges, row["mean_anomaly_score"])
    finally:
        if callable(getattr(sequence, "close", None)):
            sequence.close()
    summary = {"complete": True, "total_test_batches": len(rows), "total_test_edges_scanned": edges,
               "batch_mean_scores": score_statistics([r["mean_anomaly_score"] for r in rows]),
               "ground_truth": ground_truth_metadata, "warnings": list(warnings),
               "batches_containing_malicious_nodes": sum(r["contains_malicious_node"] for r in rows),
               "descriptive_groups": {
                   name: {"batch_count": len(group), "batch_mean_scores": score_statistics([r["mean_anomaly_score"] for r in group])}
                   for name, group in (("containing_malicious_nodes", [r for r in rows if r["contains_malicious_node"]]),
                                       ("without_malicious_nodes", [r for r in rows if not r["contains_malicious_node"]]))},
               "suggested_cases": suggested_cases(rows),
               "official_orthrus_evaluation": {"computed": False,
                   "reason": "Official evaluation requires validation/test time-window loss artifacts, configured node aggregation and thresholds; batch statistics are insufficient"},
               "restart_policy": "Restart from verified training checkpoint; CSV is not a temporal-state checkpoint"}
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def run_test_scan(config_path, output, ground_truth_path, *, logger=None):
    from scripts.run_orthrus_official_smoke import build_config, load_config
    from adapters.orthrus_runtime import run_official_orthrus_smoke_test

    output = Path(output).resolve()
    config = build_config(load_config(Path(config_path)))
    nodes, metadata = load_ground_truth(ground_truth_path, config.dataset_name)
    def consume(cfg, sequence, model, warnings):
        expected = [str(Path(cfg._ground_truth_dir) / p) for p in cfg.dataset.ground_truth_relative_path]
        if metadata.get("ground_truth_version") != cfg.detection.evaluation.ground_truth_version:
            raise ValueError("Ground-truth export version differs from official evaluation configuration")
        if metadata.get("ground_truth_relative_path") != list(cfg.dataset.ground_truth_relative_path):
            raise ValueError("Ground-truth export files differ from official evaluation configuration")
        metadata["configured_official_label_files"] = expected
        return scan_batches(sequence, RealOrthrusAnoAdapter(model, device=None, isolate_state=False),
                            output, nodes, metadata, logger=logger, warnings=warnings)
    return run_official_orthrus_smoke_test(config, scan_consumer=consume)
