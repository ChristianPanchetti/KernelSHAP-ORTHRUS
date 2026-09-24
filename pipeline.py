from __future__ import annotations

import logging
import json
from dataclasses import replace

from adapters.orthrus_ano_adapter import build_adapter, RealOrthrusAnoAdapter
from adapters.orthrus_runtime import prepare_official_orthrus_case
from config import AppConfig
from perturbation.interpretable_graph_builder import InterpretableGraphBuilder
from perturbation.perturbation_manager import PerturbationManager
from preprocessing.log_input_loader import LogInputLoader
from xai.kernel_shap_explainer import KernelSHAPExplainer, _format_explanation
from perturbation.interpretable_graph_builder import InterpretableComponent, InterpretableSpace
from perturbation.orthrus_interpretable_builder import OrthrusInterpretableBuilder
from perturbation.orthrus_perturbation_manager import OrthrusPerturbationManager
from preprocessing.orthrus_mapping import DbAssistedMappingProvider, describe_component
from xai.result_exporter import ResultExporter


def run_kernel_shap_pipeline(cfg: AppConfig, logger: logging.Logger):
    """Run the end-to-end pipeline and persist results."""

    cfg.ensure_output_dirs()

    mode = str(getattr(cfg, "pipeline_mode", "dummy"))

    if mode == "dummy":
        if cfg.adapter_type == "real":
            raise ValueError("adapter_type='real' requires pipeline_mode='orthrus'")
        return _run_dummy_pipeline(cfg, logger)

    if mode == "orthrus":
        if cfg.adapter_type == "dummy":
            raise ValueError("adapter_type='dummy' requires pipeline_mode='dummy'")
        return _run_orthrus_pipeline(cfg, logger)

    raise ValueError(f"Unsupported pipeline_mode: {mode}. Supported: ['dummy', 'orthrus']")


def _run_dummy_pipeline(cfg: AppConfig, logger: logging.Logger):
    """Current debug pipeline: LogDataset + interpretable builder + record-level perturbations."""

    loader = LogInputLoader(logger=logger)
    dataset = loader.load(cfg.input_path)

    builder = InterpretableGraphBuilder(
        grouping_mode=cfg.perturbation.grouping_mode,
        max_components=cfg.perturbation.max_components,
        time_window_seconds=cfg.perturbation.time_window_seconds,
    )
    space = builder.build(dataset)

    perturbation = PerturbationManager(
        original=dataset,
        space=space,
        mode=cfg.perturbation.perturbation_mode,
    )

    adapter = build_adapter(adapter_type=cfg.adapter_type, seed=cfg.shap.seed, logger=logger)

    explainer = KernelSHAPExplainer(
        num_samples=cfg.shap.num_samples,
        seed=cfg.shap.seed,
        top_k=cfg.shap.top_k,
        logger=logger,
    )

    result = explainer.explain(space=space, perturbation=perturbation, adapter=adapter)

    exporter = ResultExporter(logger=logger)
    exporter.export_json(result, cfg.output_json_path)
    if cfg.output_ranking_csv_path is not None:
        exporter.export_ranking_csv(result, cfg.output_ranking_csv_path)

    logger.info("Pipeline completed successfully")
    return result


def _run_orthrus_pipeline(cfg: AppConfig, logger: logging.Logger):
    """One official preparation, serial isolated SHAP, then one metadata fetch."""
    from scripts.run_orthrus_official_smoke import build_config, load_config

    if cfg.adapter_type != "real":
        raise ValueError("ORTHRUS mode requires adapter_type=real")
    if cfg.perturbation.grouping_mode != "node" or cfg.perturbation.perturbation_mode != "neutralize_edges":
        raise ValueError("ORTHRUS phase 11 requires node grouping and neutralize_edges")
    if cfg.orthrus_config_path is None:
        raise ValueError("ORTHRUS mode requires --orthrus-config (official runtime JSON)")
    if cfg.perturbation.max_components is not None and cfg.perturbation.max_components < 1:
        raise ValueError("max_components must be positive")
    if cfg.mapping_backend not in {"postgresql", "offline"}:
        raise ValueError("Expected PostgreSQL or offline DB-assisted mapping")
    offline = {}
    if cfg.mapping_backend == "offline":
        if cfg.mapping_rows_path is None:
            raise ValueError("Offline mapping requires --mapping-rows")
        with cfg.mapping_rows_path.open(encoding="utf-8") as stream:
            offline = json.load(stream)
        if not isinstance(offline, dict) or any(not isinstance(offline.get(k), list) for k in ("events", "nodes")):
            raise ValueError("Offline mapping JSON requires events and nodes arrays containing all candidates")
    elif cfg.mapping_rows_path is not None:
        raise ValueError("--mapping-rows is only valid with offline mapping")

    runtime_config = build_config(load_config(cfg.orthrus_config_path))
    case, model, rel2id, runtime_warnings = prepare_official_orthrus_case(runtime_config)
    builder = OrthrusInterpretableBuilder(grouping_mode="node", max_components=cfg.perturbation.max_components)
    info = builder.build(case)
    space = _orthrus_space(case, info, cfg.perturbation.max_components)
    component_ids = tuple(c.component_id for c in space.components)
    perturbation = OrthrusPerturbationManager(case, mode="neutralize_edges")
    adapter = RealOrthrusAnoAdapter(model, device=None, isolate_state=True)

    def score(mask):
        if tuple(case.component_to_edges) != component_ids:
            raise RuntimeError("Canonical component order changed during SHAP")
        return adapter.score_mask(perturbation, mask)

    explainer = KernelSHAPExplainer(num_samples=cfg.shap.num_samples, seed=cfg.shap.seed,
                                   top_k=cfg.shap.top_k, logger=logger)
    result = explainer.explain(space, perturbation, adapter, mask_scorer=score)
    # No database access before SHAP completes. Never hide a required mapping failure.
    provider = DbAssistedMappingProvider(db_config={} if cfg.mapping_backend == "postgresql" else None,
                                         rel2id=rel2id)
    kwargs = {"rows": offline["events"], "node_rows": offline["nodes"]} if offline else {}
    provider.enrich_case(case, component_ids=component_ids, **kwargs)
    components = [replace(c, name=_orthrus_component_name(case, c.component_id),
                          description=describe_component(case, c.component_id)) for c in result.components]
    it, en = _format_explanation(components, result.shap_values, result.baseline_score,
                                 result.original_score, cfg.shap.top_k)
    it += (" I contributi riguardano la neutralizzazione delle feature correnti per nodo e ruolo, "
           "a storia e topologia fisse; non sono prove causali sugli eventi originali. "
           "OTHER resta aggregata. Le righe riportate indicano l'effetto della singola componente disattivata.")
    en += (" Contributions describe current node features neutralized per role at fixed history and topology; "
           "they are not causal evidence about original events. OTHER remains aggregated. "
           "Reported feature rows describe disabling one component, not all SHAP coalitions.")
    metadata = {**result.metadata, "pipeline_mode": "orthrus",
                "index_semantics": "record_indices and num_records denote local batch edge positions/counts",
                "case": dict(case.metadata), "edge_to_original_metadata": case.edge_to_original_metadata,
                "edge_to_event_uuid": case.edge_to_event_uuid, "mapping_quality": case.mapping_quality,
                "mapping_backend": cfg.mapping_backend, "runtime_warnings": list(runtime_warnings),
                "device": str(runtime_config.device), "score_reduction": "mean",
                "max_components": cfg.perturbation.max_components, "num_batch_edges": case.num_edges,
                "component_edges": {cid: list(case.component_to_edges[cid]) for cid in component_ids},
                "interpretation": "fixed-history node-role feature interventions; OTHER is aggregate"}
    result = replace(result, components=components, explanation_it=it, explanation_en=en, metadata=metadata)
    exporter = ResultExporter(logger=logger)
    exporter.export_json(result, cfg.output_json_path)
    if cfg.output_ranking_csv_path is not None:
        exporter.export_ranking_csv(result, cfg.output_ranking_csv_path)
    return result


def _orthrus_space(case, info, max_components):
    ids = OrthrusInterpretableBuilder.suggested_component_ids(case)
    if not ids or any(not case.component_to_edges[cid] for cid in ids):
        raise ValueError("ORTHRUS SHAP requires nonempty components")
    if max_components is not None and len(ids) > max_components:
        raise ValueError("Builder exceeds max_components (limit 1 cannot retain a node plus OTHER); use at least 2")
    assignments = [-1] * case.num_edges
    components = []
    for j, cid in enumerate(ids):
        edges = tuple(case.component_to_edges[cid])
        for edge in edges:
            if not isinstance(edge, int) or not 0 <= edge < case.num_edges or assignments[edge] != -1:
                raise ValueError("Invalid or overlapping component edge assignment")
            assignments[edge] = j
        components.append(InterpretableComponent(cid, cid, "node_group" if cid == "OTHER" else "node",
                                                 edges, "Current node-role feature intervention"))
    if -1 in assignments:
        raise ValueError("Builder components must cover the batch")
    return InterpretableSpace(components, tuple(assignments), {"grouping_mode": info.grouping_mode,
                                                               "index_kind": "local_batch_edge"})


def _orthrus_component_name(case, cid):
    roles = case.metadata["component_mapping"][cid]
    if cid == "OTHER":
        return f"OTHER ({len(roles['source_node_ids'])} source nodes)"
    node = case.metadata["node_mapping"].get(roles["source_node_ids"][0], {})
    if node.get("status") != "resolved":
        return f"{cid} [{node.get('status', 'missing')}]"
    detail = node.get("path") or node.get("cmd") or node.get("node_uuid")
    return f"{cid} [{node['node_type']}: {detail}]"
