from __future__ import annotations

import logging

from adapters.orthrus_ano_adapter import build_adapter
from config import AppConfig
from perturbation.interpretable_graph_builder import InterpretableGraphBuilder
from perturbation.perturbation_manager import PerturbationManager
from preprocessing.log_input_loader import LogInputLoader
from xai.kernel_shap_explainer import KernelSHAPExplainer
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
    """Scaffolding for the future ORTHRUS/DARPA pipeline."""

    raise NotImplementedError(
        "ORTHRUS pipeline mode is scaffolded but not implemented yet. "
        "Planned flow: TemporalData -> OrthrusAlertCase -> ORTHRUS interpretable builder -> ORTHRUS perturbation -> "
        "RealOrthrusAnoAdapter -> Kernel SHAP."
    )
