"""Entry point for the ORTHRUS-ano Kernel SHAP explainability pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from arg_parser import parse_args
from config import AppConfig, KernelSHAPConfig, LoggingConfig, PerturbationConfig
from logger import setup_logging


def _parse_level(level: str) -> int:
    return getattr(logging, level.upper(), logging.INFO)


def build_config_from_args(args) -> AppConfig:
    """Build `AppConfig` from CLI arguments."""

    output_json = Path(args.output)
    output_csv = Path(args.output_csv) if args.output_csv else None

    # Default log file under the output directory.
    log_file = Path(args.log_file) if args.log_file else (output_json.parent / "run.log")

    is_orthrus = args.mode == "orthrus"
    if is_orthrus and (args.input != "logs_input.json" or args.time_window_seconds != 60):
        raise ValueError("--input and --time-window-seconds are legacy options; use --orthrus-config for ORTHRUS")
    if not is_orthrus and (args.orthrus_config or args.mapping_rows or args.mapping_backend != "postgresql"):
        raise ValueError("Official runtime and DB mapping options require --mode orthrus")
    return AppConfig(
        input_path=Path(args.input),
        output_json_path=output_json,
        output_ranking_csv_path=output_csv,
        adapter_type=args.adapter or ("real" if is_orthrus else "dummy"),
        pipeline_mode=str(args.mode),
        device="cpu",
        shap=KernelSHAPConfig(
            num_samples=int(args.num_samples),
            seed=int(args.seed),
            top_k=int(args.top_k),
        ),
        perturbation=PerturbationConfig(
            grouping_mode=args.grouping_mode or ("node" if is_orthrus else "exec_path"),
            max_components=int(args.max_components) if args.max_components is not None else (8 if is_orthrus else 50),
            time_window_seconds=int(args.time_window_seconds),
            perturbation_mode=args.perturbation_mode or ("neutralize_edges" if is_orthrus else "drop_records"),
        ),
        logging=LoggingConfig(level=str(args.log_level), log_file=log_file),
        orthrus_config_path=Path(args.orthrus_config) if args.orthrus_config else None,
        mapping_backend=args.mapping_backend,
        mapping_rows_path=Path(args.mapping_rows) if args.mapping_rows else None,
    )


def main() -> int:
    args = parse_args()
    if args.plot_json:
        from xai.result_visualizer import export_plots_from_json
        for path in export_plots_from_json(args.plot_json):
            print(path)
        return 0
    from pipeline import run_kernel_shap_pipeline
    cfg = build_config_from_args(args)

    logger = setup_logging(
        project_name="KSHAP_ORTHRUS",
        log_file=cfg.logging.log_file,
        console_level=_parse_level(cfg.logging.level),
        file_level=logging.DEBUG,
    )

    try:
        run_kernel_shap_pipeline(cfg, logger)
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Pipeline failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
