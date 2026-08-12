"""Entry point for the ORTHRUS-ano Kernel SHAP explainability pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from arg_parser import parse_args
from config import AppConfig, KernelSHAPConfig, LoggingConfig, PerturbationConfig
from logger import setup_logging
from pipeline import run_kernel_shap_pipeline


def _parse_level(level: str) -> int:
    return getattr(logging, level.upper(), logging.INFO)


def build_config_from_args(args) -> AppConfig:
    """Build `AppConfig` from CLI arguments."""

    output_json = Path(args.output)
    output_csv = Path(args.output_csv) if args.output_csv else None

    # Default log file under the output directory.
    log_file = Path(args.log_file) if args.log_file else (output_json.parent / "run.log")

    return AppConfig(
        input_path=Path(args.input),
        output_json_path=output_json,
        output_ranking_csv_path=output_csv,
        adapter_type=args.adapter,
        pipeline_mode=str(args.mode),
        device="cpu",
        shap=KernelSHAPConfig(
            num_samples=int(args.num_samples),
            seed=int(args.seed),
            top_k=int(args.top_k),
        ),
        perturbation=PerturbationConfig(
            grouping_mode=str(args.grouping_mode),
            max_components=int(args.max_components) if args.max_components is not None else None,
            time_window_seconds=int(args.time_window_seconds),
            perturbation_mode=str(args.perturbation_mode),
        ),
        logging=LoggingConfig(level=str(args.log_level), log_file=log_file),
    )


def main() -> int:
    args = parse_args()
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
