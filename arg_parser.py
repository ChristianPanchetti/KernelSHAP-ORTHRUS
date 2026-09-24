"""Command-line argument parsing for the Kernel SHAP pipeline."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end Kernel SHAP explainability for ORTHRUS-ano (treated as a black box)."
        )
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="logs_input.json",
        help="Path to logs_input.json (default: logs_input.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs/kernel_shap_results.json",
        help="Output JSON path (default: outputs/kernel_shap_results.json)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional CSV path for the ranking (default: not generated)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["dummy", "orthrus"],
        default="dummy",
        help=(
            "Pipeline mode: 'dummy' uses LogDataset + Dummy adapter (debug/test); "
            "'orthrus' uses the officially prepared real model and DB-assisted mapping (default: dummy)"
        ),
    )

    parser.add_argument(
        "--adapter",
        type=str,
        choices=["dummy", "placeholder", "real"],
        default=None,
        help="Which ORTHRUS-ano adapter to use (default: dummy)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of SHAP KernelExplainer samples (nsamples) (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K components used in the natural-language summary (default: 10)",
    )

    parser.add_argument(
        "--grouping-mode",
        type=str,
        choices=[
            "node",
            "exec_path",
            "record",
            "entity_id",
            "parent_child",
            "time_window",
            "event_type",
            "local_subgraph",
        ],
        default=None,
        help="Grouping (default: dummy=exec_path, orthrus=node)",
    )
    parser.add_argument(
        "--time-window-seconds",
        type=int,
        default=60,
        help="Time window size (in seconds) for grouping-mode=time_window (default: 60)",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=None,
        help="Max components (default: dummy=50, orthrus=8)",
    )
    parser.add_argument(
        "--perturbation-mode",
        type=str,
        choices=[
            "neutralize_edges",
            "drop_records",
            "neutralize_command",
            "mask_features",
            "drop_time_window",
            "drop_subgraph",
            "remove_node_with_incident_edges",
        ],
        default=None,
        help="Perturbation (default: dummy=drop_records, orthrus=neutralize_edges)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console log level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path (default: outputs/run.log)",
    )

    parser.add_argument("--orthrus-config", type=str, help="Existing official runtime JSON (phase 9)")
    parser.add_argument("--mapping-backend", choices=["postgresql", "offline"], default="postgresql",
                        help="PostgreSQL uses libpq environment/service settings, never repository credentials")
    parser.add_argument("--mapping-rows", type=str, help="Offline mapping JSON with events and nodes arrays")
    return parser.parse_args()