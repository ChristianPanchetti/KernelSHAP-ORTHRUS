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
            "'orthrus' is the future ORTHRUS/DARPA path based on OrthrusAlertCase (stub) (default: dummy)"
        ),
    )

    parser.add_argument(
        "--adapter",
        type=str,
        choices=["dummy", "placeholder", "real"],
        default="dummy",
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
            "exec_path",
            "record",
            "entity_id",
            "parent_child",
            "time_window",
            "event_type",
            "local_subgraph",
        ],
        default="exec_path",
        help="How to build binary components (default: exec_path)",
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
        default=50,
        help="Max number of interpretable components (default: 50)",
    )
    parser.add_argument(
        "--perturbation-mode",
        type=str,
        choices=[
            "drop_records",
            "neutralize_command",
            "mask_features",
            "drop_time_window",
            "drop_subgraph",
            "remove_node_with_incident_edges",
        ],
        default="drop_records",
        help="Perturbation strategy (default: drop_records)",
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

    return parser.parse_args()