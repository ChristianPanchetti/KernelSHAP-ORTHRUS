from __future__ import annotations

from pathlib import Path


def test_shap_additivity_holds_with_tolerance(
    tmp_path: Path,
    default_input_path: Path,
    run_main_cli,
    load_json,
):
    out_json = tmp_path / "additivity.json"

    run_main_cli(
        [
            "--input",
            str(default_input_path),
            "--output",
            str(out_json),
            "--num-samples",
            "50",
            "--seed",
            "0",
        ]
    )

    payload = load_json(out_json)
    baseline = float(payload["baseline_score"])
    original = float(payload["original_anomaly_score"])
    shap_sum = sum(float(c["shap_value"]) for c in payload.get("components", []))

    # Local accuracy / additivity: expected_value (baseline) + sum(phi) ~= f(x)
    assert abs((baseline + shap_sum) - original) <= 1e-6
