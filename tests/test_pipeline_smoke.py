from __future__ import annotations

from pathlib import Path


def test_cli_produces_json(tmp_path: Path, default_input_path: Path, run_main_cli, load_json):
    out_json = tmp_path / "kernel_shap_results.json"

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

    assert out_json.exists(), "Expected JSON output file to be created"
    payload = load_json(out_json)

    for key in [
        "original_anomaly_score",
        "baseline_score",
        "components",
        "ranking",
        "explanation_it",
        "explanation_en",
        "metadata",
    ]:
        assert key in payload


def test_cli_produces_csv(tmp_path: Path, default_input_path: Path, run_main_cli):
    out_json = tmp_path / "kernel_shap_results.json"
    out_csv = tmp_path / "kernel_shap_ranking.csv"

    run_main_cli(
        [
            "--input",
            str(default_input_path),
            "--output",
            str(out_json),
            "--output-csv",
            str(out_csv),
            "--num-samples",
            "50",
            "--seed",
            "0",
        ]
    )

    assert out_json.exists(), "Expected JSON output file to be created"
    assert out_csv.exists(), "Expected CSV output file to be created"

    # Basic sanity: header should be present.
    header = out_csv.read_text(encoding="utf-8").splitlines()[0]
    assert header.strip() == "rank,index,name,shap_value,abs_shap"


def _extract_component_shap(payload: dict) -> dict[str, float]:
    # Stable mapping name -> shap_value
    out: dict[str, float] = {}
    for c in payload.get("components", []):
        out[str(c["name"])] = float(c["shap_value"])
    return out


def test_reproducibility_same_seed_same_nsamples(
    tmp_path: Path,
    default_input_path: Path,
    run_main_cli,
    load_json,
):
    out1 = tmp_path / "run1.json"
    out2 = tmp_path / "run2.json"

    args = [
        "--input",
        str(default_input_path),
        "--num-samples",
        "50",
        "--seed",
        "123",
    ]

    run_main_cli([*args, "--output", str(out1)])
    run_main_cli([*args, "--output", str(out2)])

    p1 = load_json(out1)
    p2 = load_json(out2)

    assert float(p1["baseline_score"]) == float(p2["baseline_score"])
    assert float(p1["original_anomaly_score"]) == float(p2["original_anomaly_score"])

    s1 = _extract_component_shap(p1)
    s2 = _extract_component_shap(p2)
    assert s1.keys() == s2.keys()

    # Values should match exactly or be extremely close.
    for k in s1:
        assert abs(s1[k] - s2[k]) <= 1e-9


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
