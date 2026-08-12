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
