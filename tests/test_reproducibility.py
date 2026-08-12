from __future__ import annotations

from pathlib import Path


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
