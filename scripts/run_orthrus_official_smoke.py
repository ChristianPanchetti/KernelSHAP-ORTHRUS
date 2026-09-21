"""Smoke ufficiale ORTHRUS; --phase9 prepara la storia e valida quattro mask."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


class SmokeConfigError(ValueError):
    """Errore leggibile nella configurazione JSON dello smoke test."""


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SmokeConfigError(f"File di configurazione JSON non trovato: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SmokeConfigError(
            f"JSON non valido in {path}, riga {exc.lineno}, colonna {exc.colno}: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise SmokeConfigError(f"Impossibile leggere il file di configurazione: {path}") from exc
    if not isinstance(payload, dict):
        raise SmokeConfigError("La radice della configurazione JSON deve essere un oggetto")
    return payload


def build_config(payload: Mapping[str, Any]):
    # Import lazy: importare questo script non richiede Torch, PyG o ORTHRUS.
    from adapters.orthrus_runtime import OrthrusOfficialRuntimeConfig

    required = ("external_root", "config_path", "dataset_name", "model_epoch_dir")
    missing = [key for key in required if key not in payload]
    if missing:
        raise SmokeConfigError("Campi obbligatori mancanti: " + ", ".join(missing))

    allowed = {
        *required,
        "split",
        "graph_index",
        "batch_index",
        "device",
        "from_weights_path",
        "overrides",
        "require_model_epoch",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise SmokeConfigError("Campi di configurazione sconosciuti: " + ", ".join(unknown))

    values = dict(payload)
    for key in ("external_root", "config_path", "model_epoch_dir", "from_weights_path"):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    try:
        return OrthrusOfficialRuntimeConfig(**values)
    except (TypeError, ValueError) as exc:
        raise SmokeConfigError(f"Configurazione smoke test non valida: {exc}") from exc


def run_from_config(path: Path, *, phase9: bool = False):
    from adapters.orthrus_runtime import run_official_orthrus_smoke_test

    config = build_config(load_config(path))
    if phase9:
        result = run_official_orthrus_smoke_test(config, perturbative=True)
        print(json.dumps(result, indent=2))
        return result
    result = run_official_orthrus_smoke_test(config)
    print(f"score: {result.score}")
    print(f"num_edges: {result.num_edges}")
    print(f"edge_loss_count: {result.edge_loss_count}")
    print(f"dataset: {result.dataset}")
    print(f"split: {result.split}")
    print(f"batch_index: {result.batch_index}")
    print(f"device: {result.device}")
    warnings = result.warnings or ()
    print("warnings: " + ("; ".join(warnings) if warnings else "nessuno"))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Esegue uno smoke test ORTHRUS ufficiale non perturbato")
    parser.add_argument("config", type=Path, help="Path al file JSON dello smoke test")
    parser.add_argument("--phase9", action="store_true",
                        help="Verifica checkpoint, prepara prefisso temporale ufficiale ed esegue A1/B/A2/Z")
    args = parser.parse_args(argv)
    try:
        if args.phase9:
            run_from_config(args.config, phase9=True)
        else:
            run_from_config(args.config)
    except Exception as exc:
        print(f"Errore smoke test ORTHRUS: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    # Consente l'esecuzione diretta da scripts/ mantenendo gli import lazy.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    raise SystemExit(main())
