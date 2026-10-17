"""Export official THEIA_E5 node labels for the database-free test scanner."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import importlib.util
import json
import numbers
import os
from pathlib import Path
import sys


class GroundTruthExportError(RuntimeError):
    """An official label export could not be completed."""


@contextmanager
def _official_label_modules(root):
    """Bind official flat imports without importing data/model factories."""
    names = ("config", "labelling", "provnet_utils")
    previous = {name: sys.modules.get(name) for name in names}
    previous_path = sys.path[:]
    try:
        sys.path.insert(0, str(root / "src"))
        for name in names:
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location("config", root / "src" / "config.py")
        if spec is None or spec.loader is None:
            raise GroundTruthExportError("Cannot load official ORTHRUS config.py")
        config_module = importlib.util.module_from_spec(spec)
        sys.modules["config"] = config_module
        spec.loader.exec_module(config_module)
        yield config_module, importlib.import_module("labelling")
    finally:
        sys.path[:] = previous_path
        for name, module in previous.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module


def export_from_config(config_path, output):
    """Call the official helper once; no model, checkpoint or mapping enrichment."""
    from scripts.run_orthrus_official_smoke import build_config, load_config
    from adapters.orthrus_runtime import _load_official_cfg

    config = build_config(load_config(Path(config_path)))
    if config.dataset_name != "THEIA_E5":
        raise GroundTruthExportError("This export requires dataset_name=THEIA_E5")
    root = Path(config.external_root).resolve()
    output = Path(output).resolve()
    for path in (root / "src" / "config.py", root / "src" / "labelling.py", Path(config.config_path)):
        if not path.is_file():
            raise GroundTruthExportError(f"Required official ORTHRUS file is missing: {path}")
    previous_cwd = Path.cwd()
    try:
        os.chdir(root)
        with _official_label_modules(root) as (config_module, labelling):
            cfg = _load_official_cfg(config, config_module)
            if cfg.dataset.name != "THEIA_E5":
                raise GroundTruthExportError("Loaded official configuration is not THEIA_E5")
            relative_paths = list(cfg.dataset.ground_truth_relative_path)
            if not relative_paths:
                raise GroundTruthExportError("Official THEIA_E5 ground-truth file list is empty")
            files = [Path(cfg._ground_truth_dir) / name for name in relative_paths]
            for path in files:
                if not path.is_file():
                    raise GroundTruthExportError(f"Official ground-truth file is missing: {path}")
            try:
                node_ids, _, _ = labelling.get_ground_truth(cfg)
            except Exception as exc:
                # Do not print DB exception text, which may contain credentials.
                raise GroundTruthExportError(
                    "Official labelling.get_ground_truth failed "
                    f"({type(exc).__name__}); verify PostgreSQL connectivity, official DB "
                    "configuration and ground-truth UUIDs/files"
                ) from exc
            node_ids = list(node_ids)
            if any(isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 0 for n in node_ids):
                raise GroundTruthExportError("Official helper returned invalid numeric ORTHRUS node IDs")
            node_ids = sorted({int(n) for n in node_ids})
            payload = {
                "dataset": cfg.dataset.name,
                "ground_truth_level": "node",
                "official_label_function": "labelling.get_ground_truth",
                "ground_truth_source": "Official ORTHRUS labelling.get_ground_truth(cfg): "
                    "configured node-label CSVs resolved by the official database UUID lookup",
                "ground_truth_version": cfg.detection.evaluation.ground_truth_version,
                "ground_truth_relative_path": relative_paths,
                "ground_truth_files": [str(path.resolve()) for path in files],
                "malicious_node_count": len(node_ids),
                "node_ids": node_ids,
            }
    except GroundTruthExportError:
        raise
    except Exception as exc:
        raise GroundTruthExportError(
            f"Official ground-truth setup failed ({type(exc).__name__}); "
            "verify server ORTHRUS dependencies, dataset and configuration"
        ) from exc
    finally:
        os.chdir(previous_cwd)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orthrus-config", required=True, type=Path, help="Existing real runtime JSON")
    parser.add_argument("--output", required=True, type=Path, help="Scanner ground-truth JSON output")
    args = parser.parse_args(argv)
    try:
        payload = export_from_config(args.orthrus_config, args.output)
    except Exception as exc:
        print(f"Ground-truth export failed: {exc}", file=sys.stderr)
        return 1
    print(f"Exported {payload['malicious_node_count']} official THEIA_E5 malicious node IDs to {args.output}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
