from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Add project root to the Python path to resolve import errors in tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def default_input_path(project_root: Path) -> Path:
    return project_root / "logs_input.json"


@pytest.fixture()
def run_main_cli(project_root: Path):
    """Callable fixture to run `main.py` as a subprocess.

    Uses subprocess to test the real CLI entrypoint (arg parsing + file outputs).
    """

    def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
        cmd = [sys.executable, "main.py", *args]
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(
                "CLI command failed\n"
                f"cmd: {' '.join(cmd)}\n"
                f"cwd: {project_root}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )
        return proc

    return _run


@pytest.fixture()
def load_json():
    def _load(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    return _load
