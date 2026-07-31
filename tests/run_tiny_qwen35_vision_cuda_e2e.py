#!/usr/bin/env python3
"""Build a tiny Qwen3.5-VL package and compare CPU/CUDA inference."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from run_tiny_qwen3_cpu_e2e import build_qwen35_fixture_model


def main() -> int:
    if len(sys.argv) != 3:
        print(
            f"usage: {sys.argv[0]} <models/converter.py> <e2e-runner>",
            file=sys.stderr,
        )
        return 2
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("SKIP: numpy is required to build the tiny vision fixture")
        return 77

    converter = Path(sys.argv[1]).resolve()
    runner = Path(sys.argv[2]).resolve()
    if not converter.is_file() or not runner.is_file():
        print("tiny vision E2E prerequisites are missing", file=sys.stderr)
        return 2

    temp_dir = Path(tempfile.mkdtemp(prefix="mollm_tiny_qwen35_vision_"))
    try:
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        build_qwen35_fixture_model(model_dir, include_vision=True)
        package = temp_dir / "tiny-qwen35-vision.mollm"
        subprocess.run(
            [sys.executable, str(converter), str(model_dir), str(package)],
            check=True,
            cwd=str(converter.parent.parent),
        )

        environment = os.environ.copy()
        environment["MOLLM_CUDA_PROFILE"] = "1"
        completed = subprocess.run(
            [str(runner), str(package)],
            check=False,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode == 77:
            return 77
        completed.check_returncode()
        if "  fallback " in completed.stderr:
            raise AssertionError(
                "tiny Qwen3.5 vision CUDA E2E used an operator fallback")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
