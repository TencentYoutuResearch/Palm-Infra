#!/usr/bin/env python3
"""Build a tiny Qwen3.5-VL package and compare CPU/CUDA inference."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

from run_tiny_qwen3_cpu_e2e import build_qwen35_fixture_model


def write_png(path: Path, width: int = 64, height: int = 48) -> None:
    """Write a dependency-free RGBA fixture with non-square color content."""
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # PNG filter: none
        for x in range(width):
            rows.extend((
                (x * 5 + y * 3) & 0xFF,
                (x * 2 + y * 7) & 0xFF,
                (x * 11 + y) & 0xFF,
                255 if (x + y) % 5 else 0,
            ))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload)) + kind + payload +
            struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" +
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)) +
        chunk(b"IDAT", zlib.compress(bytes(rows), level=9)) +
        chunk(b"IEND", b"")
    )


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
        # The fixture uses 2x2 patches rather than the real model's 16x16
        # patches. Keep its processor budget proportional so a small PNG does
        # not expand into thousands of text-side image tokens.
        (model_dir / "preprocessor_config.json").write_text(
            '{"size":{"shortest_edge":1024,"longest_edge":65536}}\n',
            encoding="utf-8",
        )
        package = temp_dir / "tiny-qwen35-vision.mollm"
        subprocess.run(
            [sys.executable, str(converter), str(model_dir), str(package)],
            check=True,
            cwd=str(converter.parent.parent),
        )

        environment = os.environ.copy()
        environment["MOLLM_CUDA_PROFILE"] = "1"
        image = temp_dir / "fixture.png"
        write_png(image)
        commands = (
            [str(runner), str(package)],
            [str(runner), str(package), "--image", str(image)],
        )
        for command in commands:
            completed = subprocess.run(
                command,
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
