#!/usr/bin/env python3
"""Regenerate the small IQ2_XXS/IQ3_XXS codebooks from llama.cpp gguf-py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf_py", help="path to llama.cpp/gguf-py")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, args.gguf_py)
    from gguf.quants import IQ2_XXS, IQ3_XXS

    arrays = {}
    for cls, name in ((IQ2_XXS, "iq2_grid"), (IQ3_XXS, "iq3_grid")):
        cls.init_grid()
        arrays[name] = cls.grid.reshape(cls.grid_shape).astype(np.uint8)
    arrays["iq_signs"] = np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8)

    lines = [
        "// Generated from llama.cpp gguf-py IQ2_XXS/IQ3_XXS codebooks.",
        "// Source: gguf/quants.py; llama.cpp is distributed under MIT.",
        "#pragma once",
        "#include <cstdint>",
    ]
    for name, array in arrays.items():
        flat = array.reshape(-1)
        lines.append(f"inline constexpr uint8_t {name}[{flat.size}] = {{")
        for offset in range(0, flat.size, 16):
            values = ", ".join(str(int(v)) for v in flat[offset:offset + 16])
            lines.append(f"    {values},")
        lines.append("};")
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
