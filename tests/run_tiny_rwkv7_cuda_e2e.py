#!/usr/bin/env python3
"""Build an official-layout RWKV7 .pth and compare CPU/CUDA inference."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def build_checkpoint(path: Path, tokenizer_path: Path) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is unavailable") from exc

    generator = torch.Generator().manual_seed(20260801)
    hidden = 128
    heads = 2
    head_size = 64
    intermediate = 256
    vocab = 32
    rank = 8

    def random(shape: tuple[int, ...], scale: float = 0.02):
        return (torch.randn(shape, generator=generator) * scale).to(
            torch.bfloat16)

    def zeros(shape: tuple[int, ...]):
        return torch.zeros(shape, dtype=torch.bfloat16)

    def ones(shape: tuple[int, ...]):
        return torch.ones(shape, dtype=torch.bfloat16)

    checkpoint = {
        "emb.weight": random((vocab, hidden), 0.08),
        "head.weight": random((vocab, hidden), 0.04),
        "ln_out.weight": ones((hidden,)),
        "ln_out.bias": zeros((hidden,)),
        "blocks.0.ln0.weight": ones((hidden,)),
        "blocks.0.ln0.bias": zeros((hidden,)),
        "blocks.0.ln1.weight": ones((hidden,)),
        "blocks.0.ln1.bias": zeros((hidden,)),
        "blocks.0.ln2.weight": ones((hidden,)),
        "blocks.0.ln2.bias": zeros((hidden,)),
        "blocks.0.att.receptance.weight": random((hidden, hidden)),
        "blocks.0.att.key.weight": random((hidden, hidden)),
        "blocks.0.att.value.weight": random((hidden, hidden)),
        "blocks.0.att.output.weight": random((hidden, hidden)),
        "blocks.0.att.r_k": random((heads, head_size), 0.01),
        "blocks.0.att.ln_x.weight": ones((hidden,)),
        "blocks.0.att.ln_x.bias": zeros((hidden,)),
        "blocks.0.att.w0": torch.full(
            (1, 1, hidden), -2.0, dtype=torch.bfloat16),
        "blocks.0.att.a0": zeros((1, 1, hidden)),
        "blocks.0.att.v0": zeros((1, 1, hidden)),
        "blocks.0.att.k_k": ones((1, 1, hidden)),
        "blocks.0.att.k_a": zeros((1, 1, hidden)),
        "blocks.0.ffn.x_k": torch.full(
            (1, 1, hidden), 0.5, dtype=torch.bfloat16),
        "blocks.0.ffn.key.weight": random((intermediate, hidden)),
        "blocks.0.ffn.value.weight": random((hidden, intermediate)),
    }
    for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
        checkpoint[f"blocks.0.att.{name}"] = torch.full(
            (1, 1, hidden), 0.5, dtype=torch.bfloat16)
    for name in ("w", "a", "v", "g"):
        checkpoint[f"blocks.0.att.{name}1"] = random((hidden, rank))
        checkpoint[f"blocks.0.att.{name}2"] = random((rank, hidden))

    torch.save(checkpoint, path)
    tokenizer_path.write_text(
        "".join(
            f"{token} {repr(bytes([token]))} 1\n"
            for token in range(1, vocab)
        ),
        encoding="utf-8",
    )


def main() -> int:
    if len(sys.argv) != 4:
        print(
            f"usage: {sys.argv[0]} <models/rwkv7.py> <quantizer> <e2e-runner>",
            file=sys.stderr,
        )
        return 2

    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP: PyTorch is required for the RWKV7 .pth converter E2E")
        return 77

    converter = Path(sys.argv[1]).resolve()
    quantizer = Path(sys.argv[2]).resolve()
    runner = Path(sys.argv[3]).resolve()
    temp_dir = Path(tempfile.mkdtemp(prefix="mollm_tiny_rwkv7_"))
    try:
        checkpoint = temp_dir / "tiny-rwkv7.pth"
        tokenizer = temp_dir / "rwkv_vocab.txt"
        build_checkpoint(checkpoint, tokenizer)
        environment = os.environ.copy()
        environment["MOLLM_QUANT_HELPER"] = str(quantizer)
        environment["MOLLM_CUDA_PROFILE"] = "1"
        for quantization in ("fp16", "w8pc", "w4mixg32", "w4mixg128"):
            package = temp_dir / f"tiny-rwkv7-{quantization}.mollm"
            subprocess.run(
                [
                    sys.executable,
                    str(converter),
                    str(checkpoint),
                    str(package),
                    "--prefill-seq-len",
                    "8",
                    "--tokenizer",
                    str(tokenizer),
                    "--quant",
                    quantization,
                ],
                check=True,
                env=environment,
            )

            print(f"=== RWKV7 {quantization} CUDA package E2E ===")
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
                    f"tiny RWKV7 {quantization} CUDA package E2E used an "
                    "operator fallback")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
