"""GGUF-to-XMAP conversion for small Hy4 correctness packages.

The official gguf-py decoder supplies K-quant and IQ codebooks. Its location
is explicit so the converter does not silently depend on a sibling checkout.
The diagnostic path decodes complete quantization blocks to FP16 without
materializing whole tensors. Native K-quant weights are copied as raw blocks;
other types still use the FP16 path until their CPU kernels are implemented.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

from gguf_quants import dequantize_for_shape, ggml_type_info
from gguf_reader import GgufIndex
from transpile import Precision, StreamedWeight, WeightByteRange

_NATIVE_PRECISION = {
    "q8_0": Precision.GGUF_Q8_0,
    "q4_K": Precision.GGUF_Q4_K,
    "q5_K": Precision.GGUF_Q5_K,
    "q6_K": Precision.GGUF_Q6_K,
}


def gguf_native_streamed_weight(index: GgufIndex, name: str) -> StreamedWeight:
    """Keep a GGUF matrix in its original quantized block layout."""
    tensor = index.tensor(name)
    info = index.info(name)
    quant = ggml_type_info(info.type_id)
    precision = _NATIVE_PRECISION.get(quant.name)
    if precision is None:
        raise ValueError(f"{name}: no native kernel for {quant.name}")
    if (len(info.logical_shape) not in (2, 3) or
            info.logical_shape[-1] % quant.block_values or
            (len(info.logical_shape) == 3 and quant.name != "q8_0")):
        raise ValueError(
            f"{name}: native quantization needs aligned 2-D rows, or "
            "3-D Q8_0 rows")
    return StreamedWeight(
        precision=precision,
        logical_shape=info.logical_shape,
        data_ranges=[tensor.source])


def _official_decoder(gguf_py_path: str | None):
    if gguf_py_path:
        root = Path(gguf_py_path).resolve()
        if not (root / "gguf" / "quants.py").exists():
            raise FileNotFoundError(f"gguf-py not found at {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        gguf = importlib.import_module("gguf")
        quants = importlib.import_module("gguf.quants")
    except ImportError as error:
        raise ImportError(
            "K/IQ GGUF conversion needs gguf-py; pass --gguf-py "
            "/path/to/llama.cpp/gguf-py") from error
    return gguf, quants


def gguf_fp16_streamed_weight(
        index: GgufIndex, name: str, *,
        gguf_py_path: str | None = None) -> StreamedWeight:
    """Stream one GGUF tensor as row-major FP16 values into an XMAP weight."""
    tensor = index.tensor(name)
    info = index.info(name)
    quant = ggml_type_info(info.type_id)
    elements = int(np.prod(info.logical_shape, dtype=np.int64))
    if quant.name == "f16":
        transform = None
        alignment = 2
    elif quant.name == "f32":
        transform = lambda raw: np.frombuffer(raw, "<f4").astype("<f2").tobytes()
        alignment = 4
    elif quant.name == "bf16":
        def transform(raw: bytes) -> bytes:
            bits = np.frombuffer(raw, "<u2").astype(np.uint32) << 16
            return bits.view(np.float32).astype("<f2").tobytes()
        alignment = 2
    else:
        if info.logical_shape[-1] % quant.block_values:
            raise ValueError(
                f"{name}: row width {info.logical_shape[-1]} is not a whole "
                f"{quant.name} block")
        if quant.name not in ("q8_0", "stq1_0"):
            gguf, quants = _official_decoder(gguf_py_path)
            qtype = gguf.GGMLQuantizationType(info.type_id)
        else:
            gguf = quants = qtype = None

        def transform(raw: bytes) -> bytes:
            if len(raw) % quant.block_bytes:
                raise ValueError(f"partial {quant.name} block in {name}")
            blocks = np.frombuffer(raw, np.uint8).reshape(-1, quant.block_bytes)
            if quants is None:
                values = dequantize_for_shape(
                    info.type_id, blocks,
                    (blocks.shape[0], quant.block_values))
            else:
                values = quants.dequantize(blocks, qtype)
            return values.astype("<f2").tobytes()
        alignment = quant.block_bytes

    data_range = WeightByteRange(
        tensor.source.path, tensor.source.offset, tensor.source.size,
        transform, elements * 2, transform_alignment=alignment)
    return StreamedWeight(
        precision=Precision.FP16,
        logical_shape=info.logical_shape,
        data_ranges=[data_range])
