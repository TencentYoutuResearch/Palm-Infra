"""Selective, streaming access to sharded safetensors checkpoints.

This module deliberately does not depend on PyTorch or safetensors. It parses
the small JSON headers and exposes byte ranges that the package writer can
copy directly into a `.mollm` file.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from transpile import (
    Precision,
    StreamedWeight,
    WEIGHT_FLAG_INT4_BG32,
    WEIGHT_FLAG_FP8_BLOCK128,
    WEIGHT_FLAG_NVFP4_Q8_PAIR,
    WeightByteRange,
)


_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "U8": 1,
    "I8": 1,
    "I32": 4,
    "I64": 8,
}


@dataclass(frozen=True)
class SafeTensorSlice:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source: WeightByteRange

    @property
    def nbytes(self) -> int:
        return self.source.size


class SafeTensorIndex:
    """Index tensor byte ranges without materializing checkpoint tensors."""

    def __init__(self, model_dir: str | Path):
        self.root = Path(model_dir)
        self._headers: dict[str, tuple[int, dict]] = {}
        index_path = self.root / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            self.weight_map = dict(index["weight_map"])
        else:
            files = sorted(self.root.glob("*.safetensors"))
            if len(files) != 1:
                raise FileNotFoundError(
                    "expected model.safetensors.index.json or one "
                    "safetensors file")
            self.weight_map = {}
            header = self._read_shard_header(files[0].name)
            for name in header:
                if name != "__metadata__":
                    self.weight_map[name] = files[0].name

    def _read_shard_header(self, shard: str) -> dict:
        path = self.root / shard
        with path.open("rb") as source:
            raw_size = source.read(8)
            if len(raw_size) != 8:
                raise ValueError(f"truncated safetensors header: {path}")
            header_size = struct.unpack("<Q", raw_size)[0]
            if header_size <= 0 or header_size > (1 << 30):
                raise ValueError(
                    f"invalid safetensors header size in {path}")
            raw_header = source.read(header_size)
            if len(raw_header) != header_size:
                raise ValueError(f"truncated safetensors header: {path}")
        header = json.loads(raw_header)
        self._headers[shard] = (8 + header_size, header)
        return header

    def tensor(self, name: str) -> SafeTensorSlice:
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"checkpoint tensor not found: {name}")
        if shard not in self._headers:
            self._read_shard_header(shard)
        data_base, header = self._headers[shard]
        entry = header[name]
        dtype = entry["dtype"]
        shape = tuple(int(dim) for dim in entry["shape"])
        begin, end = (int(value) for value in entry["data_offsets"])
        if begin < 0 or end < begin:
            raise ValueError(f"invalid byte range for {name}")
        item_size = _DTYPE_BYTES.get(dtype)
        if item_size is None:
            raise ValueError(f"unsupported safetensors dtype {dtype}: {name}")
        expected = item_size
        for dim in shape:
            expected *= dim
        if expected != end - begin:
            raise ValueError(
                f"safetensors byte-size mismatch for {name}: "
                f"{end - begin} != {expected}")
        return SafeTensorSlice(
            name=name,
            dtype=dtype,
            shape=shape,
            source=WeightByteRange(
                str(self.root / shard), data_base + begin, end - begin),
        )

    def contains(self, name: str) -> bool:
        return name in self.weight_map


def _bf16_to_fp16(chunk: bytes) -> bytes:
    if len(chunk) % 2:
        raise ValueError("BF16 stream chunk is not element-aligned")
    bf16 = np.frombuffer(chunk, dtype="<u2")
    fp32_bits = bf16.astype(np.uint32) << 16
    return fp32_bits.view(np.float32).astype(np.float16).tobytes()


def _bf16_to_fp32(chunk: bytes) -> bytes:
    if len(chunk) % 2:
        raise ValueError("BF16 stream chunk is not element-aligned")
    bf16 = np.frombuffer(chunk, dtype="<u2")
    return (bf16.astype(np.uint32) << 16).view(np.float32).tobytes()


def _fp16_to_fp32(chunk: bytes) -> bytes:
    if len(chunk) % 2:
        raise ValueError("FP16 stream chunk is not element-aligned")
    return np.frombuffer(chunk, dtype="<f2").astype(np.float32).tobytes()


def _bf16_to_fp32_plus_one(chunk: bytes) -> bytes:
    values = np.frombuffer(_bf16_to_fp32(chunk), dtype="<f4").copy()
    values += np.float32(1.0)
    return values.tobytes()


def _fp16_to_fp32_plus_one(chunk: bytes) -> bytes:
    values = np.frombuffer(chunk, dtype="<f2").astype(np.float32)
    values += np.float32(1.0)
    return values.tobytes()


def _int64_to_int32(chunk: bytes) -> bytes:
    if len(chunk) % 8:
        raise ValueError("INT64 stream chunk is not element-aligned")
    values = np.frombuffer(chunk, dtype="<i8")
    if np.any(values < np.iinfo(np.int32).min) or np.any(
            values > np.iinfo(np.int32).max):
        raise OverflowError("INT64 routing value does not fit INT32")
    return values.astype("<i4").tobytes()


def dense_streamed_weight(index: SafeTensorIndex,
                          name: str) -> StreamedWeight:
    """Create a streamed dense weight, preserving native quantization."""
    tensor = index.tensor(name)
    if tensor.dtype in ("BF16", "F16", "F32"):
        precision = {
            "BF16": Precision.FP16,
            "F16": Precision.FP16,
            "F32": Precision.FP32,
        }[tensor.dtype]
        transform = _bf16_to_fp16 if tensor.dtype == "BF16" else None
        data_range = WeightByteRange(
            tensor.source.path, tensor.source.offset,
            tensor.source.size, transform)
        return StreamedWeight(
            precision=precision, logical_shape=tensor.shape,
            data_ranges=[data_range])

    scale = index.tensor(name.rsplit(".", 1)[0] + ".scale")
    if scale.dtype != "F8_E8M0":
        raise ValueError(f"native quantized scale is not E8M0: {name}")
    if tensor.dtype == "F8_E4M3":
        if len(tensor.shape) != 2:
            raise ValueError(f"FP8 weight must be a matrix: {name}")
        n, k = tensor.shape
        expected_scale = ((n + 127) // 128, (k + 127) // 128)
        if scale.shape != expected_scale:
            raise ValueError(
                f"FP8 scale shape mismatch for {name}: "
                f"{scale.shape} != {expected_scale}")
        return StreamedWeight(
            precision=Precision.FP8_E4M3,
            logical_shape=tensor.shape,
            data_ranges=[tensor.source],
            scale_ranges=[scale.source],
            group_size=128,
            num_groups=scale.nbytes,
            flags=WEIGHT_FLAG_FP8_BLOCK128,
        )
    if tensor.dtype == "I8":
        if len(tensor.shape) != 2:
            raise ValueError(f"MXFP4 weight must be a matrix: {name}")
        n, packed_k = tensor.shape
        logical_shape = (n, packed_k * 2)
        expected_scale = (n, logical_shape[1] // 32)
        if scale.shape != expected_scale:
            raise ValueError(
                f"MXFP4 scale shape mismatch for {name}: "
                f"{scale.shape} != {expected_scale}")
        return StreamedWeight(
            precision=Precision.MXFP4,
            logical_shape=logical_shape,
            data_ranges=[tensor.source],
            scale_ranges=[scale.source],
            group_size=32,
            num_groups=scale.nbytes,
        )
    raise ValueError(f"unsupported dense dtype {tensor.dtype}: {name}")


def _fp16_to_bg32(chunk: bytes, columns: int) -> bytes:
    """Quantize complete FP16 rows into canonical FP32-scale BG32 blocks."""
    row_bytes = columns * 2
    if columns <= 0 or columns % 32 or len(chunk) % row_bytes:
        raise ValueError(
            f"invalid FP16 BG32 chunk: {len(chunk)} bytes, K={columns}")
    values = np.frombuffer(chunk, dtype="<f2").reshape(-1, columns)
    rows = values.shape[0]
    groups = columns // 32
    output = bytearray(((rows + 7) // 8) * groups * 160)
    for row_begin in range(0, rows, 8):
        valid = min(8, rows - row_begin)
        tile = values[row_begin:row_begin + valid].astype(np.float32)
        for group in range(groups):
            block_values = tile[:, group * 32:(group + 1) * 32]
            maximum = np.max(np.abs(block_values), axis=1)
            scales = np.where(maximum > 0, maximum / 7.0, 1.0).astype("<f4")
            quantized = np.rint(block_values / scales[:, None])
            quantized = np.clip(quantized, -7, 7).astype(np.int8)
            packed = ((quantized[:, 0::2].astype(np.uint8) & 0x0f) |
                      ((quantized[:, 1::2].astype(np.uint8) & 0x0f) << 4))
            block_index = (row_begin // 8) * groups + group
            offset = block_index * 160
            output[offset:offset + valid * 4] = scales.tobytes()
            for row in range(valid):
                begin = offset + 32 + row * 16
                output[begin:begin + 16] = packed[row].tobytes()
    return bytes(output)


def quantize_w4g32_streamed_weight(weight: StreamedWeight) -> StreamedWeight:
    """Convert a streamed FP16 matrix to the one supported dense W4 layout.

    Quantization happens in row tiles while the package is written, so even
    fused projections do not need a resident FP16 copy or a runtime sidecar.
    """
    if weight.precision != Precision.FP16 or len(weight.logical_shape) != 2:
        raise ValueError("W4G32 quantization requires a streamed FP16 matrix")
    rows, columns = weight.logical_shape
    if rows <= 0 or columns <= 0 or columns % 32:
        raise ValueError(f"W4G32 requires positive N and K%32=0: {(rows, columns)}")
    row_bytes = columns * 2
    output_ranges = []
    for segment in weight.data_ranges:
        if segment.written_size != segment.size or segment.size % row_bytes:
            raise ValueError("W4G32 source range must contain complete FP16 rows")
        segment_rows = segment.size // row_bytes
        previous = segment.transform

        def transform(chunk: bytes, previous=previous, columns=columns):
            if previous is not None:
                chunk = previous(chunk)
            return _fp16_to_bg32(chunk, columns)

        full_rows = segment_rows - segment_rows % 8
        if full_rows:
            full_size = full_rows * row_bytes
            output_ranges.append(WeightByteRange(
                segment.path, segment.offset, full_size, transform,
                (full_rows // 8) * (columns // 32) * 160,
                max(segment.transform_alignment, 8 * row_bytes)))
        if full_rows != segment_rows:
            tail_rows = segment_rows - full_rows
            output_ranges.append(WeightByteRange(
                segment.path, segment.offset + full_rows * row_bytes,
                tail_rows * row_bytes, transform,
                (columns // 32) * 160,
                max(segment.transform_alignment, row_bytes)))

    expected_bytes = ((rows + 7) // 8) * (columns // 32) * 160
    actual_bytes = sum(segment.written_size for segment in output_ranges)
    if actual_bytes != expected_bytes:
        raise ValueError(
            "W4G32 concatenation must preserve eight-row tile boundaries: "
            f"{actual_bytes} != {expected_bytes}")
    return StreamedWeight(
        precision=Precision.INT4,
        logical_shape=weight.logical_shape,
        data_ranges=output_ranges,
        group_size=32,
        num_groups=rows * (columns // 32),
        flags=WEIGHT_FLAG_INT4_BG32)


def quantize_w8pc_streamed_weight(weight: StreamedWeight) -> StreamedWeight:
    """Quantize a streamed FP16 matrix to row-major W8 per-channel."""
    if weight.precision != Precision.FP16 or len(weight.logical_shape) != 2:
        raise ValueError("W8PC quantization requires a streamed FP16 matrix")
    rows, columns = weight.logical_shape
    if rows <= 0 or columns <= 0:
        raise ValueError(f"W8PC requires a non-empty matrix: {(rows, columns)}")
    row_bytes = columns * 2
    data_ranges = []
    scale_ranges = []
    for segment in weight.data_ranges:
        if segment.written_size != segment.size or segment.size % row_bytes:
            raise ValueError("W8PC source range must contain complete FP16 rows")
        segment_rows = segment.size // row_bytes
        previous = segment.transform

        def values(chunk: bytes, previous=previous, columns=columns):
            if previous is not None:
                chunk = previous(chunk)
            matrix = np.frombuffer(chunk, dtype="<f2").reshape(-1, columns)
            matrix = matrix.astype(np.float32)
            maximum = np.max(np.abs(matrix), axis=1)
            scales = np.where(maximum > 0, maximum / 127.0, 1.0)
            return np.clip(
                np.rint(matrix / scales[:, None]), -127, 127
            ).astype(np.int8).tobytes()

        def scales(chunk: bytes, previous=previous, columns=columns):
            if previous is not None:
                chunk = previous(chunk)
            matrix = np.frombuffer(chunk, dtype="<f2").reshape(-1, columns)
            maximum = np.max(np.abs(matrix.astype(np.float32)), axis=1)
            return np.where(
                maximum > 0, maximum / 127.0, 1.0
            ).astype("<f4").tobytes()

        data_ranges.append(WeightByteRange(
            segment.path, segment.offset, segment.size, values,
            segment_rows * columns, row_bytes))
        scale_ranges.append(WeightByteRange(
            segment.path, segment.offset, segment.size, scales,
            segment_rows * 4, row_bytes))
    return StreamedWeight(
        precision=Precision.INT8,
        logical_shape=weight.logical_shape,
        data_ranges=data_ranges,
        scale_ranges=scale_ranges,
        group_size=columns,
        num_groups=rows)


def aggregate_dense_experts(index: SafeTensorIndex, name: str, *,
                            rows_per_expert: int) -> StreamedWeight:
    """Flatten a fused 3-D expert tensor into expert-major rows.

    Hy4 stores its routed experts as one tensor of shape
    ``[experts, rows, K]`` (for gate_up, ``rows = 2 * moe_intermediate``).
    The MoE kernel addresses expert ``e`` at row ``e * rows_per_expert`` with
    row stride ``K``, which is exactly the row order of this tensor already,
    so the payload can be streamed verbatim and only the declared shape
    changes: ``[experts * rows, K]``.

    The element count is unchanged by the reshape, which is what makes this a
    metadata-only operation. Declaring the 3-D shape instead would leave
    ``shape[1] == rows`` where the loader and kernel expect ``K``.
    """
    tensor = index.tensor(name)
    if tensor.dtype not in ("BF16", "F16", "F32"):
        raise ValueError(f"fused experts must be unquantized: {name}")
    if len(tensor.shape) != 3:
        raise ValueError(
            f"fused experts must be 3-D [experts, rows, K]: "
            f"{name} has shape {tensor.shape}")
    experts, rows, columns = tensor.shape
    if rows != rows_per_expert:
        raise ValueError(
            f"{name} has {rows} rows per expert, expected {rows_per_expert}")
    precision = {
        "BF16": Precision.FP16,
        "F16": Precision.FP16,
        "F32": Precision.FP32,
    }[tensor.dtype]
    transform = _bf16_to_fp16 if tensor.dtype == "BF16" else None
    data_range = WeightByteRange(
        tensor.source.path, tensor.source.offset, tensor.source.size,
        transform)
    return StreamedWeight(
        precision=precision,
        logical_shape=(experts * rows, columns),
        data_ranges=[data_range],
        expert_interleave_count=experts)


def fp32_streamed_weight(
        index: SafeTensorIndex, name: str,
        logical_shape: tuple[int, ...] | None = None) -> StreamedWeight:
    """Stream an unquantized tensor as FP32.

    DeepSeek-V4 stores several numerically sensitive RMSNorm/compressor
    parameters as BF16 even though its reference modules materialize them as
    FP32. Conversion happens one package-writer chunk at a time.
    """
    tensor = index.tensor(name)
    shape = logical_shape or tensor.shape
    if np.prod(shape, dtype=np.int64) != np.prod(
            tensor.shape, dtype=np.int64):
        raise ValueError(
            f"logical FP32 shape changes element count: "
            f"{tensor.shape} -> {shape}")
    if tensor.dtype == "F32":
        data_range = tensor.source
    elif tensor.dtype == "BF16":
        data_range = WeightByteRange(
            tensor.source.path, tensor.source.offset, tensor.source.size,
            _bf16_to_fp32, tensor.source.size * 2)
    elif tensor.dtype == "F16":
        data_range = WeightByteRange(
            tensor.source.path, tensor.source.offset, tensor.source.size,
            _fp16_to_fp32, tensor.source.size * 2)
    else:
        raise ValueError(f"cannot convert {tensor.dtype} to FP32: {name}")
    return StreamedWeight(
        precision=Precision.FP32,
        logical_shape=tuple(int(dim) for dim in shape),
        data_ranges=[data_range])


def fp32_plus_one_streamed_weight(
        index: SafeTensorIndex, name: str,
        logical_shape: tuple[int, ...] | None = None) -> StreamedWeight:
    """Stream a parameter whose module uses ``1 + weight`` as FP32."""
    tensor = index.tensor(name)
    shape = logical_shape or tensor.shape
    if np.prod(shape, dtype=np.int64) != np.prod(
            tensor.shape, dtype=np.int64):
        raise ValueError(
            f"logical FP32 shape changes element count: "
            f"{tensor.shape} -> {shape}")
    if tensor.dtype == "BF16":
        transform = _bf16_to_fp32_plus_one
        output_size = tensor.source.size * 2
    elif tensor.dtype == "F16":
        transform = _fp16_to_fp32_plus_one
        output_size = tensor.source.size * 2
    elif tensor.dtype == "F32":
        def transform(chunk: bytes) -> bytes:
            values = np.frombuffer(chunk, dtype="<f4").copy()
            values += np.float32(1.0)
            return values.tobytes()
        output_size = tensor.source.size
    else:
        raise ValueError(f"cannot convert {tensor.dtype} to FP32: {name}")
    return StreamedWeight(
        precision=Precision.FP32,
        logical_shape=tuple(int(dim) for dim in shape),
        data_ranges=[WeightByteRange(
            tensor.source.path, tensor.source.offset, tensor.source.size,
            transform, output_size)])


def split_mla_kv_b(index: SafeTensorIndex, name: str, *,
                   num_heads: int, nope_dim: int, v_head_dim: int,
                   kv_lora_rank: int) -> tuple[StreamedWeight, StreamedWeight]:
    """Split a fused ``kv_b_proj`` into the absorbed ``k_b`` and ``v_b`` pair.

    The checkpoint stores one matrix ``[heads * (nope + v), kv_lora]`` that
    expands the compressed KV latent into per-head keys and values. Absorption
    needs the two halves in *opposite* orientations:

        k_b  [heads, kv_lora, nope]   -- so a query can be folded into the latent
        v_b  [heads, v_head,  kv_lora] -- values still expand out of the latent

    Only ``k_b`` is transposed; ``v_b`` keeps the checkpoint's [v, kv_lora] row
    order. Both are pure reindexing of the same bytes, so nothing here changes a
    value -- it only decides which axis the matmul consumes. The inverse
    (``k_b.transpose`` + concat) reproduces the original bit for bit, which is
    what makes it safe to do at conversion time rather than in the graph.
    """
    tensor = index.tensor(name)
    if tensor.dtype not in ("BF16", "F16"):
        raise ValueError(f"kv_b_proj must be BF16/F16 to split: {name}")
    if len(tensor.shape) != 2:
        raise ValueError(f"kv_b_proj must be a matrix: {name}")
    fused_rows, fused_cols = tensor.shape
    expected_rows = num_heads * (nope_dim + v_head_dim)
    if fused_rows != expected_rows:
        raise ValueError(
            f"{name} has {fused_rows} rows, expected "
            f"heads({num_heads}) * (nope({nope_dim}) + v({v_head_dim})) = "
            f"{expected_rows}")
    if fused_cols != kv_lora_rank:
        raise ValueError(
            f"{name} has {fused_cols} columns, expected kv_lora_rank "
            f"{kv_lora_rank}")

    stride = nope_dim + v_head_dim
    transform = _bf16_to_fp16 if tensor.dtype == "BF16" else None
    item = 2  # FP16 bytes per element after the transform

    def transpose_k_head(chunk: bytes) -> bytes:
        """Convert one [nope, kv_lora] head into [kv_lora, nope]."""
        expected = nope_dim * kv_lora_rank * item
        if len(chunk) != expected:
            raise ValueError(f"incomplete MLA K head: {len(chunk)} != {expected}")
        if tensor.dtype == "BF16":
            bits = np.frombuffer(chunk, dtype="<u2").astype(np.uint32) << 16
            values = bits.view(np.float32).astype("<f2")
        else:
            values = np.frombuffer(chunk, dtype="<f2")
        return values.reshape(nope_dim, kv_lora_rank).T.copy().tobytes()

    # Every head's k block starts at head * stride, and the k rows are the
    # first `nope_dim` of that block; v follows immediately. Element offsets
    # are per-row, so one range per head per output.
    k_ranges = []
    v_ranges = []
    for head in range(num_heads):
        k_row_begin = head * stride
        v_row_begin = k_row_begin + nope_dim
        k_ranges.append(WeightByteRange(
            tensor.source.path,
            tensor.source.offset + k_row_begin * fused_cols * item,
            nope_dim * fused_cols * item, transpose_k_head,
            transform_alignment=nope_dim * fused_cols * item))
        v_ranges.append(WeightByteRange(
            tensor.source.path,
            tensor.source.offset + v_row_begin * fused_cols * item,
            v_head_dim * fused_cols * item, transform))

    k_b = StreamedWeight(
        precision=Precision.FP16,
        logical_shape=(num_heads, kv_lora_rank, nope_dim),
        data_ranges=k_ranges)
    v_b = StreamedWeight(
        precision=Precision.FP16,
        logical_shape=(num_heads, v_head_dim, kv_lora_rank),
        data_ranges=v_ranges)
    return k_b, v_b


def concatenate_fp16_streamed_weights(
        index: SafeTensorIndex, names: Iterable[str],
        logical_shape: tuple[int, ...]) -> StreamedWeight:
    """Row-concatenate BF16/F16 tensors into one streamed FP16 tensor."""
    ranges = []
    elements = 0
    for name in names:
        tensor = index.tensor(name)
        if tensor.dtype not in ("BF16", "F16"):
            raise ValueError(f"cannot concatenate {tensor.dtype} as FP16: {name}")
        elements += int(np.prod(tensor.shape, dtype=np.int64))
        ranges.append(WeightByteRange(
            tensor.source.path, tensor.source.offset, tensor.source.size,
            _bf16_to_fp16 if tensor.dtype == "BF16" else None))
    expected = int(np.prod(logical_shape, dtype=np.int64))
    if elements != expected:
        raise ValueError(
            f"concatenated FP16 element count mismatch: {elements} != {expected}")
    return StreamedWeight(
        precision=Precision.FP16,
        logical_shape=tuple(int(dim) for dim in logical_shape),
        data_ranges=ranges)


def concatenate_fp16_row_slices(
        index: SafeTensorIndex,
        slices: Iterable[tuple[str, int, int]],
        logical_shape: tuple[int, ...]) -> StreamedWeight:
    """Concatenate contiguous row ranges from rank-2 tensors as FP16."""
    ranges = []
    elements = 0
    for name, row_begin, row_count in slices:
        tensor = index.tensor(name)
        if tensor.dtype not in ("BF16", "F16") or len(tensor.shape) != 2:
            raise ValueError(f"row slice is not a BF16/F16 matrix: {name}")
        rows, cols = tensor.shape
        if row_begin < 0 or row_count <= 0 or row_begin + row_count > rows:
            raise ValueError(
                f"invalid row slice [{row_begin}, {row_begin + row_count}) "
                f"for {name} with {rows} rows")
        row_bytes = cols * 2
        ranges.append(WeightByteRange(
            tensor.source.path,
            tensor.source.offset + row_begin * row_bytes,
            row_count * row_bytes,
            _bf16_to_fp16 if tensor.dtype == "BF16" else None))
        elements += row_count * cols
    expected = int(np.prod(logical_shape, dtype=np.int64))
    if elements != expected:
        raise ValueError(
            f"concatenated FP16 slice count mismatch: {elements} != {expected}")
    return StreamedWeight(
        precision=Precision.FP16,
        logical_shape=tuple(int(dim) for dim in logical_shape),
        data_ranges=ranges)


def raw_u8_streamed_weight(
        index: SafeTensorIndex, names: Iterable[str],
        logical_shape: tuple[int, ...], *,
        accepted_dtypes: tuple[str, ...] = ("U8", "F8_E4M3"),
        ) -> StreamedWeight:
    """Concatenate byte-exact tensors used by lookup tables."""
    tensors = [index.tensor(name) for name in names]
    for tensor in tensors:
        if tensor.dtype not in accepted_dtypes:
            raise ValueError(
                f"raw tensor has unsupported dtype {tensor.dtype}: "
                f"{tensor.name}")
    actual = sum(tensor.nbytes for tensor in tensors)
    expected = int(np.prod(logical_shape, dtype=np.int64))
    if actual != expected:
        raise ValueError(f"raw byte count mismatch: {actual} != {expected}")
    return StreamedWeight(
        precision=Precision.RAW_U8,
        logical_shape=tuple(int(dim) for dim in logical_shape),
        data_ranges=[tensor.source for tensor in tensors])


def integer_streamed_weight(index: SafeTensorIndex,
                            name: str,
                            logical_shape: tuple[int, ...] | None = None
                            ) -> StreamedWeight:
    tensor = index.tensor(name)
    if tensor.dtype == "I32":
        data_range = tensor.source
    elif tensor.dtype == "I64":
        data_range = WeightByteRange(
            tensor.source.path, tensor.source.offset, tensor.source.size,
            _int64_to_int32, tensor.source.size // 2)
    else:
        raise ValueError(f"integer tensor must be I32/I64: {name}")
    return StreamedWeight(
        precision=Precision.INT32,
        logical_shape=logical_shape or tensor.shape,
        data_ranges=[data_range])


def aggregate_mxfp4_experts(index: SafeTensorIndex,
                            names: Iterable[str], *,
                            interleave_expert_count: int = 0) -> StreamedWeight:
    """Concatenate per-expert MXFP4 matrices in expert-major row order."""
    tensors = [index.tensor(name) for name in names]
    if not tensors:
        raise ValueError("cannot aggregate an empty expert list")
    first = tensors[0]
    if first.dtype != "I8" or len(first.shape) != 2:
        raise ValueError(f"expert is not packed MXFP4: {first.name}")
    rows, packed_k = first.shape
    for tensor in tensors:
        if tensor.dtype != "I8" or tensor.shape != first.shape:
            raise ValueError(
                f"inconsistent MXFP4 expert shape: {tensor.name}")

    scales = [
        index.tensor(tensor.name.rsplit(".", 1)[0] + ".scale")
        for tensor in tensors
    ]
    expected_scale = (rows, packed_k * 2 // 32)
    for scale in scales:
        if scale.dtype != "F8_E8M0" or scale.shape != expected_scale:
            raise ValueError(
                f"inconsistent MXFP4 expert scale: {scale.name}")
    return StreamedWeight(
        precision=Precision.MXFP4,
        logical_shape=(len(tensors) * rows, packed_k * 2),
        data_ranges=[tensor.source for tensor in tensors],
        scale_ranges=[scale.source for scale in scales],
        group_size=32,
        num_groups=sum(scale.nbytes for scale in scales),
        expert_interleave_count=interleave_expert_count,
    )


def _repeat_f32_scalar(count: int):
    def transform(chunk: bytes) -> bytes:
        if len(chunk) != 4:
            raise ValueError("NVFP4 global scale must be one FP32 scalar")
        return chunk * count
    return transform


def _pair_pack_nvfp4(packed_k: int):
    """Reorder row-major NVFP4 bytes into [row4, group, row, byte]."""
    if packed_k <= 0 or packed_k % 8:
        raise ValueError(f"invalid NVFP4 packed K: {packed_k}")
    groups = packed_k // 8

    def transform(chunk: bytes) -> bytes:
        tile_bytes = 4 * packed_k
        if len(chunk) % tile_bytes:
            raise ValueError(
                "NVFP4 pair-pack chunk is not four-row aligned")
        values = np.frombuffer(chunk, dtype=np.uint8)
        return np.ascontiguousarray(
            values.reshape(-1, 4, groups, 8).transpose(0, 2, 1, 3)
        ).tobytes()

    return transform


def aggregate_nvfp4_experts(
        index: SafeTensorIndex,
        expert_projection_names: Iterable[Iterable[str]]) -> StreamedWeight:
    """Fuse and aggregate native NVFP4 expert projections.

    Each inner iterable contains projections that are row-concatenated for a
    single expert (gate + up, or just down). Packed E2M1 bytes stay native.
    The sidecar stores all E4M3 block scales followed by one FP32 global scale
    per output row, allowing fused projections to retain distinct globals.
    """
    experts = [list(names) for names in expert_projection_names]
    if not experts or not experts[0]:
        raise ValueError("cannot aggregate empty NVFP4 experts")

    data_ranges = []
    scale_ranges = []
    rows_per_expert = None
    logical_k = None
    total_groups = 0
    for expert_names in experts:
        expert_rows = 0
        expert_scales = []
        expert_globals = []
        for name in expert_names:
            tensor = index.tensor(name)
            if tensor.dtype != "U8" or len(tensor.shape) != 2:
                raise ValueError(f"expert is not packed NVFP4: {name}")
            rows, packed_k = tensor.shape
            current_k = packed_k * 2
            if current_k % 16 != 0:
                raise ValueError(f"NVFP4 K is not block-16 aligned: {name}")
            if logical_k is None:
                logical_k = current_k
            elif logical_k != current_k:
                raise ValueError(f"inconsistent NVFP4 K dimension: {name}")
            scale = index.tensor(name + "_scale")
            scale2 = index.tensor(name + "_scale_2")
            expected_scale = (rows, current_k // 16)
            if scale.dtype != "F8_E4M3" or scale.shape != expected_scale:
                raise ValueError(f"invalid NVFP4 block scale: {scale.name}")
            if scale2.dtype != "F32" or scale2.shape not in ((), (1,)):
                raise ValueError(f"invalid NVFP4 global scale: {scale2.name}")
            if rows % 4:
                raise ValueError(
                    f"NVFP4 rows are not pair-pack aligned: {name}")
            data_ranges.append(WeightByteRange(
                tensor.source.path, tensor.source.offset, tensor.source.size,
                _pair_pack_nvfp4(packed_k), tensor.source.size,
                4 * packed_k))
            expert_scales.append(scale.source)
            expert_globals.append(WeightByteRange(
                scale2.source.path, scale2.source.offset, scale2.source.size,
                _repeat_f32_scalar(rows), rows * 4))
            expert_rows += rows
            total_groups += rows * (current_k // 16)
        if rows_per_expert is None:
            rows_per_expert = expert_rows
        elif rows_per_expert != expert_rows:
            raise ValueError("NVFP4 experts have inconsistent fused row counts")
        # The runtime sidecar is [all block scales][one global per row].
        scale_ranges.extend(expert_scales)
        scale_ranges.extend(expert_globals)

    assert rows_per_expert is not None and logical_k is not None
    return StreamedWeight(
        precision=Precision.NVFP4,
        logical_shape=(len(experts) * rows_per_expert, logical_k),
        data_ranges=data_ranges,
        scale_ranges=scale_ranges,
        group_size=16,
        num_groups=total_groups,
        flags=WEIGHT_FLAG_NVFP4_Q8_PAIR,
        expert_interleave_count=len(experts),
    )
