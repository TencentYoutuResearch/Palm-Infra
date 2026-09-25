#!/usr/bin/env python3
"""GGML tensor types: byte layout, naming, and (where implemented) decoding.

The engine reaches STQ1_0 through `kernels/cpu/matmul/stq1_0.{h,cpp}`, verified
bit-exact against the reference in `tests/stq1_0_crosscheck.cpp`. This module is
the Python side of that contract: it owns the *size* of every type (so a reader
can turn a tensor directory into byte ranges without decoding anything) and,
for the types we decode, the value math.

Keeping the sizes here rather than in the reader matters because a GGUF tensor
directory records offsets, not extents -- the next tensor's offset is the only
end marker, and the last tensor has none at all. A wrong block size therefore
shows up as a silently truncated read rather than an error, so each type's
total-bytes formula is checked against the file's own tensor spacing in
`verify_gguf_reader.py`.

Only the types the Hy4 preview package actually uses are implemented: STQ1_0
for the routed experts and the K-quants plus IQ2_XXS/IQ3_XXS for everything
else. Adding a type means adding one entry and, if it is new to the engine, a
matching kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


# ---------------------------------------------------------------------------
# Block geometry
# ---------------------------------------------------------------------------
#
# Every type is described by (values per block, bytes per block). Total size is
# then values_per_block-block chunks, and a tensor whose element count is not a
# multiple of the block size is malformed.

# GGML type ids, as written into a GGUF tensor directory and the ggml_type
# enum. The gaps are types upstream reserves or removed.
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_IQ1_M = 29
GGML_TYPE_BF16 = 30
GGML_TYPE_TQ1_0 = 34
GGML_TYPE_TQ2_0 = 35
GGML_TYPE_STQ1_0 = 43


@dataclass(frozen=True)
class GgmlType:
    type_id: int
    name: str
    block_values: int
    block_bytes: int
    # How the package writer should tag these bytes, and whether the payload is
    # a direct element array (unquantized) or packed blocks.
    dtype_tag: str
    quantized: bool

    def total_bytes(self, shape: tuple[int, ...]) -> int:
        elements = 1
        for dim in shape:
            elements *= int(dim)
        if elements % self.block_values != 0:
            raise ValueError(
                f"{self.name} tensor of shape {shape} has {elements} elements, "
                f"not a multiple of the {self.block_values}-value block")
        return (elements // self.block_values) * self.block_bytes


def _type(type_id: int, name: str, values: int, size: int,
          dtype_tag: str, quantized: bool = True) -> GgmlType:
    return GgmlType(type_id, name, values, size, dtype_tag, quantized)


# Block bytes are the on-disk `block_*` structs. For the K-quants and I-quants
# the struct is packed with explicit padding, so these are measured constants,
# not sums of field sizes.
_TYPES = [
    _type(GGML_TYPE_F32, "f32", 1, 4, "F32", quantized=False),
    _type(GGML_TYPE_F16, "f16", 1, 2, "F16", quantized=False),
    _type(GGML_TYPE_BF16, "bf16", 1, 2, "BF16", quantized=False),
    _type(GGML_TYPE_Q4_0, "q4_0", 32, 18, "Q4_0"),
    _type(GGML_TYPE_Q8_0, "q8_0", 32, 34, "Q8_0"),
    _type(GGML_TYPE_Q2_K, "q2_K", 256, 84, "Q2_K"),
    _type(GGML_TYPE_Q3_K, "q3_K", 256, 110, "Q3_K"),
    _type(GGML_TYPE_Q4_K, "q4_K", 256, 144, "Q4_K"),
    _type(GGML_TYPE_Q5_K, "q5_K", 256, 176, "Q5_K"),
    _type(GGML_TYPE_Q6_K, "q6_K", 256, 210, "Q6_K"),
    _type(GGML_TYPE_IQ1_M, "iq1_m", 256, 56, "IQ1_M"),
    _type(GGML_TYPE_IQ2_XXS, "iq2_xxs", 256, 66, "IQ2_XXS"),
    _type(GGML_TYPE_IQ2_XS, "iq2_xs", 256, 74, "IQ2_XS"),
    _type(GGML_TYPE_IQ2_S, "iq2_s", 256, 82, "IQ2_S"),
    _type(GGML_TYPE_IQ3_XXS, "iq3_xxs", 256, 98, "IQ3_XXS"),
    _type(GGML_TYPE_IQ3_S, "iq3_s", 256, 110, "IQ3_S"),
    _type(GGML_TYPE_IQ4_NL, "iq4_nl", 32, 18, "IQ4_NL"),
    _type(GGML_TYPE_IQ4_XS, "iq4_xs", 256, 136, "IQ4_XS"),
    _type(GGML_TYPE_TQ1_0, "tq1_0", 256, 54, "TQ1_0"),
    _type(GGML_TYPE_TQ2_0, "tq2_0", 256, 66, "TQ2_0"),
    # AngelSlim's ternary type: fp16 scale + 32 pack bytes + 8 sign bytes.
    _type(GGML_TYPE_STQ1_0, "stq1_0", 256, 42, "STQ1_0"),
]

GGML_TYPES: dict[int, GgmlType] = {t.type_id: t for t in _TYPES}
GGML_TYPE_NAMES: dict[int, str] = {t.type_id: t.name for t in _TYPES}
GGML_TYPES_BY_NAME: dict[str, GgmlType] = {t.name: t for t in _TYPES}


def ggml_type_info(type_id: int) -> GgmlType:
    entry = GGML_TYPES.get(type_id)
    if entry is None:
        raise ValueError(f"unsupported GGML type id {type_id}")
    return entry


# ---------------------------------------------------------------------------
# STQ1_0
# ---------------------------------------------------------------------------
#
# AngelSlim's 1.3125 bpw ternary format, ggml type id 43. A 256-weight block is
# 42 bytes:
#
#     uint8_t qs[32];    // two 4-bit codes per byte
#     uint8_t sign[8];   // eight 1-bit table selects per byte
#     fp16    d;         // block scale
#
# Note the field order: the scale is *last*, not first. Every other GGML block
# type puts it first, so assuming that here reads a scale out of packed code
# bytes.
#
# Each group of four weights holds exactly one zero and three equal-magnitude
# non-zero values, so 3^4 = 81 raw combinations collapse to
# 4 zero-positions x 2^3 signs = 32, which fits five bits: four in `qs`, one in
# `sign`. The codebook turns those five bits into one byte of four 2-bit lanes,
# so decoding is a table load. Lane encoding is value+1, i.e. -1 -> 0b00,
# 0 -> 0b01, +1 -> 0b10, hence `((qpack >> 2p) & 3) - 1`.
#
# The group-to-element mapping is stride-16 within each 64-element chunk:
# group g, lane p lives at self::(g / 16) * 64 + (g % 16) + p * 16. That is not
# the natural order, and using element order here fails silently -- the values
# stay in range and simply belong to the wrong weights.
#
# This is a mirror of kernels/cpu/matmul/stq1_0.h, which
# tests/stq1_0_crosscheck.cpp pins bit-exact against the Python reference. The
# layout is asserted there rather than trusted here.

STQ1_0_BLOCK_VALUES = 256
STQ1_0_BLOCK_BYTES = 42
STQ1_0_GS = 4              # weights per sparse group
STQ1_0_QK = STQ1_0_BLOCK_VALUES
STQ1_0_QS_BYTES = STQ1_0_QK // 8       # 32
STQ1_0_SIGN_BYTES = STQ1_0_QK // 32    # 8
STQ1_0_GROUPS = STQ1_0_QK // STQ1_0_GS  # 64


def _stq1_0_codebook() -> np.ndarray:
    """Build the 32-entry codebook mapping (sign << 4) | slot to pack byte.

    Four bits are the slot: two of zero-position and two of tail signs. Bit 0
    of the tail is the sign of the second non-zero lane, bit 1 of the third;
    the first non-zero lane is always positive, and the global `sign` bit
    flips every non-zero lane at once. Each entry packs four 2-bit lanes in
    value+1 form.
    """
    table = np.zeros(32, dtype=np.uint8)
    for slot in range(16):
        zero_pos = slot >> 2
        tail = slot & 3
        # Recover the three non-zero lanes, first one positive by construction.
        signs = [1, 1 if not (tail & 1) else -1, 1 if not (tail & 2) else -1]
        lanes = [0] * 4
        tail_index = 0
        for lane in range(4):
            if lane == zero_pos:
                lanes[lane] = 0
            else:
                lanes[lane] = signs[tail_index]
                tail_index += 1
        for global_sign in range(2):
            packed = 0
            for lane, value in enumerate(lanes):
                v = -value if global_sign else value
                packed |= ((v + 1) & 3) << (2 * lane)
            table[(global_sign << 4) | slot] = packed
    return table


STQ1_0_ENTRY_SIZE = 32
_STQ1_0_CODEBOOK = _stq1_0_codebook()


def stq1_0_element_index(group: int, lane: int) -> int:
    """Element index for (group, lane) within a block.

    Stride-16 within each 64-element chunk. Shared with the C++ kernel so a
    divergence shows up as a test failure rather than as plausible-looking
    wrong numbers.
    """
    return (group // 16) * 64 + (group % 16) + lane * 16


def stq1_0_dequantize(blocks: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Decode STQ1_0 blocks into a flat value array.

    `blocks` is a uint8 array whose last axis is one 42-byte block.
    """
    blocks = _as_array(blocks)
    if blocks.shape[-1] != STQ1_0_BLOCK_BYTES:
        raise ValueError(
            f"STQ1_0 blocks must be {STQ1_0_BLOCK_BYTES} bytes, got "
            f"{blocks.shape[-1]}")
    leading = blocks.shape[:-1]
    flat = blocks.reshape(-1, STQ1_0_BLOCK_BYTES)

    # The scale trails the packed data, unlike every other GGML block type.
    d = _block_scales(
        flat[:, STQ1_0_QS_BYTES + STQ1_0_SIGN_BYTES:], 2, "<f2")
    qs = flat[:, :STQ1_0_QS_BYTES]
    sign = flat[:, STQ1_0_QS_BYTES:STQ1_0_QS_BYTES + STQ1_0_SIGN_BYTES]

    out = np.empty((flat.shape[0], STQ1_0_BLOCK_VALUES), dtype=np.float32)
    for group in range(STQ1_0_GROUPS):
        code = (qs[:, group // 2] >> (4 * (group & 1))) & 0x0F
        sgn = (sign[:, group // 8] >> (group % 8)) & 0x01
        # Widen before subtracting: these are uint8 lanes, and `0 - 1` would
        # wrap to 255 rather than producing -1. The C++ path avoids this by
        # casting to int8_t, so matching it here keeps the two bit-identical.
        qpack = _STQ1_0_CODEBOOK[(sgn << 4) | code].astype(np.int16)
        for lane in range(STQ1_0_GS):
            # Lanes are stored as value + 1, so subtract to recover {-1,0,1}.
            out[:, stq1_0_element_index(group, lane)] = (
                ((qpack >> (2 * lane)) & 3) - 1)

    values = out * d[:, None]
    return values.astype(dtype).reshape(*leading, STQ1_0_BLOCK_VALUES)


def _as_array(buffer) -> np.ndarray:
    """Copy an arbitrary buffer into a writable uint8 array, keeping its shape.

    ``np.frombuffer`` returns a read-only view, and any later reshape or cast
    on a non-owning read-only slice fails with a misleading "cannot reshape
    array of size ..." error. Reading decoders all start here so that failure
    mode cannot happen.

    Two details matter: an ndarray input must not be re-interpreted (passing it
    through ``np.frombuffer`` would flatten it to 1-D), and the copy must
    preserve shape -- ``np.array`` alone drops leading axes when the input is
    already a plain buffer.
    """
    if isinstance(buffer, np.ndarray):
        return np.array(buffer, dtype=np.uint8).reshape(buffer.shape)
    return np.array(
        np.frombuffer(buffer, dtype=np.uint8), dtype=np.uint8).reshape(-1)


def _block_scales(block_bytes: np.ndarray, scale_bytes: int,
                  dtype: str) -> np.ndarray:
    """Read the leading scale field of every block as a flat value array.

    ``block_bytes`` is (blocks, block_size). Viewing the scale slice directly
    does not do what it looks like: ``.view()`` reinterprets the *last axis*,
    so a (blocks, 2) uint8 slice becomes (blocks, 1) fp16 rather than a flat
    (blocks,) array -- and the resulting broadcast silently produces an
    n-dimensional product instead of an elementwise one. Going through the
    contiguous byte stream keeps the intent unambiguous.
    """
    raw = np.ascontiguousarray(block_bytes[:, :scale_bytes]).tobytes()
    return np.frombuffer(raw, dtype=dtype).astype(np.float32)


# ---------------------------------------------------------------------------
# q8_0
# ---------------------------------------------------------------------------
#
# The simplest block format and the one most of Hy4's attention weights use:
# 32 int8 values sharing one fp16 scale. No mins, no offsets, no codebook.
#
#   struct block_q8_0 { fp16 d; int8 qs[32]; };   // 34 bytes

Q8_0_BLOCK_VALUES = 32
Q8_0_BLOCK_BYTES = 34


def q8_0_dequantize(blocks: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Decode q8_0 blocks into a flat value array.

    `blocks` is a uint8 array whose last axis is one 34-byte block.
    """
    blocks = _as_array(blocks)
    if blocks.shape[-1] != Q8_0_BLOCK_BYTES:
        raise ValueError(
            f"q8_0 blocks must be {Q8_0_BLOCK_BYTES} bytes, got "
            f"{blocks.shape[-1]}")
    leading = blocks.shape[:-1]
    flat = blocks.reshape(-1, Q8_0_BLOCK_BYTES)

    d = _block_scales(flat, 2, "<f2")
    q = np.ascontiguousarray(flat[:, 2:]).astype(np.int8).astype(np.float32)
    return ((q * d[:, None]).reshape(*leading, Q8_0_BLOCK_VALUES)
            .astype(dtype))


def dequantize_for_shape(type_id: int, blocks: np.ndarray,
                         shape: tuple[int, ...]) -> np.ndarray:
    """Decode a tensor's blocks and reshape to its logical shape.

    Types without a decoder raise rather than returning raw bytes, so a caller
    cannot mistake undecoded storage for values.
    """
    entry = ggml_type_info(type_id)
    values = None
    if entry.name == "q8_0":
        values = q8_0_dequantize(blocks)
    elif entry.name == "stq1_0":
        values = stq1_0_dequantize(blocks)
    elif entry.name == "f32":
        return blocks.view("<f4").reshape(shape)
    elif entry.name in ("f16", "bf16"):
        if entry.name == "f16":
            return blocks.view("<f2").astype(np.float32).reshape(shape)
        return ((blocks.view("<u2").astype(np.uint32) << 16)
                .view(np.float32).reshape(shape))
    else:
        raise NotImplementedError(
            f"no decoder implemented for GGML type {entry.name!r}")
    return values.reshape(shape)


def stq1_0_row_bytes(columns: int) -> int:
    """Bytes for one output row of `columns` STQ1_0 weights."""
    if columns % STQ1_0_BLOCK_VALUES:
        raise ValueError(
            f"STQ1_0 rows must be a multiple of {STQ1_0_BLOCK_VALUES}, "
            f"got {columns}")
    return (columns // STQ1_0_BLOCK_VALUES) * STQ1_0_BLOCK_BYTES
