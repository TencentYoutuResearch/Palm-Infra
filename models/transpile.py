"""
mollm — Python graph builder + serialiser

Usage:
    g = GraphBuilder()
    hidden = g.input('hidden', (2048,))
    w_q = g.weight('layer_0_q.weights', (1536, 2048))
    q = g.matmul(hidden, w_q)
    q = g.rms_norm(q, w_norm, eps=1e-6)
    ...
    g.save('model')  # → model.graph + model_*.weights
"""

from __future__ import annotations

import struct
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import BinaryIO, Callable, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# constants (must match C++ graph.h)
# ---------------------------------------------------------------------------

GRAPH_MAGIC   = 0x4D4C4C47  # "GLLM"
GRAPH_VERSION = 3  # v3: added per-node dynamic[4] (DynamicKind)

WEIGHT_MAGIC  = 0x50414D58  # "XMAP"
WEIGHT_FLAG_INT4_Q4DOT_LEGACY = 1 << 0
WEIGHT_FLAG_INT4_BG128 = 1 << 1
WEIGHT_FLAG_INT4_BG32 = 1 << 2
WEIGHT_FLAG_FP8_BLOCK128 = 1 << 3
WEIGHT_FLAG_EXPERT_INTERLEAVED = 1 << 4
WEIGHT_FLAG_NVFP4_Q8_PAIR = 1 << 5
_CPP_QUANT_HELPER: str | None | bool = None
_CPP_QUANT_HELPER_ANNOUNCED = False
_CPP_QUANT_HELPER_MISSING_ANNOUNCED = False

class OpType(IntEnum):
    INPUT          = 0
    CONSTANT       = 1
    MATMUL         = 10
    MATMUL_BATCH   = 11
    RMS_NORM       = 20
    LAYER_NORM     = 21
    ADD_RMS_NORM   = 22
    RMS_NORM_ROPE  = 23
    QK_RMS_NORM_ROPE = 24
    GROUP_RMS_NORM = 25
    SILU           = 30
    GELU           = 31
    TANH           = 32
    ROTARY_EMBED   = 40
    SDPA           = 50
    SDPA_MLA       = 51
    RESHAPE        = 60
    PERMUTE        = 61
    CONCAT         = 62
    SLICE          = 63
    TILE           = 64
    CONTIGUOUS     = 65
    ADD            = 70
    MUL            = 71
    SIGMOID        = 72
    EXP            = 73
    SOFTPLUS       = 74
    SWIGLU         = 75
    SIGMOID_EXACT  = 76
    EXP_EXACT      = 77
    GEMV_SPARSE_A  = 78
    SIGMOID_MUL    = 79
    QUANTIZE_KV    = 80
    DEQUANTIZE_KV  = 81
    GATED_DELTANET_DECODE  = 110
    GATED_DELTANET_PREFILL = 111
    GATED_DELTANET_CONV_DECODE = 112
    GATED_DELTANET_CONV_VERIFY = 113
    MOE            = 120
    HC_PRE         = 130
    HC_POST        = 131
    HC_HEAD        = 132
    GR_REDUCE      = 133
    GR_INJECT      = 134
    PLE_LOOKUP     = 135
    PLE_GATE       = 136
    PLE_DILATED_CONV = 137
    DSV4_COMPRESSOR = 160
    DSV4_INDEXER = 161
    DSV4_SPARSE_ATTN = 162
    DSV4_GROUPED_LINEAR = 163
    SHORTCONV      = 140
    RWKV7          = 150
    RWKV_TOKEN_SHIFT = 151
    RWKV_MIX       = 152
    RWKV_L2_NORM   = 153
    RWKV_POST      = 157

class Precision(IntEnum):
    FP32 = 0
    FP16 = 1
    INT8 = 2
    INT4 = 3
    FP8_E4M3 = 4
    MXFP4 = 5
    INT32 = 6
    RAW_U8 = 7
    # NVIDIA NVFP4: packed E2M1, one E4M3 block scale per 16 values, and a
    # per-matrix FP32 global scale. Currently used by SSD-backed MoE experts.
    NVFP4 = 8


# ---------------------------------------------------------------------------
# Symbolic dim expressions (must match `struct DimExpr` in graph.h)
# ---------------------------------------------------------------------------
# Each dim of a tensor's output shape can be a symbolic expression:
#   CONST: value = out_shape[i] (static)
#   SEQ:   value = runtime_seq_len
#   MUL:   value = coeff * runtime_seq_len   (covers N * SEQ)
#   ADD:   value = coeff + runtime_seq_len   (covers N + SEQ, rare)
#   BATCH: value = runtime_batch_size         (reserved)
#   DIV:   value = runtime_seq_len // coeff   (vision spatial merge)
#
# Transpile-time symbolic propagation flows these from INPUT nodes through
# every op. The C++ runtime evaluates them via eval_dim().

class DimKind(IntEnum):
    CONST = 0
    SEQ   = 1
    MUL   = 2
    ADD   = 3
    BATCH = 4
    DIV   = 5


@dataclass
class DimExpr:
    """Symbolic expression for a single tensor dimension."""
    kind: int = DimKind.CONST
    coeff: int = 0   # multiplier (MUL) or constant term (ADD)

    @staticmethod
    def const():
        return DimExpr(DimKind.CONST, 0)

    @staticmethod
    def seq():
        return DimExpr(DimKind.SEQ, 0)

    @staticmethod
    def mul(coeff):
        return DimExpr(DimKind.MUL, coeff)

    @staticmethod
    def add(coeff):
        return DimExpr(DimKind.ADD, coeff)

    @staticmethod
    def div(coeff):
        return DimExpr(DimKind.DIV, coeff)


# Module-level SEQ symbol for use in reshape() target shapes.
# Usage: g.reshape(x, (head_dim, num_heads * SEQ))  # dim 1 = MUL(num_heads, seq)
#
# In build_graph(), callers should bind SEQ to the actual seq_len via
# SEQ.bind(seq_len) so that transpile-time shape values are correct.
# The unbound SEQ (build-time value 0) is only used for element-count
# inference; runtime evaluation uses runtime_seq_len.
class _SeqSymbol:
    """Symbol representing runtime seq_len in reshape() target shapes.

    Supports arithmetic: SEQ, N * SEQ, N + SEQ, SEQ // N. Detected by reshape()
    to construct the appropriate DimExpr.

    The build-time value (default 0) is used for shape serialization;
    runtime evaluation uses runtime_seq_len via DimExpr.
    """
    def __init__(self, build_value=0):
        self._build_value = build_value

    def bind(self, seq_len):
        """Return a new _SeqSymbol bound to a specific build-time seq_len."""
        return _SeqSymbol(seq_len)

    @property
    def build_value(self):
        return self._build_value

    def __mul__(self, other):
        if isinstance(other, int):
            return _DimExprSymbol(DimExpr(DimKind.MUL, other), self._build_value * other)
        return NotImplemented
    def __rmul__(self, other):
        if isinstance(other, int):
            return _DimExprSymbol(DimExpr(DimKind.MUL, other), self._build_value * other)
        return NotImplemented
    def __add__(self, other):
        if isinstance(other, int):
            return _DimExprSymbol(DimExpr(DimKind.ADD, other), self._build_value + other)
        return NotImplemented
    def __radd__(self, other):
        if isinstance(other, int):
            return _DimExprSymbol(DimExpr(DimKind.ADD, other), self._build_value + other)
        return NotImplemented
    def __floordiv__(self, other):
        if isinstance(other, int) and other > 0:
            if self._build_value % other != 0:
                raise ValueError(
                    f"build-time SEQ={self._build_value} is not divisible by {other}")
            return _DimExprSymbol(
                DimExpr(DimKind.DIV, other), self._build_value // other)
        return NotImplemented


@dataclass
class _DimExprSymbol:
    """Result of arithmetic on _SeqSymbol (e.g. 8 * SEQ).
    Carries both the DimExpr and the build-time value for shape serialization."""
    expr: DimExpr
    build_value: int


SEQ = _SeqSymbol()  # unbound; build_graph binds it to actual seq_len


# ---------------------------------------------------------------------------
# Activation functions (fused into MATMUL at writeback time).
# Values must match `enum class Activation` in kernels/activations.h.
# ---------------------------------------------------------------------------
class Activation(IntEnum):
    NONE = 0   # identity — fast path, no per-column branch
    SILU = 1   # x * sigmoid(x)  — SwiGLU gate
    GELU = 2   # 0.5 * x * (1 + tanh(...))  — tanh approximation
    RELU = 3   # max(0, x)
    RELU_SQUARED = 4  # max(0, x)^2


# ---------------------------------------------------------------------------
# internal node representation
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    id: int
    op_type: OpType
    inputs: list[int] = field(default_factory=list)
    out_shape: tuple[int, ...] = (0, 1, 1, 1)
    # Per-dim symbolic expression.  CONST by default (out_shape is literal).
    # When SEQ/MUL/ADD, runtime evaluates against runtime_seq_len.
    # Transpile-time symbolic propagation fills this in (see propagate_dim_exprs).
    dim_expr: tuple = field(default_factory=lambda: (DimExpr.const(),) * 4)
    out_prec: Precision = Precision.FP32
    params_i32: list[int] = field(default_factory=list)
    params_f32: list[float] = field(default_factory=list)
    params_str: list[str] = field(default_factory=list)
    weight_path: Optional[str] = None   # for weight nodes
    weight_data: Optional[np.ndarray] = None  # for inline constants


# ---------------------------------------------------------------------------
# Symbolic shape propagation — ONNX-style dynamic dim flow
# ---------------------------------------------------------------------------
#
# After the graph is built, this pass walks nodes in topological order and
# fills each node's `dim_expr[4]` field based on the dim_expr[] of its inputs.
# Runtime (C++) then reads dim_expr[] to know which dims need runtime seq_len
# evaluation; it does NOT re-derive these rules.
#
# Core invariant: a node's output dim is dynamic iff some input dim that flows
# to it is dynamic. Most ops propagate input[0]'s dim_expr directly; a few
# (MATMUL, SDPA, RESHAPE, GATED_DELTANET_PREFILL) have specialized rules.

_CONST = DimExpr.const()
_CONST4 = (_CONST, _CONST, _CONST, _CONST)


def _propagate_op(node: _Node, nodes: list) -> tuple:
    """Compute this node's dim_expr[4] from its inputs' dim_expr fields."""
    op = node.op_type
    def inp(i):
        return nodes[node.inputs[i]]

    n_in = len(node.inputs)

    if op == OpType.CONSTANT:
        return _CONST4

    if op in (OpType.RMS_NORM, OpType.GROUP_RMS_NORM,
              OpType.LAYER_NORM, OpType.ADD_RMS_NORM,
              OpType.SILU, OpType.GELU, OpType.TANH, OpType.SIGMOID, OpType.SIGMOID_EXACT,
              OpType.EXP, OpType.EXP_EXACT, OpType.SOFTPLUS,
              OpType.SWIGLU,
              OpType.ROTARY_EMBED,
              OpType.TILE, OpType.CONTIGUOUS,
              OpType.QUANTIZE_KV, OpType.DEQUANTIZE_KV,
              OpType.SHORTCONV,
              OpType.RWKV7, OpType.RWKV_TOKEN_SHIFT, OpType.RWKV_MIX,
              OpType.RWKV_L2_NORM, OpType.RWKV_POST,
              OpType.MOE,
              OpType.HC_PRE, OpType.HC_POST, OpType.HC_HEAD,
              OpType.GR_REDUCE, OpType.GR_INJECT,
              OpType.PLE_GATE, OpType.PLE_DILATED_CONV):
        return inp(0).dim_expr if n_in >= 1 else _CONST4

    if op == OpType.PLE_LOOKUP:
        token_ids = inp(0).dim_expr if n_in >= 1 else _CONST4
        return (_CONST, token_ids[0], _CONST, _CONST)

    if op in (OpType.RMS_NORM_ROPE, OpType.QK_RMS_NORM_ROPE):
        return node.dim_expr

    if op in (OpType.ADD, OpType.MUL, OpType.SIGMOID_MUL):
        return inp(0).dim_expr if n_in >= 1 else _CONST4

    if op in (OpType.MATMUL, OpType.MATMUL_BATCH, OpType.GEMV_SPARSE_A,
              OpType.DSV4_INDEXER, OpType.DSV4_SPARSE_ATTN,
              OpType.DSV4_GROUPED_LINEAR):
        a = inp(0).dim_expr if n_in >= 1 else _CONST4
        return (_CONST, a[1], _CONST, _CONST)

    if op in (OpType.SDPA, OpType.SDPA_MLA):
        q = inp(0).dim_expr if n_in >= 1 else _CONST4
        return (_CONST, q[1], _CONST, _CONST)

    if op == OpType.PERMUTE:
        order = node.params_i32[:4]
        in_de = inp(0).dim_expr if n_in >= 1 else _CONST4
        if len(order) < 4:
            return _CONST4
        return tuple(in_de[order[i]] for i in range(4))

    if op == OpType.SLICE:
        in_de = inp(0).dim_expr if n_in >= 1 else _CONST4
        return in_de  # preserve all dims' exprs

    if op == OpType.CONCAT:
        in_de = inp(0).dim_expr if n_in >= 1 else _CONST4
        return in_de

    if op == OpType.RESHAPE:
        # Reshape's dim_expr is set at construction time via SEQ symbol
        # detection (see GraphBuilder.reshape). Don't recompute from inputs.
        return node.dim_expr

    if op == OpType.GATED_DELTANET_PREFILL:
        return (_CONST, DimExpr.seq(), _CONST, _CONST)

    if op in (OpType.GATED_DELTANET_DECODE,
              OpType.GATED_DELTANET_CONV_DECODE,
              OpType.GATED_DELTANET_CONV_VERIFY):
        return _CONST4

    # Default: don't propagate (conservative).
    return _CONST4


def propagate_dim_exprs(nodes: list):
    """Fill dim_expr[] on every non-INPUT node based on inputs' dim_expr[].

    INPUT nodes must already have their dim_expr[] set (via builder.input()).
    Safe to call multiple times; idempotent.
    """
    for node in nodes:
        if node.op_type == OpType.INPUT:
            continue  # caller-set
        node.dim_expr = _propagate_op(node, nodes)


# Backward-compat alias (older code called propagate_dynamic_shapes)
propagate_dynamic_shapes = propagate_dim_exprs


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------

class GraphBuilder:
    def __init__(self):
        self._nodes: list[_Node] = []
        self._next_id = 0
        self._weight_files: dict[str, bytes] = {}  # path → binary content
        self.metadata: dict[str, str] = {}  # graph-level config (rope_theta, etc.)

    def set_metadata(self, key: str, value):
        """Set a graph metadata field. Stored in the .graph file header.
        The engine reads these to configure rope_theta, rope_dim, etc."""
        self.metadata[key] = str(value)

    def set_model_config(self, rope_dim: int, rope_theta: float,
                         hidden_size: int = 0, num_layers: int = 0,
                         vocab_size: int = 0, model_type: str = ''):
        """Convenience method to set common model config fields."""
        self.metadata['rope_dim'] = str(rope_dim)
        self.metadata['rope_theta'] = str(rope_theta)
        if hidden_size: self.metadata['hidden_size'] = str(hidden_size)
        if num_layers: self.metadata['num_layers'] = str(num_layers)
        if vocab_size: self.metadata['vocab_size'] = str(vocab_size)
        if model_type: self.metadata['model_type'] = model_type

    # ---- helpers ----

    def _add(self, op: OpType, inputs: list[int], out_shape: tuple,
             prec: Precision = Precision.FP32,
             i32: Optional[list[int]] = None,
             f32: Optional[list[float]] = None,
             s: Optional[list[str]] = None) -> int:
        nid = self._next_id
        self._next_id += 1
        self._nodes.append(_Node(
            id=nid, op_type=op, inputs=inputs,
            out_shape=self._normalize_shape(out_shape),
            out_prec=prec,
            params_i32=list(i32 or []),
            params_f32=list(f32 or []),
            params_str=list(s or []),
        ))
        return nid

    @staticmethod
    def _normalize_shape(s: tuple) -> tuple:
        """Ensure shape is exactly 4 elements."""
        s = tuple(s)
        while len(s) < 4:
            s = s + (1,)
        return s[:4]

    # ---- inputs / constants ----

    def input(self, name: str, shape: tuple,
              prec: Precision = Precision.FP32,
              dynamic: Optional[tuple] = None) -> int:
        """Declare a graph INPUT node.

        Args:
            name: input name (matched by engine at runtime).
            shape: 4-D shape (will be normalized to 4 elements).
            prec: tensor precision.
            dynamic: optional 4-tuple of DimExpr values; if given, marks
                which dims are runtime-dynamic. Default = all CONST.
                Transpile-time propagation fills downstream nodes' dim_expr
                fields automatically; callers only need to mark INPUT.
        """
        nid = self._add(OpType.INPUT, [], shape, prec, s=[name])
        if dynamic is not None:
            dyn = list(dynamic)
            while len(dyn) < 4:
                dyn.append(DimExpr.const())
            self._nodes[nid].dim_expr = tuple(dyn[:4])
        return nid

    def weight(self, path: str, shape: tuple,
               prec: Precision = Precision.FP16) -> int:
        nid = self._add(OpType.CONSTANT, [], shape, prec, s=[path])
        self._nodes[nid].weight_path = path
        return nid

    def constant(self, data: np.ndarray) -> int:
        shape = tuple(data.shape)
        prec = _numpy_to_precision(data.dtype)
        # Use a placeholder path that save() will replace with the actual
        # const file path (basename_const_<id>.weights).
        nid = self._add(OpType.CONSTANT, [], shape, prec, s=[f"__inline_const__"])
        self._nodes[nid].weight_data = data.copy()
        return nid

    # ---- linear ----

    def matmul(self, a: int, b: int, trans_b: bool = False,
               activation: Activation = Activation.NONE,
               act_n_begin: int = 0, act_n_len: int = -1) -> int:
        sa = self._nodes[a].out_shape
        sb = self._nodes[b].out_shape
        # A: [K, M], B: weight matrix (either [K, N] or [N, K])
        # out: [N, M]
        K = sa[0]
        M = sa[1]
        if trans_b:
            N = sb[1]  # B is [N, K] -> transpose -> [K, N]
            assert sb[0] == K, f"matmul K mismatch: {sa} vs {sb}"
        else:
            # Detect N by comparing sb dimensions with K
            if sb[0] == K:
                N = sb[1]  # sb[0]=K (inner), sb[1]=N (output)
            elif sb[1] == K:
                N = sb[0]  # sb[1]=K (inner), sb[0]=N (output)
            else:
                raise AssertionError(f"matmul K mismatch: {sa} vs {sb}")
        # Fused activation params: [activation, act_n_begin, act_n_len].
        # act_n_len == -1 means "apply to whole N" (fast path).
        # act_n_len == 0 means "don't apply" (handled identically to NONE).
        # act_n_len > 0 means "apply to columns [act_n_begin, act_n_begin+act_n_len)".
        return self._add(OpType.MATMUL, [a, b], (N, M),
                         prec=self._nodes[a].out_prec,
                         i32=[int(activation), int(act_n_begin), int(act_n_len)])

    def matmul_batch(self, pairs: list[tuple[int, int]]) -> list[int]:
        """Run same-shaped independent matmuls through one runtime dispatch."""
        if not pairs:
            return []
        inputs: list[int] = []
        common_n = None
        common_m = None
        for a, b in pairs:
            sa = self._nodes[a].out_shape
            sb = self._nodes[b].out_shape
            k, m = sa[0], sa[1]
            if sb[1] != k:
                raise AssertionError(f"batched matmul K mismatch: {sa} vs {sb}")
            n = sb[0]
            if common_n is None:
                common_n, common_m = n, m
            elif n != common_n or m != common_m:
                raise AssertionError(
                    f"batched matmul shape mismatch: {(n, m)} vs "
                    f"{(common_n, common_m)}")
            inputs.extend((a, b))
        merged = self._add(
            OpType.MATMUL_BATCH, inputs,
            (common_n, common_m, len(pairs)),
            prec=self._nodes[pairs[0][0]].out_prec,
            i32=[len(pairs)])
        return self.slice(merged, [1] * len(pairs), 2)

    def gemv_sparse_a(self, a: int, b: int) -> int:
        """Matmul with a decode GEMV path that skips exact-zero A entries."""
        sa = self._nodes[a].out_shape
        sb = self._nodes[b].out_shape
        K, M = sa[0], sa[1]
        if sb[1] != K:
            raise AssertionError(f"sparse GEMV K mismatch: {sa} vs {sb}")
        return self._add(OpType.GEMV_SPARSE_A, [a, b], (sb[0], M),
                         prec=self._nodes[a].out_prec)

    # ---- normalisation ----

    def rms_norm(self, x: int, weight: int, eps: float = 1e-6) -> int:
        sx = self._nodes[x].out_shape
        return self._add(OpType.RMS_NORM, [x, weight], sx,
                         prec=self._nodes[x].out_prec, f32=[eps])

    def group_rms_norm(self, x: int, weight: int, group_size: int,
                       eps: float = 1e-6) -> int:
        """RMSNorm consecutive groups while retaining per-element gamma."""
        sx = self._nodes[x].out_shape
        if group_size <= 0 or sx[0] % group_size != 0:
            raise ValueError(
                f"invalid grouped RMSNorm shape {sx} / group {group_size}")
        return self._add(
            OpType.GROUP_RMS_NORM, [x, weight], sx,
            prec=self._nodes[x].out_prec,
            i32=[group_size], f32=[eps])

    def add_rms_norm(self, residual: int, update: int, weight: int,
                     eps: float = 1e-6) -> int:
        """In-place residual add plus RMSNorm.

        The residual node remains the live residual-stream value; this op
        updates its storage before producing the normalized output.
        """
        sr = self._nodes[residual].out_shape
        su = self._nodes[update].out_shape
        assert sr == su, f"add_rms_norm shape mismatch: {sr} vs {su}"
        return self._add(
            OpType.ADD_RMS_NORM, [residual, update, weight], sr,
            prec=self._nodes[residual].out_prec, f32=[eps])

    def rms_norm_rope(self, x: int, weight: int, cos: int, sin: int,
                      seq_len, heads: int, rope_dim: int = 64,
                      interleave: bool = True, eps: float = 1e-6) -> int:
        """Fuse per-head RMSNorm, dense materialization, and RoPE."""
        dim = self._nodes[x].out_shape[0]
        dynamic_seq = isinstance(seq_len, _SeqSymbol)
        seq_static = seq_len.build_value if dynamic_seq else seq_len
        nid = self._add(
            OpType.RMS_NORM_ROPE, [x, weight, cos, sin],
            (dim, seq_static, heads), prec=self._nodes[x].out_prec,
            i32=[rope_dim, 1 if interleave else 0], f32=[eps])
        if dynamic_seq:
            self._nodes[nid].dim_expr = (
                DimExpr.const(), DimExpr.seq(),
                DimExpr.const(), DimExpr.const())
        return nid

    def qk_rms_norm_rope(self, query: int, key: int,
                         query_weight: int, key_weight: int,
                         cos: int, sin: int, seq_len,
                         query_heads: int, key_heads: int,
                         rope_dim: int = 64, interleave: bool = True,
                         eps: float = 1e-6) -> int:
        """Normalize and rotate Q and K in one parallel backend dispatch.

        The combined output is laid out as [head_dim, seq, q_heads+k_heads].
        Callers can recover Q and K with zero-copy slices along dim 2.
        """
        query_dim = self._nodes[query].out_shape[0]
        key_dim = self._nodes[key].out_shape[0]
        assert query_dim == key_dim, (
            f"Q/K head dim mismatch: {query_dim} vs {key_dim}")
        dynamic_seq = isinstance(seq_len, _SeqSymbol)
        seq_static = seq_len.build_value if dynamic_seq else seq_len
        nid = self._add(
            OpType.QK_RMS_NORM_ROPE,
            [query, key, query_weight, key_weight, cos, sin],
            (query_dim, seq_static, query_heads + key_heads),
            prec=self._nodes[query].out_prec,
            i32=[rope_dim, 1 if interleave else 0, query_heads],
            f32=[eps])
        if dynamic_seq:
            self._nodes[nid].dim_expr = (
                DimExpr.const(), DimExpr.seq(),
                DimExpr.const(), DimExpr.const())
        return nid

    def layer_norm(self, x: int, weight: int, bias: int,
                   eps: float = 1e-5) -> int:
        return self._add(OpType.LAYER_NORM, [x, weight, bias],
                         self._nodes[x].out_shape,
                         prec=self._nodes[x].out_prec, f32=[eps])

    # ---- activations ----

    def silu(self, x: int) -> int:
        return self._add(OpType.SILU, [x], self._nodes[x].out_shape,
                         prec=self._nodes[x].out_prec)

    def gelu(self, x: int) -> int:
        return self._add(OpType.GELU, [x], self._nodes[x].out_shape,
                         prec=self._nodes[x].out_prec)

    def tanh(self, x: int) -> int:
        return self._add(OpType.TANH, [x], self._nodes[x].out_shape,
                         prec=self._nodes[x].out_prec)

    def sigmoid(self, x: int) -> int:
        return self._add(OpType.SIGMOID, [x], self._nodes[x].out_shape,
                         prec=self._nodes[x].out_prec)

    def sigmoid_exact(self, x: int) -> int:
        """IEEE sigmoid for recurrent paths where approximation error compounds."""
        return self._add(OpType.SIGMOID_EXACT, [x], self._nodes[x].out_shape,
                         prec=self._nodes[x].out_prec)

    def exp(self, x: int) -> int:
        return self._add(OpType.EXP, [x], self._nodes[x].out_shape,
                         prec=self._nodes[x].out_prec)

    def exp_exact(self, x: int) -> int:
        """IEEE exp for recurrent paths where approximation error compounds."""
        return self._add(OpType.EXP_EXACT, [x], self._nodes[x].out_shape,
                         prec=self._nodes[x].out_prec)

    def softplus(self, x: int) -> int:
        return self._add(OpType.SOFTPLUS, [x], self._nodes[x].out_shape,
                         prec=self._nodes[x].out_prec)

    def swiglu(self, merged: int) -> int:
        """Fused SwiGLU over a merged [2I, ...] tensor: out = silu(gate) * up,
        where gate = merged[0:I] and up = merged[I:2I] along dim 0. Output dim0 = I.
        The op reads both halves from the single merged buffer (with merged's dim1
        row stride), so it does NOT rely on stride-aware slice views."""
        sx = list(self._nodes[merged].out_shape)
        assert sx[0] % 2 == 0, f"swiglu merged dim0 must be even, got {sx[0]}"
        out = tuple([sx[0] // 2] + sx[1:])
        return self._add(OpType.SWIGLU, [merged], out,
                         prec=self._nodes[merged].out_prec)

    def sigmoid_mul(self, value: int, gate: int) -> int:
        """Fused value * sigmoid(gate) with matching tensor shapes."""
        sv = self._nodes[value].out_shape
        sg = self._nodes[gate].out_shape
        assert sv == sg, f"sigmoid_mul shape mismatch: {sv} vs {sg}"
        return self._add(
            OpType.SIGMOID_MUL, [value, gate], sv,
            prec=self._nodes[value].out_prec)

    # ---- position encoding ----

    def rope(self, x: int, cos: int, sin: int,
             rope_dim: int = 64, interleave: bool = True) -> int:
        return self._add(OpType.ROTARY_EMBED, [x, cos, sin],
                         self._nodes[x].out_shape,
                         prec=self._nodes[x].out_prec,
                         i32=[rope_dim, 1 if interleave else 0])

    # ---- attention ----

    def sdpa(self, q: int, k_cur: int, v_cur: int, mask: int,
             k_cache: int, v_cache: int,
             kv_cache: int = 2, causal: bool = True,
             scale: float = 0.0,
             num_heads: int = 16, num_kv_heads: int = 16,
             head_dim: int = 192, v_head_dim: int = 128
             ) -> tuple[int, int, int]:
        sq = self._nodes[q].out_shape
        out_shape = (v_head_dim, sq[1], num_heads)
        attn = self._add(OpType.SDPA, [q, k_cur, v_cur, mask, k_cache, v_cache],
                         out_shape, prec=self._nodes[q].out_prec,
                         i32=[kv_cache, 1 if causal else 0,
                              num_heads, num_kv_heads, head_dim, v_head_dim],
                         f32=[scale])
        # cache outputs are views of the same cache buffers
        kc_out = self._add(OpType.RESHAPE, [k_cache],
                           self._nodes[k_cache].out_shape,
                           prec=self._nodes[k_cache].out_prec,
                           i32=list(self._nodes[k_cache].out_shape))
        vc_out = self._add(OpType.RESHAPE, [v_cache],
                           self._nodes[v_cache].out_shape,
                           prec=self._nodes[v_cache].out_prec,
                           i32=list(self._nodes[v_cache].out_shape))
        return attn, kc_out, vc_out

    def sdpa_no_cache(self, q: int, k: int, v: int,
                      causal: bool = False, scale: float = 0.0,
                      num_heads: int = 16, num_kv_heads: int = 16,
                      head_dim: int = 64, v_head_dim: int = 64) -> int:
        """Self-attention without persistent KV state, used by vision graphs."""
        sq = self._nodes[q].out_shape
        return self._add(
            OpType.SDPA, [q, k, v],
            (v_head_dim, sq[1], num_heads),
            prec=self._nodes[q].out_prec,
            i32=[0, 1 if causal else 0, num_heads, num_kv_heads,
                 head_dim, v_head_dim],
            f32=[scale])

    # ---- shape ops (zero-copy views) ----

    def reshape(self, x: int, shape: tuple) -> int:
        """Reshape input to target shape.

        Args:
            x: input node id.
            shape: target shape (4-D, will be normalized). Use -1 to infer
                   one dim from total element count. Use the module-level
                   SEQ symbol to mark runtime-dynamic dims:
                     g.reshape(x, (head_dim, num_heads, SEQ))      # dim 2 = SEQ
                     g.reshape(x, (head_dim, num_heads * SEQ))     # dim 1 = MUL
                   The C++ runtime evaluates these DimExprs against
                   runtime_seq_len via eval_dim().
        """
        sx = self._nodes[x].out_shape
        n_elems = 1
        for d in sx: n_elems *= d
        # handle -1 (infer) dimension
        resolved = list(shape)
        if -1 in resolved:
            known = 1
            for d in resolved:
                if d != -1 and not isinstance(d, (_SeqSymbol, DimExpr)):
                    known *= d
            idx = resolved.index(-1)
            resolved[idx] = n_elems // known
        # separate static dims from symbolic ones
        static_dims = []
        dim_exprs = [DimExpr.const()] * 4
        for d, val in enumerate(resolved):
            if isinstance(val, _SeqSymbol):
                static_dims.append(val.build_value)  # build-time value (for serialization)
                dim_exprs[d] = DimExpr.seq()
            elif isinstance(val, _DimExprSymbol):
                static_dims.append(val.build_value)
                dim_exprs[d] = val.expr
            elif isinstance(val, DimExpr):
                static_dims.append(0)  # no build-value available
                dim_exprs[d] = val
            else:
                static_dims.append(int(val))
        s_elems = 1
        for d in static_dims:
            if d > 0: s_elems *= d
        # For element count check, treat dynamic dims as their build-time
        # values (we don't have runtime values at transpile time).
        # Skip the strict element-mismatch assert when shape has SEQ symbols.
        has_symbolic = any(isinstance(v, (_SeqSymbol, DimExpr)) for v in resolved)
        if not has_symbolic:
            assert n_elems == s_elems, f"reshape element mismatch: {sx} vs {shape}"
        resolved_4d = list(self._normalize_shape(tuple(static_dims)))
        dynamic_dim = idx if -1 in list(shape) else -1
        i32_params = list(resolved_4d)
        i32_params.append(dynamic_dim)  # params_i32[4] = dynamic dim index
        nid = self._add(OpType.RESHAPE, [x], tuple(resolved_4d),
                        prec=self._nodes[x].out_prec,
                        i32=i32_params)

        # Apply symbolic dim_exprs. If caller used SEQ symbol, those are set
        # above. Otherwise check -1 path: if source has SEQ, mark inferred dim.
        node = self._nodes[nid]
        out_de = list(node.dim_expr)
        for d in range(4):
            if dim_exprs[d].kind != DimKind.CONST:
                out_de[d] = dim_exprs[d]
        # -1 path: only auto-mark if no explicit SEQ was given AND source has SEQ
        if dynamic_dim >= 0 and dynamic_dim < 4 and not has_symbolic:
            src_de = self._nodes[x].dim_expr
            if any(e.kind != DimKind.CONST for e in src_de):
                # inherit the first non-CONST expr from source
                for e in src_de:
                    if e.kind != DimKind.CONST:
                        out_de[dynamic_dim] = e
                        break
        node.dim_expr = tuple(out_de)
        return nid

    def permute(self, x: int, order: tuple) -> int:
        sx = self._nodes[x].out_shape
        new_shape = tuple(sx[order[i]] for i in range(4))
        return self._add(OpType.PERMUTE, [x], new_shape,
                         prec=self._nodes[x].out_prec,
                         i32=list(order))

    def concat(self, ids: list[int], dim: int) -> int:
        shapes = [self._nodes[i].out_shape for i in ids]
        concat_dim = sum(s[dim] for s in shapes)
        out = list(shapes[0])
        out[dim] = concat_dim
        return self._add(OpType.CONCAT, ids, tuple(out),
                         prec=self._nodes[ids[0]].out_prec,
                         i32=[dim])

    def slice(self, x: int, sizes: list[int], dim: int) -> list[int]:
        sx = self._nodes[x].out_shape
        results = []
        offset = 0
        for sz in sizes:
            out = list(sx)
            out[dim] = sz
            results.append(self._add(OpType.SLICE, [x], tuple(out),
                                     prec=self._nodes[x].out_prec,
                                     i32=[dim, offset, sz]))
            offset += sz
        return results

    def slice_range(self, x: int, offset: int, size: int, dim: int) -> int:
        """Create one slice without materializing unused sibling slices."""
        sx = self._nodes[x].out_shape
        if dim < 0 or dim >= len(sx):
            raise ValueError(f"slice dimension {dim} is out of range")
        if offset < 0 or size < 0 or offset + size > sx[dim]:
            raise ValueError(
                f"slice [{offset}, {offset + size}) exceeds dimension "
                f"{dim} of size {sx[dim]}")
        out = list(sx)
        out[dim] = size
        return self._add(OpType.SLICE, [x], tuple(out),
                         prec=self._nodes[x].out_prec,
                         i32=[dim, offset, size])

    def tile(self, x: int, repeats: tuple) -> int:
        sx = self._nodes[x].out_shape
        r = list(repeats) + [1] * (4 - len(repeats))
        out = tuple(sx[i] * r[i] for i in range(4))
        return self._add(OpType.TILE, [x], out,
                         prec=self._nodes[x].out_prec,
                         i32=r)

    def contiguous(self, x: int) -> int:
        """Materialize to a row-major contiguous buffer.
        
        Copies the input tensor in stride order into a new buffer whose
        physical layout matches the declared shape.  Essential after
        zero-copy permute/reshape when downstream kernels (e.g. SDPA)
        read via channel() + raw pointer arithmetic.
        
        Returns a new node with the same shape as x, backed by a
        contiguous buffer.
        """
        sx = self._nodes[x].out_shape
        return self._add(OpType.CONTIGUOUS, [x], sx,
                         prec=self._nodes[x].out_prec)

    # ---- element-wise ----

    def add(self, a: int, b: int) -> int:
        sa = self._nodes[a].out_shape
        sb = self._nodes[b].out_shape
        out = tuple(max(sa[i], sb[i]) for i in range(4))
        return self._add(OpType.ADD, [a, b], out,
                         prec=self._nodes[a].out_prec)

    def mul(self, a: int, b: int) -> int:
        sa = self._nodes[a].out_shape
        sb = self._nodes[b].out_shape
        out = tuple(max(sa[i], sb[i]) for i in range(4))
        return self._add(OpType.MUL, [a, b], out,
                         prec=self._nodes[a].out_prec)

    def scalar_mul(self, a: int, scalar: float) -> int:
        """Multiply tensor by a scalar (creates a 1-element CONSTANT node)."""
        scalar_node = self.constant(np.array([scalar], dtype=np.float32))
        return self.mul(a, scalar_node)

    def scalar_add(self, a: int, scalar: float) -> int:
        """Add a scalar to a tensor (creates a 1-element CONSTANT node)."""
        scalar_node = self.constant(np.array([scalar], dtype=np.float32))
        return self.add(a, scalar_node)

    def shortconv(self, x: int, weight: int, conv_state: int,
                  kernel_size: int) -> int:
        """Depth-wise causal conv1d + silu.

        Args:
            x: input [groups, seq_len] FP32
            weight: [groups, kernel_size] FP32 (CONSTANT)
            conv_state: [groups, kernel_size-1] FP32 (persistent INPUT, in-place modified)
            kernel_size: conv kernel size (e.g. 4)

        Returns:
            output [groups, seq_len] FP32
        """
        sx = self._nodes[x].out_shape
        return self._add(OpType.SHORTCONV, [x, weight, conv_state], sx,
                         prec=self._nodes[x].out_prec,
                         i32=[kernel_size])

    def rwkv_token_shift(self, x: int, state: int,
                         hidden_size: int, seq_len: int) -> int:
        """Return previous_x - x and update the persistent previous_x."""
        return self._add(OpType.RWKV_TOKEN_SHIFT, [x, state],
                         self._nodes[x].out_shape, prec=Precision.FP32,
                         i32=[hidden_size, seq_len, 0])

    def rwkv_mix(self, x: int, shift: int, mix: int) -> int:
        """Fused RWKV time mix: x + shift * mix."""
        return self._add(OpType.RWKV_MIX, [x, shift, mix],
                         self._nodes[x].out_shape, prec=Precision.FP32)

    def rwkv_l2_norm(self, x: int, num_heads: int, head_size: int,
                     eps: float = 1e-12) -> int:
        return self._add(OpType.RWKV_L2_NORM, [x], self._nodes[x].out_shape,
                         prec=Precision.FP32, i32=[num_heads, head_size],
                         f32=[eps])

    def rwkv_post(self, raw: int, r: int, k: int, v: int, r_k: int,
                  weight: int, bias: int, gate: int,
                  num_heads: int, head_size: int,
                  eps: float = 64e-5) -> int:
        """Fused (group_norm(raw) + RWKV bonus(r,k,v)) * gate."""
        return self._add(OpType.RWKV_POST,
                         [raw, r, k, v, r_k, weight, bias, gate],
                         self._nodes[raw].out_shape, prec=Precision.FP32,
                         i32=[num_heads, head_size], f32=[eps])

    def rwkv7_core(self, r: int, decay: int, k: int, v: int,
                   a: int, b: int, state: int,
                   num_heads: int, head_size: int, seq_len: int) -> int:
        """RWKV-7 recurrence only, matching ggml_rwkv_wkv7 boundaries."""
        return self._add(OpType.RWKV7, [r, decay, k, v, a, b, state],
                         self._nodes[r].out_shape, prec=Precision.FP32,
                         i32=[num_heads, head_size, seq_len, 0, 0])

    def gated_deltanet(self, qkv_conv: int, a_out: int, b_out: int, z_out: int,
                       A_log: int, dt_bias: int, norm_weight: int, gdn_state: int,
                       num_heads: int, k_dim: int, v_dim: int, seq_len: int,
                       use_qk_l2norm: bool = True, rms_eps: float = 1e-6,
                       num_v_heads: int = 0,
                       output_gate_type: str = "silu") -> int:
        """Fused Gated Delta Rule linear-attention core for Qwen3.5.

        Replaces: split qkv, g/beta compute, GDN recurrence, RMSNormGated.

        All four matmul-derived inputs (qkv_conv/a_out/b_out/z_out) MUST be in
        their native [seq, dim] row-major data layout (the layout matmul writes
        C in). The builder emits a `reshape` materialize on qkv_conv (shortconv
        output, which is [groups, seq]) to bring it to [seq, qkv_total] so all
        four inputs share the same convention. The kernel indexes ptr[t*dim+d]
        directly and does NOT consult Tensor shape/stride.

        Args:
            qkv_conv:    shortconv output reshaped to [seq, qkv_total] data
                         (qkv_total = 3*num_heads*k_dim)
            a_out:       matmul x@w_a, data [seq, num_heads]
            b_out:       matmul x@w_b, data [seq, num_heads]
            z_out:       matmul x@w_z, data [seq, num_v_heads*v_dim]
            A_log:       CONSTANT [num_heads] FP32
            dt_bias:     CONSTANT [num_heads] FP32
            norm_weight: CONSTANT [v_dim] FP32 (RMSNormGated gamma)
            gdn_state:   INPUT [num_heads, k_dim, v_dim] FP32, in-place modified
            seq_len:     1 for decode, N for prefill
            num_v_heads: number of value heads (defaults to num_heads)

        Returns:
            output node, declared shape [num_v_heads*v_dim, seq], data
            [seq, num_v_heads*v_dim] row-major (ready for out_proj matmul).
        """
        if num_v_heads <= 0:
            num_v_heads = num_heads
        if output_gate_type not in ("silu", "sigmoid"):
            raise ValueError(f"unsupported GDN output gate: {output_gate_type}")
        op = (OpType.GATED_DELTANET_DECODE if seq_len == 1
              else OpType.GATED_DELTANET_PREFILL)
        scale = float(k_dim ** -0.5)
        return self._add(op,
                         [qkv_conv, a_out, b_out, z_out,
                          A_log, dt_bias, norm_weight, gdn_state],
                         (num_v_heads * v_dim, seq_len),
                         prec=Precision.FP32,
                         i32=[num_heads, k_dim, v_dim, seq_len,
                              (1 if use_qk_l2norm else 0) |
                              (2 if output_gate_type == "sigmoid" else 0),
                              4, 0, num_v_heads],
                         f32=[rms_eps, 1e-6, scale])

    def gated_deltanet_conv_decode(
            self, qkv: int, a_out: int, b_out: int, z_out: int,
            A_log: int, dt_bias: int, norm_weight: int, gdn_state: int,
            conv_weight: int, conv_state: int,
            num_heads: int, k_dim: int, v_dim: int,
            num_v_heads: int = 0, conv_kernel: int = 4,
            use_qk_l2norm: bool = True, rms_eps: float = 1e-6,
            output_gate_type: str = "silu") -> int:
        """Decode-only ShortConv + Gated DeltaNet + RMSNormGated fusion."""
        if num_v_heads <= 0:
            num_v_heads = num_heads
        if output_gate_type not in ("silu", "sigmoid"):
            raise ValueError(f"unsupported GDN output gate: {output_gate_type}")
        scale = float(k_dim ** -0.5)
        return self._add(
            OpType.GATED_DELTANET_CONV_DECODE,
            [qkv, a_out, b_out, z_out, A_log, dt_bias, norm_weight,
             gdn_state, conv_weight, conv_state],
            (num_v_heads * v_dim, 1),
            prec=Precision.FP32,
            i32=[num_heads, k_dim, v_dim, 1,
                 (1 if use_qk_l2norm else 0) |
                 (2 if output_gate_type == "sigmoid" else 0),
                 conv_kernel, 0, num_v_heads],
            f32=[rms_eps, 1e-6, scale])

    def gated_deltanet_conv_verify(
            self, qkv: int, a_out: int, b_out: int, z_out: int,
            A_log: int, dt_bias: int, norm_weight: int, gdn_state: int,
            conv_weight: int, conv_state: int, gdn_checkpoint: int,
            conv_checkpoint: int, num_heads: int, k_dim: int, v_dim: int,
            seq_len: int, conv_kernel: int, use_qk_l2norm: bool = True,
            rms_eps: float = 1e-6, num_v_heads: int | None = None) -> int:
        """Verify a short sequence while checkpointing the confirmed prefix."""
        if num_v_heads is None:
            num_v_heads = num_heads
        scale = float(k_dim ** -0.5)
        return self._add(
            OpType.GATED_DELTANET_CONV_VERIFY,
            [qkv, a_out, b_out, z_out, A_log, dt_bias, norm_weight,
             gdn_state, conv_weight, conv_state, gdn_checkpoint,
             conv_checkpoint],
            (num_v_heads * v_dim, seq_len),
            prec=Precision.FP32,
            i32=[num_heads, k_dim, v_dim, seq_len,
                 1 if use_qk_l2norm else 0, conv_kernel, 0, num_v_heads, 1],
            f32=[rms_eps, 1e-6, scale])

    def moe(self, hidden: int, router: int,
            experts_gate_up: int, experts_down: int,
            shared_gate: int, shared_up: int, shared_down: int,
            shared_expert_gate: int | None,
            hidden_size: int, num_experts: int, top_k: int,
            intermediate_size: int, shared_intermediate_size: int,
            router_bias: int | None = None,
            router_score_func: int = 0,
            norm_topk_prob: bool = True,
            has_shared_expert: bool = True,
            shared_expert_has_gate: bool = True,
            n_group: int = 1,
            topk_group: int = 1,
            routed_scaling_factor: float = 1.0,
            swiglu_limit: float = 0.0,
            hash_token_ids: int | None = None,
            hash_table: int | None = None) -> int:
        """Fused Qwen-style sparse MLP.

        The expert tensors keep HF row-major layout:
          experts_gate_up [num_experts, 2*intermediate, hidden]
          experts_down    [num_experts, hidden, intermediate]
        """
        sx = self._nodes[hidden].out_shape
        inputs = [hidden, router, experts_gate_up, experts_down]
        if has_shared_expert:
            inputs.extend([shared_gate, shared_up, shared_down])
            if shared_expert_has_gate:
                if shared_expert_gate is None:
                    raise ValueError(
                        "gated shared expert requires a gate weight")
                inputs.append(shared_expert_gate)
        router_bias_input = -1
        if router_bias is not None:
            router_bias_input = len(inputs)
            inputs.append(router_bias)
        token_ids_input = -1
        hash_table_input = -1
        if hash_token_ids is not None or hash_table is not None:
            if hash_token_ids is None or hash_table is None:
                raise ValueError(
                    "hash routing requires token ids and lookup table")
            token_ids_input = len(inputs)
            inputs.append(hash_token_ids)
            hash_table_input = len(inputs)
            inputs.append(hash_table)
        return self._add(OpType.MOE,
                         inputs,
                         sx,
                         prec=Precision.FP32,
                         i32=[hidden_size, num_experts, top_k,
                              intermediate_size, shared_intermediate_size,
                              router_score_func,
                              1 if norm_topk_prob else 0,
                              1 if has_shared_expert else 0,
                              n_group, topk_group,
                              1 if shared_expert_has_gate else 0,
                              router_bias_input, token_ids_input,
                              hash_table_input],
                         f32=[float(routed_scaling_factor),
                              float(swiglu_limit)])

    def hc_pre(self, x: int, fn: int, scale: int, base: int,
               hidden_size: int, hc_mult: int = 4,
               sinkhorn_iters: int = 20, norm_eps: float = 1e-6,
               sinkhorn_eps: float = 1e-6) -> int:
        sx = self._nodes[x].out_shape
        packed_size = hidden_size + hc_mult + hc_mult * hc_mult
        return self._add(
            OpType.HC_PRE, [x, fn, scale, base],
            (packed_size, sx[1]), prec=Precision.FP32,
            i32=[hidden_size, hc_mult, sinkhorn_iters],
            f32=[norm_eps, sinkhorn_eps])

    def hc_post(self, branch: int, residual: int, packed: int,
                hidden_size: int, hc_mult: int = 4) -> int:
        sr = self._nodes[residual].out_shape
        return self._add(
            OpType.HC_POST, [branch, residual, packed], sr,
            prec=Precision.FP32, i32=[hidden_size, hc_mult])

    def hc_head(self, x: int, fn: int, scale: int, base: int,
                hidden_size: int, hc_mult: int = 4,
                norm_eps: float = 1e-6, hc_eps: float = 1e-6) -> int:
        sx = self._nodes[x].out_shape
        return self._add(
            OpType.HC_HEAD, [x, fn, scale, base],
            (hidden_size, sx[1]), prec=Precision.FP32,
            i32=[hidden_size, hc_mult], f32=[norm_eps, hc_eps])

    def gr_reduce(self, normalized: int, gates: int,
                  hidden_size: int, hc_count: int = 4) -> int:
        """Qwen Gated Residual read: mean(gate * stream) over streams."""
        sx = self._nodes[normalized].out_shape
        sg = self._nodes[gates].out_shape
        if sx != sg or sx[0] != hidden_size * hc_count:
            raise ValueError(
                f"invalid GR reduce shapes {sx}, {sg} for "
                f"hidden={hidden_size}, streams={hc_count}")
        return self._add(
            OpType.GR_REDUCE, [normalized, gates],
            (hidden_size, sx[1]), prec=self._nodes[normalized].out_prec,
            i32=[hidden_size, hc_count])

    def gr_inject(self, branch: int, residual: int, gates: int,
                  hidden_size: int, hc_count: int = 4) -> int:
        """Qwen Gated Residual write: residual_h + gate_h * branch."""
        sb = self._nodes[branch].out_shape
        sr = self._nodes[residual].out_shape
        sg = self._nodes[gates].out_shape
        if (sb[0] != hidden_size or sr[0] != hidden_size * hc_count or
                sg[0] != hc_count or sb[1] != sr[1] or sg[1] != sr[1]):
            raise ValueError(
                f"invalid GR inject shapes branch={sb}, residual={sr}, "
                f"gates={sg}")
        return self._add(
            OpType.GR_INJECT, [branch, residual, gates], sr,
            prec=self._nodes[residual].out_prec,
            i32=[hidden_size, hc_count])

    def ple_lookup(self, token_ids: int, history: int, table: int,
                   table_scale: int, head_vocab_sizes: int,
                   head_offsets: int, embedding_dim: int,
                   ngram_size: int, heads_per_ngram: int,
                   eos_token_id: int, unigram_vocab_size: int,
                   ple_layer_index: int = 0, seed: int = 1234) -> int:
        """Hash token n-grams and gather their raw FP8 embedding rows."""
        num_heads = (ngram_size - 1) * heads_per_ngram
        if num_heads <= 0 or embedding_dim % num_heads != 0:
            raise ValueError("PLE embedding dim must divide n-gram heads")
        seq_len = self._nodes[token_ids].out_shape[0]
        nid = self._add(
            OpType.PLE_LOOKUP,
            [token_ids, history, table, table_scale,
             head_vocab_sizes, head_offsets],
            (embedding_dim, seq_len), prec=Precision.FP32,
            i32=[ngram_size, heads_per_ngram, eos_token_id,
                 unigram_vocab_size, ple_layer_index,
                 int(seed & 0xffffffff), int((seed >> 32) & 0xffffffff),
                 seq_len])
        if self._nodes[token_ids].dim_expr[0].kind == DimKind.SEQ:
            self._nodes[nid].dim_expr = (
                DimExpr.const(), DimExpr.seq(),
                DimExpr.const(), DimExpr.const())
        return nid

    def ple_gate(self, query: int, key: int, value: int,
                 hidden_size: int, hc_count: int = 4) -> int:
        """Gate one PLE value into each residual stream."""
        sq, sk, sv = (self._nodes[node].out_shape
                      for node in (query, key, value))
        if (sq != sk or sq[0] != hidden_size * hc_count or
                sv[0] != hidden_size or sv[1] != sq[1]):
            raise ValueError(
                f"invalid PLE gate shapes query={sq}, key={sk}, value={sv}")
        return self._add(
            OpType.PLE_GATE, [query, key, value], sq,
            prec=self._nodes[query].out_prec,
            i32=[hidden_size, hc_count])

    def ple_dilated_conv(self, x: int, weight: int, state: int,
                         kernel_size: int, dilation: int) -> int:
        """Depthwise causal SiLU convolution used after PLE gating."""
        sx = self._nodes[x].out_shape
        return self._add(
            OpType.PLE_DILATED_CONV, [x, weight, state], sx,
            prec=self._nodes[x].out_prec,
            i32=[kernel_size, dilation, sx[1]])

    def dsv4_compressor(
            self, hidden: int, wkv: int, wgate: int, ape: int,
            norm: int, kv_state: int, score_state: int, cache: int,
            position: int, n_tokens: int, hidden_size: int, head_dim: int,
            ratio: int,
            overlap: bool, rotate: bool, rope_dim: int,
            original_context: int, norm_eps: float, rope_theta: float,
            rope_factor: float, beta_fast: float, beta_slow: float) -> int:
        """Update one DeepSeek-V4 learned-compression cache.

        The scalar output is only a dependency token. Persistent state and
        cache tensors are mutated in place.
        """
        return self._add(
            OpType.DSV4_COMPRESSOR,
            [hidden, wkv, wgate, ape, norm, kv_state, score_state,
             cache, position, n_tokens],
            (1,), prec=Precision.FP32,
            i32=[hidden_size, head_dim, ratio,
                 1 if overlap else 0, 1 if rotate else 0,
                 rope_dim, original_context],
            f32=[norm_eps, rope_theta, rope_factor, beta_fast, beta_slow])

    def dsv4_indexer(
            self, hidden: int, q_lora: int, wq_b: int,
            weights_projection: int, compressor_wkv: int,
            compressor_wgate: int, compressor_ape: int,
            compressor_norm: int, kv_state: int, score_state: int,
            cache: int, position: int, n_tokens: int, hidden_size: int,
            q_lora_rank: int, num_heads: int, head_dim: int, top_k: int,
            ratio: int,
            overlap: bool, rotate: bool, rope_dim: int,
            original_context: int, norm_eps: float, rope_theta: float,
            rope_factor: float, beta_fast: float, beta_slow: float) -> int:
        sequence = self._nodes[hidden].out_shape[1]
        return self._add(
            OpType.DSV4_INDEXER,
            [hidden, q_lora, wq_b, weights_projection,
             compressor_wkv, compressor_wgate, compressor_ape,
             compressor_norm, kv_state, score_state, cache, position,
             n_tokens],
            (top_k, sequence), prec=Precision.INT32,
            i32=[hidden_size, q_lora_rank, num_heads, head_dim, top_k,
                 ratio, 1 if overlap else 0, 1 if rotate else 0,
                 rope_dim, original_context],
            f32=[norm_eps, rope_theta, rope_factor, beta_fast, beta_slow])

    def dsv4_sparse_attention(
            self, query: int, current_kv: int, sink: int, window_cache: int,
            position: int, n_tokens: int, num_heads: int, head_dim: int,
            window_size: int, compress_ratio: int, compressed_top_k: int,
            rope_dim: int,
            original_context: int, softmax_scale: float,
            query_norm_eps: float, rope_theta: float, rope_factor: float,
            beta_fast: float, beta_slow: float,
            compressed_cache: int | None = None,
            compressed_indices: int | None = None,
            dependencies: Sequence[int] = ()) -> int:
        inputs = [query, current_kv, sink, window_cache, position, n_tokens]
        compressed_cache_input = -1
        compressed_indices_input = -1
        if compressed_cache is not None:
            compressed_cache_input = len(inputs)
            inputs.append(compressed_cache)
        if compressed_indices is not None:
            compressed_indices_input = len(inputs)
            inputs.append(compressed_indices)
        inputs.extend(dependencies)
        return self._add(
            OpType.DSV4_SPARSE_ATTN, inputs,
            self._nodes[query].out_shape, prec=Precision.FP32,
            i32=[num_heads, head_dim, window_size, compress_ratio,
                 compressed_top_k, rope_dim, original_context,
                 compressed_cache_input, compressed_indices_input],
            f32=[softmax_scale, query_norm_eps, rope_theta, rope_factor,
                 beta_fast, beta_slow])

    def dsv4_grouped_linear(self, x: int, weight: int,
                            groups: int) -> int:
        sx = self._nodes[x].out_shape
        sw = self._nodes[weight].out_shape
        if groups <= 0 or sw[0] % groups != 0:
            raise ValueError("invalid DeepSeek-V4 grouped projection shape")
        return self._add(
            OpType.DSV4_GROUPED_LINEAR, [x, weight],
            (sw[0], sx[1]), prec=Precision.FP32, i32=[groups])

    # ---- save ----

    def save(self, path_prefix: str):
        """Write <path_prefix>.graph and any referenced .weights files."""
        if path_prefix.endswith('.graph'):
            path_prefix = path_prefix[:-6]

        # Symbolic shape propagation: fill dynamic[] on all non-INPUT nodes.
        propagate_dynamic_shapes(self._nodes)

        # compute liveness
        use_count = [0] * len(self._nodes)
        for node in self._nodes:
            for inp in node.inputs:
                use_count[inp] += 1

        last_use = [-1] * len(self._nodes)
        for j, node in enumerate(self._nodes):
            for inp in node.inputs:
                last_use[inp] = j

        # identify graph inputs/outputs
        graph_inputs = []
        graph_outputs = []
        for node in self._nodes:
            if node.op_type == OpType.INPUT:
                graph_inputs.append(node.id)
        # outputs = nodes not consumed by any later node
        for i, node in enumerate(self._nodes):
            if use_count[i] == 0 and node.op_type not in (OpType.INPUT, OpType.CONSTANT):
                graph_outputs.append(node.id)

        # Fix inline constant paths before serialization.
        # Inline constants have weight_data set but need their params_str
        # updated to point at the const file that will be written below.
        for node in self._nodes:
            if node.weight_data is not None:
                const_fname = f"{os.path.basename(path_prefix)}_const_{node.id}.weights"
                node.params_str = [const_fname]

        # write graph binary
        graph_path = f"{path_prefix}.graph"
        with open(graph_path, 'wb') as f:
            # header
            f.write(struct.pack('<I', GRAPH_MAGIC))
            f.write(struct.pack('<I', GRAPH_VERSION))
            f.write(struct.pack('<I', len(self._nodes)))

            # metadata (key=value string pairs)
            f.write(struct.pack('<I', len(self.metadata)))
            for key, val in self.metadata.items():
                kb = key.encode('utf-8')
                vb = val.encode('utf-8')
                f.write(struct.pack('<I', len(kb)))
                f.write(kb)
                f.write(struct.pack('<I', len(vb)))
                f.write(vb)
            f.write(struct.pack('<I', len(graph_inputs)))
            for gid in graph_inputs:
                f.write(struct.pack('<I', gid))
            f.write(struct.pack('<I', len(graph_outputs)))
            for gid in graph_outputs:
                f.write(struct.pack('<I', gid))

            # nodes
            for node in self._nodes:
                f.write(struct.pack('<I', node.id))
                f.write(struct.pack('<I', int(node.op_type)))
                f.write(struct.pack('<I', len(node.inputs)))
                for inp in node.inputs:
                    f.write(struct.pack('<I', inp))
                shape = self._normalize_shape(node.out_shape)
                for d in shape:
                    f.write(struct.pack('<q', d))
                # dim_expr[4] (4 × 8 bytes: kind + 3 pad + coeff int32) — v3+
                de = list(node.dim_expr)
                while len(de) < 4:
                    de.append(DimExpr.const())
                for d in range(4):
                    e = de[d]
                    f.write(struct.pack('<bxxx i', int(e.kind), int(e.coeff)))
                f.write(struct.pack('<I', int(node.out_prec)))
                # i32 params
                f.write(struct.pack('<I', len(node.params_i32)))
                for v in node.params_i32:
                    f.write(struct.pack('<i', v))
                # f32 params
                f.write(struct.pack('<I', len(node.params_f32)))
                for v in node.params_f32:
                    f.write(struct.pack('<f', v))
                # str params
                f.write(struct.pack('<I', len(node.params_str)))
                for s in node.params_str:
                    data = s.encode('utf-8')
                    f.write(struct.pack('<I', len(data)))
                    f.write(data)

        print(f"Saved {graph_path} ({len(self._nodes)} nodes, "
              f"{len(graph_inputs)} inputs, {len(graph_outputs)} outputs)")

        # write weight files for inline constants
        for node in self._nodes:
            if node.weight_data is not None:
                const_fname = f"{os.path.basename(path_prefix)}_const_{node.id}.weights"
                wpath = os.path.join(os.path.dirname(path_prefix) or '.', const_fname)
                _write_weight_file(wpath, node.weight_data)


def _numpy_to_precision(dt: np.dtype) -> Precision:
    if dt == np.float32:
        return Precision.FP32
    elif dt == np.float16:
        return Precision.FP16
    elif dt == np.int8:
        return Precision.INT8
    elif dt == np.uint16:
        return Precision.FP16  # BF16 stored as uint16, treat as FP16 for now
    raise ValueError(f"unsupported dtype: {dt}")


def quantize_weight_w8_group(data: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Symmetric int8 quantization over K-dim groups for [N, K] weights."""
    if data.ndim != 2:
        raise ValueError(f"W8 quant expects 2-D [N,K] weight, got {data.shape}")
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")

    w = data.astype(np.float32, copy=False)
    n, k = w.shape
    groups_per_row = (k + group_size - 1) // group_size
    q = np.empty((n, k), dtype=np.int8)
    scales = np.empty((n, groups_per_row), dtype=np.float32)

    for row in range(n):
        for g in range(groups_per_row):
            begin = g * group_size
            end = min(begin + group_size, k)
            block = w[row, begin:end]
            max_abs = float(np.max(np.abs(block))) if block.size else 0.0
            scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
            scales[row, g] = scale
            q[row, begin:end] = np.clip(np.rint(block / scale), -127, 127).astype(np.int8)

    return q, scales.reshape(-1), group_size, n * groups_per_row


def _find_cpp_quant_helper() -> str | None:
    """Find optional C++ tensor quantizer built by CMake."""
    global _CPP_QUANT_HELPER
    if _CPP_QUANT_HELPER is not None:
        return _CPP_QUANT_HELPER if isinstance(_CPP_QUANT_HELPER, str) else None
    if os.environ.get("MOLLM_DISABLE_CPP_QUANT") == "1":
        _CPP_QUANT_HELPER = False
        return None

    candidates = []
    env = os.environ.get("MOLLM_QUANT_HELPER")
    if env:
        path = Path(env)
        if path.is_file() and os.access(path, os.X_OK):
            _CPP_QUANT_HELPER = str(path)
            return str(path)
    root = Path(__file__).resolve().parent.parent
    exe = "mollm-quantize"
    old_exe = "mollm_quantize_weight"
    candidates.extend([
        root / "build" / exe,
        root / "build_i8mm" / exe,
        root / "build" / "Release" / exe,
        root / "build_i8mm" / "Release" / exe,
        Path.cwd() / "build" / exe,
        Path.cwd() / "build_i8mm" / exe,
        Path.cwd() / exe,
        root / "build" / old_exe,
        root / "build_i8mm" / old_exe,
        root / "build" / "Release" / old_exe,
        root / "build_i8mm" / "Release" / old_exe,
        Path.cwd() / "build" / old_exe,
        Path.cwd() / "build_i8mm" / old_exe,
        Path.cwd() / old_exe,
    ])
    # Developer builds commonly use architecture- or experiment-specific
    # names such as build-x86-avx2. Keep explicit paths first, then discover
    # the same helper in other CMake build directories.
    candidates.extend(sorted(root.glob(f"build*/{exe}")))
    candidates.extend(sorted(root.glob(f"build*/Release/{exe}")))
    for path in candidates:
        if not path.is_file() or not os.access(path, os.X_OK):
            continue
        try:
            # A cross-compiled executable can still have the executable bit.
            # Starting it without arguments is a cheap, side-effect-free way
            # to reject binaries the current host cannot execute.
            subprocess.run(
                [str(path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        _CPP_QUANT_HELPER = str(path)
        return str(path)
    _CPP_QUANT_HELPER = False
    return None


def write_quantized_weight_file_cpp(path: str, data: np.ndarray,
                                    quant_kind: str, group_size: int,
                                    required: bool = False) -> bool:
    """Write one quantized 2-D weight file using the optional C++ helper.

    Returns False when the helper is unavailable and a Python fallback is
    acceptable. W4 callers should set required=True; Python W4 quantization is
    intentionally not supported because it is too slow for real models.
    """
    global _CPP_QUANT_HELPER_ANNOUNCED, _CPP_QUANT_HELPER_MISSING_ANNOUNCED
    helper = _find_cpp_quant_helper()
    if helper is None:
        if required or quant_kind == "w4":
            raise RuntimeError(
                "W4 quantization requires the C++ quantizer. "
                "Build it with: cmake --build build --target mollm-quantize"
            )
        if not _CPP_QUANT_HELPER_MISSING_ANNOUNCED:
            print("  C++ quantizer not found; falling back to Python W8 quantization")
            print("  Build it with: cmake --build build --target mollm-quantize")
            _CPP_QUANT_HELPER_MISSING_ANNOUNCED = True
        return False
    if data.ndim != 2:
        return False
    if quant_kind not in ("w8", "w4"):
        raise ValueError(f"unsupported C++ quant kind: {quant_kind}")

    arr = np.ascontiguousarray(data)
    if arr.dtype == np.float16:
        dtype = "f16"
    elif arr.dtype == np.float32:
        dtype = "f32"
    else:
        arr = arr.astype(np.float16)
        dtype = "f16"

    out_dir = os.path.dirname(path) or "."
    tmp = tempfile.NamedTemporaryFile(prefix="mollm_quant_", suffix=".raw",
                                      dir=out_dir, delete=False)
    tmp_path = tmp.name
    try:
        arr.tofile(tmp)
        tmp.close()
        cmd = [
            helper, tmp_path, path, dtype,
            str(arr.shape[0]), str(arr.shape[1]), quant_kind, str(group_size),
        ]
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.returncode != 0:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"C++ quantizer failed for {path}: {detail}")
        if quant_kind == "w4":
            try:
                _validate_package_weight_header(
                    path, _read_weight_header(path))
            except ValueError as error:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                raise RuntimeError(
                    "C++ quantizer emitted an obsolete W4 layout; rebuild "
                    f"mollm-quantize before conversion: {error}") from error
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass

    if not _CPP_QUANT_HELPER_ANNOUNCED:
        print(f"  Using C++ quantizer: {helper}")
        _CPP_QUANT_HELPER_ANNOUNCED = True
    prec = "INT8" if quant_kind == "w8" else "INT4"
    print(f"  Wrote {path} ({arr.shape}, {arr.dtype}, prec={prec}, group={group_size}, via=C++)")
    return True


def _weight_header_bytes(*, flags: int, ndim: int, precision: Precision,
                         shape: Sequence[int], data_size: int,
                         scales_size: int, group_size: int,
                         num_groups: int) -> bytes:
    padded_shape = list(shape) + [1] * (4 - len(shape))
    data_offset = 88
    scales_offset = data_offset + data_size if scales_size else 0
    header = struct.pack('<II', WEIGHT_MAGIC, flags)
    header += struct.pack('<II', ndim, int(precision))
    header += struct.pack('<qqqq', *padded_shape)
    header += struct.pack('<QQ', data_offset, data_size)
    header += struct.pack('<QQ', scales_offset, scales_size)
    header += struct.pack('<II', group_size, num_groups)
    assert len(header) == 88
    return header


@dataclass(frozen=True)
class WeightByteRange:
    """A byte range copied into a streamed XMAP weight.

    ``transform`` is intended for same-size chunk conversions such as BF16 to
    FP16. Native FP8 and MXFP4 tensors leave it unset and are copied verbatim.
    """
    path: str
    offset: int
    size: int
    transform: Optional[Callable[[bytes], bytes]] = None
    output_size: Optional[int] = None
    # Transformed chunks start and end on this byte boundary. This lets
    # layout transforms operate incrementally without materializing a large
    # source tensor. One means no additional alignment requirement.
    transform_alignment: int = 1

    @property
    def written_size(self) -> int:
        return self.size if self.output_size is None else self.output_size


@dataclass
class StreamedWeight:
    """An XMAP weight assembled directly from source-file byte ranges."""
    precision: Precision
    logical_shape: tuple[int, ...]
    data_ranges: Sequence[WeightByteRange]
    scale_ranges: Sequence[WeightByteRange] = ()
    group_size: int = 0
    num_groups: int = 0
    flags: int = 0
    expert_interleave_count: int = 0

    @property
    def data_size(self) -> int:
        return sum(segment.written_size for segment in self.data_ranges)

    @property
    def scales_size(self) -> int:
        return sum(segment.written_size for segment in self.scale_ranges)

    @property
    def size(self) -> int:
        return 88 + self.data_size + self.scales_size

    def header(self) -> dict:
        # Interleaving is only meaningful when data and an external scale
        # sidecar must alternate per expert. Canonical BG32/BG128 W4 blocks
        # embed their scales, so their expert-major data is already directly
        # consumable by both resident and SSD-backed runtimes.
        interleaved = (
            self.expert_interleave_count > 0 and self.scales_size > 0)
        data_size = self.data_size + self.scales_size if interleaved else self.data_size
        scales_size = 0 if interleaved else self.scales_size
        flags = self.flags | (
            WEIGHT_FLAG_EXPERT_INTERLEAVED if interleaved else 0)
        raw = _weight_header_bytes(
            flags=flags, ndim=len(self.logical_shape),
            precision=self.precision, shape=self.logical_shape,
            data_size=data_size, scales_size=scales_size,
            group_size=self.group_size, num_groups=self.num_groups)
        values = WEIGHT_HEADER_STRUCT.unpack(raw)
        result = {
            "flags": values[1], "ndim": values[2],
            "precision": values[3], "shape": list(values[4:8]),
            "data_offset": values[8], "data_size": values[9],
            "scales_offset": values[10], "scales_size": values[11],
            "group_size": values[12], "num_groups": values[13],
        }
        if interleaved:
            result["expert_interleave_count"] = self.expert_interleave_count
            result["expert_raw_data_size"] = self.data_size
            result["expert_raw_scales_size"] = self.scales_size
        return result

    @staticmethod
    def _copy_ranges(output: BinaryIO,
                     ranges: Sequence[WeightByteRange],
                     chunk_size: int = 8 * 1024 * 1024):
        for segment in ranges:
            alignment = segment.transform_alignment
            if alignment <= 0 or segment.size % alignment:
                raise ValueError(
                    f"invalid transform alignment for {segment.path}: "
                    f"{segment.size} bytes / {alignment}")
            remaining = segment.size
            written = 0
            with open(segment.path, "rb") as source:
                source.seek(segment.offset)
                while remaining:
                    take = min(chunk_size, remaining)
                    if remaining > take and alignment > 1:
                        take -= take % alignment
                    if take <= 0:
                        take = min(alignment, remaining)
                    chunk = source.read(take)
                    if not chunk:
                        raise EOFError(
                            f"short source range in {segment.path} at "
                            f"{segment.offset + segment.size - remaining}")
                    remaining -= len(chunk)
                    if segment.transform is not None:
                        chunk = segment.transform(chunk)
                    output.write(chunk)
                    written += len(chunk)
            if written != segment.written_size:
                raise ValueError(
                    f"transformed range size mismatch for {segment.path}: "
                    f"{written} != {segment.written_size}")

    def write_to(self, output: BinaryIO):
        header = self.header()
        output.write(_weight_header_bytes(
            flags=header["flags"], ndim=len(self.logical_shape),
            precision=self.precision, shape=self.logical_shape,
            data_size=header["data_size"], scales_size=header["scales_size"],
            group_size=self.group_size, num_groups=self.num_groups))
        interleaved = (
            self.expert_interleave_count > 0 and self.scales_size > 0)
        if not interleaved:
            self._copy_ranges(output, self.data_ranges)
            self._copy_ranges(output, self.scale_ranges)
            return

        count = self.expert_interleave_count
        if self.data_size % count or self.scales_size % count:
            raise ValueError("expert-interleaved weight is not expert-aligned")

        def split_ranges(ranges: Sequence[WeightByteRange],
                         bytes_per_expert: int):
            groups = []
            current = []
            current_bytes = 0
            for segment in ranges:
                segment_offset = 0
                remaining = segment.written_size
                while remaining:
                    take = min(remaining, bytes_per_expert - current_bytes)
                    if take != segment.written_size:
                        if (segment.transform is not None or
                                segment.written_size != segment.size):
                            raise ValueError(
                                "transformed range crosses an expert boundary")
                        piece = WeightByteRange(
                            segment.path, segment.offset + segment_offset,
                            take)
                    else:
                        piece = segment
                    current.append(piece)
                    current_bytes += take
                    segment_offset += take
                    remaining -= take
                    if current_bytes == bytes_per_expert:
                        groups.append(current)
                        current = []
                        current_bytes = 0
            if current or len(groups) != count:
                raise ValueError("expert ranges do not match expert count")
            return groups

        data_groups = split_ranges(self.data_ranges, self.data_size // count)
        scale_groups = split_ranges(self.scale_ranges, self.scales_size // count)
        for expert in range(count):
            self._copy_ranges(output, data_groups[expert])
            self._copy_ranges(output, scale_groups[expert])


def _write_weight_file(path: str, data: np.ndarray,
                       scales: np.ndarray | None = None,
                       group_size: int = 0,
                       num_groups: int = 0,
                       precision: Precision | None = None,
                       logical_shape: tuple[int, ...] | None = None,
                       flags: int = 0,
                       scale_dtype=np.float32):
    """Write a .weights file with a self-contained XMAP header.

    Quantized INT8/INT4 formats use the default float32 scales. Native
    FP8_E4M3 and MXFP4 checkpoints pass ``scale_dtype=np.uint8`` to preserve
    their encoded E8M0 scale bytes without a decode/re-encode round trip.
    """
    precision = _numpy_to_precision(data.dtype) if precision is None else precision
    data = np.ascontiguousarray(data)
    logical_shape = tuple(data.shape) if logical_shape is None else tuple(logical_shape)
    ndim = len(logical_shape)
    if ndim <= 0 or ndim > 4:
        raise ValueError(f"weights require 1-4 logical dims, got {logical_shape}")
    if precision == Precision.INT4:
        layout = flags & (
            WEIGHT_FLAG_INT4_BG32 | WEIGHT_FLAG_INT4_BG128)
        if layout not in (WEIGHT_FLAG_INT4_BG32, WEIGHT_FLAG_INT4_BG128):
            raise ValueError(
                "INT4 weights must use canonical BG32 or BG128 storage")
        expected_group = 32 if layout == WEIGHT_FLAG_INT4_BG32 else 128
        if scales is not None:
            raise ValueError(
                "canonical INT4 blocks embed scales; no sidecar is allowed")
        if (ndim != 2 or group_size != expected_group or
                logical_shape[1] % expected_group):
            raise ValueError(
                f"BG{expected_group} requires a 2-D weight, group="
                f"{expected_group}, and aligned K")
        rows, cols = logical_shape
        groups_per_row = cols // expected_group
        expected_groups = rows * groups_per_row
        block_bytes = 160 if expected_group == 32 else 544
        expected_bytes = (
            ((rows + 7) // 8) * groups_per_row * block_bytes)
        if num_groups != expected_groups or data.nbytes != expected_bytes:
            raise ValueError(
                f"invalid canonical BG{expected_group} payload: "
                f"groups={num_groups}/{expected_groups}, "
                f"bytes={data.nbytes}/{expected_bytes}")
    scales_bytes = b""
    scales_offset = 0
    scales_size = 0
    if scales is not None:
        scales = np.asarray(scales, dtype=scale_dtype).reshape(-1)
        if precision in (Precision.FP8_E4M3, Precision.MXFP4):
            if scales.dtype != np.uint8:
                raise ValueError(
                    f"{precision.name} requires raw uint8 E8M0 scales")
        elif scales.dtype != np.float32:
            raise ValueError(
                f"{precision.name} requires float32 quantization scales")
        scales_bytes = scales.tobytes()
        scales_offset = 88 + data.nbytes
        scales_size = len(scales_bytes)
        if group_size <= 0 or num_groups <= 0:
            raise ValueError("quantized weights require group_size and num_groups")
        if num_groups != scales.size:
            raise ValueError(f"num_groups={num_groups} does not match scales={scales.size}")

    data_size = data.nbytes
    header = _weight_header_bytes(
        flags=flags, ndim=ndim, precision=precision, shape=logical_shape,
        data_size=data_size, scales_size=scales_size,
        group_size=group_size, num_groups=num_groups)

    with open(path, 'wb') as f:
        f.write(header)
        f.write(data.tobytes())
        if scales_bytes:
            f.write(scales_bytes)
    qinfo = (f", group={group_size}, groups={num_groups}"
             if group_size else "")
    finfo = f", flags=0x{flags:x}" if flags else ""
    sinfo = f", logical={logical_shape}" if tuple(data.shape) != logical_shape else ""
    print(f"  Wrote {path} ({data.shape}, {data.dtype}, prec={precision.name}{sinfo}{qinfo}{finfo})")


# ---------------------------------------------------------------------------
# .mollm single-file package format
# ---------------------------------------------------------------------------
#
# Bundles prefill graph + decode graph + all weights + metadata into one file.
# The graphs are stored in the standard .graph format (unchanged), with
# CONSTANT nodes referencing weights by relative path (e.g. "./foo.weights").
# The C++ loader resolves these paths against the weights region of the mmap'd
# package instead of the filesystem.
#
# Layout:
#   [Header 128 bytes]
#     magic "MOLM" (4) + version (4)
#     metadata_offset (8) + metadata_len (8)
#     tokenizer_offset (8) + tokenizer_len (8)
#     jinja_offset (8) + jinja_len (8)
#     prefill_graph_offset (8) + prefill_graph_len (8)
#     decode_graph_offset (8) + decode_graph_len (8)
#     weights_offset (8) + weights_len (8)
#     vision_graph_offset (8) + vision_graph_len (8)
#     mtp_graph_len (8); MTP graph ends immediately before weights
#   [metadata JSON] — includes "weights" map: {filename: [offset, len]}
#   [tokenizer.json bytes]
#   [chat_template.jinja bytes]
#   [prefill graph bytes]   (standard .graph format, weight refs = "./foo.weights")
#   [decode graph bytes]    (standard .graph format, weight refs = "./foo.weights")
#   [vision graph bytes]    (optional)
#   [MTP graph bytes]       (optional; offset = weights_offset - mtp_graph_len)
#   [weights region]        — all .weights files concatenated, each self-contained

PACKAGE_MAGIC   = 0x4D4C4F4D  # "MOLM"
PACKAGE_VERSION = 1
PACKAGE_HEADER_SIZE = 128
WEIGHT_HEADER_STRUCT = struct.Struct("<IIIIQQQQQQQQII")


def _read_weight_header(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read(WEIGHT_HEADER_STRUCT.size)
    if len(raw) != WEIGHT_HEADER_STRUCT.size:
        raise ValueError(f"weight file too small: {path}")

    (magic, flags, ndim, precision,
     s0, s1, s2, s3,
     data_offset, data_size,
     scales_offset, scales_size,
     group_size, num_groups) = WEIGHT_HEADER_STRUCT.unpack(raw)
    if magic != WEIGHT_MAGIC:
        raise ValueError(f"bad weight magic in {path}: 0x{magic:08x}")
    return {
        "flags": flags,
        "ndim": ndim,
        "precision": precision,
        "shape": [s0, s1, s2, s3],
        "data_offset": data_offset,
        "data_size": data_size,
        "scales_offset": scales_offset,
        "scales_size": scales_size,
        "group_size": group_size,
        "num_groups": num_groups,
    }


def _validate_package_weight_header(ref: str, header: dict):
    """Enforce the canonical on-disk representation used by .mollm.

    Q4DOT remains an internal CPU packing primitive, but package INT4 tensors
    are self-contained BG32/BG128 blocks with embedded scales.  Rejecting a
    stale quantizer here prevents a backend from silently receiving a layout
    it cannot consume.
    """
    if int(header["precision"]) != int(Precision.INT4):
        return

    flags = int(header["flags"])
    if flags & WEIGHT_FLAG_INT4_Q4DOT_LEGACY:
        raise ValueError(
            f"legacy Q4DOT INT4 storage in {ref}; rebuild mollm-quantize "
            "and reconvert the model")
    layout = flags & (WEIGHT_FLAG_INT4_BG32 | WEIGHT_FLAG_INT4_BG128)
    if layout not in (WEIGHT_FLAG_INT4_BG32, WEIGHT_FLAG_INT4_BG128):
        raise ValueError(
            f"INT4 weight {ref} must use canonical BG32 or BG128 storage")
    unknown_flags = flags & ~(
        WEIGHT_FLAG_INT4_BG32 | WEIGHT_FLAG_INT4_BG128)
    if unknown_flags:
        raise ValueError(
            f"INT4 weight {ref} has unsupported storage flags "
            f"0x{unknown_flags:x}")

    rows, cols = map(int, header["shape"][:2])
    group_size = int(header["group_size"])
    expected_group = (
        32 if layout == WEIGHT_FLAG_INT4_BG32 else 128)
    if group_size != expected_group or cols % expected_group:
        raise ValueError(
            f"INT4 weight {ref} has incompatible shape/group for "
            f"BG{expected_group}: N={rows} K={cols} group={group_size}")
    if int(header["scales_size"]) != 0:
        raise ValueError(
            f"INT4 weight {ref} duplicates scales outside its canonical "
            f"BG{expected_group} blocks; rebuild mollm-quantize")

    groups_per_row = cols // expected_group
    expected_groups = rows * groups_per_row
    if int(header["num_groups"]) != expected_groups:
        raise ValueError(
            f"INT4 weight {ref} has {header['num_groups']} groups, "
            f"expected {expected_groups}")
    block_bytes = 160 if expected_group == 32 else 544
    expected_data_size = ((rows + 7) // 8) * groups_per_row * block_bytes
    if int(header["data_size"]) != expected_data_size:
        raise ValueError(
            f"INT4 weight {ref} has {header['data_size']} data bytes, "
            f"expected {expected_data_size} for canonical BG{expected_group}")


def _augment_moe_expert_storage(meta: dict,
                                weight_files: dict,
                                weight_paths: dict,
                                weight_headers: dict | None = None):
    """Resolve MoE expert aggregate tensors into package byte ranges.

    `qwen35_moe.py` emits logical per-layer split metadata. At package time we
    know each weight file's offset in the package weight region and can also
    parse the XMAP header, so fill in exact per-expert data/scales byte ranges.
    Existing runtimes ignore this object; SSD offload can use it later.
    """
    storage = meta.get("moe_expert_storage")
    if not isinstance(storage, dict):
        return
    layers = storage.get("layers")
    if not isinstance(layers, list):
        return

    num_experts = int(storage.get("num_experts") or meta.get("num_experts") or 0)
    if num_experts <= 0:
        return

    for layer in layers:
        if not isinstance(layer, dict):
            continue
        layer_num_experts = int(layer.get("num_experts") or num_experts)
        if layer_num_experts <= 0:
            continue
        for key in ("gate_up", "down"):
            spec = layer.get(key)
            if not isinstance(spec, dict):
                continue
            ref = spec.get("weight")
            if ref not in weight_files and ref and not ref.startswith("./"):
                ref = f"./{ref}"
            if (not ref or ref not in weight_files or
                    (ref not in weight_paths and
                     (weight_headers is None or ref not in weight_headers))):
                raise KeyError(f"MoE expert metadata references unknown weight: {spec.get('weight')}")

            header = (
                weight_headers[ref]
                if weight_headers is not None and ref in weight_headers
                else _read_weight_header(weight_paths[ref])
            )
            weight_offset, weight_size = weight_files[ref]
            rows_per_expert = int(spec["rows_per_expert"])
            expected_rows = layer_num_experts * rows_per_expert
            if int(header["shape"][0]) != expected_rows:
                raise ValueError(
                    f"MoE expert metadata row mismatch for {ref}: "
                    f"shape[0]={header['shape'][0]} expected={expected_rows}"
                )
            int4_layout = int(header["flags"]) & (
                WEIGHT_FLAG_INT4_BG32 | WEIGHT_FLAG_INT4_BG128)
            if (int(header["precision"]) == int(Precision.INT4) and
                    int4_layout and rows_per_expert % 8 != 0):
                raise ValueError(
                    f"canonical INT4 expert rows must be block-aligned for "
                    f"{ref}: rows_per_expert={rows_per_expert}")
            is_interleaved = bool(
                header["flags"] & WEIGHT_FLAG_EXPERT_INTERLEAVED)
            raw_data_size = int(
                header.get("expert_raw_data_size", header["data_size"]))
            raw_scales_size = int(
                header.get("expert_raw_scales_size", header["scales_size"]))
            if is_interleaved and int(
                    header.get("expert_interleave_count", 0)) != layer_num_experts:
                raise ValueError(
                    f"expert interleave count mismatch for {ref}")
            if raw_data_size % layer_num_experts != 0:
                raise ValueError(f"MoE expert data size is not expert-aligned: {ref}")
            if raw_scales_size and raw_scales_size % layer_num_experts != 0:
                raise ValueError(f"MoE expert scale size is not expert-aligned: {ref}")

            groups_per_row = 0
            if header["group_size"]:
                cols = int(spec.get("cols") or header["shape"][1])
                groups_per_row = (cols + int(header["group_size"]) - 1) // int(header["group_size"])

            spec["weight"] = ref
            spec["weight_offset"] = weight_offset
            spec["weight_size"] = weight_size
            spec["precision"] = header["precision"]
            spec["flags"] = header["flags"]
            spec["shape"] = header["shape"]
            data_per_expert = raw_data_size // layer_num_experts
            scales_per_expert = (
                raw_scales_size // layer_num_experts
                if raw_scales_size else 0)
            spec["data_offset"] = header["data_offset"]
            spec["data_size"] = raw_data_size
            spec["scales_offset"] = (
                header["data_offset"] + data_per_expert
                if is_interleaved and scales_per_expert
                else header["scales_offset"])
            spec["scales_size"] = raw_scales_size
            spec["group_size"] = header["group_size"]
            spec["groups_per_row"] = groups_per_row
            spec["expert_data_bytes"] = data_per_expert
            spec["expert_scales_bytes"] = scales_per_expert
            if is_interleaved:
                spec["expert_stride"] = data_per_expert + scales_per_expert


def save_package(output_path: str,
                 g_prefill: 'GraphBuilder',
                 g_decode: 'GraphBuilder',
                 weights_dir: str,
                 metadata: dict,
                 tokenizer_path: str = "",
                 jinja_path: str = "",
                 g_vision: 'GraphBuilder | None' = None,
                 g_mtp: 'GraphBuilder | None' = None,
                 g_mtp_verify: 'GraphBuilder | None' = None,
                 remove_weight_files: bool = False,
                 streamed_weights: dict[str, StreamedWeight] | None = None):
    """Pack model graphs + weights + tokenizer + jinja into one .mollm file.

    Graphs are saved via the standard save() format (to temp files), then
    their bytes are embedded in the package. Weight file paths in the graphs
    remain as relative paths (e.g. "./foo.weights"); the C++ loader resolves
    them against the package's weights region using the metadata offset map.
    """
    import json
    import tempfile
    import shutil

    tmp_dir = tempfile.mkdtemp(prefix="mollm_pkg_")

    try:
        # Step 1: save graphs to temp dir (standard format)
        g_prefill.save(os.path.join(tmp_dir, "model_prefill"))
        g_decode.save(os.path.join(tmp_dir, "model_decode"))
        if g_vision is not None:
            g_vision.save(os.path.join(tmp_dir, "model_vision"))
        if g_mtp_verify is not None:
            g_mtp_verify.save(os.path.join(tmp_dir, "model_mtp_verify"))
        if g_mtp is not None:
            g_mtp.save(os.path.join(tmp_dir, "model_mtp"))
        # Step 2: read graph bytes
        with open(os.path.join(tmp_dir, "model_prefill.graph"), 'rb') as f:
            pf_bytes = f.read()
        with open(os.path.join(tmp_dir, "model_decode.graph"), 'rb') as f:
            dc_bytes = f.read()
        vi_bytes = b""
        if g_vision is not None:
            with open(os.path.join(tmp_dir, "model_vision.graph"), 'rb') as f:
                vi_bytes = f.read()
        mtp_verify_bytes = b""
        if g_mtp_verify is not None:
            with open(os.path.join(tmp_dir, "model_mtp_verify.graph"), 'rb') as f:
                mtp_verify_bytes = f.read()
        mtp_bytes = b""
        if g_mtp is not None:
            with open(os.path.join(tmp_dir, "model_mtp.graph"), 'rb') as f:
                mtp_bytes = f.read()
        mtp_bundle_bytes = mtp_verify_bytes + mtp_bytes

        # Step 3: collect weight files referenced by both graphs
        weight_files = {}  # relative_name -> (offset, size)
        weight_paths = {}  # relative_name -> filesystem path used for packing
        weight_headers = {}  # headers for streamed entries
        # Align each embedded XMAP data payload, rather than its header, to a
        # cache line. Canonical BG32 is then both directly consumable by the
        # loader and aligned like the old heap sidecar used by its hot kernel.
        weight_entries = []  # (path-or-StreamedWeight, size, offset)
        weights_len = 0
        streamed_weights = streamed_weights or {}

        graphs = [g_prefill, g_decode]
        if g_vision is not None:
            graphs.append(g_vision)
        if g_mtp_verify is not None:
            graphs.append(g_mtp_verify)
        if g_mtp is not None:
            graphs.append(g_mtp)
        for g in graphs:
            for node in g._nodes:
                if node.op_type != OpType.CONSTANT or not node.params_str:
                    continue
                ref = node.params_str[0]
                if ref.startswith('#') or ref == "__inline_const__":
                    continue
                if ref in weight_files:
                    continue
                if ref in streamed_weights:
                    source = streamed_weights[ref]
                    header = source.header()
                    _validate_package_weight_header(ref, header)
                    size = source.size
                    data_offset = int(header["data_offset"])
                    offset = (
                        (weights_len + data_offset + 63) & ~63
                    ) - data_offset
                    weights_len = offset + size
                    weight_entries.append((source, size, offset))
                    weight_files[ref] = [offset, size]
                    weight_headers[ref] = header
                    continue
                # Find the weight file.
                wpath = os.path.join(weights_dir, ref) if not os.path.isabs(ref) else ref
                if not os.path.exists(wpath):
                    wpath = os.path.join(tmp_dir, ref)
                if not os.path.exists(wpath):
                    raise FileNotFoundError(f"Weight file not found: {ref}")
                header = _read_weight_header(wpath)
                _validate_package_weight_header(ref, header)
                size = os.path.getsize(wpath)
                data_offset = int(header["data_offset"])
                offset = (
                    (weights_len + data_offset + 63) & ~63
                ) - data_offset
                weights_len = offset + size
                weight_entries.append((wpath, size, offset))
                weight_files[ref] = [offset, size]
                weight_paths[ref] = wpath
                weight_headers[ref] = header

        meta = dict(metadata)
        meta["weights"] = weight_files
        meta["mtp_verify_graph_length"] = len(mtp_verify_bytes)
        if any(int(header["precision"]) == int(Precision.INT4)
               for header in weight_headers.values()):
            meta["int4_storage"] = "canonical_bg_block_v1"
        _augment_moe_expert_storage(
            meta, weight_files, weight_paths, weight_headers)
        meta_json = json.dumps(meta, ensure_ascii=False).encode('utf-8')

        # Step 5: read tokenizer + jinja bytes
        tok_bytes = b""
        if tokenizer_path and os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as tf:
                tok_bytes = tf.read()
        jinja_bytes = b""
        if jinja_path and os.path.exists(jinja_path):
            with open(jinja_path, 'rb') as jf:
                jinja_bytes = jf.read()
        elif tokenizer_path:
            # Some Hugging Face checkpoints store the template only inside
            # tokenizer_config.json. Preserve it in the package so the runtime
            # never has to infer formatting from an ambiguous special-token
            # vocabulary.
            tokenizer_config_path = os.path.join(
                os.path.dirname(tokenizer_path), "tokenizer_config.json")
            if os.path.exists(tokenizer_config_path):
                with open(tokenizer_config_path, encoding="utf-8") as cf:
                    tokenizer_config = json.load(cf)
                chat_template = tokenizer_config.get("chat_template", "")
                if isinstance(chat_template, dict):
                    chat_template = (
                        chat_template.get("default")
                        or next(iter(chat_template.values()), ""))
                elif isinstance(chat_template, list):
                    default = next(
                        (item.get("template", "")
                         for item in chat_template
                         if isinstance(item, dict)
                         and item.get("name") == "default"),
                        "")
                    chat_template = default or next(
                        (item.get("template", "")
                         for item in chat_template
                         if isinstance(item, dict)),
                        "")
                if isinstance(chat_template, str):
                    jinja_bytes = chat_template.encode("utf-8")

        # Step 6: write package
        hs = PACKAGE_HEADER_SIZE
        meta_off = hs
        tok_off = meta_off + len(meta_json)
        jin_off = tok_off + len(tok_bytes)
        pf_off = jin_off + len(jinja_bytes)
        dc_off = pf_off + len(pf_bytes)
        vi_off = dc_off + len(dc_bytes) if vi_bytes else 0
        mtp_off = dc_off + len(dc_bytes) + len(vi_bytes)
        weights_unaligned = mtp_off + len(mtp_bundle_bytes)
        w_off = (weights_unaligned + 63) & ~63
        with open(output_path, 'wb') as f:
            f.write(struct.pack('<II', PACKAGE_MAGIC, PACKAGE_VERSION))
            f.write(struct.pack('<QQ', meta_off, len(meta_json)))
            f.write(struct.pack('<QQ', tok_off, len(tok_bytes)))
            f.write(struct.pack('<QQ', jin_off, len(jinja_bytes)))
            f.write(struct.pack('<QQ', pf_off, len(pf_bytes)))
            f.write(struct.pack('<QQ', dc_off, len(dc_bytes)))
            f.write(struct.pack('<QQ', w_off, weights_len))
            f.write(struct.pack('<QQ', vi_off, len(vi_bytes)))
            f.write(struct.pack('<Q', len(mtp_bundle_bytes)))
            f.write(meta_json)
            f.write(tok_bytes)
            f.write(jinja_bytes)
            f.write(pf_bytes)
            f.write(dc_bytes)
            f.write(vi_bytes)
            f.write(mtp_bundle_bytes)
            f.write(b'\0' * (w_off - weights_unaligned))
            buf = bytearray(8 * 1024 * 1024)
            view = memoryview(buf)
            weights_written = 0
            for source, size, offset in weight_entries:
                f.write(b'\0' * (offset - weights_written))
                if isinstance(source, StreamedWeight):
                    source.write_to(f)
                else:
                    wpath = source
                    with open(wpath, 'rb') as wf:
                        while True:
                            n = wf.readinto(buf)
                            if not n:
                                break
                            f.write(view[:n])
                    # Large converters keep intermediate .weights files only
                    # so they can assemble this package. Reclaim each one
                    # after copying to avoid requiring 2x disk space.
                    if remove_weight_files:
                        os.remove(wpath)
                weights_written = offset + size

        total = w_off + weights_len
        print(f"Saved {output_path} ({weights_len} weights + {len(tok_bytes)} tokenizer + "
              f"{len(jinja_bytes)} jinja + {len(pf_bytes)} prefill + "
              f"{len(dc_bytes)} decode + {len(vi_bytes)} vision + "
              f"{len(mtp_bundle_bytes)} MTP = {total} bytes)")

    finally:
        shutil.rmtree(tmp_dir)
