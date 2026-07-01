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
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# constants (must match C++ graph.h)
# ---------------------------------------------------------------------------

GRAPH_MAGIC   = 0x4D4C4C47  # "GLLM"
GRAPH_VERSION = 3  # v3: added per-node dynamic[4] (DynamicKind)

WEIGHT_MAGIC  = 0x50414D58  # "XMAP"

class OpType(IntEnum):
    INPUT          = 0
    CONSTANT       = 1
    MATMUL         = 10
    RMS_NORM       = 20
    LAYER_NORM     = 21
    SILU           = 30
    GELU           = 31
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
    QUANTIZE_KV    = 80
    DEQUANTIZE_KV  = 81
    GATED_DELTANET_DECODE  = 110
    GATED_DELTANET_PREFILL = 111
    SHORTCONV      = 140

class Precision(IntEnum):
    FP32 = 0
    FP16 = 1
    INT8 = 2


# ---------------------------------------------------------------------------
# Symbolic dim expressions (must match `struct DimExpr` in graph.h)
# ---------------------------------------------------------------------------
# Each dim of a tensor's output shape can be a symbolic expression:
#   CONST: value = out_shape[i] (static)
#   SEQ:   value = runtime_seq_len
#   MUL:   value = coeff * runtime_seq_len   (covers N * SEQ)
#   ADD:   value = coeff + runtime_seq_len   (covers N + SEQ, rare)
#   BATCH: value = runtime_batch_size         (reserved)
#
# Transpile-time symbolic propagation flows these from INPUT nodes through
# every op. The C++ runtime evaluates them via eval_dim().

class DimKind(IntEnum):
    CONST = 0
    SEQ   = 1
    MUL   = 2
    ADD   = 3
    BATCH = 4


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


# Module-level SEQ symbol for use in reshape() target shapes.
# Usage: g.reshape(x, (head_dim, num_heads * SEQ))  # dim 1 = MUL(num_heads, seq)
#
# In build_graph(), callers should bind SEQ to the actual seq_len via
# SEQ.bind(seq_len) so that transpile-time shape values are correct.
# The unbound SEQ (build-time value 0) is only used for element-count
# inference; runtime evaluation uses runtime_seq_len.
class _SeqSymbol:
    """Symbol representing runtime seq_len in reshape() target shapes.

    Supports arithmetic: SEQ, N * SEQ, N + SEQ. Detected by reshape()
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

    if op in (OpType.RMS_NORM, OpType.LAYER_NORM,
              OpType.SILU, OpType.GELU, OpType.SIGMOID, OpType.EXP, OpType.SOFTPLUS,
              OpType.ROTARY_EMBED,
              OpType.TILE, OpType.CONTIGUOUS,
              OpType.QUANTIZE_KV, OpType.DEQUANTIZE_KV,
              OpType.SHORTCONV):
        return inp(0).dim_expr if n_in >= 1 else _CONST4

    if op in (OpType.ADD, OpType.MUL):
        return inp(0).dim_expr if n_in >= 1 else _CONST4

    if op == OpType.MATMUL:
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

    if op == OpType.GATED_DELTANET_DECODE:
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

    # ---- normalisation ----

    def rms_norm(self, x: int, weight: int, eps: float = 1e-6) -> int:
        sx = self._nodes[x].out_shape
        return self._add(OpType.RMS_NORM, [x, weight], sx,
                         prec=self._nodes[x].out_prec, f32=[eps])

    # ---- activations ----

    def silu(self, x: int) -> int:
        return self._add(OpType.SILU, [x], self._nodes[x].out_shape,
                         prec=self._nodes[x].out_prec)

    def gelu(self, x: int) -> int:
        return self._add(OpType.GELU, [x], self._nodes[x].out_shape,
                         prec=self._nodes[x].out_prec)

    def sigmoid(self, x: int) -> int:
        return self._add(OpType.SIGMOID, [x], self._nodes[x].out_shape,
                         prec=self._nodes[x].out_prec)

    def exp(self, x: int) -> int:
        return self._add(OpType.EXP, [x], self._nodes[x].out_shape,
                         prec=self._nodes[x].out_prec)

    def softplus(self, x: int) -> int:
        return self._add(OpType.SOFTPLUS, [x], self._nodes[x].out_shape,
                         prec=self._nodes[x].out_prec)

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

    def gated_deltanet(self, qkv_conv: int, a_out: int, b_out: int, z_out: int,
                       A_log: int, dt_bias: int, norm_weight: int, gdn_state: int,
                       num_heads: int, k_dim: int, v_dim: int, seq_len: int,
                       use_qk_l2norm: bool = True, rms_eps: float = 1e-6,
                       num_v_heads: int = 0) -> int:
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
        op = (OpType.GATED_DELTANET_DECODE if seq_len == 1
              else OpType.GATED_DELTANET_PREFILL)
        scale = float(k_dim ** -0.5)
        return self._add(op,
                         [qkv_conv, a_out, b_out, z_out,
                          A_log, dt_bias, norm_weight, gdn_state],
                         (num_v_heads * v_dim, seq_len),
                         prec=Precision.FP32,
                         i32=[num_heads, k_dim, v_dim, seq_len,
                              1 if use_qk_l2norm else 0, 4, 0, num_v_heads],
                         f32=[rms_eps, 1e-6, scale])

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


def _write_weight_file(path: str, data: np.ndarray):
    """Write a .weights file with self-contained header."""
    ndim = data.ndim
    shape = list(data.shape) + [1] * (4 - ndim)

    header = struct.pack('<II', WEIGHT_MAGIC, 0)      # magic, flags
    header += struct.pack('<II', ndim, _numpy_to_precision(data.dtype))
    header += struct.pack('<qqqq', *shape)
    data_offset = 88  # sizeof(Header)
    data_size = data.nbytes
    header += struct.pack('<QQ', data_offset, data_size)  # data offset, size
    header += struct.pack('<QQ', 0, 0)                    # no scales
    header += struct.pack('<II', 0, 0)                    # group_size, num_groups
    assert len(header) == 88

    with open(path, 'wb') as f:
        f.write(header)
        f.write(data.tobytes())
    print(f"  Wrote {path} ({data.shape}, {data.dtype})")


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
#     reserved (16)
#   [metadata JSON] — includes "weights" map: {filename: [offset, len]}
#   [tokenizer.json bytes]
#   [chat_template.jinja bytes]
#   [prefill graph bytes]   (standard .graph format, weight refs = "./foo.weights")
#   [decode graph bytes]    (standard .graph format, weight refs = "./foo.weights")
#   [weights region]        — all .weights files concatenated, each self-contained

PACKAGE_MAGIC   = 0x4D4C4F4D  # "MOLM"
PACKAGE_VERSION = 1
PACKAGE_HEADER_SIZE = 128


def save_package(output_path: str,
                 g_prefill: 'GraphBuilder',
                 g_decode: 'GraphBuilder',
                 weights_dir: str,
                 metadata: dict,
                 tokenizer_path: str = "",
                 jinja_path: str = ""):
    """Pack prefill+decode graphs + weights + tokenizer + jinja into a single .mollm file.

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

        # Step 2: read graph bytes
        with open(os.path.join(tmp_dir, "model_prefill.graph"), 'rb') as f:
            pf_bytes = f.read()
        with open(os.path.join(tmp_dir, "model_decode.graph"), 'rb') as f:
            dc_bytes = f.read()

        # Step 3: collect weight files referenced by both graphs
        weight_files = {}  # relative_name -> (offset, size)
        weights_blob = bytearray()

        for g in [g_prefill, g_decode]:
            for node in g._nodes:
                if node.op_type != OpType.CONSTANT or not node.params_str:
                    continue
                ref = node.params_str[0]
                if ref.startswith('#') or ref == "__inline_const__":
                    continue
                if ref in weight_files:
                    continue
                # Find the weight file
                wpath = os.path.join(weights_dir, ref) if not os.path.isabs(ref) else ref
                if not os.path.exists(wpath):
                    wpath = os.path.join(tmp_dir, ref)
                if not os.path.exists(wpath):
                    raise FileNotFoundError(f"Weight file not found: {ref}")
                blob = open(wpath, 'rb').read()
                offset = len(weights_blob)
                weights_blob.extend(blob)
                weight_files[ref] = [offset, len(blob)]

        # Step 4: build metadata
        meta = dict(metadata)
        meta["weights"] = weight_files
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

        # Step 6: write package
        hs = PACKAGE_HEADER_SIZE
        meta_off = hs
        tok_off = meta_off + len(meta_json)
        jin_off = tok_off + len(tok_bytes)
        pf_off = jin_off + len(jinja_bytes)
        dc_off = pf_off + len(pf_bytes)
        w_off = dc_off + len(dc_bytes)

        with open(output_path, 'wb') as f:
            f.write(struct.pack('<II', PACKAGE_MAGIC, PACKAGE_VERSION))
            f.write(struct.pack('<QQ', meta_off, len(meta_json)))
            f.write(struct.pack('<QQ', tok_off, len(tok_bytes)))
            f.write(struct.pack('<QQ', jin_off, len(jinja_bytes)))
            f.write(struct.pack('<QQ', pf_off, len(pf_bytes)))
            f.write(struct.pack('<QQ', dc_off, len(dc_bytes)))
            f.write(struct.pack('<QQ', w_off, len(weights_blob)))
            f.write(b'\x00' * (hs - 4 - 4 - 8 * 12))  # pad to 128
            f.write(meta_json)
            f.write(tok_bytes)
            f.write(jinja_bytes)
            f.write(pf_bytes)
            f.write(dc_bytes)
            f.write(weights_blob)

        total = (hs + len(meta_json) + len(tok_bytes) + len(jinja_bytes)
                 + len(pf_bytes) + len(dc_bytes) + len(weights_blob))
        print(f"Saved {output_path} ({len(weights_blob)} weights + {len(tok_bytes)} tokenizer + "
              f"{len(jinja_bytes)} jinja + {len(pf_bytes)} prefill + {len(dc_bytes)} decode = {total} bytes)")

    finally:
        shutil.rmtree(tmp_dir)
