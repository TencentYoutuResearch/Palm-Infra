"""
PROJECT_NAME — Python graph builder + serialiser

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
GRAPH_VERSION = 1

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
    ADD            = 70
    MUL            = 71
    QUANTIZE_KV    = 80
    DEQUANTIZE_KV  = 81

class Precision(IntEnum):
    FP32 = 0
    FP16 = 1
    INT8 = 2


# ---------------------------------------------------------------------------
# internal node representation
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    id: int
    op_type: OpType
    inputs: list[int] = field(default_factory=list)
    out_shape: tuple[int, ...] = (0, 1, 1, 1)
    out_prec: Precision = Precision.FP32
    params_i32: list[int] = field(default_factory=list)
    params_f32: list[float] = field(default_factory=list)
    params_str: list[str] = field(default_factory=list)
    weight_path: Optional[str] = None   # for weight nodes
    weight_data: Optional[np.ndarray] = None  # for inline constants


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------

class GraphBuilder:
    def __init__(self):
        self._nodes: list[_Node] = []
        self._next_id = 0
        self._weight_files: dict[str, bytes] = {}  # path → binary content

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
              prec: Precision = Precision.FP32) -> int:
        return self._add(OpType.INPUT, [], shape, prec, s=[name])

    def weight(self, path: str, shape: tuple,
               prec: Precision = Precision.FP16) -> int:
        nid = self._add(OpType.CONSTANT, [], shape, prec, s=[path])
        self._nodes[nid].weight_path = path
        return nid

    def constant(self, data: np.ndarray) -> int:
        shape = tuple(data.shape)
        prec = _numpy_to_precision(data.dtype)
        nid = self._add(OpType.CONSTANT, [], shape, prec)
        self._nodes[nid].weight_data = data.copy()
        return nid

    # ---- linear ----

    def matmul(self, a: int, b: int, trans_b: bool = False) -> int:
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
        return self._add(OpType.MATMUL, [a, b], (N, M), prec=self._nodes[a].out_prec)

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
        sx = self._nodes[x].out_shape
        n_elems = 1
        for d in sx: n_elems *= d
        # handle -1 (infer) dimension
        resolved = list(shape)
        if -1 in resolved:
            known = 1
            for d in resolved:
                if d != -1: known *= d
            idx = resolved.index(-1)
            resolved[idx] = n_elems // known
        s_elems = 1
        for d in resolved: s_elems *= d
        assert n_elems == s_elems, f"reshape element mismatch: {sx} vs {shape}"
        # Mark the dynamic dimension: the one that was -1 (inferred) is the
        # seq_len dimension that will change between prefill and decode.
        # Store the dim index in params_i32[4] so the C++ executor can use it.
        resolved_4d = list(self._normalize_shape(tuple(resolved)))
        dynamic_dim = idx if -1 in list(shape) else -1
        i32_params = list(resolved_4d)
        i32_params.append(dynamic_dim)  # params_i32[4] = dynamic dim index
        return self._add(OpType.RESHAPE, [x], tuple(resolved_4d),
                         prec=self._nodes[x].out_prec,
                         i32=i32_params)

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

    # ---- save ----

    def save(self, path_prefix: str):
        """Write <path_prefix>.graph and any referenced .weights files."""
        if path_prefix.endswith('.graph'):
            path_prefix = path_prefix[:-6]

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

        # write graph binary
        graph_path = f"{path_prefix}.graph"
        with open(graph_path, 'wb') as f:
            # header
            f.write(struct.pack('<I', GRAPH_MAGIC))
            f.write(struct.pack('<I', GRAPH_VERSION))
            f.write(struct.pack('<I', len(self._nodes)))
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
                wpath = os.path.join(os.path.dirname(path_prefix) or '.',
                                     f"{os.path.basename(path_prefix)}_const_{node.id}.weights")
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
