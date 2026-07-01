#!/usr/bin/env python3
"""Unit tests for transpile.propagate_dim_exprs().

Verifies that ONNX-style symbolic shape propagation correctly flows
DimExpr (SEQ, MUL, ADD) from INPUT nodes through every op in the graph.

Run:
    python3 tests/test_shape_propagation.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from transpile import (GraphBuilder, OpType, Precision, DimExpr, DimKind,
                       SEQ, propagate_dim_exprs)

# Bind SEQ to build-time seq_len=256 for tests
SEQ = SEQ.bind(256)

_CONST = DimExpr.const()
_SEQ   = DimExpr.seq()
CONST4 = (_CONST,) * 4
SEQ_DIM1 = (_CONST, _SEQ, _CONST, _CONST)  # seq on dim 1

failures = 0


def check(cond, msg):
    global failures
    if cond:
        print(f"  PASS: {msg}")
    else:
        print(f"  FAIL: {msg}")
        failures += 1


def de_of(g: GraphBuilder, nid: int):
    return tuple(g._nodes[nid].dim_expr)


def is_seq(e):
    return e.kind == DimKind.SEQ


def is_const(e):
    return e.kind == DimKind.CONST


def test_input_carries_seq():
    g = GraphBuilder()
    g.input('hidden', (1024, 256), dynamic=SEQ_DIM1)
    propagate_dim_exprs(g._nodes)
    de = de_of(g, 0)
    check(is_seq(de[1]) and is_const(de[0]), "INPUT keeps SEQ on dim 1")


def test_constant_is_const():
    g = GraphBuilder()
    g.input('hidden', (1024, 256), dynamic=SEQ_DIM1)
    g.weight('w', (4096, 1024))
    propagate_dim_exprs(g._nodes)
    de = de_of(g, 1)
    check(all(is_const(d) for d in de), "CONSTANT is all CONST")


def test_matmul_propagates_seq_dim1():
    g = GraphBuilder()
    h = g.input('hidden', (1024, 256), dynamic=SEQ_DIM1)
    w = g.weight('w', (4096, 1024))
    mm = g.matmul(h, w)
    propagate_dim_exprs(g._nodes)
    de = de_of(g, mm)
    check(is_seq(de[1]), "MATMUL inherits SEQ on dim 1 from A")


def test_rms_norm_inherits():
    g = GraphBuilder()
    h = g.input('hidden', (1024, 256), dynamic=SEQ_DIM1)
    w = g.weight('w', (1024,))
    n = g.rms_norm(h, w)
    propagate_dim_exprs(g._nodes)
    de = de_of(g, n)
    check(is_seq(de[1]), "RMS_NORM inherits input dim_expr")


def test_silu_inherits():
    g = GraphBuilder()
    h = g.input('hidden', (1024, 256), dynamic=SEQ_DIM1)
    s = g.silu(h)
    propagate_dim_exprs(g._nodes)
    de = de_of(g, s)
    check(is_seq(de[1]), "SILU inherits input dim_expr")


def test_add_inherits_first_input():
    g = GraphBuilder()
    a = g.input('a', (1024, 256), dynamic=SEQ_DIM1)
    b = g.input('b', (1024, 256))  # all CONST
    out = g.add(a, b)
    propagate_dim_exprs(g._nodes)
    de = de_of(g, out)
    check(is_seq(de[1]), "ADD inherits input[0] dim_expr")


def test_sdpa_seq_from_q():
    g = GraphBuilder()
    q = g.input('q', (128, 256), dynamic=SEQ_DIM1)
    k = g.input('k', (128, 1))
    v = g.input('v', (128, 1))
    mask = g.input('mask', (1, 256))
    kc = g.input('kc', (128, 4096, 4), prec=Precision.FP16)
    vc = g.input('vc', (128, 4096, 4), prec=Precision.FP16)
    attn, _, _ = g.sdpa(q, k, v, mask, kc, vc,
                       num_heads=4, num_kv_heads=4,
                       head_dim=128, v_head_dim=128)
    propagate_dim_exprs(g._nodes)
    de = de_of(g, attn)
    check(is_seq(de[1]), "SDPA inherits SEQ on dim 1 from Q")


def test_permute_reorders():
    g = GraphBuilder()
    h = g.input('hidden', (1024, 256), dynamic=SEQ_DIM1)
    p = g.permute(h, (1, 0, 2, 3))
    propagate_dim_exprs(g._nodes)
    de = de_of(g, p)
    # dim 0 was originally dim 1 (SEQ), dim 1 was originally dim 0 (CONST)
    check(is_seq(de[0]) and is_const(de[1]), "PERMUTE reorders dim_expr")


def test_gated_deltanet_prefill_has_seq_dim1():
    g = GraphBuilder()
    qkv = g.input('qkv', (1024, 256), dynamic=SEQ_DIM1)
    a = g.input('a', (16, 256), dynamic=SEQ_DIM1)
    b = g.input('b', (16, 256), dynamic=SEQ_DIM1)
    z = g.input('z', (1024, 256), dynamic=SEQ_DIM1)
    A_log = g.weight('A', (16,))
    dt = g.weight('dt', (16,))
    nw = g.weight('nw', (128,))
    gs = g.input('gs', (128, 128, 16), prec=Precision.FP32)
    gdn = g.gated_deltanet(qkv, a, b, z, A_log, dt, nw, gs,
                           num_heads=16, k_dim=128, v_dim=128, seq_len=256)
    propagate_dim_exprs(g._nodes)
    de = de_of(g, gdn)
    check(is_seq(de[1]), "GATED_DELTANET_PREFILL has SEQ on dim 1")


def test_decode_graph_all_const():
    g = GraphBuilder()
    g.input('hidden', (1024, 1))  # no dynamic
    g.input('mask', (1, 1))
    propagate_dim_exprs(g._nodes)
    de0 = de_of(g, 0)
    de1 = de_of(g, 1)
    check(all(is_const(d) for d in de0), "decode INPUT[hidden] all CONST")
    check(all(is_const(d) for d in de1), "decode INPUT[mask] all CONST")


def test_reshape_with_seq_symbol():
    """reshape() with SEQ symbol in target shape marks that dim as SEQ."""
    g = GraphBuilder()
    h = g.input('hidden', (1024, 256), dynamic=SEQ_DIM1)
    r = g.reshape(h, (256, SEQ))  # dim 1 = SEQ
    propagate_dim_exprs(g._nodes)
    de = de_of(g, r)
    check(is_seq(de[1]), "RESHAPE with SEQ symbol marks dim 1")


def test_reshape_with_mul_symbol():
    """reshape() with N*SEQ symbol marks that dim as MUL with coeff N."""
    g = GraphBuilder()
    # input [256, 8, 256, 1] (head_dim, num_heads, seq) — 524288 elems
    h = g.input('hidden', (256, 8, 256, 1), dynamic=(_CONST, _CONST, _SEQ, _CONST))
    # reshape to [256, 8*SEQ] — dim 1 = MUL(8, seq)
    r = g.reshape(h, (256, 8 * SEQ))
    propagate_dim_exprs(g._nodes)
    de = de_of(g, r)
    e = de[1]
    check(e.kind == DimKind.MUL and e.coeff == 8,
          f"RESHAPE with 8*SEQ marks dim 1 as MUL(coeff=8), got kind={e.kind} coeff={e.coeff}")


def test_no_seq_propagates_to_all_const():
    """Graph with no SEQ INPUTs has all CONST nodes."""
    g = GraphBuilder()
    h = g.input('h', (1024, 256))
    w = g.weight('w', (4096, 1024))
    mm = g.matmul(h, w)
    n = g.rms_norm(mm, g.weight('n', (4096,)))
    propagate_dim_exprs(g._nodes)
    for node in g._nodes:
        de = de_of(g, node.id)
        check(all(is_const(d) for d in de),
              f"node {node.id} ({node.op_type.name}) all CONST")


def main():
    print("=== test_shape_propagation ===\n")
    tests = [
        test_input_carries_seq,
        test_constant_is_const,
        test_matmul_propagates_seq_dim1,
        test_rms_norm_inherits,
        test_silu_inherits,
        test_add_inherits_first_input,
        test_sdpa_seq_from_q,
        test_permute_reorders,
        test_gated_deltanet_prefill_has_seq_dim1,
        test_decode_graph_all_const,
        test_reshape_with_seq_symbol,
        test_reshape_with_mul_symbol,
        test_no_seq_propagates_to_all_const,
    ]
    for t in tests:
        print(f"\n{t.__name__}:")
        t()
    print()
    if failures == 0:
        print("All shape propagation tests passed!")
    else:
        print(f"{failures} test(s) FAILED")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
