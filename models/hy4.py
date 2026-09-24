#!/usr/bin/env python3
"""Hy4-preview model → mollm graph converter.

Usage:
    python3 models/converter.py /path/to/Hy4-preview out.mollm

Hy4-preview is a 770B MoE whose architecture differs from the Qwen-shaped
models here in three ways:

  * MLA attention with a learned per-head attention sink and an elementwise
    gate on the decompressed attention output (`linear_gate`, applied before
    o_proj).
  * Gated DeepSeek Sparse Attention. A lightning indexer selects the top-k
    keys each layer attends to; only some layers run the indexer and the rest
    reuse the previous "full" layer's selection (`indexer_types` in config).
  * identity Hyper-Connections: the residual stream is `hc_mult` wide, is
    reduced to a single stream for each branch, then scattered back.

Layouts and formulas are documented in docs_internal/HY4_OPS_SPEC.md, which
was cross-checked against tencent/Hy4-preview config.json plus safetensors
headers, and against the reference graph in the AngelSlim GGUF patch.

Status: the layer stack, attention (MLA + sink + gate) and FFN branches are
built here. The DSA indexer and IndexCache wiring and the weight
export/quantisation path are not finished — see the notes at `build_graph`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import (
    Activation,
    DimExpr,
    GraphBuilder,
    Precision,
    SEQ,
)
from safetensors_stream import (
    SafeTensorIndex,
    dense_streamed_weight,
    fp32_streamed_weight,
)

INDEXER_FULL = "full"

# Resolved per graph inside build_graph; Hy4 always uses a prefill/decode pair.
_S = 1


def _fp32_weight(graph: GraphBuilder, streamed: dict,
                 index: SafeTensorIndex, label: str, source: str,
                 shape: tuple[int, ...]) -> int:
    streamed[label] = fp32_streamed_weight(
        index, source, logical_shape=shape)
    return graph.weight(label, shape, Precision.FP32)


def _dense_weight(graph: GraphBuilder, streamed: dict,
                  index: SafeTensorIndex, label: str, source: str,
                  shape: tuple[int, ...]) -> int:
    streamed[label] = dense_streamed_weight(index, source)
    return graph.weight(label, shape, Precision.FP16)


def _hc_branch(graph: GraphBuilder, streamed: dict,
               index: SafeTensorIndex, cfg: dict, layer: int, kind: str,
               residual: int, branch_builder) -> int:
    """Wrap one branch in Hy4 identity Hyper-Connections.

    iHC is a single-stream wrapper: hc_pre_id collapses the hc residual
    streams into one hidden state, the branch runs on that, and hc_post_id
    scatters the result back across the streams. Unlike the DeepSeek-V4
    variant there is no Sinkhorn mixing matrix, so `fn` has 2*hc_mult rows
    and `scale` is two scalars (pre and post).
    """
    hidden = int(cfg["hidden_size"])
    hc_mult = int(cfg["hc_mult"])
    wide = hidden * hc_mult
    mix = 2 * hc_mult
    prefix = f"layers.{layer}.hc_{kind}_layer.hc_pre"
    label = f"layer_{layer}_hc_{kind}"

    fn = _fp32_weight(
        graph, streamed, index, f"{label}_fn", f"{prefix}.hc_fn",
        (mix, wide))
    scale = _fp32_weight(
        graph, streamed, index, f"{label}_scale", f"{prefix}.hc_scale",
        (2,))
    base = _fp32_weight(
        graph, streamed, index, f"{label}_base", f"{prefix}.hc_base",
        (mix,))

    packed = graph.hc_pre_id(
        residual, fn, scale, base, hidden, hc_mult,
        float(cfg.get("hc_magnitude", 2.0)),
        float(cfg["rms_norm_eps"]), float(cfg["hc_eps"]))
    reduced = graph.slice_range(packed, 0, hidden, dim=0)
    branch = branch_builder(reduced)
    return graph.hc_post_id(branch, residual, packed, hidden, hc_mult)


def _add_indexer(
        graph: GraphBuilder, streamed: dict, index: SafeTensorIndex,
        cfg: dict, layer: int, hidden: int, q_lora: int, position: int,
        key_cache: int, n_ctx: int) -> int:
    """Run the Hy4 Gated-DSA indexer for one layer.

    `hidden` is the normed hidden state and `q_lora` the normed low-rank
    query the attention branch produced; both are reused rather than
    recomputed. Returns the INT32 [top_k, seq] selection that the sparse
    attention consumes.
    """
    hidden_size = int(cfg["hidden_size"])
    heads = int(cfg["index_n_heads"])
    head_dim = int(cfg["index_head_dim"])
    q_rank = int(cfg["q_lora_rank"])
    rope_dim = int(cfg["qk_rope_head_dim"])
    top_k = int(cfg["index_topk"])
    theta = float(cfg.get("rope_parameters", {}).get("rope_theta", 1e7))
    prefix = f"layers.{layer}.self_attn"

    def weight(suffix: str, shape: tuple[int, ...]) -> int:
        label = f"layer_{layer}_indexer_{suffix.replace('.', '_')}"
        return _dense_weight(
            graph, streamed, index, label, f"{prefix}.{suffix}", shape)

    wq_b = weight("indexer.wq_b.weight", (heads * head_dim, q_rank))
    wk = weight("indexer.wk.weight", (head_dim, hidden_size))
    weights_proj = weight(
        "indexer.weights_proj.weight", (heads, hidden_size))
    k_norm_w = _fp32_weight(
        graph, streamed, index, f"layer_{layer}_indexer_k_norm",
        f"{prefix}.indexer.k_norm.weight", (head_dim,))
    k_norm_b = _fp32_weight(
        graph, streamed, index, f"layer_{layer}_indexer_k_norm_b",
        f"{prefix}.indexer.k_norm.bias", (head_dim,))

    return graph.hy4_indexer(
        hidden, q_lora, wq_b, wk, weights_proj, k_norm_w, k_norm_b,
        key_cache, position,
        hidden_size=hidden_size, q_lora_rank=q_rank, num_heads=heads,
        head_dim=head_dim, rope_dim=rope_dim, top_k=top_k,
        capacity=n_ctx, theta=theta)


def _add_attention_branch(
        graph: GraphBuilder, streamed: dict, index: SafeTensorIndex,
        cfg: dict, layer: int, hidden: int, q_lora: int, position: int,
        cos: int, sin: int, mask: int, ck_in: int, cv_in: int,
        top_k: int | None) -> int:
    """MLA + sink + gate for one layer.

    `q_lora` is the already-normed low-rank query, built by the caller so the
    indexer can share it. `top_k` is the key selection owned by the DSA
    indexer; pass None for dense attention over the cache.
    """
    hidden_size = int(cfg["hidden_size"])
    heads = int(cfg["num_attention_heads"])
    nope_dim = int(cfg["qk_nope_head_dim"])
    rope_dim = int(cfg["qk_rope_head_dim"])
    head_dim = nope_dim + rope_dim
    v_head_dim = int(cfg["v_head_dim"])
    q_rank = int(cfg["q_lora_rank"])
    kv_rank = int(cfg["kv_lora_rank"])
    eps = float(cfg["rms_norm_eps"])
    prefix = f"layers.{layer}.self_attn"

    def weight(suffix: str, shape: tuple[int, ...]) -> int:
        label = f"layer_{layer}_attn_{suffix.replace('.', '_')}"
        return _dense_weight(
            graph, streamed, index, label, f"{prefix}.{suffix}", shape)

    wq_b = weight("q_b_proj.weight", (heads * head_dim, q_rank))
    wkv_a = weight(
        "kv_a_proj_with_mqa.weight", (kv_rank + rope_dim, hidden_size))
    kv_a_norm = _fp32_weight(
        graph, streamed, index, f"layer_{layer}_attn_kv_a_norm",
        f"{prefix}.kv_a_layernorm.weight", (kv_rank,))
    wk_b = weight(
        "kv_b_proj.weight", (heads * (nope_dim + v_head_dim), kv_rank))
    wo = weight("o_proj.weight", (hidden_size, heads * v_head_dim))
    gate_w = weight(
        "linear_gate.weight", (heads * v_head_dim, hidden_size))
    attn_sink = _fp32_weight(
        graph, streamed, index, f"layer_{layer}_attn_sink",
        f"{prefix}.learnable_sink_param", (heads,))

    # MLA query path; q_lora arrives already normed from the caller.
    query = graph.matmul(q_lora, wq_b)
    query = graph.reshape(query, (head_dim, heads, _S))
    query = graph.permute(query, (0, 2, 1, 3))       # [head_dim, seq, heads]
    q_nope, q_rope = graph.slice(query, [nope_dim, rope_dim], dim=0)
    q_rope = graph.rope(q_rope, cos, sin, rope_dim=rope_dim)

    # MLA KV path.
    kv = graph.matmul(hidden, wkv_a)
    kv_cmpr, k_rope = graph.slice(kv, [kv_rank, rope_dim], dim=0)
    kv_cmpr = graph.rms_norm(kv_cmpr, kv_a_norm, eps)
    k_rope = graph.reshape(k_rope, (rope_dim, _S, 1))
    k_rope = graph.rope(k_rope, cos, sin, rope_dim=rope_dim)
    k_rope = graph.tile(k_rope, (1, 1, heads))

    # Expand the compressed KV through wk_b and split into k_nope / v.
    kv_expanded = graph.matmul(kv_cmpr, wk_b)
    kv_expanded = graph.reshape(
        kv_expanded, (nope_dim + v_head_dim, heads, _S))
    kv_expanded = graph.permute(kv_expanded, (0, 2, 1, 3))
    k_nope, v = graph.slice(kv_expanded, [nope_dim, v_head_dim], dim=0)

    q_full = graph.concat([q_nope, q_rope], dim=0)
    k_full = graph.concat([k_nope, k_rope], dim=0)

    attn_out, _, _ = graph.sdpa(
        q_full, k_full, v, mask, ck_in, cv_in,
        kv_cache=2, causal=True, scale=1.0 / (head_dim ** 0.5),
        num_heads=heads, num_kv_heads=heads,
        head_dim=head_dim, v_head_dim=v_head_dim,
        sink=attn_sink, selection=top_k)

    # Gated MLA: elementwise sigmoid gate on the decompressed attention
    # output, applied BEFORE o_proj.
    attn_out = graph.permute(attn_out, (0, 2, 1, 3))
    attn_out = graph.reshape(attn_out, (heads * v_head_dim, _S))
    gate = graph.sigmoid(graph.matmul(hidden, gate_w))
    attn_out = graph.mul(attn_out, gate)

    return graph.matmul(attn_out, wo)


def _add_ffn_branch(
        graph: GraphBuilder, streamed: dict, index: SafeTensorIndex,
        cfg: dict, layer: int, hidden: int, token_ids: int) -> int:
    """Dense FFN for the leading layer, routed + shared MoE after it."""
    hidden_size = int(cfg["hidden_size"])
    intermediate = int(cfg["intermediate_size"])
    moe_intermediate = int(cfg["moe_intermediate_size"])
    experts = int(cfg["n_routed_experts"])
    top_k = int(cfg["num_experts_per_tok"])
    prefix = f"layers.{layer}.mlp"
    dense_lead = int(cfg.get("first_k_dense_replace", 1))

    if layer < dense_lead:
        gate_up = _dense_weight(
            graph, streamed, index, f"layer_{layer}_ffn_gate_up",
            f"{prefix}.gate_up_proj.weight", (2 * intermediate, hidden_size))
        down = _dense_weight(
            graph, streamed, index, f"layer_{layer}_ffn_down",
            f"{prefix}.down_proj.weight", (hidden_size, intermediate))
        # Fused gate_up matmul with SILU folded into the gate half.
        merged = graph.matmul(
            hidden, gate_up, activation=Activation.SILU,
            act_n_begin=0, act_n_len=intermediate)
        gate, up = graph.slice(merged, [intermediate, intermediate], dim=0)
        return graph.matmul(graph.mul(gate, up), down)

    # Routed experts are stored fused as [experts, 2*moe_i, hidden].
    experts_gate_up = _dense_weight(
        graph, streamed, index, f"layer_{layer}_ffn_experts_gate_up",
        f"{prefix}.experts.gate_up_proj",
        (experts, 2 * moe_intermediate, hidden_size))
    experts_down = _dense_weight(
        graph, streamed, index, f"layer_{layer}_ffn_experts_down",
        f"{prefix}.experts.down_proj",
        (experts, hidden_size, moe_intermediate))
    router = _dense_weight(
        graph, streamed, index, f"layer_{layer}_ffn_router",
        f"{prefix}.gate.weight", (experts, hidden_size))
    router_bias = None
    if index.contains(f"{prefix}.gate.e_score_correction_bias"):
        router_bias = _fp32_weight(
            graph, streamed, index, f"layer_{layer}_ffn_router_bias",
            f"{prefix}.gate.e_score_correction_bias", (experts,))

    shared_gate = _dense_weight(
        graph, streamed, index, f"layer_{layer}_ffn_shared_gate",
        f"{prefix}.shared_experts.gate_proj.weight",
        (moe_intermediate, hidden_size))
    shared_up = _dense_weight(
        graph, streamed, index, f"layer_{layer}_ffn_shared_up",
        f"{prefix}.shared_experts.up_proj.weight",
        (moe_intermediate, hidden_size))
    shared_down = _dense_weight(
        graph, streamed, index, f"layer_{layer}_ffn_shared_down",
        f"{prefix}.shared_experts.down_proj.weight",
        (hidden_size, moe_intermediate))

    return graph.moe(
        hidden, router, experts_gate_up, experts_down,
        shared_gate, shared_up, shared_down,
        shared_expert_gate=None,
        hidden_size=hidden_size, num_experts=experts, top_k=top_k,
        intermediate_size=moe_intermediate,
        shared_intermediate_size=moe_intermediate,
        router_bias=router_bias,
        # Hy4 scores experts with a sigmoid over the full expert set.
        router_score_func=0,
        norm_topk_prob=bool(cfg["norm_topk_prob"]),
        has_shared_expert=True,
        shared_expert_has_gate=False,
        routed_scaling_factor=float(cfg["routed_scaling_factor"]),
        swiglu_limit=float(cfg.get("swiglu_limit", 0.0)))


def build_graph(model_dir: str | Path, cfg: dict, seq_len: int = 1,
                n_ctx: int = 4096, is_prefill: bool = False
                ) -> tuple[GraphBuilder, dict]:
    """Build one Hy4 graph (prefill when is_prefill, else decode)."""
    global _S
    _S = SEQ.bind(seq_len) if is_prefill else seq_len

    index = SafeTensorIndex(Path(model_dir))
    streamed: dict = {}
    graph = GraphBuilder()
    hidden_size = int(cfg["hidden_size"])
    hc_mult = int(cfg["hc_mult"])
    rope_dim = int(cfg["qk_rope_head_dim"])
    heads = int(cfg["num_attention_heads"])
    q_rank = int(cfg["q_lora_rank"])
    layers = int(cfg["num_hidden_layers"])
    # Per-layer indexer role: only "full" layers ship indexer weights and run
    # the selector; "shared" layers reuse the previous full layer's top-k.
    indexer_types = list(cfg["indexer_types"])
    eps = float(cfg["rms_norm_eps"])

    if len(indexer_types) != layers:
        raise ValueError(
            f"indexer_types has {len(indexer_types)} entries but the model "
            f"has {layers} layers")
    if indexer_types[0] != INDEXER_FULL:
        raise ValueError(
            "layer 0 must run the indexer: shared layers reuse the previous "
            "full layer's selection, so a leading shared layer has nothing "
            "to reuse")

    if is_prefill:
        seq_dyn = DimExpr.seq()
        const = DimExpr.const()
        hidden_dyn = (const, seq_dyn, const, const)
        mask_dyn = cos_dyn = sin_dyn = hidden_dyn
    else:
        hidden_dyn = mask_dyn = cos_dyn = sin_dyn = None

    hidden = graph.input(
        "hidden", (hidden_size, seq_len), dynamic=hidden_dyn)
    token_ids = graph.input("token_ids", (seq_len,), prec=Precision.INT32)
    mask = graph.input("mask", (1, seq_len), dynamic=mask_dyn)
    cos = graph.input("cos", (rope_dim // 2, seq_len), dynamic=cos_dyn)
    sin = graph.input("sin", (rope_dim // 2, seq_len), dynamic=sin_dyn)
    position = graph.input("position", (1,), prec=Precision.INT32)

    caches = []
    for layer in range(layers):
        ck = graph.input(
            f"cache_k{layer}", (0, n_ctx, heads), prec=Precision.FP16)
        cv = graph.input(
            f"cache_v{layer}", (0, n_ctx, heads), prec=Precision.FP16)
        caches.append([ck, cv])

    # Expand the single embedding into hc_mult parallel residual streams.
    state = graph.tile(hidden, (hc_mult, 1))

    # IndexCache: the most recent "full" indexer layer's top-k selection is
    # reused by every following "shared" layer, which skips its own indexer
    # entirely. Only `full` layers ship indexer weights, so this is both a
    # compute saving and the mechanism the checkpoint expects.
    last_top_k: int | None = None

    for layer in range(layers):
        attn_norm = _fp32_weight(
            graph, streamed, index, f"layer_{layer}_attn_norm",
            f"layers.{layer}.input_layernorm.weight", (hidden_size,))
        ck_in, cv_in = caches[layer]
        is_full = indexer_types[layer] == INDEXER_FULL

        # q_a norm weight is shared by the attention branch and the indexer,
        # which both consume the same normed low-rank query.
        q_a_norm = _fp32_weight(
            graph, streamed, index, f"layer_{layer}_attn_q_a_norm",
            f"layers.{layer}.self_attn.q_a_layernorm.weight", (q_rank,))
        wq_a = _dense_weight(
            graph, streamed, index, f"layer_{layer}_attn_q_a_proj_weight",
            f"layers.{layer}.self_attn.q_a_proj.weight",
            (q_rank, hidden_size))

        # `selected` carries the indexer result back out of the branch
        # closure so the next layer can reuse it.
        selected: list[int] = []

        def attention_branch(reduced: int, *, current=layer,
                             norm=attn_norm, ck=ck_in, cv=cv_in,
                             full=is_full, wq=wq_a, qn=q_a_norm) -> int:
            normalized = graph.rms_norm(reduced, norm, eps)
            q_lora = graph.rms_norm(graph.matmul(normalized, wq), qn, eps)
            if full:
                top_k = _add_indexer(
                    graph, streamed, index, cfg, current, normalized,
                    q_lora, position, ck, n_ctx)
                selected.append(top_k)
            else:
                # A shared layer reuses the previous full layer's selection.
                # Hy4 marks layer 0 full, so last_top_k is set before any
                # shared layer unless the config is malformed.
                if last_top_k is None:
                    raise ValueError(
                        f"layer {current} is marked 'shared' but no earlier "
                        f"layer ran the indexer")
                top_k = last_top_k
            return _add_attention_branch(
                graph, streamed, index, cfg, current, normalized,
                q_lora, position, cos, sin, mask, ck, cv, top_k)

        state = _hc_branch(
            graph, streamed, index, cfg, layer, "attn", state,
            attention_branch)
        if selected:
            last_top_k = selected[0]

        ffn_norm = _fp32_weight(
            graph, streamed, index, f"layer_{layer}_ffn_norm",
            f"layers.{layer}.post_attention_layernorm.weight",
            (hidden_size,))

        def ffn_branch(reduced: int, *, current=layer,
                       norm=ffn_norm) -> int:
            normalized = graph.rms_norm(reduced, norm, eps)
            return _add_ffn_branch(
                graph, streamed, index, cfg, current, normalized, token_ids)

        state = _hc_branch(
            graph, streamed, index, cfg, layer, "mlp", state, ffn_branch)

    wide = hidden_size * hc_mult
    head_fn = _fp32_weight(
        graph, streamed, index, "hc_head_fn",
        "model.hc_head.hc_head_fn", (hc_mult, wide))
    head_scale = _fp32_weight(
        graph, streamed, index, "hc_head_scale",
        "model.hc_head.hc_head_scale", (1,))
    head_base = _fp32_weight(
        graph, streamed, index, "hc_head_base",
        "model.hc_head.hc_head_base", (hc_mult,))
    collapsed = graph.hc_head_id(
        state, head_fn, head_scale, head_base, hidden_size, hc_mult,
        eps, float(cfg["hc_eps"]))

    final_norm = _fp32_weight(
        graph, streamed, index, "final_norm", "model.norm.weight",
        (hidden_size,))
    output = graph.rms_norm(collapsed, final_norm, eps)
    return graph, streamed


def convert(model_dir: str, output_path: str, **kwargs) -> None:
    """Converter registry entry point.

    Deliberately not implemented yet: build_graph still needs the DSA
    indexer + IndexCache wiring, and the export path needs the streamed
    writers for the 3-D fused expert tensors. Fail loudly rather than
    produce a package that would silently skip sparse attention.
    """
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    if cfg.get("model_type") != "hy_v4":
        raise ValueError(
            f"not a Hy4 config: model_type={cfg.get('model_type')!r}")
    raise NotImplementedError(
        "Hy4 conversion is incomplete: the graph builder covers MLA, the "
        "attention sink, the gating and the MoE branches, but DSA indexer "
        "and IndexCache wiring plus the expert export path are still to do.")
