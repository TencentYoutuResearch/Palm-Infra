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

Status: the layer stack, DSA indexer, latent-cache MLA, and FFN branches are
built here. GGUF can produce a one-layer FP16 diagnostic package; low-bit
full-model export is still in progress.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transpile import (
    Activation,
    DimExpr,
    GraphBuilder,
    Precision,
    SEQ,
    save_package,
)
from safetensors_stream import (
    SafeTensorIndex,
    aggregate_dense_experts,
    dense_streamed_weight,
    fp32_streamed_weight,
    split_mla_kv_b,
)
from qwen3_moe import ROUTER_SCORE_SIGMOID
from gguf_reader import GgufIndex, config_from_gguf, open_gguf
from gguf_stream import gguf_fp16_streamed_weight

INDEXER_FULL = "full"

# Resolved per graph inside build_graph; Hy4 always uses a prefill/decode pair.
_S = 1
_GGUF_PY_PATH: str | None = None


def _fp32_weight(graph: GraphBuilder, streamed: dict,
                 index: SafeTensorIndex, label: str, source: str,
                 shape: tuple[int, ...]) -> int:
    streamed[label] = fp32_streamed_weight(
        index, source, logical_shape=shape)
    return graph.weight(label, shape, Precision.FP32)


def _dense_weight(graph: GraphBuilder, streamed: dict,
                  index: SafeTensorIndex, label: str, source: str,
                  shape: tuple[int, ...]) -> int:
    if isinstance(index, GgufIndex):
        streamed[label] = gguf_fp16_streamed_weight(
            index, source, gguf_py_path=_GGUF_PY_PATH)
    else:
        streamed[label] = dense_streamed_weight(index, source)
    if tuple(streamed[label].logical_shape) != tuple(shape):
        raise ValueError(
            f"{source}: weight shape {streamed[label].logical_shape} != {shape}")
    return graph.weight(label, shape, Precision.FP16)


def _fused_experts(graph: GraphBuilder, streamed: dict,
                   index: SafeTensorIndex, label: str, source: str,
                   shape: tuple[int, ...], rows_per_expert: int) -> int:
    """Declare a fused 3-D expert tensor as flattened expert-major rows.

    `shape` is the flattened form the MoE kernel indexes; the checkpoint
    stores [experts, rows_per_expert, K] and the row order is already
    expert-major, so the payload streams verbatim.
    """
    streamed[label] = aggregate_dense_experts(
        index, source, rows_per_expert=rows_per_expert)
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
    prefix = f"model.layers.{layer}.hc_{kind}_layer.hc_pre"
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
        n_tokens: int, key_cache: int, n_ctx: int) -> int:
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
    prefix = f"model.layers.{layer}.self_attn"

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
        key_cache, position, n_tokens,
        hidden_size=hidden_size, q_lora_rank=q_rank, num_heads=heads,
        head_dim=head_dim, rope_dim=rope_dim, top_k=top_k,
        capacity=n_ctx, theta=theta)


def _add_attention_branch(
        graph: GraphBuilder, streamed: dict, index: SafeTensorIndex,
        cfg: dict, layer: int, hidden: int, q_lora: int, position: int,
        cos: int, sin: int, mask: int, ck_in: int | None, cv_in: int | None,
        top_k: int | None, absorbed: bool = False) -> int:
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
    prefix = f"model.layers.{layer}.self_attn"

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
    if absorbed:
        if isinstance(index, GgufIndex):
            k_stream = gguf_fp16_streamed_weight(
                index, f"{prefix}.k_b.weight", gguf_py_path=_GGUF_PY_PATH)
            v_stream = gguf_fp16_streamed_weight(
                index, f"{prefix}.v_b.weight", gguf_py_path=_GGUF_PY_PATH)
        else:
            k_stream, v_stream = split_mla_kv_b(
                index, f"{prefix}.kv_b_proj.weight", num_heads=heads,
                nope_dim=nope_dim, v_head_dim=v_head_dim, kv_lora_rank=kv_rank)
        k_label = f"layer_{layer}_attn_k_b"
        v_label = f"layer_{layer}_attn_v_b"
        streamed[k_label] = k_stream
        streamed[v_label] = v_stream
        k_b = graph.weight(k_label, (heads, kv_rank, nope_dim), Precision.FP16)
        v_b = graph.weight(v_label, (heads, v_head_dim, kv_rank), Precision.FP16)
    else:
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
    # Non-interleaved RoPE: the reference composes it with `rotate_half`, which
    # pairs dimension i with i + rope_dim/2 rather than with i+1. The two
    # styles agree at position 0 (cos=1, sin=0 makes both the identity) and
    # diverge everywhere else, so an interleaved build looks correct on the
    # first token and drifts after -- exactly the signature the engine showed.
    q_rope = graph.rope(q_rope, cos, sin, rope_dim=rope_dim,
                        interleave=False)

    # MLA KV path.
    kv = graph.matmul(hidden, wkv_a)
    kv_cmpr, k_rope = graph.slice(kv, [kv_rank, rope_dim], dim=0)
    kv_cmpr = graph.rms_norm(kv_cmpr, kv_a_norm, eps)
    k_rope = graph.reshape(k_rope, (rope_dim, _S, 1))
    k_rope = graph.rope(k_rope, cos, sin, rope_dim=rope_dim,
                        interleave=False)
    if absorbed:
        attn_out = graph.sdpa_mla_absorb(
            q_nope, q_rope, kv_cmpr, k_rope, k_b, v_b, mask, ck_in,
            num_heads=heads, qk_head_dim=head_dim, v_head_dim=v_head_dim,
            kv_rank=kv_rank, rope_dim=rope_dim, sink=attn_sink,
            selection=top_k)
    else:
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
    prefix = f"model.layers.{layer}.mlp"
    dense_lead = int(cfg.get("first_k_dense_replace", 1))

    if layer < dense_lead:
        # The dense lead layer ships gate_proj and up_proj separately.
        gate = _dense_weight(
            graph, streamed, index, f"layer_{layer}_ffn_gate",
            f"{prefix}.gate_proj.weight", (intermediate, hidden_size))
        up = _dense_weight(
            graph, streamed, index, f"layer_{layer}_ffn_up",
            f"{prefix}.up_proj.weight", (intermediate, hidden_size))
        down = _dense_weight(
            graph, streamed, index, f"layer_{layer}_ffn_down",
            f"{prefix}.down_proj.weight", (hidden_size, intermediate))
        gate = graph.matmul(hidden, gate, activation=Activation.SILU)
        return graph.matmul(graph.mul(gate, graph.matmul(hidden, up)), down)

    # Routed experts are a single fused 3-D checkpoint tensor, but the graph
    # must declare the flattened [experts*rows, K] form: the loader takes
    # shape[0]/shape[1] straight from the node and the MoE kernel addresses
    # expert e at row e*rows with row stride K. Declaring [experts, rows, K]
    # would set shape[1]=rows instead of K and index wrongly. The bytes are
    # identical either way, so this is a metadata contract, not a layout one.
    experts_gate_up = _fused_experts(
        graph, streamed, index, f"layer_{layer}_ffn_experts_gate_up",
        f"{prefix}.experts.gate_up_proj",
        (experts * 2 * moe_intermediate, hidden_size), 2 * moe_intermediate)
    experts_down = _fused_experts(
        graph, streamed, index, f"layer_{layer}_ffn_experts_down",
        f"{prefix}.experts.down_proj",
        (experts * hidden_size, moe_intermediate), hidden_size)
    router = _dense_weight(
        graph, streamed, index, f"layer_{layer}_ffn_router",
        f"{prefix}.gate.weight", (experts, hidden_size))
    router_bias = None
    # The reference checkpoint names this parameter without a `.weight` suffix
    # (it is a bare bias, not a Linear child), so probing for the suffixed form
    # silently skips it and the router loses the bias it selects on.
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
        # Hy4 scores experts with a sigmoid over the full expert set, and
        # selects on sigmoid(logit) + e_score_correction_bias while weighting
        # by the un-biased sigmoid. That is ROUTER_SCORE_SIGMOID (= 1), not the
        # 0 default, which would run softmax and pick different experts -- the
        # router is the whole point of having the bias at all.
        router_score_func=ROUTER_SCORE_SIGMOID,
        norm_topk_prob=bool(cfg["norm_topk_prob"]),
        has_shared_expert=True,
        shared_expert_has_gate=False,
        routed_scaling_factor=float(cfg["routed_scaling_factor"]),
        swiglu_limit=float(cfg.get("swiglu_limit", 0.0)))


def build_graph(model_dir: str | Path, cfg: dict, seq_len: int = 1,
                n_ctx: int = 4096, is_prefill: bool = False,
                absorbed: bool = False,
                index: SafeTensorIndex | GgufIndex | None = None
                ) -> tuple[GraphBuilder, dict]:
    """Build one Hy4 graph (prefill when is_prefill, else decode)."""
    global _S
    _S = SEQ.bind(seq_len) if is_prefill else seq_len

    if index is None:
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
    n_tokens = graph.input("n_tokens", (1,), prec=Precision.INT32)

    # MLA caches are [head_width, n_ctx, heads], dim0-fast, matching the
    # layout the SDPA kernels and the engine's CacheMetadata expect. Dim 0
    # must be the real per-head width: the loader reads shape[0] as the head
    # dimension when sizing each cache, so a zero there yields a zero-length
    # buffer and the kernel faults past the mapping.
    qk_head_dim = int(cfg["qk_nope_head_dim"]) + rope_dim
    v_head_dim = int(cfg["v_head_dim"])
    caches = []
    for layer in range(layers):
        if absorbed:
            ck = graph.input(
                f"latent_cache{layer}",
                (int(cfg["kv_lora_rank"]) + rope_dim, n_ctx),
                prec=Precision.FP16)
            cv = None
        else:
            ck = graph.input(
                f"cache_k{layer}", (qk_head_dim, n_ctx, heads),
                prec=Precision.FP16)
            cv = graph.input(
                f"cache_v{layer}", (v_head_dim, n_ctx, heads),
                prec=Precision.FP16)
        caches.append([ck, cv])

    # The DSA indexer keeps its own key cache of `wk @ hidden` rows: FP32 and
    # indexed by index_head_dim, not the attention heads. It is a persistent
    # state input (like DeepSeek-V4's aux_state*), not a graph output, and is
    # only materialised on layers that run the indexer.
    #
    # Named `index_state*` rather than `aux_state*`: the loader keys
    # auxiliary states purely by the trailing digit (see
    # parse_persistent_input / auxiliary_states_), so `aux_state{layer}` would
    # collide with DeepSeek-V4's `layer * 16` numbering.
    index_dim = int(cfg["index_head_dim"])
    indexer_caches = {}
    for layer in range(layers):
        if indexer_types[layer] != INDEXER_FULL:
            continue
        indexer_caches[layer] = graph.input(
            f"index_state{layer}", (index_dim, n_ctx), prec=Precision.FP32)

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
            f"model.layers.{layer}.input_layernorm.weight", (hidden_size,))
        ck_in, cv_in = caches[layer]
        is_full = indexer_types[layer] == INDEXER_FULL

        # q_a norm weight is shared by the attention branch and the indexer,
        # which both consume the same normed low-rank query.
        q_a_norm = _fp32_weight(
            graph, streamed, index, f"layer_{layer}_attn_q_a_norm",
            f"model.layers.{layer}.self_attn.q_a_layernorm.weight", (q_rank,))
        wq_a = _dense_weight(
            graph, streamed, index, f"layer_{layer}_attn_q_a_proj_weight",
            f"model.layers.{layer}.self_attn.q_a_proj.weight",
            (q_rank, hidden_size))

        # `selected` carries the indexer result back out of the branch
        # closure so the next layer can reuse it.
        selected: list[int] = []

        def attention_branch(reduced: int, *, current=layer,
                             norm=attn_norm, ck=ck_in, cv=cv_in,
                             full=is_full, wq=wq_a, qn=q_a_norm,
                             icache=indexer_caches.get(layer)) -> int:
            normalized = graph.rms_norm(reduced, norm, eps)
            q_lora = graph.rms_norm(graph.matmul(normalized, wq), qn, eps)
            if full:
                top_k = _add_indexer(
                    graph, streamed, index, cfg, current, normalized,
                    q_lora, position, n_tokens, icache, n_ctx)
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
                q_lora, position, cos, sin, mask, ck, cv, top_k, absorbed)

        state = _hc_branch(
            graph, streamed, index, cfg, layer, "attn", state,
            attention_branch)
        if selected:
            last_top_k = selected[0]

        ffn_norm = _fp32_weight(
            graph, streamed, index, f"layer_{layer}_ffn_norm",
            f"model.layers.{layer}.post_attention_layernorm.weight",
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

    # Engine boundary weights. Every graph needs both, and the loader resolves
    # them by name, so they are registered once per graph even though the
    # prefill and decode graphs share the same streamed entry.
    vocab = int(cfg["vocab_size"])
    embed_ref = "embed_tokens"
    head_ref = "lm_head"
    _dense_weight(
        graph, streamed, index, embed_ref,
        "model.embed_tokens.weight", (vocab, hidden_size))
    _dense_weight(
        graph, streamed, index, head_ref,
        "lm_head.weight", (vocab, hidden_size))
    return graph, streamed


def convert(model_dir: str, output_path: str,
            prefill_seq_len: int = 32, n_ctx: int = 16384,
            absorbed: bool = False, layers: int | None = None,
            tokenizer_dir: str | None = None,
            gguf_py_path: str | None = None) -> None:
    """Convert a Hy4-preview checkpoint into a .mollm package.

    Safetensors and one-layer GGUF diagnostic packages use FP16 weights.
    The GGUF route decodes quantized blocks while writing the package.
    """
    global _GGUF_PY_PATH
    model_dir_path = Path(model_dir)
    is_gguf = model_dir_path.is_file() and model_dir_path.suffix == ".gguf"
    if is_gguf:
        _, index = open_gguf(model_dir_path)
        cfg = config_from_gguf(index.gguf)
        if not absorbed:
            raise ValueError("GGUF diagnostic packages require --absorbed")
        if tokenizer_dir is None:
            raise ValueError("GGUF packages require --tokenizer-dir")
        _GGUF_PY_PATH = gguf_py_path
    else:
        index = None
        cfg = json.loads((model_dir_path / "config.json").read_text())
    if cfg.get("model_type") != "hy_v4":
        raise ValueError(
            f"checkpoint is not Hy4: model_type={cfg.get('model_type')!r}")
    full_layers = int(cfg["num_hidden_layers"])
    if layers is not None:
        if not 1 <= layers <= full_layers:
            raise ValueError(f"--layers must be in 1..{full_layers}")
        cfg["num_hidden_layers"] = layers
        cfg["indexer_types"] = cfg["indexer_types"][:layers]
        cfg["mlp_layer_types"] = cfg["mlp_layer_types"][:layers]
    if is_gguf and int(cfg["num_hidden_layers"]) != 1:
        raise ValueError("GGUF FP16 diagnostic conversion currently supports --layers 1")
    assets_dir = Path(tokenizer_dir) if tokenizer_dir else model_dir_path
    for asset in ("tokenizer.json", "chat_template.jinja"):
        if not (assets_dir / asset).is_file():
            raise FileNotFoundError(assets_dir / asset)

    # One graph per execution shape. The weight registry is shared, so a
    # weight requested by both graphs is streamed into the package once.
    prefill, prefill_weights = build_graph(
        model_dir_path, cfg, prefill_seq_len, n_ctx,
        is_prefill=True, absorbed=absorbed, index=index)
    decode, decode_weights = build_graph(
        model_dir_path, cfg, 1, n_ctx,
        is_prefill=False, absorbed=absorbed, index=index)
    streamed = dict(prefill_weights)
    streamed.update(decode_weights)

    num_layers = int(cfg["num_hidden_layers"])
    metadata = {
        "model_name": "Hy4-preview-GGUF-diagnostic" if is_gguf else "Hy4-preview",
        "architecture": "hy-v4",
        "num_layers": num_layers,
        "hidden_size": int(cfg["hidden_size"]),
        "num_heads": int(cfg["num_attention_heads"]),
        "num_kv_heads": 1 if absorbed else int(cfg["num_attention_heads"]),
        "head_dim": int(cfg["qk_head_dim"]),
        "rope_dim": int(cfg["qk_rope_head_dim"]),
        "rope_theta": float(
            cfg.get("rope_parameters", {}).get("rope_theta", 1e7)),
        "prefill_seq_len": prefill_seq_len,
        "n_ctx": n_ctx,
        "vocab_size": int(cfg["vocab_size"]),
        "num_experts": int(cfg["n_routed_experts"]),
        "num_experts_per_tok": int(cfg["num_experts_per_tok"]),
        "first_k_dense_replace": int(cfg.get("first_k_dense_replace", 1)),
        "n_shared_experts": int(cfg.get("n_shared_experts", 0) or 0),
        "hc_mult": int(cfg["hc_mult"]),
        "indexer_types": list(cfg["indexer_types"]),
        "quantization": "gguf-dequant-fp16" if is_gguf else "fp16",
    }

    # A directory is required by save_package even when every weight is
    # streamed: the graphs' weight nodes carry their own names.
    with tempfile.TemporaryDirectory(prefix="mollm_hy4_") as empty_dir:
        save_package(
            output_path, prefill, decode, empty_dir, metadata,
            tokenizer_path=str(assets_dir / "tokenizer.json"),
            jinja_path=str(assets_dir / "chat_template.jinja"),
            streamed_weights=streamed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("output")
    parser.add_argument("--prefill-seq-len", type=int, default=32)
    parser.add_argument("--n-ctx", type=int, default=16384)
    parser.add_argument("--absorbed", action="store_true",
                        help="use latent-cache MLA attention")
    parser.add_argument("--layers", type=int,
                        help="export only the first N layers")
    parser.add_argument("--tokenizer-dir",
                        help="tokenizer and chat template directory for GGUF")
    parser.add_argument("--gguf-py",
                        help="path to llama.cpp/gguf-py for K-quant decoding")
    args = parser.parse_args()
    convert(args.model_dir, args.output,
            args.prefill_seq_len, args.n_ctx, args.absorbed,
            args.layers, args.tokenizer_dir, args.gguf_py)
