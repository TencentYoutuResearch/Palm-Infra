"""Text graph builder for the experimental Qwen4 architecture.

The first bring-up stage intentionally targets contexts up to the QSA token
budget, where QSA selects every visible token and is exactly equivalent to the
existing gated GQA path. PLE export/lookup and native NVFP4 expert storage are
added separately so the reusable GDN, MoE and attention graph can be validated
without coupling all new mechanisms at once.
"""

from __future__ import annotations

import os
import math
import json
import tempfile
import copy
from pathlib import Path

from transpile import GraphBuilder, Precision, save_package
from qwen35 import _build_full_attn_layer, _build_linear_attn_layer
from model_metadata import infer_hf_model_name
from safetensors_stream import (
    SafeTensorIndex,
    aggregate_nvfp4_experts,
    aggregate_nvfp4_input_scales,
    aggregate_fp16_experts,
    concatenate_fp16_row_slices,
    concatenate_fp16_streamed_weights,
    dense_streamed_weight,
    fp32_plus_one_streamed_weight,
    fp32_streamed_weight,
    integer_streamed_weight,
    quantize_w4g32_streamed_weight,
    quantize_w8pc_streamed_weight,
    raw_u8_streamed_weight,
)


def _ref(name: str) -> str:
    return f"./{name}.weights"


def _weight(g: GraphBuilder, weights_dir: str, name: str, shape: tuple,
            precision: Precision = Precision.FP16) -> int:
    return g.weight(os.path.join(weights_dir, f"{name}.weights"),
                    shape, precision)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in
               range(3, math.isqrt(value) + 1, 2))


def ple_table_layout(tc: dict, ple_layer_index: int = 0):
    """Return the official per-head prime vocabularies and flat offsets."""
    ngram_heads = ((int(tc["ngram_size"]) - 1) *
                   int(tc["heads_per_ngram"]))
    current = int(tc["ngram_vocab_size_base"]) - 1
    # A later PLE layer continues the global prime sequence. The released
    # checkpoint has one PLE layer, but retaining the index keeps this generic.
    wanted_end = (ple_layer_index + 1) * ngram_heads
    all_sizes = []
    while len(all_sizes) < wanted_end:
        current += 1
        while not _is_prime(current):
            current += 1
        all_sizes.append(current)
    sizes = all_sizes[ple_layer_index * ngram_heads:wanted_end]
    offsets = []
    total = 0
    for size in sizes:
        offsets.append(total)
        total += size
    divisor = int(tc["make_ngram_vocab_size_divisible_by"])
    padded = math.ceil(total / divisor) * divisor
    return sizes, offsets, padded


def _add_ple(g: GraphBuilder, residual: int, token_ids: int,
             layer_idx: int, ple_layer_index: int, weights_dir: str,
             tc: dict, eps: float) -> int:
    hidden = int(tc["hidden_size"])
    hc_count = int(tc["hc_count"])
    wide = hidden * hc_count
    ngram_size = int(tc["ngram_size"])
    heads_per_ngram = int(tc["heads_per_ngram"])
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    embedding_dim = int(tc["ple_embed_dim"])
    row_dim = embedding_dim // ngram_heads
    sizes, offsets, table_rows = ple_table_layout(tc, ple_layer_index)
    prefix = f"model_language_model_layers_{layer_idx}_ple"

    # The extra history element is an initialization flag. Engine auxiliary
    # states are zeroed on reset; PLE_LOOKUP replaces the history with EOS on
    # its first call and then keeps it across prefill chunks/decode.
    history = g.input(f"aux_state{2 * ple_layer_index}",
                      (ngram_size,), prec=Precision.INT32)
    table = _weight(
        g, weights_dir, f"{prefix}_ple_embedding_ngram_embedding_weight",
        (row_dim, table_rows), Precision.RAW_U8)
    table_scale = _weight(
        g, weights_dir,
        f"{prefix}_ple_embedding_ngram_embedding_weight_scale",
        (1,), Precision.FP32)
    vocab_sizes = _weight(
        g, weights_dir,
        f"{prefix}_ple_embedding_ngram_heads_vocab_sizes",
        (ngram_heads,), Precision.INT32)
    head_offsets = _weight(
        g, weights_dir,
        f"{prefix}_ple_embedding_ngram_heads_offsets",
        (ngram_heads,), Precision.INT32)
    embeddings = g.ple_lookup(
        token_ids, history, table, table_scale, vocab_sizes, head_offsets,
        embedding_dim, ngram_size, heads_per_ngram,
        int(tc["eos_token_id"]), int(tc["vocab_size"]),
        ple_layer_index, int(tc.get("seed", 1234)))

    key_proj = _weight(g, weights_dir, f"{prefix}_key_proj_weight",
                       (wide, embedding_dim))
    value_proj = _weight(g, weights_dir, f"{prefix}_value_proj_weight",
                         (hidden, embedding_dim))
    norm_key = _weight(g, weights_dir, f"{prefix}_norm_key_weight",
                       (wide,), Precision.FP32)
    norm_query = _weight(g, weights_dir, f"{prefix}_norm_query_weight",
                         (wide,), Precision.FP32)
    norm_conv = _weight(g, weights_dir, f"{prefix}_norm_conv_weight",
                        (wide,), Precision.FP32)
    key = g.group_rms_norm(g.matmul(embeddings, key_proj),
                           norm_key, hidden, eps)
    query = g.group_rms_norm(residual, norm_query, hidden, eps)
    value = g.matmul(embeddings, value_proj)
    gated = g.ple_gate(query, key, value, hidden, hc_count)
    gated_normed = g.group_rms_norm(gated, norm_conv, hidden, eps)

    kernel_size = int(tc["ple_conv_kernel_size"])
    dilation = ngram_size
    state_len = (kernel_size - 1) * dilation
    conv_state = g.input(f"aux_state{2 * ple_layer_index + 1}",
                         (wide, state_len), prec=Precision.FP32)
    conv_weight = _weight(g, weights_dir, f"{prefix}_conv1d_weight",
                          (wide, kernel_size), Precision.FP32)
    convolved = g.ple_dilated_conv(
        gated_normed, conv_weight, conv_state, kernel_size, dilation)
    return g.add(residual, g.add(gated, convolved))


def _gr_read(g: GraphBuilder, state: int, prefix: str, weights_dir: str,
             hidden_size: int, hc_count: int, hc_lowrank: int,
             eps: float, with_injection: bool) -> tuple[int, int | None]:
    """Read one 2560-wide branch from Qwen's four residual streams."""
    wide = hidden_size * hc_count
    norm = _weight(g, weights_dir, f"{prefix}_hc_norm_weight",
                   (wide,), Precision.FP32)
    down = _weight(g, weights_dir, f"{prefix}_input_mix_weight_down_weight",
                   (hc_lowrank, wide))
    up = _weight(g, weights_dir, f"{prefix}_input_mix_weight_up_weight",
                 (wide, hc_lowrank))

    normalized = g.group_rms_norm(state, norm, hidden_size, eps)
    lowrank = g.scalar_mul(g.matmul(normalized, down), 1.0 / hc_count)
    lowrank = g.silu(lowrank)
    read_gates = g.sigmoid(g.matmul(lowrank, up))
    mixed = g.gr_reduce(normalized, read_gates, hidden_size, hc_count)

    if not with_injection:
        return mixed, None
    inject_weight = _weight(
        g, weights_dir, f"{prefix}_block_inject_weight_weight",
        (hc_count, wide))
    injection = g.scalar_mul(
        g.sigmoid(g.scalar_mul(
            g.matmul(normalized, inject_weight), 1.0 / hc_count)),
        2.0)
    return mixed, injection


def _gr_branch(g: GraphBuilder, state: int, prefix: str, weights_dir: str,
               hidden_size: int, hc_count: int, hc_lowrank: int,
               eps: float, branch_builder) -> int:
    mixed, injection = _gr_read(
        g, state, prefix, weights_dir, hidden_size, hc_count, hc_lowrank,
        eps, with_injection=True)
    branch = branch_builder(mixed)
    assert injection is not None
    return g.gr_inject(branch, state, injection, hidden_size, hc_count)


def _build_mtp_input_fusion(g: GraphBuilder, target_hidden: int,
                          token_hidden: int, weights_dir: str,
                          hidden_size: int, hc_count: int,
                          eps: float) -> int:
    """Global HC normalization, shared per-stream projection, token broadcast.

    Norm files contain the effective Gemma gamma (checkpoint gamma + 1),
    like the existing Qwen4Exp norm exports. Projection files are separate
    H-by-H matrices; no expanded HC-by-HC weight is materialized.
    """
    from transpile import SEQ
    seq = SEQ.bind(g._nodes[token_hidden].out_shape[1])
    if hc_count <= 1:
        raise ValueError("Qwen4Exp MTP shared-residual fusion requires hc_count > 1")
    wide = hidden_size * hc_count
    embedding_norm = _weight(g, weights_dir, "mtp_pre_fc_norm_embedding_weight",
                             (hidden_size,), Precision.FP32)
    hidden_norm = _weight(g, weights_dir, "mtp_pre_fc_norm_hidden_weight",
                          (wide,), Precision.FP32)
    embedding_fc = _weight(g, weights_dir, "mtp_fc_embedding_weight",
                           (hidden_size, hidden_size))
    hidden_fc = _weight(g, weights_dir, "mtp_fc_hidden_weight",
                        (hidden_size, hidden_size))
    embedding = g.matmul(g.rms_norm(token_hidden, embedding_norm, eps), embedding_fc)
    normalized = g.rms_norm(target_hidden, hidden_norm, eps)
    per_stream = g.reshape(normalized, (hidden_size, hc_count * seq))
    projected = g.matmul(per_stream, hidden_fc)
    projected = g.reshape(projected, (wide, seq))
    return g.add(projected, g.tile(embedding, (hc_count, 1)))


def build_graph(weights_dir: str, cfg: dict, seq_len: int = 1,
                n_ctx: int = 2048, is_prefill: bool = False,
                export_mtp_state: bool = False,
                _mtp_mode: bool = False, enable_qsa: bool = False) -> GraphBuilder:
    """Build a text graph; opt-in QSA requires native CUDA and extra weights."""
    g = GraphBuilder()
    tc = cfg["text_config"] if "text_config" in cfg else cfg

    hidden_size = int(tc["hidden_size"])
    num_layers = int(tc["num_hidden_layers"])
    layer_types = tc["layer_types"]
    eps = float(tc.get("rms_norm_eps", 1e-6))
    hc_count = int(tc.get("hc_count", 4))
    hc_lowrank = int(tc["hc_lowrank"])
    qsa_budget = int(tc.get("indexer_budget", 2048))
    if not enable_qsa and n_ctx > qsa_budget and any(t == "full_attention" for t in layer_types):
        raise ValueError(
            f"Qwen4Exp bring-up graph supports n_ctx <= QSA budget "
            f"({qsa_budget}); got {n_ctx}")

    rope = tc["rope_parameters"]
    rope_theta = float(rope["rope_theta"])
    head_dim = int(tc["head_dim"])
    rope_dim = int(head_dim * float(rope["partial_rotary_factor"]))
    num_heads = int(tc["num_attention_heads"])
    num_kv_heads = int(tc["num_key_value_heads"])
    if enable_qsa:
        index_heads = int(tc.get("indexer_n_heads", 4))
        index_dim = int(tc.get("indexer_head_dim", 128))
        index_ratio = int(tc.get("indexer_compress_ratio", 4))
        if (_mtp_mode or int(tc.get("indexer_kv_heads", 1)) != 1 or
                not 1 <= n_ctx <= 262144 or not 1 <= index_heads <= 16 or
                not rope_dim <= index_dim <= 256 or not 1 <= index_ratio <= 16 or
                qsa_budget < 1 or qsa_budget % index_ratio or qsa_budget + index_ratio - 1 > 4096):
            raise ValueError("unsupported native QSA geometry/context/MTP")
        g.metadata["qsa"] = "contiguous-causal-fp32-v1"
    linear_num_heads = int(tc["linear_num_key_heads"])
    linear_num_v_heads = int(tc["linear_num_value_heads"])
    linear_k_dim = int(tc["linear_key_head_dim"])
    linear_v_dim = int(tc["linear_value_head_dim"])
    conv_kernel = int(tc["linear_conv_kernel_dim"])
    moe_intermediate = int(tc["moe_intermediate_size"])
    shared_intermediate = int(tc["shared_expert_intermediate_size"])
    num_experts = int(tc["num_experts"])
    top_k = int(tc["num_experts_per_tok"])

    g.set_model_config(
        rope_dim=rope_dim, rope_theta=rope_theta,
        hidden_size=hidden_size, num_layers=num_layers,
        vocab_size=int(tc["vocab_size"]), model_type="qwen4_exp")
    embed_shape = (int(tc["vocab_size"]), hidden_size)
    _weight(g, weights_dir, "embed_tokens", embed_shape)
    _weight(g, weights_dir, "lm_head", embed_shape)

    if is_prefill:
        from transpile import DimExpr
        const, seq = DimExpr.const(), DimExpr.seq()
        hidden_dynamic = (const, seq, const, const)
        mask_dynamic = cos_dynamic = sin_dynamic = hidden_dynamic
    else:
        hidden_dynamic = mask_dynamic = cos_dynamic = sin_dynamic = None
    hidden = g.input("hidden", (hidden_size, seq_len), dynamic=hidden_dynamic)
    mask = g.input("mask", (1, seq_len), dynamic=mask_dynamic)
    cos = g.input("cos", (rope_dim // 2, seq_len), dynamic=cos_dynamic)
    sin = g.input("sin", (rope_dim // 2, seq_len), dynamic=sin_dynamic)
    ple_ids = list(tc.get("ple_layer_ids", []))
    token_ids = None
    if ple_ids:
        if is_prefill:
            from transpile import DimExpr
            token_dynamic = (DimExpr.seq(), DimExpr.const(),
                             DimExpr.const(), DimExpr.const())
        else:
            token_dynamic = None
        token_ids = g.input("token_ids", (seq_len,), prec=Precision.INT32,
                            dynamic=token_dynamic)

    caches = []
    for layer, layer_type in enumerate(layer_types):
        if layer_type == "full_attention":
            cache_k = g.input(f"cache_k{layer}",
                              (head_dim, n_ctx, num_kv_heads),
                              prec=Precision.FP16)
            cache_v = g.input(f"cache_v{layer}",
                              (head_dim, n_ctx, num_kv_heads),
                              prec=Precision.FP16)
            caches.append(("kv", cache_k, cache_v))
        else:
            state = g.input(
                f"gdn_state{layer}",
                (linear_v_dim, linear_k_dim, linear_num_v_heads),
                prec=Precision.FP32)
            qkv = (2 * linear_num_heads * linear_k_dim +
                   linear_num_v_heads * linear_v_dim)
            conv = g.input(f"gdn_conv{layer}",
                           (qkv, conv_kernel - 1), prec=Precision.FP16)
            caches.append(("gdn", state, conv))

    if _mtp_mode:
        target_hidden = g.input("target_hidden", (hidden_size * hc_count, seq_len),
                                dynamic=hidden_dynamic)
        residual = _build_mtp_input_fusion(
            g, target_hidden, hidden, weights_dir, hidden_size, hc_count, eps)
    else:
        residual = g.tile(hidden, (hc_count, 1))
    for layer, layer_type in enumerate(layer_types):
        if layer + 1 in ple_ids:
            assert token_ids is not None
            residual = _add_ple(
                g, residual, token_ids, layer, ple_ids.index(layer + 1),
                weights_dir, tc, eps)
        cache_kind, first_cache, second_cache = caches[layer]
        pfx = f"model_language_model_layers_{layer}"

        def attention_branch(x: int, *, current=layer,
                             kind=layer_type,
                             ckind=cache_kind,
                             c0=first_cache, c1=second_cache) -> int:
            if kind == "linear_attention":
                return _build_linear_attn_layer(
                    g, x, current, weights_dir, c0, c1, eps, seq_len,
                    linear_num_heads, linear_k_dim, linear_v_dim,
                    linear_num_v_heads, conv_kernel, hidden_size,
                    is_prefill=is_prefill,
                    output_gate_type=(tc.get("output_gate_type") or
                                      tc.get("hidden_act", "silu")))
            selected = None
            if enable_qsa:
                import numpy as np
                ipfx = f"model_language_model_layers_{current}_self_attn_indexer"
                projection = g.matmul(x, _weight(g, weights_dir, f"{ipfx}_index_qk_proj_weight",
                                                ((index_heads + 1) * index_dim, hidden_size)))
                q_norm = _weight(g, weights_dir, f"{ipfx}_q_layernorm_weight", (index_dim,), Precision.FP32)
                k_norm = _weight(g, weights_dir, f"{ipfx}_k_layernorm_weight", (index_dim,), Precision.FP32)
                frequency = g.constant(1 / np.power(np.float32(rope_theta),
                    np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim))
                state_id = 2 * len(ple_ids) + 2 * current
                raw = g.input(f"aux_state{state_id}", (index_dim, n_ctx), prec=Precision.FP32)
                pooled = g.input(f"aux_state{state_id + 1}", (index_dim, max(1, n_ctx // index_ratio)), prec=Precision.FP32)
                selected = g.qsa_indexer(projection, q_norm, k_norm, frequency, raw, pooled,
                    c0, mask, heads=index_heads, dimension=index_dim, rope_dim=rope_dim,
                    ratio=index_ratio, budget=qsa_budget, eps=eps)
            return _build_full_attn_layer(
                g, x, current, weights_dir, cos, sin, mask, c0, c1,
                eps, seq_len, rope_dim, num_heads, num_kv_heads,
                head_dim, hidden_size, is_prefill=is_prefill, selected_indices=selected)

        residual = _gr_branch(
            g, residual, f"{pfx}_attn_hyper_connection", weights_dir,
            hidden_size, hc_count, hc_lowrank, eps, attention_branch)

        def moe_branch(x: int, *, current=layer) -> int:
            mlp = f"model_language_model_layers_{current}_mlp"
            router = _weight(g, weights_dir, f"{mlp}_gate_weight",
                             (num_experts, hidden_size))
            gate_up = _weight(
                g, weights_dir, f"{mlp}_experts_gate_up_proj",
                (num_experts * 2 * moe_intermediate, hidden_size))
            down = _weight(
                g, weights_dir, f"{mlp}_experts_down_proj",
                (num_experts * hidden_size, moe_intermediate))
            if tc.get("export_nvfp4_activation_calibration", False):
                # Package collection walks constant nodes, not every supplied
                # streamed weight. These metadata constants intentionally do
                # not alter the MoE operator's inputs or arithmetic.
                _weight(g, weights_dir, f"{mlp}_experts_gate_up_input_scales",
                        (num_experts, 2), Precision.FP32)
                _weight(g, weights_dir, f"{mlp}_experts_down_input_scales",
                        (num_experts, 1), Precision.FP32)
            shared_gate = _weight(
                g, weights_dir, f"{mlp}_shared_expert_gate_proj_weight",
                (shared_intermediate, hidden_size))
            shared_up = _weight(
                g, weights_dir, f"{mlp}_shared_expert_up_proj_weight",
                (shared_intermediate, hidden_size))
            shared_down = _weight(
                g, weights_dir, f"{mlp}_shared_expert_down_proj_weight",
                (hidden_size, shared_intermediate))
            shared_expert_gate = _weight(
                g, weights_dir, f"{mlp}_shared_expert_gate_weight",
                (1, hidden_size))
            return g.moe(
                x, router, gate_up, down,
                shared_gate, shared_up, shared_down, shared_expert_gate,
                hidden_size=hidden_size, num_experts=num_experts,
                top_k=top_k, intermediate_size=moe_intermediate,
                shared_intermediate_size=shared_intermediate,
                norm_topk_prob=bool(tc.get("norm_topk_prob", True)))

        residual = _gr_branch(
            g, residual, f"{pfx}_mlp_hyper_connection", weights_dir,
            hidden_size, hc_count, hc_lowrank, eps, moe_branch)

    if export_mtp_state:
        # An explicit graph output retains the pre-collapse HC stream through
        # execution/liveness. Keep it before the normal logits-hidden output.
        state_output = g.scalar_mul(residual, 1.0)
        g.metadata["mtp_state_output_node"] = str(state_output)
    collapsed, _ = _gr_read(
        g, residual, "model_language_model_hyper_connection_mixer",
        weights_dir, hidden_size, hc_count, hc_lowrank, eps,
        with_injection=False)
    return g


def build_mtp_graph(weights_dir: str, cfg: dict, seq_len: int = 8,
                    n_ctx: int = 2048) -> GraphBuilder:
    """Build the single-layer Qwen4Exp draft; weight export is separate.

    Uses shared token embeddings/head, no PLE, and independent MTP attention,
    MoE and hyper-connection weights. Never mutates the main model config.
    Contexts above QSA budget and non-final target-layer taps remain unsupported.
    """
    draft_cfg = copy.deepcopy(cfg)
    tc = draft_cfg["text_config"] if "text_config" in draft_cfg else draft_cfg
    mtp = tc.get("mtp", {})
    if seq_len < 1 or n_ctx < 1:
        raise ValueError("Qwen4Exp MTP needs positive sequence and context lengths")
    if int(mtp.get("num_hidden_layers", tc.get("mtp_num_hidden_layers", 1))) != 1:
        raise ValueError("Qwen4Exp MTP currently supports exactly one draft layer")
    if mtp.get("mtp_use_hidden_state_from_layer") is not None:
        raise ValueError("Qwen4Exp MTP currently uses the final target HC state")
    if tc.get("mtp_use_dedicated_embeddings", False):
        raise ValueError("Qwen4Exp MTP currently requires shared target embeddings")
    if mtp.get("layer_types", ["full_attention"]) != ["full_attention"]:
        raise ValueError("Qwen4Exp MTP currently requires a full-attention draft layer")
    if float(mtp.get("rope_theta", tc["rope_parameters"]["rope_theta"])) != float(tc["rope_parameters"]["rope_theta"]):
        raise ValueError("Qwen4Exp MTP currently shares target RoPE theta in the engine")
    tc["num_hidden_layers"] = 1
    tc["layer_types"] = ["full_attention"]
    tc["ple_layer_ids"] = []
    tc["export_nvfp4_activation_calibration"] = False
    if "rope_theta" in mtp:
        tc["rope_parameters"]["rope_theta"] = mtp["rope_theta"]
    graph = build_graph(weights_dir, draft_cfg, seq_len, n_ctx,
                        is_prefill=True, export_mtp_state=True, _mtp_mode=True)
    for node in graph._nodes:
        if node.weight_path:
            for old, new in (
                ("model_language_model_layers_0_", "mtp_layers_0_"),
                ("model_language_model_hyper_connection_mixer_", "mtp_hyper_connection_mixer_"),
            ):
                node.weight_path = node.weight_path.replace(old, new)
                node.params_str = [s.replace(old, new) for s in node.params_str]
            # The draft exporter deliberately materializes FP16 projections.
            # Reused attention builders may declare target FP8/W4 constants;
            # do not carry those storage declarations into this FP16 bundle.
            if node.out_prec != Precision.FP32:
                node.out_prec = Precision.FP16
    return graph


def build_streamed_weights(model_dir: str | Path, cfg: dict,
                           num_layers: int | None = None, *,
                           _index=None, _expert_precision="nvfp4",
                           _dense_precision="w4", _include_shared_weights=True):
    """Map a Qwen4-Exp checkpoint into graph-native streamed weights."""
    model_dir = Path(model_dir)
    index = _index if _index is not None else SafeTensorIndex(model_dir)
    if _expert_precision not in ("nvfp4", "fp16") or _dense_precision not in ("w4", "fp16"):
        raise ValueError("unsupported streamed export precision")
    tc = cfg["text_config"] if "text_config" in cfg else cfg
    layers = int(tc["num_hidden_layers"]) if num_layers is None else num_layers
    hidden = int(tc["hidden_size"])
    heads = int(tc["num_attention_heads"])
    head_dim = int(tc["head_dim"])
    kv_heads = int(tc["num_key_value_heads"])
    linear_heads = int(tc["linear_num_key_heads"])
    linear_v_heads = int(tc["linear_num_value_heads"])
    linear_k = int(tc["linear_key_head_dim"])
    linear_v = int(tc["linear_value_head_dim"])
    experts = int(tc["num_experts"])
    moe_intermediate = int(tc["moe_intermediate_size"])
    shared_intermediate = int(tc["shared_expert_intermediate_size"])
    hc_count = int(tc["hc_count"])
    wide = hidden * hc_count
    streamed = {}

    def put(name: str, weight):
        streamed[_ref(name)] = weight

    if _include_shared_weights:
        put("embed_tokens", dense_streamed_weight(
            index, "model.language_model.embed_tokens.weight"))
        put("lm_head", quantize_w8pc_streamed_weight(
            dense_streamed_weight(index, "lm_head.weight")))

    def dense(checkpoint_name: str, output_name: str | None = None):
        weight = dense_streamed_weight(index, checkpoint_name)
        put(output_name or checkpoint_name.replace(".", "_"),
            quantize_w4g32_streamed_weight(weight) if _dense_precision == "w4" else weight)

    def fp32(checkpoint_name: str, output_name: str | None = None,
             shape: tuple[int, ...] | None = None):
        put(output_name or checkpoint_name.replace(".", "_"),
            fp32_streamed_weight(index, checkpoint_name, shape))

    def norm(checkpoint_name: str, output_name: str | None = None,
             shape: tuple[int, ...] | None = None):
        put(output_name or checkpoint_name.replace(".", "_"),
            fp32_plus_one_streamed_weight(index, checkpoint_name, shape))

    for layer in range(layers):
        base = f"model.language_model.layers.{layer}"
        out = f"model_language_model_layers_{layer}"
        for branch in ("attn", "mlp"):
            hcp = f"{base}.{branch}_hyper_connection"
            hco = f"{out}_{branch}_hyper_connection"
            norm(f"{hcp}.hc_norm.weight", f"{hco}_hc_norm_weight")
            dense(f"{hcp}.input_mix_weight_down.weight",
                  f"{hco}_input_mix_weight_down_weight")
            dense(f"{hcp}.input_mix_weight_up.weight",
                  f"{hco}_input_mix_weight_up_weight")
            dense(f"{hcp}.block_inject_weight.weight",
                  f"{hco}_block_inject_weight_weight")

        if tc["layer_types"][layer] == "linear_attention":
            prefix = f"{base}.linear_attn"
            output = f"{out}_linear_attn"
            names = [
                f"{prefix}.in_proj_qkv.weight",
                f"{prefix}.in_proj_a.weight",
                f"{prefix}.in_proj_b.weight",
                f"{prefix}.in_proj_z.weight",
            ]
            fused_rows = (2 * linear_heads * linear_k +
                          linear_v_heads * linear_v +
                          2 * linear_v_heads +
                          linear_v_heads * linear_v)
            fused = concatenate_fp16_streamed_weights(index, names, (fused_rows, hidden))
            put(f"{output}_in_proj_weight",
                quantize_w4g32_streamed_weight(fused) if _dense_precision == "w4" else fused)
            fp32(f"{prefix}.conv1d.weight", f"{output}_conv1d_weight",
                 (2 * linear_heads * linear_k +
                  linear_v_heads * linear_v, int(tc["linear_conv_kernel_dim"])))
            fp32(f"{prefix}.A_log", f"{output}_A_log")
            fp32(f"{prefix}.dt_bias", f"{output}_dt_bias")
            # RMSNormGated uses its parameter directly, unlike Qwen RMSNorm.
            fp32(f"{prefix}.norm.weight", f"{output}_norm_weight")
            dense(f"{prefix}.out_proj.weight", f"{output}_out_proj_weight")
        else:
            prefix = f"{base}.self_attn"
            output = f"{out}_self_attn"
            q_name = f"{prefix}.q_proj.weight"
            slices = []
            for half in range(2):
                for head in range(heads):
                    slices.append((
                        q_name, (head * 2 + half) * head_dim, head_dim))
            slices.extend([
                (f"{prefix}.k_proj.weight", 0, kv_heads * head_dim),
                (f"{prefix}.v_proj.weight", 0, kv_heads * head_dim),
            ])
            qkv_rows = 2 * heads * head_dim + 2 * kv_heads * head_dim
            fused = concatenate_fp16_row_slices(index, slices, (qkv_rows, hidden))
            put(f"{output}_qkv_proj_weight",
                quantize_w4g32_streamed_weight(fused) if _dense_precision == "w4" else fused)
            norm(f"{prefix}.q_norm.weight", f"{output}_q_norm_weight")
            norm(f"{prefix}.k_norm.weight", f"{output}_k_norm_weight")
            dense(f"{prefix}.o_proj.weight", f"{output}_o_proj_weight")

        mlp = f"{base}.mlp"
        mlp_out = f"{out}_mlp"
        dense(f"{mlp}.gate.weight", f"{mlp_out}_gate_weight")
        for projection in ("gate_proj", "up_proj", "down_proj"):
            dense(f"{mlp}.shared_expert.{projection}.weight",
                  f"{mlp_out}_shared_expert_{projection}_weight")
        dense(f"{mlp}.shared_expert_gate.weight",
              f"{mlp_out}_shared_expert_gate_weight")
        expert_prefixes = [
            f"{mlp}.experts.{expert}" for expert in range(experts)]
        aggregate_experts = aggregate_nvfp4_experts if _expert_precision == "nvfp4" else aggregate_fp16_experts
        put(f"{mlp_out}_experts_gate_up_proj",
            aggregate_experts(index, [
                [f"{prefix}.gate_proj.weight",
                 f"{prefix}.up_proj.weight"]
                for prefix in expert_prefixes]))
        put(f"{mlp_out}_experts_down_proj",
            aggregate_experts(index, [
                [f"{prefix}.down_proj.weight"]
                for prefix in expert_prefixes]))
        # Preserve the original activation calibration for an optional native
        # W4A4 backend. Current W4/FP32-activation inference remains unchanged.
        input_scale_names = [f"{prefix}.{projection}.input_scale"
            for prefix in expert_prefixes
            for projection in ("gate_proj", "up_proj", "down_proj")]
        present = [index.contains(name) for name in input_scale_names]
        if any(present):
            if not all(present):
                raise ValueError("incomplete NVFP4 expert activation calibration")
            for output, projections in (("gate_up", ("gate_proj", "up_proj")),
                                        ("down", ("down_proj",))):
                put(f"{mlp_out}_experts_{output}_input_scales",
                    aggregate_nvfp4_input_scales(index, [
                        [f"{prefix}.{projection}.weight" for projection in projections]
                        for prefix in expert_prefixes]))

        if layer + 1 in tc.get("ple_layer_ids", []):
            ple = f"{base}.ple"
            ple_out = f"{out}_ple"
            ple_index = list(tc["ple_layer_ids"]).index(layer + 1)
            _, _, table_rows = ple_table_layout(tc, ple_index)
            row_dim = int(tc["ple_embed_dim"]) // (
                (int(tc["ngram_size"]) - 1) * int(tc["heads_per_ngram"]))
            shard_names = [
                f"{ple}.ple_embedding.ngram_embedding.shard_{shard}.weight"
                for shard in range(int(tc["split_ngram_parts"]))]
            put(f"{ple_out}_ple_embedding_ngram_embedding_weight",
                raw_u8_streamed_weight(
                    index, shard_names, (row_dim, table_rows)))
            fp32(f"{ple}.ple_embedding.ngram_embedding.weight_scale",
                 f"{ple_out}_ple_embedding_ngram_embedding_weight_scale")
            for field in ("ngram_heads_vocab_sizes", "ngram_heads_offsets"):
                put(f"{ple_out}_ple_embedding_{field}",
                    integer_streamed_weight(
                        index, f"{ple}.ple_embedding.{field}"))
            dense(f"{ple}.key_proj.weight", f"{ple_out}_key_proj_weight")
            dense(f"{ple}.value_proj.weight", f"{ple_out}_value_proj_weight")
            for field in ("norm_key", "norm_query", "norm_conv"):
                norm(f"{ple}.{field}.weight", f"{ple_out}_{field}_weight")
            fp32(f"{ple}.conv1d.weight", f"{ple_out}_conv1d_weight",
                 (wide, int(tc["ple_conv_kernel_size"])))

    mixer = "model.language_model.hyper_connection_mixer"
    mixer_out = "model_language_model_hyper_connection_mixer"
    norm(f"{mixer}.hc_norm.weight", f"{mixer_out}_hc_norm_weight")
    dense(f"{mixer}.input_mix_weight_down.weight",
          f"{mixer_out}_input_mix_weight_down_weight")
    dense(f"{mixer}.input_mix_weight_up.weight",
          f"{mixer_out}_input_mix_weight_up_weight")
    return streamed


def build_mtp_streamed_weights(model_dir: str | Path, cfg: dict):
    """MTP-only FP16 baseline descriptors, excluding shared embedding/head.

    Source experts use E4M3FN with BF16/F32 128x128 scales. Decoding is bounded
    to 128 rows at a time, not a full FP32 expert arena. These weights still
    require package integration and an appropriate native expert backend.
    """
    build_mtp_graph(".", cfg, seq_len=1, n_ctx=1)  # validate supported contract
    source = SafeTensorIndex(model_dir)
    def checkpoint_name(name):
        return name.replace("model.language_model.layers.0.", "mtp.layers.0.").replace(
            "model.language_model.hyper_connection_mixer.", "mtp.hyper_connection_mixer.")
    class MtpIndex:
        def tensor(self, name):
            return source.tensor(checkpoint_name(name))
        def contains(self, name):
            return source.contains(checkpoint_name(name))
    draft_cfg = copy.deepcopy(cfg)
    tc = draft_cfg["text_config"] if "text_config" in draft_cfg else draft_cfg
    tc["num_hidden_layers"] = 1
    tc["layer_types"] = ["full_attention"]
    tc["ple_layer_ids"] = []
    weights = build_streamed_weights(model_dir, draft_cfg, _index=MtpIndex(),
        _expert_precision="fp16", _dense_precision="fp16", _include_shared_weights=False)
    weights = {key.replace("model_language_model_layers_0_", "mtp_layers_0_").replace(
        "model_language_model_hyper_connection_mixer_", "mtp_hyper_connection_mixer_"): value
        for key, value in weights.items()}
    for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden"):
        weights[_ref(f"mtp_{name}_weight")] = fp32_plus_one_streamed_weight(
            source, f"mtp.{name}.weight")
    for name in ("fc_embedding", "fc_hidden"):
        weights[_ref(f"mtp_{name}_weight")] = dense_streamed_weight(source, f"mtp.{name}.weight")
    return weights


def convert_qwen4_exp(model_dir: str, output_path: str,
                      num_layers: int | None = None,
                      prefill_seq_len: int = 256, n_ctx: int = 2048,
                      quant: str = "w4g32"):
    """Convert native NVFP4 experts with offline W4G32 dense weights."""
    if quant.lower() != "w4g32":
        raise ValueError("Qwen4-Exp uses W4G32 dense weights")
    model_path = Path(model_dir)
    cfg = json.loads((model_path / "config.json").read_text())
    if cfg.get("model_type") != "qwen4_exp":
        raise ValueError("checkpoint is not Qwen4-Exp")
    cfg = dict(cfg)
    tc = dict(cfg["text_config"])
    total_layers = int(tc["num_hidden_layers"])
    if num_layers is None:
        num_layers = total_layers
    if num_layers <= 0 or num_layers > total_layers:
        raise ValueError(
            f"num_layers must be in [1, {total_layers}], got {num_layers}")
    tc["num_hidden_layers"] = num_layers
    tc["layer_types"] = list(tc["layer_types"][:num_layers])
    tc["ple_layer_ids"] = [
        layer for layer in tc.get("ple_layer_ids", [])
        if layer <= num_layers]
    cfg["text_config"] = tc

    streamed = build_streamed_weights(model_path, cfg, num_layers)
    tc["export_nvfp4_activation_calibration"] = any(
        name.endswith("_input_scales.weights") for name in streamed)
    prefill = build_graph(".", cfg, prefill_seq_len, n_ctx, True)
    decode = build_graph(".", cfg, 1, n_ctx, False)
    moe_layers = []
    for layer in range(num_layers):
        prefix = f"model_language_model_layers_{layer}_mlp"
        moe_layers.append({
            "layer": layer,
            "num_experts": int(tc["num_experts"]),
            "gate_up": {
                "weight": _ref(f"{prefix}_experts_gate_up_proj"),
                "rows_per_expert": 2 * int(tc["moe_intermediate_size"]),
                "cols": int(tc["hidden_size"]),
            },
            "down": {
                "weight": _ref(f"{prefix}_experts_down_proj"),
                "rows_per_expert": int(tc["hidden_size"]),
                "cols": int(tc["moe_intermediate_size"]),
            },
        })
        if tc["export_nvfp4_activation_calibration"]:
            for projection in ("gate_up", "down"):
                moe_layers[-1][projection]["input_scales_weight"] = _ref(
                    f"{prefix}_experts_{projection}_input_scales")
    metadata = {
        "model_name": infer_hf_model_name(
            model_path, cfg, "Qwen3.8-Flash-Next-NVFP4"),
        "architecture": "qwen4-exp",
        "num_layers": num_layers,
        "hidden_size": int(tc["hidden_size"]),
        "num_heads": int(tc["num_attention_heads"]),
        "num_kv_heads": int(tc["num_key_value_heads"]),
        "head_dim": int(tc["head_dim"]),
        "prefill_seq_len": prefill_seq_len,
        "n_ctx": n_ctx,
        "vocab_size": int(tc["vocab_size"]),
        "num_experts": int(tc["num_experts"]),
        "num_experts_per_tok": int(tc["num_experts_per_tok"]),
        "quantization": "native-nvfp4+dense-w4g32+lm-head-w8pc",
        "moe_expert_storage": {
            "version": 1,
            "layout": "expert_interleaved_v1",
            "num_experts": int(tc["num_experts"]),
            "layers": moe_layers,
        },
    }
    with tempfile.TemporaryDirectory(prefix="mollm_qwen4_exp_") as empty:
        save_package(
            output_path, prefill, decode, empty, metadata,
            tokenizer_path=str(model_path / "tokenizer.json"),
            jinja_path=str(model_path / "chat_template.jinja"),
            streamed_weights=streamed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert Qwen3.8-Flash-Next NVFP4 to .mollm")
    parser.add_argument("model_dir")
    parser.add_argument("output")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--prefill-seq-len", type=int, default=256)
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()
    convert_qwen4_exp(
        args.model_dir, args.output, args.layers,
        args.prefill_seq_len, args.n_ctx)
