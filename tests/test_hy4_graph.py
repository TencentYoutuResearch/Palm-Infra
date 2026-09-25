#!/usr/bin/env python3
"""Graph-structure tests for the Hy4-preview builder.

These run without a checkpoint: the weight loaders are replaced with recorders
that return correctly-shaped placeholder nodes, so what is asserted is the
*shape of the graph* — which ops are emitted, how many, with what shapes.

That is the level at which Hy4's structure can be checked before weights
exist. In particular the IndexCache contract (only "full" layers run the
indexer and carry its weights) is a property of the emitted graph, not of any
numerical result, so it is fully testable here.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import hy4
from transpile import OpType


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


class _RecordingIndex:
    """Stand-in for SafeTensorIndex: every tensor is considered present."""

    def contains(self, name):
        return True


def install_stubs():
    """Capture weight requests and return plausibly-shaped nodes."""
    captured = []

    def fp32(graph, streamed, index, label, source, shape):
        captured.append((label, source, tuple(shape), "fp32"))
        return graph.input(f"w_{label}", tuple(shape), prec=hy4.Precision.FP32)

    def dense(graph, streamed, index, label, source, shape):
        captured.append((label, source, tuple(shape), "dense"))
        return graph.input(f"w_{label}", tuple(shape), prec=hy4.Precision.FP16)

    def fused(graph, streamed, index, label, source, shape, rows_per_expert):
        captured.append((label, source, tuple(shape), "fused"))
        return graph.input(f"w_{label}", tuple(shape), prec=hy4.Precision.FP16)

    hy4._fp32_weight = fp32
    hy4._dense_weight = dense
    hy4._fused_experts = fused
    hy4.SafeTensorIndex = lambda path: _RecordingIndex()
    return captured


def tiny_cfg(layers=4, indexer_types=None):
    return {
        "model_type": "hy_v4",
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 4,
        "v_head_dim": 8,
        "q_lora_rank": 16,
        "kv_lora_rank": 8,
        "rms_norm_eps": 1e-5,
        "hc_mult": 4,
        "hc_eps": 1e-6,
        "hc_magnitude": 2.0,
        "num_hidden_layers": layers,
        "first_k_dense_replace": 1,
        "intermediate_size": 64,
        "moe_intermediate_size": 16,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.827,
        "vocab_size": 128,
        "swiglu_limit": 10.0,
        "indexer_types": indexer_types or ["full"] * layers,
        "index_n_heads": 2,
        "index_head_dim": 8,
        "index_topk": 8,
    }


def count_ops(graph):
    tally = {}
    for node in graph._nodes:
        tally[node.op_type] = tally.get(node.op_type, 0) + 1
    return tally


def build(cfg, **kwargs):
    captured = install_stubs()
    graph, _ = hy4.build_graph(".", cfg, **kwargs)
    return graph, captured


def test_layer_counts():
    """One attention + FFN branch per layer, each wrapped in iHC."""
    layers = 4
    graph, _ = build(tiny_cfg(layers=layers), seq_len=4, n_ctx=64,
                     is_prefill=True)
    tally = count_ops(graph)

    check(tally.get(OpType.SDPA) == layers,
          f"one SDPA per layer, got {tally.get(OpType.SDPA)}")
    check(tally.get(OpType.SIGMOID) == layers,
          f"one attention gate per layer, got {tally.get(OpType.SIGMOID)}")
    # iHC wraps both branches of every layer.
    check(tally.get(OpType.HC_PRE_ID) == 2 * layers,
          f"hc_pre_id per branch, got {tally.get(OpType.HC_PRE_ID)}")
    check(tally.get(OpType.HC_POST_ID) == 2 * layers,
          f"hc_post_id per branch, got {tally.get(OpType.HC_POST_ID)}")
    check(tally.get(OpType.HC_HEAD_ID) == 1,
          f"one head collapse, got {tally.get(OpType.HC_HEAD_ID)}")
    # The first layer keeps a dense FFN; the rest are MoE.
    dense_lead = 1
    check(tally.get(OpType.MOE) == layers - dense_lead,
          f"MoE on all but the dense-lead layers, got {tally.get(OpType.MOE)}")
    print("  PASS: per-layer op counts")


def test_indexer_only_on_full_layers():
    """IndexCache: shared layers must not run the indexer or carry weights.

    This is the property the whole IndexCache saving depends on, so it is
    asserted both on the emitted op count and on the requested weights.
    """
    types = ["full", "shared", "shared", "full", "shared"]
    graph, captured = build(tiny_cfg(layers=len(types), indexer_types=types),
                            seq_len=4, n_ctx=64, is_prefill=True)
    tally = count_ops(graph)

    expected = types.count("full")
    check(tally.get(OpType.HY4_INDEXER) == expected,
          f"HY4_INDEXER emitted {tally.get(OpType.HY4_INDEXER)} times, "
          f"expected one per full layer ({expected})")

    for layer, kind in enumerate(types):
        has_weights = any(
            label.startswith(f"layer_{layer}_indexer_")
            for label, _, _, _ in captured)
        want = kind == "full"
        check(has_weights == want,
              f"layer {layer} ({kind}): indexer weights={has_weights}, "
              f"expected {want}")
    print("  PASS: indexer restricted to full layers")


def test_shared_layer_reuses_selection():
    """A shared layer must consume the same node the full layer produced."""
    types = ["full", "shared", "shared"]
    graph, _ = build(tiny_cfg(layers=3, indexer_types=types),
                     seq_len=4, n_ctx=64, is_prefill=True)

    indexer_nodes = [
        node.id for node in graph._nodes
        if node.op_type == OpType.HY4_INDEXER]
    check(len(indexer_nodes) == 1, "exactly one indexer for one full layer")

    # Every sparse attention in the graph should take that one node as its
    # top-k input; otherwise the shared layers are not actually reusing.
    producer = indexer_nodes[0]
    consumers = 0
    for node in graph._nodes:
        if node.op_type != OpType.SDPA:
            continue
        if producer in node.inputs:
            consumers += 1
    check(consumers == len(types),
          f"all {len(types)} attention layers consume the cached selection, "
          f"got {consumers}")
    print("  PASS: shared layers reuse one cached selection")


def test_weight_shapes():
    """Shapes that differ from the DeepSeek-V4 iHC variant.

    The expert aggregates are the subtle one: the checkpoint stores them 3-D
    as [experts, rows, K], but the graph must declare [experts*rows, K]
    because the loader assigns shape[0]/shape[1] straight from the node and
    the kernel then indexes by row with stride K. Declaring the 3-D form
    compiles and produces the same byte count, but sets shape[1] to `rows`
    instead of K and reads the wrong elements.
    """
    cfg = tiny_cfg(layers=2)
    cfg["indexer_types"] = ["full", "shared"]
    graph, captured = build(cfg, seq_len=4, n_ctx=64, is_prefill=True)
    by_label = {label: shape for label, _, shape, _ in captured}

    hidden = cfg["hidden_size"]
    hc = cfg["hc_mult"]
    heads = cfg["num_attention_heads"]
    v_head = cfg["v_head_dim"]
    moe_i = cfg["moe_intermediate_size"]
    experts = cfg["n_routed_experts"]

    expected = {
        # 2*hc mixing rows: no Sinkhorn combination block.
        "layer_0_hc_attn_fn": (2 * hc, hc * hidden),
        "layer_0_hc_attn_scale": (2,),
        "layer_0_hc_attn_base": (2 * hc,),
        "layer_1_hc_mlp_fn": (2 * hc, hc * hidden),
        # hc rows for the head collapse.
        "hc_head_fn": (hc, hc * hidden),
        "hc_head_scale": (1,),
        "hc_head_base": (hc,),
        # one logit per attention head
        "layer_0_attn_sink": (heads,),
        "layer_0_attn_linear_gate_weight": (heads * v_head, hidden),
        # Routed experts are fused in the checkpoint but the graph declares
        # the flattened form: the loader reads shape[0]/shape[1] from the node
        # and the MoE kernel indexes expert e at row e*rows with row stride K.
        "layer_1_ffn_experts_gate_up": (experts * 2 * moe_i, hidden),
        "layer_1_ffn_experts_down": (experts * hidden, moe_i),
        "layer_1_ffn_router_bias": (experts,),
    }
    for label, want in expected.items():
        got = by_label.get(label)
        check(got == want,
              f"{label}: shape {got}, expected {want}")

    # The indexer carries no compressor tensors, unlike DeepSeek-V4.
    indexer_sources = [s for _, s, _, _ in captured if ".indexer." in s]
    check(len(indexer_sources) == 5,
          f"five indexer tensors on a full layer, got {len(indexer_sources)}")
    check(not any("compress" in s for s in indexer_sources),
          "Hy4's indexer must not reference compressor tensors")

    # Boundary weights: without these the graph cannot embed or produce logits.
    check(by_label.get("embed_tokens") == (cfg["vocab_size"], hidden),
          f"embed_tokens shape, got {by_label.get('embed_tokens')}")
    check(by_label.get("lm_head") == (cfg["vocab_size"], hidden),
          f"lm_head shape, got {by_label.get('lm_head')}")
    print("  PASS: Hy4-specific weight shapes")


def test_dense_lead_is_dense():
    """Layer 0 uses a dense FFN; later layers route.

    The dense lead layer ships gate_proj and up_proj as separate tensors (the
    checkpoint has no fused gate_up_proj for it), unlike the routed experts
    which are fused and flattened.
    """
    cfg = tiny_cfg(layers=2)
    cfg["indexer_types"] = ["full", "shared"]
    _, captured = build(cfg, seq_len=4, n_ctx=64, is_prefill=True)
    labels = [label for label, _, _, _ in captured]

    check(any(l.startswith("layer_0_ffn_gate") for l in labels),
          "layer 0 has a dense gate_proj weight")
    check(any(l.startswith("layer_0_ffn_up") for l in labels),
          "layer 0 has a dense up_proj weight")
    check(not any(l.startswith("layer_0_ffn_experts") for l in labels),
          "layer 0 has no routed experts")
    check(any(l.startswith("layer_1_ffn_experts") for l in labels),
          "layer 1 has routed experts")
    check(not any(l.startswith("layer_1_ffn_gate") for l in labels),
          "layer 1 has no dense gate weight")
    check(not any(l.startswith("layer_1_ffn_gate_up") for l in labels),
          "layer 1 has no dense gate_up weight")
    print("  PASS: dense lead layer")


def test_decode_path_builds():
    """The decode graph must assemble without the dynamic seq dimension."""
    cfg = tiny_cfg(layers=2)
    cfg["indexer_types"] = ["full", "shared"]
    graph, _ = build(cfg, seq_len=1, n_ctx=64, is_prefill=False)
    tally = count_ops(graph)
    check(tally.get(OpType.SDPA) == 2, "decode builds one SDPA per layer")
    check(tally.get(OpType.HY4_INDEXER) == 1,
          "decode builds one indexer for the single full layer")
    print("  PASS: decode path")


def test_absorbed_dense_cache():
    """The latent cache is shared across heads and DSA selection is wired."""
    cfg = tiny_cfg(layers=2, indexer_types=["full", "shared"])
    original = hy4.split_mla_kv_b
    hy4.split_mla_kv_b = lambda *args, **kwargs: (object(), object())
    try:
        graph, _ = build(cfg, seq_len=4, n_ctx=8,
                         is_prefill=True, absorbed=True)
        tally = count_ops(graph)
        check(tally.get(OpType.SDPA_MLA_ABSORB) == 2,
              "one absorbed attention op per layer")
        caches = [node for node in graph._nodes
                  if node.params_str and
                  node.params_str[0].startswith("latent_cache")]
        check(len(caches) == 2, "one latent cache per layer")
        for cache in caches:
            check(cache.out_shape == (12, 8, 1, 1),
                  f"head-less latent shape, got {cache.out_shape}")
        check(not any(node.params_str and
                      node.params_str[0].startswith(("cache_k", "cache_v"))
                      for node in graph._nodes),
              "absorbed graph must not allocate explicit KV caches")
        sparse_graph, _ = build(cfg, seq_len=4, n_ctx=16,
                                is_prefill=True, absorbed=True)
        for node in sparse_graph._nodes:
            if node.op_type == OpType.SDPA_MLA_ABSORB:
                selection_idx = node.params_i32[7]
                check(selection_idx >= 0 and
                      sparse_graph._nodes[node.inputs[selection_idx]].op_type ==
                      OpType.HY4_INDEXER,
                      "absorbed attention must consume DSA selection")
    finally:
        hy4.split_mla_kv_b = original
    print("  PASS: absorbed cache has no head axis and carries DSA selection")


def test_bad_config_rejected():
    """Malformed indexer_types fail loudly instead of building a wrong graph."""
    cfg = tiny_cfg(layers=3, indexer_types=["full", "shared"])
    try:
        build(cfg, seq_len=4, n_ctx=64, is_prefill=True)
        raise AssertionError("short indexer_types should be rejected")
    except ValueError:
        pass

    # A leading shared layer would have nothing to reuse.
    cfg = tiny_cfg(layers=2, indexer_types=["shared", "full"])
    try:
        build(cfg, seq_len=4, n_ctx=64, is_prefill=True)
        raise AssertionError("leading shared layer should be rejected")
    except ValueError:
        pass
    print("  PASS: malformed config rejected")


def test_convert_rejects_non_hy4():
    """convert() must refuse a checkpoint that is not Hy4.

    It reads config.json first, so write a non-Hy4 config and assert it
    fails on the model type rather than on a missing file.
    """
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump({"model_type": "qwen3"}, f)
        try:
            hy4.convert(tmp, os.path.join(tmp, "out.mollm"))
            raise AssertionError("convert() should reject a non-Hy4 config")
        except ValueError:
            pass
    print("  PASS: convert() rejects a non-Hy4 checkpoint")


def main():
    test_layer_counts()
    test_indexer_only_on_full_layers()
    test_shared_layer_reuses_selection()
    test_weight_shapes()
    test_dense_lead_is_dense()
    test_decode_path_builds()
    test_absorbed_dense_cache()
    test_bad_config_rejected()
    test_convert_rejects_non_hy4()
    print("\nAll Hy4 graph tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
