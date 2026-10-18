#!/usr/bin/env python3
"""Structural tests for the staged Qwen4Exp text graph."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))

from qwen4_exp import build_graph, build_mtp_graph, _build_mtp_input_fusion
from transpile import DimKind, OpType, Precision, propagate_dim_exprs


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def config():
    return {"text_config": {
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "rms_norm_eps": 1e-6,
        "hc_count": 4,
        "hc_lowrank": 8,
        "indexer_budget": 64,
        "rope_parameters": {"rope_theta": 1_000_000,
                            "partial_rotary_factor": 0.5},
        "head_dim": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "output_gate_type": "sigmoid",
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "vocab_size": 128,
        "ple_layer_ids": [],
        "ngram_size": 3,
        "heads_per_ngram": 2,
        "ngram_vocab_size_base": 101,
        "make_ngram_vocab_size_divisible_by": 8,
        "ple_embed_dim": 32,
        "ple_conv_kernel_size": 4,
        "eos_token_id": 127,
    }}


g = build_graph("/tmp/qwen4-test", config(), seq_len=1, n_ctx=64)
ops = [node.op_type for node in g._nodes]
check("mtp_state_output_node" not in g.metadata,
      "ordinary graph does not add MTP state capture overhead")
state_graph = build_graph(".", config(), seq_len=8, n_ctx=64,
                          is_prefill=True, export_mtp_state=True)
state_id = int(state_graph.metadata["mtp_state_output_node"])
check(state_graph._nodes[state_id].out_shape[:2] == (128, 8),
      "MTP state keeps all four residual streams")
check(state_id < state_graph._nodes[-1].id and
      state_graph._nodes[-1].out_shape[0] == 32,
      "normal logits output remains separate and last")
check(ops.count(OpType.GROUP_RMS_NORM) == 5,
      "two GR reads per layer plus final mixer")
check(ops.count(OpType.GR_REDUCE) == 5,
      "all GR reads reduce four streams")
check(ops.count(OpType.GR_INJECT) == 4,
      "attention and MoE write back in every layer")
check(ops.count(OpType.GATED_DELTANET_CONV_DECODE) == 1,
      "linear layer reuses fused decode GDN")
gdn_nodes = [node for node in g._nodes
             if node.op_type == OpType.GATED_DELTANET_CONV_DECODE]
check((gdn_nodes[0].params_i32[4] & 2) != 0,
      "Qwen4Exp selects sigmoid rather than legacy SiLU output gating")
check(ops.count(OpType.SDPA) == 1,
      "short-context QSA uses equivalent gated GQA")
check(ops.count(OpType.MOE) == 2,
      "every decoder layer contains routed MoE")
check(ops.count(OpType.TILE) == 1,
      "embedding is widened into four residual streams once")

try:
    build_graph("/tmp/qwen4-test", config(), seq_len=1, n_ctx=65)
except ValueError as exc:
    check("QSA budget" in str(exc), "long-context error explains QSA limit")
else:
    raise AssertionError("long context must not silently run dense attention")

qsa_cfg = config()
qsa_cfg["text_config"].update(indexer_n_heads=2, indexer_head_dim=8, indexer_compress_ratio=4)
for prefill, seq in ((False, 1), (True, 17)):
    qsa_graph = build_graph(".", qsa_cfg, seq_len=seq, n_ctx=256, is_prefill=prefill, enable_qsa=True)
    propagate_dim_exprs(qsa_graph._nodes)
    indexers = [n for n in qsa_graph._nodes if n.op_type == OpType.QSA_INDEXER]
    sparse = [n for n in qsa_graph._nodes if n.op_type == OpType.QSA_SDPA]
    check(len(indexers) == len(sparse) == 1, "QSA indexer connects to selected-key attention")
    check(not any(n.op_type == OpType.SDPA for n in qsa_graph._nodes), "QSA graph never silently substitutes dense SDPA")
    check(indexers[0].out_prec == Precision.INT32 and indexers[0].out_shape[:2] == (67, seq), "index slots include all incomplete tail tokens")
    check(sparse[0].inputs[6] == indexers[0].id, "selected IDs flow into main attention")
    check(indexers[0].params_i32 == [2, 8, 4, 4, 64], "indexer uses main RoPE width, not indexer fractional width")
    check(qsa_graph._nodes[indexers[0].inputs[4]].out_shape[:2] == (8, 256), "raw indexer state capacity")
    check(qsa_graph._nodes[indexers[0].inputs[5]].out_shape[:2] == (8, 64), "pooled complete-group capacity")
    if prefill:
        check(indexers[0].dim_expr[1].kind == DimKind.SEQ and sparse[0].dim_expr[1].kind == DimKind.SEQ,
              "indexer and attention retain dynamic query lengths")
try:
    build_graph(".", qsa_cfg, n_ctx=262145, enable_qsa=True)
except ValueError:
    pass
else:
    raise AssertionError("QSA unsupported capacity must be explicit")
for context in (32769, 65536, 262144):
    extended = build_graph(".", qsa_cfg, n_ctx=context, enable_qsa=True)
    indexer = next(node for node in extended._nodes if node.op_type == OpType.QSA_INDEXER)
    check(extended._nodes[indexer.inputs[4]].out_shape[1] == context, "extended raw capacity")
    check(extended._nodes[indexer.inputs[5]].out_shape[1] == context // 4, "extended pooled capacity")

qsa_ple_cfg = config()
qsa_ple_cfg["text_config"].update(indexer_n_heads=2, indexer_head_dim=8,
                              indexer_compress_ratio=4, ple_layer_ids=[2])
qsa_ple_graph = build_graph(".", qsa_ple_cfg, seq_len=17, n_ctx=256, is_prefill=True, enable_qsa=True)
qsa_aux = [n.params_str[0] for n in qsa_ple_graph._nodes if n.op_type == OpType.INPUT
           and n.params_str and n.params_str[0].startswith("aux_state")]
check(len(qsa_aux) == len(set(qsa_aux)) == 4, "QSA state namespace does not collide with PLE history/convolution")

ple_cfg = config()
ple_cfg["text_config"]["ple_layer_ids"] = [2]
ple_graph = build_graph("/tmp/qwen4-test", ple_cfg, seq_len=1, n_ctx=64)
ple_ops = [node.op_type for node in ple_graph._nodes]
check(ple_ops.count(OpType.PLE_LOOKUP) == 1,
      "configured PLE layer performs one deterministic table lookup")
check(ple_ops.count(OpType.PLE_GATE) == 1,
      "PLE value is gated independently into residual streams")
check(ple_ops.count(OpType.PLE_DILATED_CONV) == 1,
      "PLE injects local lexical context with a dilated convolution")
aux_names = [node.params_str[0] for node in ple_graph._nodes
             if node.op_type == OpType.INPUT and node.params_str and
             node.params_str[0].startswith("aux_state")]
check(aux_names == ["aux_state0", "aux_state1"],
      "PLE owns token history and dilated-convolution state")

prefill_graph = build_graph(
    "/tmp/qwen4-test", ple_cfg, seq_len=8, n_ctx=64, is_prefill=True)
propagate_dim_exprs(prefill_graph._nodes)
for node in prefill_graph._nodes:
    if node.op_type in (
            OpType.GROUP_RMS_NORM, OpType.GR_REDUCE, OpType.GR_INJECT,
            OpType.PLE_LOOKUP, OpType.PLE_GATE, OpType.PLE_DILATED_CONV):
        check(node.dim_expr[1].kind == DimKind.SEQ,
              f"{node.op_type.name} must preserve runtime sequence length")

calibrated_cfg = config()
calibrated_cfg["text_config"]["export_nvfp4_activation_calibration"] = True
calibrated_graph = build_graph(".", calibrated_cfg, seq_len=1, n_ctx=64, is_prefill=False)
calibration_nodes = [node for node in calibrated_graph._nodes
    if node.op_type == OpType.CONSTANT and node.params_str and
    node.params_str[0].endswith("_input_scales.weights")]
check(len(calibration_nodes) == 4, "package collection sees both calibration weights per layer")
check(sum(node.op_type == OpType.MOE for node in calibrated_graph._nodes) == 2,
      "calibration constants do not replace MoE operators")

import copy
draft_config = config()
original_config = copy.deepcopy(draft_config)
draft = build_mtp_graph(".", draft_config, seq_len=8, n_ctx=64)
check(draft_config == original_config, "draft building preserves target config")
inputs = {node.params_str[0]: node for node in draft._nodes
          if node.op_type == OpType.INPUT}
check(inputs["target_hidden"].out_shape[:2] == (128, 8), "draft consumes all HC streams")
check(set(inputs) == {"hidden", "target_hidden", "mask", "cos", "sin", "cache_k0", "cache_v0"},
      "MTP has its own attention cache and no PLE/recurrent inputs")
check(sum(n.op_type == OpType.SDPA for n in draft._nodes) == 1 and
      sum(n.op_type == OpType.MOE for n in draft._nodes) == 1,
      "draft contains one attention/MoE layer")
refs = [n.weight_path for n in draft._nodes if n.weight_path]
check(all(n.out_prec in (Precision.FP16, Precision.FP32)
          for n in draft._nodes if n.weight_path),
      "draft weight declarations match its FP16/FP32 export, not target quantization")
check(not any("model_language_model_layers_" in ref for ref in refs),
      "draft does not reference target layer weights")
check(any("mtp_layers_0_mlp_experts_gate_up_proj" in ref for ref in refs),
      "draft references separate expert weights")
check(any("mtp_hyper_connection_mixer" in ref for ref in refs),
      "draft collapse uses its own mixer")
propagate_dim_exprs(draft._nodes)
state_node = draft._nodes[int(draft.metadata["mtp_state_output_node"])]
check(state_node.dim_expr[1].kind == DimKind.SEQ,
      "MTP retained-state output has dynamic token count")
for overrides in (
    {"mtp": {"num_hidden_layers": 2}},
    {"mtp": {"mtp_use_hidden_state_from_layer": 0}},
    {"mtp_use_dedicated_embeddings": True},
    {"mtp": {"layer_types": ["linear_attention"]}},
    {"mtp": {"rope_theta": 123}},
    {"hc_count": 1},
):
    unsupported = config()
    unsupported["text_config"].update(overrides)
    try:
        build_mtp_graph(".", unsupported, seq_len=8, n_ctx=64)
    except ValueError:
        pass
    else:
        raise AssertionError(f"unsupported MTP configuration accepted: {overrides}")

# Evaluate the fusion graph independently against the token-major formula.
# Distinct stream/weight values catch accidental transpose or token mixing.
import numpy as np
from transpile import GraphBuilder, DimExpr
rng = np.random.default_rng(20261005)
for h in (4, 8):
    for streams in (2, 4, 8):
        for tokens in (1, 3):
            graph = GraphBuilder()
            target = graph.input("target", (h * streams, tokens),
                                 dynamic=(DimExpr.const(), DimExpr.seq()))
            embedding = graph.input("embedding", (h, tokens),
                                    dynamic=(DimExpr.const(), DimExpr.seq()))
            out = _build_mtp_input_fusion(graph, target, embedding, ".", h, streams, 1e-6)
            values = {target: rng.normal(size=(h * streams, tokens)),
                      embedding: rng.normal(size=(h, tokens))}
            weights = {}
            for node in graph._nodes:
                if node.op_type == OpType.INPUT:
                    continue
                if node.op_type == OpType.CONSTANT:
                    shape = node.out_shape[:2]
                    value = rng.normal(size=shape)
                    values[node.id] = value
                    weights[Path(node.weight_path).stem] = value
                elif node.op_type == OpType.RMS_NORM:
                    x, gamma = (values[i] for i in node.inputs)
                    values[node.id] = x / np.sqrt(np.mean(x * x, axis=0, keepdims=True) + 1e-6) * gamma
                elif node.op_type == OpType.RESHAPE:
                    width = node.out_shape[0]
                    values[node.id] = values[node.inputs[0]].reshape((width, -1), order="F")
                elif node.op_type == OpType.MATMUL:
                    x, weight = (values[i] for i in node.inputs)
                    values[node.id] = weight @ x
                elif node.op_type == OpType.TILE:
                    values[node.id] = np.tile(values[node.inputs[0]], node.params_i32[:2])
                elif node.op_type == OpType.ADD:
                    values[node.id] = values[node.inputs[0]] + values[node.inputs[1]]
                else:
                    raise AssertionError(node.op_type)
            t = values[target].T
            e = values[embedding].T
            t = t / np.sqrt(np.mean(t * t, axis=-1, keepdims=True) + 1e-6)
            t *= weights["mtp_pre_fc_norm_hidden_weight"].ravel()
            e = e / np.sqrt(np.mean(e * e, axis=-1, keepdims=True) + 1e-6)
            e *= weights["mtp_pre_fc_norm_embedding_weight"].ravel()
            expected = t.reshape(tokens, streams, h) @ weights["mtp_fc_hidden_weight"].T
            expected += (e @ weights["mtp_fc_embedding_weight"].T)[:, None, :]
            np.testing.assert_allclose(values[out].T, expected.reshape(tokens, streams * h), atol=1e-12)
print("Qwen4Exp MTP structure and 12 fusion numeric cases passed")
