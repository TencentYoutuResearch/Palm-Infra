#!/usr/bin/env python3
"""Graph and weight-layout tests for dense and MoE Qwen3.5 MTP drafts."""

import io
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import qwen35
import qwen35_moe
from qwen35 import _query_then_gate_rows
from transpile import OpType

SHARED = {"./embed_tokens.weights", "./lm_head.weights"}
DRAFT_INPUTS = ["hidden", "mask", "cos", "sin", "cache_k0", "cache_v0", "target_hidden"]


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def text_config(moe: bool) -> dict:
    tc = {
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "layer_types": ["full_attention"],
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.25},
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "vocab_size": 128,
        "mtp_num_hidden_layers": 1,
    }
    if moe:
        tc.update(moe_intermediate_size=8, shared_expert_intermediate_size=12,
                  num_experts=3, num_experts_per_tok=2)
    else:
        tc["intermediate_size"] = 32
    return {"model_type": "qwen3_5_moe" if moe else "qwen3_5", "text_config": tc}


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header = {}
    payload = b""
    for name, array in tensors.items():
        data = np.ascontiguousarray(array, dtype=np.float16).tobytes()
        header[name] = {"dtype": "F16", "shape": list(array.shape),
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def streamed_array(weight, dtype) -> np.ndarray:
    output = io.BytesIO()
    weight.write_to(output)
    return np.frombuffer(output.getvalue()[88:], dtype=dtype).reshape(weight.logical_shape)


def checkpoint(cfg: dict) -> dict[str, np.ndarray]:
    tc = cfg["text_config"]
    hidden, heads, head_dim = tc["hidden_size"], tc["num_attention_heads"], tc["head_dim"]
    kv = tc["num_key_value_heads"] * head_dim
    rng = np.random.default_rng(7)

    def tensor(*shape):
        return rng.standard_normal(shape).astype(np.float16)

    base = "mtp.layers.0"
    tensors = {
        "mtp.fc.weight": tensor(hidden, 2 * hidden),
        f"{base}.self_attn.q_proj.weight": tensor(2 * heads * head_dim, hidden),
        f"{base}.self_attn.k_proj.weight": tensor(kv, hidden),
        f"{base}.self_attn.v_proj.weight": tensor(kv, hidden),
        f"{base}.self_attn.o_proj.weight": tensor(hidden, heads * head_dim),
        f"{base}.self_attn.q_norm.weight": tensor(head_dim),
        f"{base}.self_attn.k_norm.weight": tensor(head_dim),
    }
    for name in ("mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_hidden", "mtp.norm",
                 f"{base}.input_layernorm", f"{base}.post_attention_layernorm"):
        tensors[f"{name}.weight"] = tensor(hidden)
    mlp = f"{base}.mlp"
    if "num_experts" in tc:
        inner, shared = tc["moe_intermediate_size"], tc["shared_expert_intermediate_size"]
        tensors[f"{mlp}.gate.weight"] = tensor(tc["num_experts"], hidden)
        for expert in range(tc["num_experts"]):
            tensors[f"{mlp}.experts.{expert}.gate_proj.weight"] = tensor(inner, hidden)
            tensors[f"{mlp}.experts.{expert}.up_proj.weight"] = tensor(inner, hidden)
            tensors[f"{mlp}.experts.{expert}.down_proj.weight"] = tensor(hidden, inner)
        tensors[f"{mlp}.shared_expert.gate_proj.weight"] = tensor(shared, hidden)
        tensors[f"{mlp}.shared_expert.up_proj.weight"] = tensor(shared, hidden)
        tensors[f"{mlp}.shared_expert.down_proj.weight"] = tensor(hidden, shared)
        tensors[f"{mlp}.shared_expert_gate.weight"] = tensor(1, hidden)
    else:
        inner = tc["intermediate_size"]
        tensors[f"{mlp}.gate_proj.weight"] = tensor(inner, hidden)
        tensors[f"{mlp}.up_proj.weight"] = tensor(inner, hidden)
        tensors[f"{mlp}.down_proj.weight"] = tensor(hidden, inner)
    return tensors


def check_graph(module, cfg: dict, final_op: OpType) -> None:
    graph = module.build_mtp_graph(".", cfg, seq_len=8, n_ctx=64)
    inputs = [node.params_str[0] for node in graph._nodes if node.op_type == OpType.INPUT]
    check(inputs == DRAFT_INPUTS, "MTP graph exposes the engine's draft inputs")
    references = {node.weight_path for node in graph._nodes if node.weight_path}
    check(not any("model_language_model" in name for name in references),
          "MTP weights never alias target layer weights")
    check(SHARED <= references, "MTP graph shares the target embedding and head")
    consumed = {i for node in graph._nodes for i in node.inputs}
    outputs = [node for node in graph._nodes
               if node.id not in consumed and node.op_type not in (OpType.INPUT, OpType.CONSTANT)]
    check(outputs[-1].op_type == final_op and outputs[-1].out_shape[0] == 16,
          "MTP graph ends with the mtp.norm state")
    without_draft = json.loads(json.dumps(cfg))
    del without_draft["text_config"]["mtp_num_hidden_layers"]
    try:
        module.build_mtp_graph(".", without_draft)
    except ValueError:
        pass
    else:
        raise AssertionError("MTP export requires a declared draft layer")


def check_weights(module, cfg: dict) -> None:
    tc = cfg["text_config"]
    hidden = tc["hidden_size"]
    tensors = checkpoint(cfg)
    base = "mtp.layers.0"
    with tempfile.TemporaryDirectory() as directory:
        write_safetensors(Path(directory) / "model.safetensors", tensors)
        weights = module.build_mtp_streamed_weights(directory, cfg)
        graph = module.build_mtp_graph(".", cfg)
        dedicated = {node.weight_path for node in graph._nodes if node.weight_path} - SHARED
        check(set(weights) == dedicated, "MTP weights match the graph references exactly")
        precisions = {node.weight_path: node.out_prec for node in graph._nodes if node.weight_path}
        check(all(weight.precision == precisions[ref] for ref, weight in weights.items()),
              "MTP weight precisions match the graph declarations")

        def get(name, dtype=np.float16):
            return streamed_array(weights[f"./{name}.weights"], dtype)

        qkv = np.concatenate([
            _query_then_gate_rows(tensors[f"{base}.self_attn.q_proj.weight"],
                                  tc["num_attention_heads"], tc["head_dim"]),
            tensors[f"{base}.self_attn.k_proj.weight"], tensors[f"{base}.self_attn.v_proj.weight"]])
        check(np.array_equal(get("mtp_layers_0_self_attn_qkv_proj_weight"), qkv),
              "MTP Q/K/V rows follow the target query-then-gate layout")
        fc = tensors["mtp.fc.weight"]
        check(np.array_equal(get("mtp_fc_embedding_weight"), fc[:, :hidden]) and
              np.array_equal(get("mtp_fc_hidden_weight"), fc[:, hidden:]),
              "mtp.fc splits into its embedding and hidden input columns")
        check(np.array_equal(get("mtp_norm_weight", np.float32),
                             tensors["mtp.norm.weight"].astype(np.float32) + 1.0),
              "MTP norms store the effective 1 + gamma scale")
        mlp = f"{base}.mlp"
        if "num_experts" in tc:
            experts = range(tc["num_experts"])
            gate_up = np.concatenate([np.concatenate([
                tensors[f"{mlp}.experts.{e}.gate_proj.weight"],
                tensors[f"{mlp}.experts.{e}.up_proj.weight"]]) for e in experts])
            down = np.concatenate([tensors[f"{mlp}.experts.{e}.down_proj.weight"] for e in experts])
            check(np.array_equal(get("mtp_layers_0_mlp_experts_gate_up_proj"), gate_up),
                  "draft experts store gate rows before up rows, expert-major")
            check(np.array_equal(get("mtp_layers_0_mlp_experts_down_proj"), down),
                  "draft expert down projections are expert-major")
            check(np.array_equal(get("mtp_layers_0_mlp_gate_weight"), tensors[f"{mlp}.gate.weight"]),
                  "draft router weight is stored unchanged")
        else:
            gate_up = np.concatenate([tensors[f"{mlp}.gate_proj.weight"],
                                      tensors[f"{mlp}.up_proj.weight"]])
            check(np.array_equal(get("mtp_layers_0_mlp_gate_up_proj_weight"), gate_up),
                  "dense draft MLP stores gate rows before up rows")


def main():
    check_graph(qwen35, text_config(moe=False), OpType.ADD_RMS_NORM)
    check_weights(qwen35, text_config(moe=False))
    check_graph(qwen35_moe, text_config(moe=True), OpType.RMS_NORM)
    check_weights(qwen35_moe, text_config(moe=True))
    print("Qwen3.5 MTP tests passed")


if __name__ == "__main__":
    main()
