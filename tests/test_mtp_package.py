#!/usr/bin/env python3
"""Check the optional MTP graph package section and graph boundaries."""

import json
import struct
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "models"))

from qwen3_moe import _mtp_quantization, build_mtp_graph  # noqa: E402
from transpile import GraphBuilder, save_package  # noqa: E402


def tiny_graph(name: str) -> GraphBuilder:
    graph = GraphBuilder()
    graph.input(name, (4, 1))
    return graph


def main() -> int:
    assert _mtp_quantization("w4g128") == "w8g128"
    assert _mtp_quantization("w4g128", "fp16") == "fp16"

    cfg = {
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "intermediate_size": 64,
        "moe_intermediate_size": 16,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 1,
        "n_shared_experts": 0,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "vocab_size": 128,
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "_attn_norm_kind": "q_layernorm",
        "_has_router_correction_bias": True,
    }
    mtp = build_mtp_graph(".", cfg, seq_len=8, n_ctx=32)
    inputs = {
        node.params_str[0]
        for node in mtp._nodes
        if node.op_type.name == "INPUT"
    }
    assert inputs == {
        "target_hidden", "hidden", "mask", "cos", "sin",
        "cache_k0", "cache_v0",
    }

    # Package a minimal graph that has no external weights. This isolates the
    # optional section layout from model conversion and quantization.
    with tempfile.TemporaryDirectory(prefix="mollm_mtp_package_") as tmp:
        root = Path(tmp)
        package = root / "mtp.mollm"
        verify = tiny_graph("hidden")
        draft = tiny_graph("target_hidden")
        save_package(
            str(package), tiny_graph("hidden"), tiny_graph("hidden"),
            str(root), {"architecture": "test"},
            g_mtp=draft, g_mtp_verify=verify)
        raw = package.read_bytes()
        values = struct.unpack("<II15Q", raw[:128])
        assert values[0] == 0x4D4C4F4D and values[1] == 1
        metadata_offset, metadata_length = values[2:4]
        weights_offset = values[12]
        mtp_length = values[-1]
        assert mtp_length > 0
        metadata = json.loads(
            raw[metadata_offset:metadata_offset + metadata_length])
        assert metadata["architecture"] == "test"
        verify_length = metadata["mtp_verify_graph_length"]
        assert 0 < verify_length < mtp_length
        mtp_offset = values[10] + values[11] + values[15]
        bundle_offset = mtp_offset
        verify_blob = raw[bundle_offset:bundle_offset + verify_length]
        draft_blob = raw[bundle_offset + verify_length:
                          bundle_offset + mtp_length]
        assert struct.unpack_from("<I", verify_blob)[0] == 0x4D4C4C47
        assert struct.unpack_from("<I", draft_blob)[0] == 0x4D4C4C47
        assert len(verify_blob) == verify_length
        assert len(draft_blob) == mtp_length - verify_length
        assert bundle_offset + mtp_length <= weights_offset
        assert raw[weights_offset:] == b""
    print("MTP package tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
