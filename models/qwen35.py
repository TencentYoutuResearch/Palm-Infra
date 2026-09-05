"""
mollm model builder for dense Qwen3.5 models.

Hybrid architecture:
  - 18 layers of linear attention (Gated Delta Rule)
  - 6 layers of full attention (GQA + QK norm + output gate)
  - Standard SwiGLU MLP

Packages include the FP16 vision tower when the checkpoint provides one.
"""

from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path

import numpy as np

from collections.abc import Mapping
from typing import Iterator

from safetensors_stream import SafeTensorIndex
from transpile import (
    GraphBuilder, Precision, _write_weight_file,
    quantize_weight_w8_group, save_package,
    write_quantized_weight_file_cpp,
)
from model_metadata import infer_hf_model_name
from qwen35_vision import build_vision_graph, export_vision_weights


def _layer_index(wname: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", wname)
    return int(match.group(1)) if match else None


def _w4mix_promote_to_w8(wname: str) -> bool:
    """Qwen3.5-specific mixed W4 policy inspired by llama.cpp Q4_K_M."""
    if wname == "lm_head.weight" or wname.endswith(".lm_head.weight"):
        return True

    layer_idx = _layer_index(wname)
    if layer_idx is None:
        return False

    # Fused linear-attention QKV is V-like/sensitive.
    if (".linear_attn.in_proj_qkv.weight" in wname
            or ".linear_attn.in_proj_qkvabz.weight" in wname):
        return True

    # Full-attention V projection is sensitive. New packages merge Q/K/V, so
    # promote the whole row-concatenated tensor rather than silently lowering
    # the V rows from W8 to W4.
    if (".self_attn.v_proj.weight" in wname
            or ".self_attn.qkv_proj.weight" in wname):
        return True

    # Our current W4 format is a symmetric per-group baseline, not Q4_K. Be
    # more quality-biased than llama.cpp's Q4_K_M for output-like projections.
    if ".linear_attn.out_proj.weight" in wname or ".self_attn.o_proj.weight" in wname:
        return True

    # llama.cpp's Q4_K_M gives selected FFN down projections more bits. With
    # the simpler W4 format, promote all FFN down projections for now.
    if ".mlp.down_proj.weight" in wname:
        return True

    return False


def _canonical_quant(quant: str) -> str:
    q = quant.lower()
    if q in ("none", "fp16", "f16"):
        return "fp16"
    return q


def _quant_spec(quant: str, k: int, wname: str = "") -> tuple[str, int] | None:
    quant = _canonical_quant(quant)
    if quant == "fp16":
        return None
    if quant == "w8pc":
        return ("w8", k)
    if quant.startswith("w8g"):
        return ("w8", int(quant[3:]))
    if quant.startswith("w4mixg"):
        group_size = int(quant[6:])
        if _w4mix_promote_to_w8(wname):
            return ("w8", group_size)
        return ("w4", group_size)
    if quant.startswith("w4g"):
        return ("w4", int(quant[3:]))
    raise ValueError(f"unsupported quant mode: {quant}")


def load_safetensors(path: str) -> dict[str, np.ndarray]:
    """Load a single-file safetensors model."""
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len).decode('utf-8'))

    tensors = {}
    with open(path, 'rb') as f:
        f.seek(8 + header_len)
        for name, meta in header.items():
            if name == '__metadata__':
                continue
            dtype_str = meta['dtype']
            shape = meta['shape']
            offsets = meta['data_offsets']
            f.seek(8 + header_len + offsets[0])
            data = f.read(offsets[1] - offsets[0])
            np_dtype = {
                'F32': np.float32, 'F16': np.float16,
                'BF16': np.uint16,
            }[dtype_str]
            arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
            if dtype_str == 'F16':
                arr = arr.astype(np.float16)
            elif dtype_str == 'BF16':
                as_u32 = arr.astype(np.uint32) << 16
                arr = as_u32.view(np.float32)
            tensors[name] = arr
    return tensors


class _LazySafeTensors(Mapping[str, np.ndarray]):
    """Load one indexed safetensor at a time during conversion."""

    _DTYPES = {
        "F16": np.dtype("<f2"),
        "F32": np.dtype("<f4"),
        "BF16": np.dtype("<u2"),
    }

    def __init__(self, model_dir: Path):
        self._index = SafeTensorIndex(model_dir)
        self._names = tuple(
            name for name in self._index.weight_map
            if name != "__metadata__"
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __getitem__(self, name: str) -> np.ndarray:
        tensor = self._index.tensor(name)
        dtype = self._DTYPES.get(tensor.dtype)
        if dtype is None:
            raise ValueError(
                f"unsupported Qwen3.5 conversion dtype {tensor.dtype}: {name}")
        with open(tensor.source.path, "rb") as source:
            source.seek(tensor.source.offset)
            raw = source.read(tensor.source.size)
        if len(raw) != tensor.source.size:
            raise ValueError(f"truncated safetensor payload: {name}")
        values = np.frombuffer(raw, dtype=dtype)
        if tensor.dtype == "BF16":
            values = (values.astype(np.uint32) << 16).view(np.float32)
        return values.reshape(tensor.shape)


def _query_then_gate_rows(
        q_weight: np.ndarray, num_heads: int, head_dim: int) -> np.ndarray:
    """Reorder q_proj rows from [head, query|gate, dim] to [query, gate].

    Hugging Face stores each head's query rows immediately followed by its
    output-gate rows. Grouping all query heads before all gate heads makes both
    decode slices contiguous without changing either projection.
    """
    expected_rows = num_heads * 2 * head_dim
    if q_weight.ndim != 2 or q_weight.shape[0] != expected_rows:
        raise ValueError(
            f"unexpected q_proj shape {q_weight.shape}; "
            f"expected ({expected_rows}, K)")
    grouped = q_weight.reshape(
        num_heads, 2, head_dim, q_weight.shape[1])
    return np.concatenate(
        [grouped[:, 0].reshape(num_heads * head_dim, q_weight.shape[1]),
         grouped[:, 1].reshape(num_heads * head_dim, q_weight.shape[1])],
        axis=0)


_QWEN35_MTP_REQUIRED_WEIGHTS = {
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
}


def _mtp_layer_count(cfg: dict) -> int:
    tc = cfg["text_config"] if "text_config" in cfg else cfg
    return int(tc.get("mtp_num_hidden_layers", 0) or 0)


def _validate_mtp_weights(weights: dict, cfg: dict) -> int:
    mtp_layers = _mtp_layer_count(cfg)
    if mtp_layers not in (0, 1):
        raise ValueError(
            "only one Qwen3.5 MTP layer is currently supported; "
            f"checkpoint declares {mtp_layers}"
        )
    if mtp_layers:
        missing = sorted(_QWEN35_MTP_REQUIRED_WEIGHTS.difference(weights))
        if missing:
            raise ValueError(
                "Qwen3.5 MTP is declared but required tensors are missing: "
                + ", ".join(missing)
            )
    return mtp_layers


def export_weights(weights: dict, weights_dir: str, cfg: dict,
                   quant: str = "fp16", include_mtp: bool = False):
    """Export text model weights. Skip vision encoder."""
    os.makedirs(weights_dir, exist_ok=True)
    quant = _canonical_quant(quant)
    quant_counts = {"w4": 0, "w8": 0}
    lm_head_source = None

    for wname, wdata in weights.items():
        if 'visual' in wname or 'vision' in wname:
            continue
        if 'language_model' not in wname:
            continue
        if wname.endswith('lm_head.weight'):
            lm_head_source = wdata
            break

    def save(name: str, data: np.ndarray, quantizable: bool = False, raw_name: str = ""):
        wpath = os.path.join(weights_dir, f"{name}.weights")
        quant_spec = (
            _quant_spec(quant, data.shape[1], raw_name)
            if quantizable and data.ndim == 2 else None
        )
        if quant_spec is not None:
            quant_kind, group_size = quant_spec
            if write_quantized_weight_file_cpp(
                wpath, data, quant_kind, group_size, required=(quant_kind == "w4")
            ):
                quant_counts[quant_kind] += 1
                return
            if quant_kind == "w8":
                q, scales, gs, ng = quantize_weight_w8_group(data, group_size)
                _write_weight_file(wpath, q, scales=scales, group_size=gs, num_groups=ng)
            else:
                raise ValueError(f"unsupported quant kind: {quant_kind}")
            quant_counts[quant_kind] += 1
        else:
            _write_weight_file(wpath, data)

    # Skip all visual/vision weights
    for wname, wdata in weights.items():
        if 'visual' in wname or 'vision' in wname:
            continue
        if 'language_model' not in wname:
            continue
        if wname.endswith('lm_head.weight'):
            continue
        # MLP gate/up weights are exported as one row-concatenated matrix
        # below.  A single projection reuses the activation tile and feeds the
        # fused SWIGLU op, avoiding two matmuls plus separate SILU/MUL kernels.
        if '.mlp.gate_proj.weight' in wname or '.mlp.up_proj.weight' in wname:
            continue
        if ('.self_attn.q_proj.weight' in wname
                or '.self_attn.k_proj.weight' in wname
                or '.self_attn.v_proj.weight' in wname):
            continue
        if ('.linear_attn.in_proj_qkv.weight' in wname
                or '.linear_attn.in_proj_a.weight' in wname
                or '.linear_attn.in_proj_b.weight' in wname
                or '.linear_attn.in_proj_z.weight' in wname):
            continue

        # Final norm (check before generic norm)
        if wname.endswith('model.language_model.norm.weight'):
            d = wdata.astype(np.float32) if wdata.dtype != np.float32 else wdata
            d = 1.0 + d  # Qwen3_5RMSNorm: output * rms(x) * (1.0 + weight)
            save('final_norm', d)
            continue

        # RMSNorm weights → FP32.
        # Qwen3_5RMSNorm (input_layernorm, post_attention_layernorm, q_norm, k_norm) uses:
        #   output * (1.0 + weight)  →  pre-compute (1.0 + weight)
        # Qwen3_5RMSNormGated (linear_attn.norm) uses:
        #   output * weight  →  standard RMSNorm, no conversion needed
        if 'norm' in wname.lower() or 'layernorm' in wname.lower():
            d = wdata.astype(np.float32) if wdata.dtype != np.float32 else wdata
            # Add 1.0 for all Qwen3_5RMSNorm variants
            # (input_layernorm, post_attention_layernorm, q_norm, k_norm)
            # but NOT for linear_attn.norm which is RMSNormGated
            if 'linear_attn' not in wname:
                d = 1.0 + d
            save(wname.replace('.', '_'), d)
            continue

        # A_log, dt_bias → keep as-is (already F32 for A_log, BF16→F32 for dt_bias)
        if 'A_log' in wname or 'dt_bias' in wname:
            d = wdata.astype(np.float32) if wdata.dtype != np.float32 else wdata
            save(wname.replace('.', '_'), d)
            continue

        # Embed tokens → FP16
        if 'embed_tokens' in wname:
            if lm_head_source is None:
                lm_head_source = wdata
            d = wdata.astype(np.float16) if wdata.dtype != np.float16 else wdata
            save('embed_tokens', d)
            continue

        # conv1d.weight → FP32 (used by ShortConv scalar kernel, needs FP32)
        if 'conv1d' in wname:
            d = wdata.reshape(wdata.shape[0], wdata.shape[2])  # [6144, 1, 4] → [6144, 4]
            d = d.astype(np.float32) if d.dtype != np.float32 else d
            save(wname.replace('.', '_'), d)
            continue

        # Projection weights are FP16 by default, optionally W8 quantized.
        d = wdata.astype(np.float16) if wdata.dtype != np.float16 else wdata
        save(wname.replace('.', '_'), d, quantizable=True, raw_name=wname)

    # Merge MLP gate/up projections per layer. Quantization is row-local, so
    # concatenating output rows preserves each original row's W8/W4 scales and
    # integer values exactly.
    gate_pat = re.compile(
        r'^model\.language_model\.layers\.(\d+)\.mlp\.gate_proj\.weight$'
    )
    layer_ids = sorted(
        int(match.group(1))
        for wname in weights
        for match in [gate_pat.match(wname)]
        if match
    )
    for layer_idx in layer_ids:
        gate_name = (
            f"model.language_model.layers.{layer_idx}.mlp.gate_proj.weight"
        )
        up_name = (
            f"model.language_model.layers.{layer_idx}.mlp.up_proj.weight"
        )
        if up_name not in weights:
            raise KeyError(f"Missing matching MLP up projection: {up_name}")
        merged = np.concatenate(
            [weights[gate_name], weights[up_name]], axis=0
        )
        if merged.dtype != np.float16:
            merged = merged.astype(np.float16)
        save(
            f"model_language_model_layers_{layer_idx}_mlp_gate_up_proj_weight",
            merged,
            quantizable=True,
            raw_name=(
                f"model.language_model.layers.{layer_idx}."
                "mlp.gate_up_proj.weight"
            ),
        )

    # Full-attention Q/K/V projections share the same input. Store their output
    # rows in one matrix so both prefill tensor GEMM and decode GEMV read the
    # activation once. The q projection already contains query and output-gate
    # rows; it remains the first contiguous slice.
    q_pat = re.compile(
        r'^model\.language_model\.layers\.(\d+)\.self_attn\.q_proj\.weight$'
    )
    attn_layer_ids = sorted(
        int(match.group(1))
        for wname in weights
        for match in [q_pat.match(wname)]
        if match
    )
    for layer_idx in attn_layer_ids:
        prefix = f"model.language_model.layers.{layer_idx}.self_attn"
        names = [
            f"{prefix}.q_proj.weight",
            f"{prefix}.k_proj.weight",
            f"{prefix}.v_proj.weight",
        ]
        missing = [name for name in names if name not in weights]
        if missing:
            raise KeyError(
                f"Missing full-attention projection(s): {', '.join(missing)}"
            )
        tc = cfg['text_config'] if 'text_config' in cfg else cfg
        q_weight = _query_then_gate_rows(
            weights[names[0]], tc['num_attention_heads'], tc['head_dim'])
        merged = np.concatenate(
            [q_weight, weights[names[1]], weights[names[2]]], axis=0)
        if merged.dtype != np.float16:
            merged = merged.astype(np.float16)
        save(
            f"model_language_model_layers_{layer_idx}_self_attn_qkv_proj_weight",
            merged,
            quantizable=True,
            raw_name=f"{prefix}.qkv_proj.weight",
        )

    # All linear-attention input projections consume the same normalized
    # activation. Store QKV, the two scalar gates, and Z as one row-concatenated
    # matrix so prefill and decode use a single matmul and activation read.
    qkv_pat = re.compile(
        r'^model\.language_model\.layers\.(\d+)\.'
        r'linear_attn\.in_proj_qkv\.weight$'
    )
    linear_layer_ids = sorted(
        int(match.group(1))
        for wname in weights
        for match in [qkv_pat.match(wname)]
        if match
    )
    for layer_idx in linear_layer_ids:
        prefix = f"model.language_model.layers.{layer_idx}.linear_attn"
        names = [
            f"{prefix}.in_proj_qkv.weight",
            f"{prefix}.in_proj_a.weight",
            f"{prefix}.in_proj_b.weight",
            f"{prefix}.in_proj_z.weight",
        ]
        missing = [name for name in names if name not in weights]
        if missing:
            raise KeyError(
                f"Missing linear-attention projection(s): {', '.join(missing)}"
            )
        merged = np.concatenate([weights[name] for name in names], axis=0)
        if merged.dtype != np.float16:
            merged = merged.astype(np.float16)
        save(
            f"model_language_model_layers_{layer_idx}_linear_attn_in_proj_weight",
            merged,
            quantizable=True,
            raw_name=f"{prefix}.in_proj_qkvabz.weight",
        )

    if include_mtp:
        tc = cfg["text_config"] if "text_config" in cfg else cfg
        hidden_size = int(tc["hidden_size"])
        num_heads = int(tc["num_attention_heads"])
        head_dim = int(tc["head_dim"])

        def save_mtp_norm(raw_name: str):
            data = weights[raw_name]
            data = data.astype(np.float32) if data.dtype != np.float32 else data
            save(raw_name.replace(".", "_"), 1.0 + data)

        for raw_name in (
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.norm.weight",
        ):
            save_mtp_norm(raw_name)

        for raw_name in (
            "mtp.fc.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.mlp.down_proj.weight",
        ):
            data = weights[raw_name]
            data = data.astype(np.float16) if data.dtype != np.float16 else data
            save(raw_name.replace(".", "_"), data)

        attn_prefix = "mtp.layers.0.self_attn"
        q_weight = _query_then_gate_rows(
            weights[f"{attn_prefix}.q_proj.weight"], num_heads, head_dim
        )
        qkv = np.concatenate(
            [q_weight,
             weights[f"{attn_prefix}.k_proj.weight"],
             weights[f"{attn_prefix}.v_proj.weight"]],
            axis=0,
        )
        if qkv.dtype != np.float16:
            qkv = qkv.astype(np.float16)
        save("mtp_layers_0_self_attn_qkv_proj_weight", qkv)

        gate_up = np.concatenate(
            [weights["mtp.layers.0.mlp.gate_proj.weight"],
             weights["mtp.layers.0.mlp.up_proj.weight"]],
            axis=0,
        )
        if gate_up.dtype != np.float16:
            gate_up = gate_up.astype(np.float16)
        save("mtp_layers_0_mlp_gate_up_proj_weight", gate_up)
    if lm_head_source is None:
        raise KeyError("No lm_head.weight or embed_tokens.weight found for lm_head export")
    d = lm_head_source.astype(np.float16) if lm_head_source.dtype != np.float16 else lm_head_source
    save('lm_head', d, quantizable=True, raw_name="lm_head.weight")

    if quant != "fp16":
        print(f"  Quantized tensors: W4={quant_counts['w4']} W8={quant_counts['w8']}")


def build_graph(weights_dir: str, cfg: dict, seq_len: int = 1,
                n_ctx: int = 4096, is_prefill: bool = False,
                expose_mtp_hidden: bool = False,
                verification: bool = False) -> GraphBuilder:
    """Build prefill or decode graph for Qwen3.5 text model.

    When is_prefill=True, the hidden INPUT's seq dim is marked DynamicKind.SEQ
    so the C++ runtime can inject actual seq_len (dynamic shape mode).
    Decode graphs (seq=1) stay all-STATIC.
    """
    if verification and (not is_prefill or seq_len != 2):
        raise ValueError(
            "Qwen3.5 verification graph requires is_prefill=True and seq_len=2")
    g = GraphBuilder()
    tc = cfg['text_config'] if 'text_config' in cfg else cfg

    hidden_size = tc['hidden_size']
    num_layers = tc['num_hidden_layers']
    layer_types = tc['layer_types']
    eps = tc.get('rms_norm_eps', 1e-6)
    rope_theta = tc['rope_parameters']['rope_theta']
    rope_dim = int(tc['head_dim'] * tc['rope_parameters']['partial_rotary_factor'])  # 256 * 0.25 = 64

    # Full attention params
    num_heads = tc['num_attention_heads']           # 8
    num_kv_heads = tc['num_key_value_heads']        # 2
    head_dim = tc['head_dim']                       # 256

    # Linear attention params
    linear_num_heads = tc['linear_num_key_heads']   # 16
    linear_k_dim = tc['linear_key_head_dim']         # 128
    linear_v_dim = tc['linear_value_head_dim']      # 128
    linear_num_v_heads = tc.get('linear_num_value_heads', linear_num_heads)  # 0.8B=16, 4B=32
    conv_kernel = tc['linear_conv_kernel_dim']       # 4

    # MLP
    intermediate = tc.get('intermediate_size', 3584)

    print(f"Qwen3.5 graph: seq_len={seq_len}, layers={num_layers}, "
          f"heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}, "
          f"linear_heads={linear_num_heads}, linear_v_heads={linear_num_v_heads}, "
          f"linear_k={linear_k_dim}, linear_v={linear_v_dim}, conv_kernel={conv_kernel}")

    # ---- set graph metadata (engine reads these instead of CLI args) ----
    g.set_model_config(
        rope_dim=rope_dim,
        rope_theta=rope_theta,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=tc['vocab_size'],
        model_type='qwen3_5',
    )

    # ---- embed_tokens ----
    embed_path = os.path.join(weights_dir, "embed_tokens.weights")
    embed_shape = (tc['vocab_size'], hidden_size)
    g.weight(embed_path, embed_shape, Precision.FP16)

    # ---- lm_head ----
    lm_head_path = os.path.join(weights_dir, "lm_head.weights")
    g.weight(lm_head_path, embed_shape, Precision.FP16)

    # ---- graph inputs ----
    # In prefill graphs, mark seq dim (shape[1]) as SEQ so the C++ runtime
    # can substitute actual seq_len. Decode graphs stay all-CONST.
    if is_prefill:
        from transpile import DimExpr
        SEQ_DIM = DimExpr.seq()
        CONST   = DimExpr.const()
        hidden_dyn = (CONST, SEQ_DIM, CONST, CONST)
        mask_dyn   = (CONST, SEQ_DIM, CONST, CONST)
        cos_dyn    = (CONST, SEQ_DIM, CONST, CONST)
        sin_dyn    = (CONST, SEQ_DIM, CONST, CONST)
    else:
        hidden_dyn = mask_dyn = cos_dyn = sin_dyn = None
    hidden = g.input('hidden', (hidden_size, seq_len), dynamic=hidden_dyn)
    mask = g.input('mask', (1, seq_len), dynamic=mask_dyn)
    cos = g.input('cos', (rope_dim // 2, seq_len), dynamic=cos_dyn)
    sin = g.input('sin', (rope_dim // 2, seq_len), dynamic=sin_dyn)

    # ---- persistent state inputs (KV cache for full attn, GDN state for linear attn) ----
    cache_inputs = []
    checkpoint_inputs = [(None, None) for _ in range(num_layers)]
    for i in range(num_layers):
        lt = layer_types[i]
        if lt == 'full_attention':
            ck = g.input(f'cache_k{i}', (head_dim, n_ctx, num_kv_heads), prec=Precision.FP16)
            cv = g.input(f'cache_v{i}', (head_dim, n_ctx, num_kv_heads), prec=Precision.FP16)
            cache_inputs.append(('kv', ck, cv))
        else:
            # GDN recurrent state: [v_dim, k_dim, num_value_heads] FP32
            gs = g.input(f'gdn_state{i}', (linear_v_dim, linear_k_dim, linear_num_v_heads), prec=Precision.FP32)
            qkv_total = linear_num_heads * linear_k_dim * 2 + linear_num_v_heads * linear_v_dim
            gc = g.input(f'gdn_conv{i}', (qkv_total, conv_kernel - 1), prec=Precision.FP32)
            if verification:
                checkpoint_inputs[i] = (
                    g.input(f'gdn_checkpoint{i}',
                            (linear_v_dim, linear_k_dim, linear_num_v_heads),
                            prec=Precision.FP32),
                    g.input(f'gdn_conv_checkpoint{i}',
                            (qkv_total, conv_kernel - 1),
                            prec=Precision.FP32),
                )
            cache_inputs.append(('gdn', gs, gc))

    input_norm_weights = [
        g.weight(
            os.path.join(
                weights_dir,
                f"model_language_model_layers_{i}_"
                "input_layernorm_weight.weights",
            ),
            (hidden_size,),
            Precision.FP32,
        )
        for i in range(num_layers)
    ]
    post_norm_weights = [
        g.weight(
            os.path.join(
                weights_dir,
                f"model_language_model_layers_{i}_"
                "post_attention_layernorm_weight.weights",
            ),
            (hidden_size,),
            Precision.FP32,
        )
        for i in range(num_layers)
    ]
    final_norm_weight = g.weight(
        os.path.join(weights_dir, "final_norm.weights"),
        (hidden_size,), Precision.FP32)

    mtp_hidden_output = None
    # ---- build layers ----
    x = hidden
    x_normed = g.rms_norm(x, input_norm_weights[0], eps=eps)
    for i in range(num_layers):
        lt = layer_types[i]
        ck_in, cv_in = (cache_inputs[i][1], cache_inputs[i][2]) if cache_inputs[i][0] == 'kv' else (None, None)
        gs_in, gc_in = (cache_inputs[i][1], cache_inputs[i][2]) if cache_inputs[i][0] == 'gdn' else (None, None)
        gs_checkpoint, gc_checkpoint = checkpoint_inputs[i]

        next_norm_weight = (
            input_norm_weights[i + 1]
            if i + 1 < num_layers
            else final_norm_weight
        )
        expose_residual = expose_mtp_hidden and i + 1 == num_layers
        x, x_normed, residual_output = _build_layer(
            g, x, x_normed, post_norm_weights[i], next_norm_weight,
            i, weights_dir, cos, sin, mask,
            ck_in, cv_in, gs_in, gc_in, gs_checkpoint, gc_checkpoint,
            eps, seq_len, rope_dim, rope_theta,
            num_heads, num_kv_heads, head_dim,
            linear_num_heads, linear_k_dim, linear_v_dim,
            linear_num_v_heads,
            conv_kernel, intermediate, hidden_size, lt,
            is_prefill=is_prefill, expose_residual=expose_residual,
            verification=verification)
        if residual_output is not None:
            mtp_hidden_output = residual_output

    x = x_normed
    if mtp_hidden_output is not None:
        g.set_metadata("mtp_hidden_output_id", mtp_hidden_output)

    print(f"  Total: {len(g._nodes)} nodes")
    return g


def _build_layer(g, x, x_normed, post_norm_weight, next_norm_weight,
                 layer_idx, weights_dir,
                 cos, sin, mask, ck_in, cv_in, gs_in, gc_in,
                 gs_checkpoint, gc_checkpoint,
                 eps, seq_len, rope_dim, rope_theta,
                 num_heads, num_kv_heads, head_dim,
                 linear_num_heads, linear_k_dim, linear_v_dim,
                 linear_num_v_heads,
                 conv_kernel, intermediate, hidden_size, layer_type,
                 is_prefill=False, expose_residual=False,
                 verification=False, weight_prefix=None):
    """Build one layer (linear_attention or full_attention)."""

    pfx_lm = weight_prefix or f'model_language_model_layers_{layer_idx}'

    if layer_type == 'linear_attention':
        attn_out = _build_linear_attn_layer(
            g, x_normed, layer_idx, weights_dir,
            gs_in, gc_in, gs_checkpoint, gc_checkpoint,
            eps, seq_len,
            linear_num_heads, linear_k_dim, linear_v_dim,
            linear_num_v_heads, conv_kernel, hidden_size,
            is_prefill=is_prefill, verification=verification,
            weight_prefix=f"{pfx_lm}_linear_attn")
    else:
        attn_out = _build_full_attn_layer(
            g, x_normed, layer_idx, weights_dir,
            cos, sin, mask, ck_in, cv_in,
            eps, seq_len, rope_dim,
            num_heads, num_kv_heads, head_dim, hidden_size,
            is_prefill=is_prefill,
            weight_prefix=f"{pfx_lm}_self_attn")

    # Update the residual stream in place while producing the normalized
    # activation for the MLP.
    x_normed2 = g.add_rms_norm(
        x, attn_out, post_norm_weight, eps=eps)

    # ---- MLP (SwiGLU) ----
    mlp_pfx = f'{pfx_lm}_mlp'
    w_gate_up = g.weight(
        os.path.join(weights_dir, f"{mlp_pfx}_gate_up_proj_weight.weights"),
        (2 * intermediate, hidden_size), Precision.FP16)
    w_down = g.weight(os.path.join(weights_dir, f"{mlp_pfx}_down_proj_weight.weights"),
                      (hidden_size, intermediate), Precision.FP16)

    gate_up = g.matmul(x_normed2, w_gate_up)
    mlp_hidden = g.swiglu(gate_up)
    mlp_out = g.matmul(mlp_hidden, w_down)
    if expose_residual:
        from transpile import SEQ
        residual = g.add(x, mlp_out)
        seq = SEQ.bind(seq_len) if is_prefill else seq_len
        residual_output = g.reshape(residual, (hidden_size, seq))
        next_x_normed = g.rms_norm(
            residual, next_norm_weight, eps=eps)
        return residual, next_x_normed, residual_output

    next_x_normed = g.add_rms_norm(
        x, mlp_out, next_norm_weight, eps=eps)
    return x, next_x_normed, None


def _build_linear_attn_layer(g, x, layer_idx, weights_dir,
                              gs_in, gc_in, gs_checkpoint, gc_checkpoint,
                              eps, seq_len,
                              num_heads, k_dim, v_dim, num_v_heads,
                              conv_kernel, hidden_size, is_prefill=False,
                              verification=False, weight_prefix=None,
                              output_gate_type="silu"):
    """Build a linear attention (Gated Delta Rule) layer.

    Uses the fused `gated_deltanet` op for the GDN core + RMSNormGated.
    See kernels/gdn.h for the op's data-layout contract — all matmul-derived
    inputs are consumed in their native [seq, dim] row-major data layout.
    """
    pfx = weight_prefix or f'model_language_model_layers_{layer_idx}_linear_attn'

    # ---- Fused input projections ----
    qkv_total = num_heads * k_dim * 2 + num_v_heads * v_dim
    z_total = num_v_heads * v_dim
    w_in = g.weight(
        os.path.join(weights_dir, f"{pfx}_in_proj_weight.weights"),
        (qkv_total + 2 * num_v_heads + z_total, hidden_size),
        Precision.FP16)
    projected = g.matmul(x, w_in)
    qkv, a_out, b_out, z_out = g.slice(
        projected,
        [qkv_total, num_v_heads, num_v_heads, z_total],
        dim=0)

    # ---- A_log and dt_bias (per-value-head constants for 4B, per-key-head for 0.8B) ----
    A_log = g.weight(os.path.join(weights_dir, f"{pfx}_A_log.weights"),
                     (num_v_heads,), Precision.FP32)
    dt_bias = g.weight(os.path.join(weights_dir, f"{pfx}_dt_bias.weights"),
                       (num_v_heads,), Precision.FP32)

    # ---- Short conv on qkv (conv1d + silu) ----
    # matmul output shape [qkv_total, seq], data [seq, qkv_total] (matmul convention).
    # ShortConv kernel reads data as [seq, groups] (x_data[s*groups + g]), so it
    # consumes the matmul output directly — no reshape/permute needed.
    # conv1d.weight: [6144, 1, 4] → reshape to [6144, 4] (groups=6144, kernel_size=4)
    w_conv = g.weight(os.path.join(weights_dir, f"{pfx}_conv1d_weight.weights"),
                     (qkv_total, conv_kernel), Precision.FP32)
    # ---- Fused GDN core + RMSNormGated ----
    # Replaces: split qkv, g/beta compute, GDN recurrence, RMSNormGated chain.
    # Output: shape [num_v_heads*v_dim, seq], data [seq, num_v_heads*v_dim] row-major.
    w_norm = g.weight(os.path.join(weights_dir, f"{pfx}_norm_weight.weights"),
                     (v_dim,), Precision.FP32)
    if verification:
        gated = g.gated_deltanet_conv_verify(
            qkv, a_out, b_out, z_out,
            A_log, dt_bias, w_norm, gs_in, w_conv, gc_in,
            gs_checkpoint, gc_checkpoint,
            num_heads=num_heads, k_dim=k_dim, v_dim=v_dim,
            seq_len=seq_len, num_v_heads=num_v_heads,
            conv_kernel=conv_kernel, use_qk_l2norm=True, rms_eps=eps)
    elif is_prefill:
        qkv_conv = g.shortconv(
            qkv, w_conv, gc_in, kernel_size=conv_kernel)
        # Prefill remains a separate ShortConv because its recurrent sequence
        # traversal has different parallelism from the GDN recurrence.
        gated = g.gated_deltanet(
            qkv_conv, a_out, b_out, z_out,
            A_log, dt_bias, w_norm, gs_in,
            num_heads=num_heads, k_dim=k_dim, v_dim=v_dim, seq_len=seq_len,
            use_qk_l2norm=True, rms_eps=eps,
            num_v_heads=num_v_heads,
            output_gate_type=output_gate_type)
    else:
        gated = g.gated_deltanet_conv_decode(
            qkv, a_out, b_out, z_out,
            A_log, dt_bias, w_norm, gs_in, w_conv, gc_in,
            num_heads=num_heads, k_dim=k_dim, v_dim=v_dim,
            num_v_heads=num_v_heads, conv_kernel=conv_kernel,
            use_qk_l2norm=True, rms_eps=eps,
            output_gate_type=output_gate_type)

    # ---- out_proj ----
    w_out = g.weight(os.path.join(weights_dir, f"{pfx}_out_proj_weight.weights"),
                     (hidden_size, num_v_heads * v_dim), Precision.FP16)
    out = g.matmul(gated, w_out)
    return out


def _build_full_attn_layer(g, x, layer_idx, weights_dir,
                            cos, sin, mask, ck_in, cv_in,
                            eps, seq_len, rope_dim,
                            num_heads, num_kv_heads, head_dim, hidden_size,
                            is_prefill=False, weight_prefix=None):
    """Build a full attention (GQA + QK norm + output gate) layer."""
    # In prefill graphs, use SEQ symbol for seq dims (runtime substitutes
    # actual seq_len). In decode graphs, use seq_len literal (=1, static).
    from transpile import SEQ
    _S = SEQ.bind(seq_len) if is_prefill else seq_len
    pfx = weight_prefix or f'model_language_model_layers_{layer_idx}_self_attn'

    # ---- Fused Q/K/V projection -----------------------------------------
    # q_proj contributes query+gate rows, followed by K and V. A single
    # projection lets Metal reuse the input activation and removes two GPU
    # dispatches per full-attention layer.
    qg_dim = num_heads * head_dim * 2
    kv_dim = num_kv_heads * head_dim
    w_qkv = g.weight(
        os.path.join(weights_dir, f"{pfx}_qkv_proj_weight.weights"),
        (qg_dim + 2 * kv_dim, hidden_size), Precision.FP16)
    qkv = g.matmul(x, w_qkv)
    qg, k, v = g.slice(qkv, [qg_dim, kv_dim, kv_dim], dim=0)

    # The converter stores all query heads before all output-gate heads.
    # Decode (S=1) can therefore use both halves as contiguous views. Prefill
    # materializes each strided slice once, without first copying all qg rows.
    query_dim = num_heads * head_dim
    query, gate = g.slice(qg, [query_dim, query_dim], dim=0)

    # ---- QK norm (RMSNorm per head) ----
    # q_norm/k_norm weight shape: [head_dim]
    # query/k are matmul outputs: declared [NH*HD, seq], data [seq, NH*HD] row-major.
    #
    # The RMSNorm kernel normalizes over dim0 (contiguous elements). We need
    # each (head, token) pair's head_dim=256 elements to be contiguous and
    # grouped into one "column" for RMSNorm.
    #
    # Correct pipeline:
    #   1. contiguous(matm_out) → reshape(HD, NH, seq)
    #      Matmul data layout [seq, NH*HD] matches [HD, NH, seq] row-major:
    #      element (d, h, s) at d + h*HD + s*HD*NH.
    #   2. permute(HD, NH, seq) → (HD, seq, NH) → contiguous
    #      Materializes as [HD, seq, NH] row-major. Each (s, h) pair's 256
    #      elements are at the correct positions.
    #   3. reshape(HD, seq, NH) → (HD, NH*seq)
    #      Columns: k = s + h*seq_len. This is s-major interleaved:
    #      s0_h0, s1_h0, s2_h0, s3_h0, s0_h1, s1_h1, ...
    #      Each column has the correct (s, h) pair's 256 elements.
    #   4. RMSNorm over dim0
    #   5. reshape(HD, NH*seq) → (HD, seq, NH)
    #      Maps column k back to (d, s=k%seq_len, h=k/seq_len).
    #      Since columns are s-major, this gives correct (s, h) pairs.
    #   6. RMSNorm's dense output already has the [HD, seq, NH] row-major
    #      storage required by RoPE/SDPA; the reshape does not require a copy.
    w_qn = g.weight(os.path.join(weights_dir, f"{pfx}_q_norm_weight.weights"),
                    (head_dim,), Precision.FP32)
    w_kn = g.weight(os.path.join(weights_dir, f"{pfx}_k_norm_weight.weights"),
                    (head_dim,), Precision.FP32)

    # No contiguous() before reshape/permute: reshape inherits stride and
    # permute is zero-copy. The fused Q/K norm+RoPE kernels consume both
    # strided views directly and materialize one dense combined output.
    query = g.reshape(query, (head_dim, num_heads, _S))  # [HD, NH, seq]
    query = g.permute(query, (0, 2, 1, 3))                   # [HD, seq, NH]
    query = g.reshape(query, (head_dim, num_heads * _S)) # [HD, NH*seq], cols s-major

    k = g.reshape(k, (head_dim, num_kv_heads, _S))       # [HD, NKV, seq]
    k = g.permute(k, (0, 2, 1, 3))                            # [HD, seq, NKV]
    k = g.reshape(k, (head_dim, num_kv_heads * _S))      # [HD, NKV*seq], cols s-major
    qk = g.qk_rms_norm_rope(
        query, k, w_qn, w_kn, cos, sin, _S, num_heads, num_kv_heads,
        rope_dim=rope_dim, interleave=False, eps=eps)
    query, k = g.slice(qk, [num_heads, num_kv_heads], dim=2)
    # v is matmul output: declared shape [nkv*hd, seq], data [seq, nkv*hd] row-major.
    # Step 1: reshape (hd, nkv, seq) — zero-copy, d0=hd innermost.
    #   flat[0..hd-1] = (d=0..hd-1, nkv=0, s=0) = (s=0, nkv=0, d=0..hd-1) ✓
    # Step 2: permute (hd, nkv, seq) → (hd, seq, nkv)
    # The SDPA cache-append kernel reads V through its position/head strides,
    # so the permuted view can flow through directly without materialization.
    v = g.reshape(v, (head_dim, num_kv_heads, _S))
    v = g.permute(v, (0, 2, 1, 3))

    # ---- SDPA ----
    scale = head_dim ** -0.5
    attn, ck_out, cv_out = g.sdpa(
        query, k, v, mask, ck_in, cv_in,
        kv_cache=2, causal=True, scale=scale,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, v_head_dim=head_dim)

    # ---- Output layout + sigmoid gate ----
    # Permuting attention to [HD,NH,seq] is a view. SIGMOID_MUL accepts strided
    # inputs and writes a dense output, so it simultaneously materializes the
    # target layout and applies the gate; a standalone CONTIGUOUS is unnecessary.
    attn = g.permute(attn, (0, 2, 1, 3))
    gate = g.reshape(gate, (head_dim, num_heads, _S))
    attn = g.sigmoid_mul(attn, gate)
    attn = g.reshape(attn, (num_heads * head_dim, _S))

    # ---- o_proj ----
    w_o = g.weight(os.path.join(weights_dir, f"{pfx}_o_proj_weight.weights"),
                   (hidden_size, num_heads * head_dim), Precision.FP16)
    out = g.matmul(attn, w_o)
    return out


def build_mtp_graph(weights_dir: str, cfg: dict, seq_len: int = 256,
                    n_ctx: int = 4096) -> GraphBuilder:
    """Build the single full-attention Qwen3.5 MTP decoder layer."""
    from transpile import DimExpr

    g = GraphBuilder()
    tc = cfg["text_config"] if "text_config" in cfg else cfg
    hidden_size = int(tc["hidden_size"])
    num_heads = int(tc["num_attention_heads"])
    num_kv_heads = int(tc["num_key_value_heads"])
    head_dim = int(tc["head_dim"])
    intermediate = int(tc["intermediate_size"])
    eps = float(tc.get("rms_norm_eps", 1e-6))
    rope_theta = float(tc["rope_parameters"]["rope_theta"])
    rope_dim = int(
        head_dim * tc["rope_parameters"]["partial_rotary_factor"])

    g.set_model_config(
        rope_dim=rope_dim, rope_theta=rope_theta,
        hidden_size=hidden_size, num_layers=1,
        vocab_size=int(tc["vocab_size"]), model_type="qwen3_5_mtp")

    const = DimExpr.const()
    seq = DimExpr.seq()
    hidden_dyn = (const, seq, const, const)
    target_hidden = g.input(
        "target_hidden", (hidden_size, seq_len), dynamic=hidden_dyn)
    token_hidden = g.input(
        "hidden", (hidden_size, seq_len), dynamic=hidden_dyn)
    mask = g.input("mask", (1, seq_len), dynamic=hidden_dyn)
    cos = g.input("cos", (rope_dim // 2, seq_len), dynamic=hidden_dyn)
    sin = g.input("sin", (rope_dim // 2, seq_len), dynamic=hidden_dyn)
    ck = g.input(
        "cache_k0", (head_dim, n_ctx, num_kv_heads),
        prec=Precision.FP16)
    cv = g.input(
        "cache_v0", (head_dim, n_ctx, num_kv_heads),
        prec=Precision.FP16)

    hidden_norm = g.weight(
        os.path.join(weights_dir, "mtp_pre_fc_norm_hidden_weight.weights"),
        (hidden_size,), Precision.FP32)
    embedding_norm = g.weight(
        os.path.join(
            weights_dir, "mtp_pre_fc_norm_embedding_weight.weights"),
        (hidden_size,), Precision.FP32)
    fc = g.weight(
        os.path.join(weights_dir, "mtp_fc_weight.weights"),
        (hidden_size, 2 * hidden_size), Precision.FP16)
    input_norm = g.weight(
        os.path.join(
            weights_dir, "mtp_layers_0_input_layernorm_weight.weights"),
        (hidden_size,), Precision.FP32)
    post_norm = g.weight(
        os.path.join(
            weights_dir,
            "mtp_layers_0_post_attention_layernorm_weight.weights"),
        (hidden_size,), Precision.FP32)
    final_norm = g.weight(
        os.path.join(weights_dir, "mtp_norm_weight.weights"),
        (hidden_size,), Precision.FP32)

    h = g.rms_norm(target_hidden, hidden_norm, eps=eps)
    e = g.rms_norm(token_hidden, embedding_norm, eps=eps)
    x = g.matmul(g.concat([e, h], dim=0), fc)
    x_normed = g.rms_norm(x, input_norm, eps=eps)
    _, output, _ = _build_layer(
        g, x, x_normed, post_norm, final_norm,
        0, weights_dir, cos, sin, mask,
        ck, cv, None, None, None, None,
        eps, seq_len, rope_dim, rope_theta,
        num_heads, num_kv_heads, head_dim,
        0, 0, 0, 0,
        0, intermediate, hidden_size, "full_attention",
        is_prefill=True, weight_prefix="mtp_layers_0")
    return g


def convert_qwen35(model_dir: str, output_path: str, num_layers: int | None = None,
                    prefill_seq_len: int = 256, n_ctx: int = 4096,
                    quant: str = "fp16", include_vision: bool = True):
    """Main entry point: export weights + build graphs → single .mollm file.

    Args:
        model_dir: path to HF model directory (with config.json + safetensors)
        output_path: output .mollm file path
        num_layers: deprecated; layer count is read from config.json
        prefill_seq_len: prefill sequence length
        n_ctx: max context length
        include_vision: package the FP16 vision tower when present
    """
    model_dir = Path(model_dir)
    quant = _canonical_quant(quant)
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="mollm_weights_")
    weights_dir = tmp_dir
    weights_rel = "."

    with open(model_dir / 'config.json') as f:
        cfg = json.load(f)
    tc = cfg['text_config']
    config_num_layers = tc['num_hidden_layers']
    if num_layers is not None and num_layers != config_num_layers:
        print(f"Warning: ignoring deprecated num_layers={num_layers}; "
              f"using config.json num_hidden_layers={config_num_layers}")
    num_layers = config_num_layers

    if (model_dir / "model.safetensors.index.json").exists():
        weights = _LazySafeTensors(model_dir)
    else:
        st_files = sorted(model_dir.glob('model.safetensors-*.safetensors'))
        if not st_files:
            st_files = list(model_dir.glob('model.safetensors'))
        if not st_files:
            raise FileNotFoundError(f"No safetensors file in {model_dir}")
        weights = {}
        for st_path in st_files:
            weights.update(load_safetensors(str(st_path)))
    mtp_layers = _validate_mtp_weights(weights, cfg)
    include_mtp = mtp_layers == 1 and quant == "fp16"
    if mtp_layers and not include_mtp:
        print(
            "Warning: Qwen3.5 MTP is currently packaged only for FP16; "
            f"converting the {quant} target model without MTP"
        )

    # ---- Step 1: Export weights to temp dir ----
    print("Exporting weights...")
    export_weights(
        weights, str(weights_dir), cfg, quant=quant,
        include_mtp=include_mtp)
    g_vision = None
    if include_vision and "vision_config" in cfg:
        print("Exporting vision weights...")
        export_vision_weights(weights, str(weights_dir), cfg)
        print("\nBuilding vision graph...")
        g_vision = build_vision_graph(weights_rel, cfg)

    # ---- Step 2: Build prefill graph ----
    print(f"\nBuilding prefill graph (seq_len={prefill_seq_len})...")
    g_prefill = build_graph(
        weights_rel, cfg, seq_len=prefill_seq_len, n_ctx=n_ctx,
        is_prefill=True, expose_mtp_hidden=include_mtp)

    # ---- Step 3: Build decode graph ----
    print(f"\nBuilding decode graph (seq_len=1)...")
    g_decode = build_graph(
        weights_rel, cfg, seq_len=1, n_ctx=n_ctx,
        expose_mtp_hidden=include_mtp)
    g_mtp = None
    g_mtp_verify = None
    if include_mtp:
        print("\nBuilding MTP verification graph...")
        g_mtp_verify = build_graph(
            weights_rel, cfg, seq_len=2, n_ctx=n_ctx, is_prefill=True,
            verification=True, expose_mtp_hidden=True)
        print("\nBuilding MTP graph...")
        g_mtp = build_mtp_graph(
            weights_rel, cfg, seq_len=prefill_seq_len, n_ctx=n_ctx)

    # ---- Step 4: Pack into single .mollm file ----
    print(f"\nPacking {output_path}...")
    fallback_model_name = f"Qwen3.5-{num_layers}L"
    metadata = {
        "model_name": infer_hf_model_name(model_dir, cfg, fallback_model_name),
        "architecture": "qwen3.5",
        "num_layers": num_layers,
        "hidden_size": tc['hidden_size'],
        "num_heads": tc['num_attention_heads'],
        "num_kv_heads": tc['num_key_value_heads'],
        "head_dim": tc['head_dim'],
        "prefill_seq_len": prefill_seq_len,
        "n_ctx": n_ctx,
        "vocab_size": tc['vocab_size'],
        "layer_types": tc['layer_types'],
        "quantization": quant,
        "mtp_num_hidden_layers": 1 if include_mtp else 0,
        "mtp_quantization": "fp16" if include_mtp else "none",
        "mtp_max_draft_tokens": 1 if include_mtp else 0,
        "mtp_transactional_state": 1 if include_mtp else 0,
    }
    if g_vision is not None:
        vc = cfg["vision_config"]
        processor_path = model_dir / "preprocessor_config.json"
        with open(processor_path) as f:
            processor_cfg = json.load(f)
        metadata.update({
            "vision": True,
            "image_token_id": cfg["image_token_id"],
            "vision_start_token_id": cfg["vision_start_token_id"],
            "vision_end_token_id": cfg["vision_end_token_id"],
            "vision_hidden_size": vc["hidden_size"],
            "vision_num_heads": vc["num_heads"],
            "vision_depth": vc["depth"],
            "vision_patch_size": vc["patch_size"],
            "vision_temporal_patch_size": vc["temporal_patch_size"],
            "vision_spatial_merge_size": vc["spatial_merge_size"],
            "vision_num_position_embeddings":
                vc["num_position_embeddings"],
            "mrope_section_t": tc["rope_parameters"]["mrope_section"][0],
            "mrope_section_h": tc["rope_parameters"]["mrope_section"][1],
            "mrope_section_w": tc["rope_parameters"]["mrope_section"][2],
            "vision_min_pixels": processor_cfg["size"]["shortest_edge"],
            "vision_max_pixels": processor_cfg["size"]["longest_edge"],
        })
    save_package(output_path, g_prefill, g_decode, weights_dir, metadata,
                 tokenizer_path=str(model_dir / "tokenizer.json"),
                 jinja_path=str(model_dir / "chat_template.jinja"),
                 g_vision=g_vision, g_mtp=g_mtp,
                 g_mtp_verify=g_mtp_verify)

    # Cleanup temp dir
    import shutil
    shutil.rmtree(tmp_dir)
    print(f"\nDone! Output: {output_path}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output.mollm> [quant=fp16|w8pc|w4g128|w4g32|w4mixg128|w4mixg32]")
        sys.exit(1)
    model_dir = sys.argv[1]
    output_path = sys.argv[2]
    quant = sys.argv[3] if len(sys.argv) > 3 else "fp16"
    convert_qwen35(model_dir, output_path, quant=quant)
