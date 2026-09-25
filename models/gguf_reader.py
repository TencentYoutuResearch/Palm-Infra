#!/usr/bin/env python3
"""Read a Hy4 GGUF checkpoint and expose it in the shape our graph asks for.

A GGUF file carries weights and hyperparameters but no compute graph, while a
`.mollm` package needs both. The graph already exists in `hy4.py`; what it is
missing is a checkpoint it can read. So this module does not build a second
graph. It presents the GGUF file through the same interface `hy4.py` already
uses for safetensors -- `SafeTensorIndex.tensor(name) -> slice` -- and maps the
`hyv4.*` metadata keys onto the `config.json` vocabulary. `build_graph` then
runs unmodified.

The payoff is that a GGUF checkpoint only has to get *tensor names and byte
offsets* right: every layout question the graph answers (transpose flags, fused
expert rows, cache widths) is shared with the safetensors path and already
tested.

Two GGUF properties drive the design:

  * Tensor dims are reversed relative to the (row-major) logical shape. A GGUF
    tensor declared ``[K, N]`` is our ``[N, K]``. Shapes returned by
    `GgufIndex.tensor` are therefore reversed back, and every producer in
    `hy4.py` asserts against the true logical shape.
  * Quantized expert tensors store the expert index as the *outer* dimension,
    with one contiguous run per expert. Fusing gate+up is then a block gather,
    not a dequantize-and-requantize.

The mapping is pinned by `models/verify_gguf_reader.py`, which checks the
translated config against the reference `config.json`, every reference tensor
against its GGUF counterpart, each type's block size against the file's own
tensor spacing, and the STQ1_0 decoder against the kernel the engine runs. None
of that writes a package: this stage exists to prove the mapping before any
bytes are repacked.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

from safetensors_stream import SafeTensorSlice
from transpile import WeightByteRange


GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (2, 3)

# GGUF metadata value type tags.
GGUF_UINT8, GGUF_INT8 = 0, 1
GGUF_UINT16, GGUF_INT16 = 2, 3
GGUF_UINT32, GGUF_INT32 = 4, 5
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
GGUF_STRING = 8
GGUF_ARRAY = 9
GGUF_UINT64, GGUF_INT64, GGUF_FLOAT64 = 10, 11, 12

# Type tag -> (struct format, item size, numpy dtype name for arrays).
_SCALAR_TYPES = {
    GGUF_UINT8: ("<B", 1), GGUF_INT8: ("<b", 1),
    GGUF_UINT16: ("<H", 2), GGUF_INT16: ("<h", 2),
    GGUF_UINT32: ("<I", 4), GGUF_INT32: ("<i", 4),
    GGUF_FLOAT32: ("<f", 4), GGUF_BOOL: ("<?", 1),
    GGUF_UINT64: ("<Q", 8), GGUF_INT64: ("<q", 8), GGUF_FLOAT64: ("<d", 8),
}
_ARRAY_DTYPES = {
    GGUF_UINT8: "<u1", GGUF_INT8: "<i1",
    GGUF_UINT16: "<u2", GGUF_INT16: "<i2",
    GGUF_UINT32: "<u4", GGUF_INT32: "<i4",
    GGUF_FLOAT32: "<f4", GGUF_BOOL: "?",
    GGUF_UINT64: "<u8", GGUF_INT64: "<i8", GGUF_FLOAT64: "<f8",
}


@dataclass(frozen=True)
class GgufTensorInfo:
    name: str
    # Dims exactly as stored in the file (fastest-varying first).
    dims: tuple[int, ...]
    type_id: int
    offset: int

    @property
    def logical_shape(self) -> tuple[int, ...]:
        """The row-major shape, i.e. the file's dims reversed."""
        return tuple(reversed(self.dims))


def _read_string(f: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", f.read(8))
    if length > (1 << 31):
        raise ValueError(f"implausible GGUF string length {length}")
    raw = f.read(length)
    if len(raw) != length:
        raise ValueError("truncated GGUF string")
    return raw.decode("utf-8", "replace")


def _read_scalar(f: BinaryIO, type_id: int):
    if type_id == GGUF_STRING:
        return _read_string(f)
    entry = _SCALAR_TYPES.get(type_id)
    if entry is None:
        raise ValueError(f"unknown GGUF metadata type {type_id}")
    fmt, size = entry
    raw = f.read(size)
    if len(raw) != size:
        raise ValueError("truncated GGUF scalar")
    return struct.unpack(fmt, raw)[0]


def _read_array(f: BinaryIO):
    element_type = struct.unpack("<I", f.read(4))[0]
    (count,) = struct.unpack("<Q", f.read(8))
    if element_type == GGUF_STRING:
        return [_read_string(f) for _ in range(count)]
    if element_type == GGUF_ARRAY:
        raise ValueError("nested GGUF arrays are not supported")
    dtype = _ARRAY_DTYPES.get(element_type)
    if dtype is None:
        raise ValueError(f"unknown GGUF array element type {element_type}")
    itemsize = np.dtype(dtype).itemsize
    raw = f.read(itemsize * count)
    if len(raw) != itemsize * count:
        raise ValueError("truncated GGUF array")
    return np.frombuffer(raw, dtype=dtype).copy()


class GgufFile:
    """Parse a GGUF header: metadata dict plus tensor directory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.metadata: dict[str, object] = {}
        self.tensors: dict[str, GgufTensorInfo] = {}
        self.data_start = 0
        self.alignment = 32
        self._parse()

    def _parse(self):
        with self.path.open("rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError(f"not a GGUF file: {self.path}")
            (version,) = struct.unpack("<I", f.read(4))
            if version not in SUPPORTED_VERSIONS:
                raise ValueError(
                    f"unsupported GGUF version {version} "
                    f"(supported: {SUPPORTED_VERSIONS})")
            (n_tensors,) = struct.unpack("<Q", f.read(8))
            (n_kv,) = struct.unpack("<Q", f.read(8))

            for _ in range(n_kv):
                key = _read_string(f)
                (type_id,) = struct.unpack("<I", f.read(4))
                value = (_read_array(f) if type_id == GGUF_ARRAY
                         else _read_scalar(f, type_id))
                self.metadata[key] = value

            for _ in range(n_tensors):
                name = _read_string(f)
                (n_dims,) = struct.unpack("<I", f.read(4))
                if n_dims > 4:
                    raise ValueError(
                        f"tensor {name} declares {n_dims} dims; GGUF allows "
                        f"at most 4")
                dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
                (type_id,) = struct.unpack("<I", f.read(4))
                (offset,) = struct.unpack("<Q", f.read(8))
                self.tensors[name] = GgufTensorInfo(
                    name=name, dims=tuple(dims), type_id=type_id,
                    offset=offset)

            self.alignment = int(
                self.metadata.get("general.alignment", 32) or 32)
            position = f.tell()
            self.data_start = (
                (position + self.alignment - 1) // self.alignment
            ) * self.alignment

    # -- convenience accessors -------------------------------------------

    def arch(self) -> str:
        return str(self.metadata.get("general.architecture", ""))

    def tensor_count(self) -> int:
        return len(self.tensors)

    def names(self, prefix: str = "") -> list[str]:
        return sorted(n for n in self.tensors if n.startswith(prefix))

    def families(self) -> dict[str, list[int]]:
        """Group ``blk.<n>.<suffix>`` names by suffix, for the audit view."""
        import re
        out: dict[str, list[int]] = {}
        for name in self.tensors:
            match = re.match(r"^blk\.(\d+)\.(.+)$", name)
            if match:
                out.setdefault(match.group(2), []).append(int(match.group(1)))
            else:
                out.setdefault(name, [])
        return out


# ---------------------------------------------------------------------------
# GGUF metadata -> Hy4 config.json
# ---------------------------------------------------------------------------
#
# Everything our graph reads is a scalar or a per-layer int/str list, so the
# mapping is a plain table. Keys are spelled out rather than derived from the
# GGUF key (strip "hyv4.", drop the trailing "_count", ...): a bad key fails
# loudly here instead of silently producing a wrong shape deep in the graph.

_MOE_LAYER_TYPES = ("dense", "sparse")


def _ggml_type_name(type_id: int) -> str:
    from gguf_quants import GGML_TYPE_NAMES
    name = GGML_TYPE_NAMES.get(type_id)
    return name if name is not None else f"unknown({type_id})"


def config_from_gguf(gguf: GgufFile) -> dict:
    """Translate GGUF ``hyv4.*`` metadata into the Hy4 config vocabulary."""
    m = gguf.metadata

    def need(key: str):
        if key not in m:
            raise KeyError(f"GGUF metadata is missing {key!r}")
        return m[key]

    layers = int(need("hyv4.block_count"))
    indexer_full_flags = need("hyv4.attention.indexer.is_full")
    if len(indexer_full_flags) != layers:
        raise ValueError(
            f"indexer.is_full has {len(indexer_full_flags)} entries but the "
            f"model has {layers} layers")
    indexer_types = [
        "full" if int(flag) else "shared" for flag in indexer_full_flags]

    # GGUF records only `leading_dense_block_count`; the reference config has a
    # per-layer list whose first `n` entries are "dense". Derive it so the list
    # length always matches block_count.
    dense_lead = int(need("hyv4.leading_dense_block_count"))
    if not 0 <= dense_lead <= layers:
        raise ValueError(
            f"leading_dense_block_count={dense_lead} exceeds {layers} layers")
    mlp_layer_types = ["dense"] * dense_lead + ["sparse"] * (layers - dense_lead)

    # GGUF folds rope_theta into a dotted key; the reference nests it.
    rope_theta = float(need("hyv4.rope.freq_base"))

    # GGUF names the *unscaled* sigmoid function by index; the reference has no
    # equivalent key, so it is validated by the loader rather than mapped.
    gating_func = int(need("hyv4.expert_gating_func"))

    # MLA geometry, and the one place a mechanical key-by-key translation goes
    # wrong.
    #
    # The authoritative relation lives in the reference configuration
    # (transformers/models/hy_v4/configuration_hy_v4.py):
    #
    #     qk_nope_head_dim = 192
    #     qk_rope_head_dim = 64
    #     qk_head_dim      = qk_nope_head_dim + qk_rope_head_dim   # 256
    #     head_dim         = qk_rope_head_dim
    #
    # `attention.key_length` (576) is the *cache* width, not any of these: it is
    # the 512-wide value latent plus the shared 64-wide rope, which is why it
    # neither equals qk_head_dim nor splits as nope + value.
    #
    # GGUF publishes `key_length_mla`/`value_length_mla`=256, which are the
    # decompressed per-head k/v widths and both happen to be 256 = qk_head_dim.
    # So 256 is `qk_head_dim` and `v_head_dim`, and the no-rope width is
    # qk_head_dim - rope = 192. Subtracting the rope dim from `key_length_mla`
    # instead gives 128, and reading `key_length_mla` as the no-rope width
    # directly gives 256; either silently breaks every q_b_proj and kv_b_proj
    # shape, so the relation is asserted rather than assumed.
    qk_nope = int(need("hyv4.attention.key_length_mla"))
    value_mla = int(need("hyv4.attention.value_length_mla"))
    rope_dim = int(need("hyv4.rope.dimension_count"))
    if qk_nope != value_mla:
        raise ValueError(
            f"hyv4.attention.key_length_mla ({qk_nope}) and "
            f"value_length_mla ({value_mla}) disagree; the reference derives "
            f"qk_nope_head_dim = qk_head_dim - rope from the key width alone, "
            f"so this mapping is unverified for asymmetric MLA")
    if qk_nope != 4 * rope_dim:
        raise ValueError(
            f"expected key_length_mla ({qk_nope}) to be 4x the rope dim "
            f"({rope_dim}) for Hy4's 256 = 192 + 64 split; the reference "
            f"derives the no-rope width from this relation, so it is asserted "
            f"before it is relied on")
    qk_head_dim = qk_nope
    qk_nope_head_dim = qk_head_dim - rope_dim

    cfg = {
        "model_type": "hy_v4",
        "architectures": ["HYV4ForCausalLM"],
        "hidden_size": int(need("hyv4.embedding_length")),
        "num_hidden_layers": layers,
        "intermediate_size": int(need("hyv4.feed_forward_length")),
        "num_attention_heads": int(need("hyv4.attention.head_count")),
        # MLA does not replicate K/V per query head: `head_count_kv` is 1
        # because all 64 heads read one shared 576-wide latent. The reference
        # config sets it to 8, which measures something else (it is inert
        # there), so the fields are deliberately *not* mapped to each other --
        # a reader that copies this key across carries a wrong number into the
        # cache geometry. `num_key_value_heads` is therefore reported as the
        # attention head count, which is what the graph uses.
        "num_key_value_heads": int(need("hyv4.attention.head_count")),
        "head_dim": rope_dim,
        "q_lora_rank": int(need("hyv4.attention.q_lora_rank")),
        "kv_lora_rank": int(need("hyv4.attention.kv_lora_rank")),
        "qk_head_dim": qk_head_dim,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": rope_dim,
        "v_head_dim": value_mla,
        "moe_intermediate_size": int(need("hyv4.expert_feed_forward_length")),
        "n_routed_experts": int(need("hyv4.expert_count")),
        "num_experts_per_tok": int(need("hyv4.expert_used_count")),
        "n_shared_experts": int(need("hyv4.expert_shared_count")),
        "norm_topk_prob": bool(need("hyv4.expert_weights_norm")),
        "routed_scaling_factor": float(need("hyv4.expert_weights_scale")),
        "first_k_dense_replace": dense_lead,
        "mlp_layer_types": mlp_layer_types,
        "rms_norm_eps": float(need("hyv4.attention.layer_norm_rms_epsilon")),
        "rope_parameters": {"rope_theta": rope_theta, "rope_type": "default"},
        "vocab_size": int(need("hyv4.vocab_size")),
        "max_position_embeddings": int(need("hyv4.context_length")),
        "hc_mult": int(need("hyv4.hyper_connection.count")),
        "hc_eps": float(need("hyv4.hyper_connection.epsilon")),
        "hc_magnitude": float(need("hyv4.hyper_connection.magnitude")),
        "index_n_heads": int(need("hyv4.attention.indexer.head_count")),
        "index_head_dim": int(need("hyv4.attention.indexer.key_length")),
        "index_topk": int(need("hyv4.attention.indexer.top_k")),
        "indexer_types": indexer_types,
        "layer_types": ["deepseek_sparse_attention"] * layers,
        "swiglu_limit": float(np.asarray(
            need("hyv4.swiglu_clamp_exp"), dtype=np.float32)[0]),
        "use_mla": True,
        "use_dsa": True,
        "gated_mla": True,
        "gating_type": "elementwise",
        "enable_ihc": True,
        "learnable_sink": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        # Kept for provenance; not consumed by the graph.
        "bos_token_id": int(m.get("tokenizer.ggml.bos_token_id", 0)),
        "eos_token_id": int(m.get("tokenizer.ggml.eos_token_id", 0)),
        "pad_token_id": int(m.get("tokenizer.ggml.padding_token_id", 0)),
        "expert_gating_func": gating_func,
        "expert_gating_func_name": ("sigmoid" if gating_func == 2
                                    else f"unsupported({gating_func})"),
    }
    return cfg


# ---------------------------------------------------------------------------
# TF -> torch name mapping
# ---------------------------------------------------------------------------
#
# GGUF spells a tensor differently from the reference checkpoint. The rules
# below reproduce the mapping audited against the 2134-tensor directory; every
# tensor the graph requests must hit exactly one rule.

# layer-suffix renames: the reference name (after `model.layers.<n>.`) becomes
# the GGUF name (after `blk.<n>.`).
_LAYER_SUFFIX_RENAMES = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_a_proj.weight": "attn_q_a.weight",
    "self_attn.q_a_layernorm.weight": "attn_q_a_norm.weight",
    "self_attn.q_b_proj.weight": "attn_q_b.weight",
    "self_attn.kv_a_proj_with_mqa.weight": "attn_kv_a_mqa.weight",
    "self_attn.kv_a_layernorm.weight": "attn_kv_a_norm.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.linear_gate.weight": "attn_gate.weight",
    "self_attn.learnable_sink_param": "attn_sinks.weight",
    "self_attn.indexer.wq_b.weight": "indexer.attn_q_b.weight",
    "self_attn.indexer.wk.weight": "indexer.attn_k.weight",
    "self_attn.indexer.weights_proj.weight": "indexer.proj.weight",
    "self_attn.indexer.k_norm.weight": "indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias": "indexer.k_norm.bias",
    "hc_attn_layer.hc_pre.hc_fn": "hc_attn_fn.weight",
    "hc_attn_layer.hc_pre.hc_scale": "hc_attn_scale.weight",
    "hc_attn_layer.hc_pre.hc_base": "hc_attn_base.weight",
    "hc_mlp_layer.hc_pre.hc_fn": "hc_ffn_fn.weight",
    "hc_mlp_layer.hc_pre.hc_scale": "hc_ffn_scale.weight",
    "hc_mlp_layer.hc_pre.hc_base": "hc_ffn_base.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.gate.e_score_correction_bias": "exp_probs_b.bias",
    "mlp.shared_experts.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_experts.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_experts.down_proj.weight": "ffn_down_shexp.weight",
}

_GLOBAL_RENAMES = {
    "model.embed_tokens.weight": "token_embd.weight",
    "lm_head.weight": "output.weight",
    "model.norm.weight": "output_norm.weight",
    "model.hc_head.hc_head_fn": "output_hc_fn.weight",
    "model.hc_head.hc_head_scale": "output_hc_scale.weight",
    "model.hc_head.hc_head_base": "output_hc_base.weight",
}


def torch_to_gguf(name: str) -> str | None:
    """Map one reference-checkpoint tensor name onto its GGUF spelling.

    Returns None for names GGUF stores differently enough to need a repack
    (`kv_b_proj`, split into `k_b`/`v_b`, and the fused expert tensors), so a
    caller that wants those handled explicitly can see them.
    """
    mapped = _GLOBAL_RENAMES.get(name)
    if mapped is not None:
        return mapped

    prefix = "model.layers."
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    layer_str, _, suffix = rest.partition(".")
    if not layer_str.isdigit():
        return None
    # The expert aggregates are fused at conversion time from separate GGUF
    # per-projection tensors; they are not a single rename.
    if suffix in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj",
                  "self_attn.kv_b_proj.weight"):
        return None
    renamed = _LAYER_SUFFIX_RENAMES.get(suffix)
    if renamed is None:
        raise KeyError(
            f"no GGUF spelling known for layer tensor {name!r} "
            f"(suffix {suffix!r})")
    return f"blk.{layer_str}.{renamed}"


# ---------------------------------------------------------------------------
# GgufIndex: the SafeTensorIndex-shaped view the graph builder consumes
# ---------------------------------------------------------------------------


class GgufIndex:
    """Expose a GGUF checkpoint through the safetensors accessor interface.

    Only the surface `hy4.py` uses is implemented: `tensor(name)`,
    `contains(name)`, and attribute access on the returned slice. Keeping the
    interface identical is what lets `build_graph` run unchanged on both
    checkpoint formats.
    """

    def __init__(self, gguf: GgufFile):
        self.gguf = gguf
        self.root = gguf.path.parent
        # Exposed for parity with SafeTensorIndex: callers that walk the
        # checkpoint rather than address it by name need the same entry point.
        self.metadata = gguf.metadata
        self._by_gguf_name = gguf.tensors
        # A reference name is present iff its GGUF spelling is.
        self._cache: dict[str, str | None] = {}

    def _resolve(self, name: str) -> str | None:
        if name not in self._cache:
            self._cache[name] = torch_to_gguf(name)
        return self._cache[name]

    def contains(self, name: str) -> bool:
        gguf_name = self._resolve(name)
        return gguf_name is not None and gguf_name in self._by_gguf_name

    def tensor(self, name: str) -> SafeTensorSlice:
        """Return a byte range for a reference-shaped tensor name.

        The returned dtype is a placeholder describing how the bytes must be
        interpreted; `precision_of` is the authority for the package writer.
        """
        gguf_name = self._resolve(name)
        if gguf_name is None or gguf_name not in self._by_gguf_name:
            raise KeyError(
                f"GGUF checkpoint has no tensor for {name!r} "
                f"(GGUF name {gguf_name!r})")
        info = self._by_gguf_name[gguf_name]
        from gguf_quants import ggml_type_info
        type_info = ggml_type_info(info.type_id)
        n_bytes = type_info.total_bytes(info.logical_shape)
        return SafeTensorSlice(
            name=name,
            dtype=type_info.dtype_tag,
            shape=info.logical_shape,
            source=WeightByteRange(
                str(self.gguf.path),
                self.gguf.data_start + info.offset,
                n_bytes),
        )

    def gguf_name_of(self, name: str) -> str | None:
        return self._resolve(name)

    def info(self, name: str) -> GgufTensorInfo:
        gguf_name = self._resolve(name)
        if gguf_name is None:
            raise KeyError(name)
        return self._by_gguf_name[gguf_name]


def open_gguf(path: str | Path) -> tuple[GgufFile, GgufIndex]:
    gguf = GgufFile(path)
    return gguf, GgufIndex(gguf)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Inspect a Hy4 GGUF and show its mapped config")
    parser.add_argument("gguf")
    parser.add_argument("--config", action="store_true",
                        help="print the translated Hy4 config.json")
    parser.add_argument("--families", action="store_true",
                        help="print tensor name families and their layer span")
    args = parser.parse_args()

    gguf = GgufFile(args.gguf)
    print(f"{args.gguf}")
    print(f"  arch      : {gguf.arch()}")
    print(f"  tensors   : {gguf.tensor_count()}")
    print(f"  metadata  : {len(gguf.metadata)} keys")
    print(f"  alignment : {gguf.alignment}")
    print(f"  data at   : {gguf.data_start}")

    if args.families:
        print("\ntensor families:")
        for suffix, layers in sorted(gguf.families().items()):
            span = (f"layers {min(layers)}..{max(layers)}" if layers
                    else "scalar")
            print(f"  {suffix:36} x{len(layers):4}  {span}")

    if args.config:
        print(json.dumps(config_from_gguf(gguf), indent=2, ensure_ascii=False))
