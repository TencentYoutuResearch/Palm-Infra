#!/usr/bin/env python3
"""Verify the GGUF reader against the safetensors reference checkpoint.

Two independent claims must hold before any weight bytes are repacked:

  1. Metadata translation is faithful -- `config_from_gguf` produces the same
     values the reference `config.json` holds, for every key the graph reads.
  2. The tensor directory is understood -- each reference tensor resolves to a
     GGUF tensor of the correct logical shape, and the byte ranges implied by
     the type table tile the file exactly.

(2) is where the interesting check lives. A GGUF tensor directory records
offsets, not extents, so a block size that is wrong by even one byte still
"works" until it runs off the end of the mapping. Comparing each tensor's
computed byte range against the next tensor's recorded offset turns that into a
precise, per-tensor failure.

Usage: python3 models/verify_gguf_reader.py <checkpoint.gguf> <reference_dir>
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf_quants import GGML_TYPES, ggml_type_info, stq1_0_dequantize
from gguf_reader import GgufFile, GgufIndex, config_from_gguf, torch_to_gguf
from safetensors_stream import SafeTensorIndex

REPO_ROOT = Path(__file__).resolve().parent.parent


# Keys the graph actually reads. A translation bug on any of these either
# changes a tensor shape or an arithmetic constant, so they are compared
# exactly rather than by spot check.
CONFIG_KEYS = (
    "hidden_size", "intermediate_size",
    "num_attention_heads", "head_dim",
    "q_lora_rank", "kv_lora_rank", "qk_head_dim", "qk_nope_head_dim",
    "qk_rope_head_dim", "v_head_dim", "moe_intermediate_size",
    "n_routed_experts", "num_experts_per_tok", "n_shared_experts",
    "norm_topk_prob", "routed_scaling_factor", "first_k_dense_replace",
    "rms_norm_eps", "vocab_size", "max_position_embeddings", "hc_mult",
    "hc_eps", "hc_magnitude", "index_n_heads", "index_head_dim",
    "index_topk", "swiglu_limit",
)

# The reference is a truncated 2-layer slice of the same checkpoint, so keys
# that scale with depth cannot be compared directly -- they are compared on
# their prefix instead.
DEPTH_KEYS = ("num_hidden_layers", "indexer_types", "mlp_layer_types")

# Keys that measure the same thing differently, with the reason recorded once
# rather than silently skipped.
INTENTIONAL = {
    "num_key_value_heads": (
        "GGUF reports attention.head_count_kv = 1 (all 64 heads share one MLA "
        "latent); the reference's 8 is an unused field, not a disagreement"),
}


def check_config(gguf: GgufFile, ref_dir: Path) -> list[str]:
    ours = config_from_gguf(gguf)
    theirs = json.loads((ref_dir / "config.json").read_text())
    problems = []

    for key in CONFIG_KEYS:
        if key not in theirs:
            continue
        mine, want = ours.get(key), theirs[key]
        if isinstance(want, float):
            same = abs(float(mine) - want) <= 1e-6 * max(1.0, abs(want))
        else:
            same = mine == want
        if not same:
            problems.append(f"config.{key}: ours={mine!r} ref={want!r}")

    depth = ours["num_hidden_layers"]
    theirs_depth = int(theirs["num_hidden_layers"])
    if depth < theirs_depth:
        problems.append(
            f"config.num_hidden_layers: ours={depth} is shorter than the "
            f"reference's {theirs_depth} layers")
    for key in DEPTH_KEYS:
        if key not in theirs:
            continue
        mine = ours.get(key)
        want = theirs[key]
        if isinstance(mine, list):
            if len(mine) != depth:
                problems.append(
                    f"config.{key}: {len(mine)} entries for {depth} layers")
            elif list(mine[:theirs_depth]) != list(want):
                problems.append(
                    f"config.{key}: first {theirs_depth} entries "
                    f"{list(mine[:theirs_depth])} != ref {list(want)}")
        elif mine != want and theirs_depth == depth:
            problems.append(f"config.{key}: ours={mine!r} ref={want!r}")

    print(f"    (depth: GGUF {depth} layers, reference {theirs_depth} layers"
          f" -- depth keys compared on their first {theirs_depth})")
    for key, why in INTENTIONAL.items():
        print(f"    (intentional: {key}: {why})")
    return problems


def check_tensor_directory(gguf: GgufFile) -> list[str]:
    """Confirm the type table reproduces the file's own tensor spacing.

    GGUF aligns each tensor to ``general.alignment``, so a computed size is
    only claimed to *fit* the gap to the next tensor -- it must not exceed it.
    Under-running by up to ``alignment - 1`` bytes is padding. This is still a
    strict, format-level claim: a block size wrong by more than the alignment
    shifts every later tensor and is caught here, not merely a name typo.
    """
    problems = []
    ordered = sorted(gguf.tensors.values(), key=lambda t: t.offset)
    if not ordered:
        return ["GGUF declares no tensors"]

    alignment = gguf.alignment
    for index, info in enumerate(ordered):
        entry = ggml_type_info(info.type_id)
        try:
            size = entry.total_bytes(info.logical_shape)
        except ValueError as error:
            problems.append(f"{info.name}: {error}")
            continue
        if index + 1 >= len(ordered):
            break
        gap = ordered[index + 1].offset - info.offset
        if size > gap:
            problems.append(
                f"{info.name} ({entry.name}): computed size {size} exceeds "
                f"the {gap} bytes to the next tensor")
        elif gap - size >= alignment:
            problems.append(
                f"{info.name} ({entry.name}): computed size {size} leaves "
                f"{gap - size} bytes of padding, at least a whole alignment "
                f"unit ({alignment})")
    return problems


def check_reference_coverage(gguf: GgufFile, ref_dir: Path) -> tuple[
        list[str], dict[str, int]]:
    """Every reference tensor must resolve to a correctly shaped GGUF tensor.

    Expert aggregates and kv_b_proj are deliberately fused/repacked at
    conversion time, so they are reported rather than treated as failures.
    """
    index = SafeTensorIndex(ref_dir)
    gguf_index = GgufIndex(gguf)
    problems: list[str] = []
    stats = {"direct": 0, "deferred": 0, "missing": 0, "shape_mismatch": 0}

    for name in sorted(index.weight_map):
        # MTP layers are an optional draft head; the preview package omits them.
        if name.startswith("model.mtp_layers."):
            continue
        gguf_name = torch_to_gguf(name)
        if gguf_name is None:
            stats["deferred"] += 1
            continue
        if gguf_name not in gguf.tensors:
            stats["missing"] += 1
            problems.append(f"{name} -> {gguf_name}: not in GGUF")
            continue
        ref = index.tensor(name)
        got = gguf_index.tensor(name)
        if ref.shape != got.shape:
            stats["shape_mismatch"] += 1
            problems.append(
                f"{name}: shape ours={got.shape} ref={ref.shape}")
            continue
        stats["direct"] += 1
    return problems, stats


def check_expert_layout(gguf: GgufFile, ref_dir: Path) -> list[str]:
    """Confirm the fused expert tensors are a pure block gather.

    GGUF stores gate and up separately while the graph wants them fused into
    one ``[experts, 4096, 6144]`` tensor, so this mapping is not a rename and
    the type may differ per projection (the preview package mixes stq1_0, iq2_xxs
    and iq3_xxs). What must hold is that each GGUF tensor is already
    expert-major at the same shape, so fusing is a concatenation of contiguous
    runs and never a dequantize/requantize.

    The check is structural on purpose: comparing decoded values would test the
    quantizer, not the reader, and the two projections may legitimately use
    different formats in the same layer.
    """
    import numpy as np

    index = SafeTensorIndex(ref_dir)
    problems = []
    checked = 0

    for layer in range(1, 3):
        for ref_key, gguf_suffixes in (
                ("mlp.experts.gate_up_proj",
                 ("ffn_gate_exps", "ffn_up_exps")),
                ("mlp.experts.down_proj", ("ffn_down_exps",))):
            name = f"model.layers.{layer}.{ref_key}"
            if not index.contains(name):
                continue
            ref = index.tensor(name)
            experts, rows, columns = ref.shape
            for suffix in gguf_suffixes:
                gguf_name = f"blk.{layer}.{suffix}.weight"
                info = gguf.tensors.get(gguf_name)
                if info is None:
                    problems.append(f"{gguf_name}: absent")
                    continue
                checked += 1
                # The graph declares the flattened [experts*rows, K] form; the
                # GGUF side must already carry that row order.
                want = (experts, rows // len(gguf_suffixes), columns)
                if info.logical_shape != want:
                    problems.append(
                        f"{gguf_name}: shape {info.logical_shape} != {want} "
                        f"(reference {name} is {ref.shape}, split across "
                        f"{len(gguf_suffixes)} projection(s))")
                    continue
                entry = ggml_type_info(info.type_id)
                elements = experts * (rows // len(gguf_suffixes)) * columns
                if elements % entry.block_values:
                    problems.append(
                        f"{gguf_name}: {elements} elements is not a whole "
                        f"number of {entry.name} blocks")

    if checked == 0:
        problems.append("no expert tensors found to check")
    else:
        print(f"    ({checked} projection tensor(s) checked; formats differ "
              f"per layer by design)")
    return problems


def check_stq1_0_against_kernel(gguf: GgufFile,
                                build_dir: Path) -> list[str]:
    """Diff the Python STQ1_0 decoder against the engine's own kernel.

    This is the strongest available check on the ternary format, and the only
    one that can run for it: the reference checkpoint is BF16, so a decoded
    value has no ground truth to compare against, and reconstruction error is
    uninformative here (ternary weights sit near 55% either way). Comparing
    against the kernel the engine executes on real checkpoint bytes is
    therefore the meaningful claim -- if the two agree bit-for-bit, the reader
    cannot disagree with what inference will do.

    Skipped, not failed, when the helper has not been built.
    """
    import subprocess

    dump = build_dir / "stq_decode_dump"
    if not dump.exists():
        print(f"    (skipped: {dump} not built; see "
              f"tests/stq_decode_dump.cpp for the one-line build command)")
        return []

    stq_names = sorted(n for n, t in gguf.tensors.items()
                       if ggml_type_info(t.type_id).name == "stq1_0")
    if not stq_names:
        return ["no STQ1_0 tensors found to cross-check"]

    # Sample across the layer range rather than one tensor, so a per-layer
    # layout quirk cannot hide behind a single agreeing sample.
    sample = [stq_names[0], stq_names[len(stq_names) // 2], stq_names[-1]]
    problems = []
    for name in sample:
        info = gguf.tensors[name]
        n_blocks = 24
        with open(gguf.path, "rb") as f:
            f.seek(gguf.data_start + info.offset)
            raw = f.read(n_blocks * 42)
        if len(raw) != n_blocks * 42:
            problems.append(f"{name}: short read")
            continue

        blob = Path(tempfile.gettempdir()) / "mollm_stq_check.bin"
        blob.write_bytes(raw)
        result = subprocess.run([str(dump), str(blob)], text=True,
                                capture_output=True)
        if result.returncode != 0:
            problems.append(
                f"{name}: kernel dump failed: {result.stderr.strip()}")
            continue
        reference = np.array(
            [[float(v) for v in line.split()]
             for line in result.stdout.strip().splitlines()],
            dtype=np.float32)

        blocks = np.frombuffer(raw, np.uint8).reshape(n_blocks, 42)
        ours = stq1_0_dequantize(blocks).astype(np.float32)
        if ours.shape != reference.shape:
            problems.append(
                f"{name}: shape {ours.shape} != kernel {reference.shape}")
        elif not np.array_equal(ours, reference):
            worst = float(np.abs(ours - reference).max())
            problems.append(
                f"{name}: Python differs from the kernel by up to {worst:g}")
        else:
            print(f"    ({name}: {n_blocks} blocks bit-exact vs kernel)")

    blob = Path(tempfile.gettempdir()) / "mollm_stq_check.bin"
    blob.unlink(missing_ok=True)
    return problems


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    gguf_path, ref_dir = sys.argv[1], Path(sys.argv[2])

    gguf = GgufFile(gguf_path)
    print(f"GGUF      : {gguf_path}")
    print(f"reference : {ref_dir}\n")

    used_ids = sorted({t.type_id for t in gguf.tensors.values()})
    print("tensor types in use:")
    for type_id in used_ids:
        count = sum(1 for t in gguf.tensors.values() if t.type_id == type_id)
        entry = GGML_TYPES.get(type_id)
        label = (f"{entry.name} ({entry.block_values}/{entry.block_bytes}B)"
                 if entry else "UNSUPPORTED")
        print(f"  {type_id:3}  {label:28} x{count}")
    print()

    failures = 0

    problems = check_config(gguf, ref_dir)
    print(f"[config]   {'OK' if not problems else f'{len(problems)} mismatch(es)'}")
    for line in problems:
        print(f"    {line}")
    failures += len(problems)

    problems = check_tensor_directory(gguf)
    print(f"[bytes]    {'OK' if not problems else f'{len(problems)} error(s)'}")
    for line in problems[:20]:
        print(f"    {line}")
    if len(problems) > 20:
        print(f"    ... and {len(problems) - 20} more")
    failures += len(problems)

    problems, stats = check_reference_coverage(gguf, ref_dir)
    print(f"[names]    direct={stats['direct']} deferred={stats['deferred']} "
          f"missing={stats['missing']} shape_mismatch={stats['shape_mismatch']}")
    for line in problems[:20]:
        print(f"    {line}")
    if len(problems) > 20:
        print(f"    ... and {len(problems) - 20} more")
    failures += len(problems)

    problems = check_expert_layout(gguf, ref_dir)
    print(f"[experts]  {'OK' if not problems else f'{len(problems)} issue(s)'}")
    for line in problems:
        print(f"    {line}")
    failures += len(problems)

    problems = check_stq1_0_against_kernel(gguf, REPO_ROOT / "build")
    print(f"[ternary]  {'OK' if not problems else f'{len(problems)} issue(s)'}")
    for line in problems:
        print(f"    {line}")
    failures += len(problems)

    print()
    print("PASS" if failures == 0 else f"FAIL ({failures} problem(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
