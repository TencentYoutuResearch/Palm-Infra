#!/usr/bin/env python3
"""Independent CPU PyTorch indexer oracle for bounded CUDA diagnostic dumps.

Equations follow Transformers Qwen4ExpTextQSAIndexer/RMSNorm/partial RoPE:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py
The dump begins AFTER projection. This is not whole-model BF16 certification,
nor a test of projection precision or earlier hidden-state drift. No CUDA
context is created by default, no weights are loaded, and no output file is
written. Optional CUDA reference runs must be serialized with model workloads.
"""
import argparse
import json
import math
import struct


def read_header(source):
    magic = source.read(8)
    if magic not in (b"QSAIDX1\0", b"QSAIDX2\0"):
        raise ValueError("invalid snapshot magic/version")
    header = source.read(44)
    if len(header) != 44:
        raise ValueError("truncated header")
    q, past, heads, dim, rope, ratio, budget, offset, stride, slots, eps = struct.unpack("<10If", header)
    bf16 = magic == b"QSAIDX2\0"
    if bf16:
        flags = source.read(4)
        if len(flags) != 4 or struct.unpack("<I", flags)[0] != 1:
            raise ValueError("invalid/truncated diagnostic precision flags")
    keys, groups = past + q, (past + q) // max(1, ratio)
    if not (1 <= q <= 2048 and 0 <= past and keys <= 8192 and 1 <= heads <= 16
            and 2 <= dim <= 256 and 2 <= rope <= dim and rope % 2 == 0
            and 1 <= ratio <= 16 and 1 <= budget <= 4096-ratio+1 and budget % ratio == 0
            and slots == budget+ratio-1 and stride >= max(1, groups)
            and offset <= 262144 and math.isfinite(eps) and eps > 0):
        raise ValueError("invalid/beyond-bound snapshot geometry")
    elements = q*((heads+1)*dim+heads*dim+groups+slots)+(keys+groups)*dim+2*dim+rope//2
    if elements > 64*1024*1024//4:
        raise ValueError("snapshot exceeds64MiB bound")
    return dict(q=q, past=past, heads=heads, dim=dim, rope=rope, ratio=ratio,
                budget=budget, offset=offset, eps=eps, keys=keys, groups=groups,
                elements=elements, slots=slots, diagnostic_bf16=bf16)


def read_snapshot(path, torch):
    with open(path, "rb") as source:
        result = read_header(source)
        q, heads, dim, rope, keys, groups, slots = (result[x] for x in
            ("q", "heads", "dim", "rope", "keys", "groups", "slots"))

        def tensor(shape, dtype=None):
            data = source.read(math.prod(shape)*4)
            if len(data) != math.prod(shape)*4:
                raise ValueError("truncated tensor")
            return torch.frombuffer(bytearray(data), dtype=dtype or torch.float32).clone().reshape(shape)

        for name, shape in (
            ("projection", (q, (heads+1)*dim)), ("qw", (dim,)), ("kw", (dim,)),
            ("frequency", (rope//2,)), ("raw", (keys, dim)), ("query", (q, heads, dim)),
            ("pooled", (groups, dim)), ("scores", (q, groups))):
            result[name] = tensor(shape)
        result["ids"] = tensor((q, slots), torch.int32)
        if source.read(1):
            raise ValueError("trailing snapshot bytes")
    for name in ("projection", "qw", "kw", "frequency", "raw", "query", "pooled"):
        if not torch.isfinite(result[name]).all():
            raise ValueError(f"nonfinite {name}")
    if not (result["frequency"] > 0).all() or groups == 0:
        raise ValueError("invalid frequency/empty group fixture")
    return result


def oracle(data, torch, dtype):
    device = data["projection"].device
    def norm(x, weight):
        value = x.float()
        normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True)+data["eps"])
        return (normalized*(1+weight.float())).to(x.dtype)

    def rope(x, positions):
        angle = positions.float()[:, None]*data["frequency"]
        cosine = torch.cat((angle.cos(), angle.cos()), -1).to(x.dtype)
        sine = torch.cat((angle.sin(), angle.sin()), -1).to(x.dtype)
        while cosine.ndim < x.ndim:
            cosine, sine = cosine.unsqueeze(1), sine.unsqueeze(1)
        prefix, suffix = x[..., :data["rope"]], x[..., data["rope"]:]
        half = data["rope"]//2
        rotated = torch.cat((-prefix[..., half:], prefix[..., :half]), -1)
        return torch.cat((prefix*cosine+rotated*sine, suffix), -1)

    q, heads, dim, groups, ratio = (data[x] for x in ("q", "heads", "dim", "groups", "ratio"))
    query = data["projection"][:, :heads*dim].to(dtype).reshape(q, heads, dim)
    query = rope(norm(query, data["qw"]), torch.arange(data["past"], data["keys"], device=device)+data["offset"])
    raw = data["raw"].to(dtype)
    pooled = raw[:groups*ratio].reshape(groups, ratio, dim).float().mean(1).to(dtype)
    pooled = rope(norm(pooled, data["kw"]), torch.arange(groups, device=device)*ratio+data["offset"])
    scores = torch.einsum("qhd,gd->qhg", query.float(), pooled.float()).relu().sum(1)/math.sqrt(dim)
    visible = (torch.arange(data["past"]+1, data["keys"]+1, device=device)//ratio)[:, None] > torch.arange(groups, device=device)[None, :]
    scores.masked_fill_(~visible, -math.inf)
    return query.float(), pooled.float(), scores, visible


def check(data, torch, *, selection_only=False, reference_device="cpu"):
    bf16_mode = data.get("diagnostic_bf16", False)
    dtype = torch.bfloat16 if bf16_mode else torch.float32
    if selection_only and not bf16_mode:
        raise ValueError("selection-only is limited to BF16 diagnostic snapshots")
    reference = {name: value.to(reference_device) if isinstance(value, torch.Tensor) else value
                 for name, value in data.items()}
    query, pooled, scores, visible = (value.cpu() for value in oracle(reference, torch, dtype))
    report = dict(queries=data["q"], past=data["past"], keys=data["keys"], groups=data["groups"],
                  reference_device=reference_device, selection_only=selection_only)
    current_raw = data["projection"][:, data["heads"]*data["dim"]:].to(dtype).float()
    if not torch.equal(data["raw"][data["past"]:], current_raw):
        raise ValueError("raw cache differs from current projected keys")
    value_gate = True
    for name, expected in (("query", query), ("pooled", pooled), ("scores", scores)):
        observed = data[name]
        if name == "scores":
            if not torch.isneginf(observed[~visible]).all():
                raise ValueError("future group score must be -inf")
            observed, expected = observed[visible], expected[visible]
        if not torch.isfinite(observed).all():
            raise ValueError(f"nonfinite visible {name}")
        report[name+"_max_abs"] = float((observed-expected).abs().max())
        report[name+"_exact_elements"] = int((observed == expected).sum())
        report[name+"_elements"] = observed.numel()
        if not torch.allclose(observed, expected, atol=3e-4, rtol=3e-5):
            value_gate = False
            if not selection_only:
                raise ValueError(f"{name} disagrees with independent {dtype} equations: {report[name+'_max_abs']}")

    bf_query, bf_pooled, bf_scores, _ = (value.cpu() for value in oracle(reference, torch, torch.bfloat16))
    report["bf16_post_projection_score_max_abs"] = float((bf_scores[visible]-scores[visible]).abs().max())
    strict_matches, bf_intersection, bf_total, bf_exact, near_cutoff = 0, 0, 0, 0, 0
    for row in range(data["q"]):
        count = data["past"]+row+1
        groups = count//data["ratio"]
        keep = min(data["budget"]//data["ratio"], groups)
        observed = data["ids"][row].tolist()
        valid = [value for value in observed if value >= 0]
        if (observed != valid+[-1]*(len(observed)-len(valid)) or valid != sorted(set(valid))
                or any(value >= count for value in valid)):
            raise ValueError(f"invalid padding/order/causal ID at row {row}")
        chosen = sorted(set(value//data["ratio"] for value in valid if value < groups*data["ratio"]))
        tokens = [block*data["ratio"]+i for block in chosen for i in range(data["ratio"])]
        tokens += list(range(groups*data["ratio"], count))
        if len(chosen) != keep or valid != sorted(tokens):
            raise ValueError(f"incomplete selected group/missing tail at row {row}")
        expected = torch.argsort(scores[row, :groups], descending=True, stable=True)[:keep].sort().values.tolist()
        strict_matches += chosen == expected
        if chosen != expected:
            near_cutoff += 1
            if selection_only:
                raise ValueError(f"selected set differs from independent BF16 reference at row {row}")
        if keep < groups and keep:
            omitted = torch.ones(groups, dtype=torch.bool)
            omitted[chosen] = False
            lowest = scores[row, chosen].min()
            highest = scores[row, :groups][omitted].max()
            # Permit only ordering changes within the same FP32 comparison
            # bound; report exact-set agreement separately, never as bit-exact.
            if highest > lowest+3e-4+3e-5*torch.maximum(lowest.abs(), highest.abs()):
                raise ValueError(f"selected set violates independent cutoff at row {row}")
        bf = torch.argsort(bf_scores[row, :groups], descending=True, stable=True)[:keep].sort().values.tolist()
        bf_exact += chosen == bf
        bf_intersection += len(set(chosen)&set(bf))
        bf_total += keep
    report.update(fp32_exact_selected_rows=strict_matches, fp32_near_cutoff_rows=near_cutoff,
                  bf16_post_projection_exact_selected_rows=bf_exact,
                  bf16_post_projection_group_overlap=bf_intersection/max(1, bf_total),
                  fp32_oracle_pass=value_gate, selected_set_pass=strict_matches == data["q"],
                  value_gate_pass=value_gate, whole_model_bf16_certified=False)
    if bf16_mode:
        report.pop("bf16_post_projection_score_max_abs")
        report["bf16_post_projection_oracle_pass"] = report.pop("fp32_oracle_pass")
        report["bf16_oracle_exact_selected_rows"] = report.pop("fp32_exact_selected_rows")
        report["bf16_oracle_near_cutoff_rows"] = report.pop("fp32_near_cutoff_rows")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--reference-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--selection-only", action="store_true",
                        help="BF16 diagnostics only: require exact selected sets, report value gate separately")
    args = parser.parse_args()
    if not 1 <= args.threads <= 64:
        parser.error("threads must be1..64")
    import torch
    torch.set_num_threads(args.threads)
    try:
        if args.reference_device == "cuda":
            import subprocess
            busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
            if busy.strip():
                raise ValueError("CUDA oracle requires an idle GPU; do not overlap model workloads")
        report = check(read_snapshot(args.snapshot, torch), torch,
                       selection_only=args.selection_only, reference_device=args.reference_device)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
