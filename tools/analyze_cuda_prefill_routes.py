#!/usr/bin/env python3
"""Validate complete expert-count traces and report route-slot fill, not GPU utilization."""
import argparse
import json
import re
from pathlib import Path

PATTERN = re.compile(r"nvfp4_expert_counts node=(\d+) rows=(\d+) unmatched=(\d+) counts=(\d+(?:,\d+)*)")


def validate_fused_token_trace(text, records):
    """Require every traced layer to use shared-token, fused activation input."""
    if infer_gate_path(text, records) != "fused-token":
        raise ValueError("fused-token accounting requires complete unmixed runtime counters")


def infer_gate_path(text, records):
    if records <= 0:
        raise ValueError("positive record count required")
    counters = []
    for name in ("prefill shared token quantization", "NVFP4 MMA activation fused prefill",
                 "NVFP4 BF16 gate output prefill", "prefill compact gate logits",
                 "NVFP4 BF16 down output prefill"):
        matches = re.findall(r"^CudaBackend: " + re.escape(name) + r" calls=(\d+)$", text, re.M)
        if len(matches) > 1:
            raise ValueError("duplicate gate path counter")
        counters.append(int(matches[0]) if matches else 0)
    token, fused, bf16, compact, down = counters
    # Current BF16-down admission proves canonical N8 geometry; BF16-gate
    # and fused activation admission prove gate N32. Compact logits are
    # required to account for the fused shared intermediate as BF16.
    if down != records:
        raise ValueError("complete BF16 down/N8 runtime counter required")
    if token == records and fused == records and compact == records and not bf16:
        return "fused-token"
    if fused == records and compact == records and not token and not bf16:
        return "fused-route"
    if bf16 == records and not token and not fused and not compact:
        return "route-bf16"
    raise ValueError("missing/mixed gate path counters; no single dataflow mode qualified")


def qwen38_nvfp4_dataflow(report, gate_path="route-bf16"):
    """Logical payload accounting for canonical gate N32/down N8, not DRAM traffic.

    Each route batch traverses one expert matrix, tiled across output rows.
    Count its unique codes + K16 scales once; caches, duplicate lane reads,
    descriptors, row globals and transaction granularity are not modeled.
    Route-BF16 quantization is expert-dependent. The fused-token path shares
    codes only with a uniform gate input global, and keeps gate logits in shared
    memory rather than publishing a global BF16 intermediate.
    """
    if gate_path not in ("route-bf16", "fused-route", "fused-token"):
        raise ValueError("unknown gate input path")
    hidden, inter = 2560, 640
    gate_payload = (2 * inter * hidden // 2) + (2 * inter * hidden // 16)
    down_payload = (hidden * inter // 2) + (hidden * inter // 16)
    chunks = []
    for chunk in report["chunks"]:
        routes = chunk["routes"]
        active = chunk["active_expert_layer_pairs"]
        gate_batches = chunk["batch_widths"]["32"]["batches"]
        down_batches = chunk["batch_widths"]["8"]["batches"]
        quant_units = chunk["rows"] * chunk["layer_records"] if gate_path == "fused-token" else 2 * routes
        chunks.append({
            "chunk": chunk["chunk"], "rows": chunk["rows"],
            "gate_weight_batch_payload_bytes": gate_batches * gate_payload,
            "down_weight_batch_payload_bytes": down_batches * down_payload,
            "active_weight_once_payload_bytes": active * (gate_payload + down_payload),
            "gate_routes_per_weight_traversal": routes / gate_batches,
            "down_routes_per_weight_traversal": routes / down_batches,
            "gate_quant_source_requested_bytes": quant_units * hidden * 4,
            "gate_quant_packed_and_scale_write_bytes": quant_units * hidden * 9 // 16,
            "gate_bf16_valid_intermediate_bytes": routes * 2 * inter * 2,
            "gate_bf16_output_write_bytes": routes * 2 * inter * 2 if gate_path == "route-bf16" else 0,
            "activation_packed_and_scale_write_bytes": routes * inter * 9 // 16,
            "down_bf16_output_write_bytes": routes * hidden * 2,
        })
    totals = {key: sum(chunk[key] for chunk in chunks)
              for key in chunks[0] if key.endswith("_bytes")}
    return {"scope": "logical_payload_not_measured_DRAM_bytes_or_speedup",
            "gate_input_path": gate_path,
            "assumptions": "Qwen3.8 hidden2560/inter640; gateN32/downN8; K16 NVFP4; "
                           f"{gate_path} gate quantization; BF16 gate/down; packed-only activation; "
                           "no cache/descriptor/duplicate-lane/transaction accounting",
            "chunks": chunks, "total": totals}


def analyze(text, chunk_rows, layers, experts=512, top_k=10):
    if not chunk_rows or any(n <= 0 for n in chunk_rows) or min(layers, experts, top_k) <= 0:
        raise ValueError("positive chunk rows/layers/experts/top-k required")
    records = []
    for line in text.splitlines():
        if "nvfp4_expert_counts" not in line:
            continue
        match = PATTERN.fullmatch(line)
        if not match:
            raise ValueError("malformed expert-count record")
        node, rows, unmatched = map(int, match.groups()[:3])
        counts = list(map(int, match[4].split(",")))
        if unmatched or len(counts) != experts or sum(counts) != rows * top_k:
            raise ValueError("unmatched expert, incorrect expert count or route total")
        records.append((node, rows, counts))
    if len(records) != len(chunk_rows) * layers:
        raise ValueError("missing/extra layer records")
    reports = []
    node_order = None
    for index, rows in enumerate(chunk_rows):
        chunk = records[index * layers:(index + 1) * layers]
        nodes = [r[0] for r in chunk]
        if any(r[1] != rows for r in chunk) or len(set(nodes)) != layers:
            raise ValueError("incorrect chunk order or duplicate layer node")
        if node_order is not None and nodes != node_order:
            raise ValueError("layer node order changed between chunks")
        node_order = nodes
        counts = [n for _, _, group in chunk for n in group]
        routes = sum(counts)
        sizes = {}
        for width in (8, 16, 32, 64):
            batches = sum((n + width - 1) // width for n in counts)
            sizes[str(width)] = {"batches": batches,
                                 "route_slot_fill_pct": 100 * routes / (width * batches),
                                 "full_batches": sum(n // width for n in counts)}
        # Gate N32 skips entirely absent N8 subgroups. N32 slot fill alone
        # therefore overstates empty MMA work; neither quantity is utilization.
        layer_reports = []
        for node, _, group in chunk:
            active_counts = sorted(n for n in group if n)
            layer_routes = sum(group)
            n8 = sum((n + 7) // 8 for n in group)
            n32 = sum((n + 31) // 32 for n in group)
            layer_reports.append({
                "node": node, "active_experts": len(active_counts),
                "routes": layer_routes, "n8_batches": n8, "n32_batches": n32,
                "n8_route_slot_fill_pct": 100 * layer_routes / (8 * n8),
                "n32_route_slot_fill_pct": 100 * layer_routes / (32 * n32),
                "active_expert_count_median": active_counts[len(active_counts) // 2],
                "maximum_routes_per_expert": active_counts[-1],
                "expert_count_ranges": {
                    "1_8": sum(1 <= n <= 8 for n in group),
                    "9_16": sum(9 <= n <= 16 for n in group),
                    "17_32": sum(17 <= n <= 32 for n in group),
                    "33_64": sum(33 <= n <= 64 for n in group),
                    "above64": sum(n > 64 for n in group),
                },
            })
        reports.append({"chunk": index, "rows": rows, "layer_records": layers,
                        "routes": routes, "active_expert_layer_pairs": sum(n > 0 for n in counts),
                        "routes_in_experts_above32_pct": 100 * sum(n for n in counts if n > 32) / routes,
                        "routes_in_experts_above64_pct": 100 * sum(n for n in counts if n > 64) / routes,
                        "maximum_routes_per_expert": max(counts), "batch_widths": sizes,
                        "layers": layer_reports,
                        "gate_n32_active_n8_subgroups": sizes["8"]["batches"],
                        "gate_n32_skipped_n8_subgroups":
                            4 * sizes["32"]["batches"] - sizes["8"]["batches"]})
    return {"scope": "arithmetic_route_slot_fill_not_tensor_utilization_or_speedup",
            "records": len(records), "chunks": reports}


def merge_routing_windows(text, chunk_rows, layers, experts=512, top_k=10):
    """Counterfactual batching of frozen histograms, not a model scheduler.

    Validation first proves matching layer order, complete records and totals.
    Actual activations/routing can change with chunk geometry; no speedup or
    DRAM inference follows from summing these expert counts.
    """
    split = analyze(text, chunk_rows, layers, experts, top_k)
    records = [PATTERN.fullmatch(line) for line in text.splitlines()
               if "nvfp4_expert_counts" in line]
    merged_lines = []
    for layer in range(layers):
        counts = [0] * experts
        for chunk in range(len(chunk_rows)):
            values = list(map(int, records[chunk * layers + layer][4].split(",")))
            counts = [a + b for a, b in zip(counts, values)]
        merged_lines.append(f"nvfp4_expert_counts node={records[layer][1]} "
                            f"rows={sum(chunk_rows)} unmatched=0 counts=" +
                            ",".join(map(str, counts)))
    merged = analyze("\n".join(merged_lines), [sum(chunk_rows)], layers, experts, top_k)
    comparison = {}
    for width in (8, 16, 32, 64):
        before = sum(c["batch_widths"][str(width)]["batches"] for c in split["chunks"])
        after = merged["chunks"][0]["batch_widths"][str(width)]["batches"]
        comparison[str(width)] = {"split_batches": before, "merged_batches": after,
                                  "batch_reduction_pct": 100 * (1 - after / before)}
    return {"scope": "counterfactual_frozen_histograms_not_model_schedule_DRAM_or_speedup",
            "split_chunk_rows": chunk_rows, "merged": merged["chunks"][0],
            "batch_comparison": comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--chunk-rows", required=True, help="ordered comma-separated actual chunk lengths")
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--merge-routing-windows", action="store_true",
                        help="counterfactual frozen-histogram window merge, not a model execution")
    parser.add_argument("--qwen38-nvfp4-dataflow", action="store_true",
                        help="also estimate current gateN32/downN8 logical payload; not DRAM traffic")
    parser.add_argument("--qwen38-gate-path", choices=("auto", "route-bf16", "fused-route", "fused-token"), default="auto",
                        help="dataflow mode (default: qualify from complete counters); explicit modes are assumptions except fused-token")
    args = parser.parse_args()
    if args.qwen38_gate_path != "auto" and not args.qwen38_nvfp4_dataflow:
        parser.error("gate path requires --qwen38-nvfp4-dataflow")
    try:
        report = analyze(args.log.read_text(), [int(n) for n in args.chunk_rows.split(",")],
                         args.layers, args.experts, args.top_k)
        if args.qwen38_nvfp4_dataflow:
            if args.experts != 512 or args.top_k != 10 or args.layers != 48:
                raise ValueError("Qwen3.8 accounting requires 48 layers, 512 experts, top-k10")
            gate_path = infer_gate_path(args.log.read_text(), report["records"]) if args.qwen38_gate_path == "auto" else args.qwen38_gate_path
            if gate_path == "fused-token":
                validate_fused_token_trace(args.log.read_text(), report["records"])
            report["qwen38_nvfp4_dataflow"] = qwen38_nvfp4_dataflow(report, gate_path)
            report["qwen38_nvfp4_dataflow"]["gate_path_evidence"] = (
                "complete_runtime_counters" if args.qwen38_gate_path == "auto" or gate_path == "fused-token"
                else "explicit_mode_assumption_not_runtime_qualification")
        if args.merge_routing_windows:
            report["window_merge"] = merge_routing_windows(args.log.read_text(),
                [int(n) for n in args.chunk_rows.split(",")], args.layers, args.experts, args.top_k)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
