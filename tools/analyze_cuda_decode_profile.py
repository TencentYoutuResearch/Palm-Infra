#!/usr/bin/env python3
"""Summarize a final decode or prefill window from an Nsight SQLite export.

The caller must establish that each token has exactly one matching LM-head
kernel, and supply the final run's decode count (excluding its prefill token).
Kernel-time sums are not uninstrumented benchmark measurements.
For --prefill-chunks, the caller must establish one matching LM-head per
chunk and that the final events are prefill, not decode. This includes all
specified chunks rather than silently reporting only the final tail chunk.
For --prefill-prompts, one matching head must terminate each complete prompt,
including all interleaved windows of a layer-major scheduler. The caller must
verify this contract; kernel names alone do not establish phase boundaries.
CUDA Graph runs must be captured with --cuda-graph-trace=node; whole-graph
intervals without their kernel events cannot establish kernel percentages.
"""
import argparse
from bisect import bisect_right
from contextlib import closing
from pathlib import Path
import sqlite3


def kernel_overlap_lookup(intervals):
    """Exact integer overlap queries over sorted, merged kernel intervals."""
    starts, ends, prefix = [], [], [0]
    for a, b in intervals:
        if b <= a or (ends and a < ends[-1]):
            raise ValueError("kernel intervals must be sorted and nonoverlapping")
        starts.append(a)
        ends.append(b)
        prefix.append(prefix[-1] + b - a)

    def covered_before(point):
        index = bisect_right(starts, point) - 1
        if index < 0:
            return 0
        return prefix[index] + min(point, ends[index]) - starts[index]

    def overlap(a, b):
        return covered_before(b) - covered_before(a) if b > a else 0

    return overlap


def kernel_family(name):
    """Name-based time grouping, not a hardware bottleneck classifier."""
    if "sdpa_" in name:
        return "attention"
    if "nvfp4_n32_fused_activation_cuda" in name or "nvfp4_n32_async_weights_cuda" in name:
        return "moe_gate"
    if "nvfp4_native_batched8_mma_cuda" in name:
        return "moe_down"
    if "gdn_recurrence" in name:
        return "gdn_recurrence"
    if "hc_up_reduce_fused_cuda" in name:
        return "hc_up_reduce"
    if "moe_combine_gr_inject_rms_deferred_cuda" in name:
        return "moe_hc_inject_fused"
    if "gr_inject" in name:
        return "hc_inject"
    if "gdn_" in name or "shortconv_" in name:
        return "recurrent_other"
    if any(part in name for part in ("quantize_nvfp4", "build_nvfp4_batches", "group_nvfp4_jobs",
                                    "moe_resident_routes", "nvfp4_token_input_map", "moe_activation", "moe_combine")):
        return "moe_auxiliary"
    if "prepare_q4" in name or "prepare_q8" in name:
        return "dense_prepare"
    if "cutlass" in name or "nvjet" in name or "gemm" in name:
        return "dense_gemm"
    return "other"


def summarize_families(rows):
    totals = {}
    for name, count, elapsed, *_ in rows:
        value = totals.setdefault(kernel_family(name), [0, 0])
        value[0] += count
        value[1] += elapsed
    return totals


def attention_pv_gemm_cost(events, expected_pairs):
    """Attribute same-stream bind->copy kernels under the verified PV source contract."""
    if expected_pairs <= 0:
        raise ValueError("positive expected PV pairs required")
    streams = {}
    for event in events:
        if event[1] <= event[0]:
            raise ValueError("invalid kernel interval")
        streams.setdefault(tuple(event[3:]), []).append(event)
    selected, pairs = [], 0
    for stream_events in streams.values():
        pending, enclosed = None, []
        for start, end, name, *_ in sorted(stream_events):
            if "sdpa_pv_bind_cuda" in name:
                if pending is not None:
                    raise ValueError("PV bind without preceding copy")
                pending, enclosed = end, []
            elif "sdpa_pv_copy_cuda" in name:
                if pending is None or start < pending or not enclosed:
                    raise ValueError("PV copy without nonempty ordered bind range")
                if any(kernel_end > start for _, kernel_end, _ in enclosed):
                    raise ValueError("kernel crosses PV copy boundary")
                selected.extend(enclosed)
                pairs += 1
                pending = None
            elif pending is not None:
                if start < pending or kernel_family(name) != "dense_gemm":
                    raise ValueError("unexpected kernel inside PV range")
                enclosed.append((start, end, name))
        if pending is not None:
            raise ValueError("missing PV copy")
    if pairs != expected_pairs:
        raise ValueError("missing/extra PV bind/copy pairs")
    return dict(pairs=pairs, events=len(selected),
                kernel_ns=sum(end-start for start, end, _ in selected))


def serial_moe_shared_cost(events, expected_pairs):
    """Infer down->combine shared ranges under a caller-verified serial contract.

    Events are (start,end,name,device,context,stream). This is not an overlap
    experiment or a general recognizer for fused/multi-stream MoE paths.
    """
    if expected_pairs <= 0 or not events:
        raise ValueError("positive expected MoE pairs and nonempty events required")
    if len({tuple(event[3:]) for event in events}) != 1:
        raise ValueError("serial shared inference requires one device/context/stream")
    ordered = sorted(events)
    pairs, pending = [], None
    for event in ordered:
        start, end, name = event[:3]
        if end <= start:
            raise ValueError("invalid kernel interval")
        if "nvfp4_native_batched8_mma_cuda" in name:
            if pending is not None:
                raise ValueError("down without preceding combine")
            pending = end
        elif "moe_combine_cuda" in name:
            if pending is None or start < pending:
                raise ValueError("combine without ordered down")
            pairs.append((pending, start))
            pending = None
    if pending is not None or len(pairs) != expected_pairs:
        raise ValueError("missing/extra MoE down/combine pairs")
    selected = [event for event in ordered
                if any(event[0] >= a and event[1] <= b for a, b in pairs)]
    # The verified source path has real shared kernels, not empty markers.
    for a, b in pairs:
        if not any(event[0] >= a and event[1] <= b for event in selected):
            raise ValueError("empty shared interval")
        if any(event[0] < b and event[1] > a and
               not (event[0] >= a and event[1] <= b) for event in ordered):
            raise ValueError("kernel crosses shared interval boundary")
    return {"pairs": len(pairs), "events": len(selected),
            "kernel_ns": sum(event[1] - event[0] for event in selected),
            "span_ns": sum(b - a for a, b in pairs)}


def kernel_interval_coverage(intervals, start, end):
    """Clipped interval union and no-kernel gaps; not device utilization."""
    if end <= start:
        raise ValueError("invalid interval window")
    cursor = start
    covered = 0
    gaps = []
    for a, b in sorted(intervals):
        a, b = max(start, a), min(end, b)
        if b <= a:
            continue
        if a > cursor:
            gaps.append(a - cursor)
        if b > cursor:
            covered += b - max(cursor, a)
            cursor = b
    if cursor < end:
        gaps.append(end - cursor)
    return covered, gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    window = parser.add_mutually_exclusive_group(required=True)
    window.add_argument("--decode-tokens", type=int)
    window.add_argument("--prefill-chunks", type=int)
    window.add_argument("--prefill-prompts", type=int,
                        help="complete prompts with one final head each, including all scheduled windows")
    parser.add_argument("--head-pattern", default="%q8_per_channel_dense_gemv%")
    parser.add_argument("--expected-head-events", type=int,
                        help="reject missing/extra matching heads in the entire capture; "
                             "caller still establishes their semantic meaning")
    parser.add_argument("--group-by-grid", action="store_true",
                        help="split each kernel by grid/block launch dimensions")
    parser.add_argument("--kernel-families", action="store_true",
                        help="name-based kernel-time groups; not hardware bottleneck attribution")
    parser.add_argument("--kernel-coverage", action="store_true",
                        help="report interval union and gaps, not hardware utilization")
    parser.add_argument("--attention-pv-pairs", type=int,
                        help="attribute verified same-stream PV bind/copy ranges; exact expected pairs")
    parser.add_argument("--runtime-api", action="store_true",
                        help="report clipped host API time and its kernel-overlapped portion")
    parser.add_argument("--serial-moe-shared", action="store_true",
                        help="infer down->combine costs; caller must verify serial shared-branch source contract")
    parser.add_argument("--expected-moe-events", type=int,
                        help="exact expected down/combine pairs for --serial-moe-shared")
    args = parser.parse_args()
    prompts = args.prefill_prompts is not None
    prefill = prompts or args.prefill_chunks is not None
    units = args.prefill_prompts if prompts else args.prefill_chunks if prefill else args.decode_tokens
    if units <= 0:
        parser.error("window count must be positive")
    if args.expected_head_events is not None and args.expected_head_events <= 0:
        parser.error("expected head-event count must be positive")
    if args.attention_pv_pairs is not None and args.attention_pv_pairs <= 0:
        parser.error("expected PV pair count must be positive")
    if args.serial_moe_shared and (not prefill or not args.expected_moe_events or args.expected_moe_events <= 0):
        parser.error("serial shared inference requires prefill and positive expected MoE events")
    if args.expected_moe_events is not None and not args.serial_moe_shared:
        parser.error("expected MoE events requires --serial-moe-shared")
    with closing(sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        joined = "CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName"
        heads = db.execute(f"SELECT k.start,k.end FROM {joined} WHERE s.value LIKE ? ORDER BY k.start",
                           (args.head_pattern,)).fetchall()
        if args.expected_head_events is not None and len(heads) != args.expected_head_events:
            parser.error(f"matching LM-head event count {len(heads)} differs from "
                         f"expected {args.expected_head_events}")
        if len(heads) <= units:
            parser.error("not enough LM-head events to establish a preceding token boundary")
        start, end = heads[-units - 1][1], heads[-1][1]
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        columns = {row[1] for row in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        geometry = ("gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ")
        if args.group_by_grid and not set(geometry).issubset(columns):
            parser.error("kernel export lacks grid/block launch dimensions")
        if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables:
        # A coarse graph event may coexist with ordinary kernel events.
            # Reject only graphs overlapping this decode window whose nodes
            # are not represented, rather than interpreting them as GPU idle.
            graph_events = db.execute("""SELECT start,end,deviceId,contextId,streamId
                FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE WHERE start<? AND end>?""",
                                      (end, start)).fetchall()
            for graph_start, graph_end, device, context, stream in graph_events:
                represented = "graphNodeId" in columns and db.execute("""
                    SELECT 1 FROM CUPTI_ACTIVITY_KIND_KERNEL
                    WHERE start>=? AND end<=? AND deviceId=? AND contextId=?
                    AND streamId=? AND graphNodeId IS NOT NULL AND graphNodeId!=0
                    LIMIT 1""", (graph_start, graph_end, device, context, stream)).fetchone()
                if not represented:
                    parser.error("decode window contains a CUDA Graph without node kernel events; "
                                 "recapture with --cuda-graph-trace=node")
        extra = "".join(",k." + field for field in geometry) if args.group_by_grid else ""
        rows = db.execute(f"""SELECT s.value,COUNT(*),SUM(k.end-k.start){extra}
            FROM {joined} WHERE k.start>=? AND k.end<=?
            GROUP BY s.value{extra} ORDER BY SUM(k.end-k.start) DESC""", (start, end)).fetchall()
        total = sum(row[2] for row in rows)
        if total <= 0 or end <= start:
            parser.error("empty decode window")
        if args.attention_pv_pairs is not None:
            events = db.execute(f"""SELECT k.start,k.end,s.value,k.deviceId,k.contextId,k.streamId
                FROM {joined} WHERE k.start>=? AND k.end<=? ORDER BY k.start""", (start, end)).fetchall()
            try:
                pv = attention_pv_gemm_cost(events, args.attention_pv_pairs)
            except ValueError as error:
                parser.error(str(error))
            print(f"attention_pv_pairs={pv['pairs']} events={pv['events']} "
                  f"kernel_ms={pv['kernel_ns']/1e6:.6f} "
                  "scope=source_verified_same_stream_bind_to_copy_not_utilization")
            families = summarize_families(rows)
            print(f"attention_including_pv_ms={(families.get('attention',[0,0])[1]+pv['kernel_ns'])/1e6:.6f} "
                  f"dense_gemm_excluding_pv_ms={(families.get('dense_gemm',[0,0])[1]-pv['kernel_ns'])/1e6:.6f}")
        if args.serial_moe_shared:
            events = db.execute(f"""SELECT k.start,k.end,s.value,k.deviceId,k.contextId,k.streamId
                FROM {joined} WHERE k.start>=? AND k.end<=? ORDER BY k.start""", (start, end)).fetchall()
            try:
                shared = serial_moe_shared_cost(events, args.expected_moe_events)
            except ValueError as error:
                parser.error(str(error))
            print(f"serial_moe_shared_pairs={shared['pairs']} events={shared['events']} "
                  f"kernel_ms={shared['kernel_ns']/1e6:.6f} span_ms={shared['span_ns']/1e6:.6f} "
                  f"kernel_over_window_pct={shared['kernel_ns']/(end-start)*100:.6f} "
                  "scope=source_verified_serial_down_to_combine_not_overlap_speedup")
        count_label = "prefill_prompts" if prompts else "prefill_chunks" if prefill else "decode_tokens"
        unit_label = "prompt" if prompts else "chunk" if prefill else "token"
        print(f"{count_label}={units} window_ms={(end-start)/1e6:.3f} "
              f"summed_gpu_ms={total/1e6:.3f} gpu_time_per_{unit_label}_ms={total/1e6/units:.3f} "
              f"kernel_sum_over_window_pct={total/(end-start)*100:.3f}")
        print(f"matching_head_events_total={len(heads)} selected_window_heads={units} "
              "scope=matching_kernel_events_not_semantic_phase_validation")
        if args.kernel_coverage:
            intervals = db.execute("""SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL
                WHERE start<? AND end>? ORDER BY start""", (end, start)).fetchall()
            covered, gaps = kernel_interval_coverage(intervals, start, end)
            print(f"kernel_union_ms={covered/1e6:.3f} "
                  f"kernel_union_window_pct={covered/(end-start)*100:.3f} "
                  f"no_kernel_ms={sum(gaps)/1e6:.3f} "
                  f"no_kernel_per_{unit_label}_us={sum(gaps)/1000/units:.3f} "
                  "scope=kernel_intervals_not_hardware_utilization")
            print("largest_no_kernel_gaps_us=" + ",".join(
                f"{gap/1000:.3f}" for gap in sorted(gaps, reverse=True)[:10]))
        if args.kernel_families:
            print("kernel_family_scope=name_based_summed_kernel_time_not_utilization")
            print("family\tinstances\tkernel_ms\twindow_pct")
            for family, (count, elapsed) in sorted(summarize_families(rows).items(), key=lambda x: x[1][1], reverse=True):
                print(f"{family}\t{count}\t{elapsed/1e6:.6f}\t{elapsed/(end-start)*100:.6f}")
        print("GPU%\tinstances\taverage_us\t" + ("grid/block\t" if args.group_by_grid else "") + "name")
        for name, count, elapsed, *launch in rows:
            dimensions = ("x".join(map(str, launch[:3])) + "/" +
                          "x".join(map(str, launch[3:])) + "\t") if args.group_by_grid else ""
            print(f"{elapsed/total*100:.3f}\t{count}\t{elapsed/count/1000:.3f}\t{dimensions}{name}")
        if args.runtime_api:
            if "CUPTI_ACTIVITY_KIND_RUNTIME" not in tables:
                parser.error("export lacks CUDA runtime API events")
            intervals = db.execute("""SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL
                WHERE start<? AND end>? ORDER BY start""", (end, start)).fetchall()
            # Merge once before intersecting many API intervals. Host waiting
            # for GPU completion is not evidence of transfer or launch cost.
            merged = []
            for a, b in intervals:
                a, b = max(a, start), min(b, end)
                if b <= a:
                    continue
                if merged and a <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
                else:
                    merged.append((a, b))
            totals = {}
            kernel_overlap = kernel_overlap_lookup(merged)
            for name, a, b in db.execute("""SELECT s.value,r.start,r.end
                    FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
                    WHERE r.start<? AND r.end>?""", (end, start)):
                a, b = max(a, start), min(b, end)
                if b <= a:
                    continue
                overlap = kernel_overlap(a, b)
                value = totals.setdefault(name, [0, 0, 0])
                value[0] += 1
                value[1] += b-a
                value[2] += overlap
            print("runtime_api_scope=host_intervals_not_transfer_cost; "
                  "sums_may_overlap_across_host_threads; no_kernel_is_not_causal_overhead")
            print("instances\thost_ms\tkernel_overlap_ms\tno_kernel_ms\tAPI")
            for name, (count, elapsed, overlap) in sorted(
                    totals.items(), key=lambda item: item[1][1], reverse=True):
                print(f"{count}\t{elapsed/1e6:.6f}\t{overlap/1e6:.6f}\t"
                      f"{(elapsed-overlap)/1e6:.6f}\t{name}")


if __name__ == "__main__":
    main()
