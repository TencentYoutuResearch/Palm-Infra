"""Offline coverage of serial Nsight decode-window validation."""
import sqlite3
from contextlib import closing
import subprocess
import sys
import tempfile
import unittest
import runpy
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools/analyze_cuda_decode_profile.py"


class DecodeProfileTest(unittest.TestCase):
    def test_indexed_kernel_overlap_matches_exhaustive_integer_reference(self):
        build = runpy.run_path(str(SCRIPT))["kernel_overlap_lookup"]
        # Boundaries, gaps, adjacent intervals and large Nsight timestamps;
        # retain integer nanoseconds rather than rounding through floats.
        for offset in (0, 2**60):
            for intervals in ([], [(2, 5)], [(2, 5), (5, 9), (12, 14)],
                              [(3*i, 3*i+2) for i in range(40)]):
                shifted = [(a+offset, b+offset) for a, b in intervals]
                overlap = build(shifted)
                for a in range(-2, 124):
                    for b in range(a, 125):
                        expected = sum(max(0, min(b+offset, y)-max(a+offset, x))
                                       for x, y in shifted)
                        self.assertEqual(overlap(a+offset, b+offset), expected)
        for intervals in ([(2, 2)], [(4, 3)], [(2, 5), (4, 8)], [(8, 9), (2, 3)]):
            with self.assertRaises(ValueError):
                build(intervals)

    def test_attention_pv_attribution_is_stream_scoped(self):
        function = runpy.run_path(str(SCRIPT))["attention_pv_gemm_cost"]
        events = [(1, 2, "sdpa_pv_bind_cuda", 0, 1, 7),
                  (3, 8, "cutlass::Kernel2", 0, 1, 7),
                  (3, 9, "nvjet_other_stream", 0, 1, 8),
                  (10, 11, "sdpa_pv_copy_cuda", 0, 1, 7)]
        self.assertEqual(function(events, 1), dict(pairs=1, events=1, kernel_ns=5))

    def test_attention_pv_rejects_incomplete_or_ambiguous_ranges(self):
        function = runpy.run_path(str(SCRIPT))["attention_pv_gemm_cost"]
        bind = (1, 2, "sdpa_pv_bind_cuda", 0, 1, 7)
        gemm = (3, 8, "cutlass::Kernel2", 0, 1, 7)
        copy = (10, 11, "sdpa_pv_copy_cuda", 0, 1, 7)
        cases = [[], [bind], [copy], [bind, copy], [bind, bind, gemm, copy],
                 [bind, (3, 12, gemm[2], 0, 1, 7), copy],
                 [bind, (3, 8, "unrelated_cuda", 0, 1, 7), copy],
                 [bind, (3, 8, gemm[2], 0, 1, 8), copy]]
        for events in cases:
            with self.subTest(events=events), self.assertRaises(ValueError):
                function(events, 1)
        with self.assertRaises(ValueError):
            function([bind, gemm, copy], 2)
        with self.assertRaises(ValueError):
            function([bind, gemm, copy], 0)

    def test_name_families_and_exact_totals(self):
        module = runpy.run_path(str(SCRIPT))
        families = {
            "void sdpa_prefill_pair_queries_cuda": "attention",
            "validate_sdpa_prefix_cuda": "attention",
            "nvfp4_n32_fused_activation_cuda": "moe_gate",
            "nvfp4_native_batched8_mma_cuda": "moe_down",
            "gdn_recurrence_quads_128_cuda": "gdn_recurrence",
            "hc_up_reduce_fused_cuda": "hc_up_reduce",
            "gr_inject_group_rms_half_cuda": "hc_inject",
            "moe_combine_gr_inject_rms_deferred_cuda": "moe_hc_inject_fused",
            "gdn_prepare_qk_128_cuda": "recurrent_other",
            "shortconv_prefill_cuda": "recurrent_other",
            "quantize_nvfp4_jobs_vector_cuda": "moe_auxiliary",
            "moe_combine_cuda": "moe_auxiliary",
            "prepare_q4_g32_dense_gemm_cuda": "dense_prepare",
            "cutlass::Kernel2": "dense_gemm",
            "nvjet_sm121_hss_mma": "dense_gemm",
            "contiguous_cuda": "other",
        }
        for name, expected in families.items():
            self.assertEqual(module["kernel_family"](name), expected)
        rows = [(name, 2, 1000, 7, 8) for name in families]
        rows.append(("another_unknown", 3, 11))
        summary = module["summarize_families"](rows)
        self.assertEqual(summary["other"], [5, 1011])
        self.assertEqual(sum(value[0] for value in summary.values()), 2*len(families)+3)
        self.assertEqual(sum(value[1] for value in summary.values()), 1000*len(families)+11)

    def shared_cost(self, events, pairs=1):
        function = runpy.run_path(str(SCRIPT))["serial_moe_shared_cost"]
        return function([(*event, 0, 1, 7) for event in events], pairs)

    def test_serial_shared_exact_ranges(self):
        events = [(0, 2, "head"), (3, 5, "nvfp4_native_batched8_mma_cuda"),
                  (6, 10, "gemm"), (11, 13, "activation"), (14, 16, "moe_combine_cuda"),
                  (17, 19, "outside"), (20, 23, "nvfp4_native_batched8_mma_cuda"),
                  (24, 26, "gemm"), (27, 29, "moe_combine_cuda")]
        self.assertEqual(self.shared_cost(events, 2),
                         dict(pairs=2, events=3, kernel_ns=8, span_ns=13))

    def test_serial_shared_rejects_bad_order_and_count(self):
        down = (1, 3, "nvfp4_native_batched8_mma_cuda")
        combine = (7, 8, "moe_combine_cuda")
        for events in ([], [down], [combine], [down, combine],
                       [down, (4, 5, down[2]), combine],
                       [down, (2, 4, "crossing"), (5, 6, "gemm"), combine]):
            with self.subTest(events=events), self.assertRaises(ValueError):
                self.shared_cost(events)
        with self.assertRaises(ValueError):
            self.shared_cost([down, (4, 6, "gemm"), combine], 2)

    def test_serial_shared_rejects_multiple_streams(self):
        function = runpy.run_path(str(SCRIPT))["serial_moe_shared_cost"]
        with self.assertRaisesRegex(ValueError, "one device/context/stream"):
            function([(1, 3, "nvfp4_native_batched8_mma_cuda", 0, 1, 7),
                      (4, 6, "gemm", 0, 1, 8), (7, 8, "moe_combine_cuda", 0, 1, 7)], 1)

    def analyze(self, graph=None, node=False, graph_table=True, geometry=False, grouped=False, coverage=False, runtime=False, runtime_table=False, prefill_chunks=None, prefill_prompts=None, extra_args=None):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "trace.sqlite"
            with closing(sqlite3.connect(database)) as db, db:
                db.executescript("""
                    CREATE TABLE StringIds(id INTEGER,value TEXT);
                    INSERT INTO StringIds VALUES(1,'q8_per_channel_dense_gemv'),(2,'expert');
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                        start INTEGER,end INTEGER,demangledName INTEGER,
                        deviceId INTEGER,contextId INTEGER,streamId INTEGER,graphNodeId INTEGER);
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                        (0,10,1,0,1,7,NULL),(90,100,1,0,1,7,NULL);
                """)
                if graph_table:
                    db.execute("""CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(
                        start INTEGER,end INTEGER,deviceId INTEGER,contextId INTEGER,streamId INTEGER)""")
                if runtime_table:
                    db.executescript("""
                        INSERT INTO StringIds VALUES(3,'cudaMemcpy'),(4,'outside');
                        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER);
                        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
                            (0,120,3),(30,50,3),(0,9,4),(101,120,4);
                    """)
                if graph:
                    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES(?,?,0,1,7)", graph)
                if node:
                    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(20,80,2,0,1,7,42)")
                if prefill_chunks is not None:
                    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(190,200,1,0,1,7,NULL)")
                    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(120,180,2,0,1,7,43)")
                if geometry:
                    for field, value in (("gridX", 1), ("gridY", 1), ("gridZ", 1),
                                         ("blockX", 32), ("blockY", 1), ("blockZ", 1)):
                        db.execute(f"ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN {field} INTEGER DEFAULT {value}")
                    if node:
                        db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=40,gridX=10,blockX=256 WHERE demangledName=2")
                        db.execute("""INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
                            (start,end,demangledName,deviceId,contextId,streamId,graphNodeId,gridX,blockX)
                            VALUES(40,80,2,0,1,7,43,20,128)""")
            return subprocess.run([sys.executable, str(SCRIPT), str(database),
                                   *( ["--prefill-prompts", str(prefill_prompts)] if prefill_prompts is not None else
                                      ["--decode-tokens", "1"] if prefill_chunks is None else
                                      ["--prefill-chunks", str(prefill_chunks)] )] + (["--group-by-grid"] if grouped else []) +
                                  (["--kernel-coverage"] if coverage else []) +
                                  (["--runtime-api"] if runtime else []) + (extra_args or []),
                                  capture_output=True, text=True)

    def test_prefill_spans_both_chunks(self):
        result = self.analyze(node=True, prefill_chunks=2, coverage=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("prefill_chunks=2", result.stdout)
        self.assertNotIn("decode_tokens=", result.stdout)
        self.assertIn("gpu_time_per_chunk_ms=", result.stdout)
        self.assertIn("\t2\t0.060\texpert", result.stdout)
        self.assertIn("kernel_union_window_pct=73.684", result.stdout)

    def test_prefill_prompt_has_one_final_head(self):
        result = self.analyze(node=True, prefill_prompts=1, coverage=True,
                              extra_args=["--expected-head-events", "2"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("prefill_prompts=1", result.stdout)
        self.assertIn("gpu_time_per_prompt_ms=", result.stdout)
        self.assertNotIn("prefill_chunks=", result.stdout)
        self.assertIn("kernel_union_window_pct=77.778", result.stdout)

    def test_prefill_prompt_rejects_invalid_or_ambiguous_contract(self):
        for count, extra in ((0, []), (-1, []), (1, ["--expected-head-events", "3"]),
                             (1, ["--prefill-chunks", "1"]), (1, ["--decode-tokens", "1"])):
            result = self.analyze(node=True, prefill_prompts=count, extra_args=extra)
            self.assertNotEqual(result.returncode, 0)

    def test_serial_shared_cli_contract(self):
        for arguments in (["--serial-moe-shared"],
                          ["--serial-moe-shared", "--expected-moe-events", "0"],
                          ["--expected-moe-events", "1"]):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.analyze(extra_args=arguments).returncode, 2)
        result = self.analyze(node=True, prefill_chunks=2,
                              extra_args=["--serial-moe-shared", "--expected-moe-events", "1"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing/extra MoE", result.stderr)

    def test_head_event_count_exact(self):
        result = self.analyze(node=True, prefill_chunks=2,
                              extra_args=["--expected-head-events", "3"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("matching_head_events_total=3 selected_window_heads=2", result.stdout)

    def test_head_event_count_rejects_missing_and_extra(self):
        for count in (2, 4):
            result = self.analyze(node=True, prefill_chunks=2,
                                  extra_args=["--expected-head-events", str(count)])
            self.assertEqual(result.returncode, 2)
            self.assertIn("matching LM-head event count 3 differs from expected", result.stderr)

    def test_head_event_count_requires_positive(self):
        for count in (0, -1):
            result = self.analyze(extra_args=["--expected-head-events", str(count)])
            self.assertEqual(result.returncode, 2)
            self.assertIn("expected head-event count must be positive", result.stderr)

    def test_prefill_requires_preceding_boundary(self):
        result = self.analyze(prefill_chunks=3)
        self.assertEqual(result.returncode, 2)
        self.assertIn("not enough LM-head", result.stderr)

    def test_prefill_invalid_count_and_mixed_mode(self):
        self.assertEqual(self.analyze(prefill_chunks=0).returncode, 2)
        self.assertEqual(self.analyze(prefill_chunks=2, extra_args=["--decode-tokens", "1"]).returncode, 2)

    def test_interval_union_clips_and_merges(self):
        coverage = runpy.run_path(str(SCRIPT))["kernel_interval_coverage"]
        self.assertEqual(coverage([(80,120),(20,60),(30,40),(-5,15),(50,70)],10,100),
                         (75,[5,10]))
        self.assertEqual(coverage([],10,100),(0,[90]))
        self.assertEqual(coverage([(0,120)],10,100),(90,[]))
        with self.assertRaises(ValueError):
            coverage([],100,10)

    def test_runtime_clips_and_intersects_kernel_union(self):
        result = self.analyze(node=True, runtime=True, runtime_table=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("2\t0.000110\t0.000090\t0.000020\tcudaMemcpy", result.stdout)
        self.assertNotIn("\toutside", result.stdout)
        self.assertIn("not_transfer_cost", result.stdout)

    def test_runtime_missing_table(self):
        result = self.analyze(runtime=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("lacks CUDA runtime", result.stderr)

    def test_interval_coverage_does_not_double_count(self):
        result = self.analyze(node=True,geometry=True,coverage=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("kernel_union_window_pct=77.778",result.stdout)
        self.assertIn("not_hardware_utilization",result.stdout)

    def test_non_graph_export(self):
        result = self.analyze(graph_table=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("kernel_sum_over_window_pct=11.111", result.stdout)

    def test_node_export(self):
        result = self.analyze(node=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("kernel_sum_over_window_pct=77.778", result.stdout)

    def test_coarse_graph_rejected(self):
        result = self.analyze(graph=(20,80))
        self.assertEqual(result.returncode, 2)
        self.assertIn("--cuda-graph-trace=node", result.stderr)

    def test_coarse_graph_with_nodes(self):
        result = self.analyze(graph=(20,80), node=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_graph_outside_decode_window(self):
        result = self.analyze(graph=(0,9))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_grouped_missing_dimensions(self):
        result = self.analyze(grouped=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("lacks grid/block", result.stderr)

    def test_grouped_distinct_launches(self):
        result = self.analyze(graph=(20,80), node=True, geometry=True, grouped=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("kernel_sum_over_window_pct=77.778", result.stdout)
        self.assertIn("10x1x1/256x1x1\texpert", result.stdout)
        self.assertIn("20x1x1/128x1x1\texpert", result.stdout)
        self.assertIn("28.571\t1\t0.020", result.stdout)
        self.assertIn("57.143\t1\t0.040", result.stdout)


if __name__ == "__main__":
    unittest.main()
