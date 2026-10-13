import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("routes", Path(__file__).resolve().parents[1] /
                                           "tools/analyze_cuda_prefill_routes.py")
routes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routes)


class RouteTraceTests(unittest.TestCase):
    def record(self, node, rows, counts, unmatched=0):
        return f"nvfp4_expert_counts node={node} rows={rows} unmatched={unmatched} counts=" + ",".join(map(str, counts))

    def test_complete_ordered_chunks(self):
        text = "\n".join([self.record(3, 2, [3, 1, 0]), self.record(7, 2, [2, 0, 2]),
                          self.record(3, 1, [1, 1, 0]), self.record(7, 1, [0, 0, 2])])
        report = routes.analyze(text, [2, 1], 2, 3, 2)
        self.assertEqual(report["records"], 4)
        self.assertEqual([r["routes"] for r in report["chunks"]], [8, 4])
        self.assertEqual(report["chunks"][0]["batch_widths"]["8"]["route_slot_fill_pct"], 25)

    def test_missing_extra_and_malformed(self):
        line = self.record(3, 2, [3, 1, 0])
        for text in ("", line + "\n" + line, line + ",", line.replace("3,1,0", "3,-1,2")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                routes.analyze(text, [2], 1, 3, 2)

    def test_logical_payload_and_tail_batches(self):
        # One expert with 33 routes crosses both N32 and N8 boundaries.
        report = routes.analyze(self.record(3, 33, [33]), [33], 1, 1, 1)
        audit = routes.qwen38_nvfp4_dataflow(report)
        chunk = audit["chunks"][0]
        gate = 1280 * 2560 * 9 // 16
        down = 2560 * 640 * 9 // 16
        self.assertEqual(chunk["gate_weight_batch_payload_bytes"], 2 * gate)
        self.assertEqual(chunk["down_weight_batch_payload_bytes"], 5 * down)
        self.assertEqual(chunk["active_weight_once_payload_bytes"], gate + down)
        self.assertEqual(chunk["gate_routes_per_weight_traversal"], 16.5)
        self.assertEqual(chunk["down_routes_per_weight_traversal"], 6.6)
        self.assertEqual(chunk["gate_quant_source_requested_bytes"], 33 * 2 * 2560 * 4)
        self.assertEqual(chunk["gate_quant_packed_and_scale_write_bytes"], 33 * 2 * 2560 * 9 // 16)
        self.assertEqual(chunk["activation_packed_and_scale_write_bytes"], 33 * 640 * 9 // 16)
        self.assertEqual(audit["total"]["down_bf16_output_write_bytes"], 33 * 2560 * 2)
        self.assertIn("not_measured_DRAM", audit["scope"])

    def test_payload_totals_exclude_empty_experts(self):
        text = "\n".join([self.record(3, 1, [1, 0]), self.record(3, 2, [1, 1])])
        audit = routes.qwen38_nvfp4_dataflow(routes.analyze(text, [1, 2], 1, 2, 1))
        for key, total in audit["total"].items():
            self.assertEqual(total, sum(chunk[key] for chunk in audit["chunks"]))
        self.assertEqual(audit["total"]["active_weight_once_payload_bytes"],
                         3 * (1280 * 2560 + 2560 * 640) * 9 // 16)

    def test_fused_token_payload_uses_tokens_not_routes_or_variants(self):
        text = "\n".join([self.record(3, 2, [3, 1]), self.record(7, 2, [2, 2])])
        report = routes.analyze(text, [2], 2, 2, 2)
        legacy = routes.qwen38_nvfp4_dataflow(report)["total"]
        fused = routes.qwen38_nvfp4_dataflow(report, "fused-token")["total"]
        self.assertEqual(fused["gate_quant_source_requested_bytes"], 2 * 2 * 2560 * 4)
        self.assertEqual(fused["gate_quant_packed_and_scale_write_bytes"], 2 * 2 * 2560 * 9 // 16)
        self.assertEqual(fused["gate_bf16_output_write_bytes"], 0)
        for key in ("gate_weight_batch_payload_bytes", "down_weight_batch_payload_bytes",
                    "down_bf16_output_write_bytes", "gate_bf16_valid_intermediate_bytes"):
            self.assertEqual(fused[key], legacy[key])
        self.assertEqual(legacy["gate_quant_source_requested_bytes"], 4 * fused["gate_quant_source_requested_bytes"])
        with self.assertRaises(ValueError):
            routes.qwen38_nvfp4_dataflow(report, "unknown")

    def test_fused_token_counter_qualification(self):
        token = "CudaBackend: prefill shared token quantization calls=192"
        fused = "CudaBackend: NVFP4 MMA activation fused prefill calls=192"
        geometry = "\nCudaBackend: prefill compact gate logits calls=192\nCudaBackend: NVFP4 BF16 down output prefill calls=192"
        routes.validate_fused_token_trace(token + "\n" + fused + geometry, 192)
        for text in ("", token, fused, token + "\n" + fused.replace("192", "191"),
                     token + "\n" + token + "\n" + fused,
                     token.replace("192", "193") + "\n" + fused,
                     token + "\n" + fused + "\nCudaBackend: NVFP4 BF16 gate output prefill calls=1"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                routes.validate_fused_token_trace(text + geometry, 192)

    def test_auto_mode_qualifies_only_complete_unmixed_counters(self):
        token = "CudaBackend: prefill shared token quantization calls=192"
        fused = "CudaBackend: NVFP4 MMA activation fused prefill calls=192"
        bf16 = "CudaBackend: NVFP4 BF16 gate output prefill calls=192"
        down = "\nCudaBackend: NVFP4 BF16 down output prefill calls=192"
        compact = "\nCudaBackend: prefill compact gate logits calls=192"
        self.assertEqual(routes.infer_gate_path(token + "\n" + fused + compact + down, 192), "fused-token")
        self.assertEqual(routes.infer_gate_path(fused + compact + down, 192), "fused-route")
        self.assertEqual(routes.infer_gate_path(bf16 + down, 192), "route-bf16")
        for text in ("", token, fused.replace("192", "191"),
                     token + "\n" + fused + "\n" + bf16,
                     token + "\n" + token + "\n" + fused):
            with self.subTest(text=text), self.assertRaises(ValueError):
                routes.infer_gate_path(text + compact + down, 192)

    def test_auto_mode_rejects_missing_partial_or_duplicate_geometry(self):
        gate = "CudaBackend: prefill shared token quantization calls=192\nCudaBackend: NVFP4 MMA activation fused prefill calls=192"
        compact = "\nCudaBackend: prefill compact gate logits calls=192"
        down = "\nCudaBackend: NVFP4 BF16 down output prefill calls=192"
        for suffix in ("", compact, down, compact + down.replace("192", "191"),
                       compact.replace("192", "191") + down, compact + compact + down,
                       compact + down + down):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                routes.infer_gate_path(gate + suffix, 192)
        with self.assertRaises(ValueError):
            routes.infer_gate_path("", 0)

    def test_fused_route_does_not_publish_global_gate_logits(self):
        report = routes.analyze(self.record(3, 33, [33]), [33], 1, 1, 1)
        legacy = routes.qwen38_nvfp4_dataflow(report)["total"]
        fused = routes.qwen38_nvfp4_dataflow(report, "fused-route")["total"]
        self.assertEqual(fused["gate_quant_source_requested_bytes"], legacy["gate_quant_source_requested_bytes"])
        self.assertEqual(fused["gate_bf16_output_write_bytes"], 0)

    def test_unmatched_expert_width_and_total(self):
        for counts, unmatched in (([3, 1, 0], 1), ([4, 0], 0), ([3, 0, 0], 0)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                routes.analyze(self.record(3, 2, counts, unmatched), [2], 1, 3, 2)

    def test_duplicate_nodes_and_changed_order(self):
        for nodes in ((3, 3, 3, 7), (3, 7, 7, 3)):
            with self.subTest(nodes=nodes), self.assertRaises(ValueError):
                routes.analyze("\n".join(self.record(n, 2, [3, 1, 0]) for n in nodes), [2, 2], 2, 3, 2)

    def test_chunk_order_and_invalid_configuration(self):
        line = self.record(3, 2, [3, 1, 0])
        for rows, layers, experts, top_k in (([1], 1, 3, 2), ([], 1, 3, 2),
                                           ([0], 1, 3, 2), ([2], 0, 3, 2), ([2], 1, 0, 2)):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                routes.analyze(line, rows, layers, experts, top_k)

    def test_merge_complete_layer_histograms(self):
        text = "\n".join([self.record(3, 2, [3, 1, 0]), self.record(7, 2, [2, 0, 2]),
                          self.record(3, 1, [1, 1, 0]), self.record(7, 1, [0, 0, 2])])
        report = routes.merge_routing_windows(text, [2, 1], 2, 3, 2)
        self.assertEqual(report["merged"]["rows"], 3)
        self.assertEqual(report["merged"]["routes"], 12)
        self.assertEqual(report["merged"]["active_expert_layer_pairs"], 4)
        self.assertEqual(report["batch_comparison"]["8"]["split_batches"], 7)
        self.assertEqual(report["batch_comparison"]["8"]["merged_batches"], 4)
        self.assertIn("not_model_schedule", report["scope"])

    def test_merge_rejects_incomplete_or_reordered_layers(self):
        for text in (self.record(3, 2, [3, 1, 0]),
                     "\n".join(self.record(n, 2, [3, 1, 0]) for n in (3, 7, 7, 3))):
            with self.assertRaises(ValueError):
                routes.merge_routing_windows(text, [2, 2], 2, 3, 2)

    def test_single_window_has_no_merge_reduction(self):
        report = routes.merge_routing_windows(self.record(3, 33, [33]), [33], 1, 1, 1)
        for width in (8, 16, 32, 64):
            self.assertEqual(report["batch_comparison"][str(width)]["batch_reduction_pct"], 0)

    def test_n32_skips_empty_n8_groups_and_layer_distribution(self):
        report = routes.analyze(self.record(3, 66, [1, 8, 9, 16, 32, 0]),
                                [66], 1, 6, 1)["chunks"][0]
        self.assertEqual(report["gate_n32_active_n8_subgroups"], 10)
        self.assertEqual(report["gate_n32_skipped_n8_subgroups"], 10)
        layer = report["layers"][0]
        self.assertEqual(layer["active_experts"], 5)
        self.assertEqual(layer["active_expert_count_median"], 9)
        self.assertEqual(layer["expert_count_ranges"],
                         {"1_8": 2, "9_16": 2, "17_32": 1, "33_64": 0, "above64": 0})
        self.assertEqual(layer["n8_route_slot_fill_pct"], 82.5)


if __name__ == "__main__":
    unittest.main()
