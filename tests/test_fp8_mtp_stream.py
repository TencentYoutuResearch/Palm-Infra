"""Bounded FP8 MTP expert decoding, scale blocks, tails and aggregation."""
import io
import sys
import tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from safetensors_stream import SafeTensorSlice, fp8_block128_to_fp16_streamed_weight, aggregate_fp16_experts
from transpile import WeightByteRange, StreamedWeight, Precision
from qwen4_exp import build_streamed_weights


def dense_export_regression(root):
    """Fused attention exports must honor FP16, while default W4 is preserved."""
    tc = dict(hidden_size=32, num_hidden_layers=2,
              layer_types=["linear_attention", "full_attention"],
              num_attention_heads=4, head_dim=8, num_key_value_heads=2,
              linear_num_key_heads=2, linear_num_value_heads=8,
              linear_key_head_dim=8, linear_value_head_dim=8,
              linear_conv_kernel_dim=4, num_experts=2, moe_intermediate_size=32,
              shared_expert_intermediate_size=32, hc_count=4, hc_lowrank=32,
              ple_layer_ids=[])
    path = root / "dense_source.bin"
    values = ((np.arange(128 * 128) % 37 - 18) / 64).astype("<f2")
    path.write_bytes(values.tobytes())
    class Index:
        def contains(self, name): return False
        def tensor(self, name):
            suffixes = {
                "hc_norm.weight": (128,), "input_mix_weight_down.weight": (32, 128),
                "input_mix_weight_up.weight": (128, 32), "block_inject_weight.weight": (4, 128),
                "in_proj_qkv.weight": (96, 32), "in_proj_a.weight": (8, 32),
                "in_proj_b.weight": (8, 32), "in_proj_z.weight": (64, 32),
                "conv1d.weight": (96, 1, 4), "A_log": (8,), "dt_bias": (8,),
                "linear_attn.norm.weight": (8,), "linear_attn.out_proj.weight": (32, 64),
                "q_proj.weight": (64, 32), "k_proj.weight": (16, 32), "v_proj.weight": (16, 32),
                "o_proj.weight": (32, 32), "q_norm.weight": (8,), "k_norm.weight": (8,),
                "gate_proj.weight": (32, 32), "up_proj.weight": (32, 32), "down_proj.weight": (32, 32),
                "mlp.gate.weight": (2, 32), "shared_expert_gate.weight": (1, 32),
            }
            shape = next((shape for suffix, shape in suffixes.items() if name.endswith(suffix)), None)
            if shape is None: raise AssertionError(name)
            return SafeTensorSlice(name, "F16", shape,
                WeightByteRange(str(path), 0, int(np.prod(shape)) * 2))
    for precision in ("fp16", "w4"):
        weights = build_streamed_weights(root, tc, _index=Index(), _expert_precision="fp16",
                                         _dense_precision=precision, _include_shared_weights=False)
        for ref in ("./model_language_model_layers_0_linear_attn_in_proj_weight.weights",
                    "./model_language_model_layers_1_self_attn_qkv_proj_weight.weights"):
            weight = weights[ref]
            assert weight.precision == (Precision.FP16 if precision == "fp16" else Precision.INT4)
            output = io.BytesIO()
            weight.write_to(output)
            assert len(output.getvalue()) == weight.size
        if precision == "fp16":
            qkv = weights["./model_language_model_layers_1_self_attn_qkv_proj_weight.weights"]
            payload = io.BytesIO()
            StreamedWeight._copy_ranges(payload, qkv.data_ranges, 19)
            source_q = values[:64 * 32].reshape(64, 32)
            expected = np.concatenate([source_q[0:8], source_q[16:24], source_q[32:40], source_q[48:56],
                                       source_q[8:16], source_q[24:32], source_q[40:48], source_q[56:64],
                                       source_q[:16], source_q[:16]])
            np.testing.assert_array_equal(np.frombuffer(payload.getvalue(), dtype="<f2").reshape(96, 32), expected)


def main():
    with tempfile.TemporaryDirectory(prefix="mollm_fp8_mtp_test_") as tmp:
        dense_export_regression(Path(tmp))
        path = Path(tmp) / "source.bin"
        data, tensors = bytearray(), {}
        def add(name, dtype, shape, payload):
            offset = len(data)
            data.extend(payload)
            tensors[name] = SafeTensorSlice(name, dtype, shape,
                WeightByteRange(str(path), offset, len(payload)))
        class Index:
            def tensor(self, name): return tensors[name]
        index = Index()
        codes = np.array([0, 1, 0x38, 0x40, 0x7e, 0x80, 0xb8], dtype=np.uint8)
        decoded = np.array([0, 2 ** -9, 1, 2, 448, -0., -1], dtype=np.float32)
        positions = np.arange(130 * 130).reshape(130, 130) % len(codes)
        scales = np.array([[2, 3], [5, 7]], dtype=np.float32)
        gold = decoded[positions] * np.repeat(np.repeat(scales, 128, axis=0), 128, axis=1)[:130, :130]
        for name, scale_type in (("gate", "BF16"), ("up", "F32")):
            payload_codes = codes[positions] if name == "gate" else codes[positions] ^ 128
            add(name + ".weight", "F8_E4M3", (130, 130), payload_codes.tobytes())
            raw_scale = ((scales.view(np.uint32) >> 16).astype("<u2").tobytes()
                         if scale_type == "BF16" else scales.astype("<f4").tobytes())
            add(name + ".weight_scale_inv", scale_type, (2, 2), raw_scale)
        add("nan.weight", "F8_E4M3", (1, 1), bytes([127]))
        add("nan.weight_scale_inv", "F32", (1, 1), np.array([1], dtype="<f4").tobytes())
        path.write_bytes(data)
        weight = fp8_block128_to_fp16_streamed_weight(index, "gate.weight")
        assert weight.precision == Precision.FP16 and weight.data_size == 130 * 130 * 2
        assert len(weight.data_ranges) == 2
        for chunk in (1, 130, 389, 8 * 1024 * 1024):
            output = io.BytesIO()
            StreamedWeight._copy_ranges(output, weight.data_ranges, chunk)
            actual = np.frombuffer(output.getvalue(), dtype="<f2").reshape(130, 130)
            np.testing.assert_array_equal(actual.view(np.uint16), gold.astype(np.float16).view(np.uint16))
        aggregate = aggregate_fp16_experts(index,
            [["gate.weight", "up.weight"], ["up.weight", "gate.weight"]])
        assert aggregate.logical_shape == (520, 130)
        output = io.BytesIO()
        StreamedWeight._copy_ranges(output, aggregate.data_ranges, 389)
        np.testing.assert_array_equal(np.frombuffer(output.getvalue(), dtype="<f2").reshape(520, 130),
                                      np.concatenate([gold, -gold, -gold, gold]).astype(np.float16))
        try:
            bad = fp8_block128_to_fp16_streamed_weight(index, "nan.weight")
            StreamedWeight._copy_ranges(io.BytesIO(), bad.data_ranges)
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite FP8 payload accepted")
        try:
            aggregate_fp16_experts(index, [["gate.weight"], ["gate.weight", "up.weight"]])
        except ValueError:
            pass
        else:
            raise AssertionError("different expert geometry accepted")
    print("FP8 MTP stream: block tails, BF16/F32 scales, repeated chunk writes and expert aggregation passed")


if __name__ == "__main__":
    main()
