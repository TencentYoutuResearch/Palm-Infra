# Quantization

*Last updated: 2026-07-01*

## Current Status

W8 weight-only quantization is implemented as a correctness-first baseline.
Both per-channel and per-group metadata are supported in the file/package
format and in the runtime matmul path.

Supported converter modes:

```bash
# FP16 / original precision
python3 models/converter.py /path/to/model out.mollm

# W8 per-channel: one scale per output row
python3 models/converter.py /path/to/model out_w8pc.mollm 24 256 w8pc

# W8 per-group: one scale per output row per K-group
python3 models/converter.py /path/to/model out_w8g128.mollm 24 256 w8g128
```

The current INT8 kernel is scalar and optimized for correctness, not speed.
Do not use W8 benchmark numbers as performance conclusions until the NEON
kernel path is implemented.

## Weight Format

`.weights` files use the existing 88-byte header. For quantized weights:

- `precision = INT8`
- `data_offset/data_size` point to row-major int8 weight data
- `scales_offset/scales_size` point to FP32 scales
- `group_size` is the K-dimension quantization group size
- `num_groups = N * ceil(K / group_size)`

For a 2-D weight `[N, K]`, quantization groups are laid out by output row:

```text
q:      int8    [N, K]
scales: float32 [N, groups_per_row]
groups_per_row = ceil(K / group_size)
scale_index = n * groups_per_row + k / group_size
```

`w8pc` is represented as `group_size = K`, so `groups_per_row = 1`.
`w8g128` uses `group_size = 128`.

## Quantization Math

The converter uses symmetric int8 quantization per row/group:

```text
scale = max(abs(weight_block)) / 127
q = round(weight / scale), clipped to [-127, 127]
dequant = q * scale
```

All-zero blocks use `scale = 1.0`.

## Runtime Semantics

The engine reads precision from the weight file/package header, not from the
graph node's original precision. This lets existing graphs load quantized
weights without changing graph schemas.

For INT8 constants, `load_graph()` validates:

- scales are present
- `group_size > 0`
- `num_groups == N * ceil(K / group_size)`
- `scales_size == num_groups * sizeof(float)`

FP16 interleaved packing is skipped for INT8 weights. `kernel_matmul_fp32()`
detects `B.prec == INT8`, dequantizes by group on the fly, and writes FP32
outputs. Activation is applied after the scalar matmul range.

## Quantized Weights

The Qwen3.5 and Youtu converters quantize projection/linear 2-D weights when
`quant != none`.

Kept in original precision:

- embeddings / tied lm_head storage
- norm weights
- convolution weights
- Gated DeltaNet `A_log` and `dt_bias`
- other non-2-D tensors

This keeps the first baseline conservative and avoids changing embedding lookup
or lm_head packing semantics.

## Validation

Validation performed on 2026-07-01:

```bash
ninja -C build
python3 -m py_compile models/converter.py models/transpile.py models/qwen35.py models/mla.py
./build/test_matmul
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8g128.mollm
ctest --test-dir build --output-on-failure -E 'test_e2e|bench_matmul|bench_sdpa'
```

Short Qwen3.5-0.8B correctness smoke test:

| Package | Size | Short CE/PPL | CE delta vs FP16 | Greedy smoke |
|---|---:|---:|---:|---|
| FP16 `qwen35_0.8b.mollm` | 1518.9 MB | 3.0911 / 22.00 | - | - |
| W8PC `qwen35_0.8b_w8pc.mollm` | 1022.9 MB | 3.0969 / 22.13 | +0.0058 | pass |
| W8G128 `qwen35_0.8b_w8g128.mollm` | 1036.8 MB | 3.0926 / 22.03 | +0.0015 | pass |

`test_quantized_e2e` checks load, finite logits, vocab/logit shape, reusable
workspace presence, short CE drift, and two greedy decode steps.

## Known Limitations

- INT8 matmul is scalar. Performance is expected to be worse than FP16 until
  NEON kernels are added.
- Full 256-token W8 PPL is not part of the default quick test because scalar
  W8 decode makes the existing `test_e2e` very slow.
- GGUF import is intentionally deferred. The `.mollm` graph/runtime semantics
  need explicit quant metadata and graph-owned execution behavior; direct GGUF
  compatibility would add format work without avoiding the kernel/runtime work.
- W4 is still future work. W8 establishes the metadata, loader, and testing
  structure first.

## Next Steps

1. Add optimized ARM NEON W8 GEMV/GEMM kernels for the current metadata layout.
2. Run strict pp256/tg64 benchmarks on 2B/4B after W8 kernels exist.
3. Compare W8PC vs W8G128 quality on full 256-token PPL.
4. Extend the same metadata structure to W4 once W8 is stable.
