# Quantization

*Last updated: 2026-07-02*

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

The first INT8 NEON kernels are now implemented. Decode/GEMV uses a
llama.cpp-style Q8 activation dot path by default. The default W8 runtime now
uses the repacked dot-product layout for both decode/GEMV and prefill/GEMM.
The older INT8 interleaved layout is created only for fallback or diagnostic
paths, so the q8dot layout replaces that copy instead of adding a second packed
INT8 weight buffer.

Experimental runtime mode:

```bash
MOLLM_W8_ONFLY_FP16=1 ./build/mollm_bench --package out_w8pc.mollm ...
MOLLM_W8_NO_Q8_DOT=1 ./build/mollm_bench --package out_w8pc.mollm ...
MOLLM_W8_NO_Q8_DOT_REPACK=1 ./build/mollm_bench --package out_w8pc.mollm ...
MOLLM_W8_NO_Q8_DOT_GEMM=1 ./build/mollm_bench --package out_w8pc.mollm ...
MOLLM_W8_Q8_DOT_GEMM=1 ./build/mollm_bench --package out_w8pc.mollm ...
MOLLM_W8_NO_I8MM=1 ./build_i8mm/mollm_bench --package out_w8pc.mollm ...
MOLLM_MATMUL_SHAPE_PROFILE=1 ./build/mollm_bench --package out_w8pc.mollm --profile ...
```

- `MOLLM_W8_ONFLY_FP16=1`: diagnostic path that dequantizes the loaded B tile
  to FP16 inside an FP16-accumulate kernel. It keeps W8 storage but is slower
  for decode.
- `MOLLM_W8_NO_Q8_DOT=1`: disables the default Q8 activation dot GEMV path and
  falls back to the older FP32-dequant W8 GEMV.
- `MOLLM_W8_NO_Q8_DOT_REPACK=1`: disables the default q8dot repacked layout and
  falls back to the older INT8 interleaved layout.
- `MOLLM_W8_NO_Q8_DOT_GEMM=1`: keeps repacked GEMV/decode enabled but disables
  repacked GEMM/prefill; the loader keeps the old interleaved copy for GEMM.
- `MOLLM_W8_Q8_DOT_GEMM=1`: legacy diagnostic switch for the old non-repacked
  Q8-dot GEMM prototype. It is not needed for the default q8dot full-repack
  path.
- `MOLLM_W8_NO_I8MM=1`: in an i8mm/native build, disables the
  `vmmlaq_s32` q8dot GEMM kernel and forces the DOTPROD repacked GEMM path for
  A/B diagnostics.
- `MOLLM_MATMUL_SHAPE_PROFILE=1`: diagnostic profiler. With
  `mollm_bench --profile`, prints matmul rows grouped by phase, dispatch path,
  `M/N/K`, quant group metadata, thread count, and repack/interleave flags.

Build-time ARM feature switches:

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DMOLLM_ARM_NATIVE=ON
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DMOLLM_ARM_I8MM=ON
```

- `MOLLM_ARM_NATIVE=ON`: passes `-mcpu=native` for local ARM64 benchmarking.
- `MOLLM_ARM_I8MM=ON`: passes `-march=armv8.6-a+i8mm+fp16+fp16fml` to expose
  `__ARM_FEATURE_MATMUL_INT8` while preserving the existing FP16FML kernels.
- Both are OFF by default so portable ARM64 builds keep the previous target.

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

FP16 interleaved packing is skipped for INT8 weights. By default, INT8 weights
are packed at load time as `[N/8, K/32, 8, 32]` for ARM dot-product kernels.
The older `[N/8, K, 8]` layout is built only when q8dot repack is disabled,
GEMM q8dot is disabled, on-the-fly FP16 diagnostics are requested, or the
weight shape/quant group cannot use 32-element q8dot blocks.

Runtime dispatch:

- GEMV/decode: quantize the FP32 activation row to temporary Q8 blocks
  (`32` K-elements per block), dot with INT8 weight tiles, then apply
  `activation_scale * weight_scale` to the int32 dot result.
- GEMV/decode: default uses `[N/8, K/32, 8, 32]` and ARM dot-product intrinsics.
- GEMM/prefill: default uses the same q8dot repacked layout and ARM dot-product
  intrinsics, or ARM i8mm (`vmmlaq_s32`) when the binary is built with
  `__ARM_FEATURE_MATMUL_INT8`.
- Weight scale dispatch distinguishes:
  - `groups_per_row == 1`: per-channel scale, hoisted per output tile
  - `group_size == 32`: true per-block scale matching the q8 dot block
  - otherwise: per-group scale, such as `w8g128`
- Fallback: disabling q8dot repack uses the older `[N/8, K, 8]` native W8
  dequant GEMM path.
- Scalar fallback still supports exact FP32 dequant by group.

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
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm 256
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8g128.mollm 256
ctest --test-dir build --output-on-failure -E 'test_e2e|bench_matmul|bench_sdpa'
```

Short Qwen3.5-0.8B correctness smoke test:

| Package | Size | Short CE/PPL | CE delta vs FP16 | Greedy smoke |
|---|---:|---:|---:|---|
| FP16 `qwen35_0.8b.mollm` | 1518.9 MB | 3.0911 / 22.00 | - | - |
| W8PC `qwen35_0.8b_w8pc.mollm` | 1022.9 MB | 3.0969 / 22.13 | +0.0058 | pass |
| W8G128 `qwen35_0.8b_w8g128.mollm` | 1036.8 MB | 3.0926 / 22.03 | +0.0015 | pass |

256-token Qwen3.5-0.8B reference-token audit:

| Package | 256-token CE/PPL | CE delta vs FP16 |
|---|---:|---:|
| FP16 `qwen35_0.8b.mollm` | 2.1388 / 8.49 | - |
| W8PC `qwen35_0.8b_w8pc.mollm` | 2.1473 / 8.56 | +0.0085 |
| W8G128 `qwen35_0.8b_w8g128.mollm` | 2.1413 / 8.51 | +0.0025 |

`test_quantized_e2e` checks load, finite logits, vocab/logit shape, reusable
workspace presence, configurable CE/PPL token count, and two greedy decode
steps.

## Known Limitations

- Default INT8 GEMM now uses the repacked Q8-dot path. i8mm builds are much
  faster than the DOTPROD GEMM path on prefill, but still trail llama.cpp Q8_0.
- Decode/GEMV now uses Q8 activation dot by default and is materially faster
  than the first W8 GEMV path, but it still does not match llama.cpp Q8_0 on
  Qwen3.5-0.8B.
- `MOLLM_W8_ONFLY_FP16=1` keeps W8 storage and dequantizes each loaded B tile to
  FP16 inside an experimental kernel. Current microbench results show it is not
  faster than the native W8 NEON path, and GEMV is significantly slower.
- Full 256-token W8 PPL passes on the current reference sample; broader
  calibration-set quality audit is still future work.
- GGUF import is intentionally deferred. The `.mollm` graph/runtime semantics
  need explicit quant metadata and graph-owned execution behavior; direct GGUF
  compatibility would add format work without avoiding the kernel/runtime work.
- W4 is still future work. W8 establishes the metadata, loader, and testing
  structure first.

## Next Steps

1. Optimize repacked Q8-dot GEMV/decode and the remaining GEMM overhead,
   especially scale handling, activation Q8 packing, stores, and kernel layout.
2. Reduce remaining runtime memory traffic and compare against llama.cpp Q8_0
   at the kernel level.
3. Compare W8PC vs W8G128 quality on a broader calibration set.
4. Extend the same metadata structure to W4 once W8 is stable.
