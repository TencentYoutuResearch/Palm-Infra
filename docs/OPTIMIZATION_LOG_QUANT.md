# Quantization Optimization Log

*Last updated: 2026-07-03*

This log tracks quantized-kernel optimization separately from the Qwen3.5 FP16
optimization log. `docs/QUANTIZATION.md` describes the format and correctness
baseline; this file records performance experiments.

Benchmark notes:

- Use `caffeinate -dimsu` on macOS to avoid sleep during measurement.
- Microbench numbers below use 4 threads unless stated otherwise.
- W8 package benchmarks use pp256 + tg64, warmup=3.
- 0.8B is noisy, so package numbers are for direction only; 2B/4B remain the
  primary targets once kernels are mature.

## Attempt 1: W8 B Interleave + NEON GEMV/GEMM

### Goal

Replace the correctness-first scalar INT8 matmul path with a first NEON path
that preserves the W8 metadata layout:

- per-channel: `group_size = K`
- per-group: `group_size = 128` or another `w8gN`
- scale indexing: `scales[n * groups_per_row + k / group_size]`

### Implementation

- Added load-time INT8 B interleaving:
  - original `[N, K]` row-major int8
  - packed `[N/8, K, 8]`, matching the FP16 B interleave layout
  - padded output rows are zero
- `Tensor::is_interleaved` marks packed weight layout.
- Added `pack_b_interleaved_int8_full()`.
- Added W8 NEON GEMV for `M == 1`:
  - load 8 int8 output channels for one `k`
  - sign-extend to int16/int32
  - convert to FP32
  - apply per-channel/per-group scale
  - FMA with FP32 activation
- Added W8 NEON GEMM for `M > 1`:
  - 4 rows x 8 output channels
  - FP32 accumulate
  - no A packing yet
- Kept scalar fallback for non-interleaved INT8.
- Extended `bench_matmul` with `--int8` and `--group-size`.
- Extended `test_matmul` to cover row-major and interleaved INT8 for both
  per-channel and per-group scales.

### Correctness

```bash
ninja -C build
./build/test_matmul
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8g128.mollm
ctest --test-dir build --output-on-failure -E 'test_e2e|bench_matmul|bench_sdpa'
```

All passed.

`test_quantized_e2e` wall time dropped from about 9s to about 2.5s for explicit
W8PC/W8G128 runs, and about 1.4s inside quick ctest after warm filesystem/cache
state.

### Microbench

After a sleep/wake event invalidated an earlier run, measurements were repeated
with `caffeinate -dimsu`.

#### GEMV: M=1 K=1024 N=4096

| Path | group | avg ms | GF/s |
|---|---:|---:|---:|
| W8 row-major scalar | 1024 | 0.92 | 9.1 |
| W8 interleaved NEON | 1024 | 0.12 | 71.5 |
| W8 interleaved NEON | 128 | 0.12 | 72.0 |
| FP16 interleaved | - | 0.03 | 296.0 |

#### GEMV: M=1 K=2048 N=6144

| Path | group | avg ms | GF/s |
|---|---:|---:|---:|
| W8 row-major scalar | 2048 | 2.74 | 9.2 |
| W8 interleaved NEON | 2048 | 0.37 | 68.6 |
| W8 interleaved NEON | 128 | 0.36 | 69.9 |
| FP16 interleaved | - | 0.15 | 164.3 |

#### GEMM: M=128 K=1024 N=4096

| Path | group | avg ms | GF/s |
|---|---:|---:|---:|
| W8 row-major scalar | 1024 | 111.26 | 9.7 |
| W8 interleaved NEON | 1024 | 5.08 | 211.4 |
| W8 interleaved NEON | 128 | 5.17 | 207.6 |
| FP16 interleaved | - | 1.44 | 745.0 |

#### GEMM: M=128 K=2048 N=6144

| Path | group | avg ms | GF/s |
|---|---:|---:|---:|
| W8 interleaved NEON | 2048 | 14.78 | 218.0 |
| W8 interleaved NEON | 128 | 14.05 | 229.3 |
| FP16 interleaved | - | 4.05 | 795.1 |

### Package Bench: Qwen3.5-0.8B

Same machine state, pp256 + tg64, warmup=3, 4 threads.

| Package | pp/tg | prefill ms | decode ms | peak RSS |
|---|---:|---:|---:|---:|
| FP16 | 623.5 / 97.6 | 410.6 | 645.8 | 3992.9 MB |
| W8PC | 194.5 / 49.7 | 1315.9 | 1268.4 | 3044.7 MB |
| W8G128 | 190.9 / 49.5 | 1341.3 | 1272.3 | 3058.1 MB |

### Interpretation

The scalar bottleneck is fixed:

- GEMV improves about 7-8x.
- GEMM improves about 20x for tested prefill shapes.
- End-to-end W8 smoke tests are much faster and remain correct.

But W8 is still slower than FP16:

- Current W8 kernel converts int8 to FP32 and applies FP32 scales inside the K
  loop.
- FP16 path uses highly optimized FP16FML kernels and FP16 accumulation.
- Weight bandwidth is lower, but compute/conversion overhead dominates on Apple
  Silicon for this first W8 kernel.

Memory improves in package benchmark: W8 0.8B peak RSS is about 3.05 GB vs FP16
about 3.99 GB. Runtime still keeps a packed INT8 copy in addition to the mmap'd
package weights, mirroring the FP16 packed-weight strategy.

### Next Ideas

1. Explore activation quantization + int8 dot product (`vdot`) / i8mm paths.
   This is the real route to W8 being faster than FP16FML, but has a larger
   correctness and calibration surface.
2. Improve W8 FP32-dequant kernel:
   - 8-row tile or A packing
   - scale hoisting / per-channel specialization
   - unroll K and reduce conversion overhead
3. Run 2B/4B package benchmarks only after one of the above improves the kernel
   enough to be competitive.

## Attempt 2: On-the-Fly W8 -> FP16 B Tile Experiment

### Goal

Test the user's intended "dequant while loading B" idea without creating a
persistent FP16 copy of the full weight tensor. This answers a narrower kernel
question:

> Can the current FP16FML matmul structure be reused by dequantizing each INT8
> B tile to FP16 at load time inside the kernel?

### Implementation

Added an opt-in environment variable for the experimental path:

```bash
MOLLM_W8_ONFLY_FP16=1 ./build/mollm_bench --package qwen35_0.8b_w8pc.mollm ...
```

When enabled:

- runtime weight storage remains INT8 interleaved `[N/8, K, 8]`
- GEMV loads an int8x8 B tile, sign-extends, converts/scales to FP16, then uses
  FP16 accumulation
- GEMM uses an 8x8 packed-A FP16-accumulate structure and dequantizes each
  int8x8 B tile on the fly
- no persistent FP16 packed weight copy is allocated

An earlier full load-time dequant-to-FP16 prototype was tried as an upper-bound
diagnostic, but removed after review. It recovered much of FP16 prefill speed
for 0.8B W8PC, but it stored the mmap'd W8 package plus a runtime packed FP16
copy of the weights, giving up too much of the runtime-memory benefit. That is
not a realistic final implementation.

### Correctness

```bash
MOLLM_W8_ONFLY_FP16=1 ./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm
MOLLM_W8_ONFLY_FP16=1 ./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8g128.mollm
```

Both passed.

Short CE/PPL:

| Package | Runtime | Short CE/PPL | CE delta vs FP16 |
|---|---|---:|---:|
| W8PC | native W8 NEON | 3.0969 / 22.13 | +0.0058 |
| W8PC | on-the-fly B tile FP16 | 3.0945 / 22.08 | +0.0034 |
| W8G128 | native W8 NEON | 3.0926 / 22.03 | +0.0015 |
| W8G128 | on-the-fly B tile FP16 | 3.0847 / 21.86 | -0.0064 |

The small CE difference is expected: the experimental path uses FP16
accumulation after tile dequant, while native W8 uses FP32 dequant/accumulate.

### Microbench

After sleep/wake invalidated an earlier run, these were repeated with
`caffeinate -dimsu`.

#### GEMV: M=1 K=2048 N=6144

| Path | group | avg ms | GF/s |
|---|---:|---:|---:|
| W8 interleaved NEON | 2048 | 0.34 | 73.7 |
| W8 on-the-fly FP16 tile | 2048 | 1.09 | 23.2 |
| W8 on-the-fly FP16 tile | 128 | 1.09 | 23.0 |
| FP16 interleaved | - | 0.16 | 154.4 |

#### GEMM: M=128 K=1024 N=4096

| Path | group | avg ms | GF/s |
|---|---:|---:|---:|
| W8 interleaved NEON | 1024 | 5.08 | 211.4 |
| W8 on-the-fly FP16 tile | 1024 | 5.09 | 210.8 |
| W8 on-the-fly FP16 tile | 128 | 5.32 | 201.8 |
| FP16 interleaved | - | 1.42 | 755.5 |

### Package Bench: Qwen3.5-0.8B

Same machine state, pp256 + tg64, warmup=3, 4 threads.

| Package | Runtime | pp/tg | prefill ms | decode ms | peak RSS |
|---|---|---:|---:|---:|---:|
| FP16 | FP16 package + FP16FML | 623.5 / 97.6 | 410.6 | 645.8 | 3992.9 MB |
| W8PC | native W8 NEON | 194.5 / 49.7 | 1315.9 | 1268.4 | 3044.7 MB |
| W8G128 | native W8 NEON | 190.9 / 49.5 | 1341.3 | 1272.3 | 3058.1 MB |
| W8PC | on-the-fly B tile FP16 | 248.0 / 23.4 | 1032.1 | 2693.0 | 3049.1 MB |
| W8G128 | on-the-fly B tile FP16 | 237.8 / 21.4 | 1076.5 | 2948.5 | 3062.2 MB |

### Interpretation

On-the-fly tile dequant is not a useful performance path in the current form.

- GEMM is roughly equal to native W8, not closer to FP16FML.
- GEMV is much slower than native W8 because it pays int8-to-FP16 conversion
  and scale handling for every loaded B tile.
- Package prefill improves from native W8 because the M>1 path gets a better
  packed-A/8x8 structure, but decode collapses from about 50 tok/s to about
  23 tok/s.
- Peak RSS stays near native W8, confirming this path keeps the W8 runtime
  memory profile and does not allocate a full FP16 copy.

This validates the design constraint: if W8 is kept compressed at runtime, the
kernel needs to do useful int8 dot work, not repeatedly expand B into FP16 just
before multiply.

### llama.cpp / GGML Comparison

Verified against local `../llama.cpp` checkout `5c7c22c3e` from 2026-06-25.
The important detail is more specific than "on-the-fly dequant":

- `src0` is the weight matrix and keeps its quantized block type.
- each quantized weight type declares a `vec_dot_type`; for `Q4_0` and `Q8_0`
  this is `Q8_0`
- when `src1` activation is FP32, CPU mul_mat quantizes `src1` into that
  `vec_dot_type` work buffer before the dot
- ARM kernels then compute `Q*_weight x Q8_activation` as int8 dot/i8mm, with
  FP32 scale application at block granularity

Relevant llama.cpp files:

- `ggml/src/ggml-cpu/ggml-cpu.c`: `type_traits_cpu`, `vec_dot_type`,
  `from_float`, and mul_mat `params->wdata` conversion
- `ggml/src/ggml-cpu/arch/arm/quants.c`: `ggml_vec_dot_q8_0_q8_0` and
  `ggml_vec_dot_q4_0_q8_0`, including `vmmlaq_s32` paths
- `ggml/src/ggml-cpu/arch/arm/repack.cpp`: multi-row/column packed kernels and
  activation `Q8_0x4` packing helpers

In other words, the llama.cpp-style route is:

```text
FP32 activation -> temporary Q8 block buffer
quantized weight block + Q8 activation block -> int8 dot/i8mm -> FP32 output
```

not:

```text
quantized weights -> FP16 expansion -> FP16 matmul
```

### llama.cpp CPU Bench: Qwen3.5-0.8B

Command shape:

```bash
../llama.cpp/build-cpu/bin/llama-bench -m ...gguf -ngl 0 -p 256 -n 64 -t 4 -r 5 -o md
```

Same machine, CPU-only binary, 4 threads. The F16 row is noisy on this run, but
the quantized direction is clear:

| GGUF | Size | pp256 | tg64 |
|---|---:|---:|---:|
| F16 | 1.44 GiB | 527.4 +/- 44.9 | 82.1 +/- 14.6 |
| Q8_0 | 784.5 MiB | 639.4 +/- 16.2 | 117.9 +/- 13.7 |
| Q4_0 | 478.8 MiB | 784.3 +/- 10.1 | 187.6 +/- 1.9 |

This was the opposite of our pre-Attempt-3 W8 behavior:

- mollm native W8PC 0.8B: ~194.5 / 49.7
- mollm on-the-fly FP16 tile W8PC: ~248.0 / 23.4
- llama.cpp Q8_0: ~639.4 / 117.9
- llama.cpp Q4_0: ~784.3 / 187.6

The gap is not explained by package format. It is the kernel strategy. llama.cpp
spends a temporary activation-quantization pass, then amortizes that cost with
block int8 dot/i8mm kernels. Before Attempt 3, our W8 path kept FP32
activations and either expanded int8 weights to FP32/FP16 inside the K loop or
used FP16 accumulation after tile expansion; both missed the main int8-dot
advantage.

## Attempt 3: Q8 Activation Dot for W8 Decode

### Goal

Move mollm W8 toward the llama.cpp/ggml strategy:

```text
FP32 activation -> temporary Q8 blocks
INT8 weight x Q8 activation -> int32 dot -> FP32 scale/output
```

The first target is decode/GEMV, where `M == 1` and activation quantization
cost is small. Prefill/GEMM is tested but not made default unless it beats the
existing native W8 GEMM.

### Implementation

- Added temporary Q8 activation quantization in 32-element K blocks.
- Added Q8-dot scalar fallback that splits accumulation at both activation-Q8
  block boundaries and W8 weight group boundaries.
- Added NEON Q8-dot GEMV for interleaved INT8 B `[N/8, K, 8]`.
- Added a first NEON 4x8 Q8-dot GEMM prototype behind
  `MOLLM_W8_Q8_DOT_GEMM=1`.
- Default W8 dispatch:
  - `M == 1`: Q8-dot GEMV, unless `MOLLM_W8_NO_Q8_DOT=1`
  - `M > 1`: native W8 dequant GEMM
  - `MOLLM_W8_ONFLY_FP16=1`: diagnostic path still takes priority

The current NEON Q8-dot kernels use int16 multiply/add rather than i8mm
`vmmla`/dot-product instructions. This is intentionally conservative for the
first correctness pass; the optimized path should repack A/B for real i8mm use.

### Correctness

```bash
./build/test_matmul
MOLLM_W8_Q8_DOT_GEMM=1 ./build/test_matmul
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8g128.mollm
```

All passed in local runs.

Short Qwen3.5-0.8B CE/PPL remains unchanged from the native W8 quick test:

| Package | Short CE/PPL | CE delta vs FP16 |
|---|---:|---:|
| W8PC | 3.0969 / 22.13 | +0.0058 |
| W8G128 | 3.0926 / 22.03 | +0.0015 |

Longer 256-token reference-token audit:

| Package | 256-token CE/PPL | CE delta vs FP16 |
|---|---:|---:|
| FP16 | 2.1388 / 8.49 | - |
| W8PC | 2.1448 / 8.54 | +0.0060 |
| W8G128 | 2.1376 / 8.48 | -0.0012 |

### Microbench

Repeated with `caffeinate -dimsu`, 4 threads, warmup=3, repeat=10.

#### GEMV: M=1 K=2048 N=6144

| Path | avg ms | GF/s |
|---|---:|---:|
| W8 Q8-dot GEMV default | 0.16 | 158.4 |
| W8 native dequant GEMV (`MOLLM_W8_NO_Q8_DOT=1`) | 0.39 | 64.9 |

#### GEMM: M=128 K=1024 N=4096

| Path | avg ms | GF/s |
|---|---:|---:|
| W8 Q8-dot GEMM prototype (`MOLLM_W8_Q8_DOT_GEMM=1`) | 5.76 | 186.4 |
| W8 native dequant GEMM | 5.00 | 214.6 |

### Package Bench: Qwen3.5-0.8B

Same machine, pp256 + tg64, warmup=3, 4 threads.

| Package | Runtime | pp/tg | prefill ms | decode ms | peak RSS |
|---|---|---:|---:|---:|---:|
| W8PC | Q8-dot GEMV + native GEMM | 184.63 / 70.54 | 1386.57 | 893.05 | 3046.2 MB |
| W8G128 | Q8-dot GEMV + native GEMM | 183.44 / 71.58 | 1395.58 | 880.19 | 3059.1 MB |
| W8PC | native dequant GEMV/GEMM | 184.44 / 44.73 | 1387.95 | 1408.49 | 3044.6 MB |

Earlier all-Q8-dot testing, before GEMM was moved behind the experimental
switch, produced about `152.5 / 64` pp/tg for W8PC/W8G128. That confirms the
first GEMM prototype hurts prefill enough that it should not be the default.

### Interpretation

- Decode improves about 1.58x versus the previous W8 path
  (`44.73 -> 70.54` tok/s for W8PC).
- Prefill stays near the native W8 baseline because GEMM remains on the old
  native dequant kernel by default.
- Q8 activation quantization is directionally correct for decode, matching the
  llama.cpp strategy at a high level.
- The GEMM prototype is not enough: without proper packed/repacked multi-row
  layout and i8mm/dot-product use, Q8-dot loses to the simpler native W8 GEMM.

### Next Ideas

1. Implement packed/repacked Q8-dot GEMM:
   - pack A as Q8 multi-row blocks
   - repack B for i8mm-friendly K/N layout
   - apply scales at block granularity after int32 accumulation
2. Check ARM feature dispatch for dot-product/i8mm and keep conservative
   fallback paths.
3. Re-run 2B/4B strict pp256/tg64 only after GEMM improves; 0.8B is still too
   noisy for final conclusions.

## Attempt 4: Opt-in Repacked Q8-dot GEMM

### Goal

Make Q8 activation dot useful for prefill/GEMM by adding a B layout that matches
ARM dot-product access patterns. The previous GEMM prototype reused the decode
interleaved B layout `[N/8, K, 8]`, which is good for loading 8 output channels
at one K but poor for dot-product instructions that want contiguous K values.

### Implementation

- Added optional INT8 B repack layout:

```text
[N/8, K/32, 8, 32]
```

For each output-channel tile and Q8 block, one column's 32 K values are stored
contiguously. Padding rows and K tail are zero.

- Added padded Q8 activation quantization for GEMM.
- Added a NEON dot-product 4x8 GEMM kernel using `vdotq_s32`.
- Dispatch is opt-in:

```bash
MOLLM_W8_Q8_DOT_GEMM=1 MOLLM_W8_Q8_DOT_REPACK=1 ...
```

- Loader only builds the extra q8dot B layout when
  `MOLLM_W8_Q8_DOT_REPACK=1`, so the default W8 memory profile is unchanged.
- Added `MOLLM_W8_DEBUG_PATHS=1` to print the first few INT8 matmul path/shape
  decisions when debugging dispatch.

### Correctness

```bash
./build/test_matmul
MOLLM_W8_Q8_DOT_GEMM=1 MOLLM_W8_Q8_DOT_REPACK=1 ./build/test_matmul
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm 256
MOLLM_W8_Q8_DOT_GEMM=1 MOLLM_W8_Q8_DOT_REPACK=1 ./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm 256
MOLLM_W8_Q8_DOT_GEMM=1 MOLLM_W8_Q8_DOT_REPACK=1 ./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8g128.mollm 256
```

All passed.

256-token PPL with repacked GEMM:

| Package | Runtime | 256-token CE/PPL | CE delta vs FP16 |
|---|---|---:|---:|
| W8PC | default | 2.1448 / 8.54 | +0.0060 |
| W8PC | repacked Q8-dot GEMM | 2.1475 / 8.56 | +0.0087 |
| W8G128 | repacked Q8-dot GEMM | 2.1369 / 8.47 | -0.0019 |

### Microbench

4 threads, warmup=3, repeat=10.

| Shape | Runtime | avg ms | GF/s |
|---|---|---:|---:|
| M=128 K=1024 N=4096 | native W8 GEMM | 5.05 | 212.7 |
| M=128 K=1024 N=4096 | old Q8-dot GEMM | 5.81 | 184.9 |
| M=128 K=1024 N=4096 | repacked Q8-dot GEMM | 2.07 | 519.1 |
| M=256 K=1024 N=1024 | native W8 GEMM | 2.16 | 248.7 |
| M=256 K=1024 N=1024 | repacked Q8-dot GEMM | 1.13 | 473.3 |
| M=256 K=1024 N=3072 | native W8 GEMM | 6.91 | 233.0 |
| M=256 K=1024 N=3072 | repacked Q8-dot GEMM | 4.43 | 363.5 |
| M=256 K=2048 N=6144 | native W8 GEMM | 27.43 | 234.8 |
| M=256 K=2048 N=6144 | repacked Q8-dot GEMM | 15.30 | 421.1 |

### Package Bench: Qwen3.5-0.8B

Same command shape: pp256 + tg64, warmup=3, 4 threads, greedy decode.

| Package | Runtime | pp/tg | prefill ms | decode ms | peak RSS |
|---|---|---:|---:|---:|---:|
| W8PC | default Q8-dot GEMV + native GEMM | 221.4 / 102.3 | 1156.45 | 615.66 | 3042.1 MB |
| W8PC | repacked Q8-dot GEMM | 387.9 / 102.8 | 660.00 | 612.94 | 3521.7 MB |
| W8G128 | default Q8-dot GEMV + native GEMM | 185.6 / 100.5 | 1379.28 | 626.73 | 3056.0 MB |
| W8G128 | repacked Q8-dot GEMM | 230.8 / 92.4 | 1109.44 | 681.56 | 3535.9 MB |

### Interpretation

- Repacked Q8-dot GEMM is the first W8 prefill path that clearly beats native
  W8 GEMM in both microbench and package benchmark.
- W8PC benefits much more than W8G128 because per-channel scaling has less
  scale handling overhead.
- The cost is memory: enabling the repacked path adds another packed INT8
  weight layout, about +480 MB RSS on Qwen3.5-0.8B.
- Decode is mostly unchanged for W8PC, but W8G128 showed some decode slowdown,
  likely cache pressure from the extra resident packed weights.

### Next Ideas

1. Keep repacked GEMM opt-in until the memory/speed tradeoff is decided.
2. Consider enabling it by default only for W8PC, where the speedup is large.
3. Reduce memory overhead by replacing the default interleaved GEMM layout when
   repacked GEMM is enabled, while preserving the decode-friendly layout.
4. Improve per-group scale handling before making W8G128 decisions.

## Reference: llama.cpp Q8_0 All-Model CPU Bench

Fresh run on 2026-07-01 after generating Q8_0 GGUFs for the larger models.
Command shape:

```bash
caffeinate -dimsu ../llama.cpp/build-cpu/bin/llama-bench \
  -m <model-Q8_0.gguf> -ngl 0 -p 256 -n 64 -t 4 -r 5 -o md
```

llama.cpp build: `5c7c22c3e` / 9803. CPU backend reports `BLAS`, 4 threads.

| GGUF | Size | Params | pp256 | tg64 |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B Q8_0 | 784.52 MiB | 772.85 M | 809.11 +/- 0.97 | 167.23 +/- 0.88 |
| Youtu-LLM-2B Q8_0 | 1.94 GiB | 1.96 B | 257.31 +/- 20.32 | 85.84 +/- 1.65 |
| Qwen3.5-4B Q8_0 | 4.28 GiB | 4.33 B | 137.45 +/- 4.96 | 38.25 +/- 1.82 |

Youtu-LLM-2B was run twice because the first prefill result had high variance:

| Run | pp256 | tg64 |
|---|---:|---:|
| 1 | 249.17 +/- 36.09 | 84.46 +/- 1.87 |
| 2 | 257.31 +/- 20.32 | 85.84 +/- 1.65 |

Interpretation:

- llama.cpp Q8_0 remains a strong target for mollm W8: its decode speed is well
  above our current 0.8B W8 decode even after Q8-dot GEMV.
- The 4B Q8_0 prefill reference is 137.45 tok/s, which is already above the
  current FP16 mollm 4B baseline in `CURRENT_STATE.md` (115 tok/s). This is the
  concrete bar for W8/W4 prefill optimization.
- Youtu-LLM-2B Q8_0 decode at about 85 tok/s is also materially above the
  current FP16 mollm 2B decode baseline, so MLA decode should benefit from the
  same quantized dot-product direction once larger W8 packages are converted.

## Attempt 5: Repacked Q8-dot GEMV for Decode

### Goal

Close the decode gap to llama.cpp Q8_0. Before this attempt, mollm decode used
Q8 activation dot, but the GEMV kernel still reused the decode-friendly
interleaved layout `[N/8, K, 8]` and did int16 multiply/add in the inner loop.
The `vdotq_s32` path existed only for the opt-in repacked GEMM kernel.

### Implementation

- Added `matmul_int8_q8dot_neon_gemv_repacked_range()`.
- Reused the existing optional B layout:

```text
[N/8, K/32, 8, 32]
```

- For `M == 1`, `MOLLM_W8_Q8_DOT_REPACK=1` now routes GEMV/decode to the
  repacked `vdotq_s32` kernel when the repacked B copy exists and
  `group_size % 32 == 0`.
- `MOLLM_W8_Q8_DOT_GEMM=1` is not required for repacked GEMV; it only controls
  whether prefill/GEMM also uses Q8-dot.
- Added an `M=1, K=32, N=8` matmul test case with `q8_repack_data`, so
  `MOLLM_W8_Q8_DOT_REPACK=1 ./build/test_matmul` covers the new path.

Default W8 behavior is unchanged: the extra repacked B copy is still built only
when `MOLLM_W8_Q8_DOT_REPACK=1`.

### Correctness

```bash
./build/test_matmul
MOLLM_W8_Q8_DOT_REPACK=1 ./build/test_matmul
MOLLM_W8_Q8_DOT_REPACK=1 ./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm 256
MOLLM_W8_Q8_DOT_REPACK=1 ./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8g128.mollm 256
MOLLM_W8_Q8_DOT_GEMM=1 MOLLM_W8_Q8_DOT_REPACK=1 ./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm 256
MOLLM_W8_Q8_DOT_GEMM=1 MOLLM_W8_Q8_DOT_REPACK=1 ./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8g128.mollm 256
```

All passed.

256-token PPL is unchanged from Attempt 4 for the full repack path:

| Package | Runtime | 256-token CE/PPL | CE delta vs FP16 |
|---|---|---:|---:|
| W8PC | repacked GEMV only | 2.1448 / 8.54 | +0.0060 |
| W8G128 | repacked GEMV only | 2.1376 / 8.48 | -0.0012 |
| W8PC | repacked GEMM + GEMV | 2.1475 / 8.56 | +0.0087 |
| W8G128 | repacked GEMM + GEMV | 2.1369 / 8.47 | -0.0019 |

### Microbench

4 threads, warmup=3, repeat=10.

| Shape | Runtime | avg ms | GF/s |
|---|---|---:|---:|
| M=1 K=2048 N=6144 | native W8 GEMV | 0.35 | 72.0 |
| M=1 K=2048 N=6144 | Q8-dot GEMV default | 0.15 | 170.9 |
| M=1 K=2048 N=6144 | repacked Q8-dot GEMV | 0.10 | 251.0 |
| M=1 K=2048 N=128256 | Q8-dot GEMV default | 2.65 | 198.2 |
| M=1 K=2048 N=128256 | repacked Q8-dot GEMV | 1.75 | 300.4 |

### Package Bench: Qwen3.5-0.8B

Same command shape: pp256 + tg64, warmup=3, 4 threads, greedy decode. These
are single fresh runs on the current machine state, so compare rows within this
table rather than against older attempts.

| Package | Runtime | pp/tg | prefill ms | decode ms | peak RSS |
|---|---|---:|---:|---:|---:|
| W8PC | default Q8-dot GEMV + native GEMM | 222.9 / 103.2 | 1148.62 | 610.36 | 3042.2 MB |
| W8PC | repacked GEMV + native GEMM | 217.0 / 133.3 | 1179.96 | 472.68 | 3519.3 MB |
| W8PC | repacked GEMV + repacked GEMM | 350.6 / 132.8 | 730.09 | 474.43 | 3521.1 MB |
| W8G128 | default Q8-dot GEMV + native GEMM | 218.7 / 102.9 | 1170.39 | 612.30 | 3057.4 MB |
| W8G128 | repacked GEMV + native GEMM | 214.9 / 115.5 | 1191.33 | 545.49 | 3531.9 MB |
| W8G128 | repacked GEMV + repacked GEMM | 271.3 / 112.8 | 943.79 | 558.31 | 3534.4 MB |

### Interpretation

- Repacked GEMV gives a real decode win: W8PC `103.2 -> 133.3` tok/s and
  W8G128 `102.9 -> 115.5` tok/s in package bench.
- The gap to llama.cpp Q8_0 on Qwen3.5-0.8B shrinks from about `1.62x`
  (`167 / 103`) to about `1.25x` (`167 / 133`), but it is not closed.
- The memory cost is the same as Attempt 4: about +480 MB RSS on 0.8B because
  the runtime keeps the default interleaved INT8 copy plus the q8dot repack
  copy.
- W8PC is the more attractive first default candidate. W8G128 benefits less
  because per-group scale handling remains heavier and the extra resident
  weight layout increases cache pressure.

### Next Ideas

1. Reduce repack memory by avoiding simultaneous ownership of both INT8 layouts
   where possible.
2. Specialize W8PC scale handling in the repacked kernels.
3. Continue comparing against llama.cpp Q8_0 at the kernel level; remaining
   decode gap is now more likely kernel overhead and runtime overhead than the
   high-level quantization strategy.

### Multi-Model Package Bench

Generated additional W8PC packages:

```bash
python3 models/converter.py ../Youtu-LLM-2B youtu-llm-2b_w8pc.mollm 32 256 w8pc
python3 models/converter.py ../Qwen3.5-4B qwen35_4b_w8pc.mollm 24 256 w8pc
```

Bench command shape:

```bash
caffeinate -dimsu ./build/mollm_bench \
  --package <model_w8pc.mollm> --prompt-tokens 256 --max-new-tokens 64 \
  --warmup 3 --threads 4 --temperature 0
```

`GEMV repack` means `MOLLM_W8_Q8_DOT_REPACK=1`.
`full repack` means `MOLLM_W8_Q8_DOT_GEMM=1 MOLLM_W8_Q8_DOT_REPACK=1`.
Rows are single fresh runs on the same machine state, so use them directionally.

| Model | Runtime | pp/tg | prefill ms | decode ms | peak RSS |
|---|---|---:|---:|---:|---:|
| Qwen3.5-0.8B W8PC | default | 206.9 / 103.6 | 1237.17 | 607.97 | 3042.4 MB |
| Qwen3.5-0.8B W8PC | GEMV repack | 226.1 / 133.5 | 1132.11 | 471.87 | 3518.9 MB |
| Qwen3.5-0.8B W8PC | full repack | 376.8 / 134.9 | 679.44 | 467.17 | 3521.5 MB |
| Youtu-LLM-2B W8PC | default | 66.6 / 39.7 | 3845.13 | 1586.25 | 6195.5 MB |
| Youtu-LLM-2B W8PC | GEMV repack | 64.0 / 57.4 | 4000.11 | 1097.69 | 7832.0 MB |
| Youtu-LLM-2B W8PC | full repack | 116.3 / 53.9 | 2201.75 | 1169.67 | 7838.0 MB |
| Qwen3.5-4B W8PC | default | 28.4 / 19.0 | 9017.54 | 3317.82 | 12170.5 MB |
| Qwen3.5-4B W8PC | GEMV repack | 27.8 / 27.8 | 9203.12 | 2266.48 | 15579.3 MB |
| Qwen3.5-4B W8PC | full repack | 52.3 / 27.6 | 4895.09 | 2281.01 | 15587.5 MB |

Comparison to the llama.cpp Q8_0 reference from the same log:

| Model | mollm W8PC full repack | llama.cpp Q8_0 | decode gap |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 376.8 / 134.9 | 809.1 / 167.2 | 1.24x |
| Youtu-LLM-2B | 116.3 / 53.9 | 257.3 / 85.8 | 1.59x |
| Qwen3.5-4B | 52.3 / 27.6 | 137.5 / 38.3 | 1.39x |

Interpretation:

- Repacked GEMV improves decode on every tested W8PC model:
  - Qwen3.5-0.8B: `103.6 -> 133.5` tok/s
  - Youtu-LLM-2B: `39.7 -> 57.4` tok/s
  - Qwen3.5-4B: `19.0 -> 27.8` tok/s
- Full repack is required for meaningful W8PC prefill recovery:
  - Youtu-LLM-2B: `66.6 -> 116.3` tok/s
  - Qwen3.5-4B: `28.4 -> 52.3` tok/s
- The remaining gap to llama.cpp Q8_0 is still large, especially in prefill.
  For 2B/4B this is not just decode GEMV anymore; GEMM kernel efficiency,
  scale handling, and runtime memory traffic are now the main suspects.
- Memory cost grows with model size because the current implementation keeps
  both the default interleaved INT8 copy and the q8dot repack copy resident.

## Attempt 6: Make Q8-dot Repack the Default W8 Layout

### Goal

Keep the fastest W8 path by default, but remove the avoidable memory cost from
Attempt 5. The previous full-repack mode kept both runtime INT8 layouts:

```text
[N/8, K, 8]          old interleaved native-W8 layout
[N/8, K/32, 8, 32]  q8dot repack layout
```

Once repacked Q8-dot became the preferred decode and prefill path, the old
interleaved copy was no longer needed for the default runtime.

### Implementation

- W8 now defaults to q8dot full repack:
  - decode/GEMV uses `matmul_int8_q8dot_neon_gemv_repacked_range()`
  - prefill/GEMM uses `matmul_int8_q8dot_neon_4x8_repacked_range()`
- Loader builds only `[N/8, K/32, 8, 32]` for supported INT8 weights by
  default.
- Loader builds the old `[N/8, K, 8]` layout only when needed:
  - `MOLLM_W8_NO_Q8_DOT_REPACK=1`
  - `MOLLM_W8_NO_Q8_REPACK=1`
  - `MOLLM_W8_NO_Q8_DOT=1`
  - `MOLLM_W8_NO_Q8_DOT_GEMM=1`
  - `MOLLM_W8_ONFLY_FP16=1`
  - q8dot unsupported, such as group size not divisible by 32 or no dotprod
    target
  - `MOLLM_W8_KEEP_INT8_INTERLEAVED=1` for diagnostics
- The old non-repacked Q8-dot GEMM prototype is no longer default fallback. It
  still requires explicit `MOLLM_W8_Q8_DOT_GEMM=1`; otherwise disabling q8dot
  repack falls back to native W8 GEMM.

### Correctness

```bash
./build/test_matmul
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm 256
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8g128.mollm 256
MOLLM_W8_NO_Q8_DOT_REPACK=1 ./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm 64
ctest --test-dir build --output-on-failure -E 'test_e2e|bench_matmul|bench_sdpa'
```

All passed. Debug path checks:

```bash
MOLLM_W8_DEBUG_PATHS=1 ...
# default: q8dot_gemm_repack

MOLLM_W8_NO_Q8_DOT_REPACK=1 MOLLM_W8_DEBUG_PATHS=1 ...
# fallback: native_w8_gemm
```

### Package Bench

Same command shape as Attempt 5, single fresh runs.

| Model | Runtime | pp/tg | prefill ms | decode ms | peak RSS |
|---|---|---:|---:|---:|---:|
| Qwen3.5-0.8B W8PC | default q8dot repack | 376.1 / 126.8 | 680.65 | 496.77 | 3045.0 MB |
| Youtu-LLM-2B W8PC | default q8dot repack | 118.4 / 57.3 | 2162.39 | 1099.38 | 6199.8 MB |
| Qwen3.5-4B W8PC | default q8dot repack | 54.5 / 27.7 | 4694.52 | 2275.67 | 12176.4 MB |

Comparison to Attempt 5 full repack:

| Model | Attempt 5 full repack RSS | Attempt 6 default RSS | delta |
|---|---:|---:|---:|
| Qwen3.5-0.8B W8PC | 3521.5 MB | 3045.0 MB | -476.5 MB |
| Youtu-LLM-2B W8PC | 7838.0 MB | 6199.8 MB | -1638.2 MB |
| Qwen3.5-4B W8PC | 15587.5 MB | 12176.4 MB | -3411.1 MB |

### Interpretation

- The q8dot path is now the actual W8 default, not an opt-in benchmark mode.
- Removing the old INT8 interleaved copy restores the W8 memory profile while
  retaining full-repack prefill speed.
- Decode remains below llama.cpp Q8_0, so the next work is still kernel/runtime
  efficiency, not choosing between layouts.

## Attempt 7: Shape-Level Matmul Profiling

### Goal

Stop guessing which W8 matmul shapes matter. The regular `--profile` output
aggregates all MATMUL nodes together, so it cannot distinguish q8dot GEMM,
q8dot GEMV, and FP16 lm_head.

### Implementation

- Added `MOLLM_MATMUL_SHAPE_PROFILE=1`.
- `kernel_matmul_fp32()` now records aggregate rows keyed by:
  `phase`, `path`, `M`, `N`, `K`, `group_size`, `groups_per_row`, thread count,
  q8 repack presence, and interleaved layout.
- Engine marks phases as:
  - `prefill_graph`
  - `prefill_lmhead`
  - `decode_graph`
  - `decode_lmhead`
- `mollm_bench --profile` prints a `matmul_shape_profile` table when the env
  flag is enabled.

### Validation

```bash
ninja -C build mollm_bench test_matmul test_quantized_e2e
./build/test_matmul
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm 64
ctest --test-dir build --output-on-failure -E 'test_e2e|bench_matmul|bench_sdpa'
```

All passed.

### Shape Profile

Command shape:

```bash
MOLLM_MATMUL_SHAPE_PROFILE=1 caffeinate -dimsu ./build/mollm_bench \
  --package <w8pc.mollm> --prompt-tokens 256 --max-new-tokens 16 \
  --warmup 1 --threads 4 --temperature 0 --profile
```

These are not strict benchmark-protocol rows; they are profiling runs with
short decode, used to identify dominant shape classes.

Youtu-LLM-2B W8PC:

| phase | path | M | N | K | calls | total ms | avg ms | GMAC/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| prefill_graph | q8dot_gemm_repack | 256 | 12288 | 2048 | 32 | 859.05 | 26.845 | 239.98 |
| prefill_graph | q8dot_gemm_repack | 256 | 2048 | 6144 | 32 | 429.07 | 13.408 | 240.24 |
| prefill_graph | q8dot_gemm_repack | 256 | 3072 | 1536 | 32 | 161.56 | 5.049 | 239.26 |
| prefill_graph | q8dot_gemm_repack | 256 | 2048 | 2048 | 32 | 143.81 | 4.494 | 238.92 |
| decode_graph | q8dot_gemv_repack | 1 | 12288 | 2048 | 480 | 93.18 | 0.194 | 129.63 |
| decode_lmhead | fp16_gemv_interleaved_fp16acc | 1 | 128256 | 2048 | 15 | 36.73 | 2.448 | 107.28 |

Qwen3.5-4B W8PC:

| phase | path | M | N | K | calls | total ms | avg ms | GMAC/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| prefill_graph | q8dot_gemm_repack | 256 | 9216 | 2560 | 64 | 1721.03 | 26.891 | 224.60 |
| prefill_graph | q8dot_gemm_repack | 256 | 2560 | 9216 | 32 | 867.29 | 27.103 | 222.85 |
| prefill_graph | q8dot_gemm_repack | 256 | 8192 | 2560 | 32 | 780.68 | 24.396 | 220.06 |
| prefill_graph | q8dot_gemm_repack | 256 | 2560 | 4096 | 32 | 383.69 | 11.990 | 223.88 |
| decode_graph | q8dot_gemv_repack | 1 | 9216 | 2560 | 960 | 178.56 | 0.186 | 126.84 |
| decode_lmhead | fp16_gemv_interleaved_fp16acc | 1 | 248320 | 2560 | 15 | 91.08 | 6.072 | 104.70 |

### Interpretation

- The remaining W8 gap is still MATMUL, but now we know exactly where:
  prefill is dominated by a handful of `M=256` q8dot GEMM shapes, and decode is
  dominated by the matching `M=1` q8dot GEMV shapes.
- The current q8dot GEMM path reaches only ~220-240 GMAC/s on 2B/4B profiling
  runs. That is much lower than the FP16FML matmul roof and aligns with the
  observed prefill gap to llama.cpp Q8_0.
- q8dot GEMV is ~120-130 GMAC/s for the major decode shapes. It is improved
  versus native W8 but still below the llama.cpp Q8_0 decode reference.
- lm_head is visible and non-zero, especially for 4B decode, but it is not the
  main W8 graph bottleneck.
- llama.cpp does not hard-code model matrix sizes; it selects repack/tile
  kernels by quant type, CPU feature, and divisibility such as 4x8/8x8. The
  corresponding next step here is to add better general tile kernels for these
  shape classes, especially ARM i8mm/`vmmlaq_s32` GEMM where available, not
  per-layer shape hardcoding.
- Current default CMake flags do not enable `__ARM_FEATURE_MATMUL_INT8` on this
  Apple build. Plain `clang++` defines DOTPROD but not MATMUL_INT8; `-mcpu=native`
  or `-march=armv8.6-a+i8mm+fp16+fp16fml` defines it while preserving FP16FML.
- Added build-time gates for local measurement:
  `-DMOLLM_ARM_NATIVE=ON` passes `-mcpu=native`, and `-DMOLLM_ARM_I8MM=ON`
  passes `-march=armv8.6-a+i8mm+fp16+fp16fml`. Both are OFF by default to keep
  the portable ARM64 target unchanged.

## Attempt 8: ARM i8mm Repacked Q8-dot GEMM

### Goal

Use ARMv8.6-A INT8 matrix multiply (`vmmlaq_s32`) for the dominant W8 prefill
shapes identified in Attempt 7, while keeping the existing q8dot repacked weight
layout and correctness behavior.

### Implementation

- Added `matmul_int8_q8dot_neon_4x8_repacked_i8mm_range()` under
  `__ARM_FEATURE_MATMUL_INT8`.
- Reuses the default q8dot weight layout `[N/8, K/32, 8, 32]`; no new runtime
  weight copy is introduced.
- Computes 4 output rows x 8 output columns per tile. Each K block of 32 is
  consumed with four `vmmlaq_s32` groups, then row accumulators are reordered,
  scaled by `activation_scale * weight_scale`, and stored with tail handling.
- Dispatch prefers `q8dot_gemm_repack_i8mm` when the binary is built with i8mm
  support. `MOLLM_W8_NO_I8MM=1` forces the previous DOTPROD GEMM path for A/B
  diagnostics.
- Fixed the explicit i8mm build flag to
  `-march=armv8.6-a+i8mm+fp16+fp16fml`; plain `+i8mm` disabled the FP16/FP16FML
  intrinsics used elsewhere in the runtime.

### Correctness

```bash
cmake -G Ninja -B build_i8mm -DCMAKE_BUILD_TYPE=Release -DMOLLM_ARM_I8MM=ON
ninja -C build_i8mm mollm_bench test_matmul test_quantized_e2e
./build_i8mm/test_matmul
./build_i8mm/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm 64
ctest --test-dir build_i8mm --output-on-failure -E 'test_e2e|bench_matmul|bench_sdpa'
ctest --test-dir build --output-on-failure -E 'test_e2e|bench_matmul|bench_sdpa'
```

All passed. Debug dispatch checks:

```bash
MOLLM_W8_DEBUG_PATHS=1 ./build_i8mm/test_matmul
# path=q8dot_gemm_repack_i8mm for supported GEMM shapes

MOLLM_W8_NO_I8MM=1 MOLLM_W8_DEBUG_PATHS=1 ./build_i8mm/test_matmul
# path=q8dot_gemm_repack
```

### Microbench

Commands used the dominant Attempt 7 prefill shapes, 4 threads, warmup=3,
repeat=10.

| Shape | Build/path | avg ms | bench GFLOP/s | speedup |
|---|---|---:|---:|---:|
| M=256,K=2048,N=12288 | default DOTPROD | 26.65 | 483.4 | - |
| M=256,K=2048,N=12288 | i8mm | 16.93 | 761.1 | 1.57x |
| M=256,K=2560,N=9216 | default DOTPROD | 24.00 | 503.4 | - |
| M=256,K=2560,N=9216 | i8mm | 16.29 | 741.7 | 1.47x |

### Package Bench

Single fresh runs, pp256 + tg64, warmup=3, 4 threads. These are not strict
5-run medians, so use them as A/B direction.

| Model | Build/path | pp/tg | prefill ms | decode ms | peak RSS |
|---|---|---:|---:|---:|---:|
| Youtu-LLM-2B W8PC | default DOTPROD | 126.78 / 57.23 | 2019.27 | 1100.78 | 6200.0 MB |
| Youtu-LLM-2B W8PC | i8mm GEMM | 203.73 / 57.35 | 1256.54 | 1098.61 | 6199.3 MB |
| Qwen3.5-4B W8PC | default DOTPROD | 53.33 / 27.08 | 4799.97 | 2326.10 | 12180.6 MB |
| Qwen3.5-4B W8PC | i8mm GEMM | 87.26 / 26.55 | 2933.77 | 2373.10 | 12175.7 MB |

Compared with the earlier llama.cpp Q8_0 reference, i8mm closes a large part of
the prefill gap but does not eliminate it:

| Model | mollm W8PC i8mm | llama.cpp Q8_0 |
|---|---:|---:|
| Youtu-LLM-2B | 203.7 / 57.4 | 257.3 / 85.8 |
| Qwen3.5-4B | 87.3 / 26.6 | 137.5 / 38.3 |

### Interpretation

- i8mm is the right direction for prefill: the main GEMM microbench shapes gain
  about 1.5x, and package prefill improves about 61-64% in single-run A/B.
- Decode is unchanged because decode is dominated by `M=1` q8dot GEMV, not GEMM.
  The next decode work remains a better GEMV tile/path and lm_head accounting.
- Remaining prefill gap is likely scale handling, activation Q8 pack overhead,
  output conversion/store cost, and broader runtime memory traffic rather than
  lack of an INT8 matrix-multiply instruction.

## Attempt 9: Vectorized GEMV Reduction + Scale Mode Split

### Goal

Fix the main decode inefficiency found after Attempt 8: the repacked q8dot GEMV
kept each output channel as a scalar and used `vaddvq_s32` for every column and
every 32-element K block. That threw away vector structure too early.

Also make the weight-scale semantics explicit:

- `groups_per_row == 1`: per-channel weight scale
- `group_size == 32`: true per-block weight scale
- otherwise: per-group weight scale, such as `w8g128`

### Implementation

- Added `W8ScaleMode::{PerChannel, PerBlock32, PerGroup}` and
  `w8_scale_group()`.
- Reworked `matmul_int8_q8dot_neon_gemv_repacked_range()`:
  - keeps 8 output channels as two `float32x4_t` accumulators
  - uses `vpaddq_s32` to reduce four column sums at once
  - removes the per-column `vaddvq_s32` loop
- Hoisted per-channel B scales for q8dot GEMV, DOTPROD GEMM fallback, and i8mm
  GEMM. True per-block/per-group modes still load scales by the current block's
  group id.
- Added q8 activation quantization profiling counters to `mollm_bench`:
  `q8_quant_a_ms` and `q8_quant_a_calls`.

### Correctness

```bash
ninja -C build_i8mm mollm_bench test_matmul test_quantized_e2e bench_matmul
ninja -C build mollm_bench test_matmul test_quantized_e2e bench_matmul
./build_i8mm/test_matmul
./build/test_matmul
./build_i8mm/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm 64
./build_i8mm/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8g128.mollm 64
./build_i8mm/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm 256
./build_i8mm/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8g128.mollm 256
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm 256
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8g128.mollm 256
```

All passed.

Post-cleanup 256-token CE/PPL:

| Package | CE/PPL | CE delta vs FP16 |
|---|---:|---:|
| FP16 | 2.1388 / 8.49 | - |
| W8PC | 2.1473 / 8.56 | +0.0085 |
| W8G128 | 2.1413 / 8.51 | +0.0025 |

### Microbench

Sequential runs on the i8mm build, 4 threads.

| Shape | avg ms | bench GFLOP/s |
|---|---:|---:|
| GEMV M=1,K=2048,N=12288 | 0.12 | 437.7 |
| GEMV M=1,K=2560,N=9216 | 0.11 | 429.1 |
| GEMM M=256,K=2048,N=12288 | 13.11 | 982.5 |
| GEMM M=256,K=2560,N=9216 | 12.20 | 990.4 |

### Package Bench

Single fresh runs, pp256 + tg64, warmup=3, 4 threads. These are not strict
5-run medians.

| Model | pp/tg | prefill ms | decode ms | q8 quant A |
|---|---:|---:|---:|---:|
| Youtu-LLM-2B W8PC | 230.42 / 87.12 | 1111.01 | 723.14 | 94.22 ms |
| Qwen3.5-4B W8PC | 106.74 / 42.44 | 2398.41 | 1484.33 | 164.29 ms |

Comparison to the earlier llama.cpp Q8_0 reference:

| Model | mollm W8PC Attempt 9 | llama.cpp Q8_0 |
|---|---:|---:|
| Youtu-LLM-2B | 230.4 / 87.1 | 257.3 / 85.8 |
| Qwen3.5-4B | 106.7 / 42.4 | 137.5 / 38.3 |

### Interpretation

- Decode is no longer the large W8 gap for 2B/4B. The 2B decode run slightly
  exceeds the recorded llama.cpp Q8_0 reference; 4B decode is also above the
  reference in this single run.
- Prefill improved but still trails llama.cpp Q8_0, especially on 4B. The next
  prefill work should focus on making the GEMM layout more like ggml's
  `block_q8_0x4`: packed A layout, scale placement, and store/reorder overhead.
- q8 activation quantization remains a minor but visible cost: about 4-6% of
  total matmul time in these package runs.

## Attempt 10: W4 Per-Group Correctness Baseline

### Goal

Add W4 per-group weight-only quantization before optimizing kernels. The intent
is to validate the file format, converter, loader metadata checks, and runtime
math first, then make a separate decision about W4 group size and kernel layout.

### Implementation

- Added `Precision::INT4` to Python/C++ precision enums.
- Added converter mode `w4gN` for Qwen3.5 and Youtu converters.
- Added `quantize_weight_w4_group()`:
  - symmetric signed W4 per K-group
  - `scale = max(abs(block)) / 7`
  - quant values clipped to `[-7, 7]`
  - two's-complement signed nibbles packed low-nibble first
- Extended `_write_weight_file()` so packed W4 data can keep a logical header
  shape `[N, K]` while the physical data buffer is `[N, ceil(K / 2)]`.
- Extended runtime loader validation:
  - INT8 data size must be `N * K`
  - INT4 data size must be `N * ceil(K / 2)`
  - both require `num_groups == N * ceil(K / group_size)` and matching scale
    bytes.
- Added INT4 scalar matmul fallback:
  - unpack signed nibble
  - multiply by the row/group FP32 scale
  - accumulate into FP32
  - apply fused activation after writeback

No W4 NEON, q8dot, or repacked layout is implemented in this attempt.

### Correctness

```bash
python3 -m py_compile models/converter.py models/transpile.py models/qwen35.py models/mla.py
ninja -C build test_matmul test_tensor test_quantized_e2e
./build/test_tensor
./build/test_matmul
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm 32
python3 models/converter.py ../Qwen3.5-0.8B qwen35_0.8b_w4g128.mollm 24 256 w4g128
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w4g128.mollm 32
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w4g128.mollm 256
ctest --test-dir build --output-on-failure
```

All passed. `test_matmul` now includes an INT4 per-group case with odd `K`
to cover packed-byte tail handling.

### Qwen3.5-0.8B Quality Smoke

Reference-token CE/PPL:

| Package | Size | 32-token CE/PPL | 256-token CE/PPL | CE delta at 256 |
|---|---:|---:|---:|---:|
| FP16 | 1518.9 MB | 3.0911 / 22.00 | 2.1388 / 8.49 | - |
| W8G128 | 1036.8 MB | 3.0926 / 22.03 | 2.1413 / 8.51 | +0.0025 |
| W4G128 | 788.0 MB | 3.3823 / 29.44 | 2.3373 / 10.35 | +0.1985 |

Greedy smoke passed for W4G128: prefill returned token `271`, followed by decode
tokens `9419` and `11`, matching the W8 smoke path on this prompt.

### Interpretation

- W4G128 is functionally correct enough for the existing smoke suite: finite
  logits, matching vocab shape, reusable workspace, and short decode all pass.
- Quality drift is much larger than W8G128 on the current 0.8B reference sample.
  This may simply mean W4 needs smaller groups or a better quantization scheme;
  the current result should not be treated as the final W4 quality point.
- Runtime speed is not meaningful yet because all W4 matmuls use scalar unpack
  and dequant. The next useful W4 performance step is a packed layout and
  NEON/q8dot-style kernel, after selecting a group-size target.

### Next Ideas

1. Convert and compare `w4g64` and `w4g32` on the same 256-token reference.
2. Add a W4 matmul microbench mode so scalar baseline cost is explicit.
3. Design W4 repack format with scale placement before writing the first NEON
   kernel.

## Attempt 11: Qwen3.5 Mixed W4 Quality Policy

### Goal

Reduce the W4G128 quality drift before investing in W4 kernels. llama.cpp does
not quantize Q4_K_M as pure 4-bit for every tensor; its quantizer promotes
sensitive classes such as output/tied embeddings, attention-V-like tensors, and
FFN down projections to higher-bit formats unless `--pure` is requested.

Mollm's current W4 format is simpler than llama.cpp Q4_K: symmetric signed int4
per K-group with one FP32 scale, no asymmetric min term, no Q4_K super-block
scale/min packing, and no imatrix. The first mixed policy is therefore
quality-biased rather than an exact Q4_K_M clone.

### Implementation

- Added Qwen3.5 converter mode `w4mixgN`.
- Base policy: W4G`N` for quantizable projection/linear weights.
- Promoted these Qwen3.5 tensor classes to W8G`N`:
  - `linear_attn.in_proj_qkv.weight`
  - `self_attn.v_proj.weight`
  - `linear_attn.out_proj.weight`
  - `self_attn.o_proj.weight`
  - `mlp.down_proj.weight`
- Existing `w4gN` remains unchanged for A/B.
- Added `tests/test_qwen35_quant_policy.py` and registered it in CTest.

An earlier, less aggressive policy promoted QKV/V plus selected FFN down layers
using llama.cpp's layer-selection pattern. It improved 256-token CE delta from
`+0.1985` to `+0.1134`, but remained too noisy for a quality target. The final
policy promotes all output-like and down-projection tensors.

### Validation

```bash
python3 -m py_compile models/converter.py models/transpile.py models/qwen35.py models/mla.py tests/test_qwen35_quant_policy.py
python3 tests/test_qwen35_quant_policy.py
python3 models/converter.py ../Qwen3.5-0.8B qwen35_0.8b_w4mixg128.mollm 24 256 w4mixg128
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w4mixg128.mollm 32
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w4mixg128.mollm 256
ninja -C build test_quantized_e2e
ctest --test-dir build --output-on-failure
```

All passed. CTest now runs 19 tests including `test_qwen35_quant_policy`.

### Qwen3.5-0.8B Quality Smoke

The final `w4mixg128` export reports `W4=114 W8=72` quantized tensors. Package
size is 915.4 MB by engine metadata (`873 MiB` on disk), versus 788.0 MB for
W4G128 and 1036.8 MB for W8G128.

| Package | Size | 32-token CE/PPL | 256-token CE/PPL | CE delta at 256 |
|---|---:|---:|---:|---:|
| FP16 | 1518.9 MB | 3.0911 / 22.00 | 2.1388 / 8.49 | - |
| W8G128 | 1036.8 MB | 3.0926 / 22.03 | 2.1413 / 8.51 | +0.0025 |
| W4G128 | 788.0 MB | 3.3823 / 29.44 | 2.3373 / 10.35 | +0.1985 |
| W4MIXG128 | 915.4 MB | 3.1879 / 24.24 | 2.1710 / 8.77 | +0.0322 |

Greedy smoke passed for W4MIXG128: prefill returned token `198`, followed by
decode tokens `9419` and `11`.

### Performance Baseline

Strict protocol: Qwen3.5-0.8B, default Release build (`MOLLM_ARM_I8MM=OFF`),
pp256 + tg64, warmup=3, 4 threads, 5-run median.

| Package | pp/tg | prefill ms | decode ms | peak RSS |
|---|---:|---:|---:|---:|
| FP16 | 625.36 / 101.74 | 409.36 | 619.21 | 3994.9 MB |
| W8G128 | 376.88 / 165.59 | 679.27 | 380.45 | 3058.2 MB |
| W4G128 | 9.47 / 9.09 | 27039.88 | 6928.74 | 2339.3 MB |
| W4MIXG128 | 19.06 / 17.60 | 13433.46 | 3578.96 | 2708.7 MB |

Note: this FP16 row supersedes the original `615.04 / 127.86` entry. A later
same-session repro attempt could not reproduce the `~128 tok/s` FP16 decode
number: current explicit-lmhead FP16 measured `625.36 / 101.74` over 5 runs,
while old pre-dynamic `cee9357` measured `624.91 / 101.14` with its own package.
Both used `generated_tokens=64` and `decode_tokens=63`. Treat the original
`~128 tok/s` FP16 decode sample as invalid. The most likely source is a
transcription/package-label mix-up with Attempt 6's W8PC row:
`Qwen3.5-0.8B W8PC | default q8dot repack | 376.1 / 126.8`, whose
`decode_ms=496.77` is very close to the invalid FP16 row's `492.72`.

W4MIXG128 is about 2.0x faster than pure W4G128 because 72 sensitive tensors
use the optimized W8 path, but it is still far slower than W8G128/FP16. This is
expected: all remaining W4 tensors still use scalar nibble unpack + FP32
accumulate, with no W4 NEON/q8dot/repacked kernel.

llama.cpp CPU reference on the same machine, using `../llama.cpp` commit
`5c7c22c3e` / build 9803 and `llama-bench -ngl 0 -p 256 -n 64 -t 4 -r 5`.
Values are `llama-bench` reported summaries:

| GGUF | Size | pp/tg |
|---|---:|---:|
| Qwen3.5-0.8B F16 | 1475.05 MiB | 735.34 / 99.27 |
| Qwen3.5-0.8B Q8_0 | 784.52 MiB | 819.99 / 171.91 |
| Qwen3.5-0.8B Q4_0 | 478.76 MiB | 809.47 / 189.80 |
| Qwen3.5-0.8B Q4_K_M | 506.34 MiB | 639.04 / 173.60 |

`Q4_K_M` was generated from the local F16 GGUF with
`llama-quantize Q4_K_M`. The quantizer promoted q6_K token embeddings, selected
`attn_qkv` and `ffn_down` tensors, and late-layer `attn_v`/`ffn_down` tensors.
Compared with llama.cpp Q4_K_M, mollm W4MIXG128 is 33.5x slower on prefill and
9.9x slower on decode. The next W4 performance step is therefore kernel/layout
work, not further benchmark analysis of the scalar fallback.

### Interpretation

- Mixed W4 recovers most of the pure W4G128 quality drift without going all the
  way to W8G128 size.
- The remaining delta is likely a mix of the simple symmetric W4 format and
  still-W4 gate/up/q/k/projection tensors. Smaller groups (`w4mixg64`,
  `w4mixg32`) and targeted ablations should be tested before treating this as
  final.
- Performance remains a separate task: W8-promoted tensors use the optimized W8
  path, but W4 tensors still use scalar unpack/dequant fallback.

## Attempt 12: Explicit lm_head Weight Boundary

### Goal

Remove tied-word-embedding handling from runtime. The package format should own
the model's weight identity: `embed_tokens.weights` is an embedding lookup
table, while `lm_head.weights` is an independent linear weight. Runtime should
not create a hidden FP16 packed lmhead copy from embeddings.

### Implementation

- Qwen3.5 and Youtu converters now always export explicit `lm_head.weights`.
  If the source model has an explicit lmhead tensor, that tensor is used;
  otherwise tied embeddings are copied into a separate lmhead file at export.
- `embed_tokens.weights` remains FP16 row-major for lookup.
- `lm_head.weights` is quantized by the normal linear policy:
  - `none`: FP16
  - `w8pc` / `w8gN`: W8
  - `w4gN`: W4
  - Qwen3.5 `w4mixgN`: W8
- Engine load now requires explicit `lm_head.weights` and records a
  `lm_head_weight_` pointer from the graph CONSTANT. `run_lmhead()` and
  `run_lmhead_raw()` call `kernel_matmul_fp32()` on that tensor directly.
- The old runtime-owned `embed_packed_` lmhead copy was removed.

Old `.mollm` packages without `lm_head.weights` must be regenerated.

### Validation

```bash
python3 -m py_compile models/qwen35.py models/mla.py models/converter.py models/transpile.py tests/test_qwen35_quant_policy.py
python3 tests/test_qwen35_quant_policy.py
ninja -C build mollm_bench test_quantized_e2e test_e2e test_engine
python3 models/converter.py ../Qwen3.5-0.8B qwen35_0.8b.mollm 24 256 none
python3 models/converter.py ../Qwen3.5-0.8B qwen35_0.8b_w8pc.mollm 24 256 w8pc
python3 models/converter.py ../Qwen3.5-0.8B qwen35_0.8b_w8g128.mollm 24 256 w8g128
python3 models/converter.py ../Qwen3.5-0.8B qwen35_0.8b_w4g128.mollm 24 256 w4g128
python3 models/converter.py ../Qwen3.5-0.8B qwen35_0.8b_w4mixg128.mollm 24 256 w4mixg128
python3 models/converter.py ../Youtu-LLM-2B youtu-llm-2b.mollm 32 256 none
./build/test_e2e qwen35_0.8b.mollm youtu-llm-2b.mollm
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm 256
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8g128.mollm 256
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w4g128.mollm 32
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w4g128.mollm 256
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w4mixg128.mollm 32
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w4mixg128.mollm 256
ctest --test-dir build --output-on-failure
```

All passed. `test_e2e` still reports Qwen3.5 prefill/decode PPL matching HF and
Youtu-LLM-2B PPL `10.2569` vs HF `10.20`.

### Package / Quality Results

Current Qwen3.5-0.8B packages now contain 321 weights, including explicit
`lm_head.weights`.

| Package | lmhead precision | Size | 32-token CE/PPL | 256-token CE/PPL | CE delta at 256 |
|---|---|---:|---:|---:|---:|
| FP16 | FP16 | 2027.5 MB | 3.0911 / 22.00 | 2.1388 / 8.49 | - |
| W8PC | INT8, group=1024 | 1278.1 MB | 3.1098 / 22.42 | 2.1475 / 8.56 | +0.0086 |
| W8G128 | INT8, group=128 | 1299.1 MB | 3.1107 / 22.44 | 2.1416 / 8.51 | +0.0028 |
| W4G128 | INT4, group=128 | 923.1 MB | 3.4903 / 32.80 | 2.4221 / 11.27 | +0.2832 |
| W4MIXG128 | INT8, group=128 | 1177.6 MB | 3.1864 / 24.20 | 2.1705 / 8.76 | +0.0317 |

Quantized tensor counts:

| Package | W4 tensors | W8 tensors |
|---|---:|---:|
| W8PC | 0 | 187 |
| W8G128 | 0 | 187 |
| W4G128 | 187 | 0 |
| W4MIXG128 | 114 | 73 |

Pure W4 quality regresses when lmhead is truly W4. W4MIX keeps quality near the
previous result because lmhead is promoted to W8.

### Performance Check

Strict protocol: Qwen3.5-0.8B, default Release build (`MOLLM_ARM_I8MM=OFF`),
pp256 + tg64, warmup=3, 4 threads, 5-run median.

| Package | pp/tg | prefill ms | decode ms | peak RSS |
|---|---:|---:|---:|---:|
| FP16 | 625.36 / 101.74 | 409.36 | 619.21 | 3994.9 MB |
| W8PC | 396.78 / 157.34 | 645.19 | 400.40 | 2561.9 MB |
| W8G128 | 392.36 / 148.78 | 652.46 | 423.45 | 2581.5 MB |

The FP16 row was updated after a same-session repro check. The earlier
`613.34 / 127.80` entry could not be reproduced under the strict pp256/tg64
protocol: 8 additional current runs ranged from `97.61` to `103.25` decode
tok/s, and old `cee9357` with its own package measured the same `~101 tok/s`
decode level. Current FP16 profile confirms `decode_lmhead` is
`fp16_gemv_interleaved_fp16acc`, not W8. The closest matching valid historical
number is Attempt 6's pre-explicit-lmhead W8PC run (`376.1 / 126.8`,
`decode_ms=496.77`), so the bad FP16 entry was likely copied or labeled from
that W8PC measurement.

The W8 rows were also remeasured with the same strict protocol. The earlier
`399.18 / 214.24` and `382.61 / 205.60` rows were not reproducible in this
same-session rerun; current W8 decode is still faster than FP16 decode, but
0.8B no longer beats llama.cpp Q8_0.

Profile check with `MOLLM_MATMUL_SHAPE_PROFILE=1` confirms both lmhead phases
take the W8 q8dot path in W8PC:

| Phase | Path | M/N/K | group | avg ms |
|---|---|---|---:|---:|
| prefill_lmhead | q8dot_gemv_repack | 1 / 248320 / 1024 | 1024 | 1.711 |
| decode_lmhead | q8dot_gemv_repack | 1 / 248320 / 1024 | 1024 | 1.104 |

### Interpretation

- The runtime boundary is cleaner: lmhead is a regular CONSTANT and regular
  matmul weight.
- W8 decode improves substantially because the large vocabulary projection no
  longer uses FP16 packed lmhead in W8 packages.
- Package size increases because tied embedding is materialized explicitly as
  lmhead. Runtime RSS still improves for W8 because the old FP16 lmhead packed
  copy is gone.
- Pure W4 should not be treated as a quality target with W4 lmhead. W4MIX or a
  more sophisticated W4 format needs high-precision handling for lmhead/output
  embeddings.

## Attempt 13: Strict Current Performance Audit

### Goal

Re-run the performance comparison after discovering that an earlier
`~128 tok/s` FP16 0.8B decode row was invalid and that old 2B/4B packages
without explicit `lm_head.weights` no longer load on current runtime.

### Protocol

- `mollm_bench --prompt-tokens 256 --max-new-tokens 64 --warmup 3 --threads 4`
- `llama-bench -ngl 0 -p 256 -n 64 -t 4 -r 1`
- 5 independent processes per row, median reported.
- Reconverted current-format packages for Youtu-LLM-2B W8PC and Qwen3.5-4B
  FP16/W8PC before benchmarking.

### Current Packages

| Package | Disk size | Notes |
|---|---:|---|
| `qwen35_0.8b.mollm` | 1.9G | explicit FP16 lmhead |
| `qwen35_0.8b_w8pc.mollm` | 1.2G | explicit W8 lmhead |
| `qwen35_0.8b_w8g128.mollm` | 1.2G | explicit W8 lmhead |
| `youtu-llm-2b.mollm` | 4.2G | explicit FP16 lmhead |
| `youtu-llm-2b_w8pc.mollm` | 2.3G | reconverted current W8PC |
| `qwen35_4b_current_reconv.mollm` | 9.0G | reconverted current FP16 |
| `qwen35_4b_w8pc.mollm` | 5.1G | reconverted current W8PC |

### FP16 / F16 Comparison

| Model | mollm pp/tg | llama.cpp pp/tg | mollm peak RSS |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 625.36 / 101.74 | 675.69 / 99.24 | 3994.9 MB |
| Youtu-LLM-2B | 241.59 / 51.27 | 266.66 / 46.89 | 9490.7 MB |
| Qwen3.5-4B | 110.66 / 24.44 | 143.22 / 22.02 | 19085.0 MB |

### W8PC / Q8_0 Comparison

| Model | mollm W8PC pp/tg | llama.cpp Q8_0 pp/tg | mollm peak RSS |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 396.78 / 157.34 | 813.58 / 171.10 | 2561.9 MB |
| Youtu-LLM-2B | 139.50 / 89.64 | 271.58 / 86.47 | 5950.5 MB |
| Qwen3.5-4B | 64.02 / 43.66 | 138.22 / 40.71 | 10972.3 MB |

### Interpretation

- FP16 mollm decode is still slightly faster than llama.cpp F16 on 2B/4B, but
  FP16 prefill remains slower.
- W8PC decode is competitive with llama.cpp Q8_0 on 2B/4B, but W8PC prefill is
  still about 2x slower. That is the main current quantized performance gap.
- 0.8B remains noisy and currently trails llama.cpp Q8_0 on decode. Use 2B/4B
  as the primary performance signal.
- Explicit tied lmhead materially increases FP16 package/RSS. For example,
  Qwen3.5-4B FP16 now peaks around 19.1 GB RSS; W8PC peaks around 11.0 GB.

## Attempt 14: W4 q4dot Kernel Baseline

### Goal

Replace the W4 scalar correctness path with a real ARM dot-product kernel while
keeping the current symmetric signed-int4 per-group format. Correctness and
fallback coverage stay first; W4 load-time repack is left for a later pass.

### Implementation

- Added W4 q8dot GEMV/GEMM dispatch for `K % 32 == 0` and
  `group_size % 32 == 0`.
- FP32 activations are quantized into temporary Q8 blocks, reusing the W8
  block size of 32 K elements.
- The W4 kernel loads 16 packed bytes for 32 weights, splits low/high nibbles,
  sign-extends them to int8, unzips the Q8 activation into even/odd K lanes,
  and computes two `vdotq_s32` dot products per 32-K block.
- The int32 dot result is scaled by `activation_scale * weight_scale` and
  accumulated into FP32 output.
- Odd K or non-32-aligned groups still fall back to scalar nibble unpack.
- New env flags:
  - `MOLLM_W4_NO_Q8_DOT=1`: force scalar W4 fallback
  - `MOLLM_W4_NO_Q8_DOT_GEMM=1`: keep W4 GEMV q4dot but force GEMM scalar
- `bench_matmul` now supports `--int4`.
- `test_matmul` now covers both scalar INT4 odd-K fallback and q4dot GEMV/GEMM.

This path uses ARM dot-product (`vdotq_s32`), not i8mm. i8mm consumes int8
matrix tiles; using it for W4 would require expanding or repacking W4 into
int8-shaped tiles, which doubles weight bandwidth unless a more specialized
layout is used.

### Validation

```bash
cmake --build build --target test_matmul bench_matmul mollm_bench test_quantized_e2e -j
./build/test_matmul
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w4g128.mollm 32
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w4g128.mollm 256
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w4mixg128.mollm 32
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w4mixg128.mollm 256
```

All passed.

Reference-token CE/PPL after q4dot:

| Package | 32-token CE/PPL | 256-token CE/PPL | CE delta at 256 |
|---|---:|---:|---:|
| W4G128 | 3.4881 / 32.73 | 2.4239 / 11.29 | +0.2851 |
| W4MIXG128 | 3.1804 / 24.06 | 2.1684 / 8.74 | +0.0296 |

The small drift versus scalar W4 comes from Q8 activation dot-product
quantization and is within the expected W8 q8dot-level noise.

### Kernel Microbench

Default q4dot versus forced scalar (`MOLLM_W4_NO_Q8_DOT=1`), 4 threads:

| Shape | q4dot avg | scalar avg | speedup |
|---|---:|---:|---:|
| M=1 K=1024 N=3584 | 0.02 ms | 0.91 ms | ~45x |
| M=1 K=1024 N=248320 | 1.25 ms | 53.88 ms | ~43x |
| M=64 K=1024 N=3584 | 1.05 ms | 50.03 ms | ~48x |
| M=64 K=1024 N=6144 | 1.83 ms | 85.37 ms | ~47x |
| M=64 K=3584 N=1024 | 0.99 ms | 50.90 ms | ~51x |

The old scalar path was around 8-9 GMAC/s. q4dot profile rows now land around
130-190 GMAC/s in end-to-end runs; standalone microbench reports roughly
150-235 GMAC/s depending on shape.

### End-to-End Profile

Short profile (`pp64 + tg8`, warmup=1, 4 threads):

| Package | Before scalar pp/tg | q4dot pp/tg | MATMUL before | MATMUL q4dot |
|---|---:|---:|---:|---:|
| W4G128 | 9.35 / 6.05 | 304.31 / 126.85 | prefill 6758 ms, decode 761 ms | prefill 170 ms, decode 25 ms |
| W4MIXG128 | 18.81 / 17.17 | 315.94 / 128.86 | prefill 3362 ms, decode 378 ms | prefill 164 ms, decode 24 ms |

Shape profile examples after q4dot:

| Phase | Shape | Path | Avg ms | GMAC/s |
|---|---|---|---:|---:|
| prefill_graph | M=64 N=3584 K=1024 | q4dot_gemm | 1.235 | 190.16 |
| prefill_graph | M=64 N=6144 K=1024 | q4dot_gemm | 2.078 | 193.77 |
| decode_lmhead | M=1 N=248320 K=1024 | q4dot_gemv | 1.397 | 181.97 |
| decode_graph | M=1 N=3584 K=1024 | q4dot_gemv | 0.025 | 149.65 |

Strict protocol (`pp256 + tg64`, warmup=3, 4 threads, 5 independent process
median):

| Package | pp/tg median |
|---|---:|
| W4G128 | 350.94 / 148.53 |
| W4MIXG128 | 331.37 / 137.19 |

### Interpretation

- W4 is no longer a scalar-only correctness baseline. It is now close to W8
  decode speed on 0.8B, though still below llama.cpp Q4_0/Q4_K_M decode.
- W4G128 is smaller and now fast, but pure W4 quality remains poor because
  `lm_head.weights` is also W4.
- W4MIXG128 keeps quality near W8 by promoting 73 tensors to W8, including
  lmhead. It is not faster than pure W4 or W8 because it mixes q4dot and W8
  q8dot and still pays on-the-fly W4 unpack.
- Next W4 performance work should focus on a W4-specific load-time dot layout,
  probably `[N/8, K/32, 8, 16 packed bytes]`, so the kernel avoids repeated
  row-major nibble addressing and can improve cache behavior without expanding
  all W4 weights to int8.

## Attempt 15: W4 q4dot Load-Time Repack

### Goal

Avoid repeated row-major W4 address jumps in q4dot kernels while preserving
4-bit storage. This is a packed dot layout, not an int8 expansion.

### Implementation

- Added `Tensor::q4_repack_data`.
- Added `pack_b_q4dot_int4_full()`:

```text
source: [N, ceil(K/2)]
target: [N/8, K/32, 8, 16 packed bytes]
```

- Engine now builds q4 repack at load time for INT4 weights when:
  - `K % 32 == 0`
  - `group_size % 32 == 0`
  - ARM dot-product is available
  - `MOLLM_W4_NO_Q8_DOT`, `MOLLM_W4_NO_Q4_REPACK`, and
    `MOLLM_W4_NO_Q8_REPACK` are not set
- W4 q4dot dispatch now reports `q4dot_gemv_repack` /
  `q4dot_gemm_repack` when the repack pointer exists.
- Original W4 data stays available for scalar fallback and for disabling
  repack via env. Because packages are mmaped, this increases RSS after
  packing; a future package-format change could store the q4dot layout directly
  if we want to avoid the second resident copy.
- `bench_matmul --int4` now creates q4 repack by default.
- `test_matmul` covers both row-major q4dot and repacked q4dot for GEMV/GEMM.

### Validation

```bash
cmake --build build --target test_matmul bench_matmul mollm_bench test_quantized_e2e -j
./build/test_matmul
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w4g128.mollm 256
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w4mixg128.mollm 256
```

All passed. 256-token CE/PPL is unchanged from Attempt 14:

| Package | 256-token CE/PPL | CE delta at 256 |
|---|---:|---:|
| W4G128 | 2.4239 / 11.29 | +0.2851 |
| W4MIXG128 | 2.1684 / 8.74 | +0.0296 |

### Microbench

Default q4 repack versus on-the-fly row-major q4dot
(`MOLLM_W4_NO_Q4_REPACK=1`) and scalar (`MOLLM_W4_NO_Q8_DOT=1`), 4 threads:

| Shape | q4 repack | row-major q4dot | scalar |
|---|---:|---:|---:|
| M=1 K=1024 N=248320 | 1.20 ms | 1.36 ms | 53.70 ms |
| M=64 K=1024 N=3584 | 1.02 ms | 1.03 ms | 49.94 ms |

The repack helps large GEMV/lmhead a bit. Dominant prefill GEMM is almost
unchanged in standalone microbench, but end-to-end profile improves because
there are many small/medium GEMV-like decode shapes and repeated graph calls.

### End-to-End A/B

Strict protocol: Qwen3.5-0.8B, pp256 + tg64, warmup=3, 4 threads, 5 independent
process median.

| Package | Mode | pp/tg median | load ms median | RSS median |
|---|---|---:|---:|---:|
| W4G128 | q4 repack | 386.71 / 153.52 | 275.7 | 1862.7 MB |
| W4G128 | no q4 repack | 329.97 / 137.99 | 196.8 | 1500.8 MB |
| W4MIXG128 | q4 repack | 341.15 / 133.81 | 342.1 | 2349.0 MB |
| W4MIXG128 | no q4 repack | 319.50 / 132.58 | 319.8 | 2233.0 MB |

Short shape profile (`pp64 + tg8`) confirms the runtime uses repacked paths:

| Phase | Shape | Path | Avg ms | GMAC/s |
|---|---|---|---:|---:|
| prefill_graph | M=64 N=3584 K=1024 | q4dot_gemm_repack | 1.137 | 206.65 |
| prefill_graph | M=64 N=6144 K=1024 | q4dot_gemm_repack | 1.920 | 209.72 |
| decode_lmhead | M=1 N=248320 K=1024 | q4dot_gemv_repack | 1.372 | 185.38 |
| decode_graph | M=1 N=3584 K=1024 | q4dot_gemv_repack | 0.023 | 162.43 |

### Interpretation

- q4 repack is worth keeping by default for pure W4: +17% prefill and +11%
  decode in same-session strict A/B.
- W4MIX benefits much less because 73 tensors are W8 and only the remaining W4
  tensors use q4 repack.
- The price is resident memory and load time. W4G128 RSS rises by about
  360 MB, roughly the resident cost of the q4dot-layout copy plus touched mmap
  pages. W4MIX rises by about 116 MB because fewer tensors are W4.
- To avoid the second resident W4 copy, the next structural step would be a
  package-level packed weight format, so `.mollm` stores q4dot layout directly
  (implemented in Attempt 16).

## Attempt 16: W4 q4dot Package Layout

### Goal

Remove the runtime-owned W4 q4dot copy introduced in Attempt 15 while keeping
the same fast q4dot kernels. Runtime should not store both original row-major
W4 bytes and q4dot-layout W4 bytes for current packages.

### Implementation

- Reused the existing `.weights` header `flags` field:
  - `FLAG_INT4_Q4DOT = 1 << 0`
  - logical shape remains `[N, K]`
  - physical data is `[ceil(N/8), K/32, 8, 16 packed bytes]`
- Added converter-side `pack_weight_w4_q4dot()` and enabled it for W4 tensors
  when both K and group size are 32-aligned.
- Runtime metadata validation now accepts two INT4 physical layouts:
  - old row-major W4: `N * ceil(K/2)` bytes, `flags=0`
  - direct q4dot W4: `ceil(N/8) * (K/32) * 8 * 16` bytes,
    `FLAG_INT4_Q4DOT`
- When `FLAG_INT4_Q4DOT` is present, runtime sets `Tensor::is_q4_repacked` and
  points `Tensor::q4_repack_data` directly at mmap package data. It does not
  build a `packed_weights_` copy for that tensor.
- Old row-major W4 packages remain compatible and still use the Attempt 15
  load-time repack path unless disabled by `MOLLM_W4_NO_Q4_REPACK=1`.
- Direct q4dot packages require ARM DOTPROD. They cannot be forced into scalar
  fallback, because the package no longer stores row-major W4 bytes.
- `test_matmul` now covers direct q4dot layout for GEMV and GEMM, including
  N padding (`N=9`). `test_mmap_file` now checks header flag preservation.

### Validation

```bash
cmake --build build --target test_matmul mollm_bench test_quantized_e2e test_mmap_file -j
python3 -m py_compile models/transpile.py models/qwen35.py models/mla.py models/converter.py
python3 models/converter.py ../Qwen3.5-0.8B qwen35_0.8b_w4g128_q4pkg.mollm 24 256 w4g128
python3 models/converter.py ../Qwen3.5-0.8B qwen35_0.8b_w4mixg128_q4pkg.mollm 24 256 w4mixg128
./build/test_matmul
./build/test_mmap_file
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w4g128_q4pkg.mollm 256
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w4mixg128_q4pkg.mollm 256
```

All passed. 256-token CE/PPL is unchanged from the old row-major packages:

| Package | 256-token CE/PPL | CE delta at 256 |
|---|---:|---:|
| W4G128 direct q4pkg | 2.4239 / 11.29 | +0.2851 |
| W4MIXG128 direct q4pkg | 2.1684 / 8.74 | +0.0296 |

Short profile for W4G128 direct q4pkg (`pp64 + tg8`) confirms direct package
data still dispatches as q4dot repack paths:

| Phase | Shape | Path | Avg ms | GMAC/s |
|---|---|---|---:|---:|
| prefill_graph | M=64 N=3584 K=1024 | q4dot_gemm_repack | 0.938 | 250.41 |
| prefill_graph | M=64 N=6144 K=1024 | q4dot_gemm_repack | 1.628 | 247.31 |
| decode_lmhead | M=1 N=248320 K=1024 | q4dot_gemv_repack | 1.196 | 212.58 |
| decode_graph | M=1 N=3584 K=1024 | q4dot_gemv_repack | 0.019 | 191.55 |

### End-to-End A/B

Strict protocol: Qwen3.5-0.8B, pp256 + tg64, warmup=3, 4 threads, 5 independent
process median.

| Package | Layout / mode | pp/tg median | load ms median | RSS median |
|---|---|---:|---:|---:|
| W4G128 | old row-major + runtime q4 repack | 386.71 / 153.52 | 275.7 | 1862.7 MB |
| W4G128 | old row-major, no q4 repack | 329.97 / 137.99 | 196.8 | 1500.8 MB |
| W4G128 | direct q4pkg | 385.88 / 154.88 | 194.2 | 1500.7 MB |
| W4MIXG128 | old row-major + runtime q4 repack | 341.15 / 133.81 | 342.1 | 2349.0 MB |
| W4MIXG128 | old row-major, no q4 repack | 319.50 / 132.58 | 319.8 | 2233.0 MB |
| W4MIXG128 | direct q4pkg | 370.07 / 143.38 | 312.3 | 2232.8 MB |

Youtu-LLM-2B follow-up on 2026-07-03, same protocol. This was run in the same
session as the 2B direct q4pkg package conversion; absolute prefill throughput
was lower than the 2026-07-02 fan-max baseline, so use these rows as same-session
FP16/W8/W4 comparison rather than as a replacement for the global baseline.

| Package | Layout / mode | pp/tg median | load ms median | RSS median | 256-token CE/PPL |
|---|---|---:|---:|---:|---:|
| Youtu-LLM-2B | FP16 | 212.14 / 50.97 | 951.3 | 9490.7 MB | 3.1139 / 22.51 |
| Youtu-LLM-2B | W8PC | 114.73 / 88.78 | 507.5 | 5948.3 MB | 3.1094 / 22.41 |
| Youtu-LLM-2B | W4G128 direct q4pkg | 108.31 / 70.20 | 110.5 | 2893.5 MB | 3.1578 / 23.52 |

### Interpretation

- The user concern was correct: runtime-owned q4 repack was the wrong long-term
  storage model. It preserved speed but made current packages keep two resident
  W4 layouts.
- Package-level q4dot keeps W4G128 speed essentially unchanged and removes the
  ~360 MB RSS penalty from runtime repack.
- W4MIX direct q4pkg also drops the W4 duplicate copy and measured faster in
  this session. Treat the exact speedup cautiously because 0.8B is noisy, but
  the RSS/load-time improvement is structural.
- On Youtu-LLM-2B, pure W4G128 quality is acceptable on the current 256-token
  reference sample (`CE delta +0.0440`; W8PC rerun is `3.1094 / 22.41`,
  delta `-0.0045`), but decode is slower than W8PC
  (`70.20` vs `88.78` tok/s). The next W4 work should look at 2B decode shapes,
  especially q4dot GEMV and activation quantization overhead.
- This makes q4dot layout a package-format concern. Runtime still keeps the old
  load-time repack path only for backwards compatibility and diagnostics.

## Attempt 17: W4 q4dot GEMV Scale/QA Prep

### Goal

Improve W4 decode speed on Youtu-LLM-2B without changing package layout or
prefill GEMM behavior. Attempt 16 showed 2B W4G128 direct q4pkg quality and RSS
were good, but decode still trailed W8PC (`70.20` vs `88.78` tok/s in that
same-session run).

### Implementation

- Added a W4 GEMV-specific activation quantizer that writes Q8 activation blocks
  directly into even/odd 16-lane streams matching the q4dot nibble layout.
  This avoids first building row-major Q8 activation data and then doing
  `vuzp` inside every qblock.
- Hoisted W4 GEMV weight scale loading by scale mode:
  - per-channel: load the 8 output-channel scales once per output tile
  - per-group: for `w4g128`, load scales once per 4 qblocks
  - true per-block32: keep the per-qblock scale load
- Kept the direct q4pkg mmap path unchanged; no runtime W4 copy is introduced.
- Tried the same scale-loop restructuring in W4 GEMM, but reverted it after
  short profiling showed a prefill regression on the dominant `M=64` shapes.
  The committed kernel change is intentionally GEMV-only.

### Validation

```bash
cmake --build build --target bench_matmul test_matmul mollm_bench test_quantized_e2e -j
./build/test_matmul
./build/test_quantized_e2e youtu-llm-2b.mollm youtu-llm-2b_w4g128_q4pkg.mollm 256
./build/test_mmap_file
```

All passed. Youtu-LLM-2B W4G128 direct q4pkg 256-token CE/PPL stayed unchanged:
`3.1578 / 23.52`, `CE delta +0.0440` versus FP16.

### Microbench

W4G128 direct q4dot, 4 threads:

| Shape | Avg ms | p50 ms | GF/s |
|---|---:|---:|---:|
| M=1 K=2048 N=12288 | 0.09 | 0.09 | 541.3 |
| M=1 K=6144 N=2048 | 0.05 | 0.05 | 484.3 |
| M=1 K=2048 N=128256 | 0.96 | 0.97 | 544.9 |

Short shape profile on Youtu-LLM-2B W4G128 direct q4pkg (`pp64 + tg8`,
warmup=1) after the GEMV-only change:

| Phase | Shape | Avg ms | GMAC/s |
|---|---|---:|---:|
| decode_graph | M=1 N=12288 K=2048 | 0.111 | 227.65 |
| decode_graph | M=1 N=2048 K=6144 | 0.061 | 205.09 |
| decode_lmhead | M=1 N=128256 K=2048 | 1.098 | 239.23 |

The same shapes before this GEMV change were about `0.123 ms`, `0.064 ms`,
and `1.212 ms` respectively in the earlier short profile. The exact GMAC/s
varies with machine state, but the decode matmul direction is consistent.

### End-to-End

Strict protocol: pp256 + tg64, warmup=3, 4 threads, 5 independent process
median. These were run in the same machine session after the GEMV change.

| Youtu-LLM-2B package | pp/tg median | prefill ms | decode ms | load ms | peak RSS |
|---|---:|---:|---:|---:|---:|
| W4G128 direct q4pkg | 118.39 / 89.61 | 2162.34 | 703.05 | 114.8 | 2893.6 MB |
| W8PC | 117.25 / 87.84 | 2183.38 | 717.18 | 510.6 | 5948.9 MB |

### llama.cpp Comparison

Local llama.cpp build: `../llama.cpp` `5c7c22c3e` / build 9803,
`../llama.cpp/build-cpu/bin/llama-bench -ngl 0 -p 256 -n 64 -t 4 -r 5 -o md`.
Youtu Q4_0 and Q4_K_M GGUFs were generated locally from the F16 GGUF with
`llama-quantize`.

| Runtime / package | Size | pp256 | tg64 |
|---|---:|---:|---:|
| mollm W4G128 direct q4pkg | 1577.9 MB | 118.39 | 89.61 |
| llama.cpp F16 | 3.65 GiB | 253.29 ± 10.31 | 45.04 ± 2.48 |
| llama.cpp Q8_0 | 1.94 GiB | 259.58 ± 1.35 | 85.09 ± 1.69 |
| llama.cpp Q4_0 | 1.09 GiB | 255.28 ± 2.17 | 90.35 ± 5.35 |
| llama.cpp Q4_K_M | 1.14 GiB | 187.96 ± 1.30 | 82.53 ± 0.63 |

Notes:

- llama.cpp reports BLAS backend in this build, which explains the large
  pp256 advantage. mollm W4 prefill is still W4 GEMM-bound.
- mollm W4G128 decode is now effectively in the llama.cpp Q4/Q8 range on 2B.
- llama.cpp Q4_0 is not pure-all-tensors-Q4: token embedding is q6_K by
  default. Q4_K_M also fell back 32 `attn_k_b` tensors to q5_0 because their
  ncols are not compatible with q4_K.

### Interpretation

- The original 2B W4 decode bottleneck was not package layout anymore; it was
  GEMV-side per-qblock overhead: repeated activation lane unzip and repeated
  per-group scale loading.
- W4G128 direct q4pkg now reaches W8PC-class decode on 2B in same-session
  strict benchmark while keeping roughly half the RSS and much lower load time.
- Prefill remains governed by the W4 GEMM kernel. The attempted GEMM loop
  rewrite regressed prefill and was not kept. The next W4 kernel work should
  target GEMM separately, likely with a different tile/layout change rather
  than mechanically applying the GEMV scale loop.

## Attempt 18: C++ Tensor Quantizer Helper

### Goal

Make W4/W8 package conversion practical for 4B-class models. The Python
converter was still doing per-element row/group quantization and q4dot packing
in nested Python/NumPy loops. That was acceptable for 0.8B and marginal for 2B,
but 4B `lm_head.weights` alone is `248320 x 2560`, so pure Python packing made
4B W4 conversion effectively unusable.

This is a converter-throughput change only. It should not change the `.weights`
format, quantization math, runtime loader, or kernel behavior.

### Implementation

- Added `tools/quantize_weight.cpp` and CMake target/binary `mollm-quantize`.
  The old `mollm_quantize_weight` build target remains as a compatibility alias.
- CLI contract:

```bash
mollm-quantize <input.raw> <output.weights> <f16|f32> <N> <K> <w8|w4> <group_size> [threads]
```

- The helper reads a contiguous row-major FP16/FP32 tensor and writes the same
  88-byte `XMAP` `.weights` header used by Python.
- W8 output matches the current row-major INT8 + flattened scales metadata.
- W4 output uses the direct q4dot package layout when `K` and `group_size` are
  32-aligned:
  - logical shape remains `[N, K]`
  - physical data is `[ceil(N/8), K/32, 8, 16 packed bytes]`
  - `FLAG_INT4_Q4DOT` is set in the header
- Odd/non-q4dot W4 shapes fall back to row-major nibble packing.
- The Python converter now auto-detects the helper in `build/`. W4 requires the
  helper and fails fast when it is missing. W8 can still fall back to the
  Python implementation because it is simple enough for small/debug packages.
  `MOLLM_QUANT_HELPER` overrides the helper path; `MOLLM_DISABLE_CPP_QUANT=1`
  only leaves Python fallback available for W8.

This intentionally avoids adopting GGUF/Q4_K formats in the converter. The
helper writes mollm's current package layout directly, while preserving room for
future per-channel/per-group policy changes.

### Validation

Byte-for-byte comparison against the Python reference passed for:

- W8 FP16 group quantization
- W8 FP32 group quantization
- W4 direct q4dot group quantization
- W4 odd-K row-major fallback

Package/e2e validation:

```bash
cmake --build build --target mollm-quantize test_mmap_file test_matmul test_quantized_e2e -j
python3 -m py_compile models/transpile.py models/qwen35.py models/mla.py models/converter.py tests/test_qwen35_quant_policy.py
python3 tests/test_qwen35_quant_policy.py
./build/test_matmul
./build/test_mmap_file
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w4g128_cppq.mollm 256
./build/test_quantized_e2e youtu-llm-2b.mollm youtu-llm-2b_w4g128_cppq.mollm 256
./build/test_quantized_e2e qwen35_4b_current_reconv.mollm qwen35_4b_w4g128_cppq.mollm 256
```

All passed.

### Conversion Time

`/usr/bin/time -p` around full package conversion, using the C++ helper:

| Model/package | Quant | real time | package size |
|---|---|---:|---:|
| Qwen3.5-0.8B | W4G128 | 3.06s | 923.1 MB |
| Youtu-LLM-2B | W4G128 | 6.34s | 1577.9 MB |
| Qwen3.5-4B | W4G128 | 20.18s | 3522.2 MB |

The 4B conversion is the important result: the previous Python path had to be
killed after a long run before producing a package, while the C++ helper reaches
and completes the large `lm_head.weights` quantization.

### PPL Audit

The C++ helper preserves the same quality results as the Python q4pkg path.

| Model | FP16 CE/PPL | W4G128 CE/PPL | CE delta |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 2.1388 / 8.49 | 2.4239 / 11.29 | +0.2851 |
| Youtu-LLM-2B | 3.1139 / 22.51 | 3.1578 / 23.52 | +0.0440 |
| Qwen3.5-4B | 1.8771 / 6.53 | 1.9568 / 7.08 | +0.0797 |

### Interpretation

- The converter bottleneck was Python per-element quantize/pack work, not the
  `.mollm` package format itself.
- Runtime memory behavior is unchanged: current W4 packages still mmap direct
  q4dot data and do not allocate a second W4 repack copy.
- This makes W4 experimentation on 4B practical enough to continue kernel and
  quantization-policy work. The next W4 performance target remains prefill
  GEMM; this helper only fixes package generation latency.

## Attempt 19: Youtu W4MIX Policy

### Goal

Stop treating Youtu-LLM-2B as a pure-W4-only path. Pure W4G128 looked much
better on Youtu than on Qwen3.5-0.8B (`CE delta +0.0440` on the 256-token
sample), but it still quantized `lm_head` and all MLA/FFN projections to W4.

### Implementation

Added `w4mixgN` support to `models/mla.py`. The first Youtu mixed policy
promotes these tensors to W8G`N`:

- `lm_head.weight`
- `self_attn.kv_b_proj.weight`
- `self_attn.o_proj.weight`
- `mlp.down_proj.weight`

It leaves these in W4G`N`:

- `self_attn.q_a_proj.weight`
- `self_attn.q_b_proj.weight`
- `self_attn.kv_a_proj_with_mqa.weight`
- merged `mlp.gate_up_proj.weight`

This keeps Youtu mixed lighter than Qwen3.5 mixed. The intent is to protect the
output/expansion side first, while avoiding a mostly-W8 package.

### Coverage

Youtu-LLM-2B `w4mixg128` export:

| Metric | Value |
|---|---:|
| W8 tensors | 97 |
| W4 tensors | 128 |
| W8 quantized params | 866,648,064 (44.2%) |
| W4 quantized params | 1,094,713,344 (55.8%) |
| Package size | 2011.2 MB |
| Conversion time | 9.27s |

For comparison, Youtu pure W4G128 package is 1577.9 MB and W8PC package is
about 2.3 GB.

### Validation

```bash
python3 -m py_compile models/transpile.py models/qwen35.py models/mla.py models/converter.py tests/test_qwen35_quant_policy.py tests/test_youtu_quant_policy.py
python3 tests/test_qwen35_quant_policy.py
python3 tests/test_youtu_quant_policy.py
cmake --build build --target mollm-quantize test_quantized_e2e test_matmul -j
python3 models/converter.py ../Youtu-LLM-2B youtu-llm-2b_w4mixg128_cppq.mollm 32 256 w4mixg128
./build/test_quantized_e2e youtu-llm-2b.mollm youtu-llm-2b_w4mixg128_cppq.mollm 256
```

All passed.

### PPL

256-token reference sample:

| Package | CE/PPL | CE delta vs FP16 |
|---|---:|---:|
| FP16 | 3.1139 / 22.51 | - |
| W4G128 | 3.1578 / 23.52 | +0.0440 |
| W4MIXG128 | 3.1511 / 23.36 | +0.0372 |

The first mixed policy improves quality, but only modestly because pure W4 is
already close on this sample.

### Performance Smoke

Single run only, pp256 + tg64, warmup=3, 4 threads:

| Package | pp/tg | load ms | peak RSS |
|---|---:|---:|---:|
| W4MIXG128 | 117.94 / 81.18 | 306.27 | 4384.1 MB |

Use this only as a smoke result. The strict baseline for Youtu W4G128 remains
the 5-run median `118.39 / 89.61`, RSS `2893.6 MB`. Mixed increases RSS and
currently slows decode, so the next policy step should be targeted ablation:
try `lm_head`-only, `lm_head+kv_b/o`, and `lm_head+mlp.down` before settling on
the W8 coverage.

## Attempt 20: W4 q4dot Kernel Cleanup, GEMM 8x8

### Goal

Improve W4 kernel throughput for both prefill and decode without changing the
W4 package layout or quantization math. The target package for this pass was
Youtu-LLM-2B W4G128 direct q4pkg.

### Kept Changes

- Added a W4 q8dot GEMM `8x8` kernel. The previous dispatcher used `tile_m=8`
  but still called the internal `4x8` kernel twice, which reloaded/unpacked the
  same q4 B tile for the second group of 4 rows. The new `8x8` path unpacks B
  once and reuses it across 8 activation rows.
- Kept `MOLLM_W4_Q8_DOT_GEMM_4X8=1` as a diagnostic fallback to the old W4
  GEMM kernel.
- Reused small thread-local W4 GEMV scratch buffers for `qA_even`, `qA_odd`,
  and activation scales. The scratch is only used in `M==1` W4 GEMV, so the
  retained capacity is a few KiB rather than a prefill-sized qA buffer.

### Reverted Experiments

- NEON-vectorized qA quantization was tested and reverted. It did not improve
  decode and made the main prefill microbench shapes slower, largely due to
  extra temporary packing in the even/odd GEMV path.
- A 16-column W4 GEMV tile was tested and removed. Synthetic microbench looked
  promising for one shape, but real Youtu profile showed slower decode,
  especially `M=1,N=2048,K=6144`.

### Microbench

Dominant Youtu W4 prefill shapes, 4 threads, `warmup=10`, `repeat=20`.
These are shape-level microbenches, not strict package benchmark rows.

| Shape `(M,K,N)` | 4x8 fallback p50 | 8x8 p50 | speedup |
|---|---:|---:|---:|
| `256,2048,12288` | 25.27 ms | 22.17 ms | 1.14x |
| `256,6144,2048` | 12.68 ms | 11.67 ms | 1.09x |

### Package Benchmark

Strict protocol: Youtu-LLM-2B W4G128 direct q4pkg, pp256 + tg64, warmup=3,
4 threads, 5 independent process median.

| Package | pp/tg median | load ms median | RSS median | CE/PPL |
|---|---:|---:|---:|---:|
| W4G128 direct q4pkg before GEMM 8x8 | 118.39 / 89.61 | 114.80 | 2893.6 MB | 3.1578 / 23.52 |
| W4G128 direct q4pkg after GEMM 8x8 | 131.73 / 91.38 | 116.17 | 2893.6 MB | 3.1578 / 23.52 |

The strict package result is thermally noisy and drifted downward across the 5
processes, but the median still shows the expected prefill improvement. Decode
only moves slightly; decode remains dominated by W4 GEMV q4 unpack + SDOT work.

Fan-max same-binary A/B, alternating default `8x8` and
`MOLLM_W4_Q8_DOT_GEMM_4X8=1` old `4x8`, same strict protocol:

| W4 GEMM path | pp/tg median | prefill ms median | decode ms median | total ms median | RSS median |
|---|---:|---:|---:|---:|---:|
| default 8x8 | 137.26 / 92.38 | 1865.13 | 681.99 | 2546.14 | 2893.2 MB |
| fallback 4x8 | 120.10 / 92.74 | 2131.50 | 679.31 | 2813.75 | 2893.5 MB |

This isolates the endpoint effect of the GEMM change better than the
historical before/after table: prefill improves by about 14% in tok/s and total
latency improves by about 9.5%, while decode is essentially unchanged.

Same current binary all-model W4G128 rerun after the GEMM 8x8 change:

| Model / package | pp/tg median | prefill ms | decode ms | load ms | peak RSS | 256-token CE/PPL |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B W4G128 direct q4pkg | 415.08 / 163.02 | 616.74 | 386.46 | 197.96 | 1499.8 MB | 2.4239 / 11.29 |
| Youtu-LLM-2B W4G128 direct q4pkg | 131.20 / 90.77 | 1951.26 | 694.07 | 119.68 | 2893.4 MB | 3.1578 / 23.52 |
| Qwen3.5-4B W4G128 direct q4pkg | 56.58 / 39.96 | 4524.21 | 1576.72 | 231.23 | 4984.4 MB | 1.9568 / 7.08 |

### Validation

```bash
cmake --build build --target test_matmul bench_matmul mollm_bench test_quantized_e2e -j
./build/test_matmul
./build/test_quantized_e2e youtu-llm-2b.mollm youtu-llm-2b_w4g128_cppq.mollm 256
```

All passed. The 256-token CE/PPL stayed `3.1578 / 23.52` with CE delta
`+0.0440` vs FP16.

## Attempt 21: Default AUTO i8mm Build Detection

### Goal

Make the W8 i8mm GEMM path the default on ARM64 machines that can actually
build and run it. W8 prefill was still far behind llama.cpp mostly because the
default build stayed on DOTPROD even though the i8mm kernel already existed.

### Implementation

- Changed `MOLLM_ARM_I8MM` from a boolean OFF-by-default option to a tri-state
  cache string: `AUTO` / `ON` / `OFF`.
- `AUTO` is the default. On ARM64, CMake probes
  `-march=armv8.6-a+i8mm+fp16+fp16fml` with a small program that requires
  `__ARM_FEATURE_MATMUL_INT8` and calls `vmmlaq_s32`.
- In native builds the probe is compiled and run, so unsupported hosts do not
  get binaries that would trap on i8mm instructions. During cross-compilation,
  CMake uses the compile-target probe only.
- `MOLLM_ARM_I8MM=ON` requires the probe to pass. `OFF` forces the portable
  DOTPROD/NEON path.

### Validation

Fresh default configure on Apple ARM64:

```text
-- Performing Test MOLLM_ARM_I8MM_COMPILES - Success
-- Performing Test MOLLM_ARM_I8MM_RUNS - Success
-- ARM64 i8mm target enabled (-march=armv8.6-a+i8mm+fp16+fp16fml)
```

Forced-off configure:

```text
-- ARM64 i8mm target disabled by MOLLM_ARM_I8MM=OFF
```

Correctness:

```bash
cmake -S . -B build -DMOLLM_ARM_I8MM=AUTO
cmake --build build --target test_matmul bench_matmul mollm_bench -j
./build/test_matmul
```

All matmul tests passed.

### Microbench

Current `build/` after AUTO reconfigure, Youtu dominant W8 prefill shape,
4 threads, `warmup=10`, `repeat=20`:

| Path | M,K,N | p50 | GFLOPS |
|---|---|---:|---:|
| default i8mm | `256,2048,12288` | 16.98 ms | 751.3 |
| `MOLLM_W8_NO_I8MM=1` DOTPROD fallback | `256,2048,12288` | 24.23 ms | 508.1 |

This turns the faster W8 prefill path into the normal build result on supported
ARM64 targets, while preserving a compile/runtime fallback for older CPUs.

### W8 vs W4 Package Benchmark

Current AUTO-i8mm build, pp256 + tg64, warmup=3, 4 threads, 5 independent
process median:

| Model | W8PC pp/tg | W4G128 pp/tg | W8/W4 prefill | W8/W4 decode | W8 RSS | W4 RSS |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 596.37 / 150.00 | 444.47 / 164.18 | 1.34x | 0.91x | 2560.7 MB | 1501.3 MB |
| Youtu-LLM-2B | 233.43 / 88.25 | 146.37 / 93.81 | 1.59x | 0.94x | 5948.4 MB | 2893.6 MB |
| Qwen3.5-4B | 108.87 / 42.51 | 68.43 / 45.54 | 1.59x | 0.93x | 10964.7 MB | 4984.4 MB |

Takeaway: enabling i8mm by default changes the W8/W4 tradeoff. W8PC is now the
faster prefill option across all tested models, while W4G128 remains the lower
RSS and slightly faster decode option.

## Attempt 22: W4 DOTPROD Vector-Reduce GEMM

This attempt supersedes the W4G128 rows in Attempt 21's W8-vs-W4 package
benchmark. Attempt 21 measured W8 after AUTO-i8mm, before the W4 GEMM
vector-reduce rewrite.

### Goal

Improve W4 prefill without changing the W4 package layout, quantization math, or
resident memory model. Attempt 20's 8x8 GEMM removed duplicate q4 B unpacking,
but the inner loop still reduced every output with scalar `vaddvq_s32` and
round-tripped partial dots through stack arrays.

### Implementation

- Rewrote the default W4 `8x8` q4dot GEMM to keep dot results in NEON vectors:
  `vdotq_s32` produces four partial sums per output column, then pairwise
  `vpaddq_s32` builds the 8 column results as `dots_lo` / `dots_hi`.
- Accumulates directly into `float32x4_t acc_lo[8]` / `acc_hi[8]` per row,
  avoiding per-output scalar reductions and `int32_t dots[8][8]` stack traffic.
- Kept `MOLLM_W4_Q8_DOT_GEMM_4X8=1` as the old 4x8 diagnostic fallback.
- Added an opt-in W4 i8mm prototype behind `MOLLM_W4_I8MM=1`. It unpacks q4
  blocks into int8 even/odd micro-panels inside the kernel and feeds
  `vmmlaq_s32`. Correctness passes, but current microbench is slower than the
  DOTPROD vector-reduce path, so the default remains DOTPROD.

### Microbench

Current AUTO-i8mm build, 4 threads, `warmup=10`, `repeat=20`:

| Shape `(M,K,N)` | default DOTPROD vector-reduce | W4 i8mm opt-in | old 4x8 fallback |
|---|---:|---:|---:|
| `256,2048,12288` | 18.96 ms / 687.5 GF/s | 21.41 ms / 606.3 GF/s | 30.86 ms / 432.8 GF/s |
| `256,6144,2048` | 10.14 ms / 635.6 GF/s | 11.74 ms / 562.5 GF/s | not rerun |

Short shape profile on Youtu-LLM-2B W4G128 direct q4pkg still dispatches the
default package path as `q4dot_gemm_repack`; the i8mm prototype reports
`q4dot_gemm_repack_i8mm` only when `MOLLM_W4_I8MM=1` is set.

### Package Benchmark

Current binary, pp256 + tg64, warmup=3, 4 threads, 5 independent process
median:

| Model / package | pp/tg median | prefill ms | decode ms | load ms | peak RSS | 256-token CE/PPL |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B W4G128 direct q4pkg | 596.96 / 164.34 | 428.84 | 383.36 | 193.25 | 1500.6 MB | 2.4239 / 11.29 |
| Youtu-LLM-2B W4G128 direct q4pkg | 196.56 / 90.20 | 1302.38 | 698.44 | 110.45 | 2893.7 MB | 3.1578 / 23.52 |
| Qwen3.5-4B W4G128 direct q4pkg | 85.79 / 39.55 | 2984.11 | 1592.89 | 222.46 | 4984.3 MB | 1.9568 / 7.08 |

Same-session W8PC vs W4G128 after this change:

| Model | W8PC pp/tg | W4G128 pp/tg | W8/W4 prefill | W8/W4 decode | W8 RSS | W4 RSS |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 596.37 / 150.00 | 596.96 / 164.34 | 1.00x | 0.91x | 2560.7 MB | 1500.6 MB |
| Youtu-LLM-2B | 233.43 / 88.25 | 196.56 / 90.20 | 1.19x | 0.98x | 5948.4 MB | 2893.7 MB |
| Qwen3.5-4B | 108.87 / 42.51 | 85.79 / 39.55 | 1.27x | 1.07x | 10964.7 MB | 4984.3 MB |

### Validation

```bash
cmake --build build --target test_matmul bench_matmul mollm_bench -j
./build/test_matmul
MOLLM_W4_I8MM=1 ./build/test_matmul
./build/test_quantized_e2e youtu-llm-2b.mollm youtu-llm-2b_w4g128_cppq.mollm 256
```

All passed. Youtu-LLM-2B W4G128 256-token CE/PPL stayed `3.1578 / 23.52`
with CE delta `+0.0440` vs FP16.

### Takeaway

The current best W4 path is direct q4dot package storage plus DOTPROD
vector-reduce GEMM. The W4 i8mm prototype shows that simply unpacking W4 to
int8 inside the hot loop does not pay for itself; revisiting i8mm needs a true
W4-aware packed layout or a different activation/weight micro-panel design.

## Attempt 23: W4 Scaled-Nibble DOTPROD

### Kernel Gap Check vs llama.cpp

Follow-up inspection compared mollm's W4 kernel against local llama.cpp
`5c7c22c3e` / build 9803:

- llama.cpp loads Q4_0 tensors into CPU repack buffers such as `q4_0_4x8`.
- Its Q4_0 ARM GEMM uses packed nibble data directly in hand-written i8mm asm:
  `sshl` / `and` form high-nibble signed values (`q4 * 16`), `smmla` accumulates,
  then `scvtf #4` divides by 16 before scale application.
- A CPU-only no-BLAS/no-Metal llama.cpp build still measured strong 4-thread
  prefill, so the gap is not just Apple BLAS:

| Runtime | Package | pp256 | tg64 |
|---|---:|---:|---:|
| llama.cpp CPU-only | Youtu-LLM-2B Q4_0 | 264.55 | 95.91 |
| llama.cpp CPU-only | Youtu-LLM-2B Q8_0 | 233.84 | 80.55 |
| llama.cpp CPU-only | Qwen3.5-4B Q4_0 | 139.06 | 43.15 |
| llama.cpp CPU-only | Qwen3.5-4B Q8_0 | 128.50 | 39.46 |

Compared with same-session mollm single runs, decode is already close on 4B
(`W4 40.98` vs `Q4_0 43.15`, `W8 41.39` vs `Q8_0 39.46`), while prefill still
trails (`W4 88.78` vs `Q4_0 139.06`, `W8 102.20` vs `Q8_0 128.50`). The next
large W4 step should therefore be a true packed-i8mm/asm-style prefill kernel,
not the existing runtime-unpack i8mm prototype.

### Implementation

This attempt applies the first small part of the llama.cpp trick to the current
DOTPROD path without changing package layout:

- Keep the existing signed-int4 two's-complement nibbles in memory.
- For DOTPROD only, load each packed byte as high-nibble signed values:
  `even_scaled = packed << 4`, `odd_scaled = packed & 0xF0`.
- Dot products therefore accumulate `q4 * 16 * q8`; conversion to float uses
  `vcvtq_n_f32_s32(..., 4)` (or `* 1/16` fallback) before applying
  `a_scale * w_scale`.
- The W4 i8mm prototype still uses the original full sign-extend helper because
  it currently expects true int8 q4 lanes.

### Microbench

AUTO-i8mm build, default W4 DOTPROD path, 4 threads, `warmup=10`, `repeat=20`:

| Shape `(M,K,N)` | Attempt 22 p50 | Attempt 23 p50 |
|---|---:|---:|
| `256,2048,12288` | 13.94 ms / 922.3 GF/s | 13.68 ms / 942.5 GF/s |
| `256,6144,2048` | 7.32 ms / 877.9 GF/s | 7.09 ms / 903.7 GF/s |
| `256,2560,9216` | not recorded | 12.81 ms / 942.6 GF/s |
| `256,9216,2560` | not recorded | 13.53 ms / 890.6 GF/s |

Diagnostic A/B on `256,2048,12288`:

| Path | p50 |
|---|---:|
| default scaled-nibble DOTPROD 8x8 | 13.68 ms / 942.5 GF/s |
| `MOLLM_W4_I8MM=1` runtime-unpack i8mm | 15.85 ms / 811.6 GF/s |
| `MOLLM_W4_Q8_DOT_GEMM_4X8=1` old fallback | 24.63 ms / 522.7 GF/s |

### Package Profile

Short profile (`pp256 + tg4`, warmup=1, 4 threads):

| Model | Dominant shape | Attempt 22 avg | Attempt 23 avg |
|---|---|---:|---:|
| Youtu-LLM-2B W4G128 | `M=256,N=12288,K=2048` | 14.698 ms | 14.456 ms |
| Qwen3.5-4B W4G128 | `M=256,N=9216,K=2560` | 13.909 ms | 13.336 ms |
| Qwen3.5-4B W4G128 | `M=256,N=2560,K=9216` | 14.468 ms | 13.666 ms |

The improvement is modest, as expected: this removes part of the W4 unpack
overhead, but it does not change the kernel's tile family or scheduling.

### Validation

```bash
cmake --build build --target test_matmul bench_matmul mollm_bench -j 10
./build/test_matmul
MOLLM_W4_I8MM=1 ./build/test_matmul
cmake --build build --target test_quantized_e2e -j 10
./build/test_quantized_e2e youtu-llm-2b.mollm youtu-llm-2b_w4g128_cppq.mollm 256
```

All passed. Youtu-LLM-2B W4G128 256-token CE/PPL remains `3.1578 / 23.52`
with CE delta `+0.0440` vs FP16.

### Takeaway

Keep this change: it is correctness-neutral and consistently improves the main
W4 GEMM microbench shapes. It does not close the llama.cpp gap by itself. The
remaining prefill gap is structural: mollm still has a DOTPROD kernel over the
current q4dot layout, while llama.cpp has packed Q4_0/Q8_0 block formats and
hand-scheduled i8mm kernels.

## Attempt 24: W4 Full-Tile Fast Path

### Motivation

After the scaled-nibble change, the default W4 prefill path was still spending
most time in the 8x8 DOTPROD GEMM. Model dimensions are almost always multiples
of 8 on the output axis, but the kernel still carried `c_valid` tail checks in
the hot inner dot and store loops. This attempt specializes the common full
`N` tile case without changing the package format or quantization math.

### Rejected Experiment: GEMM A Even/Odd Pre-Split

Before the full-tile specialization, I tried pre-quantizing GEMM activations
directly into even/odd q8 buffers so the W4 DOTPROD kernel would not repeat
`vuzp1/vuzp2` for every output-column tile.

Result: rejected and reverted. The extra split-buffer memory traffic was worse
than the saved shuffle work on current shapes.

Sequential microbench signals before revert:

| Shape `(M,K,N)` | Baseline Attempt 23 | A even/odd pre-split |
|---|---:|---:|
| `256,2048,12288` | 13.68 ms p50 | 13.93 ms p50 |
| `256,6144,2048` | 7.09 ms p50 | 7.99 ms p50 |

Two additional 4B shape probes were accidentally run concurrently and are not
used as valid benchmark numbers; they only confirmed the same regression
direction. The code was restored to contiguous qA before continuing.

### Implementation Kept

The retained change adds a fast path for `c_valid == 8` in the W4 8x8 DOTPROD
GEMM:

- load all 8 q4 columns unconditionally for full output tiles;
- run all 8 column dot products without per-column validity branches;
- store `acc_lo/acc_hi` directly with two vector stores instead of scalar tail
  stores through a temporary array.

The tail path is still present for odd `N`, and `test_matmul` covers odd-N W4
cases.

### Microbench

AUTO-i8mm build, default W4 DOTPROD path, 4 threads, `warmup=10`, `repeat=20`:

| Shape `(M,K,N)` | Attempt 23 p50 | Attempt 24 p50 |
|---|---:|---:|
| `256,2048,12288` | 13.68 ms / 942.5 GF/s | 13.25 ms / 968.0 GF/s |
| `256,6144,2048` | 7.09 ms / 903.7 GF/s | 7.07 ms / 908.0 GF/s |
| `256,2560,9216` | 12.81 ms / 942.6 GF/s | 12.38 ms / 971.8 GF/s |
| `256,9216,2560` | 13.53 ms / 890.6 GF/s | 13.15 ms / 922.8 GF/s |

### Package Profile

Short profile (`pp256 + tg4`, warmup=1, 4 threads):

| Model | Metric | Attempt 23 | Attempt 24 |
|---|---|---:|---:|
| Youtu-LLM-2B W4G128 | prefill ms | 1112.22 | 1068.03 |
| Youtu-LLM-2B W4G128 | `M=256,N=12288,K=2048` avg | 14.456 ms | 13.716 ms |
| Youtu-LLM-2B W4G128 | `M=256,N=2048,K=6144` avg | 7.384 ms | 7.075 ms |
| Qwen3.5-4B W4G128 | prefill ms | 2342.51 | 2276.66 |
| Qwen3.5-4B W4G128 | `M=256,N=9216,K=2560` avg | 13.336 ms | 12.590 ms |
| Qwen3.5-4B W4G128 | `M=256,N=2560,K=9216` avg | 13.666 ms | 13.119 ms |

### Validation

```bash
cmake --build build --target test_matmul bench_matmul mollm_bench -j 10
./build/test_matmul
cmake --build build --target test_quantized_e2e -j 10
./build/test_quantized_e2e youtu-llm-2b.mollm youtu-llm-2b_w4g128_cppq.mollm 256
```

All passed. Youtu-LLM-2B W4G128 256-token CE/PPL remains `3.1578 / 23.52`
with CE delta `+0.0440` vs FP16.

### Takeaway

Keep the full-tile fast path. It is a small but consistent win on all main W4
prefill microbench shapes and improves short package profiles. The failed A
even/odd pre-split is useful negative evidence: the current kernel is more
sensitive to memory-stream pressure than to the repeated `vuzp` shuffle. The
next major gap to llama.cpp still requires a real packed W4/i8mm microkernel
and probably scale packing, not just activation-side repacking.

## Attempt 25: W8 i8mm Full-Tile Cleanup

### Diagnosis

After Attempt 24, current strict endpoint showed that W4 is no longer the main
2B bottleneck:

| Model | W8PC pp/tg | W4G128 pp/tg |
|---|---:|---:|
| Youtu-LLM-2B | 227.47 / 88.76 | 227.31 / 91.28 |
| Qwen3.5-4B | 108.00 / 42.28 | 102.42 / 43.34 |

Same-session llama.cpp CPU-only no-BLAS rerun still shows a real prefill gap:

| Model | llama.cpp Q8_0 | llama.cpp Q4_0 |
|---|---:|---:|
| Youtu-LLM-2B | 275.90 / 83.99 | 281.04 / 99.82 |
| Qwen3.5-4B | 142.88 / 39.77 | 148.67 / 46.43 |

Short 4B profiles showed W8, W4, and FP16 dominant prefill GEMMs all in the same
range. Before this attempt, Qwen3.5-4B W8PC dominant shapes were:

| Shape `(M,K,N)` | W8PC p50 | FP16 p50 | W4G128 p50 |
|---|---:|---:|---:|
| `256,2560,9216` | 11.60 ms | 12.05 ms | 12.47 ms |
| `256,9216,2560` | 12.14 ms | 13.48 ms | 13.05 ms |

So W8 i8mm was correct direction, but it was only modestly ahead of FP16FML.

### Implementation

The W8 i8mm GEMM kernel still used the generic 4x8 tail-safe path for the common
full tile case. This attempt specializes full `M=4,N=8` tiles:

- remove per-row validity checks in the per-qblock scale/add path when the 4-row
  tile is complete;
- store full 8-column output rows with direct vector stores instead of scalar
  tail stores through a temporary array;
- keep the original tail path for odd `M`/`N`.

This does not change quantization math or package layout.

### Microbench

AUTO-i8mm build, W8PC/per-channel, 4 threads, `warmup=10`, `repeat=20`:

| Shape `(M,K,N)` | Before | After |
|---|---:|---:|
| `256,2560,9216` | 11.60 ms / 991.2 GF/s | 11.17 ms / 1077.3 GF/s |
| `256,9216,2560` | 12.14 ms / 986.8 GF/s | 11.79 ms / 1019.4 GF/s |
| `256,2048,12288` | not rerun before | 11.97 ms / 1072.9 GF/s |
| `256,6144,2048` | not rerun before | 6.41 ms / 998.7 GF/s |

### Endpoint

Strict protocol (`pp256 + tg64`, warmup=3, 4 threads, 5 independent process
median):

| Model / package | Before | After |
|---|---:|---:|
| Qwen3.5-0.8B W8PC | 596.37 / 150.00 | 604.11 / 151.23 |
| Youtu-LLM-2B W8PC | 227.47 / 88.76 | 239.59 / 87.89 |
| Qwen3.5-4B W8PC | 108.00 / 42.28 | 114.36 / 43.02 |

Short 4B W8PC profile also improved:

| Metric | Before | After |
|---|---:|---:|
| prefill ms | 2270.34 | 2096.85 |
| prefill MATMUL total | 1954.33 | 1803.04 |
| `M=256,N=9216,K=2560` avg | 12.609 ms | 11.633 ms |
| `M=256,N=2560,K=9216` avg | 13.096 ms | 12.105 ms |

### Validation

```bash
cmake --build build --target test_matmul bench_matmul mollm_bench -j 10
./build/test_matmul
./build/test_quantized_e2e youtu-llm-2b.mollm youtu-llm-2b_w8pc.mollm 256
```

Both passed. Youtu-LLM-2B W8PC 256-token CE/PPL remains `3.1094 / 22.41`
with CE delta `-0.0045` vs FP16.

### Takeaway

Keep this cleanup: it is small, correctness-neutral, and gives a measurable
W8 prefill win. The remaining gap is still prefill-heavy and mostly shared
between W8/W4/FP16 matmul families. The next larger W8 step is likely an 8-row
i8mm tile or a llama.cpp-style packed micro-panel that reuses B loads across
more rows, but that needs a careful register-pressure A/B rather than another
tail-path cleanup.
