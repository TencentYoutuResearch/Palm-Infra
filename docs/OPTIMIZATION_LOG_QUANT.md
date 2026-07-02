# Quantization Optimization Log

*Last updated: 2026-07-02*

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
