# palm-infra

AI Infra projects from [PalmAI](https://palm.tencent.com/) Team. Currently includes `mollm`.

[中文文档](README_zh.md)

## mollm

mobile-oriented LLM inference engine.
```
                 _ _
 _ __ ___   ___ | | |_ __ ___
| '_ ` _ \ / _ \| | | '_ ` _ \
| | | | | | (_) | | | | | | | |
|_| |_| |_|\___/|_|_|_| |_| |_|
```

`mollm` is a small C++ LLM runtime for ARM and x86 CPUs, with experimental
Apple Metal support. It converts supported Hugging Face model directories into
one `.mollm` file containing the graph, weights, tokenizer, and chat template,
then runs that package directly.

The current focus is fast local inference on Apple Silicon and other modern ARM
CPUs. FP16 uses NEON FP16FML kernels; quantized CPU models use weight-only int8
or int4 kernels optimized for ARM dot-product instructions. Linux x86_64 has
separately compiled scalar, AVX2/FMA/F16C, and AVX-512 providers for FP32,
FP16, W8, and packed W4G32/W4G128 matmul. Runtime CPUID dispatch selects the
widest supported tier without exposing newer instructions to older CPUs.
Set `MOLLM_X86_ISA=scalar|avx2|avx512|auto` to cap the selected tier for
testing or machine-specific tuning. The legacy
`MOLLM_X86_DISABLE_AVX2=1` scalar override remains supported.

## Large MoE models with SSD offload

`mollm` keeps dense weights in RAM and loads only routed experts from SSD into
a bounded shared cache. The same asynchronous offload path supports W4 models
as well as checkpoint-native FP8/MXFP4 and NVFP4 experts.

| Model | Expert cache | Prefill | Decode |
|---|---:|---:|---:|
| Qwen3.5-122B-A10B W4 | 16 GiB | **51.06 t/s** | **16.53 t/s** |
| Qwen3.8-Flash-Next NVFP4 | 20 GiB | **16.2 t/s** | **21.8 t/s** † |
| DeepSeek-V4-Flash | 16 GiB | **9.52 t/s** | **5.71 t/s** |
| Hy3-295B-A21B W4G128 | 16 GiB | **10.57 t/s** | **3.41 t/s** |

† Best observed real-prompt run with eight CPU threads and a 20 GiB expert
cache; the other rows use their documented four-thread / 16 GiB protocols.

[Cache sweeps, I/O behavior, and tracing](docs/ssd-offload.md) ·
[Complete performance tables and protocols](docs/performance.md)

## What Works

Existing experimental `MOLLM_CUDA_SDPA_PREFILL_HEAD_GROUP=4` is qualified on
the 1743-token logical-batch preset: four query heads reuse one KV head with
no additional workspace. Requires validated-prefix, skip-zero-PV, query-group4
and shared-score-limit1024; D256, queries16..2048, keys1025..4096 and a GQA
ratio divisible by4. Default OFF; short windows, decode, incompatible shapes,
other backends and weight formats are unchanged. Matched hot prefill gain is
about 1.15%, with identical model losses/token IDs; see `docs/performance.md`
for the complete preset and evidence.

For CUDA memory diagnostics, `MOLLM_CUDA_PREFILL_REUSE_WORKSPACE=0`
disables retention of all same-shape prefill node outputs and uses tensor
lifetimes where the graph supports them. Stateful graphs currently defer
cleanup until the end of the call; this switch only avoids cross-call retention
there. Unset preserves existing behavior; decode/MTP are unchanged.
The device pool still caches freed allocations: this is not a memory limit or
a guarantee that a larger chunk/context fits. No large-input speedup is verified.
Bounded opt-in `--cuda-prefill-reclaim` (CLI/bench/PPL, default off) enables
conservative temporary reclamation for CUDA resident+offload-PLE only. It
requires `MOLLM_RESIDENT_EXPERT_ARENA=cuda-device`,
`MOLLM_CUDA_HOST_HEADROOM_MB>=24576`, context<=4096 and per-call sequence<=1024;
invalid configurations reject rather than fall back to CPU. It changes neither
chunk selection nor decode/weight layout. After fixing materialized view-chain
lifetimes, full Qwen loss tables match fresh chunk256/512 references and
historical chunk1024 references; see `docs/performance.md` for exact fixtures.
Planner isolated speedup remains unverified; different chunk sizes do not
produce equivalent losses or necessarily identical generated text.
Experimental `--cuda-prefill-logical-batch` separately admits one dynamic
Qwen3.8 prefill batching (default off): requires reclaim,
chunk0..2048 (0 defaults to2048), output pool1..4096 MiB and disabled decode segments; rejects
MTP/padding/SSD/vision/profile/window scheduling. It enables read-only last-use
contracts itself; unset legacy stateful/liveness sequence caps.
**Important:** without `--qsa-companion`, the current Qwen3.8 bring-up package
omits the QSA indexer and uses experimental dense attention beyond its2048-token
budget. Runtime context capacity alone does not restore original-model semantics.
With the matching companion, CUDA text inference uses native QSA indexing and
selected-key attention; the loader patches the graph in memory without rewriting
the large package. Continuous causal history is supported up to262144 tokens,
subject to memory admission. MTP/static padding/vision and noncanonical masks
are not supported; unsupported QSA configurations reject rather than use dense
attention. Independent primitive references, optimization-relative token losses,
state checks and throughput evidence are recorded in
[`docs/qsa-implementation-status.md`](docs/qsa-implementation-status.md);
these are not an original-BF16 whole-model quality certification.
Context capacity is independent of chunk size: e.g. `--n-ctx 16384` with
`--prefill-chunk-size 512`. Long inputs append KV in configured chunks; short
inputs are not padded. 2048 remains the supported per-batch maximum, not a
context limit. Long KV falls back from ineligible grouped attention paths;
host memory guards remain enabled. Long-context throughput needs separate measurement.
On GB10, build `mollm_chat`, then run `./chat-qwen38-cuda.sh` for an opt-in
standalone interactive preset (context16384, chunk2048, max512 generated tokens).
CLI arguments override defaults, e.g. `--prefill-chunk-size 512`. The launcher
can enable QSA with `./chat-qwen38-cuda.sh --qsa-companion ../qwen38_qsa_bundle`.
For QSA contexts above65536, its default chunk/cache limits are512 tokens/4096
MiB; explicit chunk/cache overrides remain honored. Smaller contexts retain
the existing defaults. The companion must match the selected checkpoint.
The launcher
uses local `../qwen38_flash_next_nvfp4_calibrated.mollm`, never sources benchmark
scripts or downloads/builds automatically, and requires an idle GPU and100 GiB
MemAvailable at startup. Admission checks are not memory reservations.
Linux chat/bench additionally reuse `MOLLM_CUDA_HOST_HEADROOM_MB` to monitor
host availability every250ms (24 GiB in this preset). Below-threshold or
unreadable availability exits only the model process with137, without GPU
reset or parent/session signals. No extra flag or stdin/TTY wrapper. This is
best-effort, not protection against instantaneous allocations; emergency
termination attempts to restore TTY settings and skips model destructors.
Logical-prefill automatically selects eligible MoE input-half reuse and
token K-major views. The launcher no longer needs
`MOLLM_CUDA_MOE_PREFILL_INPUT_HALF_REUSE=1` or
`MOLLM_CUDA_NVFP4_PREFILL_TOKEN_CODES_KMAJOR=1`; explicit0 still disables
each independently. Ordinary CUDA/decode and CPU/Metal defaults are unchanged,
and existing shape/layout eligibility still applies.
Logical-prefill additionally requests qualified MoE input scale K-major,
compact logits, shared-token quantization and pair-word reads automatically.
The launcher no longer forces `MOLLM_CUDA_NVFP4_PREFILL_INPUT_SCALE_KMAJOR`,
`COMPACT_LOGITS`, `TOKEN_QUANTIZE` or `TOKEN_PAIR_WORDS` (the latter three
also use the `MOLLM_CUDA_NVFP4_PREFILL_` prefix). Explicit0 disables each;
existing fusion, uniform-scale, alignment and weight-layout guards remain.
Defaults outside logical-prefill are unchanged.
Logical-prefill also automatically requests validated prefix, skip-zero PV
and dimension-paired PV: no need to set
`MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1`,
`MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV=1` or `MOLLM_CUDA_SDPA_PREFILL_PV_PAIR=1`.
Existing validation/shape requirements remain; explicit0 disables each option,
and defaults outside logical-prefill remain off.
The long-KV optimizations/equivalence below describe the dense bring-up
diagnostic path, not the companion-enabled QSA path.
The preset now automatically selects qualified long-KV FP32 GEMM PV while
retaining native QK/softmax; no additional startup flag or model conversion.
Scope: Qwen3.8 H24/KVH2/D256,32..2048 queries and compatible layouts, with
KV capacity up to32768. GB10 (SM121) automatically admits actual4097..32768
keys; other architectures retain the automatic4097..16384 range. Selection
uses actual key length; larger capacity alone no longer causes fallback.
The extended path passes complete approximately28K input, token-loss,
512-token multi-turn/reset and operator sanitizer checks; fresh default-binary
regressions also pass nine backend fixtures,7K/14K/28K token losses and daily
multi-turn replies. Same-binary alternating approximately28K-input
benchmarks improve prefill388.9 to446.5TPS (14.8%), with decode unchanged.
This neither guarantees every capacity/chunk combination fits memory nor
permits direct comparison with7K-input throughput.
Short windows, decode, captures and unsupported configurations retain
their original paths. Logical score growth uses exact required capacity rather
than rounding2.25 GiB up to4 GiB. See [performance evidence](docs/performance.md)
for measured workloads and qualification limits.
The qualified SM121 long two-head path also automatically elides repeated
positive-zero mask loads in QK, preserving positive-zero addition and the
original numeric order. Negative-zero/noncanonical masks retain original
fallback computation; validation is not skipped. No additional switch or model
conversion. Same-binary7K/32K alternating measurement improves prefill about3.9%,
with exact7K/14K token loss and long-first-turn/follow-up/reset replies;
see the performance record for scope and remaining limitations.
On GB10 (SM121), eligible long-KV calls with actual keys<=16384 additionally pair two query heads
within each warp, preserving native ordered QK and softmax reduction trees.
No extra global workspace or startup option; other CUDA architectures and
unsupported shapes retain the prior layout.
SM121 mid-KV windows also automatically select native QK/softmax plus GEMM
PV when past>=2048, actual keys2049..4096, queries32..2048 and the above
layout/validation guards hold. This range keeps the single-head probability
producer; first short windows and captures retain prior paths. No new
startup option. Qualified with chunk1024/2048,7K/14K inputs and multiturn;
private `MOLLM_TEST_CUDA_SDPA_MID_PV_GEMM=0` retains the prior path for A/B.
The qualified mid-KV single-head path now also elides repeated validated
positive-zero mask loads automatically, sharing the long-window policy rather
than adding an independent switch. Same-binary7K/32K alternating measurement
improves prefill a further2.1%; exact7K/14K loss, chunk1024 and multi-turn/reset
verification are recorded in the performance notes.
See the performance record for measurement and qualification scope.
Logical batching also admits requested MoE K-major/compact-logit/shared-token
optimizations for short prefill, rather than requiring more than1024 rows;
native batching/layout constraints still apply. Ordinary windows and decode
retain their original policy. Each call
checks an extra4 GiB above the host guard (not a reservation or OOM guarantee).
Operator tiles stay independent; weights need no reconversion. Model quality
and warmed end-to-end throughput must be verified before adoption.
The quality-checked1743-token logical preset also sets both
`MOLLM_TEST_CUDA_HC_GATE_ROW_TILE=1024` and
`MOLLM_TEST_CUDA_HC_LOW_RANK_ROW_TILE=1024` to preserve numerical partitioning
of two HC projections. Logical-batch alone does not establish loss equality.
Independent default-off `MOLLM_TEST_CUDA_DENSE_WIDE_FULL_LT=1` admits
M1025..2048,N16480,K2560 wide projections to the screened zero-workspace Lt
algorithm; unsupported shapes/algorithms retain fallback. Other backends and
weight layout are unchanged. Do not combine with
`MOLLM_TEST_CUDA_DENSE_WIDE_ROW_TILE=1024`, which takes precedence and tiles
rows internally. See [performance records](docs/performance.md) for evidence
and validation scope.

Optional `MOLLM_CUDA_OUTPUT_POOL_MB` bounds CUDA output-pool reservations in MiB
(unset/0 disables). Before a new allocation would exceed this limit, the backend
synchronizes work, invalidates its cached CUDA Graphs and frees returned pool
blocks only; active buffers remain intact. If the request still cannot fit, or
capture/replay prevents reclamation, allocation fails. Reclamation may add
latency; this is not a speedup recommendation or a whole-system OOM guarantee.
Weights, scratch, persistent storage and driver allocations are outside this
limit. Existing host-headroom checks remain unchanged.

On Linux unified-memory CUDA systems, optional `MOLLM_CUDA_HOST_HEADROOM_MB`
checks `/proc/meminfo` MemAvailable before explicit CUDA allocations, including
the resident arena and scratch growth. Unset/0 disables it; invalid settings
or unreadable availability fail closed when configured. It is not a reservation
or a guarantee against OOM: concurrent allocations and driver overhead remain.
The value must contain decimal digits only (no sign or whitespace). A rejected
scratch growth budget returns failure without freeing the old buffer or entering
the CUDA-OOM allocation retry path; actual CUDA allocation failures retain the
existing fallback behavior.

Experimental `MOLLM_CUDA_GR_INJECT_RMS_FUSION=1` fuses Qwen gated injection
with the next group RMS normalization, retaining both FP32 outputs. It requires
the existing HC prefill activation reuse and RMS-half options; unsupported
shapes use the original path. Disabled by default and for profiling, tracing,
node dumps or partial execution. No model/weight conversion is required.
Small-graph exactness and full token-loss tables on the bounded Qwen sanity256
and engineering512 fixtures are verified. Matched engineering857-token/chunk512
OFF/ON and reverse-order runs reduce prefill time by1.30%/1.00%, with all12
complete32-token outputs identical, but use184MiB more output-pool storage.
Other shapes/model quality are not validated; this remains an opt-in experiment,
not a default performance preset. Details and memory limits: `docs/performance.md`.

Experimental `MOLLM_CUDA_HC_PREFILL_DEFERRED_RMS=1` additionally defers the HC
normalized FP32 writeback: existing FP16 projection shadows are reused and
GR_REDUCE reconstructs values from carry/gamma/inverses. Requires both fusion
options, HC activation reuse/RMS-half, and disabled decode segments. Strict
graph/output/lifetime checks keep unsupported paths ordinary; fallback reads
materialize FP32. Default off; adds64KiB/backend, no weight conversion or extra
expert copy. Bounded resident+PLE-offload engineering1743/chunk1024 tests have
identical1742-loss tables and all12 complete32-token outputs; paired model
prefill time improves1.77%/1.93% (~1090→1110tok/s). This is not validation of
other models/shapes or a default preset. Safety limits: `docs/performance.md`.

With that retained deferred-RMS/FP16-shadow configuration, optional existing
`MOLLM_CUDA_PREFILL_FP16_CACHE_HC_DOWN=1` caches the97 HC down matrices within
the same FP16 cache budget (extra606.25MiB, no expert duplicate or package reconversion).
Fresh1743-token/chunk1024 forward/reverse A/B reduces prefill time0.83%/0.51%
(matched means1107→1115tok/s); all1742 loss rows and12 complete32-token
outputs match. Decode path is unchanged; this remains default-off and only
validated for this bounded configuration, not a general performance preset.

For prefill-first use under the retained BF16 gate/down and deferred-RMS
configuration, existing optional `MOLLM_CUDA_PREFILL_FP16_CACHE_NARROW=1`
also caches48 dense2560/K6144 matrices within the same8GiB budget. Extra
1.40625GiB raises total FP16 cache to6.23GiB,no expert duplicate/reconversion;
decode uses original quantized weights. Fresh1743-token/chunk1024/ctx4096
forward/reverse tests improve prefill TPS1.25–1.37% (matched1154→1169TPS),
all1742 losses and12 complete32-token outputs match. Decode changes have
mixed signs (~±0.1%); earlier settings showed slight decode slowdown. This
remains default-off,a memory/TTFT tradeoff,not broad quality or universal speedup.

Experimental `MOLLM_TEST_CUDA_WINDOW_DEFER_MOE_INJECT=1` (default-off)
requires the existing merged layer-major window scheduler and segment fusions.
It moves eligible stateless injections across segment cuts to enable existing
inject/RMS fusion, preserving state-update order. Bounded current1743-token
resident+offload-PLE ABBA prefill1380→1391TPS (+0.78%), decode~35 unchanged;
full hidden/state/KV dump and12 token arrays agree. See `docs/performance.md`.

HC up/reduce graph preparation now shares a call-scoped structural index
instead of rebuilding it per candidate; no selector, device math or weight
layout changes. Matched current1743-token/chunk1024/ctx4096 resident+offload-PLE
ABBA improves prefill1393.35→1419.81TPS (+1.90%), decode~35 unchanged.
All100 hidden/state/KV records and12 complete token arrays agree. This is
host preparation optimization, not a measured GPU-kernel speedup; scope and
reproducible baseline details are in `docs/performance.md`.

Deferred-HC preparation also shares consumer/position indices per graph,
without changing device math or weights. Same bounded fixture ABBA prefill
1416.19→1437.83TPS (+1.53%);100 state records and12 full token arrays agree.
A32-token decode outlier is retained/reported; reverse-order128-token follow-up
has equal34.97TPS decode means and identical full outputs. This is scoped
host-planning evidence, not an all-workload no-regression guarantee.

Optional HC up/reduce controls `MOLLM_CUDA_HC_PREFILL_UP_INPUT_PITCH=336`
and `MOLLM_CUDA_HC_PREFILL_UP_TILE_COLUMNS=32` retain defaults320/16 and
use columns32 only for validated rows>=256. No new global scratch or model
conversion. Current bounded1743-token/chunk1024 resident+offload-PLE ABBA
prefill1363→1376TPS (+0.94%), decode~35TPS unchanged; full state dump/token
outputs agree. See `docs/performance.md` for scope and checks.

Optional `MOLLM_CUDA_NVFP4_PREFILL_INPUT_SCALE_KMAJOR=1` (default-off)
reorders input-scale scratch and quantizer work for fused MoE merged windows
with1024<rows<=2048. Ordinary small windows/decode retain their layouts;
no additional scratch/cache or reconversion. On the current bounded1743-token
resident+offload-PLE/query-group4/weight-scale-cache preset, same-binary ABBA
prefill1344.4→1359.6TPS (+1.13%), decode~35TPS unchanged; full hidden/state/KV
dumps and12 complete token arrays match. Scope and evidence: `docs/performance.md`.

Optional `MOLLM_CUDA_PREFILL_GATE_SCALE_CACHE_MB=<MiB>` (default0, max8192)
now budgets only the legacy canonical-gate scale views. The current chat
preset's resident gate-K-major mode automatically shares ONE K-major scale
payload across load/prefill/decode, eliminating the4.69GiB duplicate. No cache
budget or reconversion required. Arithmetic ordering is preserved; CPU
fallback restores canonical bytes, and CPU/Metal/SSD layouts are unchanged.
7K/32K and14K/16K losses plus multi-turn replies pass; whole-model throughput
has no stable improvement. This is a memory optimization, not a prefill
speedup claim. Details: `docs/performance.md`.

Optional `MOLLM_CUDA_NVFP4_PREFILL_COMPACT_LOGITS=1` (default-off) stores
already-BF16-rounded fused gate/up shared logits as lossless16-bit codes.
Requires both gate-scale views and Kmajor input scales, selecting only fused
resident merged windows1024<rows<=2048 (hidden2560/inter640); decode/fallbacks
stay unchanged. No new cache or conversion. Bounded1743-token same-binary
ABBA: prefill1483.2->1500.9TPS (+1.19%), full state dump and12 complete128-token
arrays agree. No stable decode gain; scope/checks in `docs/performance.md`.

Optional `MOLLM_CUDA_NVFP4_PREFILL_TOKEN_QUANTIZE=1` (default-off) shares
per-token gate/up quantization in the above fused merged windows. Every layer
checks that ALL expert gate/up input calibrations are identical; otherwise
the original path remains. Down/decode and weight files are unchanged, no
reconversion. Original scratch capacity remains, with an added route-index
region. Bounded1743-token same-binary ABBA prefill1507.4->1517.3TPS (+0.65%);
full state dump and12 complete128-token arrays match. Decode~35TPS has no
stable gain. Standalone gains are not whole-model gains; see performance docs.

Optional `MOLLM_CUDA_NVFP4_PREFILL_TOKEN_PAIR_WORDS=1` (default-off) requires
the above shared-token guard and pair-layout resident gates. It interleaves
temporary input words for one64-bit MMA-fragment read; no weight-file/cache/
scratch-capacity change. Bounded1743-token same-binary ABBA prefill1517.7->
1538.6TPS (+1.38%); reverse256-token follow-up+1.30%. Full state/output checks
agree.128-token decode decreases0.45%;256-token follow-up is essentially
neutral (~34.79TPS). No all-workload no-regression claim; see performance docs.

Experimental `MOLLM_CUDA_NVFP4_N32_ASYNC_WEIGHTS=1` enables asynchronous
CTA weight staging for canonical resident CUDA-device gate GEMMs only
(hidden2560/inter640, N32, four warps, prefill rows>=16). Compact/native-layout
or unsupported configurations keep the original path; down/decode are unchanged.
No weight conversion or extra resident expert copy. Kernel replay/guard and
sanitizer checks pass; bounded sanity256/engineering512 full loss tables match.
Matched engineering857-token/chunk512 runs (GR fusion ON in both variants)
reduce prefill time by1.24%/1.02% in both orders; all12 complete32-token outputs
match and output-pool reservations are unchanged. Other workloads are unproven.
Disabled by default; see `docs/performance.md` for exact scope and memory guards.

| Model family | Status |
|---|---|
| Qwen3 dense text models | FP16, W8, W4 |
| Qwen3-30B-A3B MoE | text-only W4 path |
| Qwen3.6-35B-A3B MoE | text-only W4 path |
| Qwen3.5-122B-A10B MoE | CPU W4 with SSD expert offload |
| Qwen3.8-Flash-Next | text-only CPU inference with native NVFP4 expert SSD offload |
| Tencent Hy-MT2-30B-A3B | text-only W4 MoE; CPU and Metal |
| Tencent Hy3-295B-A21B | text-only W4 MoE with CPU SSD expert offload |
| Qwen3.5-0.8B / Qwen3.5-4B | FP16, W8, W4, mixed W4; experimental single-image vision |
| Youtu-LLM-2B | FP16, W8, W4, mixed W4 |
| RWKV7 | FP16, W8, mixed W4; recurrent CPU prefill/decode |
| DeepSeek-V4-Flash | Experimental CPU inference with native FP8/MXFP4 and SSD expert offload |

The most tested runtime path today is `w4g128`: it has the lowest memory use and
the fastest decode speed in mollm. All config-based converters also accept
`w4g32` and `w4mixg32`. The smaller groups improve W4 quality at the cost of
more scales and potentially lower throughput; mixed modes reuse each model's
`w4mixg128` policy for deciding which sensitive tensors remain W8.

## Performance

### CPU

Apple M5 Pro results use four CPU threads, `pp256 + tg64`, `warmup=3`, and
independent-process medians unless noted.

Current W4 medians reach 140.77 decode tokens/s on Youtu-LLM-2B and 69.78
tokens/s on Qwen3.5-4B, respectively 1.47x and 1.73x the matched llama.cpp
Q4_0 CPU results. All dense and recurrent W4 rows in the chart use current
packages and binaries.

![CPU throughput comparison](assets/performance_cpu.svg)

### Metal (experimental)

Apple M5 Pro Metal results use the same `pp256 + tg64`, `warmup=3`, and
independent-process median protocol. Decode starts with an existing 256-token
context, and both runtimes keep model weights on Metal.

![Metal throughput comparison](assets/performance_metal.svg)

[Protocol, complete CPU and Metal tables, context scaling, and correctness gates](docs/performance.md)

## Why Decode Is Fast

- Highly optimized AArch64 GEMV kernels for FP16, W8, and W4 decode.
- Decode-friendly packed weight layouts, including direct W4G128 package
  layout.
- W4A8 decode aligns activation and weight groups at 128 values, accumulating
  four dot-product blocks in integer registers before applying their scale.
- Static reusable decode workspace to avoid per-token allocation churn.
- Prefill is still the main optimization target on dense models.

## Quick Start

```bash
cmake -G Ninja -B build_i8mm -DCMAKE_BUILD_TYPE=Release
cmake --build build_i8mm -j

# Needed for W4 conversion.
cmake --build build_i8mm --target mollm-quantize

# Convert a Hugging Face model directory.
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4g128.mollm w4g128

# Chat from the single package file.
./build_i8mm/mollm_chat --package qwen35_4b_w4g128.mollm --threads 4
```

```

                 _ _
 _ __ ___   ___ | | |_ __ ___
| '_ ` _ \ / _ \| | | '_ ` _ \
| | | | | | (_) | | | | | | | |
|_| |_| |_|\___/|_|_|_| |_| |_|

 model     : Qwen3-30B-A3B
 arch      : qwen3-moe
 layers    : 48
 hidden    : 2048
 quant     : w4g128
 ctx       : 16384
 threads   : 4

 /reset   clear context
 /quit    exit


>
```

Interactive chat commands:

```text
/reset   clear conversation context
/quit    exit
```

## Build

Requirements:

- macOS/Apple Silicon, ARM Linux, or Linux x86_64
- CMake + Ninja or Make
- Python 3
- Python packages needed by conversion, especially `numpy` and `safetensors`

Recommended local build:

```bash
cmake -G Ninja -B build_i8mm -DCMAKE_BUILD_TYPE=Release
cmake --build build_i8mm -j
```

On ARM64, a compiler with i8mm support embeds the optimized kernels as separate
objects. Runtime HWCAP/sysctl probing selects them only on a compatible CPU;
the rest of the library remains on the NEON/DOTPROD baseline.
`MOLLM_ARM_I8MM=OFF` omits the i8mm objects, while
`MOLLM_ARM_ISA=neon` forces the runtime fallback for testing. A plain `build/`
directory also works; replace `build_i8mm` in the examples with your build
directory.

## Convert Models

The converter auto-detects the model type from `config.json`.

```bash
# Default FP16 package.
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_fp16.mollm

# W8 int8 baseline.
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w8pc.mollm w8pc

# W4 performance package.
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4g128.mollm w4g128

# Mixed W4 quality package.
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4mixg128.mollm w4mixg128

# Smaller-group W4, pure or with the same model-specific mixed policy.
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4g32.mollm w4g32
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4mixg32.mollm w4mixg32

# RWKV7
python3 models/rwkv7.py /path/to/rwkv7-g1h-1.5b.pth rwkv7_1.5b_w4mixg32.mollm --tokenizer /path/to/tokenizer.txt --quant w4mixg32
```

MoE example:

```bash
python3 models/converter.py \
    /path/to/Qwen3-30B-A3B \
    qwen3_30b_a3b_w4g128.mollm \
    w4g128
```

Experimental DeepSeek-V4-Flash conversion preserves its checkpoint-native FP8
and MXFP4 weights:

```bash
python3 models/deepseek_v4.py \
    /path/to/DeepSeek-V4-Flash \
    deepseek_v4_flash_native.mollm

./build_i8mm/mollm_chat \
    --package deepseek_v4_flash_native.mollm \
    --ssd-cache-mb 10240 --ssd-io-workers 8 --threads 6
```

Qwen3.8-Flash-Next conversion preserves the checkpoint's NVFP4 routed experts,
quantizes dense weights to W4G32, and stores `lm_head` as per-channel W8:

```bash
python3 models/converter.py \
    /path/to/Qwen3.8-Flash-Next-NVFP4 \
    qwen38_flash_next_nvfp4.mollm

./build_i8mm/mollm_chat \
    --package qwen38_flash_next_nvfp4.mollm \
    --ssd-cache-mb 16384 --ssd-io-workers 8 --threads 4
```

Supported `config.json` model types:

| `model_type` | Supported models |
|---|---|
| `qwen3` | Qwen3 dense text models |
| `qwen3_moe` | Qwen3 MoE text models |
| `qwen3_5` | Qwen3.5 dense text and single-image vision models |
| `qwen3_5_moe` | Qwen3.5/3.6 MoE text models |
| `qwen4_exp` | Qwen3.8-Flash-Next text model with native NVFP4 experts |
| `hy_v3` | Tencent HY-V3 / Hy-MT2 MoE text models |
| `youtu` | Youtu-LLM MLA models |
| RWKV7 `.pth` | Use `models/rwkv7.py` directly. |
| `deepseek_v4` | Use the experimental `models/deepseek_v4.py` converter directly. |

Quantization choices:

| Mode | Use when |
|---|---|
| `fp16` | You want the simplest baseline and have enough memory. |
| `w8pc` | You want int8 weight-only quantization with small quality drift. |
| `w4g128` | You want the smallest package and fastest decode. This is the usual performance choice. |
| `w4mixg128` | Pure W4 quality is too low and you can spend more memory for selected W8 tensors. |
| `w4g32` | You want better W4 quality from smaller 32-value groups and accept extra scales and possible throughput loss. |
| `w4mixg32` | You want W4G32 plus the same model-specific W8 tensor promotions used by `w4mixg128`. |

Notes:

- DeepSeek-V4-Flash uses checkpoint-native FP8 E4M3 block-128 quantized
  dense/shared-expert weights and MXFP4 E2M1/E8M0 group-32 routed experts
  instead of the conversion modes above.
- Qwen3.8-Flash-Next uses checkpoint-native NVFP4 routed experts, offline
  W4G32 dense weights, and a per-channel W8 `lm_head`.
- W4 conversion requires the `mollm-quantize` helper built from C++.
- FP16 and W8 conversion do not require that helper.
- The prefill graph is built with an internal 256-token chunk size, but CPU
  runtime prefill is dynamic: short prompts are not padded to 256 unless you
  explicitly pass `--static-padded`.
- `--prefill-chunk-size N` overrides that runtime chunk limit; `0` keeps
  the package default. Increasing it requires a dynamic hidden SEQ dimension
  and cannot be combined with static padding. This is experimental: changing
  batching can change numerics, and larger chunks are not necessarily faster.
  Qwen3.8 CUDA batches above 4096 routed jobs currently leave the grouped
  NVFP4 path (512 tokens with top-k10 exceeds that limit). No reconversion
  or weight-layout change is needed for this option.
  The 384-token setting sped up two measured Qwen3.8 prompts, but changed
  generated tokens on one of them; it is not a generally equivalent default.
  Experimental `MOLLM_CUDA_NVFP4_PREFILL_TILED_GROUP=1` removes the4096-job
  grouping boundary for explicit larger chunks (up to65535 jobs). It is off
  by default, keeps the small-chunk/decode paths unchanged, and requires
  separate numerical validation when combined with a new chunk size.
  Experimental `MOLLM_CUDA_NVFP4_MMA_COMPACT_BATCHES=1` uses a GPU-only
  compact batch plan for resident batched NVFP4 prefill. It preserves route
  output indices and supports graph capture without a host-count readback.
  It is off by default; performance validation is still in progress.
  Experimental `MOLLM_CUDA_NVFP4_PREFILL_RADIX_GROUP=1` replaces resident
  prefill's quadratic grouping with stable GPU radix sorting of full weight
  addresses (up to 65535 routes). Equal-key route order and output indices
  are preserved, and caller-owned scratch supports CUDA Graph capture. This
  flag also permits large-route grouping without the tiled-group flag. It is
  off by default. The measured 857-token prompt at chunk1024 showed no
  meaningful end-to-end speedup; this is not a recommended speed setting.
  Experimental `MOLLM_CUDA_NVFP4_MMA_BATCH_ROUTES=16` raises resident
  batched prefill's expert batch width from the default 8 to 16. It reuses
  weight fragments across two N8 MMAs with the existing prequantized inputs;
  output route positions, weight layout and calibration tables are unchanged.
  One 857-token prompt showed 0.76–1.21% shorter prefill with matching generated
  token IDs; broader performance and fixed-history accuracy checks are pending.
  Experimental `MOLLM_CUDA_NVFP4_MMA_PREPARED_SPECIALIZED=1` uses the same
  prepared implementation at width 8 for implementation-isolation testing;
  it is off by default, and width 16 always uses that implementation.
  Experimental `MOLLM_CUDA_NVFP4_MMA_GATE_BATCH_ROUTES=32` overrides only
  batched prefill gate/up width; down/decode remain unchanged. Values 8/16/32
  are accepted; unset/invalid inherits the shared batch width. On one 857-token
  prompt the reverse-order comparison showed 0.82% shorter prefill; the first
  order was noisier. Two same-chunk fixed-history loss CSVs were byte-identical
  to N8. No reconversion needed; not a general throughput/quality guarantee.
  Experimental `MOLLM_CUDA_HC_PREFILL_ACTIVATION_REUSE=1` shares the FP16
  input conversion between paired HC prefill projections, preserving both
  original GEMMs. It is off by default; eligible contiguous inputs require
  at least 16 rows. One 857-token prompt at chunk1024 showed 2.44–2.49%
  shorter prefill, with no stable total/decode gain. Two same-chunk fixed-input
  loss CSVs were byte-identical. Shared scratch capacity was 32 MiB; growth
  retains earlier buffers until teardown. No weight changes or reconversion;
  other backends are unchanged. Broader prompts/lengths remain unvalidated.
  Experimental `MOLLM_CUDA_HC_PREFILL_RMS_HALF=1`, with activation reuse
  enabled, produces that FP16 input directly in Group RMS while preserving
  its FP32 output. It is off by default and uses the same shared scratch.
  On the same 857-token/chunk1024 fixture it adds 1.28–1.37% shorter
  prefill, but no stable total/decode gain; both tested same-chunk loss CSVs
  were byte-identical. No reconversion or changes to other backends.
  Experimental `MOLLM_CUDA_NVFP4_ACTIVATION_PREPARE_FUSION=2` fuses routed
  activation with NVFP4 input quantization for batched MMA prefill (16+ rows).
  It omits FP32 activation writes only when down minimum is 1 or singleton
  preparation is enabled; otherwise it preserves the FP32 fallback buffer.
  Default is off; value 1 always preserves that buffer and showed no stable
  model benefit. Value 2 showed 0.97–1.15% shorter prefill on one 857-token,
  chunk1024 prompt. Four same-chunk loss pairs, including both fallback
  configurations, were byte-identical. Decode, weight formats and other
  backends are unchanged; no reconversion or broad quality guarantee.
  Experimental `MOLLM_CUDA_NVFP4_PREFILL_BF16_DOWN_OUTPUT=1` stores routed
  down outputs as BF16 codes at the existing combine rounding boundary.
  Default off; resident canonical NVFP4 grouped prefill only,16+ rows,
  hidden2560/inter640,N8/minimum1/four warps,no compact/specialized down or
  singleton preparation. Unsupported configurations keep FLOAT producer
  and consumer together. Existing scratch is reused; no extra weight cache,
  reconversion or changes to decode/other backends. On one1743-token,
  chunk1024/ctx4096 fixture,matched order tests improved prefill TPS
  2.01–2.30% (combined1113→1137TPS); all1742 fixed-history token losses
  and12 full generated token arrays were byte-identical. No decode gain
  or broad quality guarantee.
  Experimental `MOLLM_CUDA_NVFP4_PREFILL_BF16_GATE_OUTPUT=1` similarly
  stores gate/up at the existing activation BF16 boundary. Default off;
  resident canonical NVFP4 prefill16+ rows,hidden2560/inter640 requires
  async N32 gate/minimum1/four warps,no compact/specialized/singleton
  preparation,and activation preparation fusion1/2. Producer and consumer
  switch together; unsupported selections retain FLOAT. Independent of
  BF16 down output; no extra cache,scratch allocation,weight conversion,
  decode or other-backend change. Same1743-token/chunk1024/ctx4096 fixture
  added0.99–1.20% prefill TPS (combined1136→1149TPS); all1742 fixed-history
  losses and12 complete generated token arrays matched. One driver internal
  allocation warning occurred in the first OFF run; no application failure,
  Xid or system OOM-kill, but its cause/timing effect is not isolated.
  See performance notes; this is not broad quality or a clean driver-log claim.
  Experimental `MOLLM_CUDA_NVFP4_PREFILL_EARLY_INPUTS=1` moves gate
  input-fragment loads before the async weight wait. Default off; requires
  the supported BF16 async N32 gate and all resident gate descriptors using
  paired layout. Other selections retain the original path; no extra cache,
  scratch,weight conversion,decode or other-backend change. Same1743-token/
  chunk1024/ctx4096 fixture adds0.49–0.63% prefill TPS (combined1171→1178);
  all1742 fixed-history losses and12 complete generated arrays match.
  This is not broad quality proof or a decode-speedup claim.
  Experimental `MOLLM_CUDA_PREFILL_LT_GEMM=1` caches advertised zero-workspace,
  split1 cuBLASLt candidates for SM121 FP16→FP32 prefill at K2560,N13312/16480,
  M512..1024. Default off. Unsupported type/shape/alignment,missing candidate
  or capture-time cache miss keeps original cuBLAS. Decode/strict FP32 remain
  unchanged; no weight layout/cache or conversion change. Same1743-token/
  chunk1024/ctx4096 fixture adds0.42–0.47% prefill TPS (combined1177→1182),
  with1742 fixed-history losses and12 complete generated arrays matching.
  This is not broad quality proof or a decode-speedup guarantee.
  Experimental `MOLLM_CUDA_NVFP4_PREFILL_PARALLEL_PLAN=1` uses warp
  ballots for sparse batch-plan metadata. Default off; canonical resident
  batched prefill without compact/specialized/singleton preparation only.
  MMA/quantization/accumulation/BF16 boundaries unchanged; no extra workspace,
  weight cache,conversion,decode or other-backend change. Same1743-token/
  chunk1024/ctx4096 fixture adds1.55–1.66% prefill TPS (combined1184→1203),
  with1742 fixed-history losses and12 complete generated arrays matching.
  Not broad quality or decode-speedup proof,or permission to raise chunk/context.
  Experimental `MOLLM_CUDA_NVFP4_PREFILL_VECTOR_QUANTIZE=1` vectorizes
  gate input quantization only for the supported resident BF16 async gate
  prefill path (16..4096 rows). Default off; requires the BF16 gate selector.
  Caches16 input values per thread,retains sequential maximum/FP8/FP4 math,
  uses aligned float4 loads and packed32-bit stores,scalar I/O fallback when
  misaligned. No weight/cache/workspace/layout/reconversion or decode change.
  Same1743-token/chunk1024/ctx4096 fixture improves prefill1.85–1.86%
  (combined1202.66→1224.96 TPS);1742 losses and12 generated arrays match.
  Fixture-specific evidence,not broad quality or permission to raise chunk.
  Experimental `MOLLM_CUDA_SDPA_PREFILL_VALIDATED_PREFIX=1` skips future
  QK work for cached FP16 D256 prefill only after GPU validation of the
  explicit causal mask and Q/K bounds. Failed validation retains full QK;
  softmax/PV order is unchanged. It is off by default. One 857-token,
  chunk1024 prompt showed 1.36–2.70% shorter prefill, but no stable
  total/decode benefit. Both tested fixed-history loss CSVs were
  byte-identical; this is not a broad quality guarantee. No reconversion
  or changes to decode, weight formats or other backends.
  With validated-prefix enabled, experimental
  `MOLLM_CUDA_SDPA_PREFILL_SKIP_ZERO_PV=1` also omits future zero-weight
  PV terms after finite-V validation. Zero/subnormal partial results retain
  the original trailing FMAs; failed validation retains full computation.
  Default is off. One 857-token/chunk1024 prompt showed 2.73–2.80% shorter
  prefill versus validated-prefix alone in both test orders.
  A 1743-token/chunk2048 fixture showed 8.20–8.26% shorter prefill and
  2.80–2.81% shorter total latency for 128 generated tokens in both orders.
  Four fixed-history loss pairs were byte-identical, including long global/shared-score paths.
  This is not a general quality/decode guarantee; no weight reconversion.
  With validated-prefix and zero-PV enabled, experimental
  `MOLLM_CUDA_SDPA_PREFILL_PV_UNROLL=1` unrolls the ordered PV loop only
  for global-score paths (keys above the selected shared-score limit).
  Default is off; shared-score paths and decode are unchanged. On GB10,
  resident+PLE-offload with one 1743-token/chunk2048 fixture showed
  1.35–1.50% shorter prefill and 0.17–0.65% shorter total latency for
  128 generated tokens in both orders. Two long same-chunk loss pairs
  (1024/2048) were byte-identical; a sanity256 inactive-path loss CSV
  matched its reference. This is not a general quality/decode guarantee;
  no weight reconversion or chunk-default change.
  At chunk1024, the same long fixture showed 1.31–1.40% shorter prefill
  in both orders, but total-latency results were directionally inconsistent.
  Experimental `MOLLM_CUDA_SDPA_PREFILL_PV_PAIR=1` pairs adjacent V
  dimensions using half2 loads and independent ordered FP32 FMA chains.
  Requires validated-prefix and zero-PV; default is off. It supersedes
  PV-unroll when both are enabled, only for global-score D256 cached prefill
  (keys above the selected shared-score limit). Shared-score paths and
  decode are unchanged. The initial shared-score model trial did not give
  reproducible throughput gains despite identical output. One long1743/
  chunk1024 fixed-history CSV matched all1742 reference losses byte-for-byte.
  On one resident+PLE-offload long1743/chunk1024 fixture, prefill was
  2.03–2.25% shorter and total latency for32 greedy tokens1.03–1.56% shorter
  in both orders versus retained global PV-unroll4. All12 full generated
  arrays matched; no stable decode gain or general quality guarantee.
  No weight reconversion or chunk change.
  With validated-prefix enabled, optional
  `MOLLM_CUDA_SDPA_PREFILL_SHARED_SCORE_LIMIT=4096` extends shared score
  storage beyond the default 1024-key threshold. One 1743-token/chunk2048
  fixture showed 0.76–1.05% shorter prefill, with no stable total/decode
  benefit. Two same-chunk fixed-history loss comparisons were byte-identical;
  broader quality is unvalidated. Global scratch remains allocated.
  With zero-PV and quad GDN enabled, the same long fixture instead showed
  1.01–1.08% slower prefill at 4096 in both orders; keep 1024 for that
  tested combination. Isolated attention timing gains did not transfer.
  Experimental `MOLLM_CUDA_GDN_PREFILL_PAIRED_VALUES=1` lets each GDN
  recurrence warp process two independent value columns, sharing Q/K loads
  while preserving each column's arithmetic order. Default is off; requires
  sequence length at least 16 and aligned state. Decode, short/serial prefill
  and butterfly-key mode retain their original paths. One 857-token/chunk1024
  prompt showed 1.12–1.43% shorter prefill in both orders, with no robust
  total/decode benefit. Three fixed-history loss pairs were byte-identical,
  including a 1743-token input; broader quality is unvalidated. No extra
  allocation, weight format changes or reconversion.
  Experimental `MOLLM_CUDA_GDN_PREFILL_QUAD_VALUES=1` processes four
  independent value columns per warp, preserving per-column arithmetic.
  Default is off; prepared prefill only, at least 16 tokens, 16-byte-aligned
  state and no butterfly-key mode. It takes precedence over paired values
  when supported; otherwise existing fallback selection remains. Decode
  and snapshots are unchanged. Against paired-values enabled, one
  857-token/chunk1024 fixture showed 1.89–1.90% shorter prefill in both
  orders, only 0.18–0.25% shorter total time; no general decode gain.
  All 12 generated sequences match and three complete fixed-history loss
  pairs (364/856/1742 targets) are byte-identical. Broader quality is
  unvalidated. No extra allocation, weight format changes or reconversion.
  A 1743-token/chunk2048 fixture also showed 3.01–3.05% shorter prefill
  and 0.88–1.00% shorter total time in both orders, with all 12 full
  generated sequences matching; this does not establish a decode gain.
  Experimental `MOLLM_CUDA_SIGMOID_GR_REDUCE_FUSION=1` fuses adjacent,
  single-consumer HC gate sigmoid and stream reduction for CUDA prefill.
  It avoids the intermediate gate allocation; decode, debug/profiling and
  partial execution retain the original path. Weight/package formats are
  unchanged. One 857-token prompt showed 3.45–4.35% shorter prefill with
  matching generated IDs; a 364-target fixed-history CE comparison is
  byte-identical. Broader prompts/lengths remain to be validated.
  At chunk256 the same prompt showed a smaller 1.24–1.40% prefill gain;
  the two-chunk 364-target CE comparison is also byte-identical.
  Experimental `MOLLM_CUDA_SHORTCONV_TRANSPOSE=1` coalesces CUDA prefill
  short-convolution output stores through a small shared-memory tile for
  unit feature-stride input. It is off by default; decode and history-prefix
  snapshots retain their original paths. One 857-token prompt at chunk1024
  showed 0.74–0.81% shorter prefill with identical generated IDs, but no
  consistent total-latency improvement. A separate two-chunk 364-target
  fixed-history loss CSV is byte-identical. No weight reconversion is needed.
- Converted packages store a user-facing model name from the Hugging Face config
  or model directory, so chat displays names such as `Qwen3-30B-A3B`.

## Run Chat

```bash
./build_i8mm/mollm_chat --package qwen35_4b_w4g128.mollm --threads 4
```

One-shot deterministic smoke test:

```bash
./build_i8mm/mollm_chat \
    --package qwen35_4b_w4g128.mollm \
    --prompt "请只输出一句话，不要解释：杭州有什么特点？" \
    --max-new-tokens 64 \
    --threads 4 \
    --temperature 0
```

Experimental Qwen3.5 single-image chat (macOS, CPU vision encoder):

```bash
./build_i8mm/mollm_chat \
    --package qwen35_0.8b_w4g128.mollm \
    --image photo.png \
    --prompt "Describe this image." \
    --max-new-tokens 128
```

The converter packages the checkpoint's FP16 vision tower alongside the
quantized text model. The first implementation supports one static image in
single-shot chat; multi-image/video input, server integration, and Metal vision
execution are not enabled yet. Images default to a 262,144-pixel resize budget
(roughly 512x512); use `--image-max-pixels` to trade speed for detail, up to
1,048,576 pixels. Use converter option `--text-only` when the vision tower is
not needed, avoiding its package and resident-memory overhead.

Sampled generation:

```bash
./build_i8mm/mollm_chat \
    --package qwen35_4b_w4g128.mollm \
    --temperature 0.8 --top-p 0.95 --top-k 40 --min-p 0.05 \
    --repeat-penalty 1.05 --repeat-last-n 128 --seed 42
```

OpenAI-style `--presence-penalty` and `--frequency-penalty` are also
available. Run `mollm_chat --help` for ranges and defaults.

MoE chat:

```bash
./build_i8mm/mollm_chat --package qwen3_30b_a3b_w4g128.mollm --threads 4
```

By default, `mollm_chat` loads package weights into resident memory. For mmap
A/B testing, pass `--mmap`; mmap page warmup is enabled unless you also pass
`--no-load-warmup`.

## Benchmark

Standard mollm benchmark:

```bash
./build_i8mm/mollm_bench \
    --package qwen35_4b_w4g128.mollm \
    --prompt-tokens 256 \
    --max-new-tokens 64 \
    --warmup 3 \
    --threads 4
```

## CUDA correctness backend

CUDA support can be built independently of the CPU provider:

```bash
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DMOLLM_CUDA=ON
cmake --build build_cuda -j
./build_cuda/mollm_chat --device cuda --package model.mollm
```

Explicitly selecting CUDA now requires a working CUDA backend instead of
silently falling back to CPU. Add `--require-native` to reject any operator
that would use the CPU reference implementation.

The CUDA backend is still a correctness-first implementation. Graph outputs
and persistent state use device-addressable managed storage, FP16/FP32 linear
layers run through cuBLAS. W8 weights remain quantized on device: decode uses
a native GEMV kernel, while prefill dequantizes one matrix at a time into
reusable FP16 scratch before cuBLAS. Package-native W4G32/W4G128 weights use
the same policy with packed decode kernels and reusable prefill scratch.
RMSNorm and fused norm paths,
dense elementwise operations, common activations, SwiGLU, RoPE, strided layout
materialization, FP32 cached SDPA, zero-copy views, and the decode lm_head also
stay on CUDA. Operators not yet implemented natively synchronize and use the
CPU reference dispatcher over the managed buffers. Set
`MOLLM_CUDA_PROFILE=1` to print native/fallback operator counts. Recurrent
Gated DeltaNet/short-convolution operators and several specialized model
families still fall back, so this is not yet a performance-complete CUDA
backend.

## Local HTTP server

```bash
./build_i8mm/mollm_server \
    --package qwen35_4b_w4g128.mollm \
    --host 127.0.0.1 --port 8080 --threads 4
```

The initial server implements `GET /v1/models` and OpenAI-compatible
`POST /v1/chat/completions`, including SSE streaming. It also retains a
single exact token-prefix KV cache between serialized requests. Sampling
parameters can be set per request; the default remains deterministic
(`temperature=0`). See [SERVER.md](SERVER.md) for fields, examples, and
limitations.

## Project Layout

```text
mollm/
├── kernels/    ARM kernels for matmul, attention, MoE, norm, rope
├── graph/      Graph format, executor, mmap package loading, BufferPool
├── engine/     LLMEngine, tokenizer, chat/generation lifecycle
├── models/     Python converters and graph builders
├── examples/   mollm_chat, mollm_server, mollm_bench, mollm_ppl
└── tests/      Unit, stress, and end-to-end tests
```

## Roadmap

- Prefill performance optimization, especially for W8/W4 dense-model prompt
  processing.
- Improve experimental Metal performance, especially quantized prefill, MoE
  prefill, and CPU/GPU synchronization overhead.
- Full prefix caching for serving workloads, building on the current single-user
  REPL cache.
- Broader accelerator coverage while keeping the CPU runtime as the portable
  baseline.
- More model families beyond the current Qwen, Youtu, and Qwen-MoE coverage.
- Broader vision support beyond the current Qwen3.5 single-image path,
  including multi-image/video input, serving, and Metal execution.
- Speculative decoding for DeepSeek-V4, starting with native MTP and exploring
  [DSpark](https://arxiv.org/abs/2607.05147)-style confidence-scheduled
  semi-autoregressive drafting.
- Keep optimizing SSD offload.

## License

Copyright 2026 Tencent. Licensed under the Apache License 2.0. See
[LICENSE](LICENSE) and [NOTICE](NOTICE). Bundled dependency notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Acknowledgments

- [Cider](https://github.com/Mininglamp-AI/cider), whose work on W8A8/W4A8
  inference with Metal 4 INT8 TensorOps on Apple Silicon informed mollm's
  experimental quantized Metal path.
- Fang et al.'s [Fate](https://arxiv.org/abs/2502.12224), whose cross-layer
  gate prediction inspired mollm's experimental expert-prefetch path.
