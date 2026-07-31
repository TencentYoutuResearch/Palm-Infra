# palm-infra

AI Infra projects from Palm Team. Currently includes `mollm`.

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

## Now it runs a 122B model on a 48GB Mac (Or even 16GB)

`mollm` can run Qwen3.5-122B-A10B W4 on a 48GB Apple Silicon Mac by keeping
dense weights in RAM and fetching only routed MoE experts from SSD. In the
current 256-token cache sweep, a bounded, shared 16 GiB expert cache and
cross-layer prefetching provide 16.53 t/s interactive decode.

The cache is configurable rather than tied to a resident copy of the model. In
the following real-prompt sweep, the 1 GiB expert-cache configuration runs with
only 5.90 GiB peak RSS; larger caches trade memory for fewer SSD reads and
higher throughput.

| Expert RAM cache | Decode | Peak RSS | Cache hit rate | Avg. SSD reads / generated token |
|---:|---:|---:|---:|---:|
| **1 GiB** | 12.38 t/s | **5.90 GiB** | 47.9% | 1.72 GB/token |
| **10 GiB** | 16.19 t/s | 14.64 GiB | 83.5% | 0.75 GB/token |
| **16 GiB** | **16.53 t/s** | 20.60 GiB | **88.6%** | **0.51 GB/token** |

This sweep uses a 16-token real prompt, 256 generated tokens, greedy decoding,
`warmup=0`, and three independent process runs per cache size. The rows were
rerun on 2026-07-29. The 10 GiB cache is within 2.1% of the 16 GiB decode
throughput while using about 6 GiB less peak RSS; 1 GiB demonstrates the
low-memory operating point. SSD reads are logical routed-expert bytes loaded
by demand or prefetch, divided by generated tokens; prompt prefill is included
in the amortized value, while dense-weight and CPU-sidecar loading is excluded.

See [Running 122B MoE models with SSD offload](docs/ssd-offload.md) for cache
policy, memory/throughput sweeps, I/O behavior, and Perfetto tracing.

## Experimental DeepSeek-V4-Flash support

`mollm` can convert the native DeepSeek-V4-Flash checkpoint without first
expanding it to FP16. Quantized dense and shared-expert weights retain their
native FP8 E4M3 data, while routed experts retain OCP MXFP4: packed E2M1 values
with one E8M0 scale per 32 values. The 157 GB package uses the same bounded,
asynchronous SSD expert offload path as other large MoE models.

Current CPU-only results on an Apple M5 Pro:

| Standard throughput | Expert RAM cache | Result |
|---|---:|---:|
| `pp256 + tg64` | 16 GiB | **9.35 pp / 4.74 tg** |

This standard sample uses four CPU threads and `warmup=3`. It is one completed
process run rather than the usual five-process median, so it is provisional.

| Expert RAM cache | Decode | Peak RSS | Avg. SSD reads / generated token |
|---:|---:|---:|---:|
| **1 GiB** | 4.37 t/s | **19.09 GiB** | 3.73 GB/token |
| **10 GiB** | **4.73 t/s** | **24.32 GiB** | 1.89 GB/token |
| **16 GiB** | 4.67 t/s | 26.91 GiB | **1.55 GB/token** |

This experimental sweep uses a 19-token real Chinese prompt, 64 generated
tokens, greedy decoding, six CPU threads, `warmup=0`, and three independent
processes per cache size. These real-prompt numbers are reported separately
from the standard `pp256 + tg64` performance chart. SSD reads use the same
amortized logical-expert-byte definition as the 122B table above.

## What Works

| Model family | Status |
|---|---|
| Qwen3 dense text models | FP16, W8, W4 |
| Qwen3-30B-A3B MoE | text-only W4 path |
| Qwen3.6-35B-A3B MoE | text-only W4 path |
| Qwen3.5-122B-A10B MoE | CPU W4 with SSD expert offload |
| Qwen3.5-0.8B / Qwen3.5-4B | FP16, W8, W4, mixed W4; experimental single-image vision |
| Youtu-LLM-2B | FP16, W8, W4, mixed W4 |
| RWKV7 | Official `.pth` conversion; FP16, W8, mixed W4; recurrent CPU/CUDA prefill and decode |
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

Supported `config.json` model types:

| `model_type` | Supported models |
|---|---|
| `qwen3` | Qwen3 dense text models |
| `qwen3_moe` | Qwen3 MoE text models |
| `qwen3_5` | Qwen3.5 dense text and single-image vision models |
| `qwen3_5_moe` | Qwen3.5/3.6 MoE text models |
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
- W4 conversion requires the `mollm-quantize` helper built from C++.
- FP16 and W8 conversion do not require that helper.
- The prefill graph is built with an internal 256-token chunk size, but CPU
  runtime prefill is dynamic: short prompts are not padded to 256 unless you
  explicitly pass `--static-padded`.
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

Experimental Qwen3.5 single-image chat (macOS image-file decoding):

```bash
./build_i8mm/mollm_chat \
    --package qwen35_0.8b_w4g128.mollm \
    --image photo.png \
    --prompt "Describe this image." \
    --max-new-tokens 128
```

The converter packages the checkpoint's FP16 vision tower alongside the
quantized text model. The first implementation supports one static image in
single-shot chat. With `--device cuda`, both the vision encoder and subsequent
text graph stay on CUDA; image-file decoding itself is currently macOS-only.
Linux callers can use the preprocessed `encode_vision_patches` API directly.
Multi-image/video input, server integration, and Metal vision execution are not
enabled yet. Images default to a 262,144-pixel resize budget (roughly 512x512);
use `--image-max-pixels` to trade speed for detail, up to 1,048,576 pixels. Use
converter option `--text-only` when the vision tower is not needed, avoiding
its package and resident-memory overhead.

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

The CUDA backend is still a correctness-first implementation. Graph outputs
and persistent state use device-addressable managed storage, FP16/FP32 linear
layers run through cuBLAS, and ordinary package-native W8, W4G32, and W4G128
weights are dequantized to FP16 device weights at load time. RMSNorm, fused
norm paths, strided and broadcast elementwise operations, common activations,
SwiGLU, RoPE, layout materialization, tile/concat, FP32 cached SDPA, zero-copy
views, and the decode lm_head also stay on CUDA. Qwen3.5's recurrent Gated
DeltaNet and short-convolution paths are native as well, including persistent
decode-state updates. RWKV7 token shift, channel mixing, normalization, WKV7
recurrence, post-processing, batched projections, and sparse-activation FFN
also run natively, with persistent FP16 or FP32 recurrent state. Qwen3.5's
single-image vision tower uses the same native matmul, LayerNorm, RoPE, SDPA,
GELU, and layout paths. Resident Qwen3-style MoE graphs support softmax,
sigmoid, grouped, correction-bias, and INT32 token-hash routing plus optional
shared experts. Hash lookup tables remain device-resident and retain their
package-native INT32 representation.
Aggregate W8, W4G32, and W4G128 expert tensors remain quantized on the GPU and
are decoded inside the selected-expert kernels instead of being expanded into
model-sized FP16 copies. W8 experts retain row-major values and per-group
scales, while W4 experts retain their package-native BG32/BG128 blocks.
Checkpoint-native FP8 E4M3 block-128 dense/shared-expert weights and resident
MXFP4 E2M1/E8M0 group-32 routed experts likewise stay byte-packed on the GPU.
Their CUDA paths reproduce the checkpoint's FP8 activation quantization and
the BF16 boundaries around MXFP4 MoE gate/up, routed scaling, and down output.
This
keeps the dense Qwen3, Qwen3.5, Youtu MLA, RWKV7, Qwen3.5 vision, Qwen3-MoE,
and Qwen3.5-MoE graphs on CUDA without CPU operator fallback. DeepSeek-V4's
FP32 Hyper-Connection stages, checkpoint-native dense matmuls, and resident
FP8/MXFP4 hash-routed MoE operator are native, but its complete graph remains
CPU-only because grouped projections and sparse-attention coverage are still
incomplete. SSD-offloaded MoE variants also remain CPU-only.
Other operators not yet implemented natively synchronize and use the CPU
reference dispatcher over the managed buffers. Set `MOLLM_CUDA_PROFILE=1` to
print native/fallback operator counts. Several specialized model families
still fall back, so this is not yet a performance-complete CUDA backend.

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
