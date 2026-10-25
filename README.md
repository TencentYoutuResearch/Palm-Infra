# palm-infra

AI Infra projects from [PalmAI](https://palm.tencent.com/) Team.

[中文文档](README_zh.md)

## mollm

**A small, portable C++ inference engine that runs large models on the hardware you already have.**

```text
                 _ _
 _ __ ___   ___ | | |_ __ ___
| '_ ` _ \ / _ \| | | '_ ` _ \
| | | | | | (_) | | | | | | | |
|_| |_| |_|\___/|_|_|_| |_| |_|
```

[![CI](https://github.com/TencentYoutuResearch/palm-infra/actions/workflows/ci.yml/badge.svg)](https://github.com/TencentYoutuResearch/palm-infra/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-ARM64%20%7C%20x86--64%20%7C%20Metal%20%7C%20CUDA-informational)](#build)

`mollm` converts a supported Hugging Face checkpoint into one `.mollm` file
containing the graph, weights, tokenizer, and chat template, then runs that
package directly. There is no Python runtime, no serving framework, and no
per-model glue code at inference time.

It targets fast local inference on Apple Silicon, ARM and x86 CPUs, with
optional Apple Metal and NVIDIA CUDA backends. Decode kernels are hand-tuned
for NEON/i8mm, AVX2/FMA/F16C, and AVX-512; quantized models use weight-only
W8/W4 layouts designed for direct execution from the package.

### Highlights

- **One file per model.** Graph, weights, tokenizer, and template in a single
  mmap-friendly package.
- **Big models, one machine.** SSD expert offload keeps dense weights in RAM
  and streams only routed experts through a bounded shared cache — 100B+ MoE
  models run on a laptop or a single workstation.
- **Fast decode first.** Packed W4/W8 layouts, AArch64 GEMV kernels, static
  decode workspaces, and runtime ISA dispatch.
- **Portable by design.** The CPU backend is the baseline; Metal and CUDA are
  opt-in accelerators. Unsupported CUDA operators fall back explicitly, and
  `--require-native` turns any fallback into an error.
- **Long context on CUDA.** Qwen3.8-Flash-Next runs with native QSA sparse
  attention and selected-key indexing up to 262,144 tokens.
- **Batteries included.** Interactive chat, benchmark, perplexity, and an
  OpenAI-compatible HTTP server.

> `mollm` is a research project. Interfaces and package formats may change;
> experimental paths are marked below and in the linked docs.

## Quick Start

```bash
# Build the CPU runtime.
cmake -G Ninja -B build_i8mm -DCMAKE_BUILD_TYPE=Release
cmake --build build_i8mm -j

# The W4 converter helper.
cmake --build build_i8mm --target mollm-quantize

# Convert a Hugging Face checkpoint into one package.
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4g128.mollm w4g128

# Chat.
./build_i8mm/mollm_chat --package qwen35_4b_w4g128.mollm --threads 4
```

```text
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

## Supported Models

| Model family | Status |
|---|---|
| Qwen3 dense text models | FP16, W8, W4 |
| Qwen3-30B-A3B MoE | text-only W4 path |
| Qwen3.6-35B-A3B MoE | text-only W4 path |
| Qwen3.5-122B-A10B MoE | CPU W4 with SSD expert offload |
| Qwen3.8-Flash-Next | CPU NVFP4 expert SSD offload; experimental CUDA with native QSA |
| Tencent Hy-MT2-30B-A3B | text-only W4 MoE; CPU and Metal |
| Tencent Hy3-295B-A21B | text-only W4 MoE with CPU SSD expert offload |
| Hy4-preview (770B MoE) | experimental CPU GGUF/FP16 with SSD expert paging |
| Qwen3.5-0.8B / Qwen3.5-4B | FP16, W8, W4, mixed W4; experimental single-image vision |
| Youtu-LLM-2B | FP16, W8, W4, mixed W4 |
| RWKV7 | FP16, W8, mixed W4; recurrent CPU prefill/decode |
| DeepSeek-V4-Flash | experimental CPU inference with native FP8/MXFP4 and SSD expert offload |

The most tested runtime path is `w4g128`: it has the lowest memory use and the
fastest decode speed. All config-based converters also accept `w4g32` and
`w4mixg32`. Smaller groups improve W4 quality at the cost of more scales and
potentially lower throughput; mixed modes reuse each model's `w4mixg128`
policy for deciding which sensitive tensors remain W8.

## Performance

### Large MoE models with SSD offload

Dense weights stay in RAM; only routed experts are read from SSD into a
bounded shared cache. The same asynchronous path supports W4, checkpoint-native
FP8/MXFP4, and NVFP4 experts.

| Model | Expert cache | Prefill | Decode |
|---|---:|---:|---:|
| Qwen3.5-122B-A10B W4 | 16 GiB | **51.06 t/s** | **16.53 t/s** |
| Qwen3.8-Flash-Next NVFP4 | 20 GiB | **16.2 t/s** | **21.8 t/s** † |
| DeepSeek-V4-Flash | 16 GiB | **9.52 t/s** | **5.71 t/s** |
| Hy3-295B-A21B W4G128 | 16 GiB | **10.57 t/s** | **3.41 t/s** |

† Best observed real-prompt run with eight CPU threads and a 20 GiB expert
cache; the other rows use their documented four-thread / 16 GiB protocols.

[Cache sweeps, I/O behavior, and tracing](docs/ssd-offload.md)

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

[Protocols, complete tables, context scaling, CUDA measurements, and
correctness gates](docs/performance.md)

### Why decode is fast

- Highly optimized AArch64 GEMV kernels for FP16, W8, and W4 decode.
- Decode-friendly packed weight layouts, including a direct W4G128 package
  layout.
- W4A8 decode aligns activation and weight groups at 128 values, accumulating
  four dot-product blocks in integer registers before applying their scale.
- Static reusable decode workspace to avoid per-token allocation churn.
- Prefill is still the main optimization target on dense models.

## Build

Requirements:

- macOS/Apple Silicon, ARM Linux, or Linux x86_64
- CMake + Ninja or Make
- Python 3 with `numpy` and `safetensors` for conversion

Recommended local build:

```bash
cmake -G Ninja -B build_i8mm -DCMAKE_BUILD_TYPE=Release
cmake --build build_i8mm -j
```

On ARM64, a compiler with i8mm support embeds the optimized kernels as separate
objects. Runtime HWCAP/sysctl probing selects them only on a compatible CPU;
the rest of the library remains on the NEON/DOTPROD baseline.
`MOLLM_ARM_I8MM=OFF` omits the i8mm objects, while `MOLLM_ARM_ISA=neon` forces
the runtime fallback for testing. A plain `build/` directory also works;
replace `build_i8mm` in the examples with your build directory.

On x86_64, separately compiled scalar, AVX2/FMA/F16C, and AVX-512 providers are
selected by CPUID at startup. Set
`MOLLM_X86_ISA=scalar|avx2|avx512|auto` to cap the selected tier; the legacy
`MOLLM_X86_DISABLE_AVX2=1` override remains supported.

Optional backends:

```bash
# Apple Metal
cmake -G Ninja -B build_metal -DCMAKE_BUILD_TYPE=Release -DMOLLM_METAL=ON

# NVIDIA CUDA
cmake -G Ninja -B build_cuda -DCMAKE_BUILD_TYPE=Release -DMOLLM_CUDA=ON
```

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

Experimental DeepSeek-V4-Flash conversion preserves checkpoint-native FP8 and
MXFP4 weights:

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
| `hy_v4` | Hy4-preview MoE (native GGUF or FP16) |
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
  explicitly pass `--static-padded`. `--prefill-chunk-size N` overrides the
  runtime chunk limit (`0` keeps the package default); changing batching can
  change numerics and is not necessarily faster.
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

Sampled generation:

```bash
./build_i8mm/mollm_chat \
    --package qwen35_4b_w4g128.mollm \
    --temperature 0.8 --top-p 0.95 --top-k 40 --min-p 0.05 \
    --repeat-penalty 1.05 --repeat-last-n 128 --seed 42
```

OpenAI-style `--presence-penalty` and `--frequency-penalty` are also
available. Run `mollm_chat --help` for ranges and defaults.

Qwen3.5 single-image chat (macOS, CPU vision encoder):

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

MoE chat:

```bash
./build_i8mm/mollm_chat --package qwen3_30b_a3b_w4g128.mollm --threads 4
```

By default, `mollm_chat` loads package weights into resident memory. For mmap
A/B testing, pass `--mmap`; mmap page warmup is enabled unless you also pass
`--no-load-warmup`.

## Benchmark

```bash
./build_i8mm/mollm_bench \
    --package qwen35_4b_w4g128.mollm \
    --prompt-tokens 256 \
    --max-new-tokens 64 \
    --warmup 3 \
    --threads 4
```

## CUDA Backend (experimental)

CUDA is built independently of the CPU provider:

```bash
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DMOLLM_CUDA=ON
cmake --build build_cuda -j
./build_cuda/mollm_chat --device cuda --package model.mollm
```

CMake detects visible GPUs and checks compiler support, selecting `121a` on
GB10. Without a usable GPU, or when cross-compiling, it uses portable targets
`75;80;86`. Set `-DCMAKE_CUDA_ARCHITECTURES=...` (or `CUDAARCHS`) only to override
the selection, for example when building for another machine. A detected GPU
unsupported by the toolkit produces a configuration error; GB10 requires a
toolkit supporting `121a`, such as the tested CUDA 13.

Selecting CUDA requires a working CUDA backend; it never silently falls back to
CPU. Add `--require-native` to reject any operator that would use the CPU
reference implementation. Set `MOLLM_CUDA_PROFILE=1` to print native/fallback
operator counts.

**Qwen3.8-Flash-Next** has a validated native path: resident NVFP4 experts,
QSA indexer and selected-key attention, grouped NVFP4 prefill, GDN /
short-convolution, and MTP state handling. On a GB10-class unified-memory
system, first prepare the package and companion from the same checkpoint
(run these commands from the repository root):

```bash
python3 models/converter.py /path/to/Qwen3.8-Flash-Next-NVFP4 \
  ../qwen38_flash_next_nvfp4_calibrated.mollm
python3 tools/export_qwen4_qsa_weights.py /path/to/Qwen3.8-Flash-Next-NVFP4 \
  ../qwen38_qsa_bundle
```

The converter preserves activation calibration when present in the checkpoint;
the output filename alone does not certify calibration. Export the companion
to a new or empty directory. It contains indexer weights, not a standalone
model. Neither the build nor the launcher downloads these files automatically.
Then start the interactive long-context preset:

```bash
./chat-qwen38-cuda.sh
```

The preset uses a 16K context and a chunked prefill of 2048 tokens. It attaches
the local `../qwen38_qsa_bundle` companion automatically for its default
package. The companion supplies the QSA indexer weights; without
`--qsa-companion`, long context uses dense attention and does not reproduce the
original model semantics. `--dense-bringup` skips QSA for diagnostic A/B, and
`--qsa-companion <dir>` selects a different bundle. CLI arguments override the
preset, for example `--prefill-chunk-size 512`.

Supported QSA scope: text only, continuous causal history up to 262,144 tokens
(memory permitting), no MTP, static padding, vision, or noncanonical masks.
Unsupported configurations reject rather than silently use dense attention.
The companion must match the selected checkpoint. The launcher checks for an
idle GPU and sufficient host memory at startup; admission checks are not
memory reservations.

For models other than Qwen3.8-Flash-Next, CUDA is a mixed native/fallback
implementation: graph outputs, persistent state, dense elementwise ops, norms,
RoPE, SDPA, and the decode `lm_head` stay on CUDA, while recurrent GDN /
short-convolution and several specialized model families may use the CPU
reference dispatcher. See
[QSA implementation status](docs/qsa-implementation-status.md) and
[performance notes](docs/performance.md) for scope and evidence.

## Local HTTP Server

```bash
./build_i8mm/mollm_server \
    --package qwen35_4b_w4g128.mollm \
    --host 127.0.0.1 --port 8080 --threads 4
```

The server implements `GET /v1/models` and OpenAI-compatible
`POST /v1/chat/completions`, including SSE streaming. It retains a single exact
token-prefix KV cache between serialized requests. Sampling parameters can be
set per request; the default remains deterministic (`temperature=0`). See
[SERVER.md](SERVER.md) for fields, examples, and limitations.

## Documentation

- [Performance protocols and complete results](docs/performance.md)
- [SSD expert offload: cache sweeps, I/O behavior, tracing](docs/ssd-offload.md)
- [Qwen3.8 QSA implementation status and limits](docs/qsa-implementation-status.md)
- [HTTP server API](SERVER.md)
- [Documentation index](docs/README.md)

## Project Layout

```text
palm-infra/
├── core/       tensors, quantization layouts, cross-backend op parameters
├── runtime/    backend interfaces, thread pool, host compute, buffer pool
├── storage/    mmap, file ranges, SSD expert cache
├── graph/      graph format, serialization, executor
├── engine/     LLMEngine, model loading, tokenizer, generation lifecycle
├── backends/   CPU, CUDA, and Metal resource management and dispatch
├── kernels/    numerical kernels under cpu/, cuda/, and metal/
├── models/     Python converters and graph builders
├── examples/   CLI, server, benchmark, and perplexity programs
├── tools/      standalone development and diagnostic tools
├── tests/      unit, backend, and end-to-end tests
└── cmake/      install packaging and downstream build support
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
- More model families beyond the current Qwen, Youtu, Hy, and DeepSeek coverage.
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
