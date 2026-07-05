# mollm — mobile-oriented LLM inference engine

A from-scratch LLM inference engine for ARM-based devices.
Currently developed and benchmarked on Apple Silicon; targeting mobile/edge
ARM (Qualcomm Oryon, MediaTek) on the roadmap.
Pure C++ runtime with a Python transpilation frontend. FP16 uses FP16FML NEON
kernels; W8/W4 weight-only quantization uses ARM dot-product kernels.

## Status

Supports three model families. Benchmark protocol is Apple M5 Pro, 4 threads,
pp256 + tg64, warmup=3, 5-run median unless noted. Higher throughput numbers
are bolded.

FP16:

| Model | Architecture | mollm pp/tg | llama.cpp pp/tg | pp winner | tg winner |
|-------|--------------|------------:|----------------:|-----------|-----------|
| Qwen3.5-0.8B | Hybrid linear/full attention | 625.36 / **101.74** | **675.69** / 99.24 | llama.cpp 1.08x | mollm 1.03x |
| Youtu-LLM-2B | MLA | 241.59 / **51.27** | **266.66** / 46.89 | llama.cpp 1.10x | mollm 1.09x |
| Qwen3.5-4B | Hybrid linear/full attention | 110.66 / **24.44** | **143.22** / 22.02 | llama.cpp 1.29x | mollm 1.11x |

W8PC / Q8_0 (current AUTO-i8mm build):

| Model | mollm W8PC pp/tg | llama.cpp Q8_0 pp/tg | pp winner | tg winner |
|-------|-----------------:|---------------------:|-----------|-----------|
| Qwen3.5-0.8B | 635.51 / **211.96** | **811.70** / 160.70 | llama.cpp 1.28x | mollm 1.32x |
| Youtu-LLM-2B | 250.81 / **94.34** | **274.46** / 84.61 | llama.cpp 1.09x | mollm 1.11x |
| Qwen3.5-4B | 117.21 / **45.50** | **139.66** / 39.71 | llama.cpp 1.19x | mollm 1.15x |

W4G128 direct BG128 / Q4_0:

| Model | mollm W4G128 pp/tg | llama.cpp Q4_0 pp/tg | pp winner | tg winner |
|-------|--------------------:|---------------------:|-----------|-----------|
| Qwen3.5-0.8B | 687.32 / **252.91** | **800.05** / 188.72 | llama.cpp 1.16x | mollm 1.34x |
| Youtu-LLM-2B | 241.99 / **114.03** | **269.87** / 97.39 | llama.cpp 1.12x | mollm 1.17x |
| Qwen3.5-4B | 110.78 / **55.75** | **143.46** / 44.32 | llama.cpp 1.29x | mollm 1.26x |


## Architecture

- **Python transpile → `.mollm` package**: PyTorch weights + model definition
  become a single mmap-friendly package with metadata, tokenizer, prefill graph,
  decode graph, and weights.
- **C++ executor**: Sequential node dispatch, BufferPool memory management,
  graph workspace reuse, mmap'd weights, and a CPU backend abstraction.
- **NEON kernels**: FP16FML lane-FMA GEMM/GEMV, W8 q8dot, W4 q4dot, fused
  Gated DeltaNet recurrence, SDPA with register-tiled PV.

### Directory layout

```
mollm/
├── kernels/         NEON kernels (matmul, attention, gdn, norm, rope)
├── graph/           Executor, graph format, BufferPool, mmap
├── engine/          LLMEngine, tokenizer, generation loop, backend abstraction
├── models/          Python transpilers + graph builder (qwen35.py, mla.py, transpile.py)
├── examples/        mollm_chat (CLI), mollm_bench (benchmark)
└── tests/           Unit + e2e tests
```

`.mollm` packages bundle graphs, weights, tokenizer, and chat template into one
mmap-friendly file. Format details are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Build

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
ninja -C build
```

Requires Apple Clang (for `__ARM_FEATURE_FP16FML`), Python 3 with `safetensors`.

## Convert a model

For W4 conversion, build the C++ tensor quantizer first. The Python converter
can still do FP16 and W8 without it, but W4 intentionally requires the helper:

```bash
cmake --build build --target mollm-quantize
```

```bash
cd mollm
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b.mollm
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w8pc.mollm 32 256 w8pc
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w8g128.mollm 32 256 w8g128
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4g128.mollm 32 256 w4g128
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4mixg128.mollm 32 256 w4mixg128
python3 models/converter.py /path/to/Youtu-LLM-2B youtu-llm-2b_w4g128.mollm 32 256 w4g128
python3 models/converter.py /path/to/Youtu-LLM-2B youtu-llm-2b_w4mixg128.mollm 32 256 w4mixg128
```

The converter auto-detects the model type from `config.json` and dispatches to the appropriate converter. Supported types:

| `model_type` | Model | Default layers |
|-------------|-------|-----------------|
| `qwen3_5` | Qwen3.5-0.8B/4B | 24 (0.8B), 32 (4B) |
| `youtu` | Youtu-LLM-2B | 32 |

**Arguments:**

| # | Name | Required | Description |
|---|------|----------|-------------|
| 1 | `model_dir` | yes | HF model directory (must contain `config.json` + `model.safetensors`) |
| 2 | `output_path` | yes | Output `.mollm` file path |
| 3 | `num_layers` | no | Override number of hidden layers (auto-detected if omitted) |
| 4 | `prefill_seq_len` | no | Prefill chunk size (default 256) |
| 5 | `quant` | no | `none`, `w8pc`, `w8gN`, `w4gN`, or `w4mixgN` |

Produces a single `.mollm` file containing graphs + weights + tokenizer + chat template.

Quantization summary:

- Supported modes: `none`, `w8pc`, `w8gN`, `w4gN`, and `w4mixgN`.
- W8 is the conservative int8 path with small quality drift. W4G128 direct
  BG128 is the current lowest-RSS quantized path and fastest decode path.
- W4 conversion requires the C++ quantizer helper; FP16 and W8 conversion can
  still run from Python alone.
- `w4mixgN` keeps selected quality-sensitive tensors in W8 when pure W4 is too
  lossy for a model.

See [docs/QUANTIZATION.md](docs/QUANTIZATION.md) for tensor formats, quality
audits, diagnostic env flags, and kernel notes.

## Run

```bash
# Interactive chat (single .mollm file — no separate tokenizer needed)
./build/mollm_chat --package qwen35_4b_w4g128_bg128.mollm --threads 4

# One-shot chat smoke; temperature 0 is deterministic
./build/mollm_chat --package qwen35_4b_w4g128_bg128.mollm \
    --prompt "请只输出一句话，不要解释：杭州有什么特点？" \
    --max-new-tokens 64 --threads 4 --temperature 0

# Other current local quantized packages
./build/mollm_chat --package qwen35_0.8b_w4g128_bg128.mollm --threads 4
./build/mollm_chat --package youtu-llm-2b_w4g128_bg128.mollm --threads 4
./build/mollm_chat --package qwen35_4b_w8pc.mollm --threads 4

# Benchmark (pp256 + tg64, prints per-op profile)
./build/mollm_bench --package qwen35_4b_w4g128_bg128.mollm \
    --prompt-tokens 256 --max-new-tokens 64 --warmup 3 --threads 4 --profile
```

In interactive mode, use `/reset` to clear the conversation and `/quit` to
exit. If your local optimized build directory is `build_i8mm`, replace
`./build/...` with `./build_i8mm/...`.

## Roadmap

### Near-term
- **W4 prefill**: W4 decode is now faster than local llama.cpp Q4_0 on all
  three benchmarked models; next work is closing the remaining prefill gap.
- **W4 quality policy**: Qwen3.5 and Youtu both have first mixed policies;
  next step is targeted ablation to reduce W8 coverage without losing PPL.
- **Graph fusion**: fuse adjacent matmul + activation + norm to reduce cache thrash between ops (end-to-end matmul utilization is 40% vs 86% microbench, the 2.5x gap is cache/DRAM traffic between matmuls)
- **Memory planning**: current workspace reuse fixes repeated allocation churn;
  next step is last-use liveness planning to reduce peak RSS on long prompts.

### Mid-term
- **Continuous batching / multi-user**: serve multiple sequences with shared prefill
- **Speculative decoding**: draft model + verify
- **Vision encoder**: Qwen3.5 is multimodal — wire up the ViT side

### Longer-term
- **MoE support**: Mixture-of-Experts routing + sparse expert matmul (DeepSeek, Qwen3-MoE, Mixtral)
- **GPU/Vulkan backend**: roadmap item, CPU-first for now
- **W4 KV cache**: reduce decode memory for long contexts
- **More models**: Llama, Mistral, DeepSeek families
