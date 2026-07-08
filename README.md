# mollm — mobile-oriented LLM inference engine

```
                 _ _
 _ __ ___   ___ | | |_ __ ___
| '_ ` _ \ / _ \| | | '_ ` _ \
| | | | | | (_) | | | | | | | |
|_| |_| |_|\___/|_|_|_| |_| |_|
```

A from-scratch LLM inference engine for ARM-based devices.
Currently developed and benchmarked on Apple Silicon; targeting mobile/edge
ARM (Qualcomm Oryon, MediaTek) on the roadmap.
Pure C++ runtime with a Python transpilation frontend. FP16 uses FP16FML NEON
kernels; W8/W4 weight-only quantization uses ARM dot-product kernels.

## Status

Supports Qwen3.5 dense, Youtu, and experimental Qwen3.5/3.6 MoE text models.
Benchmark protocol is Apple M5 Pro, 4 threads, pp256 + tg64, warmup=3,
5-run median unless noted. Higher throughput numbers are bolded. Quantized
llama.cpp rows use `GGML_BLAS=ON`, `GGML_METAL=OFF`, and
`GGML_CPU_REPACK=ON`.

FP16:

| Model | Architecture | mollm pp/tg | llama.cpp pp/tg | pp winner | tg winner |
|-------|--------------|------------:|----------------:|-----------|-----------|
| Qwen3.5-0.8B | Hybrid linear/full attention | 601.38 / **123.68** | **664.54** / 97.87 | llama.cpp 1.11x | mollm 1.26x |
| Youtu-LLM-2B | MLA | 218.67 / **52.12** | **257.93** / 45.12 | llama.cpp 1.18x | mollm 1.16x |
| Qwen3.5-4B | Hybrid linear/full attention | 102.48 / **25.00** | **142.44** / 22.33 | llama.cpp 1.39x | mollm 1.12x |

W8PC / Q8_0 (current AUTO-i8mm build):

| Model | mollm W8PC pp/tg | llama.cpp Q8_0 pp/tg | pp winner | tg winner |
|-------|-----------------:|---------------------:|-----------|-----------|
| Qwen3.5-0.8B | 671.73 / **217.69** | **782.16** / 167.63 | llama.cpp 1.16x | mollm 1.30x |
| Youtu-LLM-2B | 253.05 / **89.53** | **263.95** / 86.58 | llama.cpp 1.04x | mollm 1.03x |
| Qwen3.5-4B | 118.55 / **46.64** | **135.58** / 40.50 | llama.cpp 1.14x | mollm 1.15x |

W4G128 direct BG128 / Q4_0:

| Model | mollm W4G128 pp/tg | llama.cpp Q4_0 pp/tg | pp winner | tg winner |
|-------|--------------------:|---------------------:|-----------|-----------|
| Qwen3.5-0.8B | 678.41 / **259.43** | **775.95** / 190.89 | llama.cpp 1.14x | mollm 1.36x |
| Youtu-LLM-2B | 248.08 / **115.64** | **265.58** / 97.15 | llama.cpp 1.07x | mollm 1.19x |
| Qwen3.5-4B | 115.37 / **55.94** | **140.51** / 44.25 | llama.cpp 1.22x | mollm 1.26x |

Experimental MoE W4 / Q4_0:

| Model | mollm W4G128 pp/tg | llama.cpp Q4_0 pp/tg | pp winner | tg winner |
|-------|--------------------:|---------------------:|-----------|-----------|
| Qwen3.6-35B-A3B MoE | **131.80** / **60.31** | 116.93 / 43.73 | mollm 1.13x | mollm 1.38x |


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
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w8pc.mollm w8pc
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4g128.mollm w4g128
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4mixg128.mollm w4mixg128
python3 models/converter.py /path/to/Youtu-LLM-2B youtu-llm-2b_w4g128.mollm w4g128
python3 models/converter.py /path/to/Youtu-LLM-2B youtu-llm-2b_w4mixg128.mollm w4mixg128
python3 models/converter.py /path/to/Qwen3.6-35B-A3B qwen36_moe_40l_w4g128.mollm w4g128
```

The converter auto-detects the model type from `config.json` and dispatches to the appropriate converter. Supported types:

| `model_type` | Model |
|-------------|-------|
| `qwen3_5` | Qwen3.5-0.8B/4B |
| `qwen3_5_moe` | Qwen3.5/3.6-35B-A3B text-only MoE, FP16/W4 path |
| `youtu` | Youtu-LLM-2B |

**Arguments:**

| # | Name | Required | Description |
|---|------|----------|-------------|
| 1 | `model_dir` | yes | HF model directory (must contain `config.json` + `model.safetensors`) |
| 2 | `output_path` | yes | Output `.mollm` file path |
| 3 | `quant` | no | `fp16` (default), `w8pc`, `w4g128`, or `w4mixg128` |

Produces a single `.mollm` file containing graphs + weights + tokenizer + chat template.
Layer count is read from `config.json`. The prefill graph currently uses an
internal 256-token chunk length; runtime prefill is dynamic for shorter chunks
and splits longer prompts across chunks.

Quantization summary:

- Common modes: `fp16` by default, `w8pc` for the W8 baseline, `w4g128`
  for the W4 performance baseline, and `w4mixg128` when pure W4 needs a
  quality-biased fallback.
- W8 is the conservative int8 path with small quality drift. W4G128 direct
  BG128 is the current lowest-RSS quantized path and fastest decode path.
- W4 conversion requires the C++ quantizer helper; FP16 and W8 conversion can
  still run from Python alone.
- `w4mixg128` keeps selected quality-sensitive tensors in W8 when pure W4 is too
  lossy for a model.
- `qwen3_5_moe` supports full text-only W4 packages. Router
  `mlp.gate.weight` and `shared_expert_gate.weight` stay high precision, while
  expert `gate_proj`/`up_proj`/`down_proj` weights are flattened and quantized
  for the runtime MoE scheduler.

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
./build/mollm_chat --package qwen36_moe_40l_w4g128.mollm --threads 4

# Benchmark (pp256 + tg64, prints per-op profile)
./build/mollm_bench --package qwen35_4b_w4g128_bg128.mollm \
    --prompt-tokens 256 --max-new-tokens 64 --warmup 3 --threads 4 --profile
MOLLM_MOE_PROFILE=1 ./build/mollm_bench \
    --package qwen36_moe_40l_w4g128.mollm \
    --prompt-tokens 256 --max-new-tokens 64 --warmup 3 --threads 4 --profile

# Perplexity smoke / quantization quality checks on raw text
./build/mollm_ppl --package qwen36_moe_40l_w4g128.mollm \
    --text-file calibration.txt --max-tokens 256 --chunk-size 256 --threads 4
```

In interactive mode, use `/reset` to clear the conversation and `/quit` to
exit. If your local optimized build directory is `build_i8mm`, replace
`./build/...` with `./build_i8mm/...`.

## Roadmap

### Near-term
- **W4 prefill**: W4 decode is faster than local llama.cpp Q4_0 on the
  primary 2B/4B models; next work is closing the remaining prefill gap.
- **MoE prefill**: Qwen3.5/3.6 MoE now has a full W4 text path and matmul-backed
  routed FFN scheduler; next work is improving small-bucket expert GEMM.
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
- **MoE memory tiering**: expert-level lazy load, cache, and offload
- **GPU/Vulkan backend**: roadmap item, CPU-first for now
- **W4 KV cache**: reduce decode memory for long contexts
- **More models**: Llama, Mistral, DeepSeek families
