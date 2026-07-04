# mollm — mobile-oriented LLM inference engine

A from-scratch LLM inference engine for ARM-based devices.
Currently developed and benchmarked on Apple Silicon; targeting mobile/edge
ARM (Qualcomm Oryon, MediaTek) on the roadmap.
Pure C++ runtime with a Python transpilation frontend. FP16 uses FP16FML NEON
kernels; W8/W4 weight-only quantization uses ARM dot-product kernels.

## Status

Supports three model families. Benchmark protocol is Apple M5 Pro, 4 threads,
pp256 + tg64, warmup=3, 5-run median unless noted.

FP16:

| Model | Architecture | mollm pp/tg | llama.cpp pp/tg |
|-------|--------------|------------:|----------------:|
| Qwen3.5-0.8B | Hybrid linear/full attention (Gated DeltaNet + GQA) | 625.36 / 101.74 | 675.69 / 99.24 |
| Youtu-LLM-2B | MLA | 241.59 / 51.27 | 266.66 / 46.89 |
| Qwen3.5-4B | Hybrid linear/full attention (Gated DeltaNet + GQA) | 110.66 / 24.44 | 143.22 / 22.02 |

W8PC / Q8_0:

| Model | mollm W8PC pp/tg | llama.cpp Q8_0 pp/tg |
|-------|-----------------:|---------------------:|
| Qwen3.5-0.8B | 604.11 / 151.23 | 813.58 / 171.10 |
| Youtu-LLM-2B | 239.59 / 87.89 | 275.90 / 83.99 |
| Qwen3.5-4B | 114.36 / 43.02 | 142.88 / 39.77 |

Youtu-LLM-2B W4G128 direct q4pkg after the latest W4 GEMM cleanup:

| Runtime | pp256 | tg64 |
|---------|------:|-----:|
| mollm W4G128 | 227.31 | 91.28 |
| llama.cpp Q4_0 | 281.04 | 99.82 |
| llama.cpp Q4_K_M | 216.35 | 88.72 |
| llama.cpp Q8_0 | 275.90 | 83.99 |

Current summary: decode is competitive with llama.cpp on 2B/4B for FP16, W8,
and 2B W4. AUTO i8mm plus full-tile cleanup makes W8 prefill much closer to
llama.cpp; W4 prefill has recovered substantially with q4dot DOTPROD
vector-reduce, scaled-nibble, and full-tile GEMM cleanup. Remaining gap is
mostly 4B prefill.

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

### Package format (`.mollm`)

A single-file container bundling graphs + weights + tokenizer + metadata, so
models ship as one artifact instead of a directory of loose files. Format:

```
[Header 128B]
  magic "MOLM"
  metadata_offset, metadata_len      # JSON: config + weight offset map
  tokenizer_offset, tokenizer_len
  chat_template_offset, template_len
  prefill_graph_offset, graph_len
  decode_graph_offset, graph_len
  weights_offset, weights_len
[metadata JSON]
[tokenizer.json]
[chat_template.jinja]
[prefill graph bytes]
[decode graph bytes]
[weights bytes]                      # mmap'd at runtime
```

Runtime opens the `.mollm` file once, mmaps the weights region, and reads the
two graphs. Packages must contain explicit `lm_head.weights`; old packages from
before that boundary need reconversion.

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

Quantization status:

- `models/converter.py` auto-detects `build/mollm-quantize` for W8/W4 tensor
  quantization and q4dot packing. `MOLLM_QUANT_HELPER=/path/to/helper`
  overrides the path; `MOLLM_DISABLE_CPP_QUANT=1` forces Python fallback only
  for W8. W4 has no Python fallback.
- W8 uses weight-only int8 with Q8 activation dot-product kernels. Runtime uses
  q8dot repacked layout `[N/8, K/32, 8, 32]` by default for GEMV and GEMM.
- W4 uses signed int4 weights with Q8 activation dot-product kernels. Current
  packages store W4 directly in q4dot physical layout
  `[ceil(N/8), K/32, 8, 16B]`, so runtime does not keep a second W4 repack copy.
- `w4mixgN` promotes quality-sensitive tensors such as `lm_head.weights` and
  selected attention/FFN projections to W8. Qwen3.5 and Youtu have separate
  model-specific policies.
- `embed_tokens.weights` stays FP16 row-major for lookup; `lm_head.weights` is
  explicit in the package and follows the same quantization policy as a normal
  linear weight.

Quantized package examples:

| Package | pp256 t/s | tg64 t/s | peak RSS |
|---------|----------:|---------:|---------:|
| Qwen3.5-0.8B W8PC | 604.11 | 151.23 | 2560.6 MB |
| Qwen3.5-0.8B W4G128 direct q4pkg | 624.17 | 163.37 | 1501.2 MB |
| Youtu-LLM-2B W8PC | 239.59 | 87.89 | 5948.9 MB |
| Youtu-LLM-2B W4G128 direct q4pkg | 227.31 | 91.28 | 2894.3 MB |
| Qwen3.5-4B W8PC | 114.36 | 43.02 | 10964.4 MB |
| Qwen3.5-4B W4G128 direct q4pkg | 102.42 | 43.34 | 4984.3 MB |

Pure W4 quality is model-dependent. On the current 256-token reference sample,
Youtu-LLM-2B W4G128 has small drift from FP16 (`CE delta +0.0440`), and
Youtu `w4mixg128` improves it slightly to `+0.0372`. Qwen3.5-0.8B pure W4
drifts much more; Qwen3.5 `w4mixg128` keeps quality near W8 by selectively
promoting important tensors.

`MOLLM_ARM_I8MM` defaults to `AUTO`: on ARM64, CMake probes
`-march=armv8.6-a+i8mm+fp16+fp16fml` and enables the i8mm W8 GEMM kernel when
the compiler defines `__ARM_FEATURE_MATMUL_INT8` and accepts `vmmlaq_s32`.
Use `-DMOLLM_ARM_I8MM=OFF` to force the portable DOTPROD/NEON path, or
`-DMOLLM_ARM_I8MM=ON` to require i8mm support. `MOLLM_ARM_NATIVE=ON` passes
`-mcpu=native` for local benchmarking. W4 currently defaults to the q4dot
DOTPROD vector-reduce GEMM; `MOLLM_W4_I8MM=1` enables an experimental W4 i8mm
prototype, but it is slower on current shapes.

## Run

```bash
# Chat (single .mollm file — no separate tokenizer needed)
./build/mollm_chat --package qwen35_4b.mollm --threads 4

# Benchmark (pp256 + tg64, prints per-op profile)
./build/mollm_bench --package qwen35_4b.mollm \
    --prompt-tokens 256 --max-new-tokens 64 --warmup 3 --threads 4 --profile
```

## Roadmap

### Near-term
- **W4 prefill/decode kernels**: 2B W4 decode is now llama.cpp Q4/Q8-class and
  prefill has improved with q4dot DOTPROD vector-reduce; next work is
  activation Q8 quantization, scale handling, and a true W4 micro-panel layout
  if i8mm is revisited.
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
