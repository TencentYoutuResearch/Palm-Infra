# mollm — mobile-oriented LLM inference engine

A from-scratch LLM inference engine for ARM-based devices.
Currently developed and benchmarked on Apple Silicon; targeting mobile/edge
ARM (Qualcomm Oryon, MediaTek) on the roadmap.
Pure C++ runtime with a Python transpilation frontend. FP16 with FP16FML NEON kernels.

## Status

Supports three model families:

| Model | Architecture | pp256 t/s | tg64 t/s |
|-------|-------------|-----------|----------|
| Qwen3.5-4B | Hybrid linear/full attention (Gated DeltaNet + GQA) | 115 | 25 |
| Qwen3.5-0.8B | Same | 601 | 104 |
| Youtu-LLM-2B | MLA | 235 | 54 |

Benchmarks vs llama.cpp (Apple M5 Pro, 4 threads, pp256 + tg64, warmup=3):

| Model | mollm pp/tg | llama.cpp pp/tg | prefill gap | decode gap |
|-------|------------|-----------------|-------------|------------|
| Qwen3.5-4B | 115 / **25** | 143 / 23 | 1.25x | **0.92x (faster)** |
| Qwen3.5-0.8B | 601 / **104** | 749 / 100 | 1.25x | **0.96x (faster)** |
| Youtu-LLM-2B | 235 / **54** | 264 / 41 | 1.12x | **0.76x (faster)** |

Decode beats llama.cpp on all three models. Prefill gaps are MATMUL-bound
and await weight quantization.

## Architecture

- **Python transpile → binary graph**: PyTorch weights + model definition → serialized `.graph` + `.weights` files. No runtime JIT, all op fusion done at transpile time.
- **C++ executor**: Sequential node dispatch, BufferPool memory management, mmap'd weights.
- **NEON kernels**: FP16FML lane-FMA GEMM, dedicated GEMV with 8-way K-unroll, fused Gated DeltaNet recurrence, SDPA with register-tiled PV.

### Directory layout

```
mollm/
├── kernels/         NEON kernels (matmul, attention, gdn, norm, rope)
├── graph/           Executor, graph format, BufferPool, mmap
├── engine/          LLMEngine, tokenizer, generation loop
├── models/          Python transpilers (qwen35.py, mla.py)
├── models/               Python transpilers + graph builder (qwen35.py, mla.py, transpile.py)
├── examples/        mollm_chat (CLI), mollm_bench (benchmark)
├── tests/           Unit + e2e tests
└── docs/            Optimization logs, architecture notes
```

### Package format (`.mollm`)

A single-file container bundling graph + weights + metadata, so models ship as one
artifact instead of a directory of loose files. Planned format:

```
[Header]
  magic "MOLM"
  version
  metadata_offset, metadata_len      # JSON: model name, architecture, config,
                                     #   tokenizer ref, prefill_seq_len, n_ctx
  prefill_graph_offset, graph_len    # binary graph (existing format)
  decode_graph_offset, graph_len
  weights_offset, weights_len        # concatenated .weights blobs
[metadata JSON]
[prefill graph bytes]
[decode graph bytes]
[weights bytes]                      # mmap'd at runtime, offset within file
```

Runtime opens the `.mollm` file once, mmaps the weights region, and reads the two
graphs. No directory sprawl, no path juggling. The transpile step can emit either
the current directory layout or a single `.mollm` file.

## Build

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
ninja -C build
```

Requires Apple Clang (for `__ARM_FEATURE_FP16FML`), Python 3 with `safetensors`.

## Convert a model

```bash
cd mollm
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b.mollm
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w8pc.mollm 32 256 w8pc
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w8g128.mollm 32 256 w8g128
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
| 5 | `quant` | no | `none`, `w8pc`, or `w8gN` such as `w8g128` |

Produces a single `.mollm` file containing graphs + weights + tokenizer + chat template.

W8 quantization is currently correctness-first. The package format and runtime
support per-channel/per-group scales, but INT8 matmul is still scalar; benchmark
after the NEON W8 kernels land. See `docs/QUANTIZATION.md`.

## Run

```bash
# Chat (single .mollm file — no separate tokenizer needed)
./build/mollm_chat --package qwen35_4b.mollm --threads 4

# Benchmark (pp256 + tg64, prints per-op profile)
./build/mollm_bench --package qwen35_4b.mollm \
    --prompt-tokens 256 --max-new-tokens 64 --warmup 3 --threads 4 --profile
```

## Optimization highlights

See `docs/OPTIMIZATION_LOG_QWEN35.md` for the full Qwen3.5 optimization journey (attempts 1-12, prefill 71→115 t/s, +62%):

- **GDN recurrence kernel fusion**: 5-pass state access → 2-pass (state memory traffic -60%)
- **GDN decode multi-threading**: 21.8% → 7.2% of decode time
- **full_attn contiguous cleanup**: verified rms_norm/rope kernels handle strided inputs via unit tests, removed 5 redundant `contiguous()` calls
- **RESHAPE/CONTIGUOUS materialize**: row-level memcpy fast path (-94%, 85ms→5ms)
- **FP16FML matmul**: 988 GF/s microbench (86% of FP16FML peak), lane-FMA + 2-way K-unroll GEMM, 8-way K-unroll GEMV

## Roadmap

### Near-term
- **`.mollm` package format**: single-file model artifact (graph + weights + metadata), replacing the current directory layout
- **W8/W4 weight quantization**: W8 weight-only format/runtime is in place as a correctness baseline; optimized W8 NEON kernels and then W4 are the next prefill levers
- **Graph fusion**: fuse adjacent matmul + activation + norm to reduce cache thrash between ops (end-to-end matmul utilization is 40% vs 86% microbench, the 2.5x gap is cache/DRAM traffic between matmuls)

### Mid-term
- **Continuous batching / multi-user**: serve multiple sequences with shared prefill
- **Speculative decoding**: draft model + verify
- **Qualcomm Oryon support**: validate NEON kernels on non-Apple ARM
- **Vision encoder**: Qwen3.5 is multimodal — wire up the ViT side

### Longer-term
- **GPU/Vulkan backend**: roadmap item, CPU-first for now
- **W4 KV cache**: reduce decode memory for long contexts
- **More models**: Llama, Mistral, DeepSeek families
