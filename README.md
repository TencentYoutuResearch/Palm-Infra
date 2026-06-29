# mollm — ARM-first LLM inference engine

A from-scratch LLM inference engine targeting Apple Silicon (and Qualcomm Oryon).
Pure C++ runtime with a Python transpilation frontend. FP16 with FP16FML NEON kernels.

## Status

Supports two model families:

| Model | Architecture | pp256 t/s | tg64 t/s |
|-------|-------------|-----------|----------|
| Qwen3.5-4B | Hybrid linear/full attention (Gated DeltaNet + GQA) | 115 | 25 |
| Qwen3.5-0.8B | Same | ~550 | ~95 |
| Youtu-LLM-2B | MLA | 235 | 54 |

Benchmarks vs llama.cpp (Apple M5, 4 threads, pp256 + tg64, warmup=3):

| Model | mollm pp/tg | llama.cpp pp/tg |
|-------|------------|-----------------|
| Qwen3.5-4B | 115 / 25 | 143 / 23 |
| Qwen3.5-0.8B | 550 / 95 | 749 / 100 |

4B decode already exceeds llama.cpp; prefill gap is 1.25x (MATMUL-bound, awaits weight quantization).

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
├── python/          Graph builder + serializer (transpile.py)
├── examples/        mollm_chat (CLI), mollm_bench (benchmark)
├── tests/           Unit + e2e tests
└── docs/            Optimization logs, architecture notes
```

## Build

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
ninja -C build
```

Requires Apple Clang (for `__ARM_FEATURE_FP16FML`), Python 3 with `safetensors`.

## Convert a model

```bash
cd mollm
PYTHONPATH=. python3 models/qwen35.py /path/to/Qwen3.5-4B output_qwen35_4B 32 256
```

Produces `output_qwen35_4B/{model_prefill.graph, model_decode.graph, weights/}`.

## Run

```bash
# Chat
./build/mollm_chat --tokenizer tokenizer.json --artifacts output_qwen35_4B --threads 4

# Benchmark (pp256 + tg64, prints per-op profile)
./build/mollm_bench --tokenizer tokenizer.json --artifacts output_qwen35_4B \
    --prompt-tokens 256 --max-new-tokens 64 --warmup 3 --threads 4 --profile
```

## Optimization highlights

See `docs/OPTIMIZATION_LOG_QWEN35.md` for the full Qwen3.5 optimization journey (attempts 1-12, prefill 71→115 t/s, +62%):

- **GDN recurrence kernel fusion**: 5-pass state access → 2-pass (state memory traffic -60%)
- **GDN decode multi-threading**: 21.8% → 7.2% of decode time
- **full_attn contiguous cleanup**: verified rms_norm/rope kernels handle strided inputs via unit tests, removed 5 redundant `contiguous()` calls
- **RESHAPE/CONTIGUOUS materialize**: row-level memcpy fast path (-94%, 85ms→5ms)
- **FP16FML matmul**: 988 GF/s microbench (86% of FP16FML peak), lane-FMA + 2-way K-unroll GEMM, 8-way K-unroll GEMV
