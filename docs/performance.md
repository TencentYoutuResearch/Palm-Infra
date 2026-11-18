# Performance

This page records the current public Qwen3.8 CUDA configuration and its measured
results. It is not an experiment journal. Preserve raw profiles and benchmark
artifacts outside the source tree.

## Supported measurement configuration

The results below were measured on 2026-11-15 with:

- NVIDIA GB10 (SM121), driver 580.178.04, CUDA 13.0;
- `qwen38_flash_next_nvfp4_calibrated.mollm` with matching QSA and, where used,
  QSA MTP companion bundles;
- greedy decoding, a 1759-token English prompt, 512 generated tokens,
  `warmup=1`, and `repeat=2` per model load;
- a 32K context and 2048-token logical prefill chunks.

These are short-prompt decode measurements, not long-context throughput,
accuracy, or cross-architecture claims. SM80 was also built successfully, but
was not benchmarked.

Build the public binaries with:

```bash
cmake -S . -B build_cuda \
  -DMOLLM_CUDA=ON -DMOLLM_BUILD_TESTS=ON \
  -DMOLLM_CUDA_DECODE_SEGMENTS=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121a
cmake --build build_cuda -j 2
```

Use an explicit architecture only for a known target. The normal build can
select local architectures automatically and retains generic fallbacks.

## Public presets

The launchers enable the measured stable policy without requiring users to
assemble environment variables:

```bash
./chat-qwen38-cuda.sh --temperature 0
./chat-qwen38-cuda-mtp.sh
```

The base preset uses exact Q8 argmax screening and the stable narrow-row cache.
The MTP preset additionally enables compact draft heads, lossless FP16 expert
palettes, verification width caching, and confidence-based draft scheduling.
Approximations are limited to draft proposals; target verification uses the
original weights. Ineligible models and shapes retain their generic paths.

The MTP preset defaults to two draft tokens. The measured configuration below
uses three. Production optimizations are selected by device and shape guards;
no experimental environment settings are required.

## Current measurements

### Base decode

Two model loads used the stable narrow-row cache. Both generated the exact
independent 512-token reference:

| Measurement | Run 1 | Run 2 |
| --- | ---: | ---: |
| Prefill tok/s | 1437.98 | 1440.38 |
| Decode tok/s | 40.99 | 40.96 |

### QSA + MTP

The MTP run used three draft tokens and the stable mode1 narrow-row cache:

| Measurement | Run 1 | Run 2 |
| --- | ---: | ---: |
| Prefill tok/s | 1300.84 | 1305.26 |
| Decode tok/s | 59.94 | 60.06 |

Both runs produced the exact independent 512-token reference and reported:

```text
steps=147 draft_calls=422 drafted=413 accepted=364 verify_tokens=560
recurrent_replay_tokens=0
```

This reproduces the earlier approximately 59.5 tok/s result within ordinary
run-to-run variation. It does not establish a speedup for every prompt, draft
depth, or acceptance distribution.

## Reproducing a benchmark

Use `mollm_bench` with the same package, companions, prompt, and policy as the
presets. A benchmark prompt used for the table must contain exactly 1759 tokens
after chat templating if exact numerical comparison is required.

Base mode1:

```bash
env \
  MOLLM_RESIDENT_EXPERT_ARENA=cuda-device \
  MOLLM_CUDA_PREFILL_FP16_CACHE_MB=2048 \
  MOLLM_CUDA_HOST_HEADROOM_MB=24576 \
  MOLLM_CUDA_OUTPUT_POOL_MB=3072 \
  build_cuda/mollm_bench \
  --package ../qwen38_flash_next_nvfp4_calibrated.mollm \
  --qsa-companion ../qwen38_qsa_bundle \
  --device cuda --offload-ple --n-ctx 32768 --threads 8 \
  --cuda-prefill-reclaim --cuda-prefill-logical-batch \
  --prefill-chunk-size 2048 --prompt-file /path/to/prompt.txt \
  --max-new-tokens 512 --temperature 0 --warmup 1 --repeat 2 \
  --print-token-ids
```

For the MTP measurement, add the following environment settings and arguments:

```bash
  MOLLM_CUDA_Q4_SMALL_BATCH_EXACT=1 \
  MOLLM_CUDA_Q4_SMALL_BATCH_EXACT_REUSE=1 \
  MOLLM_CUDA_Q4_COMPACT_BATCH_REUSE=1 \
  MOLLM_CUDA_GDN_SMALL_BATCH_SERIAL=1 \
  MOLLM_CUDA_MTP_DEVICE_ARGMAX=1 \
  MOLLM_CUDA_MTP_DEVICE_CHECKPOINT=1 \
  MOLLM_CUDA_MTP_STATE_PREFIXES=1 \
  MOLLM_CUDA_MTP_VERIFY_Q8_BATCH=1 \
```

```text
  --mtp-companion ../qwen38_mtp_qsa_bundle --mtp-draft-tokens 3
```

Keep the base resource settings from the first command. Compare complete
generated token IDs, not only throughput. Serialize GPU jobs and avoid
concurrent compilation during timing.

## Regression checks

```bash
python3 tests/test_prefill_contract_selector.py build_cuda/mollm_bench
python3 tests/test_qwen38_cuda_preset.py
ctest --test-dir build_cuda --output-on-failure -j 1 -R \
  '^(test_cuda_qsa_.*|test_qsa_engine_e2e$|test_cuda_verification_workspace.*|test_cuda_verify_.*|test_cuda_q4_narrow.*|test_cuda_q8_.*|test_cuda_draft_.*|test_cuda_fp16_expert_palette$|test_recurrent_mtp_prefix_e2e$)$'
```

Run the complete CTest suite for a release candidate. The focused list above
covers the optimization-specific paths; the generic contract and preset suites
cover the public configuration contract.

## Scope and retained limitations

- QSA supports text with continuous causal history; static padding, vision, and
  noncanonical masks remain rejected.
- QSA and dense draft bundles must match the target model and each other.
- Screening is used only where exact argmax semantics are preserved. Sparse
  screened logits must not be consumed as full sampling logits.
- Qualified SM121 specializations must not bypass the generic
  fallback for unsupported devices, layouts, shapes, or model files.
- Raw traces, generated corpora, sanitizer logs, and historical round reports
  belong in external benchmark storage, not in this document or the repository.
