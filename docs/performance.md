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

The launcher sets only resource limits; the CUDA backend selects the measured
kernels by device and shape:

```bash
./chat-qwen38-cuda.sh --temperature 0
./chat-qwen38-cuda.sh --mtp
```

On SM121 the base configuration uses an exact Q8 argmax screen and cached
narrow-row projection views. With MTP the backend additionally uses compact
draft heads and FP16 expert palettes that reproduce the original weights.
Other devices, models and shapes retain their generic decode kernels.

On every device with compute capability 8.0 or newer, MTP uses
confidence-based draft scheduling and verification batches that reproduce the
device's token-by-token decode arithmetic, with cached workspaces, device
checkpoints and per-token recurrent state snapshots. Greedy MTP output therefore
matches greedy decoding. Approximations are limited to draft proposals; target
verification uses the original weights. Setting
`MOLLM_CUDA_DISABLE_QUALIFIED_VERIFICATION` selects the generic verification
paths for reference comparisons.

Outside SM121 the verification exactness tests also pass on SM89
(RTX 4060 Ti). End-to-end throughput was measured only on GB10; the resident
expert arena (about 68 GB) does not fit in the memory of that GPU.

`--mtp` defaults to two draft tokens. The measured configuration below uses
three.

## Current measurements

### Base decode

Two model loads used the narrow-row cache. Both generated the exact
independent 512-token reference:

| Measurement | Run 1 | Run 2 |
| --- | ---: | ---: |
| Prefill tok/s | 1437.98 | 1440.38 |
| Decode tok/s | 40.99 | 40.96 |

### QSA + MTP

The MTP run used three draft tokens and the narrow-row cache:

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

MTP decode throughput follows the draft acceptance rate of the prompt; base
decode does not draft. The 1759-token prompt above accepts 88% of
drafted tokens. The repository prompt
`tests/data/qwen38_cuda_prefill_engineering_long.txt` (1743 tokens) accepts
71.9% on 2026-11-21 with the same settings:

| Measurement | Base | QSA + MTP |
| --- | ---: | ---: |
| Decode tok/s | 40.7 | 50.0 |
| MTP steps for 512 tokens | – | 184 |

Each value is the mean of three model loads. An MTP step costs 55–58 ms
with either prompt; the MTP gap comes from fewer accepted tokens per step.
Use this prompt when the original one is unavailable.

### Qwen3.5-4B dense MTP

A `w4g128` package with its exported MTP companion, the repository prompt,
`--n-ctx 8192`, greedy decoding and 256 generated tokens (2026-11-22). Decode
tok/s is `1000 / tpot_ms` from one model load:

| Draft tokens | Accepted | GB10 (SM121) | RTX 4060 Ti (SM89) |
| --- | ---: | ---: | ---: |
| Base | – | 53.2 | 90.2 |
| 1 | 65.6% | 63.6 | 99.9 |
| 2 | 55.4% | 66.1 | 92.6 |
| 3 | 41.5% | 58.4 | 75.8 |

Every run on both devices generated the same 256 tokens. Verification
batches cost more than one decode step, so deeper drafts pay off only with
higher acceptance; one draft token is the better setting on the 4060 Ti.

## Reproducing a benchmark

Use `mollm_bench` with the same package, companions, prompt, and policy as the
presets. A benchmark prompt used for the table must contain exactly 1759 tokens
after chat templating if exact numerical comparison is required.

Base configuration:

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

For the MTP measurement, add:

```text
  --mtp-companion ../qwen38_mtp_qsa_bundle --mtp-draft-tokens 3
```

Keep the resource settings from the first command; the 3072 MiB output pool
is the largest that admits cached verification workspaces. Compare complete
generated token IDs, not only throughput. Serialize GPU jobs and avoid
concurrent compilation during timing.

## Regression checks

```bash
python3 tests/test_prefill_contract_selector.py build_cuda/mollm_bench
python3 tests/test_qwen38_cuda_preset.py
ctest --test-dir build_cuda --output-on-failure -j 1 -R \
  '^(test_cuda_qsa_.*|test_qsa_engine_e2e$|test_cuda_verification_workspace.*|test_cuda_verify_.*|test_cuda_q4_narrow.*|test_cuda_q8_.*|test_cuda_draft_.*|test_cuda_fp16_expert_palette$|test_recurrent_mtp_.*e2e$)$'
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
- SM121 decode specializations must not bypass the generic fallback for
  unsupported devices, layouts, shapes, or model files. Verification batches
  must reproduce the decode arithmetic of the device they run on.
- Raw traces, generated corpora, sanitizer logs, and historical round reports
  belong in external benchmark storage, not in this document or the repository.
