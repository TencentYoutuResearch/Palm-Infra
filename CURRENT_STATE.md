# CURRENT STATE

## Overall status

The core `mlllm` inference path is now in a good state for the current debug target:

- End-to-end generation for the test prompt is working.
- Greedy generation from the C++ engine matches the Python/HF-style reference implementation for both:
  - a plain prompt (`"Hello, world!"`)
  - the current chat-template prompt path
- Layer-by-layer numerical tracing confirmed that the early attention stack is aligned with the reference through:
  - layer 0: `q_full`, `k_full`, `v`, `attn_out`, `out_proj`, residual
  - layer 1: input tensor and `q_full`, `k_full`, `v`

At this point, the main bug that was producing obviously wrong text for the tested path appears fixed.

---

## What was fixed

### 1. Graph/view layout bugs in the executor

The main correctness issues were in view/materialization behavior rather than in SDPA math itself.

Key fixes:

- `SLICE` now preserves parent stride/layout instead of recomputing a fake contiguous layout.
- `CONCAT` now copies elementwise using strides instead of assuming flat contiguous memory.
- `RESHAPE` now materializes a contiguous buffer when the input is a non-contiguous view.

These fixes removed cases where K/Q/V data was being silently rearranged or aliased incorrectly before attention.

### 2. Python graph export layout bugs

The MLA exporter had incorrect `permute(...)` usage for tensors that were intended to become `[dim, seq, heads]`.

This affected:

- Q path
- KV-expanded path
- attention output flattening before `o_proj`

The exporter now uses the correct permutation order for the runtime tensor convention.

### 3. Multi-head RoPE bug

`kernel_rope(...)` previously handled the input too much like a 2D tensor and did not correctly iterate across channel/head planes with separate input/output strides.

Fixes include:

- separate input/output row strides
- iterating channel/head planes explicitly
- correct handling of multi-head RoPE outputs

This was necessary to get later-sequence and later-head values to match the reference.

### 4. Earlier stability fixes already integrated

The current code also includes the earlier fixes that were needed to get to stable execution:

- cache shape/metadata initialization fixes
- cache reset dimension fixes
- prefill hidden padding to graph sequence length
- prefill lm_head last-token selection fix
- matmul/kernel cleanup used by current passing path

---

## Validation completed

### C++ test coverage

Current tested paths include:

- decode-only execution path
- prefill + decode execution path
- tokenizer load/encode/decode
- executor shape/view regression tests
- rope/matmul/attention unit tests

`ctest` was previously passing for the current code path after these fixes.

### Reference alignment checks

A Python reference implementation was used to compare intermediate tensors and final token outputs.

Confirmed aligned checks include:

- layer 0 attention block output chain
- layer 1 input and Q/K/V construction
- plain prompt greedy generation token sequence
- chat-template greedy generation token sequence

### Final text checks

For the current checked prompts, the C++ engine and Python reference produce the same greedy output tokens and decoded text.

This means the current tested text path is not merely "plausible"; it is reference-aligned for those cases.

---

## Important current repo state

There are still local files/changes not yet committed:

- `CMakeLists.txt` has a local change adding `test_e2e`
- local debug/helper files currently untracked:
  - `tests/test_e2e.cpp`
  - `tests/dump_hf_ref.py`
  - `tests/test_tokids.cpp`

Generated artifacts are now handled more cleanly:

- `.gitignore` includes `*.weights` and `*.graph`
- tracked `*.weights` / `*.graph` files were removed from Git history
- new generated graph/weight artifacts should stay untracked locally

---

## What is still open

The main question is no longer "why is the tested prompt producing nonsense text?" for the currently checked path.

Instead, the remaining decisions are about project hygiene and broader coverage:

1. Decide whether `tests/test_e2e.cpp` should become a permanent committed regression test.
2. Decide whether `tests/dump_hf_ref.py` should remain a local debug tool or be committed as a reference/debug utility.
3. Extend validation to more prompts if desired:
   - longer generations
   - more chat cases
   - Chinese prompts
4. Clean up any temporary debug prints that are no longer needed.

---

## Recommended next step

Best next step:

- decide which of the local debug/test files should be promoted into the repo
- then run one more cleanup pass to remove temporary instrumentation that is no longer required for normal development
