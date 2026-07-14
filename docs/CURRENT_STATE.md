# Current State

*Last updated: 2026-07-14*

## 项目概述

**mollm** — mobile-oriented LLM 推理引擎，ARM NEON FP16FML kernels。
目前在 Apple Silicon 上开发和测试，移动端 ARM（Qualcomm Oryon、MediaTek）在 roadmap 中。
Python 转译前端 → `.mollm` 单文件打包 → C++ 执行器 + NEON FP16FML kernels。

支持的主要模型系列：
- **Qwen3 dense text models**（GQA）
- **Qwen3-30B-A3B MoE**（Qwen3-MoE，128 experts / top-8）
- **Qwen3.5-4B**（hybrid linear/full attention：Gated DeltaNet + GQA）
- **Qwen3.5-0.8B**（同架构）
- **Youtu-LLM-2B**（MLA）

## 性能（Apple M5 Pro, 4 threads, pp256 + tg64, warmup=3）

2026-07-06 严格复测：mollm 和 llama.cpp 都用 5 个独立进程取 median，
`mollm_bench --prompt-tokens 256 --max-new-tokens 64 --warmup 3 --threads 4`；
`llama-bench -ngl 0 -p 256 -n 64 -t 4 -r 1 -o json` 跑 5 次取 median。
llama.cpp 行统一使用 `../llama.cpp/build-cpu/bin/llama-bench`
（`GGML_BLAS=ON`, `GGML_METAL=OFF`, `GGML_CPU_REPACK=ON`）。4B/2B 包已重转为
当前显式 `lm_head.weights` 格式。

FP16 / F16:

| Model | mollm pp/tg | llama.cpp pp/tg | prefill gap | decode gap |
|-------|------------:|----------------:|------------:|-----------:|
| Qwen3.5-0.8B | 601.38 / 123.68 | 664.54 / 97.87 | 1.11x slower | 1.26x faster |
| Youtu-LLM-2B | 218.67 / 52.12 | 257.93 / 45.12 | 1.18x slower | 1.16x faster |
| Qwen3.5-4B | 102.48 / 25.00 | 142.44 / 22.33 | 1.39x slower | 1.12x faster |

W8PC / Q8_0:

W8PC rows are refreshed with the 2026-07-06 validation rerun using the current
AUTO-i8mm build with W8 GEMM 2D N-block scheduling; llama.cpp Q8_0 references
use the same BLAS-on, Metal-off build as the FP16 rows.

| Model | mollm W8PC pp/tg | llama.cpp Q8_0 pp/tg | prefill gap | decode gap |
|-------|-----------------:|---------------------:|------------:|-----------:|
| Qwen3.5-0.8B | 671.73 / 217.69 | 782.16 / 167.63 | 1.16x slower | 1.30x faster |
| Youtu-LLM-2B | 253.05 / 89.53 | 263.95 / 86.58 | 1.04x slower | 1.03x faster |
| Qwen3.5-4B | 118.55 / 46.64 | 135.58 / 40.50 | 1.14x slower | 1.15x faster |

W4G128 direct BG128 / Q4_0:

| Model | mollm W4G128 pp/tg | llama.cpp Q4_0 pp/tg | prefill gap | decode gap |
|-------|-------------------:|---------------------:|------------:|-----------:|
| Qwen3.5-0.8B | 678.41 / 259.43 | 775.95 / 190.89 | 1.14x slower | 1.36x faster |
| Youtu-LLM-2B | 248.08 / 115.64 | 265.58 / 97.15 | 1.07x slower | 1.19x faster |
| Qwen3.5-4B | 115.37 / 55.94 | 140.51 / 44.25 | 1.22x slower | 1.26x faster |

Qwen3-MoE W4G128 / Q4_0:

2026-07-14 fan-on strict rerun, 5 independent process medians. mollm uses
`qwen3_30b_a3b_w4g128.mollm`; llama.cpp uses locally converted
`Qwen3-30B-A3B-Q4_0.gguf` with the same BLAS-on, Metal-off CPU build. The
llama.cpp `--fuse-gate-up-exps` converter path failed for Qwen3Moe in this
local checkout (`Missing FFN_GATE_UP_EXP`), so the Q4_0 GGUF uses the default
separate gate/up expert tensors.

| Model | mollm W4G128 pp/tg | llama.cpp Q4_0 pp/tg | prefill gap | decode gap |
|-------|-------------------:|---------------------:|------------:|-----------:|
| Qwen3-30B-A3B | 142.92 / 65.66 | 110.34 / 60.77 | 1.30x faster | 1.08x faster |

Interpretation:
- FP16 decode remains faster than llama.cpp F16 on all three models in this
  rerun; prefill remains slower.
- Current W8PC decode is faster than llama.cpp Q8_0. AUTO i8mm plus W8 GEMM
  2D N-block scheduling narrowed W8PC prefill further: 2B is about
  1.04x slower than llama.cpp Q8_0, while 4B remains about 1.14x slower.
  0.8B remains noisy and is not the primary benchmark target.
- W4G128 direct BG128 now defaults to A4-packed activation quantization in GEMM.
  On the primary 2B/4B models this gives about 6-7% W4 prefill uplift versus
  the old non-A4 path. Prefill still trails llama.cpp Q4_0, while decode remains
  faster. W4 GEMM 2D N-block scheduling was tested after A4 but is kept opt-in
  (`MOLLM_W4_GEMM_2D=1`) because strict endpoint results did not show a stable
  default win. The 0.8B rows remain noisy and should not drive kernel decisions.
- Explicit `lm_head.weights` increases tied-embedding FP16 package/RSS: current 4B FP16 package is
  ~9.0 GiB on disk and peaks around 16.8 GB RSS in the latest full rerun. W8PC
  4B is ~5.1 GiB on disk and peaks around 11.0 GB RSS; W4G128 direct BG128 4B
  peaks around 5.0 GB RSS.
- Qwen3-30B-A3B MoE is the first large official Qwen3-MoE CPU baseline in this
  tree. With the current fused MoE op and aggregated expert package layout,
  mollm is faster than the local llama.cpp Q4_0 CPU baseline on both prefill
  and decode under the same pp256/tg64 protocol.

### 2026-07-01 内存管理分支 benchmark 审计

协议：pp256 + tg64, warmup=3, 4 threads, 5 runs median。结论：当前内存管理改动
没有相对 clean HEAD 造成可见性能回退。最新实现保留静态 decode graph 的
materialized workspace 跨 token 复用，也允许 dynamic prefill 在 runtime shape key
完全相同（`runtime_seq_len/static_padded/padded_seq_len/batch` 相同）时复用
materialized workspace。`reset()` 释放 decode workspace，但保留 prefill workspace；
`release_prefill_buffers()` 可显式释放 prefill workspace。当前机器状态下 4B 绝对值
低于历史 fan-max baseline，因此本节只用于同一时段 A/B，不替换上表 baseline。

| Model / package | Commit | pp/tg median | pool 状态 |
|---|---|---:|---|
| Qwen3.5-4B (`qwen35_4b_current_reconv.mollm`) | current + prefill/decode workspace reuse | 108.4 / 25.4 | prefill workspace ~3837.5 MB; decode workspace ~14.9 MB |
| Qwen3.5-4B (`qwen35_4b_current_reconv.mollm`) | current + decode workspace reuse | 109.7 / 25.0 | prefill active 0 MB; decode workspace ~14.9 MB |
| Qwen3.5-4B (`qwen35_4b_current_reconv.mollm`) | current before decode reuse | 106.4 / 23.8 | active 0 MB, acquire=release |
| Qwen3.5-4B (same package) | `5ecf403` clean HEAD | 104.5 / 23.8 | active ~7071 MB |
| Qwen3.5-4B (`qwen35_4b_cee9357.mollm`, old graph) | `cee9357` pre-dynamic | 108.0 / 24.5 | old bench has no pool stats |
| Qwen3.5-0.8B (`qwen35_0.8b.mollm`) | current memory branch | 612.7 / 100.2 | active 0 MB, acquire=release |
| Qwen3.5-0.8B (same package) | `5ecf403` clean HEAD | 610.8 / 99.1 | active ~2267 MB |
| Qwen3.5-0.8B (`qwen35_0.8b_cee9357.mollm`, old graph) | `cee9357` pre-dynamic | 618.2 / 98.1 | old bench has no pool stats |
| Youtu-LLM-2B (`youtu-llm-2b.mollm`) | current + prefill/decode workspace reuse | 236.7 / 52.0 | prefill workspace ~2038.0 MB; decode workspace ~7.7 MB |
| Youtu-LLM-2B (`youtu-llm-2b.mollm`) | current + decode workspace reuse | 235.6 / 51.9 | prefill active 0 MB; decode workspace ~7.7 MB |
| Youtu-LLM-2B (`youtu-llm-2b.mollm`) | current before decode reuse | 229.0 / 50.2 | active 0 MB, acquire=release |
| Youtu-LLM-2B (same package) | `5ecf403` clean HEAD | 211.4 / 47.0 | active ~23792 MB |
| Youtu-LLM-2B (same package) | `29afe99` dynamic initial | 209.0 / 46.3 | old bench has no pool stats |
| Youtu-LLM-2B (`youtu-llm-2b_cee9357.mollm`, old graph) | `cee9357` pre-dynamic | 225.3 / 48.4 | old bench has no pool stats |

Notes:
- Existing `qwen35_4b.mollm` from 2026-06-30 failed graph load; 4B was reconverted to
  `qwen35_4b_current_reconv.mollm` before measurement.
- pre-dynamic rows use old-format packages generated inside the `cee9357` worktree, so they
  include converter/graph format/runtime differences and are not isolated memory-management A/B.

### 2026-07-01 Dynamic shape 前后 benchmark 审计

协议：pp256 + tg64, warmup=3, 4 threads, 5 runs median。只测 2B/4B，跳过
0.8B（噪声较大）。`cee9357` 是 dynamic shape 前一提交；`29afe99` 是 dynamic
shape / Backend / graph v3 初版。每个提交使用该提交自己的 converter 重新生成
`.mollm` 包。

结论：**dynamic 初版提交确实有性能回退信号**。pp256 下 prefill dynamic 理论上不应减少
计算量（seq_len 仍为 256）；decode graph 仍是 seq=1 static，不做 runtime shape
injection。因此这不是“decode graph 做 dynamic”导致，而是该提交同时引入的通用
executor/Backend/graph-v3 生命周期路径带来的开销。

| Model | pre-dynamic `cee9357` | dynamic initial `29afe99` | delta |
|---|---:|---:|---:|
| Youtu-LLM-2B | 235.5 / 48.0 | 215.0 / 45.1 | pp -8.7%, tg -6.1% |
| Qwen3.5-4B | 107.7 / 24.3 | 103.0 / 22.0 | pp -4.3%, tg -9.5% |

Likely suspects in `29afe99`:
- `execute_graph()` treats `runtime_seq_len > 0` as dynamic and disables zero-copy view
  `skip_init`, causing every node to refresh shapes through the dynamic path.
- `inject_runtime_shapes()` patches INPUT shapes and stateful op params before every prefill.
- Backend dispatch adds an indirect call per node; small alone, but all nodes now run through it.
- Converter emits symbolic `SEQ/MUL` reshape dims in prefill graphs, expanding graph metadata and
  changing runtime shape handling while compute kernels are otherwise unchanged.

Root-cause analysis update:
- Main cause is **executor lifetime/reset semantics**, not `eval_dim()` arithmetic itself.
- `29afe99` added an execute-entry reset that clears every non INPUT/CONSTANT tensor's `data`.
  Because allocation still happened before dispatch, RESHAPE/PERMUTE/SLICE zero-copy view ops
  were repeatedly preallocated, then dispatch overwrote the tensor with a borrowed view. That
  created wasted pool allocations and destroyed warmup-time reuse.
- The same reset also forced non-contiguous RESHAPE materializations to be reallocated/rebuilt
  every execute call, which particularly hurts decode because decode calls the graph once per
  generated token.
- Experimental patch on `29afe99`:
  1. skip preallocation for PERMUTE/SLICE and contiguous RESHAPE,
  2. then disable the reset pass entirely as an unsafe upper-bound test.

| Model | dynamic initial | skip zero-copy prealloc | no-reset upper bound | pre-dynamic |
|---|---:|---:|---:|---:|
| Youtu-LLM-2B | 215.0 / 45.1 | 222.3 / 46.4 | 229.5 / 51.1 | 235.5 / 48.0 |
| Qwen3.5-4B | 103.0 / 22.0 | 104.8 / 23.5 | 109.9 / 24.3 | 107.7 / 24.3 |

Interpretation:
- Zero-copy prealloc waste explains a meaningful part of the regression.
- The rest is mostly lost reuse of materialized intermediate buffers after the reset pass.
- Decode graph itself remains static (`runtime_seq_len=-1`, no `inject_runtime_shapes()`), but it
  still goes through the same execute-entry reset and pre-dispatch allocation path, so the same
  buffer-reuse regression shows up in decode.
- Current memory-management branch fixes the unsafe zero-copy allocation behavior. Static decode
  graph now opts into explicit workspace reuse: borrowed views are cleared each execute, materialized
  pooled tensors are kept only when the graph has no runtime dynamic dims. Dynamic prefill uses a
  stricter same-shape policy: it keeps materialized pooled tensors only if the runtime shape key
  exactly matches the previous execution; otherwise the next execute entry releases the old
  workspace before allocating the new shape.
- Memory accounting note: repeated `release()` moves prefill buffers from `active` to BufferPool
  `freelist`; it does not return pages to the OS, so RSS/peak is roughly unchanged. With workspace
  reuse, 4B reports ~3837.5 MB prefill active and ~5.3 MB prefill freelist. With release-each-call,
  the same memory would appear as ~0 MB prefill active and ~3842.8 MB prefill freelist. Only
  `clear()` / engine teardown releases both active and freelist allocations to the system.

### DYNAMIC 模式短 prompt prefill 性能（Qwen3.5-0.8B, 4 threads）

Dynamic shape 模式启用后，短 prompt 不再 padding 到 256：

| Prompt tokens | prefill t/s | 相对 256 |
|---|---|---|
| 32 | 539 | 87% |
| 64 | 555 | 90% |
| 128 | 596 | 96% |
| 256 | 619 | 100% |

32-token prefill 从 STATIC_PADDED 模式的 ~413ms（padding 到 256）降到 59ms（**7x 加速**）。

## 优化进度（Qwen3.5-4B，尝试 5→13）

| 指标 | 尝试 5 | 尝试 12 | 尝试 13 | 累计 |
|------|--------|---------|---------|------|
| prefill_tps | 71 | 115 | 115 | **+62%** |
| decode_tps | 23 | 25 | 25 | +9% |

4B decode 已超 llama.cpp。prefill gap 1.25x，MATMUL-bound（87%）。

### 尝试 13：Dynamic shape schema + Backend 抽象 + chunked prefill 测试

架构性投资，为未来"runtime 按实际 token 数执行（无 padding）"和"多后端（CPU/NPU）支持"打基础：

1. **graph format v3**（不向前兼容 v2）：`GraphNode` 加 `dim_expr[4]` 字段（每维一个 `DimExpr`：CONST/SEQ/MUL/ADD/BATCH），序列化 8 字节/dim
2. **Symbolic DimExpr**（ONNX 风格）：支持 `SEQ`、`N*SEQ`（MUL）等复合维表达式。transpile 时 `propagate_dim_exprs()` 从 INPUT 节点传播到下游所有 tensor
3. **DYNAMIC 模式启用**：runtime 按 `eval_dim()` 求值，消除短 prompt 的 padding 浪费
4. **Backend 抽象**：`engine/backend.h` 新增 `Backend` 抽象基类 + `CPUBackend`。`.mollm` 严格后端中立，shape mode（DYNAMIC / STATIC_PADDED）由 backend 决定
5. **`.mollm` package 统一**：删除 artifacts_dir 加载路径，CLI/test 只接受 `--package`
6. **Test 5 chunked PPL**：256 token 拆多 chunk 验证跨 chunk KV/state 衔接

详见 `docs/OPTIMIZATION_LOG_QWEN35.md` 尝试 13。

### 关键优化（详见 OPTIMIZATION_LOG_QWEN35.md）

1. **GDN recurrence kernel 融合**：5-pass state 访问 → 2-pass（-60% 内存流量）
2. **GDN decode 多线程**：21.8% → 7.2% of decode time
3. **full_attn 冗余 CONTIGUOUS 消除**：验证 rms_norm/rope 支持 strided，删除 5 处冗余
4. **RESHAPE/CONTIGUOUS materialize**：行级 memcpy fast path（-94%）
5. **FP16FML matmul**：988 GF/s microbench（86% FP16FML peak），lane-FMA + 2-way K-unroll GEMM，8-way K-unroll GEMV
6. **SHORTCONV prefill 多线程 + seq-outer 拷贝**
7. **GDN RMSNormGated NEON sigmoid 向量化**
8. **Dynamic shape（尝试 13）**：DimExpr symbolic + DYNAMIC 模式 + Backend 抽象

## 当前 Profiling（4B prefill，~2260ms）

| Op | ms | pct | 状态 |
|----|----|-----|------|
| MATMUL | 1923 | 87% | 988 GF/s microbench，~393 GF/s e2e（cache/DRAM bound） |
| GDN_prefill | 150 | 6.9% | Fused + 多线程 + NEON sigmoid |
| SHORTCONV | 53 | 2.5% | 多线程 + seq-outer 拷贝 |
| SDPA | 24 | 1.1% | — |
| SILU | 19 | 0.9% | NEON 向量化 |
| RESHAPE/CONTIGUOUS | 5 | 0.3% | 行级 memcpy |
| Other | <15 | <1% | — |

**MATMUL 占 87%**，算子已到顶（988 GF/s），端到端仅 393 GF/s（2.5x overhead 来自 cache/DRAM 带宽，4B 权重 280MB >> L2 16MB）。

## 架构摘要

### 流水线
1. **Python 转译**（`models/converter.py` 自动分发 → `qwen35.py` / `mla.py`）：PyTorch weights → `.mollm` 单文件
2. **C++ 执行器**（`graph/execute.cpp`）：顺序节点 dispatch + BufferPool 内存管理 + Backend 抽象
3. **NEON kernels**（`kernels/`）：matmul, attention, gdn, norm, rope

### Quantization（W8 q8dot + W4 q4dot package layout）

2026-07-01 已加入 correctness-first W8 weight-only 量化；2026-07-02 已加入
W4 per-group correctness baseline；2026-07-03 已把 W4 q4dot layout 下沉到
`.mollm` package，避免 runtime 常驻第二份 W4 repack 权重；2026-07-05
W4G128 进一步默认写入 direct BG128 package layout，让 prefill/decode 都能使用
grouped B-side layout 而不再复制 q4 权重：

- converter 支持 `quant=none|w8pc|w8gN|w4gN|w4mixgN`
  （如 `w8g128` / `w4g128` / `w4mixg128`）
- 当 CMake build 里存在 `mollm-quantize` 时，converter 会自动用 C++
  helper 生成 W8/W4 `.weights`。W8 仍保留 Python fallback；W4 不再保留
  Python fallback，缺少 helper 会直接报错。该 helper 直接
  输出当前 `.weights` header、W8 row-major data、W4 q4dot 或 W4G128 BG128
  physical layout，不改变 runtime 格式。
  因此普通 FP16/W8 转换不强制先编译 C++，但 W4 转换必须先 build helper：
  `cmake --build build --target mollm-quantize`。
- `.weights` header 已使用 `scales_offset/scales_size/group_size/num_groups`
  表达 per-channel/per-group scale metadata
- runtime 从 weight header 读取真实 precision；INT8/INT4 CONSTANT 会校验 scale
  metadata，INT4 data size 允许旧 row-major `N * ceil(K/2)`、带
  `FLAG_INT4_Q4DOT` 的 q4dot physical layout，或带 `FLAG_INT4_BG128` 的
  direct BG128 physical layout
- `kernel_matmul_fp32()` 已有 INT8 scalar fallback、native W8 NEON dequant kernel、
  decode/GEMV 默认 Q8 activation dot path，以及 W4 scalar fallback
- Qwen3.5 / Youtu converter 保留 `embed_tokens.weights` 为 FP16 row-major
  lookup；`lm_head.weights` 现在显式存储在包内，并按普通 linear weight 的量化
  policy 处理（W8 包为 W8，W4G 包为 W4，`w4mixgN` 提升为 W8）
- norm、conv、`A_log`、`dt_bias` 等非 projection 权重保持原精度

Runtime 不再从 tied embedding 临时创建 lmhead copy；`.mollm` 包必须包含显式
`lm_head.weights`，旧包需要用当前 converter 重新生成。

W8 默认路径已按 llama.cpp/ggml 的方向改成：

```text
FP32 activation -> temporary Q8 blocks -> INT8 weight x Q8 activation dot -> FP32 output
```

decode/GEMV 和 prefill/GEMM 默认都使用 q8dot repacked layout
`[N/8, K/32, 8, 32]` + `vdotq_s32`。旧 `[N/8, K, 8]` INT8 interleaved copy
不再默认常驻；仅在 fallback/diagnostic 模式或 q8dot unsupported shape 下创建。

W4G 权重仍是 signed int4 nibble packed，header 保留逻辑 shape `[N,K]`。
当前 converter 对 `w4g128` tensor 直接把物理 data 写成 BG128 layout
`[ceil(N/8), K/128]` blocks（`float scales[8] + q4dot q[4][8][16B]`），
并设置 header flag bit1 `FLAG_INT4_BG128`。Runtime 看到该 flag 后直接把
mmap data 作为 `q4_g128_data`，prefill/GEMM 和 decode/GEMV 都默认走 BG128
kernel，不再创建 runtime sidecar copy。非 BG128 但 K/group 都 32 对齐的 W4
tensor 仍可写成 q4dot layout `[ceil(N/8), K/32, 8, 16B]` 并设置
`FLAG_INT4_Q4DOT`。旧 row-major W4 包（flag=0）仍可在 load time 创建 q4dot
repack 作为兼容路径。Direct q4dot/BG128 包需要 ARM DOTPROD build，不能通过
`MOLLM_W4_NO_Q4_REPACK=1` 强制回到 row-major/scalar。

默认 W4 matmul 使用 Q8 activation dot path：FP32 activation 先按 32-K block
量化为 Q8，W4 nibble 在 kernel 内 sign-extend 成 int8，再用 `vdotq_s32`
计算 q4 x q8 dot，最后乘 `a_scale * w_scale` 写回 FP32。旧 row-major 包的
odd K 或 group 不满足 32 对齐时仍回退 scalar nibble unpack + FP32 accumulate。
W4MIXG128 的 W8 tensors 仍走 W8 q8dot repacked path。Qwen3.5 mixed policy
提升 `lm_head`、linear-attn QKV/out、full-attn V/out 和 MLP down；Youtu
mixed policy 提升 `lm_head`、MLA `kv_b_proj`、attention output 和 MLP down。

Qwen3.5-0.8B quick validation：

| Package | Size | Short CE/PPL | CE delta vs FP16 | Greedy smoke |
|---|---:|---:|---:|---|
| FP16 `qwen35_0.8b.mollm` | 2027.5 MB | 3.0911 / 22.00 | - | - |
| W8PC `qwen35_0.8b_w8pc.mollm` | 1278.1 MB | 3.1098 / 22.42 | +0.0187 | pass |
| W8G128 `qwen35_0.8b_w8g128.mollm` | 1299.1 MB | 3.1107 / 22.44 | +0.0195 | pass |
| W4G128 `qwen35_0.8b_w4g128.mollm` | 923.1 MB | 3.4881 / 32.73 | +0.3970 | pass |
| W4MIXG128 `qwen35_0.8b_w4mixg128.mollm` | 1177.6 MB | 3.1804 / 24.06 | +0.0892 | pass |

256-token PPL audit on the same reference tokens:

| Package | 256-token CE/PPL | CE delta vs FP16 |
|---|---:|---:|
| FP16 `qwen35_0.8b.mollm` | 2.1388 / 8.49 | - |
| W8PC `qwen35_0.8b_w8pc.mollm` | 2.1475 / 8.56 | +0.0086 |
| W8G128 `qwen35_0.8b_w8g128.mollm` | 2.1416 / 8.51 | +0.0028 |
| W4G128 `qwen35_0.8b_w4g128.mollm` | 2.4239 / 11.29 | +0.2851 |
| W4MIXG128 `qwen35_0.8b_w4mixg128.mollm` | 2.1684 / 8.74 | +0.0296 |

Qwen3.5-0.8B quantized runtime baseline（default Release build,
`MOLLM_ARM_I8MM=OFF`, pp256 + tg64, warmup=3, 4 threads, 5-run median）：

| Package | pp/tg | prefill ms | decode ms | peak RSS |
|---|---:|---:|---:|---:|
| FP16 `qwen35_0.8b.mollm` | 625.36 / 101.74 | 409.36 | 619.21 | 3994.9 MB |
| W8PC `qwen35_0.8b_w8pc.mollm` | 396.78 / 157.34 | 645.19 | 400.40 | 2561.9 MB |
| W8G128 `qwen35_0.8b_w8g128.mollm` | 392.36 / 148.78 | 652.46 | 423.45 | 2581.5 MB |
| W4G128 direct q4pkg `qwen35_0.8b_w4g128_q4pkg.mollm` | 385.88 / 154.88 | 663.42 | 406.77 | 1500.7 MB |
| W4MIXG128 direct q4pkg `qwen35_0.8b_w4mixg128_q4pkg.mollm` | 370.07 / 143.38 | 691.76 | 439.39 | 2232.8 MB |

2026-07-02 follow-up A/B invalidated an earlier FP16 decode result around
`~128 tok/s`. Same-session 5-run median gives current explicit-lmhead FP16
`625.36 / 101.74`; old pre-dynamic `cee9357` with its old-format package gives
`624.91 / 101.14`. Both runs generated 64 tokens (`decode_tokens=63`) and did
not hit EOS. Profile also matches: current FP16 `decode_lmhead` uses
`fp16_gemv_interleaved_fp16acc`, and total decode MATMUL is the same order as
`cee9357`. So explicit `lm_head.weights` does not explain the transient
`~128 tok/s` reading; treat that number as a bad benchmark sample. The likely
source is a label/transcription mix-up with an older W8PC `default q8dot repack`
single run (`376.1 / 126.8`, `decode_ms=496.77`), which numerically matches the
invalid FP16 row's decode time.

W4 q4dot is now the default W4 path when K/group are 32-aligned. The old
load-time q4 repack restored 0.8B W4G128 from the scalar baseline around
`9.47 / 9.09` tok/s to `386.71 / 153.52`, but raised RSS from `1500.8 MB` to
`1862.7 MB` because runtime kept a second q4dot-layout copy. The new package
layout keeps the same speed class while removing that duplicate copy:
W4G128 direct q4pkg measures `385.88 / 154.88`, RSS `1500.7 MB`; W4MIXG128
direct q4pkg measures `370.07 / 143.38`, RSS `2232.8 MB`. Pure W4 quality is
still poor because `lm_head.weights` is W4; W4MIX keeps quality near W8 by
promoting 73 tensors, including `lm_head.weights`, to W8.

2026-07-03 follow-up measured Youtu-LLM-2B W4G128 direct q4pkg under the same
strict pp256/tg64 protocol, but in a lower-prefill-throughput machine state
than the 2026-07-02 baseline. The first table is the package-layout baseline
before the W4 GEMV kernel cleanup:

| Youtu-LLM-2B package | pp/tg median | load ms | peak RSS | 256-token CE/PPL |
|---|---:|---:|---:|---:|
| FP16 `youtu-llm-2b.mollm` | 212.14 / 50.97 | 951.33 | 9490.7 MB | 3.1139 / 22.51 |
| W8PC `youtu-llm-2b_w8pc.mollm` | 114.73 / 88.78 | 507.47 | 5948.3 MB | 3.1094 / 22.41 |
| W4G128 direct q4pkg `youtu-llm-2b_w4g128_q4pkg.mollm` | 108.31 / 70.20 | 110.47 | 2893.5 MB | 3.1578 / 23.52 |

2B pure W4 quality is much better than Qwen3.5-0.8B pure W4 on this reference
sample (`CE delta +0.0440` vs FP16; W8PC rerun is `3.1094 / 22.41`,
delta `-0.0045`), and package-level q4dot keeps RSS/load very low.

After W4 GEMV scale/qA preparation cleanup, 2B was rerun in the same machine
session. The code change is decode/GEMV-only: activation Q8 blocks are produced
directly in even/odd q4dot order, and W4 per-group scales are loaded once per
group instead of once per qblock. A similar GEMM scale-loop experiment regressed
prefill and was reverted.

| Youtu-LLM-2B package | pp/tg median | load ms | peak RSS | note |
|---|---:|---:|---:|---|
| W4G128 direct q4pkg | 118.39 / 89.61 | 114.80 | 2893.6 MB | after W4 GEMV cleanup |
| W8PC `youtu-llm-2b_w8pc.mollm` | 117.25 / 87.84 | 510.56 | 5948.9 MB | same-session comparison |

Interpretation: after the GEMV cleanup, 2B W4G128 direct q4pkg is now
W8PC-class on decode in this same-session strict run while using roughly half
the RSS and much lower load time. Prefill remains a W4 GEMM problem and still
needs a separate kernel/layout optimization.

2026-07-03 follow-up added a W4 q8dot GEMM `8x8` prefill kernel. The previous
dispatcher tiled `M` by 8 but still ran the internal `4x8` kernel twice, so each
8-row tile unpacked the same q4 B block twice. The new kernel unpacks B once and
reuses it across 8 rows. A 16-column GEMV experiment and NEON qA quantization
experiment were both reverted after real package profile failed to show stable
decode wins.

Dominant Youtu W4 prefill microbench shapes, 4 threads, `warmup=10`,
`repeat=20`:

| Shape `(M,K,N)` | 4x8 fallback p50 | 8x8 p50 |
|---|---:|---:|
| `256,2048,12288` | 25.27 ms | 22.17 ms |
| `256,6144,2048` | 12.68 ms | 11.67 ms |

Strict package rerun for Youtu-LLM-2B W4G128 direct q4pkg, pp256 + tg64,
warmup=3, 4 threads, 5 independent process median:

| Package | pp/tg median | load ms | peak RSS | 256-token CE/PPL |
|---|---:|---:|---:|---:|
| W4G128 direct q4pkg after GEMM 8x8 | 131.73 / 91.38 | 116.17 | 2893.6 MB | 3.1578 / 23.52 |

Interpretation: the new 8x8 GEMM mainly helps prefill; decode remains governed
by W4 GEMV q4 unpack + SDOT and only improves slightly from scratch-buffer
reuse/noise. `MOLLM_W4_Q8_DOT_GEMM_4X8=1` keeps the old 4x8 GEMM available for
A/B.

Fan-max same-binary alternating A/B gives a cleaner endpoint comparison:
default 8x8 `137.26 / 92.38` pp/tg, old 4x8 fallback `120.10 / 92.74`.
So the current confirmed gain is about +14% prefill and neutral decode.

2026-07-04 follow-up kept the same W4 q4dot package format and rewrote the
8x8 GEMM reduction path to stay in NEON vectors instead of doing per-output
`vaddvq_s32` scalar reductions. The experimental W4 i8mm path was also tested:
it unpacks q4 blocks into int8 micro-panels at runtime and feeds `vmmlaq_s32`,
but this is slower than the default DOTPROD kernel on current shapes, so it is
kept opt-in only via `MOLLM_W4_I8MM=1`. A later same-day update changed the
default DOTPROD path to use scaled signed nibbles (`q4 * 16`) directly, matching
part of llama.cpp's packed-nibble trick and dividing the dot result by 16 during
int32-to-float conversion. This avoids full sign-extension in the W4 DOTPROD
hot path without changing package layout or quantization math. The current best
path also specializes full 8-column output tiles, removing `c_valid` branches
and scalar tail stores from the common W4 GEMM hot path.

W4 prefill microbench, 4 threads, `warmup=10`, `repeat=20`:

| Shape `(M,K,N)` | full-tile scaled-nibble DOTPROD | W4 i8mm opt-in | old 4x8 fallback |
|---|---:|---:|---:|
| `256,2048,12288` | 13.25 ms / 968.0 GF/s | 15.85 ms / 811.6 GF/s | 24.63 ms / 522.7 GF/s |
| `256,6144,2048` | 7.07 ms / 908.0 GF/s | not rerun | not rerun |
| `256,2560,9216` | 12.38 ms / 971.8 GF/s | not rerun | not rerun |
| `256,9216,2560` | 13.15 ms / 922.8 GF/s | not rerun | not rerun |

Current same-binary all-model W4G128 rerun after W4 DOTPROD vector-reduce,
scaled-nibble, full-tile GEMM cleanup, and GEMV full-tile cleanup, pp256 +
tg64, warmup=3, 4 threads, 5 independent process median:

| Model / package | pp/tg median | prefill ms | decode ms | load ms | peak RSS | 256-token CE/PPL |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B W4G128 direct q4pkg | 611.13 / 165.53 | 418.90 | 380.60 | 201.03 | 1501.3 MB | 2.4239 / 11.29 |
| Youtu-LLM-2B W4G128 direct q4pkg | 232.76 / 97.65 | 1099.84 | 645.13 | 105.70 | 2892.8 MB | 3.1578 / 23.52 |
| Qwen3.5-4B W4G128 direct q4pkg | 104.19 / 46.16 | 2457.11 | 1364.82 | 209.54 | 4984.5 MB | 1.9568 / 7.08 |

2026-07-04 current AUTO-i8mm build W8PC vs W4G128 comparison after W8 i8mm
8-row GEMM switch, same protocol (`pp256 + tg64`, warmup=3, 4 threads,
5 independent process median). W4 rows are the latest recorded W4G128 medians:

| Model | W8PC pp/tg | W4G128 pp/tg | W8/W4 prefill | W8/W4 decode | W8 RSS | W4 RSS |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 622.17 / 148.88 | 611.13 / 165.53 | 1.02x | 0.90x | 2561.5 MB | 1501.3 MB |
| Youtu-LLM-2B | 253.76 / 87.37 | 232.76 / 97.65 | 1.09x | 0.89x | 5949.2 MB | 2892.8 MB |
| Qwen3.5-4B | 117.46 / 42.39 | 104.19 / 46.16 | 1.13x | 0.92x | 10964.5 MB | 4984.5 MB |

Interpretation: W4 DOTPROD vector-reduce recovered much of the W4 prefill gap,
and the W4 GEMV full-tile cleanup brings 2B/4B W4 decode to llama.cpp Q4_0
range. W8PC remains faster for 2B/4B prefill because it can use i8mm GEMM,
while W4G128 now wins decode on all three local W8/W4 comparisons and uses much
lower RSS. Scaled-nibble DOTPROD improves dominant W4 GEMM shapes by a few
percent, but does not change the structural conclusion: closing the remaining
W4 prefill gap likely needs a true packed-i8mm/asm-style W4 kernel rather than
runtime unpacking. The W8 8-row i8mm GEMM default closes part of the W8 prefill
gap; see `docs/OPTIMIZATION_LOG_QUANT.md` Attempt 26 and W4 GEMV cleanup notes
in Attempt 27.

2026-07-04 W4 GEMV follow-up adds a NEON full-block activation quantizer for
the `M==1` even/odd q4dot path. It is default-on and can be disabled with
`MOLLM_W4_GEMV_SCALAR_QA=1`. Youtu-LLM-2B short profile improves from
`97.86` to `102.24` decode tok/s, and same-machine strict decode median moves
from `97.65` to `98.71`; the strict prefill row in that rerun was affected by
machine-state slowdown, so the main W4 table above is kept as the stable
baseline until a full fan-max rerun. See `docs/OPTIMIZATION_LOG_QUANT.md`
Attempt 28.

2026-07-04 later follow-up vectorized the shared GEMM activation Q8 quantizer
used by W8/W4 prefill. `MOLLM_Q8_GEMM_SCALAR_QA=1` forces the old scalar path
for A/B. Qwen3.5-4B short profiles dropped `q8_quant_a_ms` from roughly
`82 ms -> 34 ms` for W8PC and `119 ms -> 34 ms` for W4G128; strict pp256+tg64
4B medians in the same post-change session were W8PC `124.06 / 42.69` and
W4G128 `108.28 / 47.38`. Youtu-LLM-2B W8PC prefill was neutral in the same
session (`251.69` vs recorded `253.76`) while decode was noisy, so the all-model
tables are not fully replaced until a clean fan-stable rerun. After this change
GEMM activation quantization is about `1.5-2%` of matmul time, not the main
remaining prefill bottleneck. See `docs/OPTIMIZATION_LOG_QUANT.md` Attempt 29.

2026-07-05 follow-up added two smaller cleanups. GDN prefill now groups
repeat-value-heads by key head and reuses the normalized q/k vectors; the new
unit coverage explicitly tests `num_v_heads != num_heads`. More importantly for
W4, the default q4dot 8x8 GEMM caches per-group B scale loads, so W4G128 loads
one scale vector per 128-wide group instead of once per 32-wide Q8 block.
Same-session Qwen3.5-4B W4G128 strict median moved from `104.82 / 47.08` to
`109.02 / 48.00` pp/tg, and short profile dominant W4 GEMM shapes improved
from roughly `455-463` to `483-491 GMAC/s`. See
`docs/OPTIMIZATION_LOG_QUANT.md` Attempt 30.

A later W4 micro-tune tried changing the 8x8 q4dot GEMM scale handling from
cache+branch inside each qblock to explicit `PerChannel` / `PerGroup` /
`PerBlock32` loops. Fan-on same-session endpoint A/B did not support keeping it
(`106.90 / 46.76` vs `107.19 / 47.13` pp/tg for the Attempt 30 cache+branch
path), so the code remains on the Attempt 30 structure. See
`docs/OPTIMIZATION_LOG_QUANT.md` Attempt 31.

CPU-only llama.cpp no-BLAS/no-Metal check（local `../llama.cpp` `5c7c22c3e` /
build 9803, `GGML_BLAS=OFF`, `GGML_METAL=OFF`, `GGML_CPU_REPACK=ON`,
`pp256 + tg64`, 4 threads, `-r 3`）:

| GGUF | pp256 | tg64 |
|---|---:|---:|
| Youtu-LLM-2B Q4_0 | 264.55 | 95.91 |
| Youtu-LLM-2B Q8_0 | 233.84 | 80.55 |
| Qwen3.5-4B Q4_0 | 139.06 | 43.15 |
| Qwen3.5-4B Q8_0 | 128.50 | 39.46 |

Later same-session rerun after W8/W4 cleanup:

| GGUF | pp256 | tg64 |
|---|---:|---:|
| Youtu-LLM-2B Q4_0 | 281.04 | 99.82 |
| Youtu-LLM-2B Q4_K_M | 216.35 | 88.72 |
| Youtu-LLM-2B Q8_0 | 275.90 | 83.99 |
| Qwen3.5-4B Q4_0 | 148.67 | 46.43 |
| Qwen3.5-4B Q8_0 | 142.88 | 39.77 |

This confirms llama.cpp's Q4_0/Q8_0 prefill advantage is not just Apple BLAS;
its CPU repack/i8mm kernels are materially stronger. Current mollm decode is
competitive on 2B/4B, but prefill still trails, especially 4B (`W4 104.19` vs
`Q4_0 148.67`, `W8 117.46` vs `Q8_0 142.88`).

2026-07-06 llama.cpp BLAS vs no-BLAS A/B used two Metal-off builds:
`build-cpu` (`GGML_BLAS=ON`, `GGML_METAL=OFF`, `GGML_CPU_REPACK=ON`) and
`build-cpu-noblas` (`GGML_BLAS=OFF`, `GGML_METAL=OFF`,
`GGML_CPU_REPACK=ON`). Protocol was pp256 + tg64, 4 threads, 5 independent
process medians. Result: BLAS is crucial for F16 prefill, but Q8_0/Q4_0 are
essentially unchanged because the quantized CPU repack kernels do not depend on
BLAS for these shapes.

| Model | Quant | BLAS pp/tg | no-BLAS pp/tg | effect |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | F16 | 725.24 / 98.78 | 290.76 / 98.75 | BLAS pp 2.49x |
| Qwen3.5-0.8B | Q8_0 | 805.81 / 168.97 | 800.41 / 169.06 | neutral |
| Qwen3.5-0.8B | Q4_0 | 802.66 / 186.09 | 804.59 / 188.18 | neutral |
| Youtu-LLM-2B | F16 | 257.42 / 46.49 | 92.30 / 46.43 | BLAS pp 2.79x |
| Youtu-LLM-2B | Q8_0 | 270.41 / 85.26 | 268.18 / 85.72 | neutral |
| Youtu-LLM-2B | Q4_0 | 269.76 / 98.32 | 268.89 / 98.44 | neutral |
| Qwen3.5-4B | F16 | 141.95 / 22.09 | 42.83 / 21.87 | BLAS pp 3.31x |
| Qwen3.5-4B | Q8_0 | 136.32 / 39.04 | 134.91 / 39.55 | neutral |
| Qwen3.5-4B | Q4_0 | 143.89 / 44.23 | 143.64 / 44.72 | neutral |

2026-07-05 llama.cpp gap survey: the main remaining quantized-prefill gap looks
structural. llama.cpp's CPU repack path uses packed B formats with embedded
fp16 scales (`block_q4_0x8`, `block_q8_0x4`), packs activation rows into
matching Q8 blocks, chooses NEON/i8mm/SVE traits by CPU feature, and runs
tile-aware work scheduling inside the repack matmul path. mollm's current W4G128
path still uses q4dot data plus side-scale arrays and a generic matmul entry.
The next meaningful direction is a new packed quantized-matmul pipeline
(`q4g128x8`/`q8_0x4`-style A/B layouts plus i8mm kernel and tile-aware
scheduling), while keeping the current DOTPROD W4G128 path as the fallback.

2026-07-05 packed-A4 activation experiment: `MOLLM_W4_PACKED_A4=1` initially
routed W4 prefill GEMM through a packed 4-row Q8 activation layout (pre-split
even/odd streams, `float` scales). That early opt-in result was noisy, but the
2026-07-06 direct-BG128 rerun showed stable 6-7% prefill gain on 2B/4B, so A4
is now the default W4 GEMM path and `MOLLM_W4_NO_PACKED_A4=1` is the rollback
switch. `MOLLM_W4_I8MM=1` plus A4 improves the existing i8mm prototype by ~5%
synthetic, but it remains slower than DOTPROD+A4, so current i8mm still needs a
true B-side packed layout. See `docs/OPTIMIZATION_LOG_QUANT.md` Attempts 33 and
38.

2026-07-05 packed-BG128 update: runtime sidecar `MOLLM_W4_PACKED_BG128=1`
proved the layout, then converter/runtime were updated so new `w4g128` packages
store BG128 directly with `FLAG_INT4_BG128`. 4B direct BG128 package
`qwen35_4b_w4g128_bg128.mollm` is 3653.6 MB vs old q4dot package 3522.2 MB,
but peak RSS is back to ~4987 MB instead of sidecar ~7166 MB. Latest 4B direct
BG128 fan-on validation rerun is `110.78 / 55.75`; earlier direct BG128+A4 was
`117.21 / 56.86` in the same machine session. 0.8B direct BG128 keeps 128-token CE/PPL
`2.6224 / 13.77` and defaults to `q4dot_gemm_bg128` / `q4dot_gemv_bg128`
without env flags. See `docs/OPTIMIZATION_LOG_QUANT.md` Attempts 34-35.

Fan-on all-model direct BG128 W4G128 rerun (`pp256 + tg64`, warmup=3, 4 threads,
5 independent process median):

| Model | mollm W4G128 direct BG128 pp/tg | llama.cpp Q4_0 pp/tg | prefill gap | decode gap |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 687.32 / 252.91 | 800.05 / 188.72 | 1.16x slower | 1.34x faster |
| Youtu-LLM-2B | 241.99 / 114.03 | 269.87 / 97.39 | 1.12x slower | 1.17x faster |
| Qwen3.5-4B | 110.78 / 55.75 | 143.46 / 44.32 | 1.29x slower | 1.26x faster |

Interpretation: direct BG128 makes W4 decode clearly faster than llama.cpp
Q4_0 on all three local models and brings 2B W4 prefill near llama.cpp Q4_0.
4B W4 prefill still has the largest remaining gap.

llama.cpp same-session CPU baseline（`../llama.cpp` `5c7c22c3e` / build 9803,
`llama-bench -ngl 0 -p 256 -n 64 -t 4 -r 5`，llama-bench reported summary）：

| GGUF | Size | pp/tg |
|---|---:|---:|
| Qwen3.5-0.8B F16 | 1475.05 MiB | 735.34 / 99.27 |
| Qwen3.5-0.8B Q8_0 | 784.52 MiB | 819.99 / 171.91 |
| Qwen3.5-0.8B Q4_0 | 478.76 MiB | 809.47 / 189.80 |
| Qwen3.5-0.8B Q4_K_M | 506.34 MiB | 639.04 / 173.60 |

Comparison: after strict rerun, 0.8B W8PC/W8G128 decode is below llama.cpp Q8_0
(`157.34/148.78` vs `171.10` tok/s). The earlier `~214/206` W8 decode rows are
not reproducible under the current strict protocol. W4G128 q4dot decode is now
near W8G128, but still below llama.cpp Q4_0/Q4_K_M decode (`189.80/173.60` tok/s).
llama.cpp Q4_K_M also quantizes token embeddings to q6_K; mollm keeps
`embed_tokens` FP16 for lookup and stores `lm_head` separately.

Youtu-LLM-2B llama.cpp CPU comparison after the W4 GEMV cleanup
（`../llama.cpp/build-cpu/bin/llama-bench -ngl 0 -p 256 -n 64 -t 4 -r 5 -o md`）：

| Runtime / package | Size | pp256 | tg64 |
|---|---:|---:|---:|
| mollm W4G128 direct q4pkg | 1577.9 MB | 118.39 | 89.61 |
| llama.cpp F16 | 3.65 GiB | 253.29 ± 10.31 | 45.04 ± 2.48 |
| llama.cpp Q8_0 | 1.94 GiB | 259.58 ± 1.35 | 85.09 ± 1.69 |
| llama.cpp Q4_0 | 1.09 GiB | 255.28 ± 2.17 | 90.35 ± 5.35 |
| llama.cpp Q4_K_M | 1.14 GiB | 187.96 ± 1.30 | 82.53 ± 0.63 |

Notes: llama.cpp uses BLAS backend in this build, so its pp256 prefill is much
faster. mollm W4G128 decode is now in the same range as llama.cpp Q4_0/Q4_K_M
and Q8_0, but prefill is still about 2.2x slower than llama.cpp Q4_0/Q8_0.
The new Youtu Q4 GGUFs were generated locally from F16 with llama.cpp
`llama-quantize`; Q4_0 keeps token embedding as q6_K by llama.cpp policy, and
Q4_K_M falls back 32 `attn_k_b` tensors to q5_0 because their ncols are not
q4_K-compatible.

`test_quantized_e2e` 覆盖 quantized package load、finite logits、short CE drift、
workspace presence 和短 decode。W4 单测还覆盖 odd-K packed nibble tail。详细设计见
`docs/QUANTIZATION.md`。

2026-07-01 后续已开始 W8 kernel 优化，详见 `docs/OPTIMIZATION_LOG_QUANT.md`：

- 加入 INT8 B interleaved packing、W8 NEON GEMV、W8 NEON 4x8 GEMM
- decode/GEMV 和 prefill/GEMM 默认启用 repacked Q8 activation dot；
  `MOLLM_W8_NO_Q8_DOT_REPACK=1` 可回退旧 interleaved layout，
  `MOLLM_W8_NO_Q8_DOT=1` 可回退 native W8 dequant 路径
- microbench 中 GEMV 从 ~9 GF/s 提升到 ~70 GF/s；GEMM 从 ~10 GF/s 提升到
  ~210-230 GF/s
- Q8-dot GEMV microbench 进一步到 ~158 GF/s；旧 W8 GEMV 同形状约 ~65 GF/s
- repacked Q8-dot GEMV opt-in path 使用 `[N/8, K/32, 8, 32]` layout +
  `vdotq_s32`，microbench `M=1 K=2048 N=6144` 从 0.15ms 降到 0.10ms
- repacked Q8-dot GEMM 现在是 W8 默认 prefill 路径；严格复测下
  0.8B W8PC 当前 5-run median 约 397 / 157 pp/tg
- ARM i8mm (`vmmlaq_s32`) q8dot GEMM 已实现；`-DMOLLM_ARM_I8MM=ON` build
  下 2B/4B dominant prefill microbench 约 1.5x，整包单次 fresh run prefill
  提升约 61-64%，decode 基本不变
- q8dot GEMV 已去掉 per-column `vaddvq_s32` 标量化；W8 scale mode 明确拆成
  per-channel / true per-block(32) / per-group。`-DMOLLM_ARM_I8MM=ON` 单次
  fresh run 中 2B/4B decode 已基本追上或超过 llama.cpp Q8_0 参考
- 显式 W8 `lm_head.weights` 后，0.8B package W8PC 当前 5-run median 约
  397 / 157 pp/tg，W8G128 约 392 / 149 pp/tg；profile 中 `prefill_lmhead`
  和 `decode_lmhead` 均走 `q8dot_gemv_repack`
- FP16FML prefill 仍明显更快（FP16 0.8B package 当前约 622 / 101 pp/tg）；
  W8PC decode 已快于 FP16 decode
- W8 0.8B peak RSS 默认约 2.56-2.58 GB；q8dot repack 已替换旧 INT8
  interleaved copy，且 runtime 不再为 tied lmhead 额外创建 FP16 packed copy

2026-07-03 W4 follow-up：direct q4pkg 之后又优化了 W4 GEMV 的 Q8 activation
准备和 per-group scale handling。Youtu-LLM-2B W4G128 strict rerun 到
`118.39 / 89.61` pp/tg，同场 W8PC 为 `117.25 / 87.84`，RSS 分别约
`2.9 GB` / `5.9 GB`。这说明 2B W4 decode 当前已是 W8PC-class；剩余 W4
性能工作重点转向 prefill GEMM。

2026-07-04 W4 GEMM 后续优化把 8x8 q4dot 路径里的 per-output scalar
`vaddvq_s32` 改成 NEON pairwise vector reduction。Youtu-LLM-2B W4G128
strict median 更新到 `196.56 / 90.20` pp/tg，4B W4G128 到
`85.79 / 39.55`。W4 i8mm 原型（runtime 展开 q4 为 int8 micro-panel 后
用 `vmmlaq_s32`）已验证正确，但当前 microbench 比 DOTPROD vector-reduce 慢，
因此仅作为 `MOLLM_W4_I8MM=1` 实验路径保留，不默认启用。随后 scaled-nibble
DOTPROD 和 full-tile fast path 把主形状 microbench 推到约 `908-972 GF/s`，
并保持 Youtu-LLM-2B W4G128 256-token CE/PPL `3.1578 / 23.52` 不变；详见
`docs/OPTIMIZATION_LOG_QUANT.md` Attempt 23/24。

2026-07-04 W8 i8mm 后续 cleanup 专门处理 full 4x8 tile：完整 tile 下去掉
per-row validity checks，并用 vector store 直接写回，保持尾部路径不变。随后
8-row i8mm GEMM 原型验证通过并切为默认：每个 8-row tile 对同一 B micro-panel
只加载一次，旧 4x8 可用 `MOLLM_W8_I8MM_4X8=1` 回退。W8PC strict median
更新为：0.8B `622.17 / 148.88`（noisy），Youtu-LLM-2B `253.76 / 87.37`，
Qwen3.5-4B `117.46 / 42.39`。详见 `docs/OPTIMIZATION_LOG_QUANT.md`
Attempt 25/26。

2026-07-03 后续把 W8/W4 tensor quantization/packing 从 Python per-element
loop 下沉到 C++ helper。它保持和 Python reference bitstream 一致，但大幅降低
转换耗时：Qwen3.5-0.8B W4G128 `3.06s`，Youtu-LLM-2B W4G128 `6.34s`，
Qwen3.5-4B W4G128 `20.18s`。4B 之前 Python 路径会卡在大 tensor / `lm_head`
量化阶段，现在可以实际生成 W4 package。

当前三个模型 W4G128 256-token PPL audit：

| Model | FP16 CE/PPL | W4G128 CE/PPL | CE delta |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 2.1388 / 8.49 | 2.4239 / 11.29 | +0.2851 |
| Youtu-LLM-2B | 3.1139 / 22.51 | 3.1578 / 23.52 | +0.0440 |
| Qwen3.5-4B | 1.8771 / 6.53 | 1.9568 / 7.08 | +0.0797 |

2026-07-03 又为 Youtu-LLM-2B 加入 `w4mixgN` policy：`lm_head`、
`self_attn.kv_b_proj`、`self_attn.o_proj`、`mlp.down_proj` 走 W8，其余
projection/merged `mlp.gate_up` 走 W4。`w4mixg128` 导出 `W4=128 W8=97`，
量化参数中 W8 约 44.2%、W4 约 55.8%；转换耗时 `9.27s`，包大小
`2011.2 MB`。256-token PPL 从纯 W4 的 `3.1578 / 23.52` 改到
`3.1511 / 23.36`，CE delta 从 `+0.0440` 降到 `+0.0372`。单次
pp256/tg64 smoke 为 `117.94 / 81.18`、RSS `4384.1 MB`；这不是 strict
5-run baseline，只用于确认 mixed 包正常运行并记录初始代价。

W8PC 多模型 package bench（pp256 + tg64, warmup=3, 4 threads, 5-run median；
2026-07-05 fan-on AUTO-i8mm build，均包含显式 `lm_head.weights`，默认 W8 为
q8dot full repack + i8mm 8-row GEMM）：

| Model | Runtime | pp/tg | peak RSS |
|---|---|---:|---:|
| Qwen3.5-0.8B W8PC | explicit W8 lmhead + q8dot full repack + i8mm 8-row GEMM | 635.51 / 211.96 | 2560.7 MB |
| Youtu-LLM-2B W8PC | explicit W8 lmhead + q8dot full repack + i8mm 8-row GEMM | 250.81 / 94.34 | 5947.6 MB |
| Qwen3.5-4B W8PC | explicit W8 lmhead + q8dot full repack + i8mm 8-row GEMM | 117.21 / 45.50 | 10964.0 MB |

Interpretation:
- q8dot full repack is now the default W8 path and keeps roughly the old default
  RSS by replacing, not adding to, the INT8 interleaved layout
- W8PC decode is faster than llama.cpp Q8_0 in this fan-on rerun; 0.8B remains
  noisy and is not the primary benchmark target
- AUTO i8mm plus 8-row GEMM narrows W8PC prefill substantially; 2B is now much
  closer to llama.cpp Q8_0, while 0.8B/4B still trail

2026-07-06 follow-up changed W8 q8dot GEMM multithread scheduling from pure
M-sharding to 2D atomic-steal over M tiles and N blocks. This does not change
the q8dot B layout: GEMV/decode and GEMM/prefill still share the same
`[N/8, K/32, 8, 32]` `q8_repack_data`, so RSS is unchanged. The old path is
available with `MOLLM_W8_GEMM_1D=1`; `MOLLM_W8_GEMM_N_BLOCK` overrides the
default W8 GEMM N block size of 1024 for A/B. Alternating 3-run same-session
A/B gave Qwen3.5-4B W8PC `121.75 / 44.96` vs old 1D `117.83 / 45.60`, and
Youtu-LLM-2B W8PC `252.70 / 93.57` vs old 1D `246.81 / 91.74`. Treat these as
directional 3-run A/B, not a replacement for the fan-on 5-run baseline table.

llama.cpp Q8_0 CPU 参考基线（`../llama.cpp` build `5c7c22c3e` / 9803，
`llama-bench -ngl 0 -p 256 -n 64 -t 4 -r 5 -o md`）：

| GGUF | Size | pp256 | tg64 |
|---|---:|---:|---:|
| Qwen3.5-0.8B Q8_0 | 784.52 MiB | 811.70 | 160.70 |
| Youtu-LLM-2B Q8_0 | 1.94 GiB | 274.46 | 84.61 |
| Qwen3.5-4B Q8_0 | 4.28 GiB | 139.66 | 39.71 |

当前判断：native W8 NEON 已修复 scalar bottleneck；llama.cpp-style Q8 activation
dot 对 decode 有明确收益，q8dot repack 在 2B/4B decode 上已基本追平或略超
llama.cpp Q8_0，但 prefill 仍约慢 2x。整包 load-time
dequant-to-FP16 曾作为 upper-bound 原型测试过，但已移除；它需要常驻 FP16 packed
weight copy，会放弃太多 W8 runtime-memory 收益，不是目标路径。

当前保留的实验路径是 `MOLLM_W8_ONFLY_FP16=1`：权重仍按 INT8 interleaved
保存，只在 kernel 加载 B tile 时现场 dequant 成 FP16 并走 FP16 accumulate 结构。
microbench 结果显示它没有改善 prefill GEMM，且 GEMV 明显变慢。下一步重点是
继续打磨默认 q8dot repack layout 的 kernel 效率：scale handling、tile layout、
runtime memory traffic，以及 per-group 量化路径。

2026-07-02 新增 `MOLLM_MATMUL_SHAPE_PROFILE=1`，`mollm_bench --profile`
会额外打印按 `phase/path/M/N/K/group` 聚合的 matmul shape profile。短 profiling
（pp256 + tg16, warmup=1）显示 2B/4B 的 W8 瓶颈集中在少数通用 shape class：

| Model | Dominant prefill q8dot GEMM | GMAC/s | Dominant decode q8dot GEMV | GMAC/s |
|---|---|---:|---|---:|
| Youtu-LLM-2B W8PC | M=256,N=12288,K=2048 | ~240 | M=1,N=12288,K=2048 | ~130 |
| Qwen3.5-4B W8PC | M=256,N=9216,K=2560 | ~225 | M=1,N=9216,K=2560 | ~127 |

这说明下一步不应该按模型/层硬编码具体矩阵尺寸，而应像 llama.cpp/ggml 一样按
quant type、GEMV/GEMM、tile（如 4x8/8x8）和 CPU feature 选择通用 kernel。
2026-07-02 已加入 ARM i8mm/`vmmlaq_s32` 的 q8dot GEMM kernel。2026-07-04
起 `MOLLM_ARM_I8MM` 默认为 `AUTO`：ARM64 configure 时会用
`-march=armv8.6-a+i8mm+fp16+fp16fml` 编译并（非交叉编译时）运行一个
`vmmlaq_s32` probe，确认 `__ARM_FEATURE_MATMUL_INT8` 可用后自动启用 i8mm
W8 GEMM。`-DMOLLM_ARM_I8MM=OFF` 可强制回到 DOTPROD/NEON，`ON` 会要求
probe 必须通过。

i8mm 单次 fresh run（pp256 + tg64, warmup=3, 4 threads；非 5-run median）：

| Model | Default DOTPROD W8PC | i8mm GEMM W8PC | effect |
|---|---:|---:|---:|
| Youtu-LLM-2B | 126.8 / 57.2 | 203.7 / 57.4 | pp +60.7%, tg flat |
| Qwen3.5-4B | 53.3 / 27.1 | 87.3 / 26.6 | pp +63.6%, tg flat |

随后 q8dot GEMV 改成 vector reduction，并区分 per-channel / per-block32 /
per-group scale mode。i8mm build 单次 fresh run：

| Model | Attempt 9 W8PC | llama.cpp Q8_0 reference |
|---|---:|---:|
| Youtu-LLM-2B | 230.4 / 87.1 | 257.3 / 85.8 |
| Qwen3.5-4B | 106.7 / 42.4 | 137.5 / 38.3 |

结论：decode gap 基本消失；剩余主要是 prefill，尤其 4B。下一步应把 GEMM
layout 进一步向 ggml `block_q8_0x4` 靠拢：A packing/layout、scale placement、
store/reorder 开销；GEMM activation Q8 quantization 已经向量化，当前约
`1.5-2%` matmul time。

### Dynamic shape（尝试 13）

- `DimExpr`（CONST/SEQ/MUL/ADD/BATCH）每维一个表达式，transpile 时 `propagate_dim_exprs()` 从 INPUT 传播
- runtime `eval_dim()` 求值，`inject_runtime_shapes()` 注入 INPUT shape + patch stateful op params
- DYNAMIC 模式（CPU）：无 padding，按实际 token 数执行
- STATIC_PADDED 模式（NPU future）：pad 到 build-time seq_len
- reshape() 用 `SEQ` 符号标记 dynamic 维，`num_heads * SEQ` 自动生成 `MUL(coeff=num_heads)`

### Backend 抽象

- `Backend` 抽象基类 + `CPUBackend`（ARM NEON）
- `.mollm` 严格后端中立，shape mode 由 backend 决定
- 未来 `NPUBackend` 由 runtime 选择

### .mollm 单文件格式
```
[Header 128B] magic MOLM + 6×(offset,len) pairs
[metadata JSON] weights offset map + model config
[tokenizer.json] [chat_template.jinja]
[prefill graph] [decode graph]
[weights region] all .weights files concatenated
```
运行时 mmap 整个文件，通过 metadata offset map 解析权重路径。

### 多 graph 架构
- prefill graph（seq_len=256, DYNAMIC 模式下 runtime 可变）+ decode graph（seq_len=1）
- 共享物理 KV cache（load-time cache migration）
- 支持 chunked prefill（past_len 可超过 graph_seq_len）

### 目录结构
```
mollm/
├── kernels/         NEON kernels（matmul, attention, gdn, norm, rope）
├── graph/           执行器、图格式、BufferPool、mmap
├── engine/          LLMEngine、tokenizer、generation loop、backend.h
├── models/          Python 转译器 + graph builder（converter.py, qwen35.py, mla.py, transpile.py）
├── examples/        mollm_chat (CLI), mollm_bench (benchmark)
├── tests/           unit + e2e + quantization smoke + shape_propagation.py
└── docs/            优化日志、架构文档
```

详见 `ARCHITECTURE.md`。

## 构建 & 运行

```bash
# 编译
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
ninja -C build

# 转换模型为 .mollm 单文件（自动检测模型类型）
python3 models/converter.py /path/to/Qwen3.5-4B model.mollm

# Chat（单文件，无需单独 tokenizer）
./build/mollm_chat --package model.mollm --prompt "Hello"

# Benchmark（DYNAMIC 模式，短 prompt 不 padding）
./build/mollm_bench --package model.mollm --prompt-tokens 64 --max-new-tokens 64 --warmup 3 --threads 4 --profile

# 测试
./build/test_e2e qwen35_0.8b.mollm [youtu-llm-2b.mollm]  # PPL + chunked prefill
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm
./build/test_gdn      # GDN kernel 正确性
./build/test_norm     # RMSNorm（含 strided）
./build/test_rope     # RoPE（含 strided）
python3 tests/test_shape_propagation.py  # DimExpr symbolic propagation
```

## 测试覆盖

| 测试 | 内容 |
|------|------|
| test_e2e | 端到端 PPL（Qwen3.5-0.8B，HF ref 8.50）+ Test 5 chunked prefill |
| test_quantized_e2e | quantized package load、短 CE drift、finite logits、短 decode |
| test_shape_propagation.py | DimExpr symbolic propagation（13 个用例，含 MUL 复合维） |
| test_gdn | fused GDN kernel（prefill/decode/state continuity） |
| test_norm | RMSNorm（含 strided 用例） |
| test_rope | RoPE halves/interleave（含 strided 用例） |
| test_matmul | matmul FP32/FP16 reference 对照 |
| test_attention | SDPA + KV cache append |
| test_execute | RESHAPE/PERMUTE/SLICE view 正确性 |
| test_graph | graph 构建 + OpParams + DimExpr |
| test_io | graph save/load round-trip（v3 格式） |
| test_tensor | Tensor create/strides/views |
| test_buffer_pool | acquire/release/reuse |
| test_mmap_file | MappedFile header/data |
| test_tokenizer | tokenizer load/encode/decode |
| test_engine | causal mask, RoPE cache, lifecycle |

### 当前测试状态

- **Qwen3.5-0.8B DYNAMIC 模式**：Test 1/2/3 全过（PPL 8.49/8.51 vs HF 8.50）
- **Test 5 chunked prefill**：
  - 128+128 chunks: PPL 8.4893 ✓
  - 64×4 chunks: PPL 8.4893 ✓
  - 256 单 chunk: PPL 8.4893 ✓
  - 100+100+56 / 56×4+32 / 57×4+28 / 40×6+16 / 33×7+25: PPL 全过（tolerance 0.1）
- **Youtu-LLM-2B MLA DYNAMIC 模式**：256 prefill PPL=10.26 vs HF 10.20 ✓（tolerance 2.0）
- **18/18 ctest 全过**；新增 W4G128 量化 smoke/PPL 通过（W4G128 256-token PPL=10.35 vs FP16 8.49）

### 已知 Bug

无（两个 DYNAMIC shape bug 已修复，见下）。

### 已知技术债：内存管理

当前内存管理是可工作的 BufferPool/reset 协议，但结构偏脆：

- `execute_graph()` 入口先清空上一轮 borrowed view metadata；默认释放 materialized temporaries。
  静态 decode graph 会保留 materialized workspace 跨 token 复用；dynamic prefill 在
  runtime shape key 完全相同时保留 materialized workspace
- zero-copy RESHAPE/PERMUTE/SLICE 借用输入指针；CONCAT/TILE/non-contiguous RESHAPE 会 materialize
- KV/GDN cache/state 已从 engine-owned persistent pool 分配，再由 prefill/decode graph INPUT tensor 共享引用
- `run_graph()` 已直接借用 `embed()` / mask / RoPE helper tensor 作为 graph INPUT，不再做 input staging copy
- `prepare_execution()` 已有实验性 view-aware last-use release queue；目前仅对无 stateful/cache-mutating op 的图启用，真实模型图仍使用 end-of-call cleanup
- `Tensor` 本身仍没有完整 owner/borrowed 类型语义，`RESHAPE` borrowed/materialized 仍依赖运行时指针判断
- `Tensor::owner_id` / `storage_id` 已记录 pooled tensor 来自哪个 `BufferPool` 以及具体 allocation；主要 release 路径会校验 owner mismatch，borrowed 判断优先用 storage id
- `BufferPool` 已记录 active allocation，析构/`clear()` 可释放 active + freelist；helper tensor 已在 output 消费或 copy 后释放
- `prefill()` / `prefill_hidden()` 在 output 消费或 copy 后清空 borrowed views，并保留
  bounded same-shape workspace；`release_prefill_buffers()` 显式释放。`decode()` /
  `decode_hidden()` 保留 bounded static workspace，`reset()` 会释放 decode workspace；
  hidden API 已返回 engine-owned copied output

这套协议已经修掉 chunked prefill 的 stale shape/data 和 pool 增长问题，但后续做
graph fusion、continuous batching、多 backend 前，建议先收敛成显式
`TensorStorage` / borrowed view / liveness release 模型。设计草案见
`docs/MEMORY_MODEL.md`。

### 已修复的 DYNAMIC shape bug

1. **chunked prefill past_len != 128 时出错**（Qwen3.5-0.8B DYNAMIC 模式）— **已修复**
   - 根因：`execute_graph` 的 `skip_init` 在 DYNAMIC 模式下错误跳过了 zero-copy view op
     (RESHAPE/PERMUTE/SLICE) 的 shape 更新，导致第二个 chunk 起读到上一个 chunk
     的 stale shape。均匀 chunk size 不暴露，最后一个不等长 chunk 才崩。
   - 修复：DYNAMIC 模式下不 skip_init；对 dynamic dim 总是 `eval_dim` 更新 shape，
     但对 zero-copy view 推迟 `compute_strides()` 给 dispatch 处理。

2. **Youtu MLA 256 prefill 崩**（DYNAMIC 模式）— **已修复**
   - 根因：zero-copy view op 的 `tensor.data` 在跨 `execute_graph()` 调用时没被清空
     （release pass 对 view op `continue` 跳过 `t.data = nullptr`）。第二次 prefill
     时这些 op 的 `out.data != nullptr`，跳过 buffer allocate，dispatch 写入已 release/
     复用的内存 → heap-buffer-overflow in RMSNorm。
   - 修复：`execute_graph` 入口清空所有非 INPUT/CONSTANT tensor 的 data 指针。

## 下一步

1. **W4 quality sweep** — 显式 lmhead 后，纯 W4G128 的 0.8B 256-token CE delta 为 +0.2851；Qwen3.5 `w4mixg128` 通过 73 个 W8 tensor（包含 `lm_head.weights`）恢复到 +0.0296。下一步比较 `w4mixg64` / `w4mixg32`、更细的 tensor ablation，或实现 Q4_K-like asymmetric block 格式
2. **W4 kernel next step** — W4 q4dot + package-level layout 已落地，runtime 不再需要常驻第二份 W4 repack copy；DOTPROD vector-reduce 和 GEMM/GEMV activation quantization 已显著恢复性能。下一步针对 2B/4B prefill shape 继续优化 q4dot kernel，或设计真正适合 i8mm 的 W4 micro-panel layout（避免 hot loop 临时展开）
3. **W8 kernel next step** — q8dot full repack 已是默认 W8 路径，decode 已基本追上 Q8_0 参考；W8 GEMM 已加入 2D N-block 调度且不改变 GEMV 复用的 B layout。剩余 prefill gap 仍主要需要更深的 packed GEMM layout/kernel 工作（A/B micro-panel、scale placement、store/reorder）和 per-group 路径
4. **W8 quality audit** — 256-token PPL 已通过；后续扩大到更多 calibration samples，确认 group-size 质量/大小 tradeoff
5. **内存模型整理** — 按 `docs/MEMORY_MODEL.md` 继续落地：下一步把 `(owner_id, storage_id)` 扩展成 explicit `TensorStorage` / storage registry
6. **Graph fusion** — 融合相邻 matmul+activation+norm，减少 cache 抢占（e2e matmul 利用率 40% vs microbench 86%）
7. **Continuous batching / Vision encoder / Qualcomm Oryon 支持**
