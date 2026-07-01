# Current State

*Last updated: 2026-07-01*

## 项目概述

**mollm** — mobile-oriented LLM 推理引擎，ARM NEON FP16FML kernels。
目前在 Apple Silicon 上开发和测试，移动端 ARM（Qualcomm Oryon、MediaTek）在 roadmap 中。
Python 转译前端 → `.mollm` 单文件打包 → C++ 执行器 + NEON FP16FML kernels。

支持三个模型系列：
- **Qwen3.5-4B**（hybrid linear/full attention：Gated DeltaNet + GQA）
- **Qwen3.5-0.8B**（同架构）
- **Youtu-LLM-2B**（MLA）

## 性能（Apple M5 Pro, 4 threads, pp256 + tg64, warmup=3, fan max）

| Model | mollm pp/tg | llama.cpp pp/tg | prefill gap | decode gap |
|-------|------------|-----------------|-------------|------------|
| Qwen3.5-4B | 115 / 25 | 143 / 23 | 1.25x | **0.92x (faster)** |
| Qwen3.5-0.8B | 601 / 104 | 749 / 100 | 1.25x | **0.96x (faster)** |
| Youtu-LLM-2B | 235 / 54 | 264 / 41 | 1.12x | **0.76x (faster)** |

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
├── tests/           19 个测试（unit + e2e + shape_propagation.py）
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
./build/test_gdn      # GDN kernel 正确性
./build/test_norm     # RMSNorm（含 strided）
./build/test_rope     # RoPE（含 strided）
python3 tests/test_shape_propagation.py  # DimExpr symbolic propagation
```

## 测试覆盖

| 测试 | 内容 |
|------|------|
| test_e2e | 端到端 PPL（Qwen3.5-0.8B，HF ref 8.50）+ Test 5 chunked prefill |
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
- **17/17 unit test 全过**（14 C++ + 1 Python + e2e）

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

1. **Weight quantization (W4)** — 4B prefill 唯一大幅杠杆。MATMUL 87% at 988 GF/s，W4 砍权重带宽 2x，预期 4B prefill 115 → ~160+（超 llama.cpp 143）
2. **内存模型整理** — 按 `docs/MEMORY_MODEL.md` 继续落地：下一步把 `(owner_id, storage_id)` 扩展成 explicit `TensorStorage` / storage registry
3. **Graph fusion** — 融合相邻 matmul+activation+norm，减少 cache 抢占（e2e matmul 利用率 40% vs microbench 86%）
4. **Continuous batching** — 多序列共享 prefill
5. **Vision encoder** — Qwen3.5 是多模态
6. **Qualcomm Oryon 支持** — 非 Apple ARM 验证
