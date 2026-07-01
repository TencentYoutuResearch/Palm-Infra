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
2. **Graph fusion** — 融合相邻 matmul+activation+norm，减少 cache 抢占（e2e matmul 利用率 40% vs microbench 86%）
3. **Continuous batching** — 多序列共享 prefill
4. **Vision encoder** — Qwen3.5 是多模态
5. **Qualcomm Oryon 支持** — 非 Apple ARM 验证
