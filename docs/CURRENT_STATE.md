# Current State

*Last updated: 2026-06-29*

## 项目概述

**mollm** — ARM-first LLM 推理引擎，针对 Apple Silicon 优化。
Python 转译前端 → `.mollm` 单文件打包 → C++ 执行器 + NEON FP16FML kernels。

支持三个模型系列：
- **Qwen3.5-4B**（hybrid linear/full attention：Gated DeltaNet + GQA）
- **Qwen3.5-0.8B**（同架构）
- **Youtu-LLM-2B**（MLA）

## 性能（Apple M5, 4 threads, pp256 + tg64, warmup=3, fan max）

| Model | mollm pp/tg | llama.cpp pp/tg | prefill gap | decode gap |
|-------|------------|-----------------|-------------|------------|
| Qwen3.5-4B | 115 / 25 | 143 / 23 | 1.25x | **0.92x (faster)** |
| Qwen3.5-0.8B | ~550 / ~95 | 749 / 100 | 1.36x | 1.05x |
| Youtu-LLM-2B | 235 / 54 | 264 / 41 | 1.12x | **0.76x (faster)** |

**4B 是主要基准** — 0.8B 测量不稳定（运行时间短，对调度敏感，decode_tps 5 次跑 62-99 波动）。4B 稳定（±2.3%）。

## 优化进度（Qwen3.5-4B，尝试 5→12）

| 指标 | 尝试 5 | 尝试 12 | 累计 |
|------|--------|---------|------|
| prefill_tps | 71 | 115 | **+62%** |
| decode_tps | 23 | 25 | +9% |

4B decode 已超 llama.cpp。prefill gap 1.25x，MATMUL-bound（87%）。

### 关键优化（详见 OPTIMIZATION_LOG_QWEN35.md）

1. **GDN recurrence kernel 融合**：5-pass state 访问 → 2-pass（-60% 内存流量）
2. **GDN decode 多线程**：21.8% → 7.2% of decode time
3. **full_attn 冗余 CONTIGUOUS 消除**：验证 rms_norm/rope 支持 strided，删除 5 处冗余
4. **RESHAPE/CONTIGUOUS materialize**：行级 memcpy fast path（-94%）
5. **FP16FML matmul**：988 GF/s microbench（86% FP16FML peak），lane-FMA + 2-way K-unroll GEMM，8-way K-unroll GEMV
6. **SHORTCONV prefill 多线程 + seq-outer 拷贝**
7. **GDN RMSNormGated NEON sigmoid 向量化**

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
1. **Python 转译**（`models/qwen35.py`, `models/mla.py`）：PyTorch weights → `.mollm` 单文件
2. **C++ 执行器**（`graph/execute.cpp`）：顺序节点 dispatch + BufferPool 内存管理
3. **NEON kernels**（`kernels/`）：matmul, attention, gdn, norm, rope

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
- prefill graph（seq_len=256）+ decode graph（seq_len=1）
- 共享物理 KV cache（load-time cache migration）
- 支持 chunked prefill（past_len 可超过 graph_seq_len）

### 目录结构
```
mollm/
├── kernels/         NEON kernels（matmul, attention, gdn, norm, rope）
├── graph/           执行器、图格式、BufferPool、mmap
├── engine/          LLMEngine、tokenizer、generation loop
├── models/          Python 转译器（qwen35.py, mla.py）
├── python/          GraphBuilder + serializer（transpile.py）
├── examples/        mollm_chat (CLI), mollm_bench (benchmark)
├── tests/           18 个测试（unit + e2e）
└── docs/            优化日志、架构文档
```

详见 `ARCHITECTURE.md`。

## 构建 & 运行

```bash
# 编译
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
ninja -C build

# 转换模型为 .mollm 单文件
PYTHONPATH=. python3 models/qwen35.py /path/to/Qwen3.5-4B model.mollm 32 256

# Chat（单文件，无需单独 tokenizer）
./build/mollm_chat --package model.mollm --prompt "Hello"

# Benchmark
./build/mollm_bench --package model.mollm --prompt-tokens 256 --max-new-tokens 64 --warmup 3 --threads 4 --profile

# 测试
./build/test_e2e     # PPL 验证（8.49 vs HF 8.50）
./build/test_gdn      # GDN kernel 正确性
./build/test_norm     # RMSNorm（含 strided）
./build/test_rope     # RoPE（含 strided）
```

## 测试覆盖

| 测试 | 内容 |
|------|------|
| test_e2e | 端到端 PPL（Qwen3.5-0.8B，HF ref 8.50） |
| test_gdn | fused GDN kernel（prefill/decode/state continuity） |
| test_norm | RMSNorm（含 strided 用例） |
| test_rope | RoPE halves/interleave（含 strided 用例） |
| test_matmul | matmul FP32/FP16 reference 对照 |
| test_attention | SDPA + KV cache append |
| test_execute | RESHAPE/PERMUTE/SLICE view 正确性 |
| test_graph | graph 构建 + OpParams |
| test_io | graph save/load round-trip |
| test_tensor | Tensor create/strides/views |
| test_buffer_pool | acquire/release/reuse |
| test_mmap_file | MappedFile header/data |
| test_tokenizer | tokenizer load/encode/decode |
| test_engine | causal mask, RoPE cache, lifecycle |

## 下一步

1. **Weight quantization (W4)** — 4B prefill 唯一大幅杠杆。MATMUL 87% at 988 GF/s，W4 砍权重带宽 2x，预期 4B prefill 115 → ~160+（超 llama.cpp 143）
2. **Graph fusion** — 融合相邻 matmul+activation+norm，减少 cache 抢占（e2e matmul 利用率 40% vs microbench 86%）
3. **Continuous batching** — 多序列共享 prefill
4. **Vision encoder** — Qwen3.5 是多模态
5. **Qualcomm Oryon 支持** — 非 Apple ARM 验证
