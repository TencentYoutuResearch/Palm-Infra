# mlllm Qwen3.5 优化日志

> 测试条件: Apple M-series (ARM64), 4 threads, pp256 + tg64, warmup=3

## 框架对比基线 (2026-06-28)

| 框架 | 模型 | pp256 t/s | tg64 t/s |
|------|------|-----------|----------|
| **llama.cpp** | 0.8B | **749.44** | **100.44** |
| **llama.cpp** | 4B | **143.32** | **22.87** |
| **mlllm** | 0.8B | 207.50 | 76.12 |
| mlllm | 4B | — | — |

llama.cpp build: 5c7c22c3e, BLAS backend, gguf F16.

---

## 尝试 1: GDN decode NEON 向量化 (2026-06-28)

### 决策理由

GDN decode (seq_len=1) 占 decode 35%，纯 scalar 实现。核心操作全部可向量化：
L2 norm (128-dim reduce), state decay (16384 元素乘法), matvec (128×128 FMA),
rank-1 update (128×128 FMA), RMSNormGated (128-dim reduce)。k_dim=128, v_dim=128 均为 4 的倍数。

### 实现

新增 `kernels/gdn_decode.cpp`，NEON decode kernel（prefill 路径不变）：
- `l2norm_neon()` — L2 reduce + rsqrt + broadcast
- `matvec_128x128_neon()` — float32x4_t FMA 内层
- `rank1_update_128x128_neon()` — float32x4_t FMA 内层
- `gdn_decode_head_neon()` — 单 head 完整 recurrence + RMSNormGated
- `kernel_gdn_decode_neon()` — 16 head 循环

### 结果

#### Qwen3.5-0.8B

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| GDN decode | 197.56ms (35.0%) | 84.99ms (19.2%) | **2.32x** |
| decode_tps | 76.12 | **89.74** | +18% |
| tpot | 13.14ms | 11.14ms | -15% |

**Decode profile**:

| Op | 优化前 | 优化后 |
|---|---|---|
| MATMUL | 295.73ms (52.4%) | 287.62ms (64.8%) |
| GDN decode | 197.56ms (35.0%) | 84.99ms (19.2%) |
| SHORTCONV | 45.12ms (8.0%) | 45.57ms (10.3%) |

#### Qwen3.5-4B

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| — | — | — |

### 验证

- test_gdn: 3/3 PASS
- test_qwen35_ref: PASS
- test_e2e: PASS (PPL=8.50 vs HF 8.50)

### 下一步

- GDN prefill NEON (65.6% prefill, 与 llama.cpp 差距主因)
- SHORTCONV 向量化
- Qwen3.5-4B 适配 + 对比
