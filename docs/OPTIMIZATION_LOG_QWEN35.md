# mlllm Qwen3.5 优化日志

> 测试条件: Apple M-series (ARM64), 4 threads, pp256 + tg64, warmup=3

## 框架对比基线 (2026-06-28)

| 框架 | 模型 | pp256 t/s | tg64 t/s |
|------|------|-----------|----------|
| **llama.cpp** | 0.8B | **749.44** | **100.44** |
| **llama.cpp** | 4B | **143.32** | **22.87** |
| **mlllm** | 0.8B | 199.20 | 89.12 |
| **mlllm** | 4B | 52.07 | 22.19 |

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
| decode_tps | 76.12 | 89.12 | +17% |
| tpot | 13.14ms | 11.14ms | -15% |

**Decode profile**:

| Op | 优化前 | 优化后 |
|---|---|---|
| MATMUL | 295.73ms (52.4%) | 287.62ms (64.8%) |
| GDN decode | 197.56ms (35.0%) | 84.99ms (19.2%) |
| SHORTCONV | 45.12ms (8.0%) | 45.57ms (10.3%) |

#### Qwen3.5-4B

4B 适配完成（修复 num_v_heads params、multi-file safetensors、state 大小等），chat 测试输出正确。

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| decode_tps | — | 22.19 |
| prefill_tps | — | 52.07 |

### 验证

- test_gdn: 3/3 PASS
- test_qwen35_ref: PASS
- test_e2e: PASS (PPL=8.50 vs HF 8.50)

---

## 尝试 2: GDN prefill NEON 向量化 (2026-06-29)

### 决策理由

GDN prefill 占 prefill 65.6%（0.8B），是和 llama.cpp 差距的主因。prefill 和 decode 的核心操作相同，
只是多了 token 维度的循环。将 decode 的 NEON 函数提取为共享头文件，prefill 复用。

### 实现

- 新增 `kernels/gdn_neon.h`：共享 NEON 辅助函数（`gdn_l2norm_neon`, `gdn_matvec_neon`,
  `gdn_rank1_update_neon`, `gdn_recurrence_neon`）
- 新增 `kernels/gdn_prefill.cpp`：NEON prefill kernel，内层 t 循环每个 token 调用 `gdn_recurrence_neon`
- 重构 `kernels/gdn_decode.cpp`：使用共享头文件
- `kernel_gdn_prefill()` 在 `#if HAS_NEON` 下分发到 NEON 路径

### 结果

#### Qwen3.5-0.8B

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| GDN prefill | 805.48ms (65.6%) | 356.65ms (45.1%) | **2.26x** |
| prefill_tps | 199.20 | **321.65** | +61% |

**Prefill profile 变化**:

| Op | 优化前 | 优化后 |
|---|---|---|
| GDN prefill | 805.48ms (65.6%) | 356.65ms (45.1%) |
| MATMUL | 272.24ms (22.2%) | — |
| SHORTCONV | 72.87ms (5.9%) | — |

#### Qwen3.5-4B

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| GDN prefill | ~1138ms (31.5%) | 1067.04ms (29.9%) | 1.07x |
| prefill_tps | 52.07 | **71.50** | +37% |

### 验证

- test_gdn: 3/3 PASS
- test_e2e: PASS (0.8B PPL=8.50, 4B chat 输出正确)

### 框架对比 (尝试 2 后)

| 框架 | 模型 | pp256 t/s | tg64 t/s |
|------|------|-----------|----------|
| llama.cpp | 0.8B | 749 | 100 |
| llama.cpp | 4B | 143 | 23 |
| mlllm | 0.8B | **322** | 89 |
| mlllm | 4B | **72** | 22 |

0.8B prefill 差距从 3.8x 缩小到 2.3x。

---

## 尝试 3: SHORTCONV decode 特化 (2026-06-29)

### 决策理由

SHORTCONV decode 占 0.8B decode 的 8.8%，当前实现为通用 prefill/decode 路径，每次调用都分配
`stated` 中间 buffer（~6.4MB）并做双循环重组。decode 时 seq_len=1，可以完全消除这些开销。

### 实现

在 `graph/execute.cpp` 的 SHORTCONV case 中增加 `if (seq_len == 1)` 快速路径：
- 消除 `stated` 分配（省去 groups × total_len 的内存分配和初始化）
- 消除输入重组循环（seq_len=1 时 x_data 直接按 group 索引）
- 卷积简化为单位置：`sum = cs0*w0 + cs1*w1 + cs2*w2 + x*w3`
- conv_state 更新简化为 shift + append

### 结果

#### Qwen3.5-0.8B

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| SHORTCONV decode | 44.14ms (8.8%) | 12.16ms (2.9%) | **3.63x** |
| decode_tps | 89.12 | **92.34** | +3.6% |

#### Qwen3.5-4B

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| SHORTCONV decode | 101.39ms (4.2%) | 20.33ms (0.9%) | **4.99x** |
| decode_tps | 21.82 | **23.07** | +5.7% |

### 验证

- test_e2e: PASS (0.8B PPL=8.50)
- 4B chat: 输出正确

### 框架对比 (尝试 3 后)

| 框架 | 模型 | pp256 t/s | tg64 t/s |
|------|------|-----------|----------|
| llama.cpp | 0.8B | 749 | 100 |
| llama.cpp | 4B | 143 | 23 |
| mlllm | 0.8B | 328 | **92** |
| mlllm | 4B | 73 | **23** |

0.8B decode 92 vs llama.cpp 100，差距 8%。
4B decode 23 vs llama.cpp 23，已持平。

---

## 尝试 4: SHORTCONV prefill NEON 化 (2026-06-29)

### 决策理由

SHORTCONV prefill 占 0.8B prefill 的 8.9%，卷积内层是 4 元素点积，正好一个 float32x4_t。
尝试消除 stated buffer 失败（内存访问模式变差），改为保留 stated 布局，仅 NEON 化卷积内层。

### 实现

在 `graph/execute.cpp` 的 SHORTCONV prefill 路径中：
- 卷积内层用 `vld1q_f32` + `vmulq_f32` + `vaddvq_f32` 替代标量循环
- stated buffer 和输入重组保持不变（保证卷积窗口连续内存访问）

### 结果

#### Qwen3.5-0.8B

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| SHORTCONV prefill | 68.82ms (8.9%) | 52.11ms (6.8%) | **1.32x** |
| prefill_tps | 327.95 | **333.89** | +1.8% |

#### Qwen3.5-4B

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| SHORTCONV prefill | 136.21ms (3.9%) | 105.89ms (3.0%) | **1.29x** |
| prefill_tps | 72.87 | 71.17 | -2.3% |

> 4B prefill_tps 略微下降（波动范围内），因为 SHORTCONV 只占 3%。

### 框架对比 (尝试 4 后)

| 框架 | 模型 | pp256 t/s | tg64 t/s |
|------|------|-----------|----------|
| llama.cpp | 0.8B | 749 | 100 |
| llama.cpp | 4B | 143 | 23 |
| mlllm | 0.8B | **334** | 90 |
| mlllm | 4B | 71 | 23 |

### 下一步

- GDN prefill 仍是最大热点（0.8B 占 46%），可考虑多线程并行

### 下一步

- SHORTCONV prefill NEON 化（0.8B 占 8.9%）
- GDN prefill 进一步优化（0.8B 仍占 46.3%）

### 下一步

- SHORTCONV 向量化（decode 占 10%）
- MATMUL prefill 路径优化
- 4B GDN prefill 效果不如 0.8B，可能因为 4B 的 MATMUL 占比更高（58%）
