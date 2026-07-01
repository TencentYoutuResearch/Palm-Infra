# mollm Qwen3.5 优化日志

> 测试条件: Apple M-series (ARM64), 4 threads, pp256 + tg64, warmup=3
>
> **注意（2026-06-29 起）**：后续优化以 **4B 为主要对比基准**。原因：
> 0.8B 运行时间短（prefill ~500ms, decode ~600ms），对系统调度干扰敏感，
> 5 次连续测量 decode_tps 可从 99 暴跌到 62（-37%），数据不稳定。
> 4B 运行时间长（prefill ~2.5s, decode ~2.6s），同样 5 次测量波动仅 ±2%，
> 数据可信度高。0.8B 仍会跑 e2e 正确性验证，但性能对比以 4B 为准。
>
> 测量协议：风扇拉最大（避免热节流），5 次 run 取中位数。

## 框架对比基线 (2026-06-28)

| 框架 | 模型 | pp256 t/s | tg64 t/s |
|------|------|-----------|----------|
| **llama.cpp** | 0.8B | **749.44** | **100.44** |
| **llama.cpp** | 4B | **143.32** | **22.87** |
| **mollm** | 0.8B | 199.20 | 89.12 |
| **mollm** | 4B | 52.07 | 22.19 |

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
| mollm | 0.8B | **322** | 89 |
| mollm | 4B | **72** | 22 |

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
| mollm | 0.8B | 334 | 90 |
| mollm | 4B | 71 | 23 |

0.8B prefill 差距从 3.8x 缩小到 2.2x。

---

## 尝试 5: GDN prefill 多线程并行 (2026-06-29)

### 决策理由

GDN prefill 占 0.8B prefill 的 46.8%（358ms），heads 之间 state 独立（repeat=1 时），
天然适合多线程并行。每个 head 处理 256 tokens 的 recurrence，16 heads 可分给 4 线程。

### 实现

在 `kernels/gdn_prefill.cpp` 中使用 `thread_pool->parallel_for` 将 heads 分片并行处理。
- repeat=1 时直接按 heads 分片
- repeat>1 时按 key_head 分组（共享 state 的 heads 必须串行）

**踩坑**：`parallel_for` 的 fn 签名是 `(thread_id, begin, end)`，lambda 参数顺序必须匹配。
初始实现参数顺序错误（`vh_start, vh_end, tid`），导致 worker 线程处理了错误的 head 范围。
通过 standalone parallel_for 单元测试定位并修复。

### 结果

#### Qwen3.5-0.8B

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| GDN prefill | 358.43ms (46.8%) | 93.39ms (18.3%) | **3.84x** |
| prefill_tps | 331.82 | **495.72** | +49% |

**Prefill profile 变化**:

| Op | 优化前 | 优化后 |
|---|---|---|
| MATMUL | 278ms (34.6%) | — |
| GDN prefill | 358ms (46.8%) | 93ms (18.3%) |
| SHORTCONV | 52ms (6.8%) | — |

### 验证

- test_e2e: PASS (PPL=8.50)
- 4B chat: 输出正确

### 框架对比 (尝试 5 后)

| 框架 | 模型 | pp256 t/s | tg64 t/s |
|------|------|-----------|----------|
| llama.cpp | 0.8B | 749 | 100 |
| llama.cpp | 4B | 143 | 23 |
| mollm | 0.8B | **496** | 90 |
| mollm | 4B | 71 | 23 |

0.8B prefill 差距从 2.2x 缩小到 1.5x (749 vs 496)。
0.8B decode 差距 1.1x (100 vs 90)。

### 下一步

- MATMUL 现在是 0.8B prefill 最大热点（~50%+），但已高度优化
- 4B 的 MATMUL 占主导（prefill 59%, decode 82%），可考虑 weight quantization
- GDN prefill 仍有优化空间（单 head recurrence 内的 token 循环可向量化）

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
| mollm | 0.8B | **334** | 90 |
| mollm | 4B | 71 | 23 |

### 下一步

- GDN prefill 仍是最大热点（0.8B 占 46%），可考虑多线程并行

### 下一步

- SHORTCONV prefill NEON 化（0.8B 占 8.9%）
- GDN prefill 进一步优化（0.8B 仍占 46.3%）

### 下一步

- SHORTCONV 向量化（decode 占 10%）
- MATMUL prefill 路径优化
- 4B GDN prefill 效果不如 0.8B，可能因为 4B 的 MATMUL 占比更高（58%）

---

## 尝试 6: full_attn 层冗余 CONTIGUOUS 消除 (2026-06-29)

### 决策理由

重新转模型并复测 profile 后发现 CONTIGUOUS op 在 full_attn 层占据显著开销：
- 0.8B prefill: 43.79ms / 8.3%（60 calls）
- 4B prefill: 120.45ms / 4.2%（80 calls）
- 0.8B decode: 11ms / 2.4%
- 4B decode: 28ms / 1.2%

CONTIGUOUS 来自 `models/qwen35.py:_build_full_attn_layer` 里 q/k/v/attn 路径的 ~10 处 `g.contiguous()` 调用，每处都是 4 重循环逐元素 memcpy（`graph/execute.cpp:840`）。许多调用是为了满足下游 kernel 的 row-major 假设，但实际下游 kernel 已经支持 strided 输入。

### Kernel strided 支持情况验证

通过单元测试验证了下游 kernel 对 strided 输入的支持：

| Kernel | strided 支持 | 验证方式 |
|--------|-------------|----------|
| `kernel_rms_norm` | ✓ | `tests/test_norm.cpp` 新增 strided 用例（D=256 N=4 ldx=320），用 `stride[1]` 索引 |
| `kernel_rope` (halves) | ✓ | `tests/test_rope.cpp` 新增 strided 用例（D=256 rope_dim=64 N=4 ldx=304） |
| `kernel_sdpa` | ✗ | 把每个 head 当 `[seq*HD]` 连续块访问（`attention.cpp:1128` `vld1q_f32(Q_head + i)` 跨行），**要求 stride[1] == HD*es** |
| `Tensor::permute` / `reshape` | ✓ | 零拷贝，继承 stride |

**关键结论**：rms_norm 和 rope 的 NEON 实现已经用 `ldx = stride[1]/sizeof(float)` 参数化行距，**同一套代码同时处理 contiguous 和 strided**——contiguous 是 strided 的特例（ldx == D），不需要双路径。

### 实现

在 `models/qwen35.py:_build_full_attn_layer` 中删除 5 处冗余 `g.contiguous()`：

| 位置 | 删除的 contiguous | 下游 | 为何冗余 |
|------|------------------|------|---------|
| qg 路径 | `contiguous(qg)` | reshape + slice | matmul 输出 layout 已匹配 [HD*2, NH, seq] view，reshape/slice 零拷贝 |
| query 路径 | `contiguous(query)` (reshape 前) | reshape + permute | reshape 继承 stride，permute 零拷贝 |
| query 路径 | `contiguous(query)` (permute 后给 rms_norm) | rms_norm | kernel_rms_norm 用 ldx=stride[1] 支持 strided |
| key 路径 | 2 处（对称于 query） | rms_norm | 同上 |
| value 路径 | `contiguous(v)` (reshape 前) | reshape + permute | reshape/permute 零拷贝，SDPA 前的 contiguous 已保留 |

保留 4 处必要的 contiguous（SDPA 和 matmul 前各保留一处）：
- query/key/value → SDPA（SDPA 要求 stride[1] == HD*es）
- attn → o_proj matmul（matmul 对 strided 输入支持未验证，保守保留）

### 结果

#### Qwen3.5-0.8B

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| prefill_tps | 482.28 | **520.20** | +7.9% |
| decode_tps | 86.21 | 84.88 | 持平（噪声） |

**Prefill CONTIGUOUS 变化**:

| Op | 优化前 | 优化后 |
|---|--------|--------|
| CONTIGUOUS | 43.79ms (8.3%, 60 calls) | 13.44ms (2.8%, 24 calls) |

#### Qwen3.5-4B

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| prefill_tps | 87.91 | **94.39** | +7.4% |
| decode_tps | 22.49 | 22.45 | 持平 |

**Prefill CONTIGUOUS 变化**:

| Op | 优化前 | 优化后 |
|---|--------|--------|
| CONTIGUOUS | 120.45ms (4.2%, 80 calls) | 40.86ms (1.5%, 32 calls) |

### 验证

- test_norm: PASS（含新增 strided 用例）
- test_rope: PASS（含新增 strided 用例）
- test_e2e: PASS（PPL 8.4745/8.5021 vs HF ref 8.50）

### 框架对比 (尝试 6 后)

| 框架 | 模型 | pp256 t/s | tg64 t/s |
|------|------|-----------|----------|
| llama.cpp | 0.8B | 749 | 100 |
| llama.cpp | 4B | 143 | 23 |
| mollm | 0.8B | **520** | 85 |
| mollm | 4B | **94** | 22 |

0.8B prefill 差距从 1.55x 缩小到 1.44x。
4B prefill 差距从 1.63x 缩小到 1.52x。

### 下一步

- 0.8B MATMUL 仍占 prefill 58%，算子已优化到位（FP16FML + lane-FMA + GEMV 8-way），单点空间小
- 0.8B SHORTCONV prefill 仍占 11%，可考虑多线程化
- 4B MATMUL 占 prefill 79%，weight quantization 是唯一大幅杠杆
- GDN prefill 仍有 head×token 2D 调度空间

---

## 尝试 7: SHORTCONV prefill 多线程并行化 (2026-06-29)

### 决策理由

尝试 6 后 SHORTCONV prefill 在 0.8B 占 11.0%（53.55ms，18 calls），是仅次于 MATMUL(58%) 和 GDN(19%) 的第三大热点。当前实现（`graph/execute.cpp:297-402`）按 group 串行处理 3 个阶段，但 groups 之间完全独立（每组只访问 `stated[g*total_len .. (g+1)*total_len)` 片段），天然适合多线程并行。

groups 维度很大（0.8B: 6144, 4B: 8192），4 线程每线程可分到 ~1500-2000 组，并行效率高。

### 实现

在 `graph/execute.cpp` 的 SHORTCONV case（seq_len > 1 路径）中，将 3 个阶段的 `for (g)` 循环合并到一个 lambda，用 `thread_pool->parallel_for` 按 g 分片并行：

```cpp
auto process_group_range = [&](int /*tid*/, int g_start, int g_end) {
    for (int g = g_start; g < g_end; g++) {
        // 阶段 1: 填充 stated[g] = conv_state[g] || x[:, g]
        // 阶段 2: conv1d + silu (NEON 内层保留)
        // 阶段 3: 更新 conv_state
    }
};

if (thread_pool && groups >= 4) {
    int chunk = (groups + thread_pool->num_threads() - 1) / thread_pool->num_threads();
    thread_pool->parallel_for(0, groups, chunk, process_group_range);
} else {
    process_group_range(0, 0, groups);
}
```

**关键设计**：
1. **stated 分配保持在外层**（单次分配，线程共享不同片段）——避免每线程分配开销
2. **3 阶段合并到 lambda**——消除阶段间 barrier（每组的 3 阶段在同一线程内串行，组间并行）
3. **NEON 内层保留**——`vld1q_f32(st + i)` 在组内连续，无跨线程冲突
4. **chunk 策略**——参考 `gdn_prefill.cpp:100-103` 模式，chunk = ceil(groups/num_threads)

**未修改**：
- decode fast path（seq_len==1）不变
- scalar fallback 路径保留（仅在 HAS_NEON=0 时使用）
- SHORTCONV 的 graph 接口、params 不变

### 结果

#### Qwen3.5-0.8B

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| prefill_tps | 520.20 | **543.15** | +4.4% |
| decode_tps | 84.88 | 89.46 | +5.4%（SHORTCONV decode 占 4%，部分受益或噪声） |

**Prefill SHORTCONV 变化**:

| Op | 优化前 | 优化后 |
|---|--------|--------|
| SHORTCONV | 53.55ms (11.0%, 18 calls) | 21.03ms (4.5%, 18 calls) |

#### Qwen3.5-4B

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| prefill_tps | 94.39 | **89.98** (中位数; run1=96.91) | 波动较大 |
| decode_tps | 22.45 | 22.45 | 持平 |

**Prefill SHORTCONV 变化**:

| Op | 优化前 | 优化后 |
|---|--------|--------|
| SHORTCONV | 107.18ms (4.0%, 24 calls) | 62.81ms (2.2%, 24 calls) |

> 4B 三次 prefill_tps: 96.91 / 89.98 / 85.70，方差比 0.8B 大。4B MATMUL 占 81% 权重过大，
> 端到端提升被 MATMUL 的系统调度波动掩盖。SHORTCONV 优化本身的收益确定（107→63ms, -44ms, -41%）。

### 验证

- test_e2e: PASS（PPL 8.4745/8.5021 vs HF ref 8.50，与优化前完全一致）

### 框架对比 (尝试 7 后)

| 框架 | 模型 | pp256 t/s | tg64 t/s |
|------|------|-----------|----------|
| llama.cpp | 0.8B | 749 | 100 |
| llama.cpp | 4B | 143 | 23 |
| mollm | 0.8B | **543** | 89 |
| mollm | 4B | **90** | 22 |

0.8B prefill 差距从 1.44x 缩小到 **1.38x**（累计尝试 6+7：1.51x → 1.38x）。

### 下一步

- 0.8B MATMUL 仍占 prefill ~58%，算子已优化到位，单点空间小
- 0.8B GDN prefill 19%（94ms），可考虑 head×token 2D 调度（当前只按 head 并行，单 head 内 token 串行）
- 4B MATMUL 占 prefill 82%，weight quantization 是唯一大幅杠杆
- CONTIGUOUS 已降到 2.7%（0.8B）/ 1.5%（4B），不再是热点

---

## 尝试 8: GDN decode 多线程并行化 (2026-06-29)

### 决策理由

profiling 发现 GDN decode 是 decode 阶段的第二大热点：
- 0.8B decode: 99.76ms / 21.8%（1134 calls）
- 4B decode: 271.14ms / 11.7%（1512 calls）

排查发现 `kernel_gdn_decode_neon`（`kernels/gdn_decode.cpp`）**完全没有多线程**——
与 prefill 路径（尝试 5 已多线程）不对称。`kernel_gdn_decode` 虽然接收 `thread_pool` 参数，
但 NEON 路径的 `kernel_gdn_decode_neon` 签名没有 thread_pool，调用时也没传。

GDN decode 的每个 value head 完全独立（state 按 vh 切片，`state_data + vh * state_sz`），
天然适合按 vh 并行。0.8B num_v_heads=16，4B num_v_heads=32，4 线程每线程 4-8 heads。

### 实现

修改 2 个文件：

**`kernels/gdn_decode.cpp`**：
- 函数签名加 `ThreadPool* thread_pool` 参数
- 加 `#include "kernels/threading.h"`
- 将 `for (vh)` 循环包装到 `process_head` lambda
- 用 `thread_pool->parallel_for(0, num_v_heads, chunk, process_head)` 并行
- 复用 prefill 的 chunk 策略（`gdn_prefill.cpp:100-115`）：repeat=1 按 vh 分片，repeat>1 按 key_head 分组
- 保留 scalar fallback（`process_head(0, 0, num_v_heads)`）

**`kernels/gdn.cpp`**：
- 前向声明 `kernel_gdn_decode_neon` 加 `ThreadPool*` 参数
- `kernel_gdn_decode` 调用时传入 `thread_pool`

### 结果

#### Qwen3.5-0.8B

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| decode_tps | 89.46 | **95.20** | +6.4% |
| prefill_tps | 543.15 | 531.36 | 持平（噪声） |

**Decode GDN 变化**:

| Op | 优化前 | 优化后 |
|---|--------|--------|
| GATED_DELTANET_DECODE | 99.76ms (21.8%, 1134 calls) | 27.33ms (7.2%, 1134 calls) |

> 4 线程理想 4x，实际 99.76→27.33ms = 3.65x，并行效率 91%。
> run 1 decode_tps 达到 **101.37 t/s**，首次追平 llama.cpp 的 100 t/s。

#### Qwen3.5-4B

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| decode_tps | 22.45 | **24.50** | +9.1% |
| prefill_tps | 89.98 | 90.67 | 持平 |

**Decode GDN 变化**:

| Op | 优化前 | 优化后 |
|---|--------|--------|
| GATED_DELTANET_DECODE | 271.14ms (11.7%, 1512 calls) | 67.79ms (3.3%, 1512 calls) |

> 4B 并行效率 271.14→67.79ms = 4.0x，达到理想加速。4B decode_tps 24.5 **已超过 llama.cpp 的 23**。

### 验证

- test_e2e: PASS（PPL 8.4745/8.5021 vs HF ref 8.50，与优化前完全一致）

### 框架对比 (尝试 8 后)

| 框架 | 模型 | pp256 t/s | tg64 t/s |
|------|------|-----------|----------|
| llama.cpp | 0.8B | 749 | 100 |
| llama.cpp | 4B | 143 | 23 |
| mollm | 0.8B | 531 | **95** |
| mollm | 4B | 91 | **24.5** |

0.8B decode 差距从 1.12x 缩小到 **1.05x**（95 vs 100，基本持平）。
4B decode **已超过 llama.cpp**（24.5 vs 23）。

### 累计优化进度 (尝试 5 → 尝试 8)

| 模型 | 指标 | 尝试 5 | 尝试 8 | 累计提升 |
|------|------|--------|--------|---------|
| 0.8B | prefill_tps | 496 | 531 | +7% |
| 0.8B | decode_tps | 90 | 95 | +6% |
| 4B | prefill_tps | 71 | 91 | +28% |
| 4B | decode_tps | 23 | 24.5 | +7% |

### 下一步

- 0.8B prefill 差距 1.41x（531 vs 749），MATMUL 占 58% 已优化到位（964 GF/s 接近 FP16FML peak）
- 0.8B GDN prefill 仍占 19%（94ms），token 维度有 recurrence 状态依赖无法切片，但内层 matvec/rank1_update 可考虑融合（减少 state 重复读取）
- 4B prefill 差距 1.57x（91 vs 143），MATMUL 占 82%，weight quantization 是唯一大幅杠杆
- 0.8B/4B decode 已基本持平或超过 llama.cpp，decode 侧无明显空间

---

## 尝试 9: GDN recurrence kernel 融合（减少 state 重复读取）(2026-06-29)

### 决策理由

GDN prefill 在 4B 占 10%（270ms），是仅次于 MATMUL 的第二大热点。分析
`gdn_recurrence_neon`（`kernels/gdn_neon.h`）发现每个 (head, token) 对 state
（128×128=64KB）访问 5 遍：

| 步骤 | 操作 | state 访问 |
|------|------|-----------|
| 1. decay | state *= g | read + write |
| 2. matvec1 | kv_mem = state @ k | read |
| 3. delta | (v - kv_mem) * beta | — |
| 4. rank1_update | state += k ⊗ delta | read + write |
| 5. matvec2 | attn_out = state @ q | read |

5 遍 × 64KB = 320KB 内存流量/head/token，bandwidth-bound。

### 实现

在 `kernels/gdn_neon.h` 的 `gdn_recurrence_neon` 中，将 5 遍 state 访问融合为 2 遍：

- **Pass 1（融合步骤 1+2）**：按 dk 行遍历，读 state 行 → decay → 累加 kv_mem → 写回 decayed state
- **步骤 3**：算 delta（标量，不访问 state）
- **Pass 2（融合步骤 4+5）**：按 dk 行遍历，读 state 行 → rank1 update → 累加 attn_out → 写回 updated state

融合可行性：`gdn_matvec_neon` 和 `gdn_rank1_update_neon` 都按 dk 外层循环遍历 state 行，
可以合并。步骤 2 和 4 之间有数据依赖（delta 需要 kv_mem 完整结果），无法进一步融合。

state 内存流量从 320KB 降到 128KB/head/token（-60%）。

### 结果

#### Qwen3.5-4B（风扇拉满，5 次中位数，数据稳定）

| 指标 | 优化前（尝试 8） | 优化后（尝试 9） | 提升 |
|------|-----------------|-----------------|------|
| prefill_tps | 90.67 | **104.33** | +15.1% |
| decode_tps | 24.50 | **24.85** | +1.4% |

**Prefill GDN 变化**:

| Op | 优化前 | 优化后 |
|---|--------|--------|
| GATED_DELTANET_PREFILL | 270ms (10.0%, 24 calls) | 160ms (6.6%, 24 calls) |

**Decode GDN 变化**:

| Op | 优化前 | 优化后 |
|---|--------|--------|
| GATED_DELTANET_DECODE | 68ms (3.3%, 1512 calls) | 56ms (2.7%, 1512 calls) |

> 4B GDN prefill -41%（270→160ms），GDN decode -18%（68→56ms）。
> prefill 5 次结果方差极小（103-105 t/s），数据可信。

#### Qwen3.5-0.8B（参考，数据波动大）

| 指标 | 优化前 | 优化后（run 1 最佳） | 备注 |
|------|--------|---------------------|------|
| prefill_tps | 531 | 577 (run 1) | 5 次波动 400-577，不稳定 |
| decode_tps | 95 | 99 (run 1) | 5 次波动 62-99，不稳定 |

> 0.8B 运行时间短，对系统调度干扰敏感，数据不可信。后续以 4B 为准。

### 验证

- test_gdn: PASS（max_err < 1e-6，含 prefill→decode state continuity）
- test_e2e: PASS（PPL 8.4745/8.5021 vs HF ref 8.50，与优化前完全一致）

### 框架对比 (尝试 9 后)

| 框架 | 模型 | pp256 t/s | tg64 t/s |
|------|------|-----------|----------|
| llama.cpp | 4B | 143 | 23 |
| mollm | 4B | **104** | **24.9** |

4B prefill 差距从 1.57x 缩小到 **1.37x**（104 vs 143）。
4B decode **已超 llama.cpp**（24.9 vs 23，1.08x）。

### 累计优化进度 (尝试 5 → 尝试 9，以 4B 为准)

| 指标 | 尝试 5 | 尝试 9 | 累计提升 |
|------|--------|--------|---------|
| 4B prefill_tps | 71 | 104 | +46% |
| 4B decode_tps | 23 | 24.9 | +8% |

### 下一步

- 4B prefill 差距 1.37x（104 vs 143），MATMUL 占 84.6%
- matmul 算子 microbench 已达 988 GF/s（FP16FML peak ~1150，利用率 86%），算子无空间
- 端到端 matmul 仅 ~549 GF/s（利用率 48%），流失来自 cache/调度，需 graph fusion
- GDN prefill 已降到 6.6%，继续优化收益递减

---

## 尝试 10: RESHAPE/CONTIGUOUS materialize 行级 memcpy 优化 (2026-06-29)

### 决策理由

profiling 发现 RESHAPE 在 4B prefill 占 2.0%（48ms，104 calls），对"只改 metadata"的 op 来说异常高。
调查发现 RESHAPE 有两条路径：
- **contiguous 输入**：零拷贝，只改 shape/stride ✓
- **非 contiguous 输入**：逐元素 memcpy（4 重循环）✗

通过临时计数器定位到 4B 每 full_attn 层有 4 次 RESHAPE 触发 materialize（8 层 × 4 = 32 次），
每次拷贝 1-4MB。来源是 `_build_full_attn_layer` 里 `permute → reshape` 链：
尝试 6 删除冗余 contiguous 后，strided view 传到 reshape 触发 materialize。

CONTIGUOUS op 也有同样的 4 重循环逐元素 memcpy。

### 实现

在 `graph/execute.cpp` 新增 `materialize_strided()` helper，替换 RESHAPE 和 CONTIGUOUS 的
4 重循环：

```cpp
static inline void materialize_strided(const Tensor& src, void* dst) {
    // Fast path: dim0 contiguous (stride[0]==es) → memcpy each row in one call
    if (src.stride[0] == (size_t)es) {
        for (i3, i2, i1) {
            memcpy(dst + flat*es, src_ptr, row_bytes);  // 整行拷贝
            flat += d0;
        }
        return;
    }
    // Slow path: element-wise (dim0 not contiguous)
    ...
}
```

**关键优化**：原 4 重循环每个元素单独 memcpy（4 bytes），开销来自循环本身。
当 dim0 连续（stride[0]==es，所有 materialize 场景都满足），i0 循环用单次 memcpy
拷贝整行（shape[0] 元素），循环开销降低 shape[0] 倍（典型 256 倍）。

### 结果

#### Qwen3.5-4B（风扇拉满，5 次中位数，数据稳定 ±2.3%）

| 指标 | 优化前（尝试 9） | 优化后（尝试 10） | 提升 |
|------|-----------------|------------------|------|
| prefill_tps | 104.33 | **112.08** | +7.4% |
| decode_tps | 24.85 | **25.01** | +0.6% |

**RESHAPE/CONTIGUOUS 变化**:

| Op | 优化前 | 优化后 | 下降 |
|---|--------|--------|------|
| RESHAPE (prefill) | 48ms (2.0%, 104 calls) | 3.5ms (0.2%, 104 calls) | -93% |
| CONTIGUOUS (prefill) | 37ms (1.5%, 32 calls) | 1.8ms (0.1%, 32 calls) | -95% |
| 合计 | 85ms (3.5%) | ~5ms (0.3%) | **-94%** |

> 5 次 prefill_tps: 115.22 / 113.39 / 112.08 / 110.13 / 110.67（波动 ±2.3%，数据可信）

### 验证

- test_e2e: PASS（PPL 8.4745/8.5021 vs HF ref 8.50，与优化前完全一致）

### 框架对比 (尝试 10 后)

| 框架 | 模型 | pp256 t/s | tg64 t/s |
|------|------|-----------|----------|
| llama.cpp | 4B | 143 | 23 |
| mollm | 4B | **112** | **25** |

4B prefill 差距从 1.37x 缩小到 **1.28x**（112 vs 143）。
4B decode **已超 llama.cpp**（25 vs 23，1.09x）。

### 累计优化进度 (尝试 5 → 尝试 10，以 4B 为准)

| 指标 | 尝试 5 | 尝试 10 | 累计提升 |
|------|--------|---------|---------|
| 4B prefill_tps | 71 | 112 | **+58%** |
| 4B decode_tps | 23 | 25 | +9% |

### 下一步

- 4B prefill 差距 1.28x（112 vs 143），MATMUL 占 87%
- RESHAPE/CONTIGUOUS 已降到 0.3%，不再是热点
- matmul 算子 microbench 988 GF/s（FP16FML peak 86%），端到端 ~580 GF/s（利用率 50%）
- **weight quantization (W4)** 是唯一大幅杠杆：理论砍 matmul 时间 2x，4B prefill 112 → ~160+
- GDN prefill 7.1%，RMSNormGated 的 sigmoid 标量循环可 NEON 化（~5ms 小收益）

---

## 尝试 11: GDN RMSNormGated NEON sigmoid 向量化 (2026-06-29)

### 决策理由

GDN prefill 在 4B 占 7.1%（157ms）。分析 `gdn_recurrence_neon` 的步骤 6
（RMSNormGated）发现标量循环：每 4 元素做 `vst1q→scalar loop→vld1q` 往返，
其中 sigmoid 用 `std::exp`（标量）。项目已有 `sigmoid_f32_neon`（多项式近似，
7-bit 精度）在 activations.h 中，可直接复用。

### 实现

在 `kernels/gdn_neon.h` 中：
- 加 `#include "kernels/activations.h"`（复用 `sigmoid_f32_neon`）
- 替换步骤 6 的标量 sigmoid 循环为 NEON 向量化：
  ```cpp
  float32x4_t zv = vld1q_f32(z_row + dv);
  float32x4_t gate = vmulq_f32(zv, sigmoid_f32_neon(zv));
  vst1q_f32(out_head + dv, vmulq_f32(normed, gate));
  ```
- 消除 `vst1q(result) → scalar loop → vld1q(result)` 往返

### 精度影响

sigmoid_f32_neon 是多项式近似（~7-bit），引入 ~1e-3 误差。
- test_gdn 容差从 1e-5 放宽到 1e-3（注释说明原因）
- test_e2e PPL 8.4893/8.5070 vs HF 8.50，差异可接受

### 结果

#### Qwen3.5-4B（风扇拉满，5 次中位数）

| 指标 | 优化前（尝试 10） | 优化后（尝试 11） | 提升 |
|------|------------------|------------------|------|
| prefill_tps | 112.08 | **114.69** | +2.3% |
| decode_tps | 25.01 | **25.09** | +0.3% |

**GDN prefill 变化**: 157ms → 153ms（-3%）

> 5 次 prefill_tps: 117.71 / 116.20 / 114.69 / 113.64 / 112.83（波动 ±2.3%）

---

## 尝试 12: SHORTCONV prefill x 拷贝 seq-outer 优化 (2026-06-29)

### 决策理由

SHORTCONV prefill 在 4B 占 2.5%（53ms），计算量仅 16.7 MFLOP/call，10x overhead
来自内存访问。分析发现 Phase 1 的 x 拷贝 `x_data[s*groups + g]` 对固定 g 遍历 s 时
stride=groups*4 bytes（4B: 32KB），cache 利用率仅 1/32。

### 实现

在 `graph/execute.cpp` SHORTCONV case 中，将 Phase 1 的 x 拷贝从 per-group 循环
（s 内层，strided）改为 seq-outer 批量拷贝（g 内层，连续访问）：

```cpp
// Phase 1a: batch-copy x (seq-outer for cache-friendly access)
for (int s = 0; s < seq_len; s++) {
    const float* x_row = x_data + s * groups;      // 连续
    float* st_row = st_base + prefix_len + s;
    for (int g = 0; g < groups; g++)
        st_row[g * total_len] = x_row[g];           // x_row 连续访问
}
```

尝试过滑动窗口（无 stated buffer）方案，但因标量 shift 开销和 x strided 访问恶化，
比原版慢 8%，已回退。

### 结果

SHORTCONV 从 53ms 降到 48-58ms（波动范围内，收益不确定）。理论上 seq-outer 访问
更 cache-friendly，但 4B 的 SHORTCONV 占比小（2.5%），收益被波动掩盖。保留此优化
（逻辑正确，理论更优）。

### 验证

- test_e2e: PASS（PPL 8.4893/8.5070 vs HF 8.50）

### 框架对比 (尝试 11+12 后)

| 框架 | 模型 | pp256 t/s | tg64 t/s |
|------|------|-----------|----------|
| llama.cpp | 4B | 143 | 23 |
| mollm | 4B | **115** | **25.1** |

4B prefill 差距从 1.28x 缩小到 **1.25x**（115 vs 143）。

### 累计优化进度 (尝试 5 → 尝试 12，以 4B 为准)

| 指标 | 尝试 5 | 尝试 12 | 累计提升 |
|------|--------|---------|---------|
| 4B prefill_tps | 71 | 115 | **+62%** |
| 4B decode_tps | 23 | 25.1 | +9% |

### 下一步

- 4B prefill 差距 1.25x（115 vs 143），MATMUL 占 87%
- 非 MATMUL 优化空间已基本耗尽：GDN（融合+多线程+NEON sigmoid）、SHORTCONV（多线程+seq-outer）、
  RESHAPE/CONTIGUOUS（行级 memcpy）、SILU/RMS_NORM（已 NEON 化）都已优化到位
- matmul 算子 microbench 988 GF/s（FP16FML peak 86%），端到端 ~393 GF/s（利用率 40%）
- 端到端 matmul 2.5x overhead 来自 cache/DRAM 带宽（4B 权重 280MB >> L2 16MB，每次 matmul 走 DRAM）
- **weight quantization (W4)** 是唯一大幅杠杆：理论砍 matmul 时间 2x，4B prefill 115 → ~160+

## 尝试 13: Dynamic shape schema + Backend 抽象 + chunked prefill 测试

### 目标

为未来"runtime 按实际 token 数执行（无 padding）"和"多后端（CPU/NPU）支持"打基础。

### 改动

1. **graph format v3**（不向前兼容 v2）：`GraphNode` 加 `dynamic[4]` 字段（每维一个 `DynamicKind` enum：STATIC/SEQ/BATCH），序列化紧跟 `out_shape[4]` 后
2. **Python transpile symbolic shape propagation**：`propagate_dynamic_shapes()` pass 从 INPUT 节点按 op 规则传播 SEQ dim 到下游（ONNX 风格）。覆盖所有 op（MATMUL/RMS_NORM/SILU/SDPA/RESHAPE/PERMUTE/SLICE/CONCAT/GDN_PREFILL 等）
3. **Backend 抽象**：新增 `engine/backend.h`，`Backend` 抽象基类 + `CPUBackend`。`ExecContext` 持有 `Backend*`，`execute_graph` 通过 `backend->dispatch()` 调用 kernel。`.mollm` 严格后端中立
4. **EngineConfig 简化**：删 `prefill_graph_path` / `decode_graph_path`，只接受 `package_path`（CLI 删 `--tokenizer`/`--artifacts` 选项）
5. **inject_runtime_shapes()**：runtime 注入实际 seq_len 到 INPUT tensor + patch stateful op 的 n_real_tokens params。当前 STATIC_PADDED 模式启用，DYNAMIC 模式 reserved（待 reshape SEQ 标记完整后启用）
6. **Test 5 chunked PPL**：256 token 拆 (128,128) 和 (100,100,56) 验证跨 chunk KV/state 衔接

### 验证

- test_e2e: 全 PASS（含新 Test 5）
  - Qwen3.5-0.8B PPL: prefill 8.4893 / decode 8.5070 / chunked 8.4893（HF ref 8.50）
  - Youtu-LLM-2B PPL: 10.2569（HF ref 10.20）
- 13/13 unit test 通过（含新增 test_shape_propagation.py 13 个用例）

### 待办

- DYNAMIC 模式启用：解决 reshape 中 head_dim=256 与 seq_len=256 数值碰撞，需要 reshape() 调用者显式标记 SEQ 维或改用 -1
- NPU backend 实现（仅占位）
- batch dim dynamic（reserved enum）

### 累计优化进度

| 指标 | 尝试 5 | 尝试 12 | 尝试 13 |
|------|--------|---------|---------|
| 4B prefill_tps | 71 | 115 | 115（不变，铺垫 W4） |
| 4B decode_tps | 23 | 25.1 | 25.1 |

本次改动是架构性投资，不直接提速。W4 量化仍是下一步主要 lever。

---

## 尝试 14: W8 weight-only quantization correctness baseline (2026-07-01)

### 目标

先建立正确、可验证、可扩展到 per-group/W4 的量化基础设施；暂不追求推理速度。

### 改动

1. **W8 metadata format**
   - 复用 `.weights` header 中的 `scales_offset/scales_size/group_size/num_groups`
   - `w8pc` 表示为 `group_size=K`
   - `w8gN` 表示为 K 维每 N 个元素一组，如 `w8g128`
2. **Converter**
   - `models/converter.py` 新增可选 `quant` 参数
   - Qwen3.5 / Youtu converter 支持 `none|w8pc|w8gN`
   - 仅量化 projection/linear 2-D weights；embedding/lm_head storage、norm、conv、
     `A_log`、`dt_bias` 保持原精度
3. **Runtime loader**
   - CONSTANT tensor precision 从 weight header 读取，而不是依赖 graph node 原始 precision
   - INT8 weight load 时校验 scales/group metadata
   - INT8 跳过 FP16 interleaved packing
4. **Kernel**
   - `kernel_matmul_fp32()` 新增 INT8 scalar path
   - scale indexing 为 `scales[n * groups_per_row + k / group_size]`
   - 输出仍为 FP32，activation 在 matmul 后应用
5. **Tests**
   - `test_matmul` 新增 INT8 per-channel/per-group reference 对照
   - 新增 `test_quantized_e2e`：加载 FP16/W8 package，检查 finite logits、短 CE drift、
     workspace presence、短 decode token 有效性

### 验证

```bash
ninja -C build
python3 -m py_compile models/converter.py models/transpile.py models/qwen35.py models/mla.py
./build/test_matmul
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8pc.mollm
./build/test_quantized_e2e qwen35_0.8b.mollm qwen35_0.8b_w8g128.mollm
ctest --test-dir build --output-on-failure -E 'test_e2e|bench_matmul|bench_sdpa'
```

Qwen3.5-0.8B quick validation:

| Package | Size | Short CE/PPL | CE delta vs FP16 | Greedy smoke |
|---|---:|---:|---:|---|
| FP16 | 1518.9 MB | 3.0911 / 22.00 | - | - |
| W8PC | 1022.9 MB | 3.0969 / 22.13 | +0.0058 | pass |
| W8G128 | 1036.8 MB | 3.0926 / 22.03 | +0.0015 | pass |

### 结果解释

这是 correctness baseline，不是性能结果。当前 INT8 matmul 是标量 dequant+dot，
因此 W8 推理速度预计会低于 FP16。下一步应先实现 W8 NEON GEMV/GEMM，再按严格
pp256/tg64 协议测 2B/4B；完整 256-token PPL 也应在 optimized kernel 之后纳入
quality audit。
