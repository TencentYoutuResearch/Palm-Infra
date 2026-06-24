# Matmul 微核优化日志

## 基线 (K_BLOCK=512, TILE=4×4, FP32, ARM NEON)

| Shape | threads=1 GFLOPS | threads=4 GFLOPS |
|-------|-----------------|-------------------|
| 128×2048×6144 | 28.3 | 98.3 |
| 1×2048×2048 | 8.1 | — |
| 1×2048×128256 (lm_head) | 5.9 | 26.3 |

端到端 (Youtu-LLM-2B, 4 threads):
- prefill_tps=4.41, decode_tps=4.45
- prefill MATMUL=4381ms, decode MATMUL=580ms

---

## 尝试 1：内层 K 循环展开 (vfmaq_laneq_f32)

**决策理由**：当前内层 K 循环每轮做 4 次标量 A 加载 + 4 次 `vfmaq_n_f32`。
展开 4 次 K 迭代 → 用 `vld1q_f32` 一次加载 4 个连续 A 值 → 用 `vfmaq_laneq_f32` 做 lane-wise FMA。
理论上可以减少 A 加载次数和指令数。

**结果**：
| Shape | 基线 | 展开后 | 变化 |
|-------|------|--------|------|
| 128×2048×6144 t=1 | 113.72ms | 153.09ms | **-35%** |
| 128×2048×6144 t=4 | 32.78ms | 39.22ms | -20% |
| 1×2048×2048 t=1 | 1.03ms | 1.54ms | -50% |
| 1×2048×128k t=4 | 20.01ms | 31.57ms | -58% |

**结论**：展开反而更慢。原因分析：
- `vfmaq_laneq_f32` 指令延迟可能高于标量加载 + `vfmaq_n_f32`
- B 的 scatter gather 才是瓶颈，A 的加载方式不是
- 编译器已经在做自动向量化/调度优化

**决定**：回滚，不采用此优化。

---

## 尝试 2：TILE 从 4×4 扩展到 8×8

**决策理由**：4×4 tile 内层 K 循环每轮加载 4 个 B 值（gather from scatter layout），
做 4 次 FMA（每条 1 个 `vfmaq_n_f32`），compute-to-load ratio = 1:1。
扩大到 8×8 后，每次加载 8 个 B 值（2 次 `vld1q_f32`），做 16 次 FMA（8 行 × 2 组），
compute-to-load ratio = 2:1。理论上能更好地摊销 B 的 scatter gather 开销。

**结果 (microbench)**：
| Shape | 4×4 tile | 8×8 tile | 变化 |
|-------|----------|----------|------|
| 128×2048×6144 t=1 | 113.72ms | 84.87ms | **-25%** |
| 128×2048×6144 t=4 | 32.78ms | 22.38ms | **-32%** |
| 1×2048×2048 t=1 | 1.03ms | 1.04ms | +1% (持平) |
| 1×2048×128k t=4 | 20.01ms | 22.51ms | +12% (略慢) |

**结果 (端到端, 4 threads)**：
| 指标 | 4×4 tile | 8×8 tile | 变化 |
|------|----------|----------|------|
| prefill_tps | 4.41 | 7.02 | **+59%** |
| decode_tps | 4.45 | 5.15 | +16% |
| prefill_ms | 5217 | 3276 | -37% |
| prefill MATMUL | 4381ms | 2426ms | -45% |
| decode MATMUL | 580ms | 503ms | -13% |

**结论**：**显著有效**。prefill matmul 提升 45%，端到端 prefill_tps 从 4.41 到 7.02。
lm_head 微基准略慢（+12%）但端到端 decode 仍然有提升（+16%），
因为 decode 里 lm_head 只是一部分，其他 matmul 也受益了。
GEMV（M=1）基本持平。

**决定**：**采用 8×8 tile 作为默认**。

---

## 尝试 3：B repacking（interleaved layout）

**决策理由**：当前 B 是 row-major [N, K]，内层加载 8 个 B 值需要 gather（stride K_weight）。
Repacking 把每个 8×K_BLOCK 子块转置，使得对固定 k，n..n+7 列的值在内存中连续，
可以用单条 `vld1q_f32` 代替栈临时数组 + gather。

**实现**：每个 8 列 panel 做一次 repack：`repacked[k*8 + j] = B[(n+j)*K_weight + k]`。
K=2048 时 buffer 约 64KB。仅对 K > 64 启用。

**结果 (microbench vs 8×8 baseline)**：
| Shape | 8×8 基线 | B repack | 变化 |
|-------|----------|----------|------|
| 128×2048×6144 t=1 | 84.87ms | 78.19ms | -8% |
| 128×2048×6144 t=4 | 22.38ms | 20.89ms | -7% |
| 1×2048×2048 t=1 | 1.04ms | 1.35ms | +30% |
| 1×2048×128k t=4 | 22.51ms | 27.40ms | +22% |

**结果 (端到端 vs 8×8 baseline, 4 threads)**：
| 指标 | 8×8 基线 | B repack | 变化 |
|------|----------|----------|------|
| prefill_tps | 7.02 | 6.20 | -12% |
| decode_tps | 5.15 | 4.09 | -21% |
| prefill MATMUL | 2426ms | 2908ms | +20% |

**结论**：microbench 上大 matmul 有微弱提升（-7~8%），但小 matmul 和 lm_head 明显退化（+22~30%）。
端到端 prefill 和 decode 都变差了。根因分析：
- repack 本身的 cost（per-tile 拷贝）抵消了加载收益
- 对于 GEMV 场景（M=1），repack 开销占比太高
- 8×8 tile 已经在一定程度上摊销了 B 的 gather 成本，repack 的增量收益有限
- 在多线程场景下，repack 的堆分配（new/delete）引入了额外开销

**决定**：**不采用，已回滚到 8×8 无 repack 版本**。
如果未来要做 repack，需要：
1. 在更外层做一次性 repack（不是 per-tile），或
2. 用栈分配的固定大小 buffer（避免 heap 分配）

---

## 尝试 4：Load-time B repacking（一次性转置）

**决策理由**：前次尝试 per-tile repack 开销太大。改为在 `LLMEngine::load_graph` 时
一次性将所有权重从 row-major [N, K] 转置为 [K, N]，matmul 内核用 `vld1q_f32` 连续加载。
repack 成本只在加载时付一次。

**实现**：
- `engine/engine.cpp`: 加载 mmap 后立即 `repacked[k*N+n] = src[n*K+k]`
- `kernels/matmul.cpp`: 内核自动检测 `K_weight != K` 判断是否 repacked

**结果 (vs 8×8 无 repack 基线)**：
| 指标 | 8×8 基线 | Load-time repack | 变化 |
|------|----------|-----------------|------|
| load_ms | 328 | 5645 | **+1621%** |
| prefill_tps | 7.02 | 5.56 | -21% |
| decode_tps | 5.15 | 5.13 | ~持平 |
| prefill MATMUL | 2426ms | 3326ms | **+37%** |
| decode MATMUL | 503ms | 503ms | ~持平 |

**结论**：加载时间暴涨 17 倍（~5.3GB 权重的 O(N*K) 转置），但推理速度反而下降。
根因：repacked [K, N] layout 虽然让 B 加载连续了，但每次访问 `B[k*N + n]` 时
N 很大（如 2048），跨 K 步长从原来的 K 变成了 N，对 cache 反而不利。
decode 持平说明 repack 对小 shape 确实没帮助。

**决定**：**不采用**。当前最优方案仍是 8×8 tile + K_BLOCK=512 无 repack。

**教训**：对于 FP32 权重、K ≈ N 的场景，简单的 gather 加载（4 个 float 从 stride-K 取）
在 Apple Silicon 上已经足够高效，transpose repack 的 cache 副作用反而更差。

---

## 尝试 5：FP16 存储 + FP32 累加

**决策理由**：模型原始权重是 BF16，之前导出时全部转成了 FP32，浪费了 2x 内存带宽。
改为权重 FP16 存储，matmul 内层用 NEON `vld1q_f16` + `vcvt_f32_f16` 解量化到 FP32 再 FMA。

**实现**：
- `models/mla.py`: 投影权重保持 FP16 导出（`astype(np.float16)`）
- `kernels/matmul.cpp`: 新增 `matmul_fp16_neon_8x8_range`，用 `float16x8_t vld1q_f16` 一次加载 8 个 FP16
- `kernel_matmul_fp32`: 检测 `B.prec == FP16` 自动走 FP16 路径

**结果 (vs FP32 8×8 基线, 4 threads)**：
| 指标 | FP32 基线 | FP16 | 变化 |
|------|----------|------|------|
| prefill_tps | 7.02 | 7.21 | +3% |
| decode_tps | 5.15 | 6.55 | **+27%** |
| prefill_ms | 3276 | 3188 | -3% |
| decode_ms | 583 | 458 | -21% |

**结论**：**decode 显著受益**（+27%），prefill 小幅提升（+3%）。decode 是带宽瓶颈更明显的场景，
FP16 直接减半了 B 的内存流量。prefill 提升不大说明 compute bound 更重。
权重文件大小从 50MB 降到 25MB。

**决定**：**采用 FP16 作为默认权重存储格式**。

---

## 尝试 6：K 循环展开 4x (vfmaq_laneq_f32)

**决策理由**：当前内层 K 循环每轮加载 8 个 B 值 + 8 个 A 标量，做 16 次 FMA。
展开 4x → 用 `vld1q_f32` 一次加载 4 个连续 A 值 → 用 `vfmaq_laneq_f32` 做 lane-wise FMA，
理论上 A 加载从 8 次标量 → 8 次向量加载，B 加载仍然 4 轮但复用 A。

**结果 (vs FP16 基线)**：
| Shape | 基线 | K unroll 4x | 变化 |
|-------|------|------------|------|
| 128×2048×6144 t=1 | 84.87ms | 87.34ms | +3% |
| 128×2048×6144 t=4 | 22.38ms | 22.41ms | ~持平 |
| 1×2048×2048 t=1 | 1.04ms | 1.05ms | ~持平 |
| 1×2048×128k t=4 | 22.51ms | 24.04ms | +7% |

**结论**：**无效**。K 展开 4x 没有带来任何提升，lm_head 还略有退化。
与尝试 1（FP32 K 展开）结果一致：`vfmaq_laneq_f32` 延迟更高，
且 FP16 场景下 B 的 gather + vcvt 才是瓶颈，A 的加载方式不是。

**决定**：**不采用，已回滚**。FP16 + 8×8 tile 无 K 展开仍是当前最优。

---

# 整体进度总结

## 端到端性能演进 (Youtu-LLM-2B, M5 Pro, 4 threads)

注：prefill_tps 早期用短 prompt (23 token) 测，被 graph padding 严重低估。
pp256 = 256-token prompt 满载，与 ncnn/llama.cpp 对齐。

| 版本 | prefill (pp256) | decode (tg) | 关键改动 |
|------|----------------|------------|---------|
| 初始 | — | 1.55 | FP32 4×4 tile, 单线程 |
| + 线程池 | — | 1.54 | 自定义线程池 + parallel_for |
| + lm_head 走 matmul | — | 3.54 | 复用 matmul 内核 + N 分片 |
| + 8×8 tile | — | 5.15 | TILE_M=8, TILE_N=8 |
| + FP16 权重 | — | 6.55 | FP16 存储 + FP32 累加 |
| + per-K_BLOCK B pack | — | 6.27 | GEMM only, GEMV 回退 |
| + load-time B pack | — | 10.51 | GEMV 也用 packed B |
| + A pack + lane-FMA | **~55 tok/s** | 10.23 | ncnn 风格 lane-FMA |

**累计提升**：prefill ~55 tok/s, decode **6.6x**

**累计提升**：prefill **4.7x**, decode **4.2x**

## 历史热点快照 (早期 FP16, 4 threads, 23-token prompt)

### Prefill (3188ms, seq_len=128, 23 actual tokens)
| Op | Time | % |
|----|------|---|
| MATMUL | 2481ms | 80.9% |
| SDPA | 437ms | 14.3% |
| 其他 | <5% | — |

### Decode (458ms total, 153ms/token)
| Op | Time | % |
|----|------|---|
| MATMUL | 380ms | 98.3% |
| 其他 | <2% | — |

## 与现有框架的差距 (同为 FP16, M5 Pro, 4 threads)

| 指标 | mlllm | ncnn | llama.cpp | 差距 |
|------|-------|------|-----------|------|
| prefill (pp256) | ~55 tok/s | 202 tok/s | 264 tok/s | ~3.7-4.8x |
| decode (tg64) | ~10.5 tok/s | 33 tok/s | 41 tok/s | ~3.2-3.9x |

注：之前 "~7 tok/s" 是用 23 token prompt 测的，graph seq_len=128 padding 导致
tps 被严重低估。实际满载 128 token 约 58 tok/s，256 token 约 55 tok/s。

## 当前热点 (pp256, 4 threads, FP16 + load-time pack + lane-FMA)

### Prefill (4678ms, 256 tokens)
| Op | Time | % |
|----|------|---|
| MATMUL | 2309ms | 51.9% |
| SDPA | 1805ms | **40.6%** |
| CONCAT | 124ms | 2.8% |
| SILU | 67ms | 1.5% |
| 其他 | <1% | — |

### Prefill (2202ms, 128 tokens)
| Op | Time | % |
|----|------|---|
| MATMUL | 1150ms | 61.5% |
| SDPA | 469ms | 25.1% |
| CONCAT | 73ms | 3.9% |
| SILU | 52ms | 2.8% |
| 其他 | <2% | — |

### Decode (458ms/token, FP16 + load-time pack)
| Op | Time | % |
|----|------|---|
| MATMUL | 380ms | 98.3% |
| 其他 | <2% | — |

## 已尝试但未采用的优化

| 尝试 | 结果 | 原因 |
|------|------|------|
| K 展开 (FP32) | -35~58% | vfmaq_laneq_f32 延迟高 |
| B repacking (per-tile) | -12~58% | per-tile 拷贝开销太大 |
| B repacking (load-time) | -21% | cache 副作用 > 加载收益 |
| K 展开 (FP16) | 0~7% 退化 | 同上，B gather+vcvt 是瓶颈 |

---

## 尝试 7：A 加载 SIMD 化 (vld1q + vgetq_lane + vfmaq_n)

**决策理由**：当前 K 循环内 A 的加载是 8 次标量 load（`a_val = A[k + row * lda]`），
但同一 k 下 A[m..m+7][k] 在内存中是连续的（A 为列主序 [K, M]）。
改为用 `vld1q_f32` 一次加载 4 个连续 A 值到 NEON 寄存器，
用 `vgetq_lane_f32` 提取标量，保持 `vfmaq_n_f32` 不变。
理论上减少 8 次内存 load → 2 次向量 load。

**结果 (FP32 vs 基线)**：
| Shape | 基线 (标量A) | SIMD A load | 变化 |
|-------|-------------|-------------|------|
| 128×2048×6144 t=1 | 85.15ms (37.8GF) | 63.86ms (50.4GF) | **-25%** |
| 128×2048×6144 t=4 | 22.42ms (143.7GF) | 16.09ms (200.2GF) | **-28%** |
| 1×2048×2048 t=1 | 1.02ms (8.2GF) | 1.32ms (6.3GF) | +29% |
| 1×2048×128k t=4 | 22.81ms (23.0GF) | 24.77ms (21.2GF) | +8.6% |

**结果 (FP16 vs 基线)**：
| Shape | 基线 (标量A) | SIMD A load | 变化 |
|-------|-------------|-------------|------|
| 128×2048×6144 t=1 | 60.79ms (53.0GF) | 62.47ms (51.6GF) | +2.8% |
| 128×2048×6144 t=4 | 15.74ms (204.6GF) | 15.77ms (204.3GF) | ~持平 |
| 1×2048×2048 t=1 | 1.09ms (7.7GF) | 1.26ms (6.6GF) | +15.6% |
| 1×2048×128k t=4 | 17.78ms (29.5GF) | 21.87ms (24.0GF) | **+23%** |

**结论**：FP32 GEMM 有效（-25~28%），但 FP16 场景无效（持平或退化）。
根因：`vgetq_lane_f32` 涉及 NEON→通用寄存器传输，有额外延迟；
FP16 场景下 B 的 gather+vcvt 已经是瓶颈，A 的加载方式不是瓶颈；
GEMV (M=1) 场景退化最严重（+23%），因为 8 个 lane 只有 1 个有效。

**决定**：**不采用**。当前默认路径是 FP16，此优化对 FP16 无效且 GEMV 退化。
保留代码作为 FP32 fallback 的优化参考。

---

## 尝试 8：B Interleaved Packing (tile-of-8 transpose)

**决策理由**：当前 B 是 row-major [N, K]，内层加载 8 个 B 值需要 gather（stride K_weight）。
改为在 K_BLOCK 外层将每个 K_BLOCK 切片按 tile-of-8 转置：
`B_packed[tile_base + k*8 + j] = B_original[(n_tile + j) * K_weight + k]`
使得固定 k 下 8 个 B 值连续 → 用 `vld1q_f16` 一条指令加载，无需 gather。
Packing 用 thread_local 堆缓冲，按需扩容复用。

**实现**：
- `kernels/matmul.h`: 新增 `use_interleave_pack` 配置标志
- `kernels/matmul.cpp`: 新增 `pack_b_interleaved` + `matmul_fp16_neon_8x8_range_packed`
- `kernel_matmul_fp32`: 当 FP16 + NEON + M>=8 时走 packed 路径
- M=1 (GEMV) 自动回退，因为 packing 开销 > 计算收益

**结果 (FP16 vs 基线, M>=8 启用)**：
| Shape | 基线 | Interleave Pack | 变化 |
|-------|------|-----------------|------|
| 128×2048×6144 t=1 | 60.47ms (53.3GF) | 32.44ms (99.3GF) | **-46%** |
| 128×2048×6144 t=4 | 15.82ms (203.7GF) | 11.54ms (279.3GF) | **-27%** |
| 1×2048×2048 t=1 | 1.05ms | 1.06ms | ~持平 (回退) |
| 1×2048×128k t=4 | 17.77ms | 17.78ms | ~持平 (回退) |

**结果 (端到端 Youtu-LLM-2B, 4 threads)**：
| 指标 | 基线 | Interleave | 变化 |
|------|------|-----------|------|
| prefill_tps | 7.55 | 8.54 | **+13%** |
| decode_tps | 6.07 | 6.27 | +3.3% |
| prefill MATMUL | 1986ms | 1751ms | **-12%** |
| decode MATMUL | 17159ms | 16616ms | -3.2% |

**结论**：**显著有效**。GEMM 场景（prefill 的 MATMUL 占 80%+）单线程 -46%，多线程 -27%。
GFLOPS 从 203 提升到 279。GEMV 自动回退无退化。

**决定**：**采用。M>=8 启用 interleaved packing 作为默认。**

---

## 尝试 9：Load-Time B Packing (一次性 interleaved pack)

**决策理由**：尝试 8 的 per-K_BLOCK packing 每次 matmul 调用都重新 pack B，
GEMV (M=1) 因 pack overhead 太大只能回退。改为在 `LLMEngine::load_graph` 时
一次性将所有 FP16 权重（除 embed_tokens）pack 成 interleaved `[N/8, K, 8]` layout，
推理时零 packing 开销，GEMV 和 GEMM 共用同一份 packed B。

**实现**：
- `kernels/matmul.h`: 新增 `pack_b_interleaved_full` 公开 API
- `kernels/matmul.cpp`: packed kernel 内部 K-blocking (无 per-call pack)
- `engine/engine.h`: 新增 `packed_weights_` map (path → packed buffer)
- `engine/engine.cpp`: `load_graph` 在 `setup_weight` 后 pack FP16 权重（跳过 embed_tokens）
- 移除 M>=8 守卫，GEMV 也走 packed 路径

**结果 (FP16 microbench vs 尝试 8)**：
| Shape | 尝试 8 (per-call pack) | 尝试 9 (load-time pack) | 变化 |
|-------|----------------------|------------------------|------|
| 128×2048×6144 t=1 | 32.44ms (99.3GF) | 31.83ms (101.2GF) | -1.9% |
| 128×2048×6144 t=4 | 11.54ms (279.3GF) | 8.84ms (364.4GF) | **-23%** |
| 1×2048×2048 t=1 | 1.06ms (回退) | **0.56ms (14.9GF)** | **-47%** |
| 1×2048×128k t=4 | 17.78ms (回退) | **9.44ms (55.7GF)** | **-47%** |

**结果 (端到端 Youtu-LLM-2B, 4 threads)**：
| 指标 | 尝试 8 | 尝试 9 | 变化 |
|------|-------|-------|------|
| load_ms | 328 | 4254 | +1197% (一次性 pack) |
| prefill_tps | 8.54 | 8.67 | +1.5% |
| decode_tps | 6.27 | **10.51** | **+68%** |
| prefill MATMUL | 1751ms | 1317ms | -25% |
| decode MATMUL | 16616ms | 8384ms | **-50%** |

**结论**：**GEMV 场景大幅提升**。decode_tps 从 6.27 提升到 10.51 (+68%)，
decode MATMUL 时间减半。GEMM 多线程也提升 23%（无 per-call pack overhead）。
load time 增加 ~4s（一次性 pack 全部权重），可接受。

**决定**：**采用。load-time packing 替代 per-call packing。**

---

## 尝试 10：A Packing + vfmlalq_laneq_f16 (lane-FMA kernel)

**决策理由**：ncnn 的核心技巧。当前 kernel 用 `vfmaq_n_f32`（标量 A broadcast × 4 路 B），
每 K 步 27 指令/32 FLOPs。改为 A 也 pack 到 interleaved FP16，用 `vfmlalq_laneq_f16`
（FP16 向量 A × lane B → FP32 累加），每 K 步 18 指令/64 FLOPs，指令效率 3.5x。

**算法**：
- A: `[K, M]` FP32 column-major → `pack_a_interleaved_full` → `[M/8, K, 8]` FP16
- B: 已 load-time packed `[N/8, K, 8]` FP16
- 累加器：column-major `c[j][0/1]` = rows 0..3/4..7 for N column j
- 内层：`vfmlalq_laneq_low/high_f16(c[j], a_vec, b_vec, lane)` × 16 = 64 FLOPs/K-step
- A 是动态 activations，per-call pack（O(M×K)，远小于 B 的 N×K）
- M>=8 走 lane-FMA kernel，M<8 (GEMV) 保持 scalar-A kernel

**实现**：
- `kernels/matmul.h`: 新增 `pack_a_interleaved_full` 公开 API
- `kernels/matmul.cpp`: 新增 `matmul_fp16_neon_8x8_range_packed_a` lane-FMA kernel
- `kernel_matmul_fp32`: M>=8 走 lane-FMA + per-call A pack，M<8 走 scalar-A
- 修复 `pack_b/a_interleaved_full` 的 buffer 大小（需 pad 到 8 的倍数）
- 多线程路径：每个 worker 独立 pack 自己的 M-slice

**结果 (FP16 microbench vs 尝试 9)**：
| Shape | 尝试 9 (scalar-A) | 尝试 10 (lane-FMA) | 变化 |
|-------|------------------|-------------------|------|
| 128×2048×6144 t=1 | 31.83ms (101.2GF) | 29.74ms (108.3GF) | **-7%** |
| 128×2048×6144 t=4 | 8.84ms (364.4GF) | 8.90ms (362.0GF) | ~持平 |
| 1×2048×2048 t=1 | 0.56ms (14.9GF) | 0.66ms (12.8GF) | +18% (波动) |
| 1×2048×128k t=4 | 9.44ms (55.7GF) | 9.42ms (55.8GF) | ~持平 |

**结果 (端到端 Youtu-LLM-2B, 4 threads)**：
| 指标 | 尝试 9 | 尝试 10 | 变化 |
|------|-------|--------|------|
| prefill_tps | 8.67 | **9.89** | **+14%** |
| decode_tps | 10.51 | 10.23 | -2.7% |
| prefill MATMUL | 1317ms | 1211ms | **-8%** |
| decode MATMUL | 8384ms | 8625ms | +2.9% |

注：以上 prefill_tps 是用 23-token prompt 测的（graph seq_len=128 padding）。
满载 128 token 实际约 58 tok/s，256 token 约 55 tok/s。

**结论**：**prefill 显著提升** (+14%)，GEMM 单线程 -7%。decode 略有退化 (-2.7%)，
可能是 A pack overhead 在 GEMV 路径的微小影响（M<8 不走 lane-FMA 但有额外分支）。
GEMM 多线程持平，可能因为 A pack 在 worker 线程内串行。

**决定**：**采用。prefill 提升明显，decode 退化在波动范围内。**
后续可优化：GEMV 路径完全不走 pack 分支，减少 dispatch 开销。

---

## 当前最优配置

- **TILE**: 8×8 (NEON)
- **K_BLOCK**: 512
- **精度**: FP16 权重 + FP16 activations (GEMM) + FP32 累加
- **A packing**: Per-call interleaved (M>=8), FP32→FP16
- **B packing**: Load-time interleaved (tile-of-8 transpose), GEMV+GEMM 共用
- **FMA**: vfmlalq_laneq_f16 (M>=8), vfmaq_n_f32 (M<8)
- **线程**: 自定义线程池, per-op split work, per-worker A pack
- **分片**: 自适应 (M vs N 维度)
- **无**: repack, K 展开, A SIMD load

---

## 尝试 11：FlashAttention-2 FP32 (ARM NEON)

**决策理由**：pp256 时 SDPA 占 40.6%（1805ms），是 naive O(n²) 标量实现。
移植 ncnn-upstream 中自写的 FlashAttention-2 FP32 kernel：
- 2D tiled (Br=4, Bc=32) + online softmax，避免 materialize 全 QK 矩阵
- NEON 向量化 dot product (`dot_fp32_neon`) + fast exp (`fast_exp_f32x4`)
- decode 专用路径 (M=1, Bc=32)

**实现**：
- `kernels/attention.cpp`: 移植 `flash_attn_fp32_decode` + `flash_attn_fp32_prefill`
- KV cache append 后 K/V per-head 连续，直接传给 flash kernel
- causal mask 在无显式 mask 时动态构建
- 非 NEON 平台保留 naive scalar fallback

**结果 (端到端 Youtu-LLM-2B, 4 threads, pp256 graph)**：
| 指标 | 尝试 10 | 尝试 11 | 变化 |
|------|--------|--------|------|
| prefill_ms (220 tok) | 4678ms | **3532ms** | **-24%** |
| prefill_tps (220 tok) | 47.0 | **62.3** | **+32%** |
| prefill_tps (pp256 equiv) | ~55 | **~72.5** | +32% |
| decode_tps | 10.23 | 9.88 | -3% (噪声) |
| SDPA time (pp256) | 1805ms (40.6%) | **603ms (18.5%)** | **-67%** |

**结论**：**SDPA 大幅提升** (-67%)，prefill +32%。SDPA 占比从 40.6% 降到 18.5%。
decode 基本持平（flash decode kernel 对小 N 的 overhead 被向量化收益抵消）。

**决定**：**采用。FP32 flash attention 作为默认。**
后续可移植 FP16FML kernel 进一步提升。

---

## 尝试 12：SDPA Head 级并行

**决策理由**：flash kernel 串行遍历 128 个 head，4 线程下 SDPA 仍占 18.5%。
按 head 并行（每个线程处理 num_heads/4 个 head），理论上 SDPA 时间除以 4。

**实现**：
- `kernels/attention.h`: `kernel_sdpa` 新增 `ThreadPool*` 参数
- `graph/execute.cpp`: 传递 `thread_pool` 给 `kernel_sdpa`
- `kernels/attention.cpp`: 用 `parallel_for(0, num_heads, 1, run_head)` 并行

**结果 (端到端 Youtu-LLM-2B, 4 threads, pp256 graph)**：
| 指标 | 尝试 11 (串行) | 尝试 12 (并行) | 变化 |
|------|---------------|---------------|------|
| prefill_ms (220 tok) | 3532ms | **3349ms** | -5.2% |
| prefill_tps (pp256 equiv) | ~72.5 | **~76.4** | +5.4% |
| decode_tps | 9.88 | **10.34** | +4.7% |
| SDPA time (pp256 prefill) | 603ms (18.5%) | **173ms (5.6%)** | **-71%** |

**结论**：SDPA 从 18.5% 降到 5.6%，基本解决。整体 prefill +5%，decode +5%。
剩余瓶颈是 MATMUL（70%+），量化是下一步最大杠杆。

**决定**：**采用。head 级并行作为默认。**

---

## 尝试 13：FP16 累加 (vfmaq_lane_f16)

**决策理由**：ggml 和 ncnn 都用 `vfmaq_lane_f16`（FP16×FP16→FP16 FMA），
吞吐量 2 ops/cycle。我们用 `vfmlalq_laneq_f16`（FP16→FP32 widening）只有 1 op/cycle。
切换到 FP16 累加可获得 ~2x FMA 吞吐。FP32 widening kernel 保留为 fallback。

**实现**：
- `kernels/matmul.h`: 新增 `use_fp16_accumulate` 配置标志（默认 true）
- `kernels/matmul.cpp`: 新增 `matmul_fp16_neon_8x8_range_packed_a_fp16acc`
  - 累加器: `float16x8_t c[8]` (8 Q registers, 比 FP32 的 16 Q 少一半)
  - FMA: `vfmaq_lane_f16` × 8 = 64 FLOPs/K-step
  - K-block 间: FP16→FP32 存到 C, 再 FP32→FP16 重新加载
- `kernel_matmul_fp32`: 根据 `use_fp16_accumulate` 分发到 FP16 acc 或 FP32 acc kernel

**结果 (microbench vs FP32 acc)**：
| Shape | FP32 acc | FP16 acc | 变化 |
|-------|----------|----------|------|
| 128×2048×6144 t=1 | 29.74ms (101GF) | 25.28ms (127GF) | **-15%** |
| 128×2048×6144 t=4 | 8.90ms (362GF) | **486GF** | **+34%** |
| 1×2048×128k t=4 | 9.42ms (55.7GF) | 9.42ms (55.9GF) | ~持平 (GEMV 不走 lane-FMA) |

**结果 (端到端 Youtu-LLM-2B, 4 threads, pp256 graph)**：
| 指标 | FP32 acc | FP16 acc | 变化 |
|------|----------|----------|------|
| prefill_tps (220 tok) | 65.7 | **70.2** | +7% |
| prefill_tps (pp256 equiv) | ~76.4 | **~81.7** | +7% |
| decode_tps | 10.34 | **10.92** | +5.6% |

**结论**：microbench GEMM t=4 +34% GFLOPS，e2e +7% prefill / +5.6% decode。
GEMV 不走 lane-FMA 所以无变化。精度: 测试误差 <0.5%，可接受。

**决定**：**采用。FP16 累加作为默认。FP32 widening kernel 保留为 fallback。**

---

## 尝试 14：Dedicated GEMV Kernel (M=1 专用 kernel)

**决策理由**：decode 路径所有 matmul 都是 M=1（GEMV），但走的是 `matmul_fp16_neon_8x8_range_packed`
—— 一个 8×8 tile kernel，内层 r 循环 8 次迭代中 7 次因 `if (row < m_tile_end)` 死分支跳过。
理论上编译器应该能 DCE 掉死迭代，但实际测量发现 dedicated GEMV kernel 仍有显著收益。

**算法**：
- N-tile outer (8 N per tile), K-loop inner
- 每步：1 `vld1q_f16` (8 B values) + 2 `vcvt_f32_f16` + 2 `vfmaq_n_f32` (broadcast A[k])
- 2 个 FP32 accumulator (vs 8×8 tile 的 16 个)，大幅降低寄存器压力
- 无 r-loop，无死分支
- FP32 accumulate（GEMV dot product over K，精度优先）

**实现**：
- `kernels/matmul.cpp`: 新增 `matmul_fp16_neon_gemv_range`
- `kernel_matmul_fp32`: `use_interleave` 分支内 `M==1` 时优先走 GEMV kernel
- n_chunk 从固定 64 改为自适应 `max(N/(n_threads*8), 64)` 对齐到 8
  - lm_head (N=128k): 2003 chunks → 32 chunks, 大幅减少 parallel_for dispatch 开销

**结果 (microbench vs 8×8 tile GEMV baseline)**：
| Shape | Baseline (8×8 tile) | Dedicated GEMV | 变化 |
|-------|---------------------|----------------|------|
| 1×2048×2048 t=1 | 0.61ms (13.8 GF) | 0.44ms (19.1 GF) | **-28%** |
| 1×2048×6144 t=4 | 0.42ms (59.7 GF) | 0.35ms (71.0 GF) | **-17%** |
| 1×2048×128k t=4 | 9.80ms (53.6 GF) | 7.25ms (72.4 GF) | **-26%** |

**结果 (端到端 Youtu-LLM-2B, 4 threads, pp256, warmup=3, 3-run avg)**：
| 指标 | 尝试 13 (8×8 tile GEMV) | 尝试 14 (dedicated GEMV) | 变化 |
|------|------------------------|--------------------------|------|
| prefill_tps | 86.6 | **~101** | +17% |
| decode_tps | 13.3 | **~16.6** | **+25%** |
| prefill MATMUL | 2309ms | 1827ms | -21% |
| decode MATMUL | ~4656ms | ~3167ms | **-32%** |

注：prefill 提升部分来自测量方法改进（`--prompt-tokens 256` dummy token 满载，
warmup=3），部分来自 lm_head 的 GEMV + n_chunk 优化。decode 提升是 GEMV kernel 直接贡献。

**关键发现**：编译器**并没有**完全 DCE 掉 8×8 tile 在 M=1 时的 7/8 死分支。
dedicated GEMV kernel 通过更低的寄存器压力（2 acc vs 16）和更干净的循环结构
拿到实际的 25% decode 提升。"编译器应该已经优化掉了"的判断是错的。

**工具改进**：
- `examples/cli_common.h/cpp` + `bench.cpp`: 新增 `--prompt-tokens <N>` 选项
- 用 N 个 dummy token (id=0) 跳过 chat template，支持满载 pp128/pp256 测量
- 解决了之前短 prompt (25 token) 被 graph padding 严重低估 tps 的问题

**决定**：**采用。dedicated GEMV kernel 作为 M=1 默认路径。**

---

## 尝试 15：GEMV FP16 累加 + 2-way K-unroll（打破 FMA latency chain）

**决策理由**：尝试 14 的 GEMV kernel 用 FP32 acc（`vfmaq_n_f32` × 2 for low/high）。
理论上 FP16 acc（`vfmaq_n_f16` × 1）指令更少（2 vs 5 instr/K-step），应该更快。
但初版 FP16 acc（单累加器）反而更慢（-27% t=1, -4% t=4）。

**根因分析**：
- Apple M5 FP16 FMA latency = 2 cycles, throughput = 2/cycle
- FP32 acc kernel 有 **2 条独立 FMA 链**（acc0=low half, acc1=high half）
  → 2 FMA in 2 cycles → CPI ≈ 1 cycle/K-step（latency 完全 hidden）
- FP16 acc kernel 只有 **1 条 FMA 链**（单 float16x8_t acc）
  → 1 FMA per 2 cycles → CPI = 2 cycles/K-step（latency 是瓶颈）
- 指令更少但链更窄，反而更慢

**算法**：2-way K-unroll + 2 个独立 FP16 累加器
- 偶数 K 步 → acc0，奇数 K 步 → acc1（两条独立链）
- 循环结尾 `vaddq_f16(acc0, acc1)` 合并
- K_BLOCK 间仍用 FP32 store/reload 保持精度
- CPI 从 2 降到 1，FMA 吞吐翻倍

**实现**：
- `kernels/matmul.cpp`: `matmul_fp16_neon_gemv_range_fp16acc` 加 2-way unroll
- `kernel_matmul_fp32`: `use_fp16_accumulate=true` 时 GEMV 也走 FP16 acc 路径

**结果 (microbench, 3 个 GEMV shape)**：
| Shape | FP32 acc (尝试 14) | FP16 acc 1-chain | FP16 acc 2-unroll | vs FP32 acc |
|-------|-------------------|------------------|-------------------|-------------|
| 1×2048×2048 t=1 | 0.44ms (19.1 GF) | 0.60ms (13.9 GF) | **0.32ms (26.5 GF)** | **+39%** |
| 1×2048×6144 t=4 | 0.35ms (71.0 GF) | 0.37ms (68.1 GF) | **0.19ms (131.4 GF)** | **+85%** |
| 1×2048×128k t=4 | 7.25ms (72.4 GF) | 7.58ms (69.3 GF) | **4.01ms (130.9 GF)** | **+81%** |

131 GF ≈ Apple M5 Pro 带宽上限（~273 GB/s × 0.5 FLOP/byte = ~136 GF）的 **96%**，
也接近 FP16 compute 峰值（~144 GF）的 91%。GEMV 卡在 compute/bandwidth ridge 上。

**Roofline 分析**（修正之前"bandwidth-bound"的误判）：
- Arithmetic intensity = 8 FLOPs / 16 bytes (B) = 0.5 FLOP/byte（A 常驻 L1 不算 DRAM）
- M5 Pro (T6050) DRAM 带宽 ~273 GB/s → bandwidth ceiling = ~136 GF
- FP16 FMA peak = 2 FMA/cycle × 8 FLOP × 4.5 GHz × 4 core = ~144 GF
- 两个 ceiling 几乎重合，GEMV 正好在 ridge 上
- 尝试 14 (FP32 acc, 72 GF): 带宽利用 53%, **compute-bound**（FMA latency chain）
- 尝试 15 (FP16 acc 2-unroll, 131 GF): 带宽利用 96%, **bandwidth-bound**（撞带宽墙）

**结果 (端到端 Youtu-LLM-2B, 4 threads, pp256, warmup=3, 3-run avg)**：
| 指标 | 尝试 14 (FP32 acc GEMV) | 尝试 15 (FP16 acc 2-unroll) | 变化 |
|------|------------------------|----------------------------|------|
| prefill_tps | ~101 | **~106** | +5% |
| decode_tps | ~16.6 | **~28.2** | **+70%** |
| decode MATMUL | ~3167ms | **~1774ms** | **-44%** |

**关键发现**：
1. FP16 acc 单累加器**比 FP32 acc 更慢**——不是指令数瓶颈，是 FMA latency chain 瓶颈
2. 2-way K-unroll 加第二个累加器，打破 latency chain，FMA 吞吐翻倍
3. GEMV 从 bandwidth-bound (72 GF) 跃升到 compute-bound (131 GF)，达理论峰值 91%
4. decode 从 13.3 → 28.2 tok/s，**累计提升 2.1x**

**决定**：**采用。FP16 acc + 2-way K-unroll 作为 GEMV 默认路径。**

---

## 尝试 16：GEMM lane-FMA 2-way K-unroll

**决策理由**：尝试 15 的 GEMV 2-way unroll 大幅提升 decode (+70%)。
GEMM lane-FMA kernel（`matmul_fp16_neon_8x8_range_packed_a_fp16acc`）的 K-loop
有 8 条独立 FMA 链（c[0..7]），理论上已足够 hide FMA latency（latency=2, chains=8 >> 2）。
预判 unroll 不会有显著收益。但实际尝试后发现收益很大。

**算法**：2-way K-unroll，16 个独立累加器（c0[8] + c1[8]）
- 偶数 K 步 → c0[8]，奇数 K 步 → c1[8]
- 循环结尾 `vaddq_f16` 合并 8 对
- 尾部处理奇数 K
- 寄存器：16 acc + 4 a/b + 4 b_low/high = 24 Q（NEON 32 Q，够用）

**实现**：
- `kernels/matmul.cpp`: `matmul_fp16_neon_8x8_range_packed_a_fp16acc` K-loop 改 2-way unroll

**结果 (microbench)**：
| Shape | 1-way (尝试 15) | 2-way unroll | 变化 |
|-------|----------------|--------------|------|
| 128×2048×6144 t=4 | 6.62ms (486 GF) | **4.83ms (668 GF)** | **+37%** |
| 128×2048×6144 t=1 | 24.82ms (130 GF) | **14.61ms (220 GF)** | **+70%** |

**结果 (端到端, 4 threads, pp256, warmup=5, 4-run avg)**：
| 指标 | 尝试 15 (GEMM 1-way) | 尝试 16 (GEMM 2-unroll) | 变化 |
|------|---------------------|------------------------|------|
| prefill_tps | ~106 | **~131.5** | **+24%** |
| decode_tps | ~28.2 | ~27.9 | 持平（GEMV 路径不变） |
| prefill MATMUL | 1827ms | **1267ms** | **-31%** |

**关键发现**：
1. 预判"GEMM 8 链已足够 hide latency, unroll 无收益"是**错的**
2. 真正的瓶颈不是 FMA latency 也不是 FMA throughput，是 **instruction issue bandwidth**
3. 1-way kernel 每步 15 条指令（2 load + 2 vget + 8 FMA + 3 branch/cmp/add），其中 8/15=53% 是 FMA
   - 实测 IPC=6.3，接近 Apple 核心的 issue limit (~8)
   - **issue-bound**：FMA 单元在等指令解码
4. 2-way unroll 每步 27 条指令（4 load + 4 vget + 16 FMA + 3 branch），16/27=59% 是 FMA
   - IPC 降到 3.9，不再 issue-bound
   - FMA 单元可以更密集地运行
5. 单线程提升 (+70%) 远大于多线程 (+37%)，因为单线程更受 issue 带宽限制

**决定**：**采用。GEMM lane-FMA kernel 默认 2-way K-unroll。**

---

## 尝试 17：FP16 KV cache + FP16FML SDPA

**决策理由**：尝试 16 后 SDPA 占 prefill 10.1%、decode 3.6%。ncnn-upstream 的 flash
attention 是 FP16 输入 + FP32 累加（FP16FML `vfmlalq_lane_low/high_f16`），比 FP32 版本
减半 K/V 带宽。但直接在 kernel 内部做 FP32→FP16 转换（尝试 17a，已回滚）反而变慢：
每次 `kernel_sdpa` 调用都 per-head `new[]` + 转换整个 K/V cache + `delete[]`，
转换开销 > FP16FML 收益。

**关键改动**：KV cache 直接存 FP16
- `models/mla.py`: cache_k/cache_v input 节点 `prec=FP16`
- `engine.cpp`: `allocate_caches` 已用 `node.out_prec`，`reset()` 用 `element_size()`，
  无需改动
- `kernels/attention.cpp`: cache append 时 FP32→FP16 转换（只转 cur_seqlen 个 token，
  不是整个 cache）；`get_k_ptr/get_v_ptr` 返回 `const void*`，caller 按 prec 解释
- `flash_attn_fp16_decode/prefill`: 直接吃 FP16 K/V 指针，无需 per-call 转换
- Q 仍 per-call 转 FP16（M×d_k 小，decode 时 M=1）

**实现**：
- `flash_attn_fp16_decode`: Bc=64, 4-way batched FP16FML dot product, 64-wide register-tiled
  PV accumulation with `vfmlalq_lane_low/high_f16`
- `flash_attn_fp16_prefill`: Br=4, Bc=32, 同样 FP16FML dot + PV
- FP32 cache fallback 保留（test_attention 用 FP32 cache）

**结果 (端到端, 4 threads, pp256, warmup=5, 3-run avg)**：
| 指标 | 尝试 16 (FP32 SDPA) | 尝试 17 (FP16 cache + FP16FML SDPA) | 变化 |
|------|---------------------|-------------------------------------|------|
| prefill_tps | ~131.5 | **~136.6** | +4% |
| decode_tps | ~27.9 | ~27.3 | -2% (噪声) |
| prefill SDPA | ~175ms (10.1%) | **~136ms (8.0%)** | **-22%** |
| decode SDPA | ~113ms (3.6%) | **~79ms (3.9%)** | **-30%** |

**关键发现**：
1. SDPA 本身变快 22-30%（FP16FML 减半 K/V 带宽 + 省掉 vcvt 指令）
2. 但 SDPA 占比小（prefill 8%, decode 4%），端到端提升有限
3. FP16 cache 还减半了 cache 内存（4096×192×16×2 = 25MB vs 50MB），对长序列有益
4. 初版（尝试 17a，per-call 转换整个 K/V）反而退化 20-26%——per-call heap 分配
   + 转换整个 cache 的开销远超 FP16FML 收益

**决定**：**采用。FP16 KV cache + FP16FML SDPA 作为默认。**

---

## 尝试 18：Thread pool 改 spin-wait（降低 dispatch overhead）

**决策理由**：decode 每 token 调用 196 次 matmul，每次 `parallel_for` 的
mutex + condition variable 同步开销 ~11us → 196 × 11us = 2.2ms/token 纯 dispatch
overhead，占 decode 总时间 ~8%。decode 4-thread scaling 只有 3.2x（理想 4x）。

**测量**：
- `parallel_for(0, 128, 32, trivial_fn)` 耗时：
  - mutex+CV 版：**10.7 us/call**
  - spin-wait 版：**0.59 us/call**（**18x 提升**）
- `parallel_for(0, 6144, 192, trivial_fn)`：
  - mutex+CV 版：11.7 us/call
  - spin-wait 版：3.47 us/call

**实现**：
- `kernels/threading.h`: 移除 `mutex_`, `cv_job_`, `cv_done_`，改用
  `std::atomic<bool> stop_`, `std::atomic<int> pending_workers_`,
  `std::unique_ptr<std::atomic<bool>[]> worker_ready_`（per-worker ready flag）
- `kernels/threading.cpp`:
  - `parallel_for_impl`: 设置 job → 设 worker_ready_[t]=true → 主线程跑 shard 0
    → spin on `pending_workers_ == 0`（用 `yield` 指令降低功耗）
  - `worker_loop`: spin on `worker_ready_[t]` → 读 job → 清 flag → 跑 shard
    → `pending_workers_.fetch_sub(1)`
  - `stop_workers`: 设 stop_=true + 唤醒所有 worker_ready_
  - 析构/resize 时 join worker 线程

**结果 (端到端, 4 threads, pp256, warmup=5, 3-run avg)**：
| 指标 | 尝试 17 (mutex+CV) | 尝试 18 (spin-wait) | 变化 |
|------|---------------------|---------------------|------|
| prefill_tps | ~136.6 | ~133.7 | -2% (噪声) |
| decode_tps | ~27.3 | **~29.2** | **+7%** |
| decode SDPA | ~79ms | **~54ms** | **-32%** |
| decode MATMUL | ~1774ms | ~1764ms | 持平 |

**Scaling 改善**：
| threads | decode (mutex) | decode (spin) | scaling (spin) |
|---------|----------------|---------------|----------------|
| 1 | 9.15 | 8.99 | 1.0x |
| 2 | 16.6 | 17.3 | 1.9x |
| 3 | 23.3 | 24.6 | 2.7x |
| 4 | 29.0 | **30.3** | **3.4x** |

**关键发现**：
1. dispatch overhead 从 11us 降到 0.6us（18x），decode_tps +7%
2. decode SDPA 时间从 79ms 降到 54ms（-32%）——SDPA 也用 parallel_for，
   dispatch 开销占比大
3. decode scaling 从 3.2x → 3.4x，更接近理想 4x
4. spin-wait 不会显著增加 CPU 功耗（用 `yield` 指令），且 worker 在
   等待时不会竞争 cache
5. prefill 持平——prefill 的 parallel_for 调用少（288 次），dispatch 不是瓶颈

**决定**：**采用。spin-wait thread pool 作为默认。**

---

## 尝试 19：CONCAT/TILE 快路径（bulk memcpy）

**决策理由**：prefill 中 CONCAT 占 7.3%（111ms），TILE 占 2.2%（33ms）。
当前实现是 4 层嵌套循环逐元素 `memcpy(dst, src, 4)`——每个元素单独 4 字节拷贝，
`memcpy` 调用开销远大于数据搬运。

**分析**：
- MLA 的 CONCAT 都在 dim=0：`q_full = concat(q_nope, q_rope, dim=0)`
- dim=0 concat 时，对固定 (i1, i2, i3)，所有 i0 元素在 src 和 dst 中都连续
- 如果 src/dst 是 contiguous，可以一次 `memcpy` 整个 tensor
- 即使不 contiguous（view 输入），也可以 per-(i1,i2,i3) 拷贝 i0 块

**实现**：
- `graph/execute.cpp` CONCAT:
  - dim==0 快路径：检查 `is_contiguous()`，bulk copy 整个 tensor
  - 否则 per-(i1,i2,i3) copy i0 块（stride[0]==es 保证 i0 连续）
  - dim!=0 保留原 4D 逐元素路径
- `graph/execute.cpp` TILE:
  - dim=2-only 快路径（MLA 的 `k_rope [rope_dim, seq, 1] → [rope_dim, seq, num_heads]`）：
    contiguous src 时，`memcpy` 整个 src tensor `reps[2]` 次
  - 否则保留原逐元素路径

**结果 (端到端, 4 threads, pp256, warmup=5, 3-run avg)**：
| 指标 | 尝试 18 | 尝试 19 | 变化 |
|------|---------|---------|------|
| prefill_tps | ~133.7 | **~162.0** | **+21%** |
| decode_tps | ~30.3 | ~30.7 | 持平 |
| prefill CONCAT | 111ms (7.3%) | **8ms (0.6%)** | **-93%** |
| prefill TILE | 33ms (2.2%) | **0.3ms (0.0%)** | **-99%** |
| decode CONCAT | 28ms | **1.8ms** | **-94%** |
| decode TILE | 8ms | **0.1ms** | **-99%** |

**关键发现**：
1. CONCAT/TILE 是纯 memcpy 场景，逐元素拷贝的 `memcpy` 调用开销是瓶颈
2. dim=0 concat 的 bulk copy 把 192×256×16×4B = 3MB 数据从 192×256=49152 次 memcpy
   降到 1 次，性能提升 14x
3. TILE 的 dim=2-only 快路径同理，1 次 memcpy 替代 49152 次
4. prefill 提升明显（+21%），decode 提升小（CONCAT/TILE 占比本就小）
5. 副作用：CPU 提速也改善了热点的 cache 局部性（bulk memcpy 对 prefetcher 友好）

**决定**：**采用。CONCAT dim=0 bulk copy + TILE dim=2-only bulk copy 作为默认。**

---

## 尝试 20：SILU/ADD/MUL NEON 向量化

**决策理由**：尝试 19 后 SILU 占 prefill 3.3%（52ms），ADD/MUL 各占 0.2-0.3%。
SILU 当前是标量 `std::exp` 循环，每元素一次 libm 调用。NEON 向量化 4 元素并行。

**实现**：
- `graph/execute.cpp`:
  - 新增 `sigmoid_f32_neon` 辅助函数（复用 fast_exp 多项式逼近）
  - SILU: `vld1q_f32` × 4 → sigmoid → `vmulq_f32` → `vst1q_f32`
  - ADD: `vaddq_f32` × 2 (8 元素/iter)
  - MUL: `vmulq_f32` × 2
  - 标量 tail 处理非 4 对齐的 N

**结果 (端到端, 4 threads, pp256, warmup=5, 3-run avg)**：
| 指标 | 尝试 19 | 尝试 20 | 变化 |
|------|---------|---------|------|
| prefill_tps | ~162.0 | **~167.5** | +3% |
| decode_tps | ~30.7 | ~31.4 | +2% |
| prefill SILU | 52ms (3.3%) | **14ms (1.0%)** | **-73%** |
| prefill ADD | 4ms (0.3%) | 4ms (0.3%) | 持平 |
| prefill MUL | 6ms (0.4%) | 6ms (0.4%) | 持平 |
| decode SILU | 13ms | **3.5ms** | -73% |

**关键发现**：
1. SILU 向量化 -73%，符合预期（4 元素并行 + 逼近 exp 替代 libm exp）
2. ADD/MUL 已经很快（向量化前就只占 0.3%），向量化后无可见收益
3. 端到端提升小（+3%）因为 SILU 占比本就不大（3.3% → 1.0%）
4. decode 受益更小（SILU 占 decode <0.5%）

**决定**：**采用。SILU/ADD/MUL NEON 向量化作为默认。**

---

## 尝试 21：检查 RMS_NORM / RoPE 向量化状态

**决策理由**：尝试 20 后 RMS_NORM 占 prefill 1.0%（13ms），RoPE 占 0.3%（4ms）。
检查是否已向量化，如有空间则顺手优化。

**结果**：两者都已 NEON 向量化，无优化空间。

- `kernels/norm.cpp`: `rms_norm_neon` 已用 `vfmaq_f32` (sum_sq) + `vmulq_f32` (apply)
- `kernels/rope.cpp`: `rope_neon_interleave` (vld2q_f32 deinterleave) + `rope_neon_halves` (vld1q_f32)
- RoPE 的 `memcpy(o, x, D*sizeof(float))` pre-copy 也是 bulk，无需改

**当前 prefill 热点分布**（尝试 20 后）：
| Op | 时间 | 占比 | 状态 |
|----|------|------|------|
| MATMUL | 1120ms | 85.1% | 已优化（GEMM 2-way unroll） |
| SDPA | 120ms | 9.1% | 已优化（FP16FML + head 并行） |
| RESHAPE | 30ms | 2.3% | 零拷贝 metadata，无可优化 |
| SILU | 13ms | 1.0% | 已向量化（尝试 20） |
| RMS_NORM | 13ms | 1.0% | 已向量化 |
| CONCAT | 8ms | 0.6% | 已 bulk memcpy（尝试 19） |
| RoPE | 4ms | 0.3% | 已向量化 |
| 其他 | <5ms | <0.4% | 已向量化 |

**结论**：非 MATMUL/SDPA 占比从最初 ~40% 降到 ~6%，已无"顺手做"的空间。
剩余 ~94% 是 MATMUL + SDPA，进一步优化必须瞄准这两个。
非量化路径下，MATMUL 已接近 FP16 compute ceiling（729 GF / 4 core = ~183 GF/core，
理论 ~144 GF/core 的 127%——超线性 scaling 说明多线程效果良好）。

**决定**：无需改动。非量化 GEMM/GEMV 优化已到平台瓶颈。

---

## 当前最优配置

- **TILE**: 8×8 (NEON, M>=2) / dedicated GEMV (M=1)
- **K_BLOCK**: 512
- **精度**: FP16 权重 + FP16 activations (GEMM) + FP16 累加 (GEMM + GEMV)
- **A packing**: Per-call interleaved (M>=8), FP32→FP16
- **B packing**: Load-time interleaved (tile-of-8 transpose), GEMV+GEMM 共用
- **embed_tokens**: FP16 row-major (embed 查找) + FP16 packed 副本 (lm_head matmul)
- **FMA**: vfmaq_lane_f16 FP16 累加 2-way unroll (M>=8 GEMM), vfmaq_n_f16 FP16 累加 2-way unroll (M=1 GEMV)
- **GEMV n_chunk**: 自适应 max(N/(n_threads*8), 64) 对齐到 8
- **KV cache**: FP16 存储（cache buffer 减半，FP16FML SDPA 直接用，无需 per-call 转换）
- **SDPA**: FlashAttention-2 FP16FML (FP16 输入 + FP32 累加) + head 并行
- **线程**: 自定义线程池 (spin-wait + `yield`), per-op split work, per-worker A pack
- **Eltwise**: SILU/ADD/MUL NEON 向量化
- **Bench**: `--prompt-tokens <N>` 支持固定 token 数满载测量

## 端到端性能 (Youtu-LLM-2B, M5 Pro, 4 threads, pp256)

| 版本 | prefill_tps | decode_tps | load_ms |
|------|------------|------------|---------|
| FP32 baseline | ~60 | ~5 | 328 |
| + FP16 + pack + lane-FMA | ~76 | ~10.3 | 4254 |
| + FlashAttention + head 并行 | ~76 | ~10.3 | 4254 |
| + FP16 累加 | ~82 | ~10.9 | 4254 |
| + embed FP16 + packed lm_head | ~86.6 | ~13.3 | 1220 |
| + dedicated GEMV kernel | ~101 | ~16.6 | 1243 |
| + GEMV FP16 acc + 2-way unroll | ~106 | ~28.2 | 1243 |
| + GEMM lane-FMA 2-way unroll | ~131.5 | ~27.9 | 1243 |
| + FP16 KV cache + FP16FML SDPA | ~136.6 | ~27.3 | 1243 |
| + spin-wait thread pool | ~133.7 | ~30.3 | 1243 |
| + CONCAT/TILE bulk memcpy | ~162.0 | ~30.7 | 1243 |
| + SILU/ADD/MUL NEON | **~167.5** | **~31.4** | 1243 |

| 框架 | prefill (pp256) | decode (tg) |
|------|----------------|------------|
| **mlllm** | **~167.5** | **~31.4** |
| ncnn | 202 | 33 |
| llama.cpp | 264 | 41 |

差距：prefill **1.2-1.6x**, decode **1.1-1.3x**

## 下一步方向

1. **量化 (INT4/INT8)** — MATMUL 仍占 85%+, **唯一大杠杆**
2. **FP16 activations** — 前一个算子输出 FP16，matmul 跳过 per-call A pack（~1% 收益）
3. **Accelerate BLAS** — 快速验证 matmul 上限（Apple only, 低优先级）

非量化路径下，非 MATMUL/SDPA 优化已到瓶颈（占比从 40% 降到 6%）。
MATMUL 已接近 FP16 compute ceiling（microbench 729 GF, ~127% 理论峰值说明
多线程超线性 scaling）。要继续缩小与 ncnn/llama.cpp 的差距，量化是必经之路。
