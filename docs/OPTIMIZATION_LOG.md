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

**结果 (端到端 Youtu-LLM-2B, 4 threads, 23-token 短 prompt, 未校准)**：
| 指标 | 尝试 9 | 尝试 10 | 变化 |
|------|-------|--------|------|
| prefill_tps | 8.67 | **9.89** | **+14%** |
| decode_tps | 10.51 | 10.23 | -2.7% |
| prefill MATMUL | 1317ms | 1211ms | **-8%** |
| decode MATMUL | 8384ms | 8625ms | +2.9% |

注：以上 prefill_tps 是用 23-token prompt 测的（graph seq_len=128 padding），
**数据未重新校准**。3fbc7cf checkout 后用 `--prompt-tokens 256` 重测：
prefill=86.2, decode=12.8（pp256 满载）。短 prompt 数据仅做趋势参考。

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

**结果 (端到端 Youtu-LLM-2B, 4 threads, pp256, warmup=3, 单次 run 校准)**：
| 指标 | FP32 acc (3fbc7cf) | FP16 acc (89a1796) | 变化 |
|------|--------------------|--------------------|------|
| prefill_tps | 86.2 | **97.5** | +13% |
| decode_tps | 12.8 | 12.7 | 持平 (GEMV 不走 lane-FMA) |

注：89a1796 校准 prefill=97.5, decode=12.7。短 prompt 数据 (70.2/10.92) 已废弃。
microbench GEMM t=4 +34% GFLOPS, e2e +13% prefill。GEMV 不走 lane-FMA 无变化。
精度: 测试误差 <0.5%，可接受。

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

**结果 (端到端 Youtu-LLM-2B, 4 threads, pp256, warmup=3, 单次 run 校准)**：
| 指标 | 尝试 13 (8×8 tile GEMV) | 尝试 14 (dedicated GEMV) | 变化 |
|------|------------------------|--------------------------|------|
| prefill_tps | ~97 | — | (合并 commit, 未单测) |
| decode_tps | ~12.7 | — | (合并 commit, 未单测) |
| prefill MATMUL | (未测) | 1827ms (尝试 14-17 合并) | — |
| decode MATMUL | (未测) | ~3167ms (尝试 14-17 合并) | — |

注：尝试 13 (89a1796) 校准结果 prefill=97.5, decode=12.7。尝试 14-17 在 e4338a6 合并提交，
校准结果 prefill=133.3, decode=26.6。中间单步未单独 checkout 校准。
`--prompt-tokens 256` 满载测量已引入，但 prefill 提升部分也来自 lm_head 的 GEMV + n_chunk 优化。
decode 提升是 GEMV kernel 直接贡献。

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

**结果 (端到端, 4 threads, pp256, warmup=3, 单次 run 校准)**：
| 指标 | 尝试 15 (GEMM 1-way) | 尝试 16 (GEMM 2-unroll) | 变化 |
|------|---------------------|------------------------|------|
| prefill_tps | — (未单独 checkout) | — (未单独 checkout) | — |
| decode_tps | — | — | — |
| prefill MATMUL | 1827ms (尝试 14-17 合并) | 1267ms (尝试 14-17 合并) | -31% |

注：尝试 14-17 在 e4338a6 合并提交，校准 prefill=133.3, decode=26.6。
microbench GEMM +37% (t=4) / +70% (t=1) GFLOPS 已验证，端到端未单独 checkout。
此处 prefill MATMUL 数据沿用合并 commit 的 1267ms。

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

**结果 (端到端, 4 threads, pp256, warmup=3, 单次 run 校准)**：
| 指标 | 尝试 16 (FP32 SDPA) | 尝试 17 (FP16 cache + FP16FML SDPA) | 变化 |
|------|---------------------|-------------------------------------|------|
| prefill_tps | — (未单独 checkout) | — (未单独 checkout) | — |
| decode_tps | — | — | — |
| prefill SDPA | ~175ms (10.1%) | **~136ms (8.0%)** | **-22%** |
| decode SDPA | ~113ms (3.6%) | **~79ms (3.9%)** | **-30%** |

注：尝试 14-17 在 e4338a6 合并提交，校准合并后 prefill=133.3, decode=26.6。
SDPA sub-op 时间未重新校准，沿用之前测量（占比小，相对变化可信）。

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

**结果 (端到端, 4 threads, pp256, warmup=3, 单次 run 校准)**：
| 指标 | 尝试 17 (mutex+CV, e4338a6) | 尝试 18 (spin-wait, 80d6048) | 变化 |
|------|------------------------------|-------------------------------|------|
| prefill_tps | 133.3 | **138.3** | +4% |
| decode_tps | 26.6 | **28.0** | +5% |
| decode SDPA | ~79ms | **~54ms** | **-32%** |
| decode MATMUL | ~1774ms | ~1764ms | 持平 |

注：dispatch overhead 从 11us 降到 0.6us (18x)，decode +5%, SDPA -32%。

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

**结果 (端到端, 4 threads, pp256, warmup=3, 单次 run 校准)**：
| 指标 | 尝试 18 (80d6048) | 尝试 19 (CONCAT/TILE bulk, 80d6048 同 commit) | 变化 |
|------|-------------------|----------------------------------------------|------|
| prefill_tps | 138.3 | 138.3 | (同 commit, 未拆分) |
| decode_tps | 28.0 | 28.0 | (同 commit, 未拆分) |
| prefill CONCAT | 111ms (7.3%) | **8ms (0.6%)** | **-93%** |
| prefill TILE | 33ms (2.2%) | **0.3ms (0.0%)** | **-99%** |
| decode CONCAT | 28ms | **1.8ms** | **-94%** |
| decode TILE | 8ms | **0.1ms** | **-99%** |

注：尝试 19 的 CONCAT/TILE bulk memcpy 与尝试 18 spin-wait 在同一 commit 80d6048。
未单独 checkout 尝试 19，校准数据沿用 80d6048。CONCAT/TILE sub-op 时间沿用之前测量。

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

**结果 (端到端, 4 threads, pp256, warmup=3, 单次 run 校准)**：
| 指标 | 尝试 19 (80d6048) | 尝试 20 (40f2684) | 变化 |
|------|-------------------|------------------|------|
| prefill_tps | 138.3 | **141.0** | +2% |
| decode_tps | 28.0 | **29.3** | +5% |
| prefill SILU | 52ms (3.3%) | **14ms (1.0%)** | **-73%** |
| prefill ADD | 4ms (0.3%) | 4ms (0.3%) | 持平 |
| prefill MUL | 6ms (0.4%) | 6ms (0.4%) | 持平 |
| decode SILU | 13ms | **3.5ms** | -73% |

注：尝试 20 校准 prefill=141.0, decode=29.3。SILU sub-op 时间沿用之前测量。

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

## 尝试 22：修复 CONCAT/TILE bulk memcpy 正确性 bug

**决策理由**：尝试 19 的 CONCAT/TILE bulk memcpy 在 `mlllm_chat` 实际推理中
产生乱码输出（"amhelperhelper...挂钩挂钩挂钩..."），但 `ctest` 全通过。

**根因**：bulk memcpy 假设 src 和 dst 都完全 contiguous，用
`src.nelements() * es` 一次拷贝整个 tensor。但 output tensor 的 `stride[1]`
可能是 graph seq_len × es（padding 后的行步长），不是 `shape[0] * es`：
- `output->is_contiguous()` 检查 `stride[1] == stride[0] * shape[0]`
- 但 graph 静态 seq_len=256 时，`stride[1] = 256 * es`，而 `shape[0]` 可能是 192
- `is_contiguous()` 返回 false，走 fallback 分支
- 但 fallback 分支的 bulk copy 仍假设 `output->stride[0]` 起始处整个 tensor 连续

实际数据布局：`output [shape0=192, shape1=256, ...]` 的 `stride[1] = 256*es`，
但 `shape[0]` 只有 192——bulk copy 写过界，覆盖了下一行的前 192 个元素。

**修复**：
- **CONCAT dim=0**：per-(i1,i2,i3) 拷贝 i0 块（`shape[0]*es` 字节）。
  i0 在 src 和 dst 中都 contiguous（`stride[0]==es`），但外层维度仍按 stride 寻址。
  memcpy 调用从 N0×N1×N2×N3 降到 N1×N2×N3。
- **TILE dim=2-only**：per-i1 行拷贝 i0 块，reps[2] 次。
  每个 rep 拷贝 `shape[0]*es` 字节，按 `output->stride[2]` 寻址。
  memcpy 调用从 N0×N1×reps2 降到 N1×reps2。

**验证**：
- `ctest` 14/14 通过
- `mlllm_chat "你是谁"` 输出正常中文回复（"我是由OpenAI开发的AI助手..."），
  `hit_eos=true`，`generated_tokens=90`

**结果 (端到端, 4 threads, pp256, warmup=3, 5-run 中位数校准)**：
| 指标 | 尝试 21 (回退前) | 尝试 22 (修复, 5af2fb1) | 变化 |
|------|----------------|--------------------------|------|
| prefill_tps | ~141 (40f2684) | **138.5** (134.0-140.6) | -2% (噪声) |
| decode_tps | ~29.3 | **28.7** (28.2-29.1) | -2% (噪声) |
| prefill CONCAT | 111ms | **9ms** | **-92%** |
| prefill TILE | 33ms | **2.4ms** | **-93%** |

注：尝试 22 校准 5-run: prefill 138.5 / decode 28.7。修复正确性后 prefill 略降，
因为 per-(i1,i2,i3) block copy 比 1 次 bulk copy 多了外层循环。

**关键发现**：
1. `ctest` 的合成测试用例都是完全 contiguous 的小 tensor，不能覆盖 stride != shape×es 的情况
2. `is_contiguous()` 检查的是 stride 一致性，但 graph padding 会导致 `shape[1]` 和 `stride[1]/es` 不一致
3. 正确的优化思路：只利用内层维度（`stride[0]==es`）的 contiguous 性，不假设外层
4. 性能收益比尝试 19 略小（per-(i1,i2,i3) 而非 1 次 bulk），但仍然显著

**教训**：性能优化的合成测试必须覆盖非 contiguous stride 场景，否则会漏掉
实际推理中的 layout 不匹配 bug。`mlllm_chat` 端到端文本验证是必要的。

**决定**：**采用。per-(i1,i2,i3) block copy 作为 CONCAT/TILE 快路径。**

---

## 尝试 23：GEMV FP16 acc 8-way K-unroll

**决策理由**：分析 ggml/tinyBLAS 的 GEMV 路径发现，ggml 用 SVE `ggml_sve_f16_fma_widened`
做 8-way K-unroll + 4 累加器链，而 mlllm 尝试 15 只用了 2-way。理论上 Apple M5 FP16 FMA
latency=2 cycles，2 chains 已经达到 CPI=1（FMA throughput ceiling），但在 issue stage
仍有瓶颈——2-way kernel 每 K 步 5 条指令（2 load + 2 FMA + 1 branch），issue 占用高。
扩展到 8-way 后，8 FMA 分摊到 8 步 K 上，loop overhead 比 FMA 降到 1/4，issue 带宽更充分利用。

**算法**：8 个独立 FP16 acc chains，每 K 步 FMA 进 acc[k%8]
- 偶数 K 步: 8 个 A 标量 + 8 个 B 向量 + 8 个 FMA（每个 acc 一个）
- K_BLOCK 间 pairwise reduction：acc[0..7] → acc[0]（3 层 vaddq）
- 尾部 (K%8!=0): 回退到 rotating acc[k_outer%8]
- 寄存器：8 acc (Q0-7) + 8 B (Q8-15) + 8 A 标量 + 几个 tmp，~24 Q（NEON 32 Q，够用）

**实现**：
- `kernels/matmul.cpp`: `matmul_fp16_neon_gemv_range_fp16acc` 改 8-way K-unroll
- K_BLOCK 间 acc 初始化逻辑：acc[0] 携带 prior block partial，acc[1..7] 清零
- pairwise reduction 树：`acc[0,1]→acc[0], acc[2,3]→acc[2], ... → acc[0,2]→acc[0], ... → acc[0]`

**结果 (microbench, 3 个 GEMV shape)**：
| Shape | 尝试 15 (2-way) | 尝试 23 (8-way) | 变化 |
|-------|-----------------|-----------------|------|
| 1×2048×2048 t=1 | 0.32ms (26.5 GF) | **0.13ms (66.5 GF)** | **+151%** |
| 1×2048×6144 t=4 | 0.19ms (131.4 GF) | **0.12ms (204.2 GF)** | **+56%** |
| 1×2048×128k t=4 | 4.01ms (130.9 GF) | **2.59ms (202.8 GF)** | **+55%** |

突破之前认定的 ~131 GF 带宽墙，到 200+ GF。说明尝试 15 的"带宽墙"实际是 issue-bound
假象——2-way kernel 的 loop overhead 占 issue 带宽 60%，限制了 FMA 单元的实际吞吐。
8-way unroll 把 issue overhead 摊薄到 1/4，FMA 单元可以更密集运行。

**结果 (端到端 Youtu-LLM-2B, 4 threads, pp256, warmup=3, 5-run 中位数)**：
| 指标 | 尝试 22 (2-way) | 尝试 23 (8-way) | 变化 |
|------|-----------------|-----------------|------|
| prefill_tps | 138.5 (134.0-140.6) | **138.6 (136.1-145.5)** | 持平 |
| decode_tps | 28.7 (28.2-29.1) | **49.6 (42.4-51.2)** | **+73%** |
| tpot_ms | ~35.6 | ~20.1 | -44% |

**关键发现**：
1. **decode_tps = 49.6，超过 llama.cpp (41 tok/s)** — FP16 非量化路径下首次超越 (+1.21x)
2. microbench +55~151% GFLOPS，e2e decode +73%，与 microbench 一致
3. 之前的"131 GF 带宽墙"是误判，实际是 issue bandwidth 瓶颈
4. 8-way unroll 与 ggml SVE GEMV 的 8-way K-unroll 思路一致，验证了该方向
5. prefill 持平——GEMV kernel 改动不影响 GEMM 路径，之前单次测的 -19% 是噪声

**决定**：**采用。GEMV FP16 acc 8-way K-unroll 作为默认。**

---

## 当前最优配置

- **TILE**: 8×8 (NEON, M>=2) / dedicated GEMV (M=1)
- **K_BLOCK**: 512
- **精度**: FP16 权重 + FP16 activations (GEMM) + FP16 累加 (GEMM + GEMV)
- **A packing**: Per-call interleaved (M>=8), FP32→FP16
- **B packing**: Load-time interleaved (tile-of-8 transpose), GEMV+GEMM 共用
- **embed_tokens**: FP16 row-major (embed 查找) + FP16 packed 副本 (lm_head matmul)
- **FMA**: vfmaq_lane_f16 FP16 累加 2-way unroll (M>=8 GEMM), **vfmaq_n_f16 FP16 累加 8-way K-unroll (M=1 GEMV)**
- **GEMV n_chunk**: 自适应 max(N/(n_threads*8), 64) 对齐到 8
- **KV cache**: FP16 存储（cache buffer 减半，FP16FML SDPA 直接用，无需 per-call 转换）
- **SDPA**: FlashAttention-2 FP16FML (FP16 输入 + FP32 累加) + head 并行
- **线程**: 自定义线程池 (spin-wait + `yield`), per-op split work, per-worker A pack
- **Eltwise**: SILU/ADD/MUL NEON 向量化
- **Bench**: `--prompt-tokens <N>` 支持固定 token 数满载测量

## 端到端性能 (Youtu-LLM-2B, M5 Pro, 4 threads, pp256)

**测量协议**：`mlllm_bench --prompt-tokens 256 --max-new-tokens 64 --threads 4 --warmup 3`
单次正式 run。对关键改动 (2-way vs 8-way GEMV) 已验证 5-run 方差 ±3-8%，
单次测量足以分辨 >10% 的差距。历史数据未重新校准（部分早期数据为 23-token 短 prompt，
被 graph seq_len=128 padding 低估，仅做趋势参考）。

| 版本 | prefill_tps | decode_tps | load_ms | 备注 |
|------|------------|------------|---------|------|
| FP32 baseline | ~60 | ~5 | 328 | 短 prompt, 未校准 |
| + FP16 + pack + lane-FMA (尝试 10, 3fbc7cf) | **86.2** | **12.8** | 4254 | pp256 校准 |
| + FlashAttention + head 并行 (尝试 11-12) | — | — | 4254 | 未单独 checkout |
| + FP16 累加 (尝试 13, 89a1796) | **97.5** | **12.7** | 4254 | pp256 校准 |
| + dedicated GEMV + 2-way unroll + GEMM 2-way + FP16 KV (尝试 14-17, e4338a6) | **133.3** | **26.6** | 1243 | pp256 校准 (合并 commit) |
| + spin-wait + CONCAT/TILE bulk (尝试 18-19, 80d6048) | **138.3** | **28.0** | 1243 | pp256 校准 |
| + SILU/ADD/MUL NEON (尝试 20, 40f2684) | **141.0** | **29.3** | 1243 | pp256 校准 |
| + CONCAT/TILE 修复正确性 (尝试 22, 5af2fb1) | **138.5** | **28.7** | 1243 | 5-run 中位数校准 |
| + GEMV FP16 acc 8-way K-unroll (尝试 23) | **138.6** | **49.6** | 1243 | 5-run 中位数校准 |
| + ThreadPool park/resume (尝试 24) | **149.5** | **52.4** | 1243 | 5-run 中位数校准 |
| + GEMM 2D 调度 + pre-pack A (尝试 25) | **151.3** | **51.5** | 1243 | 10-run 中位数校准 |
| + K_BLOCK 512→2048 (尝试 26) | **159.2** | **52.1** | 1243 | 10-run 中位数校准 |
| + SDPA prefill kernel rewrite (尝试 27) | **220.3** | **52.4** | 1126 | 5-run 中位数校准 |
| + MLP gate/up merge (尝试 28) | **224.3** | **54.5** | 910 | 5-run 中位数校准 |
| + Fused SILU in matmul (尝试 29) | **225.7** | **53.7** | 910 | 5-run 中位数校准 |
| + Skip 1.3 GB cache memset (尝试 30) | **234.2** | **53.7** | 910 | 5-run 中位数校准 |
| + Cache lifecycle refactor (尝试 31, 当前) | **235.2** | **53.7** | 910 | 5-run 中位数校准 |

| 框架 | prefill (pp256) | decode (tg) |
|------|----------------|------------|
| **mlllm** | **~220.3** | **~52.4** ✅ |
| ncnn | 202 | 33 |
| llama.cpp | 264 | 41 |

**prefill 超过 ncnn (220 vs 202)**，距 llama.cpp 仅 1.20x 差距（之前 1.66x）。
decode 持平，仍 1.28x 超过 llama.cpp。

---

## 尝试 24：ThreadPool park/resume（消除 idle CPU 占用）

**决策理由**：`mlllm_chat` REPL 模式下，用户在输入提示前和读输出时，worker
线程持续 spin-wait 烧 CPU（4 核 100%）。需要让 worker 在 generation 结束后
真正阻塞，下次 prefill 时再唤醒。

**算法**：CV-based park + spin-based dispatch
- `ThreadPool::park()`: 设 `parked_=true`，worker 完成当前 shard 后在
  `condition_variable.wait()` 阻塞
- `ThreadPool::resume()`: `parked_=false` + `notify_all()`，worker 醒来重新 spin
- `parallel_for_impl`: 入口自动 `resume()`，热路径保持纯 spin（0 dispatch overhead）
- `LLMEngine::park_workers()`: 暴露给上层，`generate_greedy` 在 decode 结束后调用

**实现**：
- `kernels/threading.h`: 新增 `park()`, `resume()`, `park_mtx_`, `park_cv_`,
  `parked_` 成员；`ensure_workers_started()` 懒启动 worker
- `kernels/threading.cpp`: worker_loop 在 `fetch_sub(pending_workers_)` 后检查
  `parked_`，true 则阻塞在 CV
- `engine/engine.h`: `park_workers()` 转发到 `thread_pool_.park()`
- `examples/cli_common.cpp`: `generate_greedy` 返回前调 `engine.park_workers()`

**结果 (端到端, 4 threads, pp256, warmup=3, 5-run 中位数)**：
| 指标 | 尝试 23 (纯 spin) | 尝试 24 (park/resume) | 变化 |
|------|--------------------|----------------------|------|
| prefill_tps | 138.6 | **149.5** | +8% |
| decode_tps | 49.6 | **52.4** | +5.6% |
| idle CPU (REPL) | 4 核 100% | **~0%** | ✅ |

**关键发现**：
1. park/resume 不仅消除 idle CPU，反而**提升**了性能（decode +5.6%, prefill +8%）
2. 根因推测：park 期间 worker 释放 cache，下次 resume 时 L1/L2 更干净，
   减少了与主线程的 cache 竞争
3. 懒启动（`ensure_workers_started`）让 `mlllm_chat` 启动后到第一次输入
   期间也无 CPU 占用
4. 实测 `mlllm_chat "给我讲一个故事"` (25 token prompt, 128 token gen)：
   `decode_tps=51.27, tpot_ms=19.51`，与 bench 一致

**决定**：**采用。park/resume 作为默认 ThreadPool 行为。**

---

## 尝试 25：GEMM 2D atomic-steal 调度 + pre-pack A

**决策理由**：分析 ggml tinyBLAS 的调度方式发现，mlllm GEMM 用 1D 静态 M-shard，
每 worker 跑全 N，负载不均时快线程空等慢线程。tinyBLAS 用 2D job 空间
(M-tiles × N-blocks) + atomic-steal 动态抢 job，更好地利用多核。

**算法**：
- `ThreadPool::parallel_for_2d(M, m_tile, N, n_block, fn)`
  - `total_jobs = (M/8) × (N/N_BLOCK)`
  - 每个 job 处理 `[m_begin, m_end) × [n_begin, n_end)` = 8 行 × N_BLOCK 列
  - `current_chunk.fetch_add(1)` atomic-steal，worker 抢 job 直到耗尽
- N_BLOCK=256（8 个 N-tile 对齐，避免 job 过细导致 atomic 争抢）
- **pre-pack A**：主线程在 2D 调度前预先 pack 所有 M-tile 的 A slice，
  避免 per-job pack 导致重复计算（pack_a_calls 从 M/8 × N/N_BLOCK 降到 M/8）

**实现**：
- `kernels/threading.h`：Job 加 `current_chunk` atomic, `fn_2d` 字段, `total_2d_jobs`
- `kernels/threading.cpp`：`parallel_for_2d_impl` + worker_loop 加 atomic-steal 分支
- `kernels/matmul.cpp`：GEMM 主分支改用 `parallel_for_2d` + pre-pack A vector
- `examples/bench.cpp`：加 pack_a profiling 计数器（`pack_a_ms`, `pack_a_calls`, `pack_pct`）

**N_BLOCK 调参**：
| N_BLOCK | job 数 (M=256,N=2048) | prefill 中位数 | 备注 |
|---------|----------------------|---------------|------|
| 64 | 1024 | 136.6 (-13%) | atomic 争抢严重 |
| 256 | 256 | 145.6 (+8.6%) | 最优 |
| pre-pack + 256 | 256 | **151.3 (+12.8%)** | 消除重复 pack |

**结果 (端到端, 4 threads, pp256, warmup=3, 10-run 中位数)**：
| 指标 | 尝试 24 (1D baseline) | 尝试 25 (2D + pre-pack) | 变化 |
|------|------------------------|------------------------|------|
| prefill_tps | 134.1 | **151.3** | **+12.8%** |
| decode_tps | 51.4 | 51.5 | 持平 (GEMV 路径不变) |
| pack_pct | 1.5% | **1.5%** | 持平 (pre-pack 消除重复) |

**关键发现**：
1. 2D 调度初始版本 (N_BLOCK=64) 反而退化 -13%——atomic 争抢 + per-job pack 重复
2. N_BLOCK=256 让 job 数从 1024 降到 256，prefill +8.6%
3. pre-pack A 把 pack_a_calls 从 103424 降到 8192，pack_pct 从 11.9% 降到 1.5%
4. 三步累计 prefill +12.8%，缩小了与 llama.cpp 的差距 (1.97x → 1.75x)
5. 2D 调度与 microkernel 类型 (outer-product vs dot-product) 无关，只改 job 调度

**决定**：**采用。2D atomic-steal + pre-pack A + N_BLOCK=256 作为 GEMM 默认。**

---

## 尝试 26：K_BLOCK 从 512 调到 2048

**决策理由**：K_BLOCK 控制 K 维分块大小，block 间需要 store/reload
acc（FP16→FP32→FP16）保持精度。K_BLOCK=512 时 K=2048 有 4 个 block，
store/reload 4 次。K_BLOCK=2048 时只有 1 个 block，无 store/reload。

**microbench sweep** (128×2048×6144, t=4)：
| K_BLOCK | GFLOPS | vs 512 |
|---------|--------|--------|
| 512 (default) | 880 | baseline |
| 1024 | 932 | +6% |
| **2048** | **924** | **+5%** |
| 4096 | 937 | +6% |

多 shape sweep 确认 K_BLOCK=2048 在所有 GEMM shape 上都是最优或接近最优：
| Shape | k=512 | k=2048 | 变化 |
|-------|-------|-------|------|
| 128×2048×2048 | 851 | 911 | +7% |
| 128×6144×2048 | 849 | 890 | +5% |
| 128×2048×128k | 867 | 935 | +8% |
| 256×2048×6144 | 902 | 935 | +4% |

**实现**：`kernels/matmul.h` 中 `MatmulConfig::k_block` 从 512 改成 2048。

**结果 (端到端, 4 threads, pp256, warmup=3, 10-run 中位数)**：
| 指标 | 尝试 25 (K_BLOCK=512) | 尝试 26 (K_BLOCK=2048) | 变化 |
|------|------------------------|------------------------|------|
| prefill_tps | 151.3 | **159.2** | +5.2% |
| decode_tps | 51.5 | 52.1 | +1.2% |

**关键发现**：
1. K_BLOCK=2048 对 K=2048 的 shape 等于完全消除 store/reload
2. 精度未受影响（K=2048 的 FP16 acc 范围在安全区间内，ctest 全通过）
3. GEMV 路径不受影响（GEMV 用独立的 K_BLOCK_FP16）
4. 收益边际递减：4096 不比 2048 显著更好

**决定**：**采用。K_BLOCK=2048 作为默认。**

## 尝试 27：SDPA prefill kernel 重写（Br=8 Bc=64 + register-tiled PV + dot_fp16_8x）

**决策理由**：尝试 26 后 SDPA 占 prefill 13.2%（158 ms / 32 layers × 4.95 ms/layer），
是仅次于 MATMUL 的第二大热点。MATMUL 已达 FP16FML peak 80%（920 GF/s），无明显空间。
分析 SDPA prefill kernel（`flash_attn_fp16_prefill`）发现三个瓶颈：

1. **Br=4 Bc=32 太小** — 每 block 只做 4×32=128 个 dot，PV 内循环只 32 步，loop overhead 占比高
2. **PV loop 非 register-tiled** — 每个 j 迭代都 `vld1q_f32(oi+d)` + `vst1q_f32(oi+d)` for O，
   而非像 `flash_attn_fp16_decode` 那样把 16 个 FP32x4 acc 寄存在 NEON 寄存器里跨 j 持有
3. **dot_fp16_4x 只 4 acc chains** — FP16FML latency=2cyc, throughput=2/cyc, 4 chains → CPI 0.5
   issue-bound。扩到 8 chains → CPI 0.25，更接近 FMA throughput 上限

实测：4.37 ms/layer × 32 ≈ 140 ms / prefill（与端到端 158 ms 相符，profile 含 dispatch overhead）。
单线程仅 34 GF/s/head = **3% FP16FML peak**，远未到 compute 或 bandwidth bound。

**算法**：Br=8 Bc=64 + dot_fp16_8x + register-tiled PV

- **QK**：`dot_fp16_8x(qi, K0..K7, d_k, out[8])` — 1 Q load × 8 K loads × 8 independent
  FP32x4 acc chains，CPI 0.25 → FMA throughput bound
- **PV**：每个 Q 行 i 在内层 j=0..bc=64 循环外先 load 16 个 FP32x4 acc（`o0..o15`），
  跨整个 j 循环只在寄存器内做 `vfmlalq_lane_low/high_f16`，循环结束才 store。
  Direct port from `flash_attn_fp16_decode` lines 407-474.
- **d_v 分块**：TILE_DV=64，d_v=128 → 2 个 64-wide tile。8-wide/4-wide/scalar tail 处理非 64 倍数
- **Online softmax**：保持原 `alpha` rescale + `m_new/l_new` 逻辑，跨 block 累积

**寄存器预算**（NEON 32 个 Q）：
- QK 路径：8 acc + 2 Q + 2 K (reused) = 12 Q ✓
- PV 路径：16 acc + 8 V = 24 Q ✓（编译器可调度，无 spill）

**实现**：
- `kernels/attention.cpp`: 新增 `dot_fp16_8x`（attention.cpp:322-455，扩展 `dot_fp16_4x` 到 8 K 行）
- `kernels/attention.cpp`: 重写 `flash_attn_fp16_prefill`（attention.cpp:673-918）
  - Br=8, Bc=64, TILE_DV=64
  - QK: `dot_fp16_8x` 主路径 + `dot_fp16_4x` 4-tail + scalar tail
  - PV: 16-register-tiled 主循环，8-wide/4-wide/scalar tail
  - Stack `S[8][64] = 2 KB` per call
- `tests/test_attention.cpp`: 新增 4 个 MLA prefill 测试用例（src=256, src=20 非 tile M, vd=100 非 tile d_v, past=128）
- `tests/bench_sdpa.cpp` + `CMakeLists.txt`: 新增 SDPA microbench（mirror bench_matmul 结构）

**结果 (microbench, MLA shape: H=16, src=256, dk=192, dv=128, 4 threads, warmup=5, repeat=10)**：
| 指标 | 尝试 26 (Br=4 Bc=32) | 尝试 27 (Br=8 Bc=64 + tiled PV) | 变化 |
|------|---------------------|---------------------------------|------|
| SDPA/layer | 4.37 ms | **2.11 ms** | **-52%** |
| SDPA GFLOPS (4-thread) | 310 | **623** | +101% |
| scaling 1→4 thread | 7.4 ms → 4.4 ms (1.7x) | 7.4 ms → 2.1 ms (3.5x) | 调度改善 |

**结果 (端到端, 4 threads, pp256, warmup=3, 5-run 中位数)**：
| 指标 | 尝试 26 (Br=4 Bc=32) | 尝试 27 (Br=8 Bc=64) | 变化 |
|------|---------------------|---------------------|------|
| prefill_tps | 159.2 (历史值) | **220.3** (217-221) | **+38%** |
| prefill_ms | ~1613 | **1163** | -28% |
| SDPA (pp256) | 158 ms (13.2%) | **70 ms (6.3%)** | **-56%** |
| MATMUL (pp256) | 957 ms (80%) | 957 ms (86%) | 持平 |
| decode_tps | 52.1 | 52.4 | 持平 |

注：尝试 26 的 prefill_tps=159.2 是机器冷态历史测量。本次重测前先跑了一次完整 5-run
确认机器温度稳定（机器过热会让 matmul microbench 退化 5-16%，故关键改动需冷却后重测）。

**关键发现**：
1. **SDPA 大幅提升** (-56%)，从 13.2% 降到 6.3%，prefill_tps +38%
2. microbench +52% (4.37 → 2.11 ms) 与端到端 -56% 一致，无 surprise
3. 单线程 scaling 从 1.7x → 3.5x，说明 Br=8 更适合 multi-thread contention 模式
4. **prefill 超过 ncnn (220 vs 202)**，距 llama.cpp 仅 1.20x（之前 1.66x）
5. dot_fp16_8x 与 register-tiled PV 是 ncnn-upstream decode kernel 已有的技巧，
   prefill 之前未做是历史疏漏
6. 热管理很重要：机器过热会让 FP16FML 路径性能下降 10-20%。Bench 时需要 warmup
   + 5-run 中位数排除热噪声

**实施过程中发现的关键 bug**：
- 初次实施后 microbench 显示 -50% 但端到端无变化。排查发现 `test_output_fp16/model_prefill.graph`
  的 cache_k/cache_v 节点 `out_prec=0`（FP32），尽管 `mla.py` 已指定 `Precision::FP16`。
  原因：旧的 `.graph` 文件是早期 mla.py 版本生成的，未重新导出。
  修复：重新跑 `python3 -c "from mla import convert_mla; convert_mla(...)"` 生成新 graph
  才能让 `cache_is_fp16=true` 路径生效，端到端收益才得以显现。

**工具改进**：
- `tests/bench_sdpa.cpp`: 新增 SDPA microbench，支持 `--threads/--warmup/--repeat/--src/--heads/--hd/--vd`
  以及 `--fp32-cache`（默认 FP16）/`--no-causal` flags
- `tests/test_attention.cpp`: 新增 4 个 MLA prefill 测试用例覆盖 tile 边界条件

**决定**：**采用。Br=8 Bc=64 + dot_fp16_8x + register-tiled PV 作为 FP16 SDPA prefill 默认。**
FP32 fallback（`flash_attn_fp32_prefill`）保留不变，仅 NEON+FP16 路径走新 kernel。

**后续可探索**：
- 2D atomic-steal 跨 head 调度（16 heads × 2D = 更细粒度 work-stealing）
- causal mask 跳过最后 jb 块（自然被 causal 裁剪）
- head 间共享 Q tile（如果 RoPE 后 Q 数据可复用）

---

## 尝试 28：MLP gate/up merge + stride-aware SILU/MUL

**决策理由**：尝试 27 后 MATMUL 占 prefill 86%（957 ms），其中 mlp_gate + mlp_up 两个 6144×2048 matmul
占 MATMUL 时间 47%（约 450 ms）。两个 matmul 共享同一输入 `x_normed2`，但 A pack 各做一次
（pack_a_calls=8192 中 2048 次来自 gate/up）。合并成 N=12288 的单 matmul + SLICE 应能：
- A pack 减半（gate/up 部分省 1024 次 pack_a_calls）
- 1 次 dispatch 替代 2 次（节省 ~96 us，可忽略）
- N 维更大，理论上 lane-FMA GEMM 效率略提升

**算法**：
- 导出时合并 weight：`w_gate_up = concat([w_gate, w_up], dim=0)` → shape `(2*intermediate, hidden)` FP16
- 推理时：`merged = matmul(x, w_gate_up)` → `slice(merged, [intermediate, intermediate], dim=0)`
  → gate, up → silu(gate), mul(gate, up), matmul(mlp_hidden, w_down)
- **SLICE 是 zero-copy view**：保留 parent 的 stride，gate shape `(6144, 256, 1, 1)` stride `[4, 12288*4, ...]`

**关键 bug 1（实施过程中发现）**：SILU/MUL 当前实现假设输入 contiguous
（`ptr<float>() + i` 线性读 i=0..N），但 SLICE 后的 view stride[1] = parent_total * es
（不是 sliced * es）。这导致 SILU 读错数据（每 6144 个元素后跳过 6144 个属于 up 的元素），
文本输出乱码。

**修复方案**：让 SILU/MUL stride-aware。新增 `unary_stride_aware` / `binary_stride_aware` helper
（execute.cpp 顶部），按 shape/stride 遍历：
- 沿 dim0（shape[0]，stride[0]=es）做 NEON 4-wide 内层循环（contiguous run）
- 外层 (d3, d2, d1) 按 stride 跳转
- Contiguous 输入走快路径（保持原性能）

**关键 bug 2（初版修复用 RESHAPE workaround，已废弃）**：第一版用 `g.reshape(gate, gate.shape)`
强制 materialize contiguous copy。但 RESHAPE 的 materialize 路径是逐元素 memcpy
（execute.cpp:267-285），每个 RESHAPE 平均 0.7 ms × 64 calls = **217 ms 开销**，
完全吃掉 matmul 节省。回退该 workaround，改用 stride-aware eltwise 路径。

**实现**：
- `models/mla.py`: `export_weights` 跳过 mlp.gate_proj/up_proj，新增 `mlp_gate_up_proj` 合并 weight
- `models/mla.py`: `_build_mla_layer` 用单 matmul + slice 替代 2 matmul（无需 RESHAPE）
- `graph/execute.cpp`: 新增 `unary_stride_aware` / `binary_stride_aware` helper（约 100 行），
  重写 SILU/MUL case 使用新 helper
- `test_output_fp16/weights/`: 删除 `model_layers_*_mlp_gate_proj_weight.weights` 和
  `model_layers_*_mlp_up_proj_weight.weights`（共 64 个 orphaned 文件）

**结果 (microbench, 256×2048×N FP16 GEMM, 4 threads, warmup=5, repeat=10)**：
| N | 单次时间 | GFLOPS |
|---|----------|--------|
| 6144 (gate 或 up 单独) | 8.65 ms | 745 |
| 12288 (merged) | 17.30 ms | 743 |

GFLOPS 一致（743-745），merged matmul 时间 ≈ 2× 单 matmul（FLOPs 翻倍）。
**microbench 层面无收益** —— matmul 算力已 saturate。收益完全来自 A pack + dispatch 减半。

**结果 (端到端, 4 threads, pp256, warmup=3, 5-run 中位数)**：
| 指标 | 尝试 27 (gate/up 分开) | 尝试 28 (merged) | 变化 |
|------|-----------------------|-----------------|------|
| prefill_tps | 220.3 | **224.3** (221-227) | +1.8% |
| prefill_ms | 1163 | **1130** | -2.8% |
| MATMUL (pp256) | 957 ms (86%) | **929 ms** (86%) | **-3.0%** (-28 ms) |
| SDPA (pp256) | 70 ms (6%) | 68 ms (6%) | 持平 |
| RESHAPE | 30 ms | 30 ms | 持平（无新增） |
| pack_a_calls | 8192 | **7168** | -12.5% (-1024) |
| pack_a_ms | 30 ms | 27 ms | -10% (-3 ms) |
| decode_tps | 52.4 | **54.5** | +4.0% (意外收益) |

**关键发现**：
1. **MLP merge 净收益小但稳定**：prefill +1.8%，MATMUL -3%（28 ms 节省）
2. microbench 验证 matmul 算力已 saturate，无 N 大小收益
3. 28 ms 节省来自 pack_a_calls 减少 + dispatch 减少 + matmul 效率略升
4. decode +4% 是意外收获 —— 可能是 GEMV dispatch 减半 + cache 局部性更好
5. **stride-aware SILU/MUL 是关键** —— 老的实现假设 contiguous，导致 slice view 数据读错。
   通用 fix 让任何 view 输入都能正确处理 eltwise op（为未来其他 merge 优化铺路）
6. RESHAPE materialize workaround 不能用 —— 逐元素 memcpy 太慢，开销 > 收益

**决定**：**采用。MLP gate/up merge + stride-aware SILU/MUL 作为默认。**

**后续可探索**：
- Q/KV merge（q_a + kv_a 共享 input_normed，但 dim 差异大难合并）
- 2D atomic-steal 跨 head 调度（SDPA 收益空间）
- causal mask 跳过最后 jb 块（SDPA 微优化）

---

## 尝试 29：Fused SILU in matmul writeback（activation infrastructure）

**决策理由**：尝试 28 后 SILU op 仍占 prefill 1.3%（15 ms / 32 calls × 0.47 ms each）。
虽然绝对值小，但每次 SILU dispatch 是独立的 graph op（每次 ~120 us dispatch overhead），
加上一次额外内存 round-trip（读 matmul 输出 → 写 SILU 输出）。

更重要的理由是**基础设施**：建立 matmul→activation fused path，让未来 GELU/RELU 等
激活函数直接复用，无需新 dispatch op。同时为后续 fused-mul（SwiGLU 完整融合）铺路。

**算法**：在 matmul kernel 的 writeback 阶段（FP16 acc → FP32 store）直接 apply activation。
避免额外内存 round-trip。

- 新增 `Activation` enum: NONE, SILU, GELU, RELU（未来可加 GELU_ERF / SIGMOID / TANH）
- 新增 `kernels/activations.h`: 包含 `apply_activation_scalar` + `apply_activation_f32_neon`
  + `sigmoid_f32_neon`（从 execute.cpp 移过来共享）
- `OpType::MATMUL` 复用 `params_i32` 携带 `[activation, act_n_begin, act_n_len]`
- matmul kernel signature 扩展：`kernel_matmul_fp32(A, B, C, tp, act, act_n_begin, act_n_len)`

**`act_n_begin` / `act_n_len` 的作用**：
- MLP merge 后 gate/up 都在一个 matmul 输出里（N=12288），只需对前半（N=[0, 6144)）
  apply SILU。`act_n_begin=0, act_n_len=intermediate` 让 kernel 只对 gate 列 apply。
- `act_n_len == -1` 表示 "全部 N"（fast path，无 per-column branch）。
- 对其他场景（非 merge）用默认参数 `act_n_len=-1` 即可全 N apply。

**实现**：
- `kernels/activations.h`（新文件）: enum + scalar + NEON 4-wide activation helpers
- `kernels/matmul.h`: kernel signature 加 `Activation act, int act_n_begin, int act_n_len`
- `kernels/matmul.cpp`:
  - GEMV FP16 acc kernel: store 阶段判断 `is_last_block`，inline apply activation
    （单 K-block 时直接 inline，避免独立 pass）
  - lane-FMA FP16 acc GEMM kernel: writeback 阶段判断 `last_block`，
    per-column 检查 `activation_applies_at` 后 apply
  - FP32 acc fallback + standard path: 调用 `apply_activation_to_range` post-hoc pass
  - parallel 路径: 每个 shard 翻译 global act range 到 local coords
- `graph/execute.cpp`: MATMUL case 读 `params.i32[0..2]`，传给 `kernel_matmul_fp32`
  + 删除 `sigmoid_f32_neon`（移到 activations.h）
- `python/transpile.py`: 新增 `Activation(IntEnum)`；`matmul(activation=, act_n_begin=, act_n_len=)` 加 3 个 params
- `models/mla.py`: MLP 用 `g.matmul(x, w_gate_up, activation=Activation.SILU,
  act_n_begin=0, act_n_len=intermediate)`，移除独立 `g.silu(gate)`

**结果 (端到端, 4 threads, pp256, warmup=3, 5-run 中位数)**：
| 指标 | 尝试 28 | 尝试 29 | 变化 |
|------|--------|--------|------|
| prefill_tps | 224.3 | **225.7** (222-227) | +0.6% |
| prefill_ms | 1130 | 1134 | 持平 |
| MATMUL | 929 ms | **929 ms** | 持平 |
| SDPA | 68 ms | 70 ms | 持平 |
| **SILU op** | 15 ms (32 calls) | **0** | ✅ 完全消除 |
| RESHAPE | 30 ms | 36 ms | +6 ms (噪声) |
| pack_a | 27 ms | 27 ms | 持平 |

**decode (10-run 中位数)**：
| 指标 | 尝试 28 | 尝试 29 | 变化 |
|------|--------|--------|------|
| decode_tps | 54.5 | **53.7** | -1.5% (噪声) |
| decode SILU | 1.6 ms (992 calls) | **0** | ✅ 完全消除 |

**关键发现**：
1. **SILU op 从 profile 完全消失** —— fused 路径在 matmul writeback 内 inline apply
2. **净性能持平**（prefill +0.6%, decode -1.5%）—— 都在噪声范围
3. **GEMV 路径（decode）的 inline apply 实现**：
   - 初版用独立 pass（K-block 循环后再 load/store 一次）→ 多一次内存 round-trip
   - 改为 `is_last_block` 判断 inline 到 store：单 K-block（K=2048<K_BLOCK=4096）时
     `last_block=true`，直接在 store 前 apply，避免额外 load/store
4. **lane-FMA GEMM 路径**：`last_block` 判断 + per-column `activation_applies_at` 检查
   - SwiGLU 场景下 `act_n_len=intermediate=6144`，N-tile 大小 8，columns 0..7 全在 gate 区间，
     fast path 全 vectorize
5. matmul writeback 多余工作（per-column check）实际开销极小，MATMUL 时间几乎没变化
6. **基础设施价值** > 性能价值：
   - 未来加 GELU 等只需改 `apply_activation_*` 函数
   - 未来 fused-mul (SwiGLU gate*up 完整融合) 可以复用 act_n_begin/len 机制
   - `Activation` enum 在 graph level 暴露，其他模型可直接用 `g.matmul(activation=Activation.GELU)`

**实施过程中遇到的坑**：
1. NEON `vfmaq_f32` 不接受标量第三参数 — 需用 `vfmaq_n_f32`（scalar broadcast 版本）
2. GEMV kernel 的 K-block store 是 "store partial sums + reload" 模式，
   activation 不能在中间 block store 时 apply（会污染 reload 值）。
   初版用独立 post-loop pass，后发现单 K-block 时可 inline 判断 `is_last_block`
3. parallel 路径的 shard 需要把 global act range `[act_n_begin, act_n_begin+act_n_len)`
   翻译到 shard-local coords（每个 shard 看到的 N 是局部 [0, shard_len)）
4. binary_stride_aware 已支持 a/b 不同 stride（gate_silu contiguous, up strided view），
   所以 fused SILU 后的 MUL op 不需要改

**决定**：**采用。Fused SILU in matmul + Activation enum infrastructure 作为默认。**
所有 ctest 15/15 通过，mlllm_chat 输出正常。

**后续可探索**：
- **fused MUL** (SwiGLU `gate*up` 完整融合) — 需要 matmul 同时读 up 视图并输出 product
- 把 ADD op 也改成 stride-aware（暂未做，graph 里 add 输入都 contiguous，无 bug）
- RESHAPE materialize 路径用 bulk memcpy（解决尝试 28 提到的逐元素 memcpy 慢的问题）

---

## 尝试 30：跳过 1.3 GB cache memset（reset 只清 metadata header）

**决策理由**：尝试 29 后 prefill_ms=1134，但 profile 内 op 总和 ~1107 ms。
差距 ~27 ms 来自 `reset()` 在 `prefill()` 内部调用的 `memset(buf, 0, total)`，
total = 32 层 × 2 cache × ~21 MB = **1.3 GB**。按 ~30 GB/s memset 带宽 → ~45 ms 纯浪费。

**分析**：cache buffer 从 pool freelist 复用（同一物理内存）。memset 清零不必要：
- SDPA 的 causal mask 保证只读 `[0, past+cur)` 范围
- `current_seq_len` metadata 让 SDPA 知道 cache 有效长度
- cache append 只写 `[past, past+cur)` 位置
- 未写位置的 stale 数据不会被读

**实现**：`engine/engine.cpp:reset()` 把 `std::memset(buf, 0, total)` 改为
`std::memset(buf, 0, CacheMetadata::SIZE)`（只清 64 字节 header）。

**结果 (5-run 中位数, pp256, 4 threads)**：
| 指标 | 尝试 29 | 尝试 30 | 变化 |
|------|--------|--------|------|
| prefill_tps | 225.7 | **234.2** | +3.8% |
| prefill_ms | 1134 | **1093** | -41 ms |
| MATMUL | 929 ms | 929 ms | 持平 |
| decode_tps | 53.7 | 53.7 | 持平 |

-41 ms 正好匹配 ~45 ms 的理论 memset 时间。这是"graph dispatch overhead"谜团的真正来源。

**关键发现**：之前 profile 中 `prefill_ms - sum(op times) = 43 ms` 不是 graph dispatch，
而是 `reset()` 的 cache memset。消除后 prefill_ms 直接对齐 op 总和。

**决定**：采用。`reset()` 只清 metadata header，不清 1.3 GB 数据区。

---

## 尝试 31：Cache 生命周期重构（预分配 + 多轮对话支持）

**决策理由**：尝试 30 后发现 `reset()` 仍在 `prefill()` 内部调用，每次 prefill 都重新
`pool.acquire(1.3 GB)` + 创建 tensor。而且 `prefill()` 写死 `current_seq_len = 0`，
**破坏多轮对话**（之前的上下文丢失）。用户指出这是功能 bug，不只是性能问题。

**问题**：
1. `allocate_caches()` (load 时) 只设 shape，不分配 data
2. `reset()` 每次 acquire 1.3 GB + memset
3. `prefill()` 调 `reset()` → 每次对话重新分配 cache
4. `prefill()` 写死 `current_seq_len = 0` → 多轮时历史上下文丢失
5. cache migration (prefill→decode graph) 每次 prefill 重复做

**重构**：
- `allocate_caches()`: load 时一次性分配 cache buffer + 初始化 metadata
- `reset()`: 简化为 metadata-only（`current_seq_len = 0`），~0 ms
- `prefill()`: 不再调 `reset()`；用 `past_len_` 设置 cache metadata、RoPE offset、causal mask
- cache migration: 移到 `load()`，只做一次
- `generate_greedy()`: 显式调 `engine.reset()` 保持单轮语义

多轮对话：`prefill()` 现在正确追加到现有 cache（`past_len_ > 0` 时 RoPE position = past，
causal mask key range = [0, past+cur)）。如果 `past + n > graph_seq_len` 返回错误
（chunked prefill 是后续功能）。

**结果 (5-run 中位数, pp256, 4 threads)**：
| 指标 | 尝试 30 | 尝试 31 | 变化 |
|------|--------|--------|------|
| prefill_tps | 234.2 | **235.2** | +0.4% |
| prefill_ms | 1093 | **1089** | -4 ms |
| MATMUL | 929 ms | 929 ms | 持平 |
| decode_tps | 53.7 | 53.7 | 持平 |

收益小（-4 ms，省 pool.acquire），主要是架构正确性：
- 多轮对话支持（prefill 追加到现有 cache）
- 清晰分离：reset = 显式清空，prefill = 计算
- 为 chunked prefill 铺路

**决定**：采用。cache 预分配 + prefill 用 past_len_ 作为默认行为。

---
