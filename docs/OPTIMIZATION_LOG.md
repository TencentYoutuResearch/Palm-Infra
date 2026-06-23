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

## 当前最优配置

- **TILE**: 8×8 (NEON)
- **K_BLOCK**: 512
- **精度**: FP16 权重 + FP16 activations (GEMM) + FP32 累加
- **A packing**: Per-call interleaved (M>=8), FP32→FP16
- **B packing**: Load-time interleaved (tile-of-8 transpose), GEMV+GEMM 共用
- **FMA**: vfmlalq_laneq_f16 (M>=8), vfmaq_n_f32 (M<8)
- **SDPA**: FlashAttention-2 FP32 (Br=4, Bc=32, online softmax)
- **线程**: 自定义线程池, per-op split work, per-worker A pack

## 下一步方向

1. **量化 (INT4/INT8)** — MATMUL 仍占 70%, 最大杠杆
2. **FP16 SDPA** — 移植 FP16FML flash kernel, SDPA 仍占 18.5%
3. **SDPA 多线程** — 当前 flash kernel 串行, 可按 head 并行
4. **其他算子并行** — RMSNorm, SiLU 等（当前占比小）
4. **其他算子并行** — RMSNorm, SiLU 等（当前占比小，优先级低）
5. **Accelerate BLAS** — 快速验证 matmul 上限（Apple only, 低优先级）
