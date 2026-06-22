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

