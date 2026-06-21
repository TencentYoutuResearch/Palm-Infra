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

