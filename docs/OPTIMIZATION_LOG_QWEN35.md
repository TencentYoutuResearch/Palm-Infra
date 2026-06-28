# Qwen3.5-0.8B 优化日志

## 框架对比基线 (pp256 + tg64, 4 threads)

| 框架 | pp256 t/s | tg64 t/s | 备注 |
|------|-----------|----------|------|
| **llama.cpp** | **749.44** | **100.44** | BLAS backend, gguf F16, build 5c7c22c3e |
| **mlllm (本框架)** | 207.50 | 89.74 | ARM NEON, 自研 kernel, scalar GDN prefill |

mlllm decode 接近 llama.cpp (89.74 vs 100.44, 差 11%)，prefill 差距较大 (207 vs 749, 3.6x)，
主要因为 GDN prefill 仍为 scalar 实现（占 prefill 65.6%）。

---

## 优化前基线 (2026-06-28)

**环境**: Apple M-series (ARM64), 4 threads, Qwen3.5-0.8B

**端到端 (pp256 + tg64)**:

| 指标 | 值 |
|------|-----|
| prefill_tps | 207.50 |
| decode_tps | 89.74 |
| ttft (ms) | 1233.73 |
| tpot (ms/token) | 11.14 |

**Prefill profile (256 tokens)**:

| Op | Time (ms) | % |
|---|---|---|
| **GATED_DELTANET_PREFILL** | **805.48** | **65.6** |
| MATMUL | 272.24 | 22.2 |
| SHORTCONV | 72.87 | 5.9 |
| CONTIGUOUS | 43.42 | 3.5 |
| SDPA | 9.02 | 0.7 |
| 其他 | 24.7 | 2.0 |

**Decode profile (63 tokens)**:

| Op | Time (ms) | % |
|---|---|---|
| MATMUL | 287.62 | 64.8 |
| **GATED_DELTANET_DECODE** | **84.99** | **19.2** |
| SHORTCONV | 45.57 | 10.3 |
| CONTIGUOUS | 10.72 | 2.4 |
| SDPA | 6.53 | 1.5 |
| 其他 | 8.3 | 1.9 |

**分析**:
- Prefill: GDN scalar 占 65.6%（256 tokens × 18 layers，每层做 256 次 recurrence）
- Decode: MATMUL 64.8% + GDN 19.2%（已 NEON 优化）+ SHORTCONV 10.3%
- 与 llama.cpp 差距主要在 prefill GDN，NEON 化后可大幅缩小

---

## 尝试 1: GDN decode NEON 向量化 (2026-06-28)

### 决策理由

GDN decode (seq_len=1) 的核心操作：
- L2 norm q/k (128-dim) — 可向量化 reduce
- state *= exp(g_t) (128×128=16384 元素) — 纯向量乘法
- state @ k / state @ q (matvec 128×128) — 内层循环可向量化 FMA
- delta = (v - kv_mem) * beta (128-dim) — 纯向量运算
- state += outer(k, delta) (rank-1 update 128×128) — 内层循环可向量化 FMA
- RMSNormGated (128-dim) — 可向量化

所有维度 (k_dim=128, v_dim=128) 均为 4 的倍数，适合 NEON float32x4_t。

### 实现

新增 `kernels/gdn_decode.cpp`，写独立于 prefill scalar 路径的 NEON decode kernel：
- `l2norm_neon()` — 向量化 L2 reduce + rsqrt + broadcast
- `matvec_128x128_neon()` — 向量化 matvec（内层 float32x4_t FMA）
- `rank1_update_128x128_neon()` — 向量化 rank-1 update
- `gdn_decode_head_neon()` — 单 head 完整 recurrence
- `kernel_gdn_decode_neon()` — 16 head 循环 + 参数解析

`kernel_gdn_decode()` 在 `#if HAS_NEON` 下调用新实现，prefill 路径保持不变。

### 结果 (pp256 + tg64)

| 指标 | 优化前 (scalar) | 优化后 (NEON) | 提升 |
|------|----------------|---------------|------|
| GDN decode 总耗时 | 197.56ms | 84.99ms | **2.32x** |
| GDN decode % | 35.0% | 19.2% | -15.8pp |
| decode_tps | 76.12 | **89.74** | +18% |
| tpot (ms/token) | 13.14 | 11.14 | -15% |

**Decode profile 变化**:

| Op | 优化前 | 优化后 |
|---|---|---|
| MATMUL | 295.73ms (52.4%) | 287.62ms (64.8%) |
| GDN decode | 197.56ms (35.0%) | **84.99ms (19.2%)** |
| SHORTCONV | 45.12ms (8.0%) | 45.57ms (10.3%) |
| 其他 | 25.9ms | 25.7ms |

### 验证

- test_gdn: 3/3 PASS (prefill/decode/multi-step, scalar reference 对比, tol=1e-5)
- test_qwen35_ref: PASS (真实权重 numpy reference 对比)
- test_e2e: PASS (PPL=8.50 vs HF ref 8.50, argmax 逐 token 匹配)

### 下一步

- **GDN prefill NEON** — 当前占 65.6% prefill，是最大热点和与 llama.cpp 差距的主因
- SHORTCONV decode 占 10.3%，可向量化
- CONTIGUOUS prefill (3.5%) 可通过减少 layout 转换消除
