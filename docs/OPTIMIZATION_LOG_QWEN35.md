# Qwen3.5-0.8B 优化日志

## 优化前基线 (2026-06-28)

**环境**: Apple M-series (ARM64), 4 threads, Qwen3.5-0.8B, prefill seq_len=256

**端到端** (prompt="Hello, world!" 16 tokens, max_new_tokens=64):

| 指标 | 值 |
|------|-----|
| prefill_tps | 34.56 |
| decode_tps | 73.26 |
| tpot (ms/token) | 13.65 |
| prefill_ms | 462.90 |
| decode_ms | 122.85 |

**Prefill profile** (16 tokens):

| Op | Time (ms) | % |
|---|---|---|
| MATMUL | 285.53 | 62.3 |
| GATED_DELTANET_PREFILL | 53.63 | 11.7 |
| CONTIGUOUS | 48.64 | 10.6 |
| SHORTCONV | 27.04 | 5.9 |
| RESHAPE | 12.12 | 2.6 |
| SDPA | 9.65 | 2.1 |
| 其他 | 21.4 | 4.7 |

**Decode profile** (9 tokens):

| Op | Time (ms) | % |
|---|---|---|
| MATMUL | 41.77 | 48.2 |
| **GATED_DELTANET_DECODE** | **32.74** | **37.8** |
| SHORTCONV | 8.34 | 9.6 |
| CONTIGUOUS | 1.89 | 2.2 |
| 其他 | 2.0 | 2.3 |

**分析**: decode 阶段 GDN 占 37.8%（0.20ms/call × 162 calls），是仅次于 MATMUL 的最大热点。
GDN 当前为纯 scalar 实现，NEON SIMD 化空间巨大。

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

### 结果

| 指标 | 优化前 (scalar) | 优化后 (NEON) | 提升 |
|------|----------------|---------------|------|
| GDN decode 总耗时 | 32.74ms | 13.41ms | **2.44x** |
| GDN decode % | 37.8% | 19.2% | -18.6pp |
| decode_tps | 73.26 | **82.85** | +13% |
| tpot (ms/token) | 13.65 | 12.07 | -12% |

**Decode profile 变化**:

| Op | 优化前 | 优化后 |
|---|---|---|
| MATMUL | 41.77ms (48.2%) | 44.66ms (63.8%) |
| GDN decode | 32.74ms (37.8%) | **13.41ms (19.2%)** |
| SHORTCONV | 8.34ms (9.6%) | 8.11ms (11.6%) |
| 其他 | 3.9ms | 3.6ms |

### 验证

- test_gdn: 3/3 PASS (prefill/decode/multi-step, scalar reference 对比, tol=1e-5)
- test_qwen35_ref: PASS (真实权重 numpy reference 对比)
- test_e2e: PASS (PPL=8.50 vs HF ref 8.50, argmax 逐 token 匹配)

### 下一步

- SHORTCONV decode 仍占 11.6%，可向量化
- GDN prefill 仍为 scalar (11.1% prefill)，可考虑 NEON 化或 fused 优化
- CONTIGUOUS prefill (9.6%) 可通过减少 layout 转换消除
