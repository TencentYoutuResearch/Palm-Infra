# Qwen3.5-0.8B 适配记录

## 已修复的问题

1. **Tokenizer 不匹配**: test_e2e.cpp 硬编码路径改为 Qwen3.5 tokenizer
2. **QK norm weight 预处理**: q_norm/k_norm 需要 `1.0 + w`，export 脚本已处理
3. **OWNED tensor use-after-free**: engine.cpp 改为 pool.acquire + memcpy
4. **重复 dispatch_kernel**: execute.cpp 删除重复行
5. **Fused GDN op**: linear_attention 路径的 ~25 个细粒度 op 融合为单个 GATED_DELTANET_PREFILL/DECODE
6. **QK norm 列排列**: 通过 permute+contiguous 正确处理 [HD,NH*seq] ↔ [HD,seq,NH] 的布局转换
7. **Gate chunk 布局**: 匹配 HF 的 view+chunk 操作（contiguous → reshape [HD*2,NH,seq] → slice → reshape）
8. **SDPA 输出 reshape**: 通过 permute [HD,seq,NH] → [HD,NH,seq] → contiguous → reshape 确保正确的 (s,h,d) 排列

## 当前状态

- **Prefill**: 24 层全部对齐 HF transformers（L0-L22 cos≈1.0）
- **Decode**: 文本输出正常，greedy 和 sampling 模式均可用
- **权重**: test_output_qwen35_s4/weights/（FP16 投影权重 + FP32 norm 权重，含 1.0+w 预处理）

## 常用命令

```bash
# 构建
cd mollm/build && make -j$(sysctl -n hw.ncpu)

# 运行测试
cd mollm/build && ctest

# 生成 HF reference
cd mollm && KMP_DUPLICATE_LIB_OK=TRUE python3 tests/dump_qwen35_full_np.py

# 逐层对比（需要先生成 C++ dumps 和 HF reference）
cd mollm && python3 tests/compare_layers.py

# 重新导出权重 + graph
cd mollm && PYTHONPATH=. python3 models/qwen35.py \
    /path/to/Qwen3.5-0.8B test_output_qwen35_s4 24 256
```
