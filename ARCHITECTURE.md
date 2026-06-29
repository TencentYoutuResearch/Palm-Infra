# mollm — ARM-first LLM inference engine

> **项目名占位符**: `mollm` — 所有文档和代码中的项目名均使用此占位符，待正式命名后全局替换。

## 1. 定位

- **全新项目**，不 fork 任何现有框架
- **ARM CPU 优先**（Apple Silicon + Qualcomm Oryon），GPU/Vulkan 作为 roadmap
- **只做 LLM 推理**，不兼顾 CNN/传统 CV
- **优先支持**：Youtu-LLM (MLA) 和 Qwen3.5 (Gated Attention + Gated DeltaNet)
- **第一阶段 FP16 only**，量化放 roadmap

## 2. 核心架构决策

### 2.1 图构建在 Python 端，C++ 端是纯执行器

- 借鉴 cactus：Python 转译 PyTorch 模型 → 序列化 `.mollm_graph` 文件
- C++ 端：顺序遍历节点列表，每个节点调用对应的 kernel
- 不做运行时 JIT、不做运行时 op fusion（所有融合在 Python 端完成）
- 优势：C++ 代码极简（cactus 的 execute.cpp 只有 ~500 行）

### 2.2 BufferPool 内存管理

- 分桶空闲链表（power-of-2 对齐，最小 1KB）
- Liveness 分析：计算每个节点输出的最后使用位置，用完即释放
- 三种所有权：Owned / Pooled / External (mmap)
- 不做图级 liveness bin-packing（ggml gallocr 的做法）—— 复杂度高但收益有限

### 2.3 mmap 权重

- 权重单独存 `.mollm_weights` 文件
- 运行时 mmap，OS 按需分页
- 支持 `MADV_WILLNEED` / `MADV_DONTNEED` 控制
- 量化时：scales/codebook 也在 mmap 文件中

### 2.4 图格式

```
[Header: magic, version, node_count, input_ids, output_ids]
[Node 0: op_type, input_count, input_indices[], shape, precision, params]
[Node 1: ...]
```

- 二进制格式 + 可选 JSON 调试转储
- 权重引用路径嵌入节点（指向外部 .weights 文件 + offset）

### 2.5 Tensor 表示

```cpp
struct Tensor {
    Precision prec;           // FP16/FP32/INT8
    int ndim;
    int64_t shape[4];         // 固定 4 维
    size_t stride[4];         // bytes，支持 view/permute
    void* data;               // 可能指向 pool/external/owned
};
```

- 不把 tensor 当 graph node（跟 ggml 不同）—— 图结构独立管理
- 视图系统：设置 stride 即可 slice/permute，零拷贝

### 2.6 KV Cache 设计

```
[CacheMetadata: 64B] [fp16/int8 data: capacity * kv_heads * head_dim * elemsize]
```

- 内置 INT8 per-group 量化（group_size=32，借鉴 cactus）
- View ABI：`cache.h = valid_seqlen`，`cache.capacity = cstep / head_dim`（借鉴 persistent-kvcache）
- 滑动窗口：支持 sink tokens + circular buffer（借鉴 cactus）
- 每层独立 cache：K 和 V 各自独立 buffer
- 延迟增长：初始小容量，需要时翻倍

### 2.7 多线程

- 图执行单线程，kernel 内部 `parallel_for`
- 自适应线程数：小任务单线程，大任务多线程
- ARM CPU affinity：绑定大核

## 3. 算子体系

### Phase 1 核心算子

| 类别 | 算子 | 说明 |
|------|------|------|
| 线性 | Matmul, MatmulTransposed | FP16 GEMM/GEMV，NEON SIMD |
| 归一化 | RMSNorm, LayerNorm | FP16 |
| 激活 | SiLU, GELU | FP16 |
| 位置编码 | RotaryEmbed | 支持 interleave/neox 模式 |
| 注意力 | SDPA, SDPA_MLA | kv_cache=2 in-place |
| 内存 | Reshape, Transpose, Concat, Slice | 零拷贝视图 |
| 量化 | QuantizeKV, DequantizeKV | KV cache INT8 量化 |

### Roadmap 算子

- SDPA Flash Attention (online softmax)
- GatedDeltaNet (decode + chunked prefill)
- MoE dispatch/combine
- Gated Attention (Q/K Norm + Gate)
- Conv1D (causal depthwise)

## 4. 项目结构

```
mollm/
├── kernels/              # ARM NEON SIMD 内核
│   ├── matmul.cpp        # FP16 GEMM/GEMV
│   ├── attention.cpp     # SDPA + MLA + flash attention
│   ├── deltanet.cpp      # Gated DeltaNet
│   ├── norm.cpp          # RMSNorm, LayerNorm
│   ├── rope.cpp          # Rotary embedding
│   ├── kv_cache.cpp      # KV cache quantization
│   ├── threading.cpp     # ThreadPool + parallel_for
│   └── simd.h            # NEON intrinsics 封装
├── graph/                # 图执行引擎
│   ├── graph.h/cpp       # 图结构定义 + 加载
│   ├── execute.cpp       # 节点调度 + BufferPool
│   ├── tensor.h/cpp      # Tensor 定义
│   ├── buffer_pool.h/cpp
│   ├── mmap_file.h/cpp
│   └── io.cpp            # 图文件序列化
├── engine/               # 高层 LLM 推理
│   ├── engine.h/cpp      # 模型加载 + prefill/decode
│   ├── tokenizer.cpp     # BPE tokenizer
│   └── sampler.cpp       # Top-K/Top-P 采样
├── python/               # 模型转译
│   ├── transpile.py      # PyTorch → graph 转换
│   ├── graph_ir.py       # 中间表示
│   ├── ops.py            # 算子映射
│   └── quantize.py       # 权重量化（roadmap）
├── models/               # 模型特定转换
│   ├── mla.py            # Youtu-LLM MLA 架构
│   └── qwen35.py         # Qwen3.5 GDA/GDN 架构
├── examples/
│   └── main.cpp          # CLI 推理入口
└── tests/
    ├── test_kernels.cpp
    └── test_graph.cpp
```

## 5. 与现有框架的关键差异

| | mollm | cactus | ggml | ncnn |
|---|---|---|---|---|
| 图构建 | Python 端 | Python 端 | C 运行时 lazy | .param 文本 |
| Tensor 含 graph info | 否 | 否 | 是 (op/src) | 否 |
| 内存管理 | BufferPool + liveness | BufferPool + liveness | 图级 gallocr | 每层临时分配 |
| 权重 | mmap | mmap | mmap (gguf) | 全量加载 |
| KVCache | 内置 INT8 量化 | 内置 INT8 量化 | 上层管理 | 外部管理 |
| GPU | Roadmap | 无 | CUDA/Metal/Vulkan | Vulkan |
| 代码量 (预估) | ~20K lines | ~34K lines | ~250K lines (core) | ~89K lines (layer only) |
| MLA 原生支持 | 是 | 否 | 否 (上层实现) | WIP |
| GatedDeltaNet | 是 | 是 | 否 (上层实现) | 否 |

## 6. 实现计划

### Phase 1：骨架 + 基础推理
1. `tensor.h/cpp` — Tensor + 视图
2. `buffer_pool.h/cpp` — 内存池
3. `mmap_file.h/cpp` — mmap 权重
4. `graph.h/cpp` + `io.cpp` — 图结构 + 序列化
5. `execute.cpp` — 节点调度 + liveness
6. `matmul.cpp` — FP16 GEMM (NEON)
7. `norm.cpp` — RMSNorm
8. `rope.cpp` — RotaryEmbed
9. `attention.cpp` — SDPA (Gemm-based, kv_cache=2)
10. Python 转译器 — PyTorch → graph
11. `engine/` — 模型加载 + prefill/decode
12. CLI 推理入口

### Phase 2：优化
13. `attention.cpp` — Flash attention (online softmax)
14. `kv_cache.cpp` — INT8 量化
15. `deltanet.cpp` — Gated DeltaNet
16. `matmul.cpp` — 量化 matmul
17. 多线程 parallel_for

### Phase 3：扩展
18. MoE
19. Gated Attention (Q/K Norm + Gate)
20. Conv1D causal
21. GPU/Vulkan backend
