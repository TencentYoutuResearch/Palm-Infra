# mollm Architecture

*Last updated: 2026-06-29*

## 1. 概述

mollm 是一个 mobile-oriented 的 LLM 推理引擎，面向 ARM-based 设备。
目前在 Apple Silicon（M 系列）上开发和测试，移动端 ARM（Qualcomm Oryon、MediaTek）在 roadmap 中。
Python 转译前端 → 二进制图 → C++ 执行器 + NEON FP16FML kernels。

**设计原则**：
- 图构建在 Python 端，C++ 端是纯执行器（顺序遍历节点，dispatch kernel）
- 所有 op fusion 在转译时完成，运行时无 JIT
- FP16 权重 + FP16FML lane-FMA NEON kernels（GEMM/GEMV）
- 单文件 `.mollm` 打包格式（graph + weights + tokenizer）

## 2. 数据流

```
PyTorch safetensors
       │
       ▼  Python transpile (models/*.py)
       │  - export weights to .weights files
       │  - build prefill graph (seq_len=256) + decode graph (seq_len=1)
       │  - pack into single .mollm file
       ▼
.mollm package
       │
       ▼  C++ engine (engine/)
       │  - mmap .mollm, extract graphs to temp files
       │  - graph_load: parse binary graph, set up CONSTANT weights
       │  - allocate KV caches + GDN state buffers
       │
       ▼  Execution (graph/execute.cpp)
       │  - execute_graph: sequential node dispatch
       │  - per-node: dispatch_kernel → NEON kernel
       │  - BufferPool memory reuse
       │
       ▼  Output: token logits → sampling → next token
```

## 3. 核心数据结构

### Tensor（`kernels/tensor.h`）

固定 4-D `[d0, d1, d2, d3]`，d0 是 innermost（column）。stride 单位是 bytes。

```cpp
struct Tensor {
    Precision prec;        // FP32 / FP16 / INT8
    MemoryType mem_type;   // NONE / OWNED / POOLED / EXTERNAL
    int64_t shape[4];
    size_t stride[4];
    void* data;
    // zero-copy views: view_1d, view_2d, reshape, permute
};
```

关键特性：
- `is_contiguous()` 检查默认 row-major 布局
- `permute(a0,a1,a2,a3)` 零拷贝（只改 shape/stride）
- `reshape(s0,s1,s2,s3)` 零拷贝（继承 stride）
- `channel(c)` 用 `stride[2]` 访问 head plane

### Graph（`graph/graph.h`）

```cpp
struct Graph {
    std::vector<GraphNode> nodes;
    std::vector<uint32_t> graph_inputs, graph_outputs;
    std::unordered_map<std::string, std::string> metadata;  // rope_dim, model_type, ...
    struct Runtime {
        std::vector<Tensor> tensors;       // indexed by node.id
        std::vector<MappedFile> weights;   // file-mode mmap'd weights
        BufferPool pool;                   // intermediate buffer reuse
    };
};
```

**GraphNode**：`{id, op_type, inputs[], params{i32,f32,str}, out_shape[4], out_prec}`

**OpType 枚举**（24 个 op）：
- 数据：INPUT, CONSTANT
- 计算：MATMUL, RMS_NORM, SILU, GELU, SIGMOID, EXP, SOFTPLUS
- Attention：SDPA, SDPA_MLA
- GDN：GATED_DELTANET_DECODE, GATED_DELTANET_PREFILL
- 卷积：SHORTCONV
- 变换：ROTARY_EMBED, RESHAPE, PERMUTE, CONCAT, SLICE, TILE, CONTIGUOUS
- 逐元素：ADD, MUL
- 量化：QUANTIZE_KV, DEQUANTIZE_KV

### CacheMetadata（`engine/engine.h`）

64 字节 header 嵌入 KV cache buffer 头部：

```cpp
struct CacheMetadata {
    uint64_t current_seq_len;    // valid past_len
    uint64_t max_seq_len;        // buffer capacity
    uint64_t num_kv_heads, head_dim, v_head_dim;
    uint64_t reserved[3];
};
```

## 4. 图执行（`graph/execute.cpp`）

### execute_graph

顺序遍历节点列表：
1. 跳过 INPUT/CONSTANT
2. 从 `node.out_shape` 初始化 output shape
3. 从 BufferPool acquire output buffer（如果需要）
4. 收集 inputs（Tensor 指针数组）
5. `dispatch_kernel(op, params, inputs, output, thread_pool)`
6. 执行 release_queue（释放已完成 tensor）

### dispatch_kernel 关键路径

| Op | 实现 | 关键点 |
|----|------|--------|
| MATMUL | `kernel_matmul_fp32` | FP16FML lane-FMA GEMM / dedicated GEMV，支持 fused activation |
| SDPA | `kernel_sdpa` | Flash attention（FP16FML），KV cache append，head 级并行 |
| GATED_DELTANET | `kernel_gdn_prefill/decode` | Fused recurrence（2-pass），head 级并行 |
| SHORTCONV | 内联 | Depth-wise causal conv1d + silu，group 级并行 |
| RESHAPE | 零拷贝 or `materialize_strided` | contiguous 零拷贝；strided 行级 memcpy |
| CONTIGUOUS | `materialize_strided` | 行级 memcpy fast path |
| PERMUTE/SLICE | 零拷贝 | 只改 stride/shape |
| ADD/MUL | NEON 8-wide | stride-aware，支持 broadcast |
| RMS_NORM | `kernel_rms_norm` | NEON，支持 strided（ldx=stride[1]） |
| ROTARY_EMBED | `kernel_rope` | halves/interleave 模式，支持 strided |
| SILU/SIGMOID | `unary_stride_aware` | NEON polynomial exp 近似 |

### materialize_strided

RESHAPE/CONTIGUOUS 在非 contiguous 输入时调用。dim0 连续时用整行 memcpy（shape[0] 倍加速）。

## 5. NEON Kernels（`kernels/`）

### Matmul（`matmul.cpp`）

**布局约定**：A `[K, M]`，B `[N, K]`（weight），C `[N, M]` FP32

**Kernel 变体**：
- **GEMM lane-FMA**（`matmul_fp16_neon_8x8_range_packed_a_fp16acc`）：A+B 都 pre-packed interleaved `[M/8, K, 8]`，`vfmaq_lane_f16` FP16×FP16→FP16 FMA，2-way K-unroll。988 GF/s（86% FP16FML peak）
- **GEMV**（`matmul_fp16_neon_gemv_range_fp16acc`）：M=1，8-way K-unroll，8 独立 FP16 acc chain
- **FP32 acc fallback**：`vfmlalq_laneq_low/high_f16` FP16→FP32 widening

**调度**：
- M=1 → GEMV（shard by N）
- M≥8 + 多线程 → `parallel_for_2d`（atomic-steal，M-tile=8, N_BLOCK=256）
- Load-time B interleaved packing（一次性，跳过 embed_tokens）
- A pre-packing per M-tile

### SDPA（`attention.cpp`）

Flash attention，head 级并行（`parallel_for(0, num_heads, 1, run_head)`）

**FP16FML 路径**（默认）：
- `flash_attn_fp16_decode`（Bc=64，register-tiled PV 64-wide）
- `flash_attn_fp16_prefill`（Br=8, Bc=64，`dot_fp16_8x` + register-tiled PV）
- KV cache FP16，Q 运行时转换

**FP32 fallback**：`flash_attn_fp32_decode/prefill`

### GDN（`gdn.cpp`, `gdn_neon.h`, `gdn_decode.cpp`, `gdn_prefill.cpp`）

Gated DeltaNet fused kernel（8 inputs，1 output + in-place state）。

**Recurrence 数学**：
```
state_t = state_{t-1} * exp(g_t) + outer(k_t, (v_t - state_{t-1} @ k_t) * beta_t)
out_t   = (state_t @ q_t) * scale
```

**Fused 2-pass 实现**（`gdn_recurrence_neon`）：
- Pass 1: decay（state *= g）+ matvec1（kv_mem = state @ k）
- Pass 2: rank1_update（state += k ⊗ delta）+ matvec2（attn_out = state @ q）
- state 内存流量从 5-pass（320KB/head/token）降到 2-pass（128KB，-60%）
- RMSNormGated 用 `sigmoid_f32_neon`（多项式近似）

**并行**：prefill 和 decode 都按 value-head 并行（`parallel_for(0, num_v_heads, chunk, process_head)`）

### 其他 kernels

- **RMSNorm**（`norm.cpp`）：NEON 8-wide，支持 strided（`ldx = stride[1]/sizeof(float)`）
- **RoPE**（`rope.cpp`）：halves/interleave 模式，支持 strided
- **ThreadPool**（`threading.h`）：`parallel_for`（1D static）+ `parallel_for_2d`（atomic-steal），park/resume

## 6. 内存管理

### BufferPool（`graph/buffer_pool.h`）

Power-of-2 bucketed freelist allocator：
- MIN_BUCKET=1KB，ALIGNMENT=64（cache line）
- `acquire(bytes)` → 从 freelist 复用或 `aligned_alloc`
- `release(ptr, bytes)` → 返回 freelist（不 free）
- `reset()` → 把所有 active buffer 移回 freelist（prefill → decode 切换）

### MappedFile（`graph/mmap_file.h`）

权重文件 mmap（88 字节 header：magic, ndim, prec, shape, data_offset, data_size, scales）
- `data()` 返回 header 之后的权重数据
- `prefetch()`（MADV_WILLNEED）/ `release_pages()`（MADV_DONTNEED）

### .mollm Package（`python/transpile.py`）

单文件打包格式：

```
[Header 128B]
  magic "MOLM" + version
  6×(offset, len) pairs: metadata, tokenizer, jinja, prefill_graph, decode_graph, weights
[metadata JSON] — weights offset map + model config
[tokenizer.json]
[chat_template.jinja]
[prefill graph] (standard .graph format)
[decode graph]
[weights region] — all .weights files concatenated
```

C++ `load_package`：mmap 整个文件，解析 metadata JSON 建 weight filename → offset 映射，提取 graph/tokenizer 到临时文件。`load_graph` 里对每个 CONSTANT 节点，用 map 解析路径到 mmap 内 offset。

## 7. 模型转译（`models/`）

### Qwen3.5（`qwen35.py`）

混合线性/全注意力架构：
- 18 层 linear attention（Gated DeltaNet）+ 6 层 full attention（GQA + QK norm + output gate）
- 每 4 层一个 full attention 层
- SwiGLU MLP

**Linear attention 层**：qkv_proj → shortconv → fused_gdn → out_proj
**Full attention 层**：q/k/v_proj → QK norm（per-head RMSNorm）→ RoPE → SDPA → output gate → o_proj

### Youtu-LLM（`mla.py`）

MLA（Multi-head Latent Attention）+ SwiGLU MLP with merged gate_up + fused SILU

## 8. 引擎（`engine/`）

### LLMEngine

- `load(cfg)`：加载 package 或 artifacts，设置权重 + 分配 cache
- `prefill(token_ids)`：分块 prefill（每块 ≤ graph_seq_len），支持 past_len > graph_seq_len
- `decode(token_id)`：单步 decode
- `reset()`：清 KV cache + GDN state
- `park_workers()`：挂起线程（auto-resume on next prefill/decode）

**多 graph 架构**：prefill graph（seq_len=256）+ decode graph（seq_len=1），共享物理 KV cache（load-time cache migration）

### Tokenizer（`tokenizer.cpp`）

HF tokenizer.json 格式 → vocab + merges + added_tokens
- 3 阶段 GPT-4 风格 pre-tokenizer（CJK 隔离 + word split + byte-level）
- `apply_chat`：ChatML（Qwen3.5）或 Llama-3 格式，硬编码 token id

## 9. 文件结构

```
mollm/
├── CMakeLists.txt          项目配置（C++17, NEON, nlohmann/json）
├── CODEBUDDY.md            Agent 指引
├── README.md               项目概览 + roadmap
├── ARCHITECTURE.md         本文档
├── docs/
│   ├── CURRENT_STATE.md    当前状态（性能 + profiling + next steps）
│   ├── OPTIMIZATION_LOG_QWEN35.md  Qwen3.5 优化日志（12 attempts）
│   ├── OPTIMIZATION_LOG.md         Youtu-LLM 优化日志（17 attempts）
│   └── DEBUG_QWEN35.md     调试笔记
├── engine/                 LLMEngine + tokenizer + generation
├── graph/                  图执行（execute, io, buffer_pool, mmap）
├── kernels/                NEON kernels（matmul, attention, gdn, norm, rope）
├── models/                 Python 转译器（qwen35.py, mla.py）
├── python/                 GraphBuilder + serializer（transpile.py）
├── examples/               mollm_chat + mollm_bench
├── tests/                  18 个测试
└── third_party/nlohmann/   JSON 库
```
