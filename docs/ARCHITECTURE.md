# mollm Architecture

*Last updated: 2026-07-01*

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
- `reset()` → `clear()` 的兼容别名；会释放 active + freelist

### 当前内存生命周期

内存管理当前是工程上可工作的 BufferPool/reset 模式，但还不是严格的
ownership/liveness 模型：

- CONSTANT 权重指向 `.mollm` mmap 或文件 mmap，`MemoryType::EXTERNAL`
- KV cache、GDN state、GDN conv state 在 load 时从 `LLMEngine` 的 persistent
  pool 分配，prefill/decode graph INPUT tensor 共享物理指针
- 每次 `execute_graph()` 入口都会先清空上一轮 borrowed view metadata；默认会释放
  materialized temporaries。静态 decode graph 可通过 `reuse_static_workspace` 保留
  materialized POOLED tensor 跨 token 复用；dynamic prefill 可通过
  `reuse_same_shape_workspace` 在 runtime shape key 完全相同时复用 materialized workspace
- zero-copy RESHAPE/PERMUTE/SLICE 只借用输入指针，入口先清空 borrowed view；
  CONCAT/TILE/CONTIGUOUS 和 non-contiguous RESHAPE 会 materialize output
- `BufferPool` 记录 active allocations；`clear()`/析构释放 active + freelist，
  `release()` 校验 foreign pointer、double release 和 bucket mismatch
- pooled tensor 带 `owner_id` / `storage_id`，主要 release 路径会校验 tensor
  owner 是否匹配目标 pool，borrowed view 判断优先使用 storage id
- `run_graph()` 直接借用 `embed()` / RoPE / mask helper tensor 作为 graph INPUT；
  helper 在 output 已消费或 copy out 后由调用方释放
- `prefill()` / `prefill_hidden()` 在 output 消费或 copy 后清空 borrowed views，并保留
  bounded same-shape workspace；`release_prefill_buffers()` 显式释放
- `decode()` / `decode_hidden()` 消费或 copy output 后保留静态 decode workspace；`reset()`
  会释放该 workspace
- `prefill_hidden()` 返回 engine-owned contiguous copy，不暴露 graph pool 内部 output
- `prepare_execution()` 有实验性 view-aware last-use release queue；当前仅对无
  stateful/cache-mutating op 的图启用，真实模型图仍使用 end-of-call cleanup

这个策略解决了 DYNAMIC/chunked prefill 中暴露的 stale view pointer 和 pool
增长问题，并恢复了 decode 以及同 shape prefill 的主要 buffer 复用。注意：
`release()` 只把 buffer 移到 BufferPool freelist，不降低 RSS；workspace 复用主要把同一批
内存从 freelist 变为 active，避免下一次同 shape 执行的 release/reacquire 开销。仍有明显
技术债：

- view ownership 仍依赖 op 类型 + 指针比较，缺少类型层面的 owner/borrowed 区分
- `MemoryType::POOLED` 仍没有携带 owner id，persistent pool 与 graph pool 的区分
  依赖调用协议和 INPUT cleanup 规则
- `Tensor` 本身没有析构释放语义，所有释放依赖 engine/executor 的外部协议

后续如果继续做 graph fusion、continuous batching 或多 backend，建议先把内存模型
收敛成显式的 `TensorStorage`/borrowed view/liveness release 三层，否则正确性会越
来越依赖约定。详细设计草案见 `docs/MEMORY_MODEL.md`。

### MappedFile（`graph/mmap_file.h`）

权重文件 mmap（88 字节 header：magic, ndim, prec, shape, data_offset, data_size,
scales_offset, scales_size, group_size, num_groups）
- `data()` 返回 header 之后的权重数据
- `scales()` 返回 W8 per-channel/per-group FP32 scales（非量化权重为 `nullptr`）
- `prefetch()`（MADV_WILLNEED）/ `release_pages()`（MADV_DONTNEED）

### .mollm Package（`models/transpile.py`）

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

### 统一入口（`converter.py`）

读 `config.json` 的 `model_type` 字段，自动分发到对应的 converter：
- `qwen3_5` → `qwen35.convert_qwen35()`
- `youtu` → `mla.convert_mla()`

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
├── AGENTS.md               Agent 指引
├── README.md               项目概览 + roadmap
├── ARCHITECTURE.md         本文档
├── docs/
│   ├── CURRENT_STATE.md    当前状态（性能 + profiling + next steps）
│   ├── MEMORY_MODEL.md     内存 ownership/lifetime 设计草案
│   ├── OPTIMIZATION_LOG_QWEN35.md  Qwen3.5 优化日志（13 attempts）
│   ├── OPTIMIZATION_LOG.md         Youtu-LLM 优化日志（17 attempts）
│   └── DEBUG_QWEN35.md     调试笔记
├── engine/                 LLMEngine + tokenizer + generation
├── graph/                  图执行（execute, io, buffer_pool, mmap）
├── kernels/                NEON kernels（matmul, attention, gdn, norm, rope）
├── models/                 Python 转译器（converter.py, qwen35.py, mla.py, transpile.py）
├── examples/               mollm_chat + mollm_bench
├── tests/                  19 个测试（unit + e2e + shape_propagation.py）
└── third_party/nlohmann/   JSON 库
```

## 10. Dynamic Shape & Backend Abstraction（v3 graph format）

### Dynamic shape schema

每个 `GraphNode` 携带 `dim_expr[4]` 字段（每维一个 `DimExpr`）：

```cpp
struct DimExpr {
    int8_t  kind  = 0;  // CONST/SEQ/MUL/ADD/BATCH
    int32_t coeff = 0;  // MUL multiplier or ADD constant term
};

// CONST: value = out_shape[i]
// SEQ:   value = runtime_seq_len
// MUL:   value = coeff * runtime_seq_len
// ADD:   value = coeff + runtime_seq_len
// BATCH: reserved for future batch dim
```

- **transpile 时**：`propagate_dim_exprs()` 从 INPUT 节点传播 symbolic shape 到下游所有 tensor
- **runtime 不做 shape inference**：只读 `node.dim_expr[]`，在 `inject_runtime_shapes()` 和 `execute_graph()` 中用 `eval_dim()` 替换运行时 seq_len
- **graph format v3**：`dim_expr[4]` 紧跟 `out_shape[4]` 序列化（8 字节/dim）。不向前兼容 v2
- **复合维支持**：reshape 可以显式标记 `SEQ`，`num_heads * SEQ` 会表达成 `MUL(coeff=num_heads)`，避免 `head_dim=256` 与 `seq_len=256` 这种数值碰撞

### Backend 抽象

```cpp
class Backend {
public:
    virtual ShapeMode shape_mode() const = 0;  // DYNAMIC (CPU) | STATIC_PADDED (NPU future)
    virtual int padded_seq_len() const { return -1; }
    virtual void dispatch(const GraphNode& node,
                          const std::vector<const Tensor*>& inputs,
                          Tensor* output, ThreadPool* thread_pool) = 0;
};

class CPUBackend : public Backend { /* routes to NEON kernels */ };
// 未来: class NPUBackend : public Backend { ... };
```

- **`.mollm` 后端中立**：transpile 不做任何后端特定优化、不写 PAD 节点、不假设 shape mode
- **engine load 时选 backend**：CPUBackend 是当前唯一实现，未来 NPUBackend 由 runtime 选择
- **Shape mode 由 backend 决定**：CPU=DYNAMIC（无 padding），NPU=STATIC_PADDED（pad 到编译时 seq_len）

### 当前状态

- **DYNAMIC 模式已启用**：CPUBackend 默认无 padding，短 prompt 按实际 token 数执行
- **STATIC_PADDED 仍保留**：`--static-padded` 用于 A/B 对比和未来 NPU fixed-shape 路径
- **chunked prefill 已验证**：Test 5 跑 256 token 拆成多种 chunk 组合，PPL 一致（8.4893 vs HF 8.50）
- **已修复 DYNAMIC shape bug**：stale view shape、view op borrowed pointer 和跨 `execute_graph()` stale data 指针问题已在 executor 入口清理逻辑中处理
