# mollm Memory Model

*Draft: 2026-07-01*

## 背景

当前内存管理是可工作的局部协议：`Tensor` 持有裸 `data` 指针，`BufferPool`
按 power-of-2 bucket 复用中间 buffer，`execute_graph()` 在每次执行入口清理上一轮
borrowed view metadata，并按 graph policy 释放或复用 materialized pooled tensor。
这个方案已经支撑了 DYNAMIC shape、chunked prefill、静态 decode workspace 复用，
以及同 runtime shape 的 dynamic prefill workspace 复用，但 ownership 语义还没有
完全类型化：

- `Tensor` 没有析构释放语义，`MemoryType` 只是约定，不是所有权机制
- `BufferPool` 已记录 active allocation，pooled `Tensor` 也带 `owner_id` / `storage_id`；
  但释放仍靠调用协议，而不是 RAII storage handle
- zero-copy view 依赖 op/指针判断避免 double-release；RESHAPE 可能 zero-copy，也可能 materialize
- cache/state 已迁出 graph pool，由 engine persistent pool 持有；graph INPUT 只借用其 tensor metadata
- `run_graph()` 已直接借用 `embed()` / mask / RoPE helper tensor 作为 graph INPUT；
  调用方在输出已消费或 copy out 后释放 helper
- `prefill()` / `prefill_hidden()` 在 output 消费或 copy 后清空 borrowed views，并保留
  bounded same-shape workspace；`release_prefill_buffers()` 显式释放；
  `decode()` / `decode_hidden()` 保留 bounded static workspace，`reset()` 释放；
  `prefill_hidden()` / `decode_hidden()` 已 copy 到 engine-owned contiguous buffer

目标不是继续补单点 release，而是建立清晰的 lifetime/ownership 模型。

## 设计目标

1. **所有分配都可追踪**：engine 析构时能释放所有内存，包括 active buffer。
2. **Tensor 默认不拥有内存**：`Tensor` 表达 shape/stride/view，不承担释放责任。
3. **owned storage 与 borrowed view 显式区分**：view 不参与释放，materialized output 有明确 owner。
4. **cache/state 独立于 graph temporary**：KV cache、GDN state 的 lifetime 属于 engine，不属于某个 graph pool。
5. **graph output lifetime 明确**：返回值必须声明是 borrowed-until-next-execute，还是 copied/owned by caller。
6. **先正确后优化**：第一阶段允许执行结束统一释放 temporaries；liveness/early-free 后续再加。

## Lifetime 分层

### Package Lifetime

属于 `LLMEngine::load()` 到 `LLMEngine` 析构：

- `.mollm` whole-file mmap
- package metadata weight offset map
- extracted graph/tokenizer temp files
- load-time packed weights (`packed_weights_`, `embed_packed_`)

这些内存不应进入 graph temporary pool。

### Engine Lifetime

属于一次 engine load 到 engine 析构：

- KV cache buffers
- GDN recurrent state
- GDN conv state
- backend-owned persistent allocations

这些 storage 应由 `LLMEngine` 显式持有，例如 `engine_storage_` 或专用
`PersistentArena`。prefill/decode graph 的 cache INPUT tensor 只引用这些 storage。

### Execution Lifetime

属于一次 `execute_graph()` 调用：

- graph INPUT staging buffers（如果仍需要 copy）
- materialized intermediate outputs
- materialized view fallback output（如 non-contiguous RESHAPE）
- temporary mask/RoPE/embed buffers，如果不直接作为 borrowed input

这些 storage 由 `ExecutionScope` 或 graph runtime arena 管理。scope 结束时统一释放，
除非 output 明确被转移或 copied out。

### Borrowed View Lifetime

zero-copy RESHAPE、PERMUTE、SLICE 等 view tensor 不拥有内存。它们只引用某个
owner storage 的 `[data + offset, shape, stride]`。view 的有效期不能超过 owner。
CONCAT、TILE、CONTIGUOUS 和 non-contiguous RESHAPE 会 materialize output。

## 建议数据结构

### Tensor

`Tensor` 应逐步收敛为轻量 view：

```cpp
struct Tensor {
    Precision prec;
    int64_t shape[4];
    size_t stride[4];
    void* data;
    uint32_t owner_id;     // debug: owning pool id, 0 = external/unknown
    uint64_t storage_id;   // debug: allocation identity, copied by views
    size_t byte_offset;    // debug / optional
};
```

`MemoryType` 可以短期保留作为兼容字段，但不应作为释放决策的唯一依据。
当前代码已先加入 `Tensor::owner_id` 和 `Tensor::storage_id`：`owner_id` 标记
pooled tensor 来自哪个 `BufferPool`，`storage_id` 标记该 pool 内具体 allocation。
zero-copy view 会复制这两个字段；主要 release 路径会校验 owner mismatch，borrowed
判断会优先使用 `(owner_id, storage_id)`，再 fallback 到指针相等。

### TensorStorage

```cpp
enum class StorageKind {
    Pool,
    External,
    Mmap,
    Persistent,
};

struct TensorStorage {
    void* data = nullptr;
    size_t bytes = 0;
    StorageKind kind = StorageKind::Pool;
    uint32_t owner_id = 0;
    bool releasable = true;
};
```

storage 是 owner，tensor 是 view。

### BufferPool

`BufferPool` 应记录 active 和 free allocation：

```cpp
struct Allocation {
    void* ptr;
    size_t requested;
    size_t bucket;
    bool active;
};
```

最低要求：

- `acquire()` 注册 active allocation
- `release()` 校验 pointer 属于 pool、当前 active、bucket 匹配
- `clear()` 释放 active + free 全部 buffer
- debug build 下 double-release / foreign pointer / wrong size 直接 assert

## ExecutionScope

建议引入一次执行级别的 scope：

```cpp
class ExecutionScope {
public:
    Tensor make_temp(Precision prec, shape...);
    Tensor borrow_input(const Tensor& t);
    Tensor materialize(const Tensor& view);
    void release_all_except(uint32_t output_storage_id);
};
```

第一阶段可以不做 liveness。`execute_graph()` 结束时：

- 如果 output 是 borrowed internal tensor：保留 output owner 到下一次 execute 前
- 如果 caller 需要长期持有：copy out 到 caller-owned buffer
- 其他 execution temporary 全部 release

## Output 策略

需要显式选择两种 API：

### Borrowed Output

用于 chat/bench 快路径：

```cpp
Tensor run_graph_borrowed(...);
```

约定：返回 tensor 有效到同一个 graph 下一次 `execute_graph()` 或 engine reset 前。

### Owned Output

用于 tests / ppl / debug：

```cpp
std::vector<float> run_graph_output_copy(...);
```

调用者或 engine-owned buffer 持有结果，graph 可以立刻释放 temporaries。

当前 `prefill_hidden()` / `decode_hidden()` 返回 engine-owned copied output。
`prefill()` 只需要 logits sampling，可以用 borrowed output 并在 sampling 后 cleanup。
`decode()` 消费 output 后保留静态 decode graph 的 materialized workspace 供下一 token
复用；`reset()` 或 engine teardown 释放该 workspace。

## 迁移计划

### Phase 1: 修 BufferPool 所有权

不改 graph 语义，只修 allocator：

- 记录 active allocations
- `clear()` 释放 active + freelist
- debug assert double-release / foreign pointer / wrong bucket
- 增加测试：active 未 release 时析构/clear 不泄漏；double release 被检测

收益：engine repeated load/unload 不泄漏，pool stats 可信。

**状态：已落地第一版。** `BufferPool` 现在记录 active allocations，`clear()` /
析构会释放 active + freelist，`release()` 会校验 foreign pointer、double release
和 bucket mismatch。

### Phase 2: 消除 helper tensor 泄漏

处理 `embed()` / mask / RoPE：

- 短期：在 `run_graph()` copy 完输入后释放 original helper tensor
- 更好：让 `run_graph()` 直接借用 helper tensor 作为 INPUT，不再 copy

收益：decode/prefill per-call active 增长消失。

**状态：已落地。** `run_graph()` 现在直接把 `hidden`、`mask`、`cos`、`sin`
绑定为 graph INPUT，不再分配 staging buffer 或 memcpy。`prefill_hidden()` /
`decode_hidden()` 在 output copy 后释放 helper；`prefill()` / `decode()` 在
`run_lmhead()` 消费 borrowed output 后释放 helper。

### Phase 3: cache/state 迁出 graph pool

- 新增 engine-owned persistent allocator/storage list
- `allocate_caches()` 从 engine storage 分配
- prefill/decode graph INPUT tensor 都 borrow engine storage

收益：cache lifetime 与 graph temporary 解耦。

**状态：已落地第一版。** `LLMEngine` 现在持有 `persistent_pool_`，
`allocate_caches()` 从该 pool 分配 KV cache、GDN recurrent state 和 GDN conv state。
prefill/decode graph 的 cache INPUT tensor 只引用这些长期 buffer；graph runtime pool
仅用于 execution temporaries。后续仍需要用 storage id/owner id 替代单一
`MemoryType::POOLED`，避免 owner 只存在于调用约定里。

### Phase 4: 定义 output lifetime 并恢复 cleanup

- dynamic prefill 在 output 消费/copy 后清空 borrowed views，并在 runtime shape key
  相同时保留 materialized workspace
- 静态 `decode()` / `decode_hidden()` 允许 materialized workspace 跨 token 复用
- `prefill_hidden()` / `decode_hidden()` 返回 copied/owned output
- `release_prefill_buffers()` 使用统一 graph temporary cleanup，避免释放 borrowed view

收益：最后一个 chunk 中间 buffer 不再悬挂到下一次 execute。

**状态：已落地。** executor 不再给 zero-copy RESHAPE/PERMUTE/SLICE 预分配 buffer。
静态 decode graph 通过 `ExecContext::reuse_static_workspace` 保留 materialized pooled
tensors；dynamic prefill 通过 `ExecContext::reuse_same_shape_workspace` 只在
`runtime_seq_len/static_padded/padded_seq_len/batch` 完全一致时复用 materialized
workspace。每次 output 消费/copy 后都会清空 borrowed views。`reset()` 会释放 decode
workspace 但保留 prefill workspace，`release_prefill_buffers()` 显式释放 prefill
workspace。`test_execute` 覆盖 zero-copy view 不分配、CONCAT/non-contiguous RESHAPE
连续执行 active bytes 不增长，并检查静态和 dynamic same-shape workspace 二次执行不再
reacquire。`prefill_hidden()` / `decode_hidden()` 已 copy 到 engine-owned contiguous
buffer，PPL/debug 路径不依赖 graph pool 内部 output。

### Phase 5: liveness/early release

保守第一版已落地：

- graph load 后计算 `last_use`
- view-aware owner dependency：释放 owner 前必须确认所有 derived view 不再使用
- materialized op output 可按 last_use 提前 release

收益：降低 peak memory，同时不牺牲语义清晰度。

**状态：实验性第一版已落地，仅对无 stateful/cache-mutating op 的图启用。**
`prepare_execution()` 可以计算 view-aware release queue。`PERMUTE`/`SLICE`/
`RESHAPE` 按可能 borrowed 来传播 owner last-use；release 时 borrowed view 只清空
tensor metadata，不释放 storage。`RESHAPE` 如果运行时实际 materialize，则在
last-use 后释放；如果 zero-copy borrow，则跳过释放并清空 stale view pointer。

包含 SDPA、Gated DeltaNet、SHORTCONV 的真实模型图仍使用 end-of-call cleanup。
这些 op 涉及 KV cache / recurrent state in-place 更新，需要等 `TensorStorage` /
storage registry 能表达 persistent storage 和 borrowed view 依赖后再打开 early release。

## 不变量

这些规则应在 debug build 中 assert：

- 非 owner tensor 不能 release
- view tensor 的 owner storage 必须仍 active
- graph INPUT 中 cache/state 不能被 execution cleanup release
- borrowed output 有明确 owning storage，下一次 execute 前要么仍 valid，要么先 copy out
- prefill graph pool active bytes 可以保留 bounded same-shape workspace；连续相同
  runtime shape 不得增长，`release_prefill_buffers()` 后必须回到 0。cache/state 属于
  engine persistent pool，不计入 graph pool baseline
- static decode graph pool active bytes 可以保留 bounded workspace；连续 decode 不得增长，
  `reset()` 后必须回到 0

## 当前优先级

Phase 1、Phase 2、Phase 3 第一版、output lifetime cleanup、owner/storage-id debug
检查第一版和 view-aware last-use 第一版已落地。下一步优先级：

1. 将 `(owner_id, storage_id)` 扩展成 explicit `TensorStorage` / storage registry。
2. 用 storage registry 覆盖 persistent cache/state、external mmap 和 view byte_offset。
3. 增加 peak-memory benchmark / regression guard，量化 liveness 收益。
