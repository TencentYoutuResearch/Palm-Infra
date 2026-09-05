#pragma once

#include "graph/graph.h"
#include <vector>

class ThreadPool;
class Backend;

// ---------------------------------------------------------------------------
// mollm — Execution context
//
// prepare_execution() initializes per-graph execution bookkeeping and computes
// a conservative view-aware last-use release queue.
// execute_graph() runs one full pass.
// ---------------------------------------------------------------------------

struct NodeProfileStat {
    uint32_t node_id = 0;
    OpType op_type = OpType::INPUT;
    uint64_t calls = 0;
    uint64_t total_ns = 0;
};

struct ExecContext {
    Graph*        graph   = nullptr;
    BufferPool*   pool    = nullptr;
    ThreadPool*   thread_pool = nullptr;
    Backend*      backend = nullptr;   // op dispatcher (CPU/NPU/...)
    // Optional decode-only routed-MoE accelerator. All other graph operators
    // continue through `backend`, so host intermediates remain authoritative.
    Backend*      moe_backend = nullptr;
    bool          execution_failed = false;
    bool          profile_enabled = false;
    // DeepSeek-V4 activations are BF16 in the published inference graph.
    // prepare_execution() derives this from DeepSeek-specific graph nodes.
    bool          emulate_bf16_activations = false;
    // Decode-only experimental SSD prefetch using the current gate input and
    // the next MoE layer's router.
    bool          moe_cross_layer_prefetch = false;
    // Exact next-layer prefetch for token-id hash-routed MoE layers.
    bool          moe_hash_cross_layer_prefetch = false;
    // Label used by optional Chrome Trace events (normally prefill/decode).
    const char*   trace_label = "graph";

    // Keep materialized tensors alive across execute_graph() calls when the
    // graph is fully static. Borrowed views are still cleared each call.
    bool          reuse_static_workspace = false;

    // Keep materialized tensors alive for dynamic graphs only when the runtime
    // shape key is exactly the same as the previous execution.
    bool          reuse_same_shape_workspace = false;
    bool          workspace_shape_valid = false;
    int           workspace_runtime_seq_len = -1;
    int           workspace_runtime_batch = -1;
    bool          workspace_static_padded = false;
    int           workspace_padded_seq_len = -1;

    // ---- Dynamic shape injection ----
    // When the graph contains dynamic DimExpr dims (SEQ/MUL/ADD), runtime
    // injects the actual seq_len here before execution. inject_runtime_shapes()
    // reads this and fills tensor shapes accordingly.
    //   static_padded=false (DYNAMIC mode): runtime_seq_len = actual token count
    //   static_padded=true  (STATIC_PADDED mode): runtime_seq_len = actual token
    //     count (for n_real_tokens in stateful ops), padded_seq_len = graph's
    //     fixed seq_len (for shape + buffer allocation).
    int  runtime_seq_len = -1;     // actual token count (must be set if graph has SEQ)
    int  runtime_batch   = -1;     // reserved for future BATCH dim support
    bool static_padded   = false;  // STATIC_PADDED mode (NPU backend future)
    int  padded_seq_len  = -1;     // padded_seq_len for STATIC_PADDED mode
    int  confirmed_prefix_tokens = 0;  // transactional verification checkpoint

    // liveness: release_queue[i] = nodes that can be freed after node i
    std::vector<std::vector<uint32_t>> release_queue;
    // Node input Tensor addresses are stable after graph_load(). Resolve them
    // once instead of allocating a temporary vector for every node/token.
    std::vector<std::vector<const Tensor*>> resolved_inputs;
    std::vector<NodeProfileStat> profile_stats;
};

void reset_profile_stats(ExecContext& ctx);

/// Initialize execution bookkeeping after graph_load().
/// The release queue frees materialized intermediates after their last use.
/// Borrowed views are cleared, not released; RESHAPE is conservatively treated
/// as view-like for owner propagation because it may borrow or materialize at
/// runtime depending on input contiguity.
void prepare_execution(ExecContext& ctx);

/// Inject runtime seq_len into the graph's INPUT tensors and patch stateful
/// op params (GDN_PREFILL / SHORTCONV n_real_tokens). Must be called before
/// execute_graph() when the graph contains dynamic DimExpr dims.
void inject_runtime_shapes(ExecContext& ctx);

/// Execute the graph once. Input tensors must already have data set. A
/// non-negative stop index executes through that node, including its stateful
/// side effects, then returns without running the unused graph suffix.
/// Allocates intermediate tensors from the BufferPool.
void execute_graph(ExecContext& ctx, int stop_after_node_index = -1);
