#pragma once

#include "graph/graph.h"
#include <vector>

class ThreadPool;
class Backend;
class HostMatmulBackend;

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
    // Selected dense matmuls; tensors and persistent state remain host-owned.
    HostMatmulBackend* matmul_backend = nullptr;
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
    // Experimental planner: retain every stateful op and all its storage
    // dependencies, but permit last-use reclamation for unrelated nodes.
    bool          conservative_stateful_liveness = false;
    // Diagnostic only: honor explicit backend state/output lifetime contracts.
    // Requires conservative_stateful_liveness; engine selection test-only,
    // prefill-only with bounded shape/memory guards and no captured segments.
    bool          stateful_readonly_last_use = false;
    // Opt-in: fuse only complete producer/consumer chains inside a segment.
    bool          segment_fusions_enabled = false;
    bool          segment_fusions_identity = false;
    int           conservative_liveness_seq_limit = 0; // diagnostic cap, 0 = uncapped
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

    // liveness: release_queue[i] = nodes that can be freed after node i
    std::vector<std::vector<uint32_t>> release_queue;
    // Node input Tensor addresses are stable after graph_load(). Resolve them
    // once instead of allocating a temporary vector for every node/token.
    std::vector<std::vector<const Tensor*>> resolved_inputs;
    // Runtime-only adjacent gate fusion plan. Serialized graph/IDs unchanged.
    std::vector<int32_t> sigmoid_gr_reduce_producers;
    std::vector<std::vector<const Tensor*>> sigmoid_gr_reduce_inputs;
    std::vector<std::vector<uint32_t>> sigmoid_gr_reduce_release_queue;
    std::vector<int32_t> hc_up_reduce_producers, hc_up_reduce_consumers;
    std::vector<std::vector<const Tensor*>> hc_up_reduce_inputs;
    std::vector<std::vector<uint32_t>> hc_up_reduce_release_queue;
    // Only INPUT/CONSTANT nodes may intervene; the carry is not eliminated.
    std::vector<int32_t> gr_inject_rms_producers;
    std::vector<int32_t> gr_inject_rms_consumers;
    std::vector<std::vector<const Tensor*>> gr_inject_rms_inputs;
    std::vector<std::vector<uint32_t>> gr_inject_rms_release_queue;
    std::vector<int32_t> moe_gr_inject_rms_producers,moe_gr_inject_rms_consumers;
    std::vector<std::vector<uint32_t>> moe_gr_inject_rms_release_queue;
    std::vector<NodeProfileStat> profile_stats;

    // A suspended, unfused pass. The graph/tensors and caller-owned inputs
    // must remain unchanged until completion; separate windows need separate
    // graph runtime storage, not merely separate ExecContext objects.
    int segment_next = -1;
    Graph* segment_graph = nullptr;
    BufferPool* segment_pool = nullptr;
    Backend* segment_backend = nullptr;
    Backend* segment_moe_backend = nullptr;
    int segment_seq = -1, segment_batch = -1, segment_padded_seq = -1;
    bool segment_static_padded = false;
};

void reset_profile_stats(ExecContext& ctx);

/// Initialize execution bookkeeping after graph_load().
/// The release queue frees materialized intermediates after their last use.
/// Borrowed views are cleared, not released; RESHAPE is conservatively treated
/// as view-like for owner propagation because it may borrow or materialize at
/// runtime depending on input contiguity.
void prepare_execution(ExecContext& ctx, bool prepare_backend_plan = true);

/// Initialize an EMPTY runtime frame without copying mapped weight ownership.
/// Constants and persistent INPUT tensors borrow source storage; private_inputs
/// names INPUT IDs whose storage must instead be allocated/bound by the caller.
/// All computed tensors start unallocated, including source's materialized
/// outputs/views. Source mappings, shared state, and backend must outlive frame.
/// Construct a separate ExecContext and call prepare_execution() for the frame.
/// Caller owns output/private-input cleanup; never free borrowed shared inputs.
/// Invalid graphs/IDs or a nonempty destination return false without mutation.
bool initialize_execution_frame(const Graph& source, Graph& frame,
                                const std::vector<uint32_t>& private_inputs);

/// Inject runtime seq_len into the graph's INPUT tensors and patch stateful
/// op params (GDN_PREFILL / SHORTCONV n_real_tokens). Must be called before
/// execute_graph() when the graph contains dynamic DimExpr dims.
void inject_runtime_shapes(ExecContext& ctx);

/// Execute the graph once. Input tensors must already have data set. A
/// non-negative stop index executes through that node, including its stateful
/// side effects, then returns without running the unused graph suffix.
/// Allocates intermediate tensors from the BufferPool.
void execute_graph(ExecContext& ctx, int stop_after_node_index = -1);

/// Execute [first, end) in node-index order. Begin with first=0 and resume
/// exactly at the preceding end. Only the first segment resets tensors;
/// ordinary last-use reclamation still runs inside every segment. Fusions
/// are disabled by default; the explicit opt-in permits only complete chains
/// within this range, never a deferred producer across a suspension boundary.
/// Each segment invalidates backend execution-local caches. The caller owns
/// synchronization and must not mutate runtime/input storage while suspended.
/// Invalid ranges or changed execution identity fail before dispatch.
void execute_graph_segment(ExecContext& ctx, int first, int end);

/// Consume the next stateless MOE node in two suspended frames on one backend.
/// Keeps normal shape/liveness/cursor semantics; optional backend aggregation
/// materializes two independently owned outputs, with no graph mutation.
/// Return1 merged,0 ordinary two-dispatch fallback,-1 rejected/failed.
int execute_graph_moe_pair(ExecContext& a, ExecContext& b, int node_index);
