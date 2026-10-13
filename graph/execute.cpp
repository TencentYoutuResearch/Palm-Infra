#include "graph/execute.h"
#include "graph/hc_deferred_rms.h"
#include "runtime/backend.h"
#include "core/bf16.h"
#include "storage/ssd_expert_cache/cache.h"
#include "core/tensor.h"
#include "runtime/trace.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool device_reshape_borrows(const Backend* backend, const Tensor& source,
                                   const Tensor& target) {
    size_t strides[4];
    return source.is_contiguous() ||
        (backend && backend->reshape_view_strides(source, target, strides));
}

// ---------------------------------------------------------------------------
// DimExpr evaluation — substitute runtime seq_len into a symbolic dim expr.
// ---------------------------------------------------------------------------

bool initialize_execution_frame(const Graph& source, Graph& frame,
                                const std::vector<uint32_t>& private_inputs) {
    const size_t count = source.nodes.size();
    if (&source == &frame || count == 0 || source.runtime.tensors.size() != count ||
        !frame.nodes.empty() || !frame.runtime.tensors.empty() || !frame.runtime.weights.empty() ||
        !frame.graph_inputs.empty() || !frame.graph_outputs.empty() || !frame.metadata.empty() ||
        frame.runtime.pool.pool_bytes() != 0) return false;
    std::vector<uint8_t> seen(count, 0), input(count, 0), private_input(count, 0);
    for (const auto& node : source.nodes) {
        if (node.id >= count || seen[node.id]) return false;
        // Consumers must follow all their producers, even when IDs are not
        // in execution order. Tensor storage is still indexed by ID.
        for (uint32_t dependency : node.inputs)
            if (dependency >= count || !seen[dependency]) return false;
        seen[node.id] = 1;
        input[node.id] = node.op_type == OpType::INPUT;
    }
    for (uint32_t id : source.graph_inputs)
        if (id >= count || !input[id]) return false;
    for (uint32_t id : source.graph_outputs) if (id >= count) return false;
    for (uint32_t id : private_inputs) {
        if (id >= count || !input[id] || private_input[id]) return false;
        private_input[id] = 1;
    }
    // Validate everything before touching the destination. No mapped file or
    // pool is copied; all weight/quantization sidecars remain source-owned.
    frame.nodes = source.nodes;
    frame.graph_inputs = source.graph_inputs;
    frame.graph_outputs = source.graph_outputs;
    frame.metadata = source.metadata;
    frame.runtime.tensors.resize(count);
    for (const auto& node : source.nodes) {
        if (node.op_type == OpType::CONSTANT || (input[node.id] && !private_input[node.id]))
            frame.runtime.tensors[node.id] = source.runtime.tensors[node.id];
        else
            frame.runtime.tensors[node.id] = Tensor::create(node.out_prec, MemoryType::NONE,
                node.out_shape[0], node.out_shape[1], node.out_shape[2], node.out_shape[3]);
    }
    return true;
}

static inline int64_t eval_dim(const DimExpr& e, int64_t out_shape_val,
                                int64_t seq_val, int64_t batch_val) {
    switch (e.kind) {
        case DIM_SEQ:   return seq_val;
        case DIM_MUL:   return (int64_t)e.coeff * seq_val;
        case DIM_ADD:   return (int64_t)e.coeff + seq_val;
        case DIM_BATCH: return batch_val;
        case DIM_DIV:
            if (e.coeff <= 0 || seq_val % e.coeff != 0) {
                std::fprintf(
                    stderr,
                    "execute_graph: sequence length %lld is not divisible "
                    "by dynamic shape divisor %d\n",
                    static_cast<long long>(seq_val), e.coeff);
                return 0;
            }
            return seq_val / e.coeff;
        case DIM_CONST:
        default:        return out_shape_val;
    }
}

// ---------------------------------------------------------------------------
// prepare_execution
// ---------------------------------------------------------------------------

void reset_profile_stats(ExecContext& ctx) {
    ctx.profile_stats.clear();
    if (!ctx.graph) return;

    const auto& nodes = ctx.graph->nodes;
    ctx.profile_stats.resize(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        ctx.profile_stats[i].node_id = nodes[i].id;
        ctx.profile_stats[i].op_type = nodes[i].op_type;
        ctx.profile_stats[i].calls = 0;
        ctx.profile_stats[i].total_ns = 0;
    }
}

void prepare_execution(ExecContext& ctx, bool prepare_backend_plan) {
    auto& nodes = ctx.graph->nodes;
    const size_t N = nodes.size();
    ctx.emulate_bf16_activations = std::any_of(
        nodes.begin(), nodes.end(),
        [](const GraphNode& node) {
            return node.op_type == OpType::DSV4_COMPRESSOR ||
                   node.op_type == OpType::DSV4_INDEXER ||
                   node.op_type == OpType::DSV4_SPARSE_ATTN;
        });
    ctx.release_queue.assign(N, {});
    ctx.resolved_inputs.assign(N, {});
    for (size_t i = 0; i < N; ++i) {
        auto& resolved = ctx.resolved_inputs[i];
        resolved.reserve(nodes[i].inputs.size());
        for (uint32_t input_id : nodes[i].inputs)
            resolved.push_back(&ctx.graph->runtime.tensors[input_id]);
    }
    ctx.sigmoid_gr_reduce_producers.assign(N, -1);
    ctx.sigmoid_gr_reduce_inputs.assign(N, {});
    ctx.sigmoid_gr_reduce_release_queue.assign(N, {});
    std::vector<size_t> consumers(ctx.graph->runtime.tensors.size(), 0);
    for (const auto& node : nodes)
        for (uint32_t id : node.inputs)
            if (id < consumers.size()) ++consumers[id];
    for (size_t i = 1; i < N; ++i) {
        const auto& sigmoid = nodes[i - 1];
        const auto& reduce = nodes[i];
        if (sigmoid.op_type != OpType::SIGMOID || sigmoid.inputs.size() != 1 ||
            sigmoid.out_prec != Precision::FP32 || reduce.op_type != OpType::GR_REDUCE ||
            reduce.inputs.size() != 2 || reduce.inputs[1] != sigmoid.id ||
            sigmoid.id >= consumers.size() || consumers[sigmoid.id] != 1 ||
            sigmoid.inputs[0] >= consumers.size() ||
            std::find(ctx.graph->graph_outputs.begin(), ctx.graph->graph_outputs.end(),
                sigmoid.id) != ctx.graph->graph_outputs.end()) continue;
        ctx.sigmoid_gr_reduce_producers[i] = static_cast<int32_t>(i - 1);
        ctx.sigmoid_gr_reduce_inputs[i] = {
            &ctx.graph->runtime.tensors[reduce.inputs[0]],
            &ctx.graph->runtime.tensors[sigmoid.inputs[0]]};
    }
    ctx.hc_up_reduce_producers.assign(N,-1);
    ctx.hc_up_reduce_consumers.assign(N,-1);
    ctx.hc_up_reduce_inputs.assign(N,{});
    ctx.hc_up_reduce_release_queue.assign(N,{});
    const HcUpGraphIndex hc_up_index(*ctx.graph);
    for(size_t c=2;c<N;++c){
        if(nodes[c].op_type!=OpType::GR_REDUCE)continue;
        const auto* up=hc_up_reduce_producer(*ctx.graph,nodes[c],hc_up_index);
        if(!up)continue;
        ctx.hc_up_reduce_producers[c]=static_cast<int32_t>(c-2);
        ctx.hc_up_reduce_consumers[c-2]=static_cast<int32_t>(c);
        ctx.hc_up_reduce_inputs[c]={&ctx.graph->runtime.tensors[nodes[c].inputs[0]],
            &ctx.graph->runtime.tensors[up->inputs[0]],&ctx.graph->runtime.tensors[up->inputs[1]]};
    }
    ctx.gr_inject_rms_producers.assign(N, -1);
    ctx.gr_inject_rms_consumers.assign(N, -1);
    ctx.gr_inject_rms_inputs.assign(N, {});
    ctx.gr_inject_rms_release_queue.assign(N, {});
    ctx.moe_gr_inject_rms_producers.assign(N,-1);
    ctx.moe_gr_inject_rms_consumers.assign(N,-1);
    ctx.moe_gr_inject_rms_release_queue.assign(N,{});
    for (size_t p = 0; p < N; ++p) {
        const auto& inject = nodes[p];
        if (inject.op_type != OpType::GR_INJECT || inject.inputs.size() != 3 ||
            inject.out_prec != Precision::FP32) continue;
        size_t c = p + 1;
        while (c < N && (nodes[c].op_type == OpType::INPUT ||
                         nodes[c].op_type == OpType::CONSTANT)) ++c;
        if (c == N) continue;
        const auto& rms = nodes[c];
        if (rms.op_type != OpType::GROUP_RMS_NORM || rms.inputs.size() != 2 ||
            rms.inputs[0] != inject.id || rms.out_prec != Precision::FP32 ||
            rms.inputs[1] >= consumers.size() ||
            std::any_of(inject.inputs.begin(), inject.inputs.end(),
                [&](uint32_t id) { return id >= consumers.size(); })) continue;
        ctx.gr_inject_rms_producers[c] = static_cast<int32_t>(p);
        ctx.gr_inject_rms_consumers[p] = static_cast<int32_t>(c);
        for (uint32_t id : inject.inputs)
            ctx.gr_inject_rms_inputs[c].push_back(&ctx.graph->runtime.tensors[id]);
        ctx.gr_inject_rms_inputs[c].push_back(&ctx.graph->runtime.tensors[rms.inputs[1]]);
        if(const auto* moe=hc_moe_inject_producer(*ctx.graph,inject,rms,hc_up_index)){
            const auto m=hc_up_index.position.at(moe->id);
            ctx.moe_gr_inject_rms_producers[c]=static_cast<int32_t>(m);
            ctx.moe_gr_inject_rms_consumers[m]=static_cast<int32_t>(c);
        }
    }
    reset_profile_stats(ctx);
    if (ctx.backend && prepare_backend_plan) ctx.backend->prepare_graph(*ctx.graph);

    // Stateful/cache-mutating graphs need a stronger storage dependency model:
    // SDPA updates KV cache, GDN/SHORTCONV update recurrent state, and their
    // inputs may alias persistent engine storage. Keep them on end-of-call
    // cleanup until TensorStorage/storage-id tracking replaces pointer-based
    // borrowed-view inference.
    std::vector<uint32_t> protected_stateful_nodes;
    for (const auto& node : nodes) {
        if (node.op_type == OpType::SDPA || node.op_type == OpType::SDPA_MLA ||
            node.op_type == OpType::GATED_DELTANET_PREFILL ||
            node.op_type == OpType::GATED_DELTANET_DECODE ||
            node.op_type == OpType::GATED_DELTANET_CONV_DECODE ||
            node.op_type == OpType::SHORTCONV ||
            node.op_type == OpType::RMS_NORM_ROPE ||
            node.op_type == OpType::QK_RMS_NORM_ROPE ||
            node.op_type == OpType::ADD_RMS_NORM ||
            node.op_type == OpType::DSV4_COMPRESSOR ||
            node.op_type == OpType::DSV4_INDEXER ||
            node.op_type == OpType::DSV4_SPARSE_ATTN ||
            node.op_type == OpType::PLE_LOOKUP ||
            node.op_type == OpType::PLE_DILATED_CONV) {
            if (!ctx.conservative_stateful_liveness) return;
            protected_stateful_nodes.push_back(node.id);
        }
    }

    uint32_t max_id = 0;
    for (const auto& node : nodes) max_id = std::max(max_id, node.id);
    const size_t M = (size_t)max_id + 1;

    std::vector<uint32_t> owner_root(M, 0);
    std::vector<int> node_index(M, -1);
    std::vector<int> last_direct_use(M, -1);
    std::vector<int> last_storage_use(M, -1);
    std::vector<uint8_t> keep(M, 0);

    auto is_view_like = [&ctx](OpType op) {
        // RESHAPE may materialize at runtime for non-contiguous inputs, but it
        // may also borrow. Treating it as view-like for owner propagation keeps
        // producers alive for the borrowed case; runtime release skips it when
        // it actually borrowed and releases it when it materialized.
        // CUDA CONTIGUOUS may alias an already-contiguous input. In the
        // diagnostic stateful planner conservatively propagate its owner
        // even when it later materializes; never free a borrowed producer.
        return op == OpType::RESHAPE || op == OpType::PERMUTE || op == OpType::SLICE ||
            (ctx.conservative_stateful_liveness && op == OpType::CONTIGUOUS);
    };

    for (size_t i = 0; i < N; i++) {
        const auto& node = nodes[i];
        if (node.id >= M) continue;
        node_index[node.id] = (int)i;

        if (is_view_like(node.op_type) && !node.inputs.empty() && node.inputs[0] < M) {
            owner_root[node.id] = owner_root[node.inputs[0]];
        } else {
            owner_root[node.id] = node.id;
        }
        if (owner_root[node.id] >= M) owner_root[node.id] = node.id;

        for (uint32_t inp_id : node.inputs) {
            if (inp_id >= M) continue;
            last_direct_use[inp_id] = (int)i;
            uint32_t root = owner_root[inp_id];
            if (root < M) last_storage_use[root] = std::max(last_storage_use[root], (int)i);
        }
    }

    // Graph outputs are returned to the caller. If the output is a borrowed
    // view, its storage owner must also stay alive past execute_graph().
    for (uint32_t out_id : ctx.graph->graph_outputs) {
        if (out_id >= M) continue;
        keep[out_id] = 1;
        uint32_t root = owner_root[out_id];
        if (root < M) keep[root] = 1;
    }

    // By default do not infer ownership for state/cache-mutating operators. Retain
    // their outputs AND input storage owners for the entire call, including
    // producers reached through view chains. This intentionally over-retains.
    // Diagnostic exception: explicit backend contracts classify mutable state
    // inputs and read-only operands/independent output. Unknown fails closed.
    for (uint32_t id : protected_stateful_nodes) {
        const auto& stateful=nodes[node_index[id]];
        const uint64_t mutable_inputs=ctx.stateful_readonly_last_use && ctx.backend
            ? ctx.backend->independent_stateful_mutable_inputs(stateful) : UINT64_MAX;
        if(mutable_inputs != UINT64_MAX && stateful.inputs.size() < 64 &&
           (mutable_inputs >> stateful.inputs.size()) == 0) {
            for (size_t index = 0; index < stateful.inputs.size() && index < 64; ++index) {
                if (!(mutable_inputs & (uint64_t(1) << index))) continue;
                const uint32_t input=stateful.inputs[index];
                if(input<M) {
                    keep[input]=1;
                    keep[owner_root[input]]=1;
                }
            }
            continue; // Other inputs and independent output use normal last-use.
        } // Missing/invalid contract fails closed to existing over-retention.
        keep[id] = 1;
        keep[owner_root[id]] = 1;
        for (uint32_t input : nodes[node_index[id]].inputs) {
            if (input >= M) continue;
            keep[input] = 1;
            keep[owner_root[input]] = 1;
        }
    }

    if (ctx.conservative_stateful_liveness) {
        // A possible view can also materialize (notably RESHAPE of a strided
        // SLICE/PERMUTE). Its descendants may therefore borrow intermediate
        // storage, not just owner_root. Propagate lifetime and retention along
        // EVERY immediate view edge, backwards through the topological graph.
        // Root-only propagation can free a materialized reshape while a later
        // permute still refers to it. Default execution policy stays unchanged.
        for (size_t i = N; i-- > 0;) {
            const auto& node = nodes[i];
            if (!is_view_like(node.op_type) || node.inputs.empty()) continue;
            const uint32_t parent = node.inputs[0];
            if (parent >= M) continue;
            last_storage_use[parent] = std::max(last_storage_use[parent],
                std::max(last_direct_use[node.id], last_storage_use[node.id]));
            if (keep[node.id]) keep[parent] = 1;
        }
    }

    for (const auto& node : nodes) {
        if (node.id >= M) continue;
        if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT) continue;
        if (keep[node.id]) continue;

        int rel_at = std::max(last_direct_use[node.id], last_storage_use[node.id]);
        if (rel_at < 0) rel_at = node_index[node.id];  // dead materialized output
        if (rel_at >= 0 && (size_t)rel_at < ctx.release_queue.size()) {
            ctx.release_queue[rel_at].push_back(node.id);
        }
    }
    for (size_t i = 1; i < N; ++i) {
        if (ctx.sigmoid_gr_reduce_producers[i] < 0) continue;
        auto& fused = ctx.sigmoid_gr_reduce_release_queue[i];
        fused = ctx.release_queue[i - 1];
        fused.insert(fused.end(), ctx.release_queue[i].begin(), ctx.release_queue[i].end());
        std::sort(fused.begin(), fused.end());
        fused.erase(std::unique(fused.begin(), fused.end()), fused.end());
    }
    for(size_t c=2;c<N;++c){
        const int32_t p=ctx.hc_up_reduce_producers[c];if(p<0)continue;
        auto& fused=ctx.hc_up_reduce_release_queue[c];
        for(size_t j=static_cast<size_t>(p);j<=c;++j)
            fused.insert(fused.end(),ctx.release_queue[j].begin(),ctx.release_queue[j].end());
        std::sort(fused.begin(),fused.end());
        fused.erase(std::unique(fused.begin(),fused.end()),fused.end());
    }
    for (size_t c = 0; c < N; ++c) {
        const int32_t p = ctx.gr_inject_rms_producers[c];
        if (p < 0) continue;
        auto& fused = ctx.gr_inject_rms_release_queue[c];
        for (size_t j = static_cast<size_t>(p); j <= c; ++j)
            fused.insert(fused.end(), ctx.release_queue[j].begin(), ctx.release_queue[j].end());
        std::sort(fused.begin(), fused.end());
        fused.erase(std::unique(fused.begin(), fused.end()), fused.end());
    }
    for(size_t c=0;c<N;++c){const int32_t m=ctx.moe_gr_inject_rms_producers[c];if(m<0)continue;
        auto& fused=ctx.moe_gr_inject_rms_release_queue[c];
        for(size_t j=static_cast<size_t>(m);j<=c;++j)
            fused.insert(fused.end(),ctx.release_queue[j].begin(),ctx.release_queue[j].end());
        std::sort(fused.begin(),fused.end());fused.erase(std::unique(fused.begin(),fused.end()),fused.end());}
}

// ---------------------------------------------------------------------------
// inject_runtime_shapes
//
// Fills INPUT tensors' actual shapes (substituting runtime_seq_len for any
// dim tagged DynamicKind::SEQ) and patches stateful op parameters with the
// number of real tokens. This keeps recurrent state unchanged by static
// padding in GDN, ShortConv, and PLE.
//
// Called once before execute_graph() by the engine. After this call:
//   - INPUT tensors have their runtime shape filled in
//   - Downstream tensors are filled during execute_graph() dispatch
//   - stateful op params are set so kernels can skip padding positions
//
// In DYNAMIC mode (static_padded=false): runtime_seq_len is the actual token
// count and the graph executes with that seq_len — no padding.
// In STATIC_PADDED mode (static_padded=true): the graph executes with
// padded_seq_len, but stateful ops still need to know runtime_seq_len (the
// real token count) to skip padding positions.
// ---------------------------------------------------------------------------

void inject_runtime_shapes(ExecContext& ctx) {
    auto& nodes  = ctx.graph->nodes;
    auto& tensors = ctx.graph->runtime.tensors;
    const int seq_val = ctx.static_padded ? ctx.padded_seq_len : ctx.runtime_seq_len;
    const int n_real  = ctx.runtime_seq_len;  // always the real token count

    // Phase 1: INPUT tensors' actual shape.
    // Engine fills INPUT tensor data + shape before calling execute_graph;
    // for dynamic dims we override the shape[dim] with the evaluated DimExpr.
    for (auto& n : nodes) {
        if (n.op_type != OpType::INPUT) continue;
        auto& t = tensors[n.id];
        for (int d = 0; d < 4; d++) {
            if (n.dim_expr[d].is_dynamic()) {
                t.shape[d] = eval_dim(n.dim_expr[d], n.out_shape[d], seq_val, n_real);
            }
        }
        t.compute_strides();
    }

    // Phase 2: patch stateful op n_real_tokens + seq_len.
    // In STATIC_PADDED mode (current): graph runs at graph_seq_len (build-time
    // seq_len = 256). Stateful kernels read seq_len from params.i32[3] (GDN)
    // and need it to stay at build-time value so they iterate the full padded
    // range, but skip padding positions via n_real_tokens (params.i32[6]).
    // In DYNAMIC mode (future): runtime_seq_len replaces both seq_len and
    // n_real_tokens in params, kernels iterate only the real tokens.
    if (n_real <= 0) return;
    const bool patch_seq_len = !ctx.static_padded;  // DYNAMIC mode only
    for (auto& n : nodes) {
        if (n.op_type == OpType::RWKV7) {
            if(n.params.i32.size()<=3) n.params.i32.resize(4,0);
            n.params.i32[3]=n_real;
            if(patch_seq_len && n.params.i32.size()>2) n.params.i32[2]=n_real;
        } else if (n.op_type == OpType::RWKV_TOKEN_SHIFT) {
            if(n.params.i32.size()<=2) n.params.i32.resize(3,0);
            n.params.i32[2]=n_real;
            if(patch_seq_len && n.params.i32.size()>1) n.params.i32[1]=n_real;
        } else if (n.op_type == OpType::GATED_DELTANET_PREFILL) {
            if (n.params.i32.size() <= 6) n.params.i32.resize(7, 0);
            n.params.i32[6] = n_real;  // n_real_tokens (skip padding positions)
            if (patch_seq_len && n.params.i32.size() > 3) {
                n.params.i32[3] = n_real;  // seq_len = n_real in DYNAMIC mode
            }
        } else if (n.op_type == OpType::SHORTCONV) {
            if (n.params.i32.size() <= 1) n.params.i32.resize(2, 0);
            n.params.i32[1] = n_real;
            // SHORTCONV reads seq_len from input tensor shape (x.shape[1]),
            // which is set by the engine to graph_seq_len (padding mode) or
            // n_real (dynamic mode).
        } else if (n.op_type == OpType::PLE_LOOKUP) {
            if (n.params.i32.size() <= 7) n.params.i32.resize(8, 0);
            n.params.i32[7] = n_real;
        } else if (n.op_type == OpType::PLE_DILATED_CONV) {
            if (n.params.i32.size() <= 2) n.params.i32.resize(3, 0);
            n.params.i32[2] = n_real;
        }
    }
}


// ---------------------------------------------------------------------------
// execute_graph — pure executor, no shape inference
//
// With the multi-graph architecture, all shapes are static at graph-build
// time.  The prefill graph is built with seq_len=N, the decode graph with
// seq_len=1.  The only runtime-dynamic element is the KV cache, which uses
// embedded CacheMetadata for past_len tracking.
// ---------------------------------------------------------------------------

static void execute_graph_impl(ExecContext& ctx, int stop_after_node_index,
                               int first_node_index, bool segmented, int materialized_moe = -1) {
    auto& nodes  = ctx.graph->nodes;
    auto& tensors = ctx.graph->runtime.tensors;
    auto* pool   = ctx.pool;
    ctx.execution_failed = false;
    if (!ctx.backend) {
        std::fprintf(stderr, "execute_graph: no backend configured\n");
        ctx.execution_failed = true;
        return;
    }
    ctx.backend->clear_dispatch_error();
    ctx.backend->begin_execution();
    if (ctx.moe_backend && ctx.moe_backend != ctx.backend)
        ctx.moe_backend->begin_execution();
    // Device-resident backends (Metal) keep intermediates in device buffers,
    // so borrowed-view detection can't rely on host-pointer equality; classify
    // views by op type instead, and skip the host owner-id assertions.
    const bool device_resident = ctx.backend && ctx.backend->is_device_resident();

    // Compute seq_val once for dynamic shape injection.
    // In DYNAMIC mode this = runtime_seq_len; in STATIC_PADDED mode = padded_seq_len.
    const int seq_val = ctx.static_padded ? ctx.padded_seq_len : ctx.runtime_seq_len;
    const bool has_dynamic = (seq_val > 0);
    if (ctx.conservative_stateful_liveness && ctx.conservative_liveness_seq_limit > 0 &&
        seq_val > ctx.conservative_liveness_seq_limit) {
        std::fprintf(stderr, "execute_graph: conservative liveness diagnostic sequence cap exceeded\n");
        ctx.execution_failed = true;
        return;
    }
    const bool fusion_range = (!segmented && stop_after_node_index < 0) ||
        (segmented && ctx.segment_fusions_enabled);
    auto within_segment = [&](int consumer) {
        return !segmented || consumer <= stop_after_node_index;
    };
    const bool gate_fusion_enabled = ctx.backend->supports_sigmoid_gr_reduce_fusion() &&
        fusion_range && !ctx.profile_enabled && !ctx.emulate_bf16_activations &&
        !mollm_trace::enabled() && !getenv("MOLLM_DUMP_NODES") &&
        ctx.sigmoid_gr_reduce_producers.size() == nodes.size();
    int32_t pending_sigmoid = -1;
    const bool hc_up_enabled=ctx.backend->supports_hc_up_reduce_fusion()&&
        fusion_range&&!ctx.profile_enabled&&!ctx.emulate_bf16_activations&&
        !mollm_trace::enabled()&&!getenv("MOLLM_DUMP_NODES")&&
        ctx.hc_up_reduce_consumers.size()==nodes.size();
    int32_t pending_up=-1;
    const bool inject_rms_enabled = ctx.backend->supports_gr_inject_rms_fusion() &&
        fusion_range && !ctx.profile_enabled && !ctx.emulate_bf16_activations &&
        !mollm_trace::enabled() && !getenv("MOLLM_DUMP_NODES") &&
        ctx.gr_inject_rms_consumers.size() == nodes.size();
    int32_t pending_inject = -1;
    const bool moe_inject_enabled=inject_rms_enabled&&!segmented&&
        ctx.backend->supports_moe_gr_inject_rms_fusion()&&
        ctx.moe_gr_inject_rms_consumers.size()==nodes.size();
    int32_t pending_moe=-1;
    const bool same_shape_workspace =
        ctx.reuse_same_shape_workspace &&
        ctx.workspace_shape_valid &&
        ctx.workspace_runtime_seq_len == ctx.runtime_seq_len &&
        ctx.workspace_runtime_batch == ctx.runtime_batch &&
        ctx.workspace_static_padded == ctx.static_padded &&
        ctx.workspace_padded_seq_len == ctx.padded_seq_len;
    const bool reuse_workspace =
        (ctx.reuse_static_workspace && !has_dynamic) ||
        (has_dynamic && same_shape_workspace);

    const std::vector<uint32_t> empty_release_batch;
    if (first_node_index == 0) {
    // Reset non-INPUT/CONSTANT tensor state before execution. Borrowed views
    // must always be cleared because their source storage may be overwritten
    // or released between calls. Materialized tensors are released by default;
    // fully static graphs may opt into keeping them as reusable workspace.
    //
    // IMPORTANT: only release buffers owned by materialized ops. View ops
    // (zero-copy RESHAPE/PERMUTE/SLICE) borrow data from their inputs;
    // releasing their borrowed pointer would double-free the real owner.
    // INPUT/CONSTANT tensors keep their data (set by engine / load-time,
    // e.g. cache_k/cache_v which are mmap'd or load-time allocated).
    auto is_always_borrowed_view_op = [](OpType op) {
        return op == OpType::PERMUTE || op == OpType::SLICE;
    };
    // First classify borrowed views while all producer pointers are still
    // intact. View chains are common; mutating an earlier view before
    // classifying a later one would lose the pointer equality evidence.
    std::vector<uint8_t> borrowed_view(nodes.size(), 0);
    for (size_t i = 0; i < nodes.size(); i++) {
        auto& node = nodes[i];
        if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT) continue;
        auto& t = tensors[node.id];

        bool borrowed = is_always_borrowed_view_op(node.op_type);
        if (node.op_type == OpType::RESHAPE && !node.inputs.empty()) {
            const Tensor& src = tensors[node.inputs[0]];
            // On a device backend the output host pointer may be a device alias
            // (or null), so host pointer equality is unreliable.
            // Match dispatch's exact layout predicate, including opted-in
            // strided views; classify before clearing any producer metadata.
            borrowed = device_resident ? device_reshape_borrows(ctx.backend, src, t)
                                       : t.shares_storage_with(src);
        } else if (node.op_type == OpType::CONTIGUOUS &&
                   !node.inputs.empty()) {
            const Tensor& src = tensors[node.inputs[0]];
            borrowed = device_resident && src.is_contiguous();
        }
        borrowed_view[node.id] = borrowed ? 1 : 0;
    }

    // Clear borrowed views without releasing their borrowed pointer.
    for (size_t i = 0; i < nodes.size(); i++) {
        auto& node = nodes[i];
        if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT) continue;
        if (!borrowed_view[node.id]) continue;
        auto& t = tensors[node.id];
        t.data = nullptr;
        t.mem_type = MemoryType::NONE;
        t.owner_id = 0;
        t.storage_id = 0;
    }

    if (!reuse_workspace) {
        // Then release materialized outputs. Borrowed views were nulled above,
        // so RESHAPE only reaches this path when it actually materialized a
        // buffer.
        for (size_t i = 0; i < nodes.size(); i++) {
            auto& node = nodes[i];
            if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT) continue;
            if (borrowed_view[node.id]) continue;
            auto& t = tensors[node.id];
            if (t.data && t.mem_type == MemoryType::POOLED && t.nbytes() > 0) {
                if (!device_resident && t.owner_id != 0 && t.owner_id != pool->id()) {
                    std::fprintf(stderr,
                                 "execute_graph: owner mismatch before release for node %u (%p owner=%u pool=%u)\n",
                                 node.id, t.data, t.owner_id, pool->id());
                    assert(false && "execute_graph owner mismatch");
                    ctx.execution_failed = true;
                    return;
                }
                ctx.backend->free_output(t, pool);
            }
            t.data = nullptr;
            t.device.buffer = nullptr;
            t.device.offset = 0;
            t.mem_type = MemoryType::NONE;
            t.owner_id = 0;
            t.storage_id = 0;
        }
    }

    // The first hash-routed layer does not need an activation-based
    // prediction: its exact expert set is known from the decode token before
    // the graph starts. Queue it here; later hash layers are already covered
    // by the ordinary one-layer-ahead scheduler, and eagerly queueing all of
    // them would delay the layer-0 reads that are on the critical path.
    if (ctx.moe_hash_cross_layer_prefetch) {
        for (const GraphNode& moe : nodes) {
            if (moe.op_type != OpType::MOE || moe.inputs.size() < 4)
                continue;
            const auto& moe_inputs = moe.inputs;
            const int token_ids_input =
                graph_params::get_i32(moe.params, 12, -1);
            const int hash_table_input =
                graph_params::get_i32(moe.params, 13, -1);
            if (token_ids_input >= 0 && hash_table_input >= 0 &&
                static_cast<size_t>(token_ids_input) < moe_inputs.size() &&
                static_cast<size_t>(hash_table_input) < moe_inputs.size()) {
                const Tensor& gate = tensors[moe_inputs[2]];
                const Tensor& down = tensors[moe_inputs[3]];
                schedule_moe_hash_cross_layer_prefetch(
                    tensors[moe_inputs[token_ids_input]],
                    tensors[moe_inputs[hash_table_input]],
                    dynamic_cast<const MoeSsdTensorSource*>(
                        gate.moe_ssd_source),
                    dynamic_cast<const MoeSsdTensorSource*>(
                        down.moe_ssd_source),
                    graph_params::get_i32(moe.params, 1, 0),
                    graph_params::get_i32(moe.params, 2, 0),
                    true);
                break;
            }
        }
    }

    } // Only a fresh pass clears tensors and schedules initial prefetch.

    for (size_t i = static_cast<size_t>(first_node_index); i < nodes.size(); i++) {
        auto& node = nodes[i];

        // skip inputs / constants — data is pre-set by the caller
        if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT) {
            if (stop_after_node_index >= 0 && static_cast<int>(i) >= stop_after_node_index)
                break;
            continue;
        }

        auto& out = tensors[node.id];
        auto start_time = std::chrono::steady_clock::time_point();
        if (ctx.profile_enabled && node.id < ctx.profile_stats.size()) {
            start_time = std::chrono::steady_clock::now();
        }

        // initialise output shape from node definition.
        // Dynamic dims evaluated against runtime seq_val via DimExpr.
        //
        // For zero-copy ops (RESHAPE/PERMUTE/SLICE) we must NOT skip this step
        // entirely in DYNAMIC mode: the dispatch reads output->shape[d] for the
        // dynamic dims, and skipping would leave stale values from the previous
        // execute_graph() call. Only skip when the graph is fully STATIC
        // (has_dynamic == false) — then shapes are constants and reuse is safe.
        //
        // When out.data is already set (zero-copy view inheriting from input),
        // we still must update shape[d] for dynamic dims, but we must NOT call
        // compute_strides() unconditionally — the dispatch path will handle
        // strides itself (see RESHAPE dispatch).
        bool skip_init = (node.op_type == OpType::RESHAPE ||
                          node.op_type == OpType::PERMUTE ||
                          node.op_type == OpType::SLICE)
                         && out.data != nullptr
                         && !has_dynamic;
        if (!skip_init) {
            // Always update dynamic dims via eval_dim, even for zero-copy views.
            // For static dims, fall back to build-time literal.
            bool any_dynamic = false;
            for (int d = 0; d < 4; d++) {
                if (has_dynamic && node.dim_expr[d].is_dynamic()) {
                    out.shape[d] = eval_dim(node.dim_expr[d], node.out_shape[d],
                                            seq_val, ctx.runtime_batch);
                    any_dynamic = true;
                } else {
                    out.shape[d] = node.out_shape[d];
                }
            }
            out.prec     = node.out_prec;
            // For zero-copy views with dynamic dims, defer stride computation
            // to dispatch (which knows whether to inherit from src or
            // materialise). Computing strides here with the new (correct)
            // shape would overwrite the inherited contiguous layout.
            if (!(any_dynamic && out.data != nullptr &&
                  (node.op_type == OpType::RESHAPE ||
                   node.op_type == OpType::PERMUTE ||
                   node.op_type == OpType::SLICE))) {
                out.compute_strides();
            }
        }

        // collect inputs before allocation so zero-copy views can avoid
        // acquiring a buffer they will immediately overwrite with borrowed data.
        const std::vector<const Tensor*>& inputs = ctx.resolved_inputs[i];
        const bool fused_consumer = pending_sigmoid >= 0 &&
            i == static_cast<size_t>(pending_sigmoid) + 1;
        const bool hc_up_consumer=pending_up>=0&&ctx.hc_up_reduce_producers[i]==pending_up;
        const bool defer_up=hc_up_enabled&&ctx.hc_up_reduce_consumers[i]>=0&&
            within_segment(ctx.hc_up_reduce_consumers[i])&&
            out.shape[1]>=16&&out.shape[1]<=ctx.backend->hc_up_reduce_max_rows();
        const bool defer_sigmoid = (pending_up>=0&&i==size_t(pending_up)+1)||
            (gate_fusion_enabled && i + 1 < nodes.size() &&
            within_segment(int(i + 1)) &&
            ctx.sigmoid_gr_reduce_producers[i + 1] == static_cast<int32_t>(i) &&
            out.shape[1] > 1);
        const bool inject_rms_consumer = pending_inject >= 0 &&
            ctx.gr_inject_rms_producers[i] == pending_inject;
        const bool defer_inject = inject_rms_enabled &&
            ctx.gr_inject_rms_consumers[i] >= 0 &&
            within_segment(ctx.gr_inject_rms_consumers[i]) && out.shape[1] > 1;
        const bool defer_moe=moe_inject_enabled&&ctx.moe_gr_inject_rms_consumers[i]>=0&&
            within_segment(ctx.moe_gr_inject_rms_consumers[i])&&out.shape[1]>=16&&out.shape[1]<=2048;

        // Static device graphs rebuild hundreds of zero-copy projection views
        // on every decode token. Handle the metadata-only cases directly in
        // the executor instead of crossing the virtual backend boundary. A
        // non-contiguous RESHAPE still goes through the backend because it
        // needs a materialization kernel.
        bool inline_zero_copy_view = false;
        if (device_resident && !has_dynamic && inputs.size() == 1 &&
            inputs[0]) {
            const Tensor& src = *inputs[0];
            if (node.op_type == OpType::RESHAPE && src.is_contiguous()) {
                const int64_t shape[4] = {
                    out.shape[0], out.shape[1], out.shape[2], out.shape[3]
                };
                out = src;
                for (int d = 0; d < 4; ++d) out.shape[d] = shape[d];
                out.compute_strides();
                out.device.buffer = src.device.buffer;
                out.device.offset = src.device.offset;
                inline_zero_copy_view = true;
            } else if (node.op_type == OpType::PERMUTE) {
                const int axis[4] = {
                    graph_params::get_i32(node.params, 0, 0),
                    graph_params::get_i32(node.params, 1, 1),
                    graph_params::get_i32(node.params, 2, 2),
                    graph_params::get_i32(node.params, 3, 3),
                };
                Tensor view = src;
                for (int d = 0; d < 4; ++d) {
                    view.shape[axis[d]] = src.shape[d];
                    view.stride[axis[d]] = src.stride[d];
                }
                out = view;
                out.device.buffer = src.device.buffer;
                out.device.offset = src.device.offset;
                inline_zero_copy_view = true;
            } else if (node.op_type == OpType::SLICE) {
                const int dim =
                    graph_params::get_i32(node.params, 0, 0);
                const int offset =
                    graph_params::get_i32(node.params, 1, 0);
                const int size = graph_params::get_i32(
                    node.params, 2, (int)src.shape[dim]);
                out = src;
                out.device.buffer = src.device.buffer;
                const size_t byte_offset =
                    (size_t)offset * src.stride[dim];
                if (src.data)
                    out.data = static_cast<uint8_t*>(src.data) + byte_offset;
                out.device.offset = src.device.offset + byte_offset;
                out.shape[dim] = size;
                inline_zero_copy_view = true;
            } else if (node.op_type == OpType::CONTIGUOUS &&
                       src.is_contiguous()) {
                const int64_t shape[4] = {
                    out.shape[0], out.shape[1],
                    out.shape[2], out.shape[3],
                };
                out = src;
                for (int d = 0; d < 4; ++d) out.shape[d] = shape[d];
                out.compute_strides();
                out.device.buffer = src.device.buffer;
                out.device.offset = src.device.offset;
                inline_zero_copy_view = true;
            }
        }

        // allocate output if needed
        if (out.data == nullptr && !defer_sigmoid && !defer_up) {
            size_t nbytes = out.nbytes();
            bool needs_allocation = true;
            if (node.op_type == OpType::PERMUTE || node.op_type == OpType::SLICE) {
                needs_allocation = false;
            } else if (node.op_type == OpType::RESHAPE) {
                needs_allocation = !(inputs.size() >= 1 && inputs[0] &&
                    (inputs[0]->is_contiguous() || (device_resident &&
                     device_reshape_borrows(ctx.backend, *inputs[0], out))));
            } else if (node.op_type == OpType::CONTIGUOUS) {
                needs_allocation =
                    !(device_resident && !has_dynamic &&
                      inputs.size() >= 1 && inputs[0] &&
                      inputs[0]->is_contiguous());
            }
            if (nbytes > 0) {
                if (needs_allocation) {
                    // alloc_output() sets out.data/mem_type/owner_id/storage_id.
                    // Default (CPU) impl is the old host BufferPool path;
                    // a device backend allocates an MTLBuffer and records it in
                    // out.device.buffer.
                    void* buf = ctx.backend->alloc_output(out, nbytes, pool);
                    if (!buf) {
                        fprintf(stderr, "execute: pool acquire failed for node %u (%zu bytes)\n",
                                node.id, nbytes);
                        ctx.execution_failed = true;
                        return;
                    }
                }
            }
        }

        // Fate-style cross-layer gate prediction. At the beginning of MoE
        // layer i, the gate input is already available. Feed its copied decode
        // vector to the real router from the next MoE layer on an idle SSD
        // worker, so its speculative reads can overlap this layer and the next
        // attention block. The next layer always recomputes its exact route.
        if ((ctx.moe_cross_layer_prefetch ||
             ctx.moe_hash_cross_layer_prefetch) &&
            node.op_type == OpType::MOE && inputs.size() >= 4 && inputs[0] &&
            inputs[0]->shape[1] == 1) {
            if (device_resident) {
                ctx.backend->synchronize_for_host_read();
                if (ctx.backend->dispatch_failed()) {
                    ctx.execution_failed = true;
                    return;
                }
            }
            const auto next_it = std::find_if(nodes.begin() + static_cast<ptrdiff_t>(i + 1),
                                              nodes.end(),
                                              [](const auto& candidate) {
                                                  return candidate.op_type == OpType::MOE;
                                              });
            if (next_it != nodes.end() && next_it->inputs.size() >= 4) {
                const auto& next_inputs = next_it->inputs;
                const Tensor& next_router = tensors[next_inputs[1]];
                const Tensor& next_gate = tensors[next_inputs[2]];
                const Tensor& next_down = tensors[next_inputs[3]];
                const int next_bias_input =
                    graph_params::get_i32(next_it->params, 11, -1);
                const Tensor* next_bias =
                    next_bias_input >= 0 &&
                            static_cast<size_t>(next_bias_input) <
                                next_inputs.size()
                        ? &tensors[next_inputs[next_bias_input]]
                        : nullptr;
                const int num_experts =
                    graph_params::get_i32(next_it->params, 1, 0);
                const int top_k =
                    graph_params::get_i32(next_it->params, 2, 0);
                const int token_ids_input =
                    graph_params::get_i32(next_it->params, 12, -1);
                const int hash_table_input =
                    graph_params::get_i32(next_it->params, 13, -1);
                const bool exact_hash_route =
                    ctx.moe_hash_cross_layer_prefetch &&
                    token_ids_input >= 0 && hash_table_input >= 0 &&
                    static_cast<size_t>(token_ids_input) <
                        next_inputs.size() &&
                    static_cast<size_t>(hash_table_input) <
                        next_inputs.size();
                if (exact_hash_route) {
                    schedule_moe_hash_cross_layer_prefetch(
                        tensors[next_inputs[token_ids_input]],
                        tensors[next_inputs[hash_table_input]],
                        dynamic_cast<const MoeSsdTensorSource*>(
                            next_gate.moe_ssd_source),
                        dynamic_cast<const MoeSsdTensorSource*>(
                            next_down.moe_ssd_source),
                        num_experts, top_k);
                } else if (ctx.moe_cross_layer_prefetch) {
                    MoeSsdPredictConfig config;
                    config.hidden_size =
                        graph_params::get_i32(next_it->params, 0, 0);
                    config.num_experts = num_experts;
                    config.top_k = top_k;
                    config.router_score_func =
                        graph_params::get_i32(next_it->params, 5, 0);
                    config.n_group =
                        graph_params::get_i32(next_it->params, 8, 1);
                    config.topk_group =
                        graph_params::get_i32(next_it->params, 9, 1);
                    Tensor prediction_input = *inputs[0];
                    std::vector<uint32_t> prediction_storage;
                    if (device_resident) {
                        const size_t bytes = inputs[0]->nbytes();
                        prediction_storage.resize(
                            (bytes + sizeof(uint32_t) - 1) /
                            sizeof(uint32_t));
                        if (!ctx.backend->copy_to_host(
                                *inputs[0], prediction_storage.data(),
                                bytes)) {
                            ctx.execution_failed = true;
                            return;
                        }
                        prediction_input.data = prediction_storage.data();
                        prediction_input.device.buffer = nullptr;
                        prediction_input.device.offset = 0;
                    }
                    schedule_moe_cross_layer_prefetch(
                        prediction_input, next_router, next_bias,
                        dynamic_cast<const MoeSsdTensorSource*>(
                            next_gate.moe_ssd_source),
                        dynamic_cast<const MoeSsdTensorSource*>(
                            next_down.moe_ssd_source),
                        config);
                }
            }
        }

        // Dispatch is the useful unit in a trace: it captures one graph op
        // (matmul, attention, MoE, norm, ...) without recording allocator and
        // liveness bookkeeping as fake compute work.
        const uint64_t trace_start = mollm_trace::now_ns();
        if (static_cast<int>(i) != materialized_moe &&
            !inline_zero_copy_view && !defer_sigmoid && !defer_up && !defer_inject && !defer_moe) {
            bool delegated = false;
            if(hc_up_consumer){
                delegated=ctx.backend->dispatch_hc_up_reduce(nodes[pending_up],node,
                    ctx.hc_up_reduce_inputs[i],&out,ctx.thread_pool);
                if(!delegated&&!ctx.backend->dispatch_failed()){
                    auto& up=tensors[nodes[pending_up].id];
                    if(!up.data&&up.nbytes()>0&&!ctx.backend->alloc_output(up,up.nbytes(),pool)){
                        ctx.execution_failed=true;return;
                    }
                    ctx.backend->dispatch(nodes[pending_up],ctx.resolved_inputs[pending_up],&up,ctx.thread_pool);
                }
            }
            if (inject_rms_consumer) {
                auto& carry = tensors[nodes[pending_inject].id];
                if(pending_moe>=0){
                    auto& branch=tensors[nodes[pending_moe].id];
                    delegated=ctx.backend->dispatch_moe_gr_inject_rms(nodes[pending_moe],nodes[pending_inject],node,
                        ctx.resolved_inputs[pending_moe],ctx.gr_inject_rms_inputs[i],&branch,&carry,&out,ctx.thread_pool);
                    if(!delegated&&!ctx.backend->dispatch_failed())
                        ctx.backend->dispatch(nodes[pending_moe],ctx.resolved_inputs[pending_moe],&branch,ctx.thread_pool);
                }
                if(!delegated&&!ctx.backend->dispatch_failed())delegated = ctx.backend->dispatch_gr_inject_rms(
                    nodes[pending_inject], node, ctx.gr_inject_rms_inputs[i],
                    &carry, &out, ctx.thread_pool);
                if (!delegated && !ctx.backend->dispatch_failed())
                    ctx.backend->dispatch(nodes[pending_inject],
                        ctx.resolved_inputs[pending_inject], &carry, ctx.thread_pool);
            }
            if (fused_consumer&&!delegated&&!ctx.backend->dispatch_failed()) {
                delegated = ctx.backend->dispatch_sigmoid_gr_reduce(
                    node, ctx.sigmoid_gr_reduce_inputs[i], &out, ctx.thread_pool);
                if (!delegated && !ctx.backend->dispatch_failed()) {
                    // A backend may reject a dynamic shape. Materialize the
                    // deferred gate before ordinary consumer dispatch.
                    auto& gate = tensors[nodes[pending_sigmoid].id];
                    if (!gate.data && gate.nbytes() > 0 &&
                        !ctx.backend->alloc_output(gate, gate.nbytes(), pool)) {
                        std::fprintf(stderr, "execute: deferred sigmoid allocation failed\n");
                        ctx.execution_failed = true;
                        return;
                    }
                    ctx.backend->dispatch(nodes[pending_sigmoid],
                        ctx.resolved_inputs[pending_sigmoid], &gate, ctx.thread_pool);
                }
            }
            if (ctx.moe_backend && node.op_type == OpType::MOE &&
                out.shape[1] == 1) {
                bool success = false;
                delegated = ctx.moe_backend->dispatch_host_moe(
                    node, inputs, &out, ctx.thread_pool, success);
                if (delegated && !success) {
                    std::fprintf(
                        stderr,
                        "execute: routed-MoE accelerator failed at node %u\n",
                        node.id);
                    ctx.execution_failed = true;
                    return;
                }
            }
            if (!delegated && !ctx.backend->dispatch_failed())
                ctx.backend->dispatch(node, inputs, &out, ctx.thread_pool);
        }
        if (ctx.backend->dispatch_failed()) {
            std::fprintf(
                stderr, "execute: backend rejected node %u (%s)\n",
                node.id, op_type_name(node.op_type));
            ctx.execution_failed = true;
            return;
        }
        if (ctx.emulate_bf16_activations) {
            bool round_output =
                node.op_type == OpType::RMS_NORM ||
                node.op_type == OpType::MOE ||
                node.op_type == OpType::HC_POST ||
                node.op_type == OpType::HC_HEAD ||
                node.op_type == OpType::DSV4_SPARSE_ATTN ||
                node.op_type == OpType::DSV4_GROUPED_LINEAR;
            if (node.op_type == OpType::MATMUL && inputs.size() >= 2 &&
                inputs[1]) {
                round_output =
                    inputs[1]->prec == Precision::FP8_E4M3 ||
                    inputs[1]->prec == Precision::MXFP4;
            }
            if (round_output && !ctx.backend->round_to_bf16(out)) {
                std::fprintf(
                    stderr, "execute: backend failed BF16 boundary for "
                    "node %u (%s)\n",
                    node.id, op_type_name(node.op_type));
                ctx.execution_failed = true;
                return;
            }
        }
        if (trace_start != 0) {
            const std::string args =
                "{\"graph\":\"" + std::string(ctx.trace_label ? ctx.trace_label : "graph") +
                "\",\"node\":" + std::to_string(node.id) +
                ",\"shape\":[" + std::to_string(out.shape[0]) + "," +
                std::to_string(out.shape[1]) + "," + std::to_string(out.shape[2]) + "," +
                std::to_string(out.shape[3]) + "]}";
            mollm_trace::record_duration("graph", op_type_name(node.op_type), trace_start,
                                         mollm_trace::now_ns(), args);
        }

        // Node dumping is an opt-in diagnostic.  Do not probe the process
        // environment for every graph node in normal inference.
        static const bool dump_nodes_enabled = getenv("MOLLM_DUMP_NODES") != nullptr;
        static const bool dump_first_row = getenv("MOLLM_DUMP_FIRST_ROW") != nullptr;
        static const bool dump_node_hash = getenv("MOLLM_DUMP_NODE_HASH") != nullptr;
        static const int dump_row_offset = [] {
            const char* value = getenv("MOLLM_DUMP_ROW_OFFSET");
            if (!value) return 0;
            char* end = nullptr;
            long parsed = std::strtol(value, &end, 10);
            return end != value && !*end && parsed >= 0 && parsed <= 65535
                ? static_cast<int>(parsed) : 0;
        }();
        static const int dump_first_row_max_seq = [] {
            const char* value = getenv("MOLLM_DUMP_FIRST_ROW_MAX_SEQ");
            if (!value) return 4;
            char* end = nullptr;
            long parsed = std::strtol(value, &end, 10);
            return end != value && !*end && parsed > 0 && parsed <= 65535 ? static_cast<int>(parsed) : 4;
        }();
        static const int dump_node_limit = [] {
            const char* value = getenv("MOLLM_DUMP_NODE_LIMIT");
            if (!value) return -1;
            char* end = nullptr;
            long parsed = std::strtol(value, &end, 10);
            return end != value && !*end && parsed >= 0 && parsed <= 1000000 ? static_cast<int>(parsed) : -1;
        }();
        if (dump_nodes_enabled && (dump_node_limit < 0 || node.id <= dump_node_limit) &&
            (!dump_first_row || ctx.runtime_seq_len <= dump_first_row_max_seq) &&
            (!dump_first_row || dump_row_offset < out.shape[1]) &&
            out.data && out.prec == Precision::FP32) {
            std::vector<float> host_dump;
            const float* d = out.ptr<float>();
            if (device_resident) {
                if (!out.is_contiguous()) {
                    std::fprintf(
                        stderr, "NODE %u op=%d non-contiguous device "
                        "output omitted\n",
                        node.id, static_cast<int>(node.op_type));
                    d = nullptr;
                } else {
                    host_dump.resize(
                        static_cast<size_t>(dump_first_row && node.op_type != OpType::SHORTCONV
                            ? out.shape[0] : out.nelements()));
                    if (!ctx.backend->copy_to_host(
                            out, host_dump.data(),
                            host_dump.size() * sizeof(float),
                            dump_first_row && node.op_type != OpType::SHORTCONV
                                ? size_t(dump_row_offset) * out.stride[1] : 0)) {
                        ctx.execution_failed = true;
                        return;
                    }
                    d = host_dump.data();
                }
            }
            if (d && dump_first_row &&
                (!device_resident || node.op_type == OpType::SHORTCONV))
                d += node.op_type == OpType::SHORTCONV ? dump_row_offset
                    : size_t(dump_row_offset) * out.stride[1] / sizeof(float);
            if (d) {
                double sum = 0.0, sum_sq = 0.0;
                float max_abs = 0.0f;
                uint64_t hash = 14695981039346656037ull;
                // Compare one selected token of the first outer slice.
                // ShortConv has a special dim-major physical layout.
                // Other outer slices and omitted strided views are not covered.
                const int64_t dump_count = dump_first_row ? out.shape[0] : out.nelements();
                for (int64_t j = 0; j < dump_count; ++j) {
                    const int64_t index = dump_first_row && node.op_type == OpType::SHORTCONV
                        ? j * out.shape[1] : j;
                    sum += d[index];
                    sum_sq += (double)d[index] * d[index];
                    max_abs = std::max(max_abs, std::fabs(d[index]));
                    uint32_t bits;
                    std::memcpy(&bits, d + index, sizeof(bits));
                    for (int byte = 0; byte < 4; ++byte) {
                        hash ^= (bits >> (byte * 8)) & 255;
                        hash *= 1099511628211ull;
                    }
                }
                const int64_t sample_stride = dump_first_row && node.op_type == OpType::SHORTCONV
                    ? out.shape[1] : 1;
                fprintf(stderr, "NODE %u op=%d shape=%lld,%lld,%lld,%lld  "
                        "%.5f %.5f %.5f sum=%.9g sq=%.9g max=%.9g\n",
                        node.id, (int)node.op_type,
                        (long long)out.shape[0], (long long)out.shape[1],
                        (long long)out.shape[2], (long long)out.shape[3],
                         d[0], dump_count>1?d[sample_stride]:0,
                         dump_count>2?d[2*sample_stride]:0, sum, sum_sq, max_abs);
                if (dump_first_row)
                    fprintf(stderr, "ROW graph=%s node=%u op=%s count=%lld hash=%016llx row=%d\n",
                            ctx.trace_label, node.id, op_type_name(node.op_type),
                            (long long)dump_count, static_cast<unsigned long long>(hash), dump_row_offset);
                else if (dump_node_hash)
                    fprintf(stderr, "HASH graph=%s node=%u op=%s count=%lld hash=%016llx\n",
                            ctx.trace_label, node.id, op_type_name(node.op_type),
                            (long long)dump_count, static_cast<unsigned long long>(hash));
            }
        }
        // Release completed tensors. Classify borrowed views before mutating
        // any producer in this release batch; a producer and its view can have
        // the same last-use node.
        const auto& rels = reuse_workspace || defer_sigmoid || defer_up || defer_inject || defer_moe ||
            (pending_moe>=0&&!inject_rms_consumer) ? empty_release_batch :
            (hc_up_consumer ? ctx.hc_up_reduce_release_queue[i] :
             inject_rms_consumer&&pending_moe>=0 ? ctx.moe_gr_inject_rms_release_queue[i] :
             inject_rms_consumer ? ctx.gr_inject_rms_release_queue[i] :
             fused_consumer ? ctx.sigmoid_gr_reduce_release_queue[i] : ctx.release_queue[i]);
        std::vector<uint8_t> rel_borrowed(rels.size(), 0);
        for (size_t r = 0; r < rels.size(); r++) {
            uint32_t rel_id = rels[r];
            if (rel_id >= nodes.size()) continue;
            auto op = nodes[rel_id].op_type;
            if (op == OpType::SLICE || op == OpType::PERMUTE) {
                rel_borrowed[r] = 1;
            } else if (op == OpType::RESHAPE && !nodes[rel_id].inputs.empty()) {
                auto& t = tensors[rel_id];
                const Tensor& src = tensors[nodes[rel_id].inputs[0]];
                rel_borrowed[r] = (device_resident ? device_reshape_borrows(ctx.backend, src, t)
                                                   : t.shares_storage_with(src)) ? 1 : 0;
            } else if (op == OpType::CONTIGUOUS &&
                       !nodes[rel_id].inputs.empty()) {
                const Tensor& src =
                    tensors[nodes[rel_id].inputs[0]];
                rel_borrowed[r] =
                    device_resident && src.is_contiguous() ? 1 : 0;
            }
        }

        for (size_t r = 0; r < rels.size(); r++) {
            uint32_t rel_id = rels[r];
            auto& t = tensors[rel_id];
            if (rel_id < nodes.size()) {
                auto op = nodes[rel_id].op_type;
                if (op == OpType::INPUT || op == OpType::CONSTANT)
                    continue;
                if (rel_borrowed[r]) {
                    t.data = nullptr;
                    t.device.buffer = nullptr;
                    t.device.offset = 0;
                    t.mem_type = MemoryType::NONE;
                    t.owner_id = 0;
                    t.storage_id = 0;
                    continue;
                }
            }
            if (t.data && t.mem_type == MemoryType::POOLED && t.nbytes() > 0) {
                if (!device_resident && t.owner_id != 0 && t.owner_id != pool->id()) {
                    std::fprintf(stderr,
                                 "execute_graph: owner mismatch in release_queue for node %u (%p owner=%u pool=%u)\n",
                                 rel_id, t.data, t.owner_id, pool->id());
                    assert(false && "execute_graph release_queue owner mismatch");
                    ctx.execution_failed = true;
                    return;
                }
                ctx.backend->free_output(t, pool);
                t.data     = nullptr;
                t.device.buffer = nullptr;
                t.device.offset = 0;
                t.mem_type = MemoryType::NONE;
                t.owner_id = 0;
                t.storage_id = 0;
            }
        }

        if (defer_sigmoid) pending_sigmoid = static_cast<int32_t>(i);
        else if (fused_consumer) pending_sigmoid = -1;
        if(defer_up)pending_up=static_cast<int32_t>(i);
        else if(hc_up_consumer)pending_up=-1;
        if (defer_inject) pending_inject = static_cast<int32_t>(i);
        else if (inject_rms_consumer) pending_inject = -1;
        if(defer_moe)pending_moe=static_cast<int32_t>(i);
        else if(inject_rms_consumer)pending_moe=-1;
        if (ctx.profile_enabled && node.id < ctx.profile_stats.size()) {
            auto end_time = std::chrono::steady_clock::now();
            auto elapsed_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
            auto& stat = ctx.profile_stats[node.id];
            stat.calls += 1;
            stat.total_ns += elapsed_ns;
        }
        if (stop_after_node_index >= 0 &&
            static_cast<int>(i) >= stop_after_node_index)
            break;
    }

    if (ctx.reuse_same_shape_workspace && has_dynamic &&
        (!segmented || stop_after_node_index == static_cast<int>(nodes.size()) - 1)) {
        ctx.workspace_shape_valid = true;
        ctx.workspace_runtime_seq_len = ctx.runtime_seq_len;
        ctx.workspace_runtime_batch = ctx.runtime_batch;
        ctx.workspace_static_padded = ctx.static_padded;
        ctx.workspace_padded_seq_len = ctx.padded_seq_len;
    }
}

void execute_graph(ExecContext& ctx, int stop_after_node_index) {
    ctx.segment_next = -1;
    execute_graph_impl(ctx, stop_after_node_index, 0, false);
}

static bool validate_graph_segment(ExecContext& ctx, int first, int end) {
    auto reject = [&] {
        std::fprintf(stderr, "execute_graph_segment: invalid range or suspended execution identity\n");
        ctx.execution_failed = true;
    };
    if (!ctx.graph || !ctx.pool || !ctx.backend || first < 0 || end <= first ||
        static_cast<size_t>(end) > ctx.graph->nodes.size()) {
        reject(); return false;
    }
    if (first == 0) {
        if (ctx.segment_next > 0) { reject(); return false; }
        ctx.segment_graph = ctx.graph; ctx.segment_pool = ctx.pool;
        ctx.segment_backend = ctx.backend; ctx.segment_moe_backend = ctx.moe_backend;
        ctx.segment_seq = ctx.runtime_seq_len; ctx.segment_batch = ctx.runtime_batch;
        ctx.segment_static_padded = ctx.static_padded;
        ctx.segment_fusions_identity = ctx.segment_fusions_enabled;
        ctx.segment_padded_seq = ctx.padded_seq_len;
    } else if (ctx.segment_next != first || ctx.segment_graph != ctx.graph ||
               ctx.segment_pool != ctx.pool || ctx.segment_backend != ctx.backend ||
               ctx.segment_moe_backend != ctx.moe_backend ||
               ctx.segment_seq != ctx.runtime_seq_len || ctx.segment_batch != ctx.runtime_batch ||
               ctx.segment_static_padded != ctx.static_padded ||
               ctx.segment_fusions_identity != ctx.segment_fusions_enabled ||
               ctx.segment_padded_seq != ctx.padded_seq_len || ctx.execution_failed) {
        reject(); return false;
    }
    return true;
}

void execute_graph_segment(ExecContext& ctx, int first, int end) {
    if (!validate_graph_segment(ctx, first, end)) return;
    execute_graph_impl(ctx, end - 1, first, true);
    ctx.segment_next = ctx.execution_failed || static_cast<size_t>(end) == ctx.graph->nodes.size()
        ? -1 : end;
}

int execute_graph_moe_pair(ExecContext& a, ExecContext& b, int index) {
    auto reject = [&] {
        std::fprintf(stderr, "execute_graph_moe_pair: unsupported frame/node identity\n");
        a.execution_failed = b.execution_failed = true;
        return -1;
    };
    if (&a == &b || a.graph == b.graph || index <= 0 || a.backend != b.backend ||
        !a.backend || a.moe_backend || b.moe_backend || a.static_padded || b.static_padded ||
        a.profile_enabled || b.profile_enabled || a.emulate_bf16_activations || b.emulate_bf16_activations ||
        a.runtime_seq_len < 1 || a.runtime_seq_len > 1024 ||
        b.runtime_seq_len < 1 || b.runtime_seq_len > 1024 ||
        !validate_graph_segment(a, index, index + 1) ||
        !validate_graph_segment(b, index, index + 1)) return reject();
    const auto& na = a.graph->nodes[index]; const auto& nb = b.graph->nodes[index];
    if (na.op_type != OpType::MOE || nb.op_type != OpType::MOE || na.id != nb.id ||
        na.inputs != nb.inputs || na.params.i32 != nb.params.i32 ||
        na.params.f32 != nb.params.f32 || na.params.str != nb.params.str ||
        na.out_prec != nb.out_prec || a.resolved_inputs.size() <= size_t(index) ||
        b.resolved_inputs.size() <= size_t(index)) return reject();
    for (int d = 0; d < 4; ++d)
        if (na.out_shape[d] != nb.out_shape[d] || na.dim_expr[d].kind != nb.dim_expr[d].kind ||
            na.dim_expr[d].coeff != nb.dim_expr[d].coeff) return reject();
    for (ExecContext* c : {&a, &b}) {
        const auto& n = c->graph->nodes[index]; auto& out = c->graph->runtime.tensors[n.id];
        if (out.data || out.device.buffer) return reject();
        for (int d = 0; d < 4; ++d)
            out.shape[d] = eval_dim(n.dim_expr[d], n.out_shape[d], c->runtime_seq_len, c->runtime_batch);
        out.prec = n.out_prec; out.compute_strides();
        if (!out.nbytes() || !c->backend->alloc_output(out, out.nbytes(), c->pool)) return reject();
    }
    a.backend->clear_dispatch_error(); a.backend->begin_execution();
    const bool merged = a.backend->dispatch_moe_pair(na, a.resolved_inputs[index],
        b.resolved_inputs[index], &a.graph->runtime.tensors[na.id], &b.graph->runtime.tensors[nb.id]);
    if (a.backend->dispatch_failed()) return reject();
    for (ExecContext* c : {&a, &b}) {
        execute_graph_impl(*c, index, index, true, merged ? index : -1);
        c->segment_next = c->execution_failed || size_t(index + 1) == c->graph->nodes.size()
            ? -1 : index + 1;
        if (c->execution_failed || c->backend->dispatch_failed()) return reject();
    }
    return merged ? 1 : 0;
}
