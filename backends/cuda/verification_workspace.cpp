#include "backends/cuda/backend.h"
#include "backends/cuda/backend_internal.h"
#include "backends/cuda/state_prefix_plan.h"
#include "graph/execute.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
bool computed(const GraphNode& node) {
    return node.op_type != OpType::INPUT && node.op_type != OpType::CONSTANT;
}

void detach(Tensor& tensor) {
    tensor.data = nullptr;
    tensor.device = {};
    tensor.mem_type = MemoryType::NONE;
    tensor.owner_id = 0;
    tensor.storage_id = 0;
}

// A failed partial pass may still contain borrowed views. Keep persistent
// buffers protected and release each computed allocation exactly once.
std::vector<Tensor> output_leases(const Graph& graph) {
    struct Buffer {
        uintptr_t address;
        uint32_t node;
        bool persistent;
    };
    std::vector<Buffer> buffers;
    buffers.reserve(graph.nodes.size());
    size_t candidates = 0;
    for (const auto& node : graph.nodes) {
        const Tensor& tensor = graph.runtime.tensors[node.id];
        const bool persistent = !computed(node);
        if (tensor.device.buffer &&
            (persistent || (tensor.data && tensor.mem_type == MemoryType::POOLED))) {
            buffers.push_back({reinterpret_cast<uintptr_t>(tensor.device.buffer), node.id, persistent});
            candidates += !persistent;
        }
    }
    // A contiguous list avoids one host allocation per graph node on every
    // width transition. Persistent owners win ties; otherwise preserve the
    // earliest topological owner.
    std::sort(buffers.begin(), buffers.end(), [](const Buffer& a, const Buffer& b) {
        if (a.address != b.address) return a.address < b.address;
        if (a.persistent != b.persistent) return a.persistent;
        return a.node < b.node;
    });
    std::vector<Tensor> leases;
    leases.reserve(candidates);
    for (size_t i = 0; i < buffers.size();) {
        size_t next = i + 1;
        while (next < buffers.size() && buffers[next].address == buffers[i].address) ++next;
        if (!buffers[i].persistent) leases.push_back(graph.runtime.tensors[buffers[i].node]);
        i = next;
    }
    return leases;
}
} // namespace

bool CudaBackend::release_verification_workspaces(Graph& graph, ExecContext& ctx) {
    if (!clear_verification_workspaces(&graph, &ctx)) return false;
    impl_->verification_cache_suspended = false;
    return true;
}

bool CudaBackend::clear_verification_workspaces(Graph* graph, ExecContext* ctx) {
    if (!impl_->verification_workspaces) return true;
    auto& cache = *impl_->verification_workspaces;
    if ((graph == nullptr) != (ctx == nullptr)) return false;
    if (graph && (ctx->graph != graph || ctx->backend != this || graph != cache.graph ||
                  graph->nodes.data() != cache.nodes || graph->runtime.tensors.data() != cache.tensors ||
                  graph->nodes.size() != cache.count)) return false;
    if (!graph) {
        // Engine has already discarded graph/node storage during teardown.
        // No consumer remains; never materialize a deferred graph-owned result.
        impl_->hc_deferred = {};
        impl_->hc_prefill_owner = nullptr;
        impl_->hc_prefill_input = nullptr;
    }
    if (impl_->failed) abandon_decode_segment();
    else finish_decode_segment();
    for (auto stream : {cudaStreamPerThread, impl_->moe_capture_stream, impl_->moe_shared_stream}) {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &status) != cudaSuccess ||
            status != cudaStreamCaptureStatusNone) return false;
    }
    if (!impl_->failed && !impl_->materialize_deferred_rms()) return false;
    if (cudaDeviceSynchronize() != cudaSuccess) return false;
    end_state_prefixes();
    for (auto& slot : cache.slots) {
        for (auto& entry : slot.segments) {
            if (entry.second.graph && cudaGraphExecDestroy(entry.second.graph) != cudaSuccess) return false;
            entry.second.graph = nullptr;
        }
        slot.segments.clear();
    }
    if (!impl_->failed) {
        size_t invalidated = 0;
        if (!retire_cached_output_graphs(&invalidated)) return false;
    } else {
        // Do not clear a latched execution error to make teardown succeed.
        // Failed captures have already been abandoned; retire remaining readers.
        for (auto& entry : impl_->decode_segments) {
            if (entry.second.graph && cudaGraphExecDestroy(entry.second.graph) != cudaSuccess) return false;
            entry.second.graph = nullptr;
        }
        impl_->decode_segments.clear();
        for (auto& entry : impl_->resident_experts) {
            if (entry.second.graph && cudaGraphExecDestroy(entry.second.graph) != cudaSuccess) return false;
            entry.second.graph = nullptr;
            entry.second.graph_input = entry.second.graph_output = nullptr;
        }
        impl_->hc_deferred = {};
        impl_->hc_prefill_owner = nullptr;
        impl_->hc_prefill_input = nullptr;
    }
    if (!retire_verify_prefix_bank()) return false;
    for (auto& slot : cache.slots) {
        for (auto& binding : slot.prefixes) {
            if (binding.storage)
                mollm_cuda::release_device_buffer(impl_->state_snapshot_pool.get(), binding.storage);
            binding.storage = nullptr;
        }
        slot.prefixes.clear();
        for (auto& tensor : slot.leases) free_output(tensor, nullptr);
        slot.leases.clear();
        slot.outputs.clear();
    }
    // A null graph is used only during backend destruction, after Engine has
    // discarded its graph. Active allocations then remain owned by output_pool.
    if (graph) {
        for (auto& tensor : output_leases(*graph)) free_output(tensor, nullptr);
        for (const auto& node : graph->nodes)
            if (computed(node)) detach(graph->runtime.tensors[node.id]);
        ctx->workspace_shape_valid = false;
        ctx->workspace_runtime_seq_len = ctx->workspace_runtime_batch = -1;
        ctx->workspace_static_padded = false;
        ctx->workspace_padded_seq_len = -1;
    }
    // A reset returns leases but does not immediately shrink the pool. Keep
    // reclamation pending so the next ordinary prefill trims these free blocks
    // before checking its host-memory admission margin.
    impl_->verification_pool_retained = graph != nullptr;
    impl_->verification_width_switches += cache.switches;
    impl_->verification_width_hits += cache.hits;
    impl_->verification_peak_output_bytes =
        std::max(impl_->verification_peak_output_bytes, cache.peak_output_bytes);
    impl_->verification_workspaces.reset();
    return true;
}

bool CudaBackend::select_verification_workspace(Graph& graph, ExecContext& ctx, int rows) {
    if (impl_->verification_cache_suspended) return true;
    if (rows < 2 || rows > verification_workspace_max_rows()) return clear_verification_workspaces(&graph, &ctx);
    if (ctx.graph != &graph || ctx.backend != this || !available() || dispatch_failed() ||
        !impl_->state_prefixes.empty()) return false;
    // Unsupported execution modes retain their existing workspace policy.
    // The cache is sized for output pools of at most 3 GiB.
    constexpr size_t max_output_pool_bytes = size_t(3) << 30;
    if (ctx.static_padded || ctx.profile_enabled ||
        (ctx.runtime_batch != -1 && ctx.runtime_batch != 1) ||
        !impl_->segment_requested || !impl_->policy.qualified_verification ||
        !impl_->output_pool_limit_bytes || impl_->output_pool_limit_bytes > max_output_pool_bytes)
        return clear_verification_workspaces(&graph, &ctx);
    if (graph.nodes.empty() || graph.runtime.tensors.size() != graph.nodes.size()) return false;
    for (size_t i = 0; i < graph.nodes.size(); ++i)
        if (graph.nodes[i].id != i) return false;
    if (!impl_->verification_workspaces) {
        // Never orphan an existing single-width workspace during first admission.
        for (const auto& node : graph.nodes)
            if (computed(node) && (graph.runtime.tensors[node.id].data ||
                                   graph.runtime.tensors[node.id].device.buffer)) return true;
        auto cache = std::make_unique<Impl::VerificationWorkspaces>();
        cache->graph = &graph;
        cache->nodes = graph.nodes.data();
        cache->tensors = graph.runtime.tensors.data();
        cache->count = graph.nodes.size();
        impl_->verification_workspaces = std::move(cache);
    }
    auto& cache = *impl_->verification_workspaces;
    if (cache.graph != &graph || cache.nodes != graph.nodes.data() ||
        cache.tensors != graph.runtime.tensors.data() || cache.count != graph.nodes.size()) return false;
    finish_decode_segment();
    if (impl_->active_segment || impl_->segment_mode != Impl::SegmentMode::NONE) return false;
    for (auto stream : {cudaStreamPerThread, impl_->moe_capture_stream, impl_->moe_shared_stream}) {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &status) != cudaSuccess ||
            status != cudaStreamCaptureStatusNone) return false;
    }
    if (cache.active_width == rows) return true;
    cache.max_width = std::max(cache.max_width, rows);
    if (!impl_->materialize_deferred_rms() || cudaDeviceSynchronize() != cudaSuccess) return false;
    auto leases = output_leases(graph);
    size_t held = 0;
    auto account = [&](const Tensor& tensor) {
        const size_t limit = mollm::cuda_backend_detail::verification_workspace_budget_bytes(cache.max_width);
        if (tensor.nbytes() > limit - held) return false;
        held += tensor.nbytes();
        return true;
    };
    bool admitted = true;
    for (const auto& slot : cache.slots)
        for (const auto& tensor : slot.leases) admitted = admitted && account(tensor);
    for (const auto& tensor : leases) admitted = admitted && account(tensor);
    if (!admitted) {
        if (!clear_verification_workspaces(&graph, &ctx)) return false;
        impl_->verification_cache_suspended = true;
        return true;
    }
    cache.peak_output_bytes = std::max(cache.peak_output_bytes, held);
    if (cache.active_width) {
        auto& old = cache.slots[cache.active_width];
        if (!old.outputs.empty() || !old.leases.empty() || !old.prefixes.empty() || !old.segments.empty()) return false;
        old.outputs.resize(cache.count);
        for (const auto& node : graph.nodes)
            if (computed(node)) old.outputs[node.id] = graph.runtime.tensors[node.id];
        old.leases = std::move(leases);
        old.prepared = ctx.workspace_shape_valid && ctx.workspace_runtime_seq_len == cache.active_width &&
            ctx.workspace_runtime_batch == ctx.runtime_batch && !ctx.workspace_static_padded &&
            ctx.workspace_padded_seq_len == -1;
        old.prefixes.swap(impl_->verify_prefix_bank);
        old.segments.swap(impl_->decode_segments);
        old.proven = impl_->verify_prefix_bank_proven;
        old.warmups = impl_->verify_warmups;
    } else if (!retire_verify_prefix_bank()) return false;
    auto& next = cache.slots[rows];
    for (const auto& node : graph.nodes) {
        if (!computed(node)) continue;
        auto& tensor = graph.runtime.tensors[node.id];
        if (!next.outputs.empty()) tensor = next.outputs[node.id];
        else detach(tensor);
    }
    next.outputs.clear();
    next.leases.clear(); // ownership has returned to the graph, not the pool
    impl_->verify_prefix_bank.swap(next.prefixes);
    impl_->decode_segments.swap(next.segments);
    impl_->verify_prefix_bank_proven = next.proven;
    impl_->verify_warmups = next.warmups;
    impl_->verify_prefix_bank_active = impl_->retained_verification = false;
    impl_->segment_candidate = impl_->segment_ready = impl_->segment_disabled = false;
    impl_->segment_cursor = 0;
    ctx.workspace_shape_valid = next.prepared;
    ctx.workspace_runtime_seq_len = next.prepared ? rows : -1;
    ctx.workspace_runtime_batch = ctx.runtime_batch;
    ctx.workspace_static_padded = false;
    ctx.workspace_padded_seq_len = -1;
    ctx.runtime_seq_len = rows;
    ctx.static_padded = false;
    ctx.padded_seq_len = -1;
    inject_runtime_shapes(ctx);
    cache.active_width = rows;
    ++cache.switches;
    if (next.prepared) ++cache.hits;
    return true;
}
