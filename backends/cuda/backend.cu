#include <cuda_runtime.h>

#include "backends/cuda/backend.h"
#include "backends/cuda/backend_internal.h"
#include "backends/cuda/device_runtime.h"
#include "graph/hc_deferred_rms.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <limits>
#include <string>

using mollm::cuda_backend_detail::device_pointer;
using mollm::cuda_backend_detail::device_pointer_const;

bool CudaBackend::reshape_view_strides(const Tensor& source, const Tensor& target,
                                     size_t strides[4]) const {
    if (!impl_->strided_reshape_views || impl_->segment_enabled || !strides ||
        !impl_->strided_reshape_targets.count(&target) ||
        source.prec != Precision::FP32 || target.prec != Precision::FP32 ||
        source.is_contiguous() || source.shape[2] != 1 || source.shape[3] != 1 ||
        target.shape[3] != 1 || source.stride[0] != sizeof(float)) return false;
    for (int d = 0; d < 4; ++d)
        if (source.shape[d] <= 0 || target.shape[d] <= 0) return false;
    const size_t limit = std::numeric_limits<size_t>::max();
    if (uint64_t(source.shape[0]) > limit / sizeof(float) ||
        uint64_t(target.shape[0]) > uint64_t(source.shape[0]) ||
        source.shape[0] % target.shape[0] ||
        source.shape[0] / target.shape[0] != target.shape[1] ||
        target.shape[2] != source.shape[1] ||
        source.stride[1] < size_t(source.shape[0]) * sizeof(float) ||
        source.stride[1] % sizeof(float) ||
        uint64_t(target.shape[2]) > limit / source.stride[1]) return false;
    const size_t result[4] = {sizeof(float), size_t(target.shape[0]) * sizeof(float),
                             source.stride[1], source.stride[1] * size_t(target.shape[2])};
    std::copy(result, result + 4, strides);
    return true;
}

size_t CudaBackend::strided_reshape_view_calls() const { return impl_->strided_reshape_calls; }
size_t CudaBackend::prefill_shared_cache_bytes() const { return impl_->prefill_shared_cache_bytes; }
size_t CudaBackend::prefill_shared_cache_hits() const { return impl_->prefill_shared_cache_hits; }

namespace {
class CudaStateSnapshot final : public BackendStateSnapshot {
public:
    explicit CudaStateSnapshot(std::shared_ptr<mollm_cuda::DeviceBufferPool> pool)
        : pool_(std::move(pool)) {}
    ~CudaStateSnapshot() override {
        if (storage_) mollm_cuda::release_device_buffer(pool_.get(), storage_);
    }
    bool capture(const std::vector<Tensor*>& states) {
        size_t allocation = 0;
        for (Tensor* state : states) {
            if (!state || !state->view_span_bytes()) continue;
            if (std::any_of(entries_.begin(), entries_.end(),
                    [state](const Entry& e) { return e.state == state; })) continue;
            const size_t size = state->view_span_bytes();
            const size_t limit = std::numeric_limits<size_t>::max();
            if (size > limit - 127 || allocation > limit - (size + 127)) return false;
            entries_.push_back({state, device_pointer<uint8_t>(*state), size, allocation});
            allocation += (size + 127) & ~size_t(127);
            bytes_ += size;
        }
        if (!allocation) return true;
        if (!mollm_cuda::report_cuda(cudaDeviceSynchronize(), "state snapshot producer completion")) return false;
        storage_ = mollm_cuda::acquire_device_buffer(pool_.get(), allocation);
        if (!storage_) return false;
        for (const auto& e : entries_)
            if (!mollm_cuda::report_cuda(cudaMemcpyAsync(
                    static_cast<uint8_t*>(storage_) + e.offset, e.pointer, e.size,
                    cudaMemcpyDeviceToDevice, nullptr), "state snapshot capture")) return false;
        return mollm_cuda::report_cuda(cudaStreamSynchronize(nullptr), "state snapshot capture completion");
    }
    bool restore() const override {
        // Reject every changed span/address before writing any state. This is
        // deliberately raw storage, not a cached KV-length rewind operation.
        for (const auto& e : entries_)
            if (e.state->view_span_bytes() != e.size ||
                device_pointer<uint8_t>(*e.state) != e.pointer) return false;
        if (!mollm_cuda::report_cuda(cudaDeviceSynchronize(), "state snapshot verification completion")) return false;
        for (const auto& e : entries_)
            if (!mollm_cuda::report_cuda(cudaMemcpyAsync(e.pointer,
                    static_cast<const uint8_t*>(storage_) + e.offset, e.size,
                    cudaMemcpyDeviceToDevice, nullptr), "state snapshot restore")) return false;
        return mollm_cuda::report_cuda(cudaStreamSynchronize(nullptr), "state snapshot restore completion");
    }
    size_t bytes() const override { return bytes_; }
private:
    struct Entry { Tensor* state; uint8_t* pointer; size_t size, offset; };
    std::vector<Entry> entries_;
    void* storage_ = nullptr;
    size_t bytes_ = 0;
    // A live snapshot owns its lease. Shared ownership also lets snapshots
    // outlive the backend object without dereferencing a destroyed pool.
    std::shared_ptr<mollm_cuda::DeviceBufferPool> pool_;
};
} // namespace

void CudaBackend::end_state_prefixes() {
    for(auto& entry:impl_->state_prefixes)
        if(entry.storage)mollm_cuda::release_device_buffer(impl_->state_snapshot_pool.get(),entry.storage);
    impl_->state_prefixes.clear();impl_->state_prefix_tokens=0;
}

bool CudaBackend::begin_state_prefixes(const std::vector<Tensor*>& states,int tokens) {
    end_state_prefixes();
    const char* flag=std::getenv("MOLLM_CUDA_MTP_STATE_PREFIXES");
    if(!flag||std::strcmp(flag,"1")||tokens<2||tokens>8||states.empty())return false;
    finish_decode_segment();
    if(!impl_->state_snapshot_pool)
        impl_->state_snapshot_pool=std::shared_ptr<mollm_cuda::DeviceBufferPool>(
            mollm_cuda::create_device_buffer_pool(),mollm_cuda::destroy_device_buffer_pool);
    if(!impl_->state_snapshot_pool)return false;
    for(Tensor* tensor:states) {
        if(!tensor)continue;
        void* pointer=device_pointer<uint8_t>(*tensor);
        const size_t bytes=tensor->view_span_bytes();
        if(!bytes)continue;
        if(!pointer||impl_->persistent_host_mirrors.count(tensor->device.buffer)||
            (tensor->prec!=Precision::FP32 && tensor->prec!=Precision::INT32)||!tensor->is_contiguous()||
            bytes>std::numeric_limits<size_t>::max()/size_t(tokens)) {
            static int reports=0;if(reports++<4)std::fprintf(stderr,
                "CUDA prefix unavailable: bytes=%zu precision=%d contiguous=%d device=%d mirror=%d\n",
                bytes,int(tensor->prec),tensor->is_contiguous(),pointer!=nullptr,
                int(impl_->persistent_host_mirrors.count(tensor->device.buffer)));
            end_state_prefixes();return false;
        }
        auto duplicate=std::find_if(impl_->state_prefixes.begin(),impl_->state_prefixes.end(),
            [pointer](const auto& entry){return entry.pointer==pointer;});
        if(duplicate!=impl_->state_prefixes.end()) {
            if(duplicate->bytes!=bytes){end_state_prefixes();return false;}
            continue;
        }
        void* storage=mollm_cuda::acquire_device_buffer(impl_->state_snapshot_pool.get(),bytes*tokens);
        if(!storage){end_state_prefixes();return false;}
        impl_->state_prefixes.push_back({tensor,pointer,storage,bytes,false});
    }
    impl_->state_prefix_tokens=tokens;
    size_t payload=0;for(const auto& entry:impl_->state_prefixes)payload+=entry.bytes*size_t(tokens);
    impl_->state_prefix_peak_payload_bytes=std::max(impl_->state_prefix_peak_payload_bytes,payload);
    return !impl_->state_prefixes.empty();
}

bool CudaBackend::restore_state_prefix(int kept) {
    if(kept<1||kept>impl_->state_prefix_tokens||impl_->state_prefixes.empty())return false;
    for(const auto& entry:impl_->state_prefixes)
        if(!entry.captured||entry.tensor->view_span_bytes()!=entry.bytes||
            device_pointer<uint8_t>(*entry.tensor)!=entry.pointer) {
            static int reports=0;if(reports++<4)std::fprintf(stderr,
                "CUDA prefix incomplete: tokens=%d bytes=%zu captured=%d shape=%lld,%lld,%lld,%lld\n",
                impl_->state_prefix_tokens,entry.bytes,entry.captured,
                (long long)entry.tensor->shape[0],(long long)entry.tensor->shape[1],
                (long long)entry.tensor->shape[2],(long long)entry.tensor->shape[3]);
            return false;
        }
    if(!mollm_cuda::report_cuda(cudaDeviceSynchronize(),"prefix producer completion"))return false;
    for(const auto& entry:impl_->state_prefixes)
        if(!mollm_cuda::report_cuda(cudaMemcpyAsync(entry.pointer,
            static_cast<uint8_t*>(entry.storage)+size_t(kept-1)*entry.bytes,
            entry.bytes,cudaMemcpyDeviceToDevice,nullptr),"prefix restore"))return false;
    return mollm_cuda::report_cuda(cudaStreamSynchronize(nullptr),"prefix restore completion");
}

bool CudaBackend::supports_state_snapshot(const std::vector<Tensor*>& states) const {
    static const bool enabled = [] {
        const char* value = std::getenv("MOLLM_CUDA_MTP_DEVICE_CHECKPOINT");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (!enabled || !available()) return false;
    for (const Tensor* state : states) {
        if (!state || !state->view_span_bytes()) continue;
        // Host-authoritative/mirrored metadata requires the existing transfer
        // hooks. A mixed collection falls back as a whole rather than losing it.
        if (!state->device.buffer || impl_->persistent_host_mirrors.count(state->device.buffer))
            return false;
    }
    return true;
}

std::unique_ptr<BackendStateSnapshot> CudaBackend::capture_state_snapshot(
    const std::vector<Tensor*>& states) {
    if (!supports_state_snapshot(states)) return nullptr;
    finish_decode_segment();
    if (!impl_->state_snapshot_pool)
        impl_->state_snapshot_pool = std::shared_ptr<mollm_cuda::DeviceBufferPool>(
            mollm_cuda::create_device_buffer_pool(), mollm_cuda::destroy_device_buffer_pool);
    if (!impl_->state_snapshot_pool) return nullptr;
    auto snapshot = std::make_unique<CudaStateSnapshot>(impl_->state_snapshot_pool);
    if (!snapshot->capture(states)) { impl_->failed = true; return nullptr; }
    return snapshot;
}

CudaBackend::CudaBackend(FusionPolicy policy) : impl_(std::make_unique<Impl>()) {
    if (policy == FusionPolicy::REFERENCE) {
        impl_->qualified_fusions = false;
        impl_->hc_projection_enabled = false;
        impl_->hc_prefill_reuse_enabled = false;
        impl_->hc_prefill_rms_half_enabled = false;
        impl_->hc_deferred_enabled = false;
        impl_->hc_compact_enabled = false;
        impl_->strided_reshape_views = false;
    }
    if (const char* flag = std::getenv("MOLLM_CUDA_STRIDED_RESHAPE_VIEWS"))
        impl_->strided_reshape_views = !std::strcmp(flag, "1");
    if (const char* option = std::getenv("MOLLM_CUDA_OUTPUT_POOL_MB")) {
        errno = 0;
        char* end = nullptr;
        const unsigned long long mb = std::strtoull(option, &end, 10);
        const bool digits = *option && std::all_of(option, option + std::strlen(option),
            [](char c) { return c >= '0' && c <= '9'; });
        if (!digits || errno || !end || *end || mb > std::numeric_limits<size_t>::max() / (1024 * 1024)) {
            std::fprintf(stderr, "CudaBackend: invalid MOLLM_CUDA_OUTPUT_POOL_MB\n");
            return;
        }
        impl_->output_pool_limit_bytes = static_cast<size_t>(mb) * 1024 * 1024;
    }
    int count = 0;
    if (!mollm_cuda::report_cuda(
            cudaGetDeviceCount(&count), "cudaGetDeviceCount") ||
        count <= 0)
        return;
    if (!mollm_cuda::report_cuda(cudaSetDevice(0), "cudaSetDevice") ||
        !mollm_cuda::report_cublas(
            cublasCreate(&impl_->cublas), "cublasCreate"))
        return;
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    // cuBLAS does not inherit nvcc's per-thread default-stream setting.
    // Match our conversion/activation kernels, including during graph capture.
    if (!mollm_cuda::report_cublas(
            cublasSetStream(impl_->cublas, cudaStreamPerThread), "cublasSetStream"))
        return;
#endif
    // Allocate before any graph capture; the independent shared-expert
    // branch uses explicit fork/join events outside and inside CUDA Graphs.
    if (!mollm_cuda::report_cuda(cudaStreamCreateWithFlags(&impl_->moe_shared_stream,
            cudaStreamNonBlocking), "shared expert stream") ||
        !mollm_cuda::report_cuda(cudaEventCreateWithFlags(&impl_->moe_shared_ready,
            cudaEventDisableTiming), "shared expert ready event") ||
        !mollm_cuda::report_cuda(cudaEventCreateWithFlags(&impl_->moe_shared_done,
            cudaEventDisableTiming), "shared expert done event")) return;
    impl_->ok = true;
    int pageable = 0, host_tables = 0;
    if (cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0) == cudaSuccess &&
        cudaDeviceGetAttribute(&host_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0) == cudaSuccess)
        impl_->coherent_host_weights = pageable && host_tables;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess) {
        impl_->supports_native_nvfp4_mma = properties.major == 12 && properties.minor <= 1;
        impl_->supports_extended_nvfp4_prefill_sort = properties.major == 12 && properties.minor == 1;
        impl_->supports_sdpa_head_lane_pair=properties.major==12&&properties.minor==1&&
            properties.maxThreadsPerBlock>=512&&properties.regsPerBlock>=32768;
        if(properties.major==12&&properties.minor==1)
            impl_->prefill_lt=std::make_unique<mollm_cuda::PrefillLtGemm>();
        std::fprintf(stderr, "CudaBackend: using %s (sm_%d%d)\n",
                     properties.name, properties.major, properties.minor);
    }
}

CudaBackend::~CudaBackend() {
    if (std::getenv("MOLLM_CUDA_STRIDED_RESHAPE_VIEWS"))
        std::fprintf(stderr, "CUDA strided reshape views: enabled=%d calls=%zu\n",
                     impl_->strided_reshape_views, impl_->strided_reshape_calls);
    if(impl_->state_prefix_peak_payload_bytes)std::fprintf(stderr,
        "CUDA state prefixes: peak_payload_bytes=%zu state_pool_reserved_bytes=%zu\n",
        impl_->state_prefix_peak_payload_bytes,
        mollm_cuda::device_buffer_pool_reserved_bytes(impl_->state_snapshot_pool.get()));
    end_state_prefixes();
}

bool CudaBackend::available() const { return impl_ && impl_->ok; }

bool CudaBackend::set_logical_prefill_batch(bool enabled) {
    if(logical_prefill_batch_active_==enabled)return true;
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    if(!mollm_cuda::report_cuda(cudaStreamIsCapturing(cudaStreamPerThread,&capture),
            "logical prefill policy capture check")||capture!=cudaStreamCaptureStatusNone)return false;
    logical_prefill_batch_active_=enabled;
    return true;
}

void CudaBackend::begin_execution() {
    if(!impl_->failed&&!impl_->materialize_deferred_rms())return;
    impl_->cpu.begin_execution();
    // free_output invalidates metadata before operands can be recycled.
    // Healthy partial passes flush; failed passes discard cached pointers.
    impl_->hc_deferred={};
    impl_->hc_prefill_owner = nullptr;
    ++impl_->hc_execution_epoch;
    if (!impl_->hc_execution_epoch) {
        for (auto& entry : impl_->hc_projection_plans)
            for (auto& pair : entry.second.pairs) pair.computed_epoch = 0;
        ++impl_->hc_execution_epoch;
    }
}

uint64_t CudaBackend::independent_stateful_mutable_inputs(const GraphNode& node) const {
    if(!impl_->segment_enabled&&node.op_type==OpType::QSA_INDEXER&&
       node.out_prec==Precision::INT32&&node.inputs.size()==8)
        return (uint64_t(1)<<4)|(uint64_t(1)<<5);
    if (!impl_->segment_enabled && node.out_prec == Precision::FP32 &&
        node.op_type == OpType::QK_RMS_NORM_ROPE && node.inputs.size() == 6) {
        // This fused normalization/rotation is read-only despite the generic
        // planner grouping it with stateful ops. Native and staged reference
        // paths write an independent destination and retain no operand owners.
        return 0;
    }
    if (!impl_->segment_enabled && node.out_prec == Precision::FP32 &&
        (node.op_type == OpType::SDPA || node.op_type == OpType::QSA_SDPA) &&
        (node.inputs.size() == 6 || node.inputs.size() == 7)) {
        // Native SDPA stores into K/V and a separately allocated destination.
        // The synchronous reference fallback stages/writes back inputs before
        // returning. Neither path retains Q/K/V/mask/output addresses beyond
        // stream-ordered dispatch. Paired-frame tests cover both cache types,
        // GQA, causal masks, view owners, and full persistent cache payloads.
        return (uint64_t(1) << 4) | (uint64_t(1) << 5);
    }
    return Backend::independent_stateful_mutable_inputs(node);
}

int CudaBackend::independent_stateful_mutable_input(const GraphNode& node) const {
    // Cached decode segments may retain operand addresses beyond dispatch.
    // Fail closed even for non-engine callers until that lifetime is proved.
    if(impl_->segment_enabled)return -1;
    // dispatch.cu writes destination separately from state; prefix snapshots
    // retain state/prefix allocations, never qkv/input or destination buffers.
    // The explicit host reference fallback also writes an independent output.
    if(node.out_prec!=Precision::FP32)return -1;
    if(node.op_type==OpType::SHORTCONV && node.inputs.size()==3)return 2;
    if(node.op_type==OpType::GATED_DELTANET_PREFILL && node.inputs.size()==8)return 7;
    return -1;
}

void CudaBackend::release_graph_plan(const Graph& graph) {
    for (auto it = impl_->strided_reshape_targets.begin(); it != impl_->strided_reshape_targets.end();) {
        if (it->second == &graph) it = impl_->strided_reshape_targets.erase(it);
        else ++it;
    }
    auto old = impl_->hc_projection_plans.find(&graph);
    if (old == impl_->hc_projection_plans.end()) return;
    if (!impl_->failed) {
        if (!impl_->materialize_deferred_rms()) impl_->failed = true;
        if (!impl_->failed) finish_decode_segment(); else abandon_decode_segment();
    } else abandon_decode_segment();
    if (!mollm_cuda::report_cuda(cudaDeviceSynchronize(), "release HC graph plan")) impl_->failed = true;
    impl_->hc_deferred = {}; impl_->hc_prefill_owner = nullptr;
    for (auto& entry : impl_->decode_segments)
        if (entry.second.graph) cudaGraphExecDestroy(entry.second.graph);
    impl_->decode_segments.clear();
    for (const auto& pair : old->second.pairs) {
        impl_->hc_projection_nodes.erase(pair.first);
        impl_->hc_projection_nodes.erase(pair.second);
        if (pair.rms) impl_->hc_prefill_rms_nodes.erase(pair.rms);
    }
    if (old->second.slots) {
        auto& allocations = impl_->device_allocations;
        allocations.erase(std::remove(allocations.begin(), allocations.end(), old->second.slots), allocations.end());
        if (!mollm_cuda::report_cuda(cudaFree(old->second.slots), "free HC graph slots")) impl_->failed = true;
    }
    impl_->hc_projection_plans.erase(old);
}

void CudaBackend::prepare_graph(const Graph& graph) {
    if (!available()) return;
    if(!impl_->failed&&!impl_->materialize_deferred_rms())return;
    impl_->hc_deferred={};
    impl_->hc_prefill_owner = nullptr;
    auto old = impl_->hc_projection_plans.find(&graph);
    if (old != impl_->hc_projection_plans.end()) {
        release_graph_plan(graph);
        if (impl_->failed) return;
    }
    for (auto it = impl_->strided_reshape_targets.begin(); it != impl_->strided_reshape_targets.end();) {
        if (it->second == &graph) it = impl_->strided_reshape_targets.erase(it);
        else ++it;
    }
    if (impl_->strided_reshape_views && !impl_->segment_enabled) {
        std::unordered_map<uint32_t, std::vector<std::pair<const GraphNode*, size_t>>> users;
        std::unordered_map<uint32_t, const GraphNode*> layout_nodes;
        for (const auto& n : graph.nodes) {
            layout_nodes[n.id] = &n;
            for (size_t i = 0; i < n.inputs.size(); ++i) users[n.inputs[i]].push_back({&n, i});
        }
        const auto exposed = [&](uint32_t id) {
            return std::find(graph.graph_outputs.begin(), graph.graph_outputs.end(), id) != graph.graph_outputs.end();
        };
        for (const auto& n : graph.nodes) {
            if (n.op_type != OpType::RESHAPE || n.inputs.size() != 1 ||
                n.out_prec != Precision::FP32 || n.id >= graph.runtime.tensors.size() ||
                exposed(n.id) || users[n.id].size() != 1) continue;
            const auto parent = layout_nodes.find(n.inputs[0]);
            // The qualified path is a feature SLICE, whose previous runtime
            // metadata survives until reset classification. Direct INPUTs may
            // be replaced/resized before execute; do not infer their old view
            // ownership from the new input's shape or address.
            if (parent == layout_nodes.end() || parent->second->op_type != OpType::SLICE ||
                parent->second->inputs.size() != 1 || parent->second->params.i32.size() < 3 ||
                parent->second->params.i32[0] != 0 || parent->second->out_prec != Precision::FP32) continue;
            auto user = users[n.id][0];
            // Only verified stride-aware consumers. Generic dense consumers
            // must continue receiving materialized contiguous reshape output.
            if (user.first->op_type == OpType::PERMUTE && user.first->inputs.size() == 1 &&
                user.first->params.i32.size() == 4 &&
                user.first->params.i32 == std::vector<int32_t>({0, 2, 1, 3}) &&
                !exposed(user.first->id) && users[user.first->id].size() == 1)
                user = users[user.first->id][0];
            const bool safe = user.first->out_prec == Precision::FP32 &&
                ((user.first->op_type == OpType::RESHAPE && user.first->inputs.size() == 1) ||
                 (user.first->op_type == OpType::SDPA && user.second == 2 &&
                  (user.first->inputs.size() == 6 || user.first->inputs.size() == 7)) ||
                 (user.first->op_type == OpType::SIGMOID_MUL && user.second == 1 && user.first->inputs.size() == 2));
            if (safe) impl_->strided_reshape_targets[&graph.runtime.tensors[n.id]] = &graph;
        }
    }
    if (!impl_->hc_projection_enabled && !impl_->hc_prefill_reuse_enabled) return;
    std::unordered_map<uint32_t, const GraphNode*> producers;
    for (const auto& node : graph.nodes) producers[node.id] = &node;
    std::unique_ptr<HcDeferredGraphIndex> deferred_index;
    if(impl_->hc_deferred_enabled&&!impl_->segment_enabled)
        deferred_index=std::make_unique<HcDeferredGraphIndex>(graph);
    const auto plain_matmul = [](const GraphNode& node) {
        return node.op_type == OpType::MATMUL && node.inputs.size() == 2 &&
            node.out_prec == Precision::FP32 &&
            graph_params::get_i32(node.params, 0, 0) == 0 &&
            graph_params::get_i32(node.params, 1, 0) == 0 &&
            graph_params::get_i32(node.params, 2, -1) == -1;
    };
    const auto constant_weight = [&](uint32_t id, int rows) {
        auto found = producers.find(id);
        return found != producers.end() && found->second->op_type == OpType::CONSTANT &&
            id < graph.runtime.tensors.size() &&
            graph.runtime.tensors[id].shape[0] == rows &&
            graph.runtime.tensors[id].shape[1] == 10240;
    };
    Impl::HcProjectionPlan plan;
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& first = graph.nodes[i];
        if (!plain_matmul(first) || first.out_shape[0] != 320 ||
            first.inputs[0] >= graph.runtime.tensors.size() ||
            !constant_weight(first.inputs[1], 320)) continue;
        for (size_t j = i + 1; j < std::min(graph.nodes.size(), i + 13); ++j) {
            const auto& second = graph.nodes[j];
            if (second.op_type == OpType::MATMUL && !second.inputs.empty() &&
                second.inputs[0] == first.inputs[0]) {
                if (plain_matmul(second) && second.id < graph.runtime.tensors.size() && second.out_shape[0] == 4 &&
                    constant_weight(second.inputs[1], 4)) {
                    Impl::HcProjectionPair pair;
                    pair.first = &first;
                    pair.second = &second;
                    pair.input = &graph.runtime.tensors[first.inputs[0]];
                    pair.second_weight = &graph.runtime.tensors[second.inputs[1]];
                    // Only constants may separate the producer from first use.
                    // No other shadow producer can overwrite scratch in between.
                    for (size_t before=i; before>0;) {
                        const auto& producer=graph.nodes[--before];
                        if (producer.op_type==OpType::CONSTANT) continue;
                        if (producer.id==first.inputs[0] &&
                            producer.op_type==OpType::GROUP_RMS_NORM) pair.rms=&producer;
                        break;
                    }
                    if(pair.rms&&deferred_index)
                        pair.deferred_reduce=hc_deferred_rms_consumer(graph,*pair.rms,first,second,*deferred_index);
                    plan.pairs.push_back(pair);
                }
                break;
            }
            // Only fresh-output, pure operators may intervene. In particular,
            // no views, state mutation, recurrent ops or host boundaries.
            if (second.op_type != OpType::CONSTANT && second.op_type != OpType::MUL &&
                second.op_type != OpType::SILU && second.op_type != OpType::MATMUL &&
                second.op_type != OpType::SIGMOID && second.op_type != OpType::GR_REDUCE) break;
        }
    }
    if (plan.pairs.empty()) return;
    if(!impl_->hc_deferred_inverses&&std::any_of(plan.pairs.begin(),plan.pairs.end(),
        [](const Impl::HcProjectionPair& p){return p.deferred_reduce!=nullptr;})) {
        if(!mollm_cuda::malloc_device(reinterpret_cast<void**>(&impl_->hc_deferred_inverses),
            4096*4*sizeof(float),"deferred HC RMS inverses")) {impl_->failed=true;return;}
        impl_->device_allocations.push_back(impl_->hc_deferred_inverses);
    }
    float* slots = nullptr;
    if (!mollm_cuda::malloc_device(reinterpret_cast<void**>(&slots),
            plan.pairs.size() * 4 * sizeof(float), "HC projection slots")) return;
    impl_->device_allocations.push_back(slots);
    auto& stored = impl_->hc_projection_plans.emplace(&graph, std::move(plan)).first->second;
    stored.slots = slots;
    for (size_t i = 0; i < stored.pairs.size(); ++i) {
        auto& pair = stored.pairs[i];
        pair.cache = slots + 4 * i;
        impl_->hc_projection_nodes[pair.first] = &pair;
        impl_->hc_projection_nodes[pair.second] = &pair;
        if (pair.rms) impl_->hc_prefill_rms_nodes[pair.rms] = &pair;
    }
    std::fprintf(stderr, "CudaBackend HC projection plan: pairs=%zu slot_bytes=%zu\n",
        stored.pairs.size(), stored.pairs.size() * 4 * sizeof(float));
}

void CudaBackend::clear_dispatch_error() {
    if(impl_->failed)impl_->hc_deferred={}; // failed operands may already be released
    impl_->failed = false;
    impl_->cpu.clear_dispatch_error();
}

bool CudaBackend::dispatch_failed() const {
    return impl_->failed || impl_->cpu.dispatch_failed();
}

bool CudaBackend::set_operator_fallback_policy(
    OperatorFallbackPolicy policy) {
    impl_->operator_fallback = policy;
    return true;
}

bool CudaBackend::register_weight_region(void*, size_t) { return true; }

void CudaBackend::wrap_weight(Tensor& tensor) {
    if (!available() || !tensor.data || tensor.shape[0] <= 0 ||
        tensor.shape[1] <= 0)
        return;
    const int n = static_cast<int>(tensor.shape[0]);
    const int k = static_cast<int>(tensor.shape[1]);
    if (tensor.prec == Precision::FP16) {
        impl_->upload_weight(tensor, tensor.data, tensor.data,
                             static_cast<size_t>(n) * k * sizeof(__half),
                             CUDA_R_16F, n, k);
    } else if (tensor.prec == Precision::FP32) {
        impl_->upload_weight(tensor, tensor.data, tensor.data,
                             static_cast<size_t>(n) * k * sizeof(float),
                             CUDA_R_32F, n, k);
    } else if (tensor.prec == Precision::INT8 && tensor.scales &&
               tensor.group_size > 0) {
        const void* quantized =
            tensor.rowmajor_data ? tensor.rowmajor_data : tensor.data;
        const int group_size = static_cast<int>(tensor.group_size);
        const int groups_per_row = static_cast<int>(tensor.groups_per_row);
        if (quantized && groups_per_row ==
                (k + group_size - 1) / group_size) {
            impl_->upload_quantized_weight(
                tensor, quantized, quantized, static_cast<size_t>(n) * k,
                tensor.scales, static_cast<size_t>(n) * groups_per_row,
                n, k, group_size, groups_per_row);
        }
    }
}

void CudaBackend::wrap_weight_int4(Tensor& tensor,
                                   bool keep_native_experts) {
    if (!available() || keep_native_experts || tensor.prec != Precision::INT4 ||
        tensor.shape[0] <= 0 || tensor.shape[1] <= 0)
        return;
    const int n = static_cast<int>(tensor.shape[0]);
    const int k = static_cast<int>(tensor.shape[1]);

    if (tensor.is_q4_g32_packed && tensor.q4_g32_data && k % 32 == 0 &&
        tensor.group_size == 32 &&
        tensor.groups_per_row == static_cast<uint32_t>(k / 32)) {
        const size_t bytes = static_cast<size_t>((n + 7) / 8) * (k / 32) *
            sizeof(Q4B8G32Block);
        // Lossless compact layouts; each scale must round-trip exactly.
        constexpr bool compact_enabled=true;
        constexpr bool compact_integer_enabled=true;
        const bool integer_layout = compact_integer_enabled && n == 13312;
        // Independent geometry opt-ins; every scale must round-trip exactly.
        if (((compact_enabled && n == 16480) || integer_layout) && k == 2560 &&
            impl_->weights.find(tensor.q4_g32_data) == impl_->weights.end()) {
            const size_t count = bytes / sizeof(Q4B8G32Block);
            const auto* original = static_cast<const Q4B8G32Block*>(tensor.q4_g32_data);
            std::vector<mollm_cuda::Q4CompactG32> encoded(count);
            bool exact = true;
            for (size_t i = 0; i < count && exact; ++i) {
                for (int row = 0; row < 8; ++row) {
                    const float scale = original[i].scales[row];
                    const float maximum = scale * 7.f;
                    if (!std::isfinite(scale) || scale < 0.f || !std::isfinite(maximum)) {
                        exact = false; break;
                    }
                    uint32_t bits;
                    std::memcpy(&bits, &maximum, sizeof(bits));
                    const uint16_t code = static_cast<uint16_t>(
                        (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
                    const uint32_t restored_bits = static_cast<uint32_t>(code) << 16;
                    float restored;
                    std::memcpy(&restored, &restored_bits, sizeof(restored));
                    restored /= 7.f;
                    if (std::memcmp(&scale, &restored, sizeof(scale))) {
                        exact = false; break;
                    }
                    encoded[i].scales[row] = code;
                    for (int byte = 0; byte < 16; ++byte)
                        encoded[i].q[row][byte] = original[i].q[row][byte] ^ 0x88u;
                }
            }
            if (exact) {
                if (std::getenv("MOLLM_CUDA_Q4_PALETTE_AUDIT")) {
                    std::array<bool,65536> seen{};
                    size_t unique=0;
                    for(const auto& block:encoded) for(uint16_t code:block.scales)
                        if(!seen[code]){seen[code]=true;++unique;}
                    std::fprintf(stderr,"Q4 scale palette audit n=%d k=%d unique=%zu blocks=%zu\n",
                        n,k,unique,count);
                    // Storage-order segments keep dictionary metadata local.
                    // Count the exact codes and include each BF16 dictionary
                    // entry in the hypothetical byte footprint.
                    for (size_t banks : {size_t{1}, size_t{4}, size_t{16}}) {
                        const size_t segment_blocks = banks * static_cast<size_t>(k / 32);
                        size_t dictionary_codes = 0, maximum_codes = 0, overflow_segments = 0;
                        std::array<uint32_t,65536> stamps{};
                        uint32_t stamp = 0;
                        for (size_t begin = 0; begin < count; begin += segment_blocks) {
                            ++stamp;
                            size_t codes = 0;
                            const size_t end = std::min(count, begin + segment_blocks);
                            for (size_t block = begin; block < end; ++block)
                                for (uint16_t code : encoded[block].scales)
                                    if (stamps[code] != stamp) {
                                        stamps[code] = stamp;
                                        ++codes;
                                    }
                            dictionary_codes += codes;
                            maximum_codes = std::max(maximum_codes, codes);
                            overflow_segments += codes > 256;
                        }
                        std::fprintf(stderr,
                            "Q4 segmented palette audit n=%d banks=%zu max_unique=%zu overflow=%zu dictionary_bytes=%zu original_bytes=%zu indexed_bytes=%zu\n",
                            n, banks, maximum_codes, overflow_segments,
                            dictionary_codes * sizeof(uint16_t), count * sizeof(encoded[0]),
                            count * size_t{136} + dictionary_codes * sizeof(uint16_t));
                    }
                }
                const bool uploaded = impl_->upload_weight(tensor, tensor.q4_g32_data, encoded.data(),
                    count * sizeof(encoded[0]), CUDA_R_8I, n, k,
                    integer_layout ? Impl::WeightLayout::Q4Bg32CompactInteger : Impl::WeightLayout::Q4Bg32Compact);
                if (uploaded)
                    std::fprintf(stderr, "CudaBackend: compact W4 %dx%d saved_bytes=%zu\n",
                        n, k, bytes - count * sizeof(encoded[0]));
                return;
            }
        }
        impl_->upload_weight(
            tensor, tensor.q4_g32_data, tensor.q4_g32_data, bytes,
            CUDA_R_8I, n, k, Impl::WeightLayout::Q4Bg32Biased);
    } else if (tensor.is_q4_g128_packed && tensor.q4_g128_data &&
               k % 128 == 0 && tensor.group_size == 128 &&
               tensor.groups_per_row == static_cast<uint32_t>(k / 128)) {
        const size_t bytes = static_cast<size_t>((n + 7) / 8) * (k / 128) *
            sizeof(Q4B8G128Block);
        impl_->upload_weight(
            tensor, tensor.q4_g128_data, tensor.q4_g128_data, bytes,
            CUDA_R_8I, n, k, Impl::WeightLayout::Q4Bg128);
    }
}

void* CudaBackend::alloc_output(Tensor& output, size_t nbytes, BufferPool*) {
    if (!available() || nbytes == 0)
        return nullptr;
    const size_t limit = impl_->output_pool_limit_bytes ?
        impl_->output_pool_limit_bytes-impl_->hc_compact_scratch_bytes : 0;
    if (limit && !mollm_cuda::device_buffer_pool_can_reuse(impl_->output_pool, nbytes)) {
        size_t reserved = mollm_cuda::device_buffer_pool_reserved_bytes(impl_->output_pool);
        if (nbytes > limit) {
            std::fprintf(stderr, "CudaBackend: output pool request exceeds limit requested=%zu limit=%zu\n", nbytes, limit);
            impl_->failed = true;
            return nullptr;
        }
        if (reserved > limit - nbytes) {
            size_t reclaimed = 0, invalidated = 0;
            if (!trim_cached_outputs(&reclaimed, &invalidated)) {
                impl_->failed = true;
                return nullptr;
            }
            if (reclaimed) {
                ++impl_->output_pool_reclaims;
                impl_->output_pool_reclaimed_bytes += reclaimed;
                std::fprintf(stderr, "CudaBackend: output pool reclaim requested=%zu reclaimed=%zu invalidated_graphs=%zu\n",
                    nbytes, reclaimed, invalidated);
            }
            reserved = mollm_cuda::device_buffer_pool_reserved_bytes(impl_->output_pool);
            if (reserved > limit - nbytes) {
                std::fprintf(stderr, "CudaBackend: output pool active allocations exceed limit requested=%zu reserved=%zu limit=%zu\n",
                    nbytes, reserved, limit);
                impl_->failed = true;
                return nullptr;
            }
        }
    }
    void* pointer = mollm_cuda::acquire_device_buffer(
        impl_->output_pool, nbytes);
    if (!pointer) {
        impl_->failed = true;
        return nullptr;
    }
    output.data = pointer;
    output.device.buffer = pointer;
    output.device.offset = 0;
    output.mem_type = MemoryType::POOLED;
    output.owner_id = 0;
    output.storage_id = 0;
    if(impl_->hc_compact_enabled){
        if(!impl_->hc_compact_carries.empty())impl_->hc_compact_carries.erase(pointer);
        // Only allocations capable of becoming a compact prefill carry need
        // generations. Decode/ordinary outputs must not populate this map.
        const bool eligible=output.prec==Precision::FP32&&output.shape[0]==10240&&
            output.shape[1]>=16&&output.shape[1]<=mollm::hc::prefill_max_rows&&output.shape[2]==1&&output.shape[3]==1&&
            output.is_contiguous()&&nbytes>=size_t(output.shape[1])*10240*sizeof(float);
        if(eligible)impl_->hc_allocations[pointer]={nbytes,++impl_->hc_allocation_generation};
        else if(!impl_->hc_allocations.empty())impl_->hc_allocations.erase(pointer);
    }
    return pointer;
}

void CudaBackend::free_output(Tensor& tensor, BufferPool*) {
    if(impl_->hc_deferred.owner&&tensor.device.buffer) {
        const auto* pointer=device_pointer_const<float>(tensor);
        if(impl_->deferred_rms_overlaps(pointer,tensor.view_span_bytes())) {
            if(!impl_->failed)impl_->materialize_deferred_rms();
            else impl_->hc_deferred={};
        }
    }
    if (tensor.device.buffer) {
        if(!impl_->hc_compact_carries.empty())impl_->hc_compact_carries.erase(tensor.device.buffer);
        if(!impl_->hc_allocations.empty())impl_->hc_allocations.erase(tensor.device.buffer);
        mollm_cuda::release_device_buffer(
            impl_->output_pool, tensor.device.buffer);
    }
}

bool CudaBackend::copy_to_host(const Tensor& source, void* destination,
                               size_t nbytes, size_t source_offset) {
    finish_decode_segment();
    if (!destination || source_offset > source.view_span_bytes() ||
        nbytes > source.view_span_bytes() - source_offset) {
        impl_->failed = true;
        return false;
    }
    if(impl_->deferred_rms_overlaps(device_pointer_const<float>(source),source.view_span_bytes(),true)&&
       !impl_->materialize_deferred_rms())return false;
    if(!impl_->restore_hc_compact(device_pointer_const<uint8_t>(source),source.view_span_bytes()))return false;
    size_t host_prefix_bytes = 0;
    const auto mirror = impl_->persistent_host_mirrors.find(
        source.device.buffer);
    const size_t absolute_offset = source.device.offset + source_offset;
    if (mirror != impl_->persistent_host_mirrors.end() &&
        !mirror->second.device_coherent &&
        absolute_offset < mirror->second.size) {
        host_prefix_bytes = std::min(
            nbytes, mirror->second.size - absolute_offset);
        std::memcpy(
            destination, mirror->second.data() + absolute_offset,
            host_prefix_bytes);
        if (host_prefix_bytes == nbytes)
            return true;
    }
    const auto* device = device_pointer_const<uint8_t>(source);
    if (!device) {
        if (!source.data) {
            impl_->failed = true;
            return false;
        }
        std::memcpy(
            static_cast<uint8_t*>(destination) + host_prefix_bytes,
            static_cast<const uint8_t*>(source.data) + source_offset +
                host_prefix_bytes,
            nbytes - host_prefix_bytes);
        return true;
    }
    if (!mollm_cuda::copy_memory(
            static_cast<uint8_t*>(destination) + host_prefix_bytes,
            device + source_offset + host_prefix_bytes,
            nbytes - host_prefix_bytes,
            cudaMemcpyDeviceToHost, "cudaMemcpy tensor to host")) {
        impl_->failed = true;
        return false;
    }
    return true;
}

bool CudaBackend::copy_from_host(const void* source, Tensor& destination,
                                 size_t nbytes,
                                 size_t destination_offset) {
    if (!source || destination_offset > destination.view_span_bytes() ||
        nbytes > destination.view_span_bytes() - destination_offset) {
        impl_->failed = true;
        return false;
    }
    if(impl_->deferred_rms_overlaps(device_pointer_const<float>(destination),destination.view_span_bytes())&&
       !impl_->materialize_deferred_rms())return false;
    if(!impl_->restore_hc_compact(device_pointer_const<uint8_t>(destination),destination.view_span_bytes()))return false;
    const auto mirror = impl_->persistent_host_mirrors.find(
        destination.device.buffer);
    const size_t absolute_offset =
        destination.device.offset + destination_offset;
    size_t host_prefix_bytes = 0;
    if (mirror != impl_->persistent_host_mirrors.end() &&
        absolute_offset < mirror->second.size) {
        host_prefix_bytes = std::min(
            nbytes, mirror->second.size - absolute_offset);
        if (!mirror->second.device_coherent) {
            std::memcpy(
                mirror->second.data() + absolute_offset, source,
                host_prefix_bytes);
            if (host_prefix_bytes == nbytes)
                return true;
        }
    }
    auto* device = device_pointer<uint8_t>(destination);
    if (!device) {
        if (!destination.data) {
            impl_->failed = true;
            return false;
        }
        std::memcpy(
            static_cast<uint8_t*>(destination.data) + destination_offset,
            source, nbytes);
        return true;
    }
    const size_t device_skip =
        mirror != impl_->persistent_host_mirrors.end() &&
            !mirror->second.device_coherent
        ? host_prefix_bytes : 0;
    if (!mollm_cuda::copy_memory(
            device + destination_offset + device_skip,
            static_cast<const uint8_t*>(source) + device_skip,
            nbytes - device_skip,
            cudaMemcpyHostToDevice, "cudaMemcpy tensor from host")) {
        impl_->failed = true;
        return false;
    }
    if (mirror != impl_->persistent_host_mirrors.end() &&
        mirror->second.device_coherent && host_prefix_bytes > 0)
        std::memcpy(
            mirror->second.data() + absolute_offset, source,
            host_prefix_bytes);
    return true;
}

bool CudaBackend::zero_tensor(Tensor& tensor, size_t nbytes,
                              size_t destination_offset) {
    if (destination_offset > tensor.view_span_bytes() ||
        nbytes > tensor.view_span_bytes() - destination_offset) {
        impl_->failed = true;
        return false;
    }
    if(impl_->deferred_rms_overlaps(device_pointer_const<float>(tensor),tensor.view_span_bytes())&&
       !impl_->materialize_deferred_rms())return false;
    if(!impl_->restore_hc_compact(device_pointer_const<uint8_t>(tensor),tensor.view_span_bytes()))return false;
    const auto mirror = impl_->persistent_host_mirrors.find(
        tensor.device.buffer);
    const size_t absolute_offset = tensor.device.offset + destination_offset;
    size_t host_prefix_bytes = 0;
    if (mirror != impl_->persistent_host_mirrors.end() &&
        absolute_offset < mirror->second.size) {
        host_prefix_bytes = std::min(
            nbytes, mirror->second.size - absolute_offset);
        if (!mirror->second.device_coherent) {
            std::memset(
                mirror->second.data() + absolute_offset, 0,
                host_prefix_bytes);
            if (host_prefix_bytes == nbytes)
                return true;
        }
    }
    auto* device = device_pointer<uint8_t>(tensor);
    if (!device) {
        if (!tensor.data) {
            impl_->failed = true;
            return false;
        }
        std::memset(
            static_cast<uint8_t*>(tensor.data) + destination_offset,
            0, nbytes);
        return true;
    }
    const size_t device_skip =
        mirror != impl_->persistent_host_mirrors.end() &&
            !mirror->second.device_coherent
        ? host_prefix_bytes : 0;
    if (!mollm_cuda::zero_memory(
            device + destination_offset + device_skip, nbytes - device_skip,
            "cudaMemset tensor")) {
        impl_->failed = true;
        return false;
    }
    if (mirror != impl_->persistent_host_mirrors.end() &&
        mirror->second.device_coherent && host_prefix_bytes > 0)
        std::memset(
            mirror->second.data() + absolute_offset, 0, host_prefix_bytes);
    return true;
}

bool CudaBackend::round_to_bf16(Tensor& tensor) {
    float* values = device_pointer<float>(tensor);
    if (!values || tensor.prec != Precision::FP32 ||
        !tensor.is_contiguous()) {
        impl_->failed = true;
        return false;
    }
    if(!impl_->restore_hc_compact(values,tensor.view_span_bytes()))return false;
    if (impl_->segment_mode == Impl::SegmentMode::REPLAY) {
        const auto& record = impl_->active_segment->records[impl_->segment_cursor - 1];
        if (record.rounded && record.output == &tensor) return true;
        abandon_decode_segment();
    }
    if (impl_->segment_mode == Impl::SegmentMode::CAPTURE)
        impl_->active_segment->records.back().rounded = true;
    if (!mollm_cuda::launch_round_to_bf16(
            values, static_cast<size_t>(tensor.nelements()))) {
        impl_->failed = true;
        return false;
    }
    return true;
}

void CudaBackend::synchronize_for_host_read() {
    finish_decode_segment();
    if (!mollm_cuda::report_cuda(
            cudaDeviceSynchronize(), "cudaDeviceSynchronize"))
        impl_->failed = true;
}

void CudaBackend::begin_graph() {
    impl_->segment_candidate = false;
    impl_->segment_ready = false;
    impl_->segment_disabled = false;
}

void CudaBackend::abandon_decode_segment() {
    auto* segment = impl_->active_segment;
    const size_t count = impl_->segment_cursor;
    impl_->segment_disabled = true;
    impl_->segment_mode = Impl::SegmentMode::NONE;
    impl_->active_segment = nullptr;
    if (!segment) return;
    ++impl_->segment_invalidations;
    if (segment->graph) cudaGraphExecDestroy(segment->graph);
    segment->graph = nullptr;
    for (size_t i = 0; i < count && !impl_->failed; ++i) {
        const auto& record = segment->records[i];
        dispatch(*record.node, record.inputs, record.output, record.thread_pool);
        if (record.rounded) round_to_bf16(*record.output);
    }
    segment->records.clear();
}

void CudaBackend::finish_decode_segment() {
    auto* segment = impl_->active_segment;
    if (!segment) return;
    if (impl_->segment_mode == Impl::SegmentMode::REPLAY) {
        if (impl_->segment_cursor != segment->records.size()) {
            abandon_decode_segment();
            return;
        }
        if (!mollm_cuda::report_cuda(cudaGraphLaunch(segment->graph, cudaStreamPerThread), "decode segment replay"))
            impl_->failed = true;
        ++impl_->segment_hits;
    } else if (impl_->segment_mode == Impl::SegmentMode::CAPTURE) {
        cudaGraph_t graph = nullptr;
        const cudaError_t captured = cudaStreamEndCapture(cudaStreamPerThread, &graph);
        if (captured != cudaSuccess || !graph) {
            cudaGetLastError();
            if (graph) cudaGraphDestroy(graph);
            abandon_decode_segment();
            return;
        }
        const cudaError_t instantiated = cudaGraphInstantiate(&segment->graph, graph, 0);
        cudaGraphDestroy(graph);
        if (instantiated != cudaSuccess) {
            cudaGetLastError();
            abandon_decode_segment();
            return;
        }
        if (!mollm_cuda::report_cuda(cudaGraphLaunch(segment->graph, cudaStreamPerThread), "decode segment initial launch"))
            impl_->failed = true;
        ++impl_->segment_builds;
    }
    impl_->active_segment = nullptr;
    impl_->segment_mode = Impl::SegmentMode::NONE;
    impl_->segment_cursor = 0;
}

bool CudaBackend::try_decode_segment(const GraphNode& node,
    const std::vector<const Tensor*>& inputs, Tensor* output, ThreadPool* thread_pool,
    uint64_t fusion_tag, const void* fusion_weight, const void* fusion_cache) {
    // Prefix buffers are scoped leases; never cache their addresses in a
    // persistent decode segment or replay a graph that omits prefix stores.
    if (impl_->state_prefix_tokens>0 || !impl_->segment_enabled || !impl_->segment_ready || impl_->segment_disabled) return false;
    if (!impl_->active_segment) {
        auto& segment = impl_->decode_segments[&node];
        impl_->active_segment = &segment;
        impl_->segment_cursor = 0;
        if (segment.graph) impl_->segment_mode = Impl::SegmentMode::REPLAY;
        else {
            segment.records.clear();
            if (cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
                cudaGetLastError();
                impl_->active_segment = nullptr;
                impl_->segment_disabled = true;
                return false;
            }
            impl_->segment_mode = Impl::SegmentMode::CAPTURE;
        }
    }
    auto& segment = *impl_->active_segment;
    // Enumerate the exact signature in place: normal replay must not allocate
    // and copy temporary vectors for every operator of every decode token.
    auto visit_storage = [&](auto&& emit) {
        auto append = [&](const Tensor* tensor) {
            if (!tensor) { emit(uint64_t{0}); return; }
            emit(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor->device.buffer ? tensor->device.buffer : tensor->data)));
            emit(static_cast<uint64_t>(tensor->device.offset));
            emit(static_cast<uint64_t>(tensor->prec));
            for (int d = 0; d < 4; ++d) {
                emit(static_cast<uint64_t>(tensor->shape[d]));
                emit(static_cast<uint64_t>(tensor->stride[d]));
            }
        };
        append(output);
        for (const auto* input : inputs) append(input);
        for (const void* scratch : {impl_->moe_scratch, impl_->moe_jobs,
             impl_->recurrent_scratch, impl_->activation_fp16,
             impl_->quantized_weight_scratch, impl_->attention_scores})
            emit(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(scratch)));
        if (fusion_tag) {
            emit(fusion_tag);
            emit(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fusion_weight)));
            emit(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fusion_cache)));
        }
    };
    if (impl_->segment_mode == Impl::SegmentMode::REPLAY) {
        const size_t index = impl_->segment_cursor;
        bool matches = index < segment.records.size() && segment.records[index].node == &node;
        if (matches) {
            const auto& expected = segment.records[index].signature;
            size_t field = 0;
            visit_storage([&](uint64_t value) {
                if (field >= expected.storage.size() || expected.storage[field] != value) matches = false;
                ++field;
            });
            matches = matches && field == expected.storage.size() &&
                expected.i32 == node.params.i32 && expected.f32 == node.params.f32;
        }
        if (!matches) {
            abandon_decode_segment();
            return false;
        }
        ++impl_->segment_cursor;
        return true;
    }
    Impl::CaptureSignature signature;
    signature.storage.reserve((inputs.size() + 1) * 11 + 6);
    visit_storage([&](uint64_t value) { signature.storage.push_back(value); });
    signature.i32 = node.params.i32;
    signature.f32 = node.params.f32;
    segment.records.push_back({&node, inputs, output, thread_pool, std::move(signature), false});
    ++impl_->segment_cursor;
    return false;
}

void CudaBackend::end_graph() { finish_decode_segment(); synchronize_for_host_read(); }

bool CudaBackend::trim_cached_outputs(size_t* reclaimed_bytes, size_t* invalidated_graphs) {
    if (reclaimed_bytes) *reclaimed_bytes = 0;
    if (invalidated_graphs) *invalidated_graphs = 0;
    if (!available() || !reclaimed_bytes || !invalidated_graphs || dispatch_failed() ||
        impl_->segment_mode != Impl::SegmentMode::NONE || impl_->active_segment) return false;
    // Backend calls are serialized. External captures using these allocations
    // are not supported; inspect every stream owned/used by this backend.
    for (cudaStream_t stream : {cudaStreamPerThread, impl_->moe_capture_stream, impl_->moe_shared_stream}) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (!mollm_cuda::report_cuda(cudaStreamIsCapturing(stream, &capture), "output trim capture check") ||
            capture != cudaStreamCaptureStatusNone) return false;
    }
    if (!mollm_cuda::report_cuda(cudaDeviceSynchronize(), "output trim synchronize")) return false;
    // Invalidate BEFORE returning addresses to CUDA, including caches whose
    // current outputs remain live: this also avoids address-reuse ABA hits.
    for (auto& entry : impl_->decode_segments) {
        if (entry.second.graph) {
            if (!mollm_cuda::report_cuda(cudaGraphExecDestroy(entry.second.graph), "output trim decode graph")) return false;
            entry.second.graph = nullptr;
            ++*invalidated_graphs;
            ++impl_->segment_invalidations;
        }
    }
    impl_->decode_segments.clear();
    for (auto& entry : impl_->resident_experts) {
        auto& resident = entry.second;
        if (resident.graph) {
            if (!mollm_cuda::report_cuda(cudaGraphExecDestroy(resident.graph), "output trim MoE graph")) return false;
            resident.graph = nullptr;
            ++*invalidated_graphs;
        }
        resident.graph_input = nullptr;
        resident.graph_output = nullptr;
    }
    impl_->segment_candidate = impl_->segment_ready = impl_->segment_disabled = false;
    impl_->segment_cursor = 0;
    impl_->hc_prefill_owner = nullptr;
    impl_->hc_prefill_input = nullptr;
    impl_->hc_prefill_epoch = 0;
    for (auto& entry : impl_->hc_projection_plans)
        for (auto& pair : entry.second.pairs) pair.computed_epoch = 0;
    return mollm_cuda::trim_free_device_buffers(impl_->output_pool, cudaStreamPerThread, reclaimed_bytes);
}

void CudaBackend::alloc_persistent(
    Tensor& tensor, size_t nbytes, PersistentHostAccess host_access,
    size_t host_prefix_bytes) {
    void* storage = nullptr;
    const bool has_host_prefix =
        host_access == PersistentHostAccess::MIRRORED_PREFIX ||
        host_access == PersistentHostAccess::HOST_AUTHORITATIVE_PREFIX;
    if (!available() || nbytes == 0 ||
        (has_host_prefix &&
         (host_prefix_bytes == 0 || host_prefix_bytes > nbytes))) {
        impl_->failed = true;
        return;
    }
    if (host_access == PersistentHostAccess::FULL) {
        if (!mollm_cuda::malloc_managed(
                &storage, nbytes,
                "cudaMallocManaged host-coherent persistent")) {
            impl_->failed = true;
            return;
        }
        impl_->managed_allocations.push_back(storage);
        std::memset(storage, 0, nbytes);
        tensor.data = storage;
    } else {
        if (!mollm_cuda::malloc_device(
                &storage, nbytes, "cudaMalloc persistent") ||
            !mollm_cuda::zero_memory(
                storage, nbytes, "cudaMemset persistent")) {
            if (storage)
                cudaFree(storage);
            impl_->failed = true;
            return;
        }
        impl_->device_allocations.push_back(storage);
        if (has_host_prefix) {
            auto [entry, inserted] =
                impl_->persistent_host_mirrors.emplace(
                    storage,
                    Impl::PersistentHostMirror(
                        host_prefix_bytes,
                        host_access ==
                            PersistentHostAccess::MIRRORED_PREFIX));
            (void)inserted;
            tensor.data = entry->second.data();
        } else {
            // Tensor::data remains a non-null storage handle for executor
            // ownership checks. PersistentHostAccess::NONE forbids host
            // dereference; all access goes through explicit transfer methods.
            tensor.data = storage;
        }
    }
    tensor.device.buffer = storage;
    tensor.device.offset = 0;
    tensor.mem_type = MemoryType::EXTERNAL;
}

void CudaBackend::upload_input(Tensor& tensor, const std::string& key,
                               const void* host_source, size_t nbytes) {
    if (!available() || key.empty() || !host_source || nbytes == 0)
        return;
    auto& buffer = impl_->boundary_buffers[key];
    if (!impl_->reserve(buffer.data, buffer.capacity, nbytes)) {
        impl_->failed = true;
        return;
    }
    if (!mollm_cuda::copy_memory(
            buffer.data, host_source, nbytes, cudaMemcpyHostToDevice,
            "cudaMemcpy input")) {
        impl_->failed = true;
        return;
    }
    tensor.device.buffer = buffer.data;
    tensor.device.offset = 0;
}
