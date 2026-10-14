#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "backends/cpu/backend.h"
#include "backends/cuda/backend.h"
#include "backends/cuda/device_runtime.h"
#include "backends/cuda/dense_lt_gemm.h"
#include "runtime/cuda_host_memory_budget.h"
#include "core/activation.h"
#include "core/quant_layouts.h"
#include "graph/graph.h"
#include "graph/hc_deferred_rms.h"
#include "kernels/cuda/elementwise.h"
#include "kernels/cuda/matmul.h"
#include "kernels/cuda/moe.h"
#include "runtime/expert_provider.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

struct CudaBackend::Impl {
    enum class WeightLayout : uint8_t {
        Dense = 0,
        Q8RowMajor,
        Q4Bg32Biased,
        Q4Bg128,
        Q4Bg32Compact,
        Q4Bg32CompactInteger,
    };

    struct DeviceWeight {
        void* data = nullptr;
        float* scales = nullptr;
        cudaDataType type = CUDA_R_16F;
        int n = 0;
        int k = 0;
        int group_size = 0;
        int groups_per_row = 0;
        WeightLayout layout = WeightLayout::Dense;
        mollm_cuda::Q4RowG32* row_g32 = nullptr;
        void* prefill_fp16 = nullptr;
    };

    struct StatePrefix {
        Tensor* tensor = nullptr;
        void* pointer = nullptr;
        void* storage = nullptr;
        size_t bytes = 0;
        bool captured = false;
    };
    std::vector<StatePrefix> state_prefixes;
    int state_prefix_tokens = 0;
    size_t state_prefix_peak_payload_bytes = 0;
    float* state_prefix_output(void* pointer, size_t bytes, int tokens) {
        if(tokens!=state_prefix_tokens)return nullptr;
        for(auto& entry:state_prefixes)if(entry.pointer==pointer && entry.bytes==bytes) {
            entry.captured=true;return static_cast<float*>(entry.storage);
        }
        return nullptr;
    }

    struct BoundaryBuffer {
        void* data = nullptr;
        size_t capacity = 0;
    };

    struct PersistentHostMirror {
        PersistentHostMirror(size_t bytes, bool coherent)
            : size(bytes),
              device_coherent(coherent),
              words((bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t)) {}

        uint8_t* data() {
            return reinterpret_cast<uint8_t*>(words.data());
        }

        size_t size = 0;
        bool device_coherent = true;
        std::vector<uint64_t> words;
    };

    bool ok = false;
    bool failed = false;
    bool strided_reshape_views = false;
    size_t strided_reshape_calls = 0;
    std::unordered_map<const Tensor*, const Graph*> strided_reshape_targets;
    bool coherent_host_weights = false;
    bool supports_native_nvfp4_mma = false;
    bool supports_sdpa_head_lane_pair = false;
    bool supports_extended_nvfp4_prefill_sort = false;
    cublasHandle_t cublas = nullptr;
    std::unique_ptr<mollm_cuda::PrefillLtGemm> prefill_lt;
    CPUBackend cpu;
    std::unordered_map<const void*, DeviceWeight> weights;
    std::unordered_map<const void*, const DeviceWeight*> weights_by_device;
    size_t prefill_fp16_cache_bytes = 0;
    size_t prefill_shared_cache_bytes = 0, prefill_shared_cache_hits = 0;
    std::vector<void*> device_allocations;
    size_t row_g32_bytes = 0;
    size_t row_g32_weights = 0;
    mollm_cuda::DeviceBufferPool* output_pool =
        mollm_cuda::create_device_buffer_pool();
    size_t output_pool_limit_bytes = 0;
    size_t output_pool_reclaims = 0, output_pool_reclaimed_bytes = 0;
    std::vector<void*> managed_allocations;
    std::shared_ptr<mollm_cuda::DeviceBufferPool> state_snapshot_pool;
    std::unordered_map<void*, PersistentHostMirror> persistent_host_mirrors;
    std::unordered_map<std::string, BoundaryBuffer> boundary_buffers;
    std::unordered_map<uint32_t, uint64_t> native_ops;
    std::unordered_map<uint32_t, uint64_t> fallback_ops;
    OperatorFallbackPolicy operator_fallback =
        OperatorFallbackPolicy::ALLOW_REFERENCE;
    void* activation = nullptr;
    size_t activation_bytes = 0;
    void* activation_fp16 = nullptr;
    size_t activation_fp16_bytes = 0;
    void* quantized_weight_scratch = nullptr;
    size_t quantized_weight_scratch_bytes = 0;
    void* diagnostic_weight_fp32 = nullptr;
    size_t diagnostic_weight_fp32_bytes = 0;
    void* output = nullptr;
    size_t output_bytes = 0;
    void* argmax_partial = nullptr;
    size_t argmax_partial_bytes = 0;
    void* argmax_result = nullptr;
    size_t argmax_result_bytes = 0;
    void* attention_scores = nullptr;
    void* qsa_query = nullptr;
    size_t qsa_query_bytes = 0;
    void* qsa_scores = nullptr;
    size_t qsa_scores_bytes = 0;
    void* qsa_mask_invalid = nullptr;
    size_t qsa_mask_invalid_bytes = 0;
    void* qsa_topk_scores = nullptr;
    size_t qsa_topk_scores_bytes = 0;
    void* qsa_topk_ids = nullptr;
    size_t qsa_topk_ids_bytes = 0;
    size_t qsa_streaming_topk_enqueues = 0;
    size_t qsa_indexer_enqueues = 0, qsa_attention_enqueues = 0;
    size_t qsa_bf16_indexer_enqueues = 0;
    bool qsa_indexer_dump_done = false;
    void* qsa_selected_scores = nullptr;
    size_t qsa_selected_scores_bytes = 0;
    size_t qsa_gqa_enqueues = 0, qsa_split_enqueues = 0;
    void* qsa_attention_shadow = nullptr;
    size_t qsa_attention_shadow_bytes = 0;
    size_t attention_scores_bytes = 0;
    void* attention_prefix_valid = nullptr;
    size_t attention_prefix_valid_bytes = 0;
    // Bounded native-probability PV; separate output preserves the
    // original kernel's invalid-prefix fallback. No weight/cache duplication.
    cublasHandle_t attention_pv_handle = nullptr;
    void* attention_pv_values = nullptr;
    size_t attention_pv_values_bytes = 0;
    void* attention_pv_output = nullptr;
    size_t attention_pv_output_bytes = 0;
    void* attention_pv_pointers = nullptr;
    size_t attention_pv_pointers_bytes = 0;
    size_t attention_pv_gemm_enqueues = 0;
    size_t attention_head_lane_pair_enqueues = 0;
    size_t attention_query_slice_enqueues = 0;
    size_t attention_prefix_enqueues = 0;
    size_t attention_prefix_long_shared_enqueues = 0;
    size_t attention_prefix_zero_pv_enqueues = 0;
    size_t attention_prefix_pv_unroll_enqueues = 0;
    size_t attention_prefix_pv_pair_enqueues = 0;
    void* norm_scratch = nullptr;
    size_t norm_scratch_bytes = 0;
    void* recurrent_scratch = nullptr;
    size_t recurrent_scratch_bytes = 0;
    void* moe_scratch = nullptr;
    size_t moe_scratch_bytes = 0;
    struct Fp16MoeTables {
        const void* gate;
        const void* down;
        int experts, hidden, inter;
        mollm_cuda::Nvfp4Projection* tables;
    };
    std::vector<Fp16MoeTables> fp16_moe_tables;
    struct ResidentExperts {
        bool uniform_gate_input_global = false;
        float gate_input_global = 1.f;
        bool gate_async_aligned = false;
        bool gate_pair_layout = false;
        const ExpertSource* down_source = nullptr;
        int hidden = 0, inter = 0;
        std::vector<ExpertLease> leases;
        mollm_cuda::Nvfp4Projection* tables = nullptr;
        mollm_cuda::Nvfp4Projection* prefill_scale_tables = nullptr;
        uint8_t* prefill_scale_view = nullptr;
        cudaGraphExec_t graph = nullptr;
        const float* graph_input = nullptr;
        float* graph_output = nullptr;
        void* graph_scratch = nullptr;
        void* graph_jobs = nullptr;
        std::array<const void*, 5> graph_weights{};
        std::vector<int32_t> graph_i32;
        std::vector<float> graph_f32;
    };
    std::unordered_map<const ExpertSource*, ResidentExperts> resident_experts;
    size_t prefill_gate_scale_cache_bytes = 0;
    uint64_t prefill_gate_scale_view_calls = 0;
    uint64_t prefill_input_scale_kmajor_calls = 0;
    uint64_t prefill_compact_gate_logit_calls = 0;
    uint64_t prefill_token_quantization_calls = 0;
    uint64_t prefill_token_pair_word_calls = 0;
    uint64_t prefill_token_code_kmajor_calls = 0;
    void* moe_jobs = nullptr;
    size_t moe_jobs_bytes = 0;
    void* moe_mma_partial = nullptr;
    size_t moe_mma_partial_bytes = 0;
    void* moe_mma_group_scratch = nullptr;
    size_t moe_mma_group_scratch_bytes = 0;
    void* moe_radix_group_scratch = nullptr;
    size_t moe_radix_group_scratch_bytes = 0;
    size_t moe_radix_group_calls = 0;
    size_t moe_batched_prefill_calls = 0;
    size_t moe_n16_prefill_calls = 0;
    size_t moe_gate_n32_prefill_calls = 0;
    size_t moe_gate_async_weights_calls = 0;
    size_t moe_gate_compact_prefill_calls = 0;
    size_t moe_down_compact_prefill_calls = 0;
    size_t sigmoid_gr_reduce_fused_calls = 0;
    size_t gr_inject_rms_fused_calls = 0;
    // Atomic dispatch_moe_gr_inject_rms scope only; never survives a graph op.
    struct MoeHcChain {
        bool armed=false,ready=false;
        const uint16_t* routed=nullptr;
        const float* shared=nullptr;
        const float* scale=nullptr;
        float* branch=nullptr;
        int rows=0;
        cudaStream_t stream=nullptr;
    } moe_hc_chain;
    size_t moe_gr_inject_rms_calls=0;
    bool materialize_moe_hc_branch() {
        auto& c=moe_hc_chain;if(!c.ready)return true;
        const auto result=mollm_cuda::launch_moe_combine_bf16(c.routed,c.shared,c.scale,c.branch,c.rows,2560,10,c.stream);
        c.ready=false;
        if(!mollm_cuda::report_cuda(result,"atomic MoE HC branch fallback")){failed=true;return false;}
        return true;
    }
    size_t shortconv_transpose_dispatches = 0;
    size_t moe_sorted_prefill_calls = 0;
    size_t moe_canonical_mma_prefill_calls = 0;
    size_t moe_activation_prepare_fused_calls = 0;
    size_t moe_activation_prepare_packed_only_calls = 0;
    size_t moe_bf16_down_output_prefill_calls = 0;
    size_t moe_bf16_gate_output_prefill_calls = 0;
    size_t moe_gate_early_inputs_prefill_calls = 0;
    size_t moe_gate_vector_quantize_prefill_calls = 0;
    size_t moe_mma_activation_fused_prefill_calls = 0;
    size_t moe_parallel_batch_plan_prefill_calls = 0;
    size_t moe_prefill_input_half_conversions = 0;
    size_t moe_prefill_input_half_consumers = 0;
    cudaStream_t moe_capture_stream = nullptr;
    cudaStream_t moe_shared_stream = nullptr;
    cudaEvent_t moe_shared_ready = nullptr, moe_shared_done = nullptr;
    uint64_t moe_shared_overlap_enqueues = 0;
    struct HcProjectionPair {
        const GraphNode* first = nullptr;
        const GraphNode* second = nullptr;
        const GraphNode* rms = nullptr;
        const GraphNode* deferred_reduce = nullptr;
        const Tensor* input = nullptr;
        const Tensor* second_weight = nullptr;
        float* cache = nullptr;
        const float* computed_input = nullptr;
        const DeviceWeight* computed_weight = nullptr;
        uint64_t computed_epoch = 0;
    };
    struct HcProjectionPlan {
        std::vector<HcProjectionPair> pairs;
        float* slots = nullptr;
    };
    bool hc_projection_enabled = [] {
        const char* value = std::getenv("MOLLM_CUDA_HC_PROJECTION_PAIR");
        return value && std::strcmp(value, "1") == 0;
    }();
    std::unordered_map<const Graph*, HcProjectionPlan> hc_projection_plans;
    std::unordered_map<const GraphNode*, HcProjectionPair*> hc_projection_nodes;
    std::unordered_map<const GraphNode*, HcProjectionPair*> hc_prefill_rms_nodes;
    bool hc_prefill_reuse_enabled = [] {
        const char* value=std::getenv("MOLLM_CUDA_HC_PREFILL_ACTIVATION_REUSE");
        const char* diagnostic=std::getenv("MOLLM_CUDA_DENSE_PREFILL_FP32_REFERENCE");
        return value && std::strcmp(value,"1")==0 &&
            !(diagnostic && std::strcmp(diagnostic,"1")==0);
    }();
    bool hc_prefill_rms_half_enabled = [] {
        const char* value=std::getenv("MOLLM_CUDA_HC_PREFILL_RMS_HALF");
        return value && std::strcmp(value,"1")==0;
    }();
    void* hc_prefill_half = nullptr;
    size_t hc_prefill_half_bytes = 0;
    const HcProjectionPair* hc_prefill_owner = nullptr;
    const float* hc_prefill_input = nullptr;
    int hc_prefill_rows = 0;
    uint64_t hc_prefill_epoch = 0, hc_prefill_reuses = 0;
    bool hc_prefill_from_rms = false;
    uint64_t hc_prefill_rms_half_calls = 0;
    bool hc_deferred_enabled = [] {
        const char* value=std::getenv("MOLLM_CUDA_HC_PREFILL_DEFERRED_RMS");
        return value&&std::strcmp(value,"1")==0;
    }();
    float* hc_deferred_inverses=nullptr;
    bool hc_compact_enabled=[] {
        const char* value=std::getenv("MOLLM_CUDA_HC_COMPACT_CARRY");
        return value&&!std::strcmp(value,"1");
    }();
    struct HcAllocation {size_t bytes;uint64_t generation;};
    struct HcCompactCarry {int rows;uint64_t generation;};
    std::unordered_map<void*,HcAllocation> hc_allocations;
    std::unordered_map<void*,HcCompactCarry> hc_compact_carries;
    uint64_t hc_allocation_generation=0;
    uint16_t* hc_compact_scratch=nullptr;
    size_t hc_compact_scratch_bytes=0;
    uint64_t hc_compact_producers=0,hc_compact_residual_reads=0,hc_compact_consumers=0,hc_compact_restores=0;
    bool hc_compact_exact(const void* pointer,int rows) const {
        if(hc_compact_carries.empty())return false;
        const auto tag=hc_compact_carries.find(const_cast<void*>(pointer));
        const auto allocation=hc_allocations.find(const_cast<void*>(pointer));
        return tag!=hc_compact_carries.end()&&allocation!=hc_allocations.end()&&
            tag->second.rows==rows&&tag->second.generation==allocation->second.generation;
    }
    bool reserve_hc_compact_scratch() {
        if(hc_compact_scratch)return true;
        constexpr size_t bytes=size_t(mollm::hc::prefill_max_rows)*10240*sizeof(uint16_t);
        cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
        if(!mollm_cuda::report_cuda(cudaStreamIsCapturing(cudaStreamPerThread,&capture),"compact HC reserve capture check")){
            failed=true;return false;
        }
        if(capture!=cudaStreamCaptureStatusNone)return false;
        const size_t reserved=mollm_cuda::device_buffer_pool_reserved_bytes(output_pool);
        if(output_pool_limit_bytes&&(bytes>=output_pool_limit_bytes||reserved>output_pool_limit_bytes-bytes))return false;
        if(!mollm_cuda::malloc_device(reinterpret_cast<void**>(&hc_compact_scratch),bytes,"compact HC restore scratch")){
            failed=true;return false;
        }
        hc_compact_scratch_bytes=bytes;device_allocations.push_back(hc_compact_scratch);return true;
    }
    bool restore_hc_compact(const void* pointer,size_t bytes) {
        if(hc_compact_carries.empty()||!pointer||!bytes)return true;
        for(auto entry=hc_compact_carries.begin();entry!=hc_compact_carries.end();) {
            const size_t logical_bytes=size_t(entry->second.rows)*10240*sizeof(float);
            if(!hc_byte_ranges_overlap(reinterpret_cast<uintptr_t>(pointer),bytes,
                reinterpret_cast<uintptr_t>(entry->first),logical_bytes)){++entry;continue;}
            const auto allocation=hc_allocations.find(entry->first);
            // A stale generation is an invariant violation, never read recycled storage.
            if(allocation==hc_allocations.end()||allocation->second.generation!=entry->second.generation){failed=true;return false;}
            cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
            if(!mollm_cuda::report_cuda(cudaStreamIsCapturing(cudaStreamPerThread,&capture),"compact HC restore capture check")||
                capture!=cudaStreamCaptureStatusNone||!hc_compact_scratch||
                !mollm_cuda::launch_hc_compact_carry_materialize(static_cast<float*>(entry->first),hc_compact_scratch,
                    hc_compact_scratch_bytes,entry->second.rows)) {failed=true;return false;}
            ++hc_compact_restores;entry=hc_compact_carries.erase(entry);
        }
        return true;
    }
    struct DeferredRms {
        const HcProjectionPair* owner=nullptr;
        float* normalized=nullptr;
        const float* carry=nullptr;
        const float* gamma=nullptr;
        int rows=0;
        uint64_t epoch=0;
        bool reduced=false;
    } hc_deferred;
    uint64_t hc_deferred_producers=0,hc_deferred_reduces=0,hc_deferred_materializations=0;
    uint64_t hc_up_reduce_fused_calls=0;
    uint64_t hc_up_reduce_padded_calls=0,hc_up_reduce_wide_calls=0;
    bool deferred_rms_overlaps(const void* pointer,size_t bytes,bool normalized_only=false) const {
        if(!hc_deferred.owner)return false;
        const uintptr_t p=reinterpret_cast<uintptr_t>(pointer);
        const size_t tensor_bytes=size_t(hc_deferred.rows)*10240*sizeof(float);
        if(hc_byte_ranges_overlap(p,bytes,reinterpret_cast<uintptr_t>(hc_deferred.normalized),tensor_bytes))return true;
        if(normalized_only)return false;
        return hc_byte_ranges_overlap(p,bytes,reinterpret_cast<uintptr_t>(hc_deferred.carry),tensor_bytes)||
            hc_byte_ranges_overlap(p,bytes,reinterpret_cast<uintptr_t>(hc_deferred.gamma),10240*sizeof(float));
    }
    bool materialize_deferred_rms() {
        if(!hc_deferred.owner)return true;
        if(!mollm_cuda::launch_gr_rms_deferred_materialize(hc_deferred.carry,
            hc_deferred.gamma,hc_deferred_inverses,hc_deferred.normalized,hc_deferred.rows,nullptr,
            hc_compact_exact(hc_deferred.carry,hc_deferred.rows))||
           !mollm_cuda::report_cuda(cudaGetLastError(),"materialize deferred HC RMS")) {
            failed=true;return false;
        }
        ++hc_deferred_materializations;
        hc_deferred={};return true;
    }
    // Grow outside capture only; retain retired storage referenced by old graphs.
    bool reserve_hc_prefill_half(size_t bytes) {
        if (bytes <= hc_prefill_half_bytes) return true;
        cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
        if (!mollm_cuda::report_cuda(cudaStreamIsCapturing(cudaStreamPerThread,&capture),
                "HC prefill activation capture check")) { failed=true; return false; }
        if (capture!=cudaStreamCaptureStatusNone) return false;
        size_t capacity=std::max<size_t>(4096,hc_prefill_half_bytes);
        while (capacity<bytes && capacity<=SIZE_MAX/2) capacity*=2;
        if (capacity<bytes) capacity=bytes;
        void* replacement=nullptr;
        if (!mollm_cuda::malloc_device(&replacement,capacity,"HC prefill activation scratch")) {
            failed=true; return false;
        }
        if (hc_prefill_half) device_allocations.push_back(hc_prefill_half);
        hc_prefill_half=replacement; hc_prefill_half_bytes=capacity;
        return true;
    }
    uint64_t hc_execution_epoch = 0, hc_projection_enqueues = 0, hc_projection_copies = 0;
    uint64_t moe_graph_builds = 0, moe_graph_hits = 0;
    bool capture_audit = std::getenv("MOLLM_CUDA_CAPTURE_AUDIT") != nullptr;
    bool attention_gemm_shadow = [] {
        const char* value=std::getenv("MOLLM_TEST_CUDA_SDPA_GEMM_SHADOW");
        return value&&!std::strcmp(value,"1");
    }();
    unsigned attention_gemm_shadow_count = 0;
    uint64_t attention_prefix_query_group_enqueues = 0;
    uint64_t attention_prefix_head_group_enqueues = 0;
    bool exact_scratch_growth = std::getenv("MOLLM_CUDA_DISABLE_GEOMETRIC_GROWTH") != nullptr;
    struct CaptureSignature {
        std::vector<uint64_t> storage;
        std::vector<int32_t> i32;
        std::vector<float> f32;
    };
    struct CaptureAuditCounts {
        uint64_t first = 0, stable = 0, storage_changed = 0, params_changed = 0;
    };
    std::unordered_map<const GraphNode*, CaptureSignature> capture_signatures;
    std::unordered_map<uint32_t, CaptureAuditCounts> capture_audit_counts;
    struct SegmentRecord {
        const GraphNode* node = nullptr;
        std::vector<const Tensor*> inputs;
        Tensor* output = nullptr;
        ThreadPool* thread_pool = nullptr;
        CaptureSignature signature;
        bool rounded = false;
    };
    struct DecodeSegment {
        cudaGraphExec_t graph = nullptr;
        std::vector<SegmentRecord> records;
    };
    std::unordered_map<const GraphNode*, DecodeSegment> decode_segments;
    DecodeSegment* active_segment = nullptr;
    enum class SegmentMode { NONE, CAPTURE, REPLAY };
    SegmentMode segment_mode = SegmentMode::NONE;
    size_t segment_cursor = 0;
    bool segment_candidate = false, segment_ready = false, segment_disabled = false;
    uint64_t segment_decode_steps = 0, segment_builds = 0, segment_hits = 0, segment_invalidations = 0;
    const GraphNode* segment_decode_node = nullptr;
    bool segment_enabled =
#ifdef MOLLM_CUDA_DECODE_SEGMENTS
        std::getenv("MOLLM_CUDA_DISABLE_DECODE_SEGMENTS") == nullptr;
#else
        false;
#endif

    ~Impl() {
        if(hc_compact_enabled)std::fprintf(stderr,
            "CudaBackend compact HC: producers=%llu consumers=%llu residual_reads=%llu restorations=%llu scratch_bytes=%zu live_tags=%zu allocation_registrations=%llu\n",
            static_cast<unsigned long long>(hc_compact_producers),static_cast<unsigned long long>(hc_compact_consumers),
            static_cast<unsigned long long>(hc_compact_residual_reads),static_cast<unsigned long long>(hc_compact_restores),
            hc_compact_scratch_bytes,hc_compact_carries.size(),static_cast<unsigned long long>(hc_allocation_generation));
        if (output_pool_limit_bytes)
            std::fprintf(stderr, "CudaBackend: output_pool_limit_bytes=%zu reclaims=%zu reclaimed_bytes=%zu\n",
                output_pool_limit_bytes, output_pool_reclaims, output_pool_reclaimed_bytes);
        const char* memory_audit = std::getenv("MOLLM_CUDA_MEMORY_AUDIT");
        if (memory_audit && std::strcmp(memory_audit, "1") == 0) {
            std::fprintf(stderr, "CudaBackend: output_pool_reserved_bytes=%zu\n",
                mollm_cuda::device_buffer_pool_reserved_bytes(output_pool));
            std::fprintf(stderr,"CudaBackend: attention_score_workspace_bytes=%zu\n",attention_scores_bytes);
        }
        if (prefill_fp16_cache_bytes)
            std::fprintf(stderr, "CudaBackend: prefill FP16 weight cache bytes=%zu\n",
                prefill_fp16_cache_bytes);
        if (prefill_gate_scale_cache_bytes)
            std::fprintf(stderr,"CudaBackend: prefill gate scale cache bytes=%zu calls=%llu\n",
                prefill_gate_scale_cache_bytes,
                static_cast<unsigned long long>(prefill_gate_scale_view_calls));
        if (prefill_input_scale_kmajor_calls)
            std::fprintf(stderr,"CudaBackend: prefill K64-major input scale calls=%llu\n",
                static_cast<unsigned long long>(prefill_input_scale_kmajor_calls));
        if (prefill_compact_gate_logit_calls)
            std::fprintf(stderr,"CudaBackend: prefill compact gate logits calls=%llu\n",
                static_cast<unsigned long long>(prefill_compact_gate_logit_calls));
        if (prefill_token_quantization_calls)
            std::fprintf(stderr,"CudaBackend: prefill shared token quantization calls=%llu\n",
                static_cast<unsigned long long>(prefill_token_quantization_calls));
        if (prefill_token_code_kmajor_calls)
            std::fprintf(stderr,"CudaBackend: prefill K64-major token code calls=%llu\n",
                static_cast<unsigned long long>(prefill_token_code_kmajor_calls));
        if (prefill_token_pair_word_calls)
            std::fprintf(stderr,"CudaBackend: prefill paired token word calls=%llu\n",
                static_cast<unsigned long long>(prefill_token_pair_word_calls));
        if (row_g32_weights)
            std::fprintf(stderr, "CudaBackend Q4 row-major sidecars: weights=%zu bytes=%zu\n",
                row_g32_weights, row_g32_bytes);
        if (capture_audit) {
            std::fprintf(stderr, "CudaBackend capture audit (diagnostic only; cache lengths are not kernel parameters here):\n");
            for (const auto& entry : capture_audit_counts)
                std::fprintf(stderr, "  %-24s first=%llu stable=%llu storage_changed=%llu params_changed=%llu\n",
                    op_type_name(static_cast<OpType>(entry.first)),
                    static_cast<unsigned long long>(entry.second.first),
                    static_cast<unsigned long long>(entry.second.stable),
                    static_cast<unsigned long long>(entry.second.storage_changed),
                    static_cast<unsigned long long>(entry.second.params_changed));
        }
        // Resident leases and descriptor tables outlive all asynchronous MoE
        // reads. Engine teardown destroys this backend before the provider.
        if (segment_mode == SegmentMode::CAPTURE) {
            cudaGraph_t unfinished = nullptr;
            cudaStreamEndCapture(cudaStreamPerThread, &unfinished);
            if (unfinished) cudaGraphDestroy(unfinished);
        }
        if (ok) cudaDeviceSynchronize();
        for (auto& entry : decode_segments)
            if (entry.second.graph) cudaGraphExecDestroy(entry.second.graph);
        if (segment_enabled)
            std::fprintf(stderr, "CudaBackend decode segments: builds=%llu hits=%llu invalidations=%llu\n",
                static_cast<unsigned long long>(segment_builds),
                static_cast<unsigned long long>(segment_hits),
                static_cast<unsigned long long>(segment_invalidations));
        for (auto& entry : resident_experts) {
            if (entry.second.graph) cudaGraphExecDestroy(entry.second.graph);
            if (entry.second.tables) cudaFree(entry.second.tables);
            if (entry.second.prefill_scale_tables) cudaFree(entry.second.prefill_scale_tables);
            if (entry.second.prefill_scale_view) cudaFree(entry.second.prefill_scale_view);
        }
        if (moe_capture_stream) cudaStreamDestroy(moe_capture_stream);
        if (moe_shared_stream) cudaStreamDestroy(moe_shared_stream);
        if (moe_shared_ready) cudaEventDestroy(moe_shared_ready);
        if (moe_shared_done) cudaEventDestroy(moe_shared_done);
        if (moe_shared_overlap_enqueues)
            std::fprintf(stderr, "CudaBackend shared-expert overlap: enqueues=%llu\n",
                static_cast<unsigned long long>(moe_shared_overlap_enqueues));
        if (hc_projection_enqueues || hc_projection_copies)
            std::fprintf(stderr, "CudaBackend HC projection pairs: enqueues=%llu copies=%llu\n",
                static_cast<unsigned long long>(hc_projection_enqueues),
                static_cast<unsigned long long>(hc_projection_copies));
        resident_experts.clear();
        if (moe_jobs) cudaFree(moe_jobs);
        if (moe_mma_partial) cudaFree(moe_mma_partial);
        if (moe_mma_group_scratch) cudaFree(moe_mma_group_scratch);
        if (moe_radix_group_scratch) cudaFree(moe_radix_group_scratch);
        if (moe_radix_group_calls)
            std::fprintf(stderr, "CudaBackend: radix NVFP4 prefill group calls=%zu scratch_bytes=%zu\n",
                moe_radix_group_calls, moe_radix_group_scratch_bytes);
        if (moe_batched_prefill_calls)
            std::fprintf(stderr, "CudaBackend: batched NVFP4 prefill calls=%zu scratch_bytes=%zu\n",
                moe_batched_prefill_calls, moe_mma_group_scratch_bytes);
        if (moe_n16_prefill_calls)
            std::fprintf(stderr, "CudaBackend: N16 NVFP4 prefill calls=%zu\n", moe_n16_prefill_calls);
        if (moe_gate_async_weights_calls)
            std::fprintf(stderr, "CudaBackend: async N32 gate weight calls=%zu\n", moe_gate_async_weights_calls);
        if (moe_gate_n32_prefill_calls)
            std::fprintf(stderr, "CudaBackend: gate-only N32 NVFP4 prefill calls=%zu\n", moe_gate_n32_prefill_calls);
        if (moe_gate_compact_prefill_calls)
            std::fprintf(stderr, "CudaBackend: compact gate NVFP4 prefill calls=%zu\n", moe_gate_compact_prefill_calls);
        if (moe_down_compact_prefill_calls)
            std::fprintf(stderr, "CudaBackend: compact down NVFP4 prefill calls=%zu\n", moe_down_compact_prefill_calls);
        if (gr_inject_rms_fused_calls)
            std::fprintf(stderr, "CudaBackend: GR_INJECT RMS fused calls=%zu\n", gr_inject_rms_fused_calls);
        if(moe_gr_inject_rms_calls)
            std::fprintf(stderr,"CudaBackend: MoE GR_INJECT RMS fused calls=%zu\n",moe_gr_inject_rms_calls);
        if(hc_deferred_producers)
            std::fprintf(stderr,"CudaBackend deferred HC RMS: producers=%llu reduces=%llu materializations=%llu\n",
                static_cast<unsigned long long>(hc_deferred_producers),
                static_cast<unsigned long long>(hc_deferred_reduces),
                static_cast<unsigned long long>(hc_deferred_materializations));
        if (sigmoid_gr_reduce_fused_calls)
            std::fprintf(stderr, "CudaBackend: sigmoid GR_REDUCE fused calls=%zu\n", sigmoid_gr_reduce_fused_calls);
        if (shortconv_transpose_dispatches)
            std::fprintf(stderr, "CudaBackend: tiled shortconv dispatches=%zu\n", shortconv_transpose_dispatches);
        if (moe_sorted_prefill_calls)
            std::fprintf(stderr, "CudaBackend: extended NVFP4 prefill sort calls=%zu\n", moe_sorted_prefill_calls);
        if (moe_canonical_mma_prefill_calls)
            std::fprintf(stderr, "CudaBackend: canonical NVFP4 prefill MMA calls=%zu\n", moe_canonical_mma_prefill_calls);
        if (moe_activation_prepare_fused_calls)
            std::fprintf(stderr,"CudaBackend: NVFP4 activation preparation fused calls=%zu\n",moe_activation_prepare_fused_calls);
        if (moe_activation_prepare_packed_only_calls)
            std::fprintf(stderr,"CudaBackend: NVFP4 activation preparation packed-only calls=%zu\n",moe_activation_prepare_packed_only_calls);
        if (moe_bf16_down_output_prefill_calls)
            std::fprintf(stderr,"CudaBackend: NVFP4 BF16 down output prefill calls=%zu\n",moe_bf16_down_output_prefill_calls);
        if (moe_bf16_gate_output_prefill_calls)
            std::fprintf(stderr,"CudaBackend: NVFP4 BF16 gate output prefill calls=%zu\n",moe_bf16_gate_output_prefill_calls);
        if (moe_gate_early_inputs_prefill_calls)
            std::fprintf(stderr,"CudaBackend: NVFP4 early-input gate prefill calls=%zu\n",moe_gate_early_inputs_prefill_calls);
        if (moe_gate_vector_quantize_prefill_calls)
            std::fprintf(stderr,"CudaBackend: NVFP4 vector-quantize gate prefill calls=%zu\n",moe_gate_vector_quantize_prefill_calls);
        if(hc_up_reduce_fused_calls)
            std::fprintf(stderr,"CudaBackend: HC up reduce fused calls=%llu\n",static_cast<unsigned long long>(hc_up_reduce_fused_calls));
        if(hc_up_reduce_padded_calls||hc_up_reduce_wide_calls)
            std::fprintf(stderr,"CudaBackend: HC up reduce padded=%llu columns32=%llu\n",
                static_cast<unsigned long long>(hc_up_reduce_padded_calls),
                static_cast<unsigned long long>(hc_up_reduce_wide_calls));
        if(moe_mma_activation_fused_prefill_calls)
            std::fprintf(stderr,"CudaBackend: NVFP4 MMA activation fused prefill calls=%zu\n",moe_mma_activation_fused_prefill_calls);
        if (moe_parallel_batch_plan_prefill_calls)
            std::fprintf(stderr,"CudaBackend: NVFP4 parallel batch-plan prefill calls=%zu\n",moe_parallel_batch_plan_prefill_calls);
        if(moe_prefill_input_half_conversions)
            std::fprintf(stderr,"CudaBackend: MoE prefill input-half conversions=%zu consumers=%zu\n",
                moe_prefill_input_half_conversions,moe_prefill_input_half_consumers);
        if (cublas)
            cublasDestroy(cublas);
        for (void* allocation : device_allocations)
            cudaFree(allocation);
        mollm_cuda::destroy_device_buffer_pool(output_pool);
        for (void* allocation : managed_allocations)
            cudaFree(allocation);
        for (auto& entry : boundary_buffers)
            if (entry.second.data)
                cudaFree(entry.second.data);
        if (activation)
            cudaFree(activation);
        if (activation_fp16)
            cudaFree(activation_fp16);
        if (hc_prefill_half) cudaFree(hc_prefill_half);
        if (hc_prefill_reuses)
            std::fprintf(stderr,"CudaBackend: HC prefill activation reuses=%llu scratch_bytes=%zu\n",
                static_cast<unsigned long long>(hc_prefill_reuses),hc_prefill_half_bytes);
        if (hc_prefill_rms_half_calls)
            std::fprintf(stderr,"CudaBackend: HC prefill RMS half calls=%llu\n",
                static_cast<unsigned long long>(hc_prefill_rms_half_calls));
        if (quantized_weight_scratch)
            cudaFree(quantized_weight_scratch);
        if (diagnostic_weight_fp32)
            cudaFree(diagnostic_weight_fp32);
        if (output)
            cudaFree(output);
        if (argmax_partial)
            cudaFree(argmax_partial);
        if (argmax_result)
            cudaFree(argmax_result);
        if (attention_scores)
            cudaFree(attention_scores);
        if(qsa_query)cudaFree(qsa_query);
        if(qsa_scores)cudaFree(qsa_scores);
        if(qsa_mask_invalid)cudaFree(qsa_mask_invalid);
        if(qsa_topk_scores)cudaFree(qsa_topk_scores);
        if(qsa_topk_ids)cudaFree(qsa_topk_ids);
        if(qsa_streaming_topk_enqueues)
            std::fprintf(stderr,"CudaBackend: QSA streaming_topk=%zu workspace=%zu bytes\n",
                qsa_streaming_topk_enqueues,qsa_topk_scores_bytes+qsa_topk_ids_bytes);
        if(qsa_selected_scores)cudaFree(qsa_selected_scores);
        if(qsa_attention_shadow)cudaFree(qsa_attention_shadow);
        if(qsa_gqa_enqueues||qsa_split_enqueues)
            std::fprintf(stderr,"CudaBackend: QSA grouped=%zu split=%zu selected_workspace=%zu bytes\n",
                qsa_gqa_enqueues,qsa_split_enqueues,qsa_selected_scores_bytes);
        if(qsa_indexer_enqueues||qsa_attention_enqueues)
            std::fprintf(stderr,"CudaBackend: QSA indexer=%zu attention=%zu query_workspace=%zu score_workspace=%zu bytes\n",
                qsa_indexer_enqueues,qsa_attention_enqueues,qsa_query_bytes,qsa_scores_bytes);
        if(qsa_bf16_indexer_enqueues)std::fprintf(stderr,
            "CudaBackend: QSA diagnostic_bf16_post_projection=%zu\n",qsa_bf16_indexer_enqueues);
        if (attention_prefix_valid) cudaFree(attention_prefix_valid);
        if (attention_pv_handle) cublasDestroy(attention_pv_handle);
        if (attention_pv_values) cudaFree(attention_pv_values);
        if (attention_pv_output) cudaFree(attention_pv_output);
        if (attention_pv_pointers) cudaFree(attention_pv_pointers);
        if(attention_query_slice_enqueues)
            std::fprintf(stderr,"CudaBackend: query-sliced attention enqueues=%zu tile_rows=1024\n",attention_query_slice_enqueues);
        if(attention_head_lane_pair_enqueues)
            std::fprintf(stderr,"CudaBackend: native-probability head-lane-pair enqueues=%zu\n",attention_head_lane_pair_enqueues);
        if (attention_pv_gemm_enqueues)
            std::fprintf(stderr,"CudaBackend: native-probability GEMM PV enqueues=%zu values_bytes=%zu output_bytes=%zu pointer_bytes=%zu\n",
                attention_pv_gemm_enqueues,attention_pv_values_bytes,attention_pv_output_bytes,attention_pv_pointers_bytes);
        if (attention_prefix_enqueues)
            std::fprintf(stderr,"CudaBackend: validated-prefix attention enqueues=%zu\n",attention_prefix_enqueues);
        if (attention_prefix_long_shared_enqueues)
            std::fprintf(stderr,"CudaBackend: validated-prefix long shared scores enqueues=%zu\n",attention_prefix_long_shared_enqueues);
        if (attention_prefix_zero_pv_enqueues)
            std::fprintf(stderr,"CudaBackend: validated-prefix zero-PV enqueues=%zu\n",attention_prefix_zero_pv_enqueues);
        if (attention_prefix_pv_unroll_enqueues)
            std::fprintf(stderr,"CudaBackend: validated-prefix PV-unroll enqueues=%zu\n",attention_prefix_pv_unroll_enqueues);
        if (attention_prefix_pv_pair_enqueues)
            std::fprintf(stderr,"CudaBackend: validated-prefix PV-dimension-pair enqueues=%zu\n",attention_prefix_pv_pair_enqueues);
        if(attention_prefix_query_group_enqueues)
            std::fprintf(stderr,"CudaBackend: TEST-ONLY validated-prefix query-group enqueues=%llu\n",
                static_cast<unsigned long long>(attention_prefix_query_group_enqueues));
        if(attention_prefix_head_group_enqueues)
            std::fprintf(stderr,"CudaBackend: validated-prefix long-window head-group enqueues=%llu\n",
                static_cast<unsigned long long>(attention_prefix_head_group_enqueues));
        if (norm_scratch)
            cudaFree(norm_scratch);
        if (recurrent_scratch)
            cudaFree(recurrent_scratch);
        if (moe_scratch)
            cudaFree(moe_scratch);
        if (std::getenv("MOLLM_CUDA_PROFILE")) {
            std::fprintf(stderr, "\nCudaBackend MoE graphs: builds=%llu hits=%llu\n",
                static_cast<unsigned long long>(moe_graph_builds),
                static_cast<unsigned long long>(moe_graph_hits));
            std::fprintf(stderr, "\nCudaBackend operator coverage:\n");
            for (const auto& entry : native_ops)
                std::fprintf(stderr, "  native   %-24s %llu\n",
                             op_type_name(static_cast<OpType>(entry.first)),
                             static_cast<unsigned long long>(entry.second));
            for (const auto& entry : fallback_ops)
                std::fprintf(stderr, "  fallback %-24s %llu\n",
                             op_type_name(static_cast<OpType>(entry.first)),
                             static_cast<unsigned long long>(entry.second));
        }
    }

    bool reserve(void*& pointer, size_t& capacity, size_t requested, bool exact = false) {
        if (capacity >= requested)
            return true;
        // Growing attention context must not force cudaFree/cudaMalloc (and
        // their synchronization) on every token. Keep spare capacity bounded
        // to less than 2x the request, with a small minimum allocation.
        size_t expanded = std::max<size_t>(capacity, 4096);
        while (expanded < requested && expanded <= SIZE_MAX / 2) expanded *= 2;
        if (expanded < requested) expanded = requested;
        if (exact_scratch_growth || exact) expanded = requested;
        // A host-budget rejection is a safety decision, not a CUDA OOM.
        // Do not destroy the existing buffer or retry through the OOM path.
        if (!mollm_memory::cuda_host_allocation_allowed(expanded, "scratch growth"))
            return false;
        void* replacement = nullptr;
        cudaError_t status = cudaMalloc(&replacement, expanded);
        if (status != cudaSuccess) {
            // Spare capacity and allocating before freeing are optional on
            // memory-constrained devices. Retain the old exact-size behavior
            // as an OOM fallback rather than requiring two live buffers.
            cudaGetLastError();
            if (pointer) cudaFree(pointer);
            pointer = nullptr;
            capacity = 0;
            replacement = nullptr;
            expanded = requested;
            status = mollm_memory::cuda_host_allocation_allowed(expanded, "scratch exact fallback")
                ? cudaMalloc(&replacement, expanded) : cudaErrorMemoryAllocation;
        }
        if (!mollm_cuda::report_cuda(status, "cudaMalloc scratch"))
            return false;
        if (pointer) cudaFree(pointer);
        pointer = replacement;
        capacity = expanded;
        return true;
    }

    const DeviceWeight* find_weight(const Tensor& tensor) const {
        if (!tensor.device.buffer)
            return nullptr;
        const auto found = weights_by_device.find(tensor.device.buffer);
        return found == weights_by_device.end() ? nullptr : found->second;
    }

    bool upload_weight(Tensor& tensor, const void* cache_key,
                       const void* source, size_t bytes, cudaDataType type,
                       int n, int k,
                       WeightLayout layout = WeightLayout::Dense) {
        if (!cache_key || !source || bytes == 0)
            return false;
        auto found = weights.find(cache_key);
        if (found == weights.end()) {
            void* device = nullptr;
            if (!mollm_cuda::malloc_device(
                    &device, bytes, "cudaMalloc weight") ||
                !mollm_cuda::copy_memory(
                    device, source, bytes, cudaMemcpyHostToDevice,
                    "cudaMemcpy weight")) {
                if (device)
                    cudaFree(device);
                return false;
            }
            // CUDA's hot W4G32 kernels consume an unsigned nibble minus 8.
            // Bias the private device copy once so every decode FMA avoids
            // recovering two's-complement INT4 signs. The package and all
            // other backends retain the canonical representation.
            if (layout == WeightLayout::Q4Bg32Biased) {
                mollm_cuda::launch_bias_q4_g32_weight(
                    static_cast<Q4B8G32Block*>(device),
                    bytes / sizeof(Q4B8G32Block));
                if (!mollm_cuda::report_cuda(
                        cudaGetLastError(), "bias_q4_g32_weight_cuda")) {
                    cudaFree(device);
                    return false;
                }
            }
            mollm_cuda::Q4RowG32* row_g32 = nullptr;
            static const int row_major = [] {
                const char* option = std::getenv("MOLLM_CUDA_Q4_ROW_MAJOR");
                if (option && std::strcmp(option, "2") == 0) return 2;
                return option && std::strcmp(option, "1") == 0 ? 1 : 0;
            }();
            if (row_major && layout == WeightLayout::Q4Bg32Biased &&
                ((n == 2560 && k == 6144) || (n == 320 && k == 10240) ||
                 (row_major == 2 && n == 640 && k == 2560))) {
                const size_t row_bytes = static_cast<size_t>(n) * (k / 32) * sizeof(mollm_cuda::Q4RowG32);
                if (!mollm_cuda::malloc_device(reinterpret_cast<void**>(&row_g32), row_bytes, "cudaMalloc Q4 row-major")) {
                    cudaFree(device); return false;
                }
                mollm_cuda::launch_repack_q4_row_g32(static_cast<const Q4B8G32Block*>(device), row_g32, n, k);
                if (!mollm_cuda::report_cuda(cudaGetLastError(), "repack_q4_row_g32_cuda")) {
                    cudaFree(row_g32); cudaFree(device); return false;
                }
                device_allocations.push_back(row_g32);
                row_g32_bytes += row_bytes;
                ++row_g32_weights;
            }
            device_allocations.push_back(device);
            found = weights.emplace(
                cache_key,
                DeviceWeight{device, nullptr, type, n, k, 0, 0, layout, row_g32})
                        .first;
            weights_by_device.emplace(device, &found->second);
        }
        static const size_t prefill_cache_limit = [] {
            const char* value = std::getenv("MOLLM_CUDA_PREFILL_FP16_CACHE_MB");
            if (!value || !*value) return size_t{0};
            char* end = nullptr;
            const unsigned long long mb = std::strtoull(value, &end, 10);
            return end && !*end && mb <= 32768 ? static_cast<size_t>(mb) * 1024 * 1024 : size_t{0};
        }();
        auto& cached = found->second;
        static const bool cache_narrow = [] {
            const char* value=std::getenv("MOLLM_CUDA_PREFILL_FP16_CACHE_NARROW");
            return value && std::strcmp(value,"1")==0;
        }();
        static const bool cache_hc_down = [] {
            const char* value=std::getenv("MOLLM_CUDA_PREFILL_FP16_CACHE_HC_DOWN");
            return value && std::strcmp(value,"1")==0;
        }();
        static const bool cache_shared = [] {
            const char* value = std::getenv("MOLLM_CUDA_PREFILL_FP16_CACHE_SHARED");
            return value && !std::strcmp(value, "1");
        }();
        const bool shared_shape = (n == 640 && k == 2560) || (n == 2560 && k == 640);
        const bool cache_q4 = cached.layout == WeightLayout::Q4Bg32Biased ||
            cached.layout == WeightLayout::Q4Bg32Compact ||
            cached.layout == WeightLayout::Q4Bg32CompactInteger;
        const size_t cache_bytes = static_cast<size_t>(n) * k * sizeof(__half);
        if (prefill_cache_limit && cache_q4 && cached.n == n && cached.k == k &&
            (n > 4096 || (cache_narrow && n == 2560 && k == 6144) ||
                (cache_hc_down && n == 320 && k == 10240) || (cache_shared && shared_shape)) &&
            k > 0 && k % 32 == 0 && !cached.prefill_fp16 &&
            cache_bytes <= prefill_cache_limit - prefill_fp16_cache_bytes) {
            cudaStreamCaptureStatus capture;
            if (!mollm_cuda::report_cuda(cudaStreamIsCapturing(cudaStreamPerThread, &capture),
                    "prefill weight cache capture check")) return false;
            if (capture == cudaStreamCaptureStatusNone) {
                void* expanded = nullptr;
                if (!mollm_cuda::malloc_device(&expanded, cache_bytes, "prefill FP16 weight cache")) return false;
                device_allocations.push_back(expanded);
                if (cached.layout == WeightLayout::Q4Bg32Biased)
                    mollm_cuda::launch_prepare_q4_g32_dense_gemm(
                        static_cast<const Q4B8G32Block*>(cached.data),
                        static_cast<__half2*>(expanded), nullptr, nullptr, 0, n, k / 32);
                else mollm_cuda::launch_prepare_q4_compact_g32_dense_gemm(
                    static_cast<const mollm_cuda::Q4CompactG32*>(cached.data),
                    static_cast<__half2*>(expanded), nullptr, nullptr, 0, n, k / 32);
                if (!mollm_cuda::report_cuda(cudaGetLastError(), "prefill FP16 weight expansion")) return false;
                cached.prefill_fp16 = expanded;
                prefill_fp16_cache_bytes += cache_bytes;
                if (cache_shared && shared_shape) prefill_shared_cache_bytes += cache_bytes;
            }
        }
        tensor.device.buffer = found->second.data;
        tensor.device.offset = 0;
        return true;
    }

    bool upload_quantized_weight(
        Tensor& tensor, const void* cache_key, const void* source,
        size_t bytes, const float* host_scales, size_t scale_count,
        int n, int k, int group_size, int groups_per_row) {
        if (!host_scales || scale_count == 0 || group_size <= 0 ||
            groups_per_row <= 0 ||
            !upload_weight(
                tensor, cache_key, source, bytes, CUDA_R_8I, n, k,
                WeightLayout::Q8RowMajor))
            return false;
        auto found = weights.find(cache_key);
        if (found == weights.end())
            return false;
        DeviceWeight& prepared = found->second;
        if (!prepared.scales) {
            float* device_scales = nullptr;
            const size_t scale_bytes = scale_count * sizeof(float);
            if (!mollm_cuda::malloc_device(
                    reinterpret_cast<void**>(&device_scales), scale_bytes,
                    "cudaMalloc weight scales") ||
                !mollm_cuda::copy_memory(
                    device_scales, host_scales, scale_bytes,
                    cudaMemcpyHostToDevice, "cudaMemcpy weight scales")) {
                if (device_scales)
                    cudaFree(device_scales);
                return false;
            }
            device_allocations.push_back(device_scales);
            prepared.scales = device_scales;
        }
        prepared.group_size = group_size;
        prepared.groups_per_row = groups_per_row;
        return true;
    }

    bool run_matmul_device(const float* device_a, int lda,
                           const Tensor& weight, float* device_c, int ldc,
                           int m, int n, int k,
                           Activation activation_kind, int act_begin,
                           int act_len, cudaStream_t stream = nullptr,
                           __half* supplied_activation = nullptr, bool supplied_ready = false) {
        const DeviceWeight* prepared = find_weight(weight);
        if (!prepared || prepared->n != n || prepared->k != k || !device_a ||
            !device_c || ldc != n)
            return false;

        const bool valid_q8 =
            prepared->layout == WeightLayout::Q8RowMajor &&
            prepared->scales && prepared->group_size > 0 &&
            prepared->groups_per_row ==
                (k + prepared->group_size - 1) / prepared->group_size;
        const bool integer_compact = prepared->layout == WeightLayout::Q4Bg32CompactInteger;
        const bool compact_q4_g32 = prepared->layout == WeightLayout::Q4Bg32Compact || integer_compact;
        const bool valid_q4_g32 =
            (prepared->layout == WeightLayout::Q4Bg32Biased || compact_q4_g32) && k % 32 == 0;
        const bool valid_q4_g128 =
            prepared->layout == WeightLayout::Q4Bg128 && k % 128 == 0;
        const bool valid_quantized =
            valid_q8 || valid_q4_g32 || valid_q4_g128;
        // Non-default streams are used only by resident Q4 MoE graph capture.
        if (stream && !(valid_q4_g32 && m == 1 && activation_kind == Activation::NONE)) return false;
        static const bool small_batch = [] {
            const char* value = std::getenv("MOLLM_CUDA_Q4_SMALL_BATCH");
            return value && std::strcmp(value, "1") == 0;
        }();
        static const bool exact_batch = [] {
            const char* value = std::getenv("MOLLM_CUDA_Q4_SMALL_BATCH_EXACT");
            return value && std::strcmp(value, "1") == 0;
        }();
        if (exact_batch && valid_q4_g32 && m >= 2 && m <= 8) {
            static const bool exact_reuse = [] {
                const char* value = std::getenv("MOLLM_CUDA_Q4_SMALL_BATCH_EXACT_REUSE");
                return value && std::strcmp(value, "1") == 0;
            }();
            static const bool compact_reuse = [] {
                const char* value = std::getenv("MOLLM_CUDA_Q4_COMPACT_BATCH_REUSE");
                return value && std::strcmp(value, "1") == 0;
            }();
            if (exact_reuse && compact_reuse && compact_q4_g32) {
                if (!mollm_cuda::report_cuda(mollm_cuda::launch_q4_compact_small_batch_exact(
                        device_a, static_cast<const mollm_cuda::Q4CompactG32*>(prepared->data),
                        device_c, m, n, k, lda, ldc, integer_compact, stream), "q4_compact_batch_reuse"))
                    return false;
            } else if (exact_reuse && !compact_q4_g32 && !prepared->row_g32) {
                if (!mollm_cuda::report_cuda(mollm_cuda::launch_q4_g32_small_batch_exact(
                        device_a, static_cast<const Q4B8G32Block*>(prepared->data),
                        device_c, m, n, k, lda, ldc, stream), "q4_small_batch_exact_reuse"))
                    return false;
            } else {
                // Diagnostic: run each token through the exact decode projection,
                // including small router/shared weights, without FP16 expansion.
                for (int token = 0; token < m; ++token) {
                    const float* a = device_a + static_cast<size_t>(token) * lda;
                    float* c = device_c + static_cast<size_t>(token) * ldc;
                    if (integer_compact) {
                        if (!mollm_cuda::report_cuda(mollm_cuda::launch_q4_compact_integer_gemv(a,
                            static_cast<const mollm_cuda::Q4CompactG32*>(prepared->data), c, n, k, stream),
                            "compact_integer_gemv")) return false;
                    } else if (compact_q4_g32)
                        mollm_cuda::launch_q4_g32_compact_gemv(a,
                            static_cast<const mollm_cuda::Q4CompactG32*>(prepared->data), c, n, k, stream);
                    else if (prepared->row_g32 && reinterpret_cast<uintptr_t>(a) % alignof(float2) == 0)
                        mollm_cuda::launch_q4_row_g32_gemv(a, prepared->row_g32, c, n, k, stream);
                    else
                        mollm_cuda::launch_q4_g32_dense_gemv(a,
                            static_cast<const Q4B8G32Block*>(prepared->data), c, n, k, stream);
                }
                if (!mollm_cuda::report_cuda(cudaGetLastError(), "q4_small_batch_exact_decode")) return false;
            }
        } else if (small_batch && valid_q4_g32 && !compact_q4_g32 && m >= 2 && m <= 8 &&
            k >= 2048 && static_cast<int64_t>(n) * k >= 4 * 1024 * 1024) {
            if (!mollm_cuda::report_cuda(mollm_cuda::launch_q4_g32_small_batch(
                    device_a, static_cast<const Q4B8G32Block*>(prepared->data),
                    device_c, m, n, k, lda, ldc, stream), "q4_g32_small_batch_cuda"))
                return false;
        } else if (valid_quantized && m == 1) {
            const char* label = nullptr;
            if (valid_q8) {
                mollm_cuda::launch_q8_dense_gemv(
                    device_a, static_cast<const int8_t*>(prepared->data),
                    prepared->scales, prepared->group_size,
                    prepared->groups_per_row, device_c, n, k);
                label = "q8_dense_gemv_cuda";
            } else if (valid_q4_g32) {
                if (integer_compact) {
                    if (!mollm_cuda::report_cuda(mollm_cuda::launch_q4_compact_integer_gemv(device_a,
                        static_cast<const mollm_cuda::Q4CompactG32*>(prepared->data), device_c, n, k, stream),
                        "compact_integer_gemv")) return false;
                } else if (compact_q4_g32)
                    mollm_cuda::launch_q4_g32_compact_gemv(device_a,
                        static_cast<const mollm_cuda::Q4CompactG32*>(prepared->data),
                        device_c, n, k, stream);
                else if (prepared->row_g32 && reinterpret_cast<uintptr_t>(device_a) % alignof(float2) == 0)
                    mollm_cuda::launch_q4_row_g32_gemv(device_a, prepared->row_g32,
                        device_c, n, k, stream);
                else
                    mollm_cuda::launch_q4_g32_dense_gemv(
                        device_a,
                        static_cast<const Q4B8G32Block*>(prepared->data),
                        device_c, n, k, stream);
                label = "q4_g32_dense_gemv_cuda";
            } else {
                mollm_cuda::launch_q4_g128_dense_gemv(
                    device_a,
                    static_cast<const Q4B8G128Block*>(prepared->data),
                    device_c, n, k);
                label = "q4_g128_dense_gemv_cuda";
            }
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), label))
                return false;
        } else {
            const void* linear_weight = prepared->data;
            cudaDataType linear_weight_type = prepared->type;
            const size_t a_elements = static_cast<size_t>(m) * lda;
            bool activation_prepared = supplied_activation && supplied_ready;
            __half* half_activation = supplied_activation;
            if (valid_q4_g32 && prepared->prefill_fp16) {
                linear_weight = prepared->prefill_fp16;
                linear_weight_type = CUDA_R_16F;
                if ((n == 640 && k == 2560) || (n == 2560 && k == 640)) ++prefill_shared_cache_hits;
            } else if (valid_quantized) {
                const size_t weight_elements = static_cast<size_t>(n) * k;
                if (!reserve(
                        quantized_weight_scratch,
                        quantized_weight_scratch_bytes,
                        weight_elements * sizeof(__half)))
                    return false;
                const char* label = nullptr;
                if (valid_q8) {
                    mollm_cuda::launch_dequantize_q8_dense_weight(
                        static_cast<const int8_t*>(prepared->data),
                        prepared->scales, prepared->group_size,
                        prepared->groups_per_row,
                        static_cast<__half*>(quantized_weight_scratch),
                        weight_elements, k);
                    label = "dequantize_q8_dense_weight_cuda";
                } else if (valid_q4_g32) {
                    if (!half_activation && !reserve(
                            activation_fp16, activation_fp16_bytes,
                            a_elements * sizeof(__half)))
                        return false;
                    if (!half_activation) half_activation=static_cast<__half*>(activation_fp16);
                    if (compact_q4_g32)
                        mollm_cuda::launch_prepare_q4_compact_g32_dense_gemm(
                            static_cast<const mollm_cuda::Q4CompactG32*>(prepared->data),
                            reinterpret_cast<__half2*>(quantized_weight_scratch),
                            activation_prepared ? nullptr : device_a,
                            activation_prepared ? nullptr : half_activation,
                            activation_prepared ? 0 : a_elements, n, k / 32);
                    else mollm_cuda::launch_prepare_q4_g32_dense_gemm(
                        static_cast<const Q4B8G32Block*>(prepared->data),
                        reinterpret_cast<__half2*>(quantized_weight_scratch),
                        activation_prepared ? nullptr : device_a,
                        activation_prepared ? nullptr : half_activation,
                        activation_prepared ? 0 : a_elements, n, k / 32);
                    activation_prepared = true;
                    label = "prepare_q4_g32_dense_gemm_cuda";
                } else {
                    const size_t packed_count =
                        static_cast<size_t>((n + 7) / 8) * (k / 128) *
                        4 * 8 * 16;
                    mollm_cuda::launch_dequantize_q4_g128_dense_weight(
                        static_cast<const Q4B8G128Block*>(prepared->data),
                        reinterpret_cast<__half2*>(quantized_weight_scratch),
                        packed_count, n, k / 128);
                    label = "dequantize_q4_g128_dense_weight_cuda";
                }
                if (!mollm_cuda::report_cuda(
                        cudaGetLastError(), label))
                    return false;
                linear_weight = quantized_weight_scratch;
                linear_weight_type = CUDA_R_16F;
            } else if (prepared->layout != WeightLayout::Dense) {
                return false;
            }

            const void* gemm_activation = device_a;
            cudaDataType activation_type = CUDA_R_32F;
            static const bool fp32_reference = [] {
                const char* value = std::getenv("MOLLM_CUDA_DENSE_PREFILL_FP32_REFERENCE");
                return value && std::strcmp(value, "1") == 0;
            }();
            const bool use_fp32_reference = fp32_reference && m > 1 &&
                linear_weight_type == CUDA_R_16F;
            if (use_fp32_reference) {
                const size_t elements = static_cast<size_t>(n) * k;
                if (!reserve(diagnostic_weight_fp32, diagnostic_weight_fp32_bytes,
                             elements * sizeof(float))) return false;
                mollm_cuda::launch_fp16_to_fp32(static_cast<const __half*>(linear_weight),
                    static_cast<float*>(diagnostic_weight_fp32), elements);
                if (!mollm_cuda::report_cuda(cudaGetLastError(), "diagnostic FP16 weight to FP32")) return false;
                linear_weight = diagnostic_weight_fp32;
                linear_weight_type = CUDA_R_32F;
            }
            if (linear_weight_type == CUDA_R_16F) {
                if (!activation_prepared) {
                    if (!half_activation && !reserve(activation_fp16, activation_fp16_bytes,
                                 a_elements * sizeof(__half)))
                        return false;
                    if (!half_activation) half_activation=static_cast<__half*>(activation_fp16);
                    mollm_cuda::launch_fp32_to_fp16(
                        device_a, half_activation,
                        a_elements);
                    if (!mollm_cuda::report_cuda(
                            cudaGetLastError(), "fp32_to_fp16"))
                        return false;
                }
                gemm_activation = half_activation;
                activation_type = CUDA_R_16F;
            }

            bool lt_handled=false;
            static const bool hc_gate_row_tile = [] {
                const char* value = std::getenv("MOLLM_TEST_CUDA_HC_GATE_ROW_TILE");
                return value && !std::strcmp(value, "1024");
            }();
            static const bool hc_low_rank_row_tile = [] {
                const char* value = std::getenv("MOLLM_TEST_CUDA_HC_LOW_RANK_ROW_TILE");
                return value && !std::strcmp(value, "1024");
            }();
            if(prefill_lt&&!use_fp32_reference&&linear_weight_type==CUDA_R_16F&&activation_type==CUDA_R_16F&&
               !prefill_lt->run(cublas,linear_weight,gemm_activation,device_c,m,n,k,lda,lt_handled))return false;
            if (!lt_handled&&!mollm_cuda::run_dense_matmul(
                    cublas, linear_weight, linear_weight_type,
                    gemm_activation, activation_type, device_c, m, n, k,
                    lda, use_fp32_reference,
                    ((hc_gate_row_tile && n == 4) ||
                     (hc_low_rank_row_tile && n == 320)) ? 1024 : 0))
                return false;
        }

        if (activation_kind != Activation::NONE && act_len != 0) {
            const int begin = std::max(0, act_begin);
            const int end = act_len < 0 ? n : std::min(n, begin + act_len);
            mollm_cuda::launch_apply_activation(device_c, m, n,
                           static_cast<int>(activation_kind), begin, end);
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "apply_activation_cuda"))
                return false;
        }
        return true;
    }

    bool run_matmul(const float* host_a, int lda, const Tensor& weight,
                    float* host_c, int ldc, int m, int n, int k,
                    Activation activation_kind, int act_begin, int act_len) {
        const size_t a_bytes = static_cast<size_t>(m) * lda * sizeof(float);
        const size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);
        if (!host_a || !host_c ||
            !reserve(activation, activation_bytes, a_bytes) ||
            !reserve(output, output_bytes, c_bytes) ||
            !mollm_cuda::copy_memory(
                activation, host_a, a_bytes, cudaMemcpyHostToDevice,
                "cudaMemcpy activation") ||
            !run_matmul_device(
                static_cast<const float*>(activation), lda, weight,
                static_cast<float*>(output), ldc, m, n, k, activation_kind,
                act_begin, act_len) ||
            !mollm_cuda::copy_memory(
                host_c, output, c_bytes, cudaMemcpyDeviceToHost,
                "cudaMemcpy output"))
            return false;
        return true;
    }
};


namespace mollm::cuda_backend_detail {

template <typename T>
T* device_pointer(const Tensor& tensor) {
    if (!tensor.device.buffer)
        return nullptr;
    return reinterpret_cast<T*>(
        static_cast<uint8_t*>(tensor.device.buffer) + tensor.device.offset);
}

template <typename T>
const T* device_pointer_const(const Tensor& tensor) {
    if (!tensor.device.buffer)
        return nullptr;
    return reinterpret_cast<const T*>(
        static_cast<const uint8_t*>(tensor.device.buffer) +
        tensor.device.offset);
}

inline bool fp32_contiguous(const Tensor& tensor) {
    return tensor.prec == Precision::FP32 && tensor.is_contiguous();
}

inline bool same_shape(const Tensor& lhs, const Tensor& rhs) {
    for (int dimension = 0; dimension < 4; ++dimension)
        if (lhs.shape[dimension] != rhs.shape[dimension])
            return false;
    return true;
}

}  // namespace mollm::cuda_backend_detail
