#include "engine/engine.h"
#include "engine/byte_ranges.h"
#include "engine/weight_metadata.h"

#include "kernels/matmul.h"
#include "kernels/moe_ssd.h"
#include "kernels/trace.h"
#ifdef MOLLM_METAL
#include "engine/metal_backend.h"

namespace {
MetalBackend* as_metal(const std::unique_ptr<Backend>& backend) {
    return static_cast<MetalBackend*>(backend.get());
}
}  // namespace
#endif

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

namespace {

enum class PersistentInputKind {
    NONE,
    KV_KEY,
    KV_VALUE,
    GDN_STATE,
    GDN_CONV,
    STATE,
    ATT_SHIFT,
    FFN_SHIFT,
};

struct PersistentInput {
    PersistentInputKind kind = PersistentInputKind::NONE;
    int layer = -1;
};

bool parse_indexed_input_name(const std::string& name, const char* prefix,
                              int& layer) {
    const size_t prefix_len = std::strlen(prefix);
    if (name.compare(0, prefix_len, prefix) != 0)
        return false;

    const char* suffix = name.c_str() + prefix_len;
    char* end = nullptr;
    const long value = std::strtol(suffix, &end, 10);
    if (*suffix == '\0' || *end != '\0' || value < 0 ||
        value > std::numeric_limits<int>::max()) {
        return false;
    }
    layer = static_cast<int>(value);
    return true;
}

PersistentInput parse_persistent_input(const std::string& name) {
    PersistentInput input;
    if (parse_indexed_input_name(name, "cache_k", input.layer)) {
        input.kind = PersistentInputKind::KV_KEY;
    } else if (parse_indexed_input_name(name, "cache_v", input.layer)) {
        input.kind = PersistentInputKind::KV_VALUE;
    } else if (parse_indexed_input_name(name, "gdn_state", input.layer)) {
        input.kind = PersistentInputKind::GDN_STATE;
    } else if (parse_indexed_input_name(name, "gdn_conv", input.layer)) {
        input.kind = PersistentInputKind::GDN_CONV;
    } else if (parse_indexed_input_name(name, "rwkv_state", input.layer)) {
        input.kind = PersistentInputKind::STATE;
    } else if (parse_indexed_input_name(name, "rwkv_att_shift", input.layer)) {
        input.kind = PersistentInputKind::ATT_SHIFT;
    } else if (parse_indexed_input_name(name, "rwkv_ffn_shift", input.layer)) {
        input.kind = PersistentInputKind::FFN_SHIFT;
    }
    return input;
}

void initialize_input_tensor(Tensor& tensor, const GraphNode& node) {
    tensor.prec = node.out_prec;
    for (int dim = 0; dim < 4; ++dim)
        tensor.shape[dim] = node.out_shape[dim];
    tensor.compute_strides();
}

volatile uint8_t g_package_warmup_sink = 0;

}  // namespace

LLMEngine::~LLMEngine() {
    clear_model_state();
}

void LLMEngine::clear_model_state() {
    // Stop background readers before invalidating their package mapping and
    // before serializing the final events they may have emitted.
    moe_ssd_cache_.reset();
    if (!cfg_.trace_path.empty())
        mollm_trace::write();

    thread_pool_.park();

    // Graph tensors borrow package/shared-weight storage and backend buffers.
    // Drop the graphs and their execution pools before either owner.
    graph_prefill_ = Graph{};
    graph_decode_ = Graph{};
    exec_ctx_prefill_ = ExecContext{};
    exec_ctx_decode_ = ExecContext{};
    metal_backend_.reset();

    caches_.clear();
    embed_weight_ = nullptr;
    lm_head_weight_ = nullptr;
    persistent_pool_.clear();
    hidden_output_copy_.clear();

    for (const auto& range : locked_dense_ranges_) {
        munlock(range.first, range.second);
    }
    locked_dense_ranges_.clear();

    // Standalone graph weights own their mappings. Package graph tensors point
    // into package_mmap_ or package_weights_storage_ instead.
    weight_map_.clear();
    shared_weights_.clear();
    packed_weights_.clear();
    package_weight_map_.clear();
    moe_ssd_expert_ranges_.clear();

    if (package_mmap_) {
        munmap(package_mmap_, package_mmap_size_);
    }
    package_mmap_ = nullptr;
    package_mmap_size_ = 0;
    package_weights_base_ = nullptr;
    package_weights_size_ = 0;
    package_weights_storage_.clear();
    package_weights_resident_ = false;

    for (const auto& path : temp_files_) {
        std::remove(path.c_str());
    }
    temp_files_.clear();

    package_metadata_.clear();
    past_len_ = 0;
    cfg_ = EngineConfig{};
}

size_t LLMEngine::warmup_package_weights() {
    if (!package_weights_base_ || package_weights_size_ == 0)
        return 0;
    if (package_weights_resident_)
        return 0;

    long page_size_long = sysconf(_SC_PAGESIZE);
    size_t page_size = page_size_long > 0 ? (size_t)page_size_long : 4096;

    const uint8_t* p = package_weights_base_;
    const size_t len = package_weights_size_;

    const auto expert_ranges =
        mollm::detail::normalize_byte_ranges(moe_ssd_expert_ranges_, len);

#if defined(MADV_WILLNEED)
    // Preserve the eager readahead behaviour for ordinary mmap packages. In
    // SSD mode it would pull aggregate expert tensors into the kernel cache,
    // so dense-only warmup below intentionally relies on page touches alone.
    if (expert_ranges.empty()) {
        uintptr_t start = reinterpret_cast<uintptr_t>(package_weights_base_);
        uintptr_t aligned_start = (start / page_size) * page_size;
        size_t prefix = static_cast<size_t>(start - aligned_start);
        madvise(reinterpret_cast<void*>(aligned_start),
                prefix + package_weights_size_, MADV_WILLNEED);
    }
#endif

    uint8_t sink = 0;
    size_t warmed = 0;
    for (size_t off = 0; off < len; off += page_size) {
        if (mollm::detail::range_contains(expert_ranges, off))
            continue;
        sink ^= p[off];
        warmed += page_size;
    }
    if (len > 0 &&
        !mollm::detail::range_contains(expert_ranges, len - 1)) {
        sink ^= p[len - 1];
        warmed = std::min(len, warmed + 1);
    }
    g_package_warmup_sink ^= sink;
    return std::min(len, warmed);
}

size_t LLMEngine::lock_dense_package_weights() {
    if (!package_weights_mmap_backed() || !moe_ssd_cache_ ||
        !locked_dense_ranges_.empty()) {
        return 0;
    }

    // Touch first: mlock guarantees residency of mapped pages but does not
    // turn absent file pages into useful cache content by itself.
    const size_t warmed = warmup_package_weights();
    const size_t len = package_weights_size_;
    const uint8_t* base = package_weights_base_;
    const long system_page = sysconf(_SC_PAGESIZE);
    const size_t page_size =
        system_page > 0 ? static_cast<size_t>(system_page) : 4096;

    const auto expert_ranges =
        mollm::detail::normalize_byte_ranges(moe_ssd_expert_ranges_, len);

    auto lock_range = [&](uint64_t begin, uint64_t end) -> bool {
        if (begin >= end)
            return true;
        const uintptr_t raw_begin = reinterpret_cast<uintptr_t>(base) + begin;
        const uintptr_t raw_end = reinterpret_cast<uintptr_t>(base) + end;
        const uintptr_t aligned_begin = raw_begin / page_size * page_size;
        const uintptr_t aligned_end =
            (raw_end + page_size - 1) / page_size * page_size;
        const size_t bytes = aligned_end - aligned_begin;
        void* address = reinterpret_cast<void*>(aligned_begin);
        if (mlock(address, bytes) != 0)
            return false;
        locked_dense_ranges_.push_back({address, bytes});
        return true;
    };

    uint64_t cursor = 0;
    bool complete = true;
    for (const auto& range : expert_ranges) {
        if (!lock_range(cursor, range.begin)) {
            complete = false;
            break;
        }
        cursor = std::max(cursor, range.end);
    }
    if (complete)
        complete = lock_range(cursor, len);
    if (!complete) {
        const int err = errno;
        for (const auto& range : locked_dense_ranges_)
            munlock(range.first, range.second);
        locked_dense_ranges_.clear();
        std::fprintf(stderr, "Engine: could not lock dense mmap weights: %s\n",
                     std::strerror(err));
        return 0;
    }
    std::fprintf(stderr, "Engine: locked %.1f MB of dense mmap weights\n",
                 warmed / 1e6);
    return warmed;
}

// ---------------------------------------------------------------------------
// load_graph — load one graph and set up CONSTANT nodes from shared weights
// ---------------------------------------------------------------------------

bool LLMEngine::load_graph(Graph& g, ExecContext& exec_ctx, const char* path) {
    if (!graph_load(g, path)) {
        fprintf(stderr, "Engine: failed to load graph %s\n", path);
        return false;
    }

    // Resolve graph directory for relative weight paths
    std::string graph_dir = path;
    size_t slash = graph_dir.find_last_of("/\\");
    if (slash != std::string::npos)
        graph_dir = graph_dir.substr(0, slash + 1);
    else
        graph_dir = "./";

    for (auto& node : g.nodes) {
        if (node.op_type != OpType::CONSTANT || node.params.str.empty())
            continue;

        std::string wref = node.params.str[0];
        std::string wpath = wref;
        if (wpath[0] != '/' && (wpath.size() < 2 || wpath[1] != ':')) {
            wpath = graph_dir + wpath;
        }

        auto& t = g.runtime.tensors[node.id];

        // Helper: set up weight tensor.
        auto setup_weight = [&](void* data, Precision file_prec) {
            t.prec = file_prec;
            int64_t dim0 = node.out_shape[0]; // N
            int64_t dim1 = node.out_shape[1]; // K
            t.shape[0] = dim0;
            t.shape[1] = dim1;
            t.shape[2] = node.out_shape[2];
            t.shape[3] = node.out_shape[3];
            t.compute_strides();
            t.data = data;
            t.rowmajor_data = data;
            t.mem_type = MemoryType::EXTERNAL;
            t.is_interleaved = false;
            t.is_q4_repacked = false;
            t.is_q4_g128_packed = false;
            t.q8_repack_data = nullptr;
            t.q4_repack_data = nullptr;
            t.q4_g128_data = nullptr;
#ifdef MOLLM_METAL
            // Alias this weight into the registered device weight buffer NOW,
            // while t.data still points at the raw mmap region (before any CPU
            // load-time repacking rewrites t.data to an out-of-region buffer).
            // INT4 g128 weights need a second pass after quant metadata is set
            // (see finalize_metal_weight) to decode the packed block layout.
            if (metal_backend_)
                as_metal(metal_backend_)->wrap_weight(t);
#endif
        };

#ifdef MOLLM_METAL
        // Second pass for INT4 g128 weights: once quant metadata + q4_g128_data
        // are populated, decode the CPU Q4B8G128Block layout into a Metal raw
        // nibble+scale device buffer. No-op for non-INT4 weights.
        auto finalize_metal_weight = [&]() {
            if (metal_backend_) {
                bool is_aggregate_expert =
                    wref.find("_experts_") != std::string::npos;
                as_metal(metal_backend_)
                    ->wrap_weight_int4_g128(t, is_aggregate_expert);
            }
        };
#else
        auto finalize_metal_weight = [&]() {};
#endif

        // Package mode: resolve weight from package mmap via offset map.
        // The weight path (e.g. "./foo.weights") is looked up in
        // package_weight_map_ to find (offset, size) within the weights region.
        if (package_weights_base_ != nullptr) {
            auto pit = package_weight_map_.find(wref);
            if (pit == package_weight_map_.end())
                pit = package_weight_map_.find(wpath);
            if (pit != package_weight_map_.end()) {
                const MoeSsdTensorSource* ssd_source = nullptr;
                if (moe_ssd_cache_) {
                    ssd_source = moe_ssd_cache_->find_source(wref);
                    if (!ssd_source)
                        ssd_source = moe_ssd_cache_->find_source(wpath);
                }
                if (ssd_source) {
                    if (node.out_shape[0] != (int64_t)ssd_source->spec.rows *
                                                 ssd_source->spec.num_experts ||
                        node.out_shape[1] != ssd_source->spec.cols) {
                        fprintf(stderr,
                                "Engine: MoE SSD shape mismatch for %s\n",
                                wref.c_str());
                        return false;
                    }
                    setup_weight(nullptr, ssd_source->spec.precision);
                    t.moe_ssd_source = ssd_source;
                    t.group_size = ssd_source->spec.group_size;
                    t.groups_per_row = ssd_source->spec.groups_per_row;
                    t.num_groups = static_cast<uint32_t>(node.out_shape[0]) *
                                   ssd_source->spec.groups_per_row;
                    // Do not set data/scales or invoke load-time packing: that
                    // would fault or duplicate every routed expert tensor.
                    continue;
                }
                const uint8_t* blob =
                    package_weights_base_ + pit->second.first;
                MappedFile::Header weight_header;
                if (!MappedFile::parse_header(
                        blob, static_cast<size_t>(pit->second.second),
                        weight_header)) {
                    fprintf(stderr,
                            "Engine: package weight %s has an invalid header "
                            "or range\n",
                            wref.c_str());
                    return false;
                }
                void* data = const_cast<uint8_t*>(
                    blob + weight_header.data_offset);
                setup_weight(
                    data, static_cast<Precision>(weight_header.precision));
                const void* scales =
                    weight_header.scales_size
                        ? blob + weight_header.scales_offset
                        : nullptr;
                if (!mollm::detail::configure_weight_metadata(
                        t, weight_header, scales, wref.c_str())) {
                    return false;
                }

                prepare_matmul_weight(t, wref, data, packed_weights_,
                                      wref.find("embed_tokens") ==
                                          std::string::npos);
                finalize_metal_weight();
                continue;
            }
        }

        // File mode: resolve the shared mapping once, then configure the
        // graph tensor identically for cache hits and newly opened files.
        auto it = weight_map_.find(wpath);
        size_t weight_index = 0;
        if (it != weight_map_.end()) {
            weight_index = it->second;
        } else {
            MappedFile mf;
            if (!mf.open(wpath.c_str())) {
                fprintf(stderr, "Engine: failed to load weight %s\n",
                        wpath.c_str());
                return false;
            }
            weight_index = shared_weights_.size();
            weight_map_[wpath] = weight_index;
            shared_weights_.push_back(std::move(mf));
        }

        const MappedFile& mapped = shared_weights_[weight_index];
        setup_weight(
            const_cast<void*>(mapped.data()),
            static_cast<Precision>(mapped.header().precision));
        if (!mollm::detail::configure_weight_metadata(
                t, mapped.header(), mapped.scales(), wpath.c_str())) {
            return false;
        }

        // embed_tokens stays row-major for lookup; all other linear weights,
        // including lm_head, receive their CPU matmul layouts at load time.
        prepare_matmul_weight(t, wpath, t.data, packed_weights_,
                              node.params.str[0].find("embed_tokens") ==
                                  std::string::npos);
        finalize_metal_weight();
    }

    // Find special externally-driven weights. lm_head is stored explicitly in
    // the package and is treated as a normal matmul weight.
    for (auto& node : g.nodes) {
        if (node.op_type == OpType::CONSTANT && !node.params.str.empty()) {
            const std::string& wref = node.params.str[0];
            if (wref.find("embed_tokens") != std::string::npos) {
                embed_weight_ = &g.runtime.tensors[node.id];
            } else if (wref.find("lm_head") != std::string::npos) {
                lm_head_weight_ = &g.runtime.tensors[node.id];
            }
        }
    }

    exec_ctx.graph = &g;
    exec_ctx.pool = &g.runtime.pool;
    exec_ctx.thread_pool = &thread_pool_;
    prepare_execution(exec_ctx);

    return true;
}

// ---------------------------------------------------------------------------
// allocate_caches — allocate KV cache buffers with metadata header
// ---------------------------------------------------------------------------

void LLMEngine::allocate_caches(Graph& g, int n_ctx) {
    // Find persistent INPUT nodes and initialise their tensor shapes.
    // All recurrent state follows this path: KV cache, GDN state, and RWKV
    // state.
    for (auto& node : g.nodes) {
        if (node.op_type != OpType::INPUT || node.params.str.empty())
            continue;
        const std::string& name = node.params.str[0];
        const PersistentInput input = parse_persistent_input(name);
        if (input.kind == PersistentInputKind::NONE)
            continue;

        if (input.layer >= (int)caches_.size())
            caches_.resize(input.layer + 1);
        Tensor& tensor = g.runtime.tensors[node.id];
        initialize_input_tensor(tensor, node);
        CachePair& cache = caches_[input.layer];

        switch (input.kind) {
        case PersistentInputKind::KV_KEY:
            cache.k = &tensor;
            cache.k_head_dim = (int)node.out_shape[0];
            cache.k_num_heads = (int)node.out_shape[2];
            break;
        case PersistentInputKind::KV_VALUE:
            cache.v = &tensor;
            cache.v_head_dim = (int)node.out_shape[0];
            cache.v_num_heads = (int)node.out_shape[2];
            break;
        case PersistentInputKind::GDN_STATE:
            cache.gdn_state = &tensor;
            cache.gdn_v_dim = (int)node.out_shape[0];
            cache.gdn_k_dim = (int)node.out_shape[1];
            cache.gdn_num_heads = (int)node.out_shape[2];
            cache.is_linear_attn = true;
            break;
        case PersistentInputKind::GDN_CONV:
            cache.gdn_conv = &tensor;
            cache.gdn_conv_groups = (int)node.out_shape[0];
            cache.gdn_conv_kernel = (int)node.out_shape[1] + 1;
            break;
        case PersistentInputKind::STATE:
            cache.rwkv_state = &tensor;
            break;
        case PersistentInputKind::ATT_SHIFT:
            cache.rwkv_att_shift = &tensor;
            break;
        case PersistentInputKind::FFN_SHIFT:
            cache.rwkv_ffn_shift = &tensor;
            break;
        case PersistentInputKind::NONE:
            break;
        }
    }

    // Allocate cache/state data buffers from engine-owned persistent storage
    // (once, at load time). Graph runtime pools are execution-temporary only.
    // Allocate a persistent buffer for a cache tensor. On Metal, allocate a
    // device MTLBuffer (Shared storage, so the CacheMetadata header stays
    // host-readable/writable and SDPA appends device-side). On CPU, use the
    // engine persistent_pool_. Returns the host-visible pointer (buf).
    auto alloc_cache_buf = [&](Tensor* t, size_t total) -> void* {
#ifdef MOLLM_METAL
        if (metal_backend_) {
            as_metal(metal_backend_)->alloc_persistent(*t, total);
            return t->data; // = [buffer contents], host-visible
        }
#endif
        void* buf = persistent_pool_.acquire(total);
        t->data = buf;
        t->owner_id = persistent_pool_.id();
        t->storage_id = persistent_pool_.storage_id(buf);
        return buf;
    };

    for (auto& cp : caches_) {
        // Standard KV cache
        if (cp.k) {
            int hd = cp.k_head_dim;
            int nkv = cp.k_num_heads;
            size_t es = cp.k->element_size();
            size_t data_bytes = (size_t)hd * n_ctx * nkv * es;
            size_t total = CacheMetadata::SIZE + data_bytes;

            void* buf = alloc_cache_buf(cp.k, total);
            std::memset(buf, 0, CacheMetadata::SIZE);

            auto* meta = cache_meta(buf);
            meta->current_seq_len = 0;
            meta->max_seq_len = (uint64_t)n_ctx;
            meta->num_kv_heads = (uint64_t)nkv;
            meta->head_dim = (uint64_t)hd;

            cp.k->mem_type = MemoryType::POOLED;
            cp.k->shape[0] = (int64_t)total / (int64_t)es;
            cp.k->compute_strides();
        }
        if (cp.v) {
            int vd = cp.v_head_dim;
            int nkv = cp.v_num_heads;
            size_t es = cp.v->element_size();
            size_t data_bytes = (size_t)vd * n_ctx * nkv * es;
            size_t total = CacheMetadata::SIZE + data_bytes;

            void* buf = alloc_cache_buf(cp.v, total);
            std::memset(buf, 0, CacheMetadata::SIZE);

            auto* meta = cache_meta(buf);
            meta->current_seq_len = 0;
            meta->max_seq_len = (uint64_t)n_ctx;
            meta->num_kv_heads = (uint64_t)nkv;
            meta->v_head_dim = (uint64_t)vd;

            cp.v->mem_type = MemoryType::POOLED;
            cp.v->shape[0] = (int64_t)total / (int64_t)es;
            cp.v->compute_strides();
        }

        // GDN recurrent state: [v_dim, k_dim, num_heads] FP32
        // No CacheMetadata header — GDN state is a plain FP32 buffer.
        if (cp.gdn_state) {
            size_t data_bytes = (size_t)cp.gdn_v_dim * cp.gdn_k_dim *
                                cp.gdn_num_heads * sizeof(float);
            void* buf = alloc_cache_buf(cp.gdn_state,
                                        data_bytes); // Metal: sets device_data
            std::memset(buf, 0, data_bytes);
            cp.gdn_state->mem_type = MemoryType::POOLED;
            cp.gdn_state->shape[0] = (int64_t)cp.gdn_v_dim;
            cp.gdn_state->shape[1] = (int64_t)cp.gdn_k_dim;
            cp.gdn_state->shape[2] = (int64_t)cp.gdn_num_heads;
            cp.gdn_state->shape[3] = 1;
            cp.gdn_state->compute_strides();
        }
        // GDN conv state: [groups, kernel-1] FP32
        if (cp.gdn_conv) {
            int kernel_m1 = cp.gdn_conv_kernel - 1;
            size_t data_bytes =
                (size_t)cp.gdn_conv_groups * kernel_m1 * sizeof(float);
            void* buf = alloc_cache_buf(cp.gdn_conv,
                                        data_bytes); // Metal: sets device_data
            std::memset(buf, 0, data_bytes);
            cp.gdn_conv->mem_type = MemoryType::POOLED;
            cp.gdn_conv->shape[0] = (int64_t)cp.gdn_conv_groups;
            cp.gdn_conv->shape[1] = (int64_t)kernel_m1;
            cp.gdn_conv->shape[2] = 1;
            cp.gdn_conv->shape[3] = 1;
            cp.gdn_conv->compute_strides();
        }
        if (cp.rwkv_state) {
            size_t bytes = cp.rwkv_state->nbytes();
            void* buf = alloc_cache_buf(cp.rwkv_state, bytes);
            std::memset(buf, 0, bytes);
            cp.rwkv_state->mem_type = MemoryType::POOLED;
        }
        for (Tensor* shift : {cp.rwkv_att_shift, cp.rwkv_ffn_shift})
            if (shift) {
                size_t bytes = shift->nbytes();
                void* buf = alloc_cache_buf(shift, bytes);
                std::memset(buf, 0, bytes);
                shift->mem_type = MemoryType::POOLED;
            }
    }
}

// ---------------------------------------------------------------------------
// load — load both graphs, set up shared weights and caches
// ---------------------------------------------------------------------------

bool LLMEngine::load(const EngineConfig& cfg) {
    clear_model_state();
    if (load_impl(cfg))
        return true;

    // A failed load must leave the engine in the same empty, reusable state
    // as a freshly constructed instance.
    clear_model_state();
    return false;
}

bool LLMEngine::load_impl(const EngineConfig& cfg) {
    cfg_ = cfg;
    cfg_.num_threads = std::max(cfg_.num_threads, 1);
    mollm_trace::start(cfg_.trace_path);
    mollm_trace::set_thread_name("main");
    if (cfg_.moe_ssd_cache_bytes != 0) {
        if (cfg_.device != Device::CPU) {
            fprintf(stderr, "Engine: MoE SSD offload is currently CPU-only\n");
            return false;
        }
        // Expert tensors must remain unmaterialized; resident package loading
        // would copy the complete aggregate tensors before the cache can help.
        cfg_.weight_loading = WeightLoadingMode::MMAP;
    }
    thread_pool_.resize(cfg_.num_threads);
    exec_ctx_prefill_.thread_pool = &thread_pool_;
    exec_ctx_decode_.thread_pool = &thread_pool_;
    exec_ctx_prefill_.trace_label = "prefill";
    exec_ctx_decode_.trace_label = "decode";
    exec_ctx_prefill_.moe_cross_layer_prefetch = false;
    exec_ctx_decode_.moe_cross_layer_prefetch =
        cfg_.moe_ssd_global_cache && cfg_.moe_ssd_cross_layer_prefetch;
    exec_ctx_prefill_.backend = &cpu_backend_;
    exec_ctx_decode_.backend = &cpu_backend_;
    metal_backend_.reset();
    if (cfg_.device == Device::METAL) {
#ifdef MOLLM_METAL
        metal_backend_.reset(new MetalBackend());
        if (!as_metal(metal_backend_)->available()) {
            fprintf(stderr,
                    "Engine: Metal backend unavailable; falling back to CPU\n");
            metal_backend_.reset();
            cfg_.device = Device::CPU;
        } else {
            exec_ctx_prefill_.backend = metal_backend_.get();
            exec_ctx_decode_.backend = metal_backend_.get();
            // The Metal backend wraps the weight region via
            // newBufferWithBytesNoCopy. mmap'd file-backed pages are NOT
            // reliably GPU-accessible that way (the GPU reads zeros), so force
            // RESIDENT weights when Metal is active.
            if (cfg_.weight_loading == WeightLoadingMode::MMAP) {
                fprintf(stderr,
                        "Engine: Metal backend requires resident weights; "
                        "ignoring --mmap\n");
                cfg_.weight_loading = WeightLoadingMode::RESIDENT;
            }
        }
#else
        fprintf(stderr,
                "Engine: built without MOLLM_METAL; using CPU backend\n");
        cfg_.device = Device::CPU;
#endif
    }
    exec_ctx_prefill_.reuse_static_workspace = false;
    exec_ctx_prefill_.reuse_same_shape_workspace = true;
    exec_ctx_decode_.reuse_static_workspace = true;
    exec_ctx_decode_.reuse_same_shape_workspace = false;

    // Load the .mollm package (sets up weights mmap, extracts graphs to temp
    // files, parses metadata for weight offset map).
    if (cfg.package_path.empty()) {
        fprintf(stderr,
                "Engine: package_path is required (use .mollm package)\n");
        return false;
    }
    std::string pf_path, dc_path;
    {
        std::string tok_tmp;
        if (!load_package(cfg.package_path, pf_path, dc_path, tok_tmp)) {
            return false;
        }
        if (!tok_tmp.empty()) {
            cfg_.tokenizer_path = tok_tmp;
        }
    }

    if (cfg_.lock_dense_weights && moe_ssd_cache_) {
        lock_dense_package_weights();
    }
#ifdef MOLLM_METAL
    // Wrap the whole package weight region as one zero-copy MTLBuffer so each
    // weight tensor can alias it via device_offset (set in setup_weight).
    if (metal_backend_ && package_weights_base_ && package_weights_size_) {
        if (!as_metal(metal_backend_)
                 ->register_weight_region(
                     const_cast<uint8_t*>(package_weights_base_),
                     package_weights_size_)) {
            fprintf(stderr,
                    "Engine: failed to register weight region with Metal\n");
        }
    }
#endif

    // Load prefill graph first (establishes shared weights)
    const std::string& pf_path_load =
        cfg_.use_decode_as_prefill ? dc_path : pf_path;
    if (!load_graph(graph_prefill_, exec_ctx_prefill_, pf_path_load.c_str())) {
        return false;
    }
    // RWKV state and token-shift buffers are persistent INPUT tensors, so the
    // regular release queue can safely manage all materialized intermediates.
    // Keeping every same-shape node output resident defeats that liveness and
    // costs multiple gigabytes for a 256-token prefill graph.
    for (const auto& node : graph_prefill_.nodes) {
        if (node.op_type == OpType::RWKV7 ||
            node.op_type == OpType::RWKV_TOKEN_SHIFT) {
            exec_ctx_prefill_.reuse_same_shape_workspace = false;
            break;
        }
    }

    // Override config from graph metadata (takes precedence over CLI defaults)
    auto get_meta = [&](const char* key, const char* def) -> const char* {
        auto it = graph_prefill_.metadata.find(key);
        return it != graph_prefill_.metadata.end() ? it->second.c_str() : def;
    };
    auto get_meta_int = [&](const char* key, int def) -> int {
        const char* v = get_meta(key, nullptr);
        return v ? std::atoi(v) : def;
    };
    auto get_meta_float = [&](const char* key, float def) -> float {
        const char* v = get_meta(key, nullptr);
        return v ? (float)std::atof(v) : def;
    };
    cfg_.rope_dim = get_meta_int("rope_dim", cfg_.rope_dim);
    cfg_.rope_theta = get_meta_float("rope_theta", cfg_.rope_theta);

    allocate_caches(graph_prefill_, cfg.n_ctx);

    // Load decode graph (reuses shared weights via weight_map_)
    if (!load_graph(graph_decode_, exec_ctx_decode_, dc_path.c_str())) {
        return false;
    }

    if (!embed_weight_ || !embed_weight_->data) {
        fprintf(stderr,
                "Engine: package missing explicit embed_tokens weight\n");
        return false;
    }
    if (!lm_head_weight_ || !lm_head_weight_->data) {
        fprintf(stderr, "Engine: package missing explicit lm_head weight; "
                        "reconvert with current converter\n");
        return false;
    }

    // Migrate cache tensor pointers from prefill graph to decode graph.
    // Done once at load time — the physical cache buffers are shared across
    // both graphs for the engine's entire lifetime.
    for (auto& node : graph_decode_.nodes) {
        if (node.op_type != OpType::INPUT || node.params.str.empty())
            continue;
        const std::string& name = node.params.str[0];
        const PersistentInput input = parse_persistent_input(name);
        if (input.kind == PersistentInputKind::NONE ||
            input.layer >= (int)caches_.size()) {
            continue;
        }

        const CachePair& cache = caches_[input.layer];
        const Tensor* source = nullptr;
        switch (input.kind) {
        case PersistentInputKind::KV_KEY:
            source = cache.k;
            break;
        case PersistentInputKind::KV_VALUE:
            source = cache.v;
            break;
        case PersistentInputKind::GDN_STATE:
            source = cache.gdn_state;
            break;
        case PersistentInputKind::GDN_CONV:
            source = cache.gdn_conv;
            break;
        case PersistentInputKind::STATE:
            source = cache.rwkv_state;
            break;
        case PersistentInputKind::ATT_SHIFT:
            source = cache.rwkv_att_shift;
            break;
        case PersistentInputKind::FFN_SHIFT:
            source = cache.rwkv_ffn_shift;
            break;
        case PersistentInputKind::NONE:
            break;
        }
        if (source)
            graph_decode_.runtime.tensors[node.id] = *source;
    }

    reset();
    return true;
}
