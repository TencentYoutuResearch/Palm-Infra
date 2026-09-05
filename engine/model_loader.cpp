#include "engine/engine.h"
#include "engine/byte_ranges.h"
#include "engine/weight_metadata.h"

#include "kernels/matmul.h"
#include "kernels/moe_ssd.h"
#include "kernels/trace.h"
#include "kernels/cpu_platform.h"
#ifdef MOLLM_METAL
#include "engine/metal_backend.h"
#endif
#ifdef MOLLM_CUDA
#include "engine/cuda_backend.h"
#endif

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_set>
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
    GDN_CHECKPOINT,
    GDN_CONV_CHECKPOINT,
    STATE,
    ATT_SHIFT,
    FFN_SHIFT,
    AUX_STATE,
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
    } else if (parse_indexed_input_name(name, "gdn_checkpoint", input.layer)) {
        input.kind = PersistentInputKind::GDN_CHECKPOINT;
    } else if (parse_indexed_input_name(
                   name, "gdn_conv_checkpoint", input.layer)) {
        input.kind = PersistentInputKind::GDN_CONV_CHECKPOINT;
    } else if (parse_indexed_input_name(name, "rwkv_state", input.layer)) {
        input.kind = PersistentInputKind::STATE;
    } else if (parse_indexed_input_name(name, "rwkv_att_shift", input.layer)) {
        input.kind = PersistentInputKind::ATT_SHIFT;
    } else if (parse_indexed_input_name(name, "rwkv_ffn_shift", input.layer)) {
        input.kind = PersistentInputKind::FFN_SHIFT;
    } else if (parse_indexed_input_name(name, "aux_state", input.layer)) {
        input.kind = PersistentInputKind::AUX_STATE;
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
    graph_vision_ = Graph{};
    graph_mtp_ = Graph{};
    graph_mtp_verify_ = Graph{};
    exec_ctx_prefill_ = ExecContext{};
    exec_ctx_decode_ = ExecContext{};
    exec_ctx_vision_ = ExecContext{};
    exec_ctx_mtp_ = ExecContext{};
    accelerator_backend_.reset();
    exec_ctx_mtp_verify_ = ExecContext{};

    caches_.clear();
    mtp_caches_.clear();
    auxiliary_states_.clear();
    embed_weight_ = nullptr;
    lm_head_weight_ = nullptr;
    vision_pos_embed_ = nullptr;
    persistent_pool_.clear();
    hidden_output_copy_.clear();
    mtp_hidden_output_copy_.clear();
    mtp_draft_hidden_device_ = Tensor{};
    mtp_pending_hidden_.clear();
    mtp_target_hidden_copy_.clear();
    mtp_stats_ = MtpStats{};

    for (const auto& range : locked_dense_ranges_) {
        munlock(range.first, range.second);
    }
    locked_dense_ranges_.clear();

    // Standalone graph weights own their mappings. Package graph tensors point
    // into package_mmap_ or package_weights_storage_ instead.
    weight_map_.clear();
    shared_weights_.clear();
    packed_weights_.clear();
    prepared_weights_.clear();
    package_weight_map_.clear();
    mmap_weight_exclusion_ranges_.clear();

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
    mtp_past_len_ = 0;
    rope_position_delta_ = 0;
    multimodal_position_ids_.clear();
    active_vision_ = nullptr;
    active_image_token_id_ = -1;
    active_vision_cursor_ = 0;
    cfg_ = EngineConfig{};
    sampler_.configure(cfg_.sampling, nullptr, true);
    sampler_.reset();
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
        mollm::detail::normalize_byte_ranges(
            mmap_weight_exclusion_ranges_, len);

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
        mollm::detail::normalize_byte_ranges(
            mmap_weight_exclusion_ranges_, len);

#if defined(MADV_DONTNEED)
    // Expert aggregates and dense weights with complete CPU sidecars are no
    // longer read through this mapping. Graph loading touched the latter to
    // build their packed representation, so proactively release those clean
    // file pages before locking the real dense working set and expert cache.
    for (const auto& range : expert_ranges) {
        if (range.begin >= range.end)
            continue;
        const uintptr_t raw_begin =
            reinterpret_cast<uintptr_t>(base) + range.begin;
        const uintptr_t raw_end =
            reinterpret_cast<uintptr_t>(base) + range.end;
        const uintptr_t aligned_begin =
            raw_begin / page_size * page_size;
        const uintptr_t aligned_end =
            (raw_end + page_size - 1) / page_size * page_size;
        madvise(
            reinterpret_cast<void*>(aligned_begin),
            aligned_end - aligned_begin, MADV_DONTNEED);
    }
#endif

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
    size_t locked_sidecar_bytes = 0;
    size_t fp8_q8_sidecar_bytes = 0;
    size_t fp8_bf16_sidecar_bytes = 0;
    size_t other_sidecar_bytes = 0;
    if (complete) {
        // Matmul sidecars are the actual CPU working set for repacked
        // FP8/W8/W4/FP16 weights. Locking only their now-unused mmap sources
        // lets macOS compress or page these heap buffers under a large expert
        // cache, which makes the same GEMV several times slower in a full
        // layer rotation than in isolation.
        for (auto& [key, buffer] : packed_weights_) {
            (void)key;
            if (buffer.empty())
                continue;
            if (mlock(buffer.data(), buffer.size()) != 0) {
                complete = false;
                break;
            }
            locked_dense_ranges_.push_back(
                {buffer.data(), buffer.size()});
            locked_sidecar_bytes += buffer.size();
            constexpr char q8_suffix[] = "#fp8_q8dot";
            constexpr char bf16_suffix[] = "#fp8_bf16_fp16";
            const bool is_fp8_q8 =
                key.size() >= sizeof(q8_suffix) - 1 &&
                key.compare(key.size() - (sizeof(q8_suffix) - 1),
                            sizeof(q8_suffix) - 1, q8_suffix) == 0;
            const bool is_fp8_bf16 =
                key.size() >= sizeof(bf16_suffix) - 1 &&
                key.compare(key.size() - (sizeof(bf16_suffix) - 1),
                            sizeof(bf16_suffix) - 1, bf16_suffix) == 0;
            if (is_fp8_q8)
                fp8_q8_sidecar_bytes += buffer.size();
            else if (is_fp8_bf16)
                fp8_bf16_sidecar_bytes += buffer.size();
            else
                other_sidecar_bytes += buffer.size();
        }
        for (auto& [key, prepared] : prepared_weights_) {
            (void)key;
            if (!complete)
                break;
            for (auto& buffer : prepared.layouts) {
                if (buffer.empty())
                    continue;
                if (mlock(buffer.data(), buffer.size()) != 0) {
                    complete = false;
                    break;
                }
                locked_dense_ranges_.push_back(
                    {buffer.data(), buffer.size()});
                locked_sidecar_bytes += buffer.size();
                other_sidecar_bytes += buffer.size();
            }
        }
    }
    if (!complete) {
        const int err = errno;
        for (const auto& range : locked_dense_ranges_)
            munlock(range.first, range.second);
        locked_dense_ranges_.clear();
        std::fprintf(stderr, "Engine: could not lock dense mmap weights: %s\n",
                     std::strerror(err));
        return 0;
    }
    std::fprintf(
        stderr,
        "Engine: locked %.1f MB of dense mmap weights and %.1f MB of CPU "
        "sidecars (FP8-Q8 %.1f MB, FP8-BF16 %.1f MB, other %.1f MB)\n",
        warmed / 1e6, locked_sidecar_bytes / 1e6,
        fp8_q8_sidecar_bytes / 1e6, fp8_bf16_sidecar_bytes / 1e6,
        other_sidecar_bytes / 1e6);
    return warmed + locked_sidecar_bytes;
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

    // DSV4_GROUPED_LINEAR consumes the checkpoint's native FP8 bytes
    // directly. Its specialized kernel deliberately emulates the reference
    // BF16 materialization and does not use the generic Q8-dot sidecar.
    std::unordered_set<uint32_t> native_fp8_weight_nodes;
    for (const auto& node : g.nodes) {
        if (node.op_type == OpType::DSV4_GROUPED_LINEAR &&
            node.inputs.size() >= 2) {
            native_fp8_weight_nodes.insert(node.inputs[1]);
        }
    }

    const bool build_cpu_weight_sidecars =
        !accelerator_backend_ ||
        exec_ctx.backend != accelerator_backend_.get() ||
        accelerator_backend_->wants_cpu_weight_sidecars();

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
            t.device_data = nullptr;
            t.device_offset = 0;
            t.scales_device_data = nullptr;
            t.scales_device_offset = 0;
            t.mem_type = MemoryType::EXTERNAL;
            t.is_interleaved = false;
            t.is_q4_repacked = false;
            t.is_q4_g32_packed = false;
            t.is_q4_g128_packed = false;
            t.q8_repack_data = nullptr;
            t.fp8_bf16_fp16_data = nullptr;
            t.fp32_bf16_data = nullptr;
            t.q4_repack_data = nullptr;
            t.q4_g32_data = nullptr;
            t.q4_g128_data = nullptr;
            t.prepared_weight = nullptr;
            t.prepared_weight_row_offset = 0;
            // Prepare accelerator storage while t.data still points at the raw
            // package bytes. CPU load-time packing may replace t.data later.
            if (accelerator_backend_ &&
                exec_ctx.backend == accelerator_backend_.get() &&
                t.prec != Precision::INT4 &&
                t.prec != Precision::INT8)
                accelerator_backend_->wrap_weight(t);
        };

        // Quantized layout metadata is only known after parsing the weight
        // header, so let the accelerator perform its second preparation pass.
        auto finalize_accelerator_weight = [&]() {
            if (accelerator_backend_ &&
                exec_ctx.backend == accelerator_backend_.get()) {
                if (t.prec == Precision::INT8)
                    accelerator_backend_->wrap_weight(t);
                const bool is_aggregate_expert =
                    mollm::detail::is_routed_expert_aggregate_ref(wref);
                accelerator_backend_->wrap_weight_int4(
                    t, is_aggregate_expert);
            }
        };

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
                const void* scales = weight_header.scales_size
                    ? blob + weight_header.scales_offset : nullptr;
                const Precision precision =
                    static_cast<Precision>(weight_header.precision);
                const bool cpu_weight = build_cpu_weight_sidecars;
                auto align_package_region =
                    [&](const void* source, uint64_t size, size_t alignment,
                        const char* suffix) -> void* {
                    if (!source || alignment <= 1 ||
                        reinterpret_cast<uintptr_t>(source) % alignment == 0) {
                        return const_cast<void*>(source);
                    }
                    if (size > SIZE_MAX - (alignment - 1))
                        return nullptr;
                    auto& storage = packed_weights_[wref + suffix];
                    if (storage.empty()) {
                        storage.resize(static_cast<size_t>(size) +
                                       alignment - 1);
                        const uintptr_t base =
                            reinterpret_cast<uintptr_t>(storage.data());
                        const uintptr_t aligned =
                            (base + alignment - 1) & ~(alignment - 1);
                        std::memcpy(reinterpret_cast<void*>(aligned), source,
                                    static_cast<size_t>(size));
                    }
                    const uintptr_t base =
                        reinterpret_cast<uintptr_t>(storage.data());
                    return reinterpret_cast<void*>(
                        (base + alignment - 1) & ~(alignment - 1));
                };
                if (cpu_weight) {
                    size_t data_alignment = 1;
                    if (precision == Precision::FP32 ||
                        precision == Precision::INT32 ||
                        (precision == Precision::INT4 &&
                         (weight_header.flags &
                          (MappedFile::FLAG_INT4_BG32 |
                           MappedFile::FLAG_INT4_BG128)))) {
                        data_alignment = alignof(float);
                    } else if (precision == Precision::FP16) {
                        data_alignment = alignof(uint16_t);
                    }
                    data = align_package_region(
                        data, weight_header.data_size, data_alignment,
                        "#aligned_data");
                    if (scales &&
                        (precision == Precision::INT8 ||
                         precision == Precision::INT4)) {
                        scales = align_package_region(
                            scales, weight_header.scales_size, alignof(float),
                            "#aligned_scales");
                    }
                    if (!data || (weight_header.scales_size && !scales)) {
                        fprintf(stderr,
                                "Engine: failed to align package weight %s\n",
                                wref.c_str());
                        return false;
                    }
                }
                if ((weight_header.flags &
                     MappedFile::FLAG_EXPERT_INTERLEAVED) != 0) {
                    const auto experts_it =
                        package_metadata_.find("num_experts");
                    const int experts = experts_it != package_metadata_.end()
                        ? std::atoi(experts_it->second.c_str()) : 0;
                    const uint64_t rows = weight_header.shape[0];
                    const uint64_t cols = weight_header.shape[1];
                    if (weight_header.precision !=
                            static_cast<uint32_t>(Precision::MXFP4) ||
                        experts <= 0 || rows % experts != 0 ||
                        cols % 32 != 0 ||
                        rows > std::numeric_limits<uint64_t>::max() / cols) {
                        fprintf(stderr,
                                "Engine: unsupported expert-interleaved "
                                "weight %s\n", wref.c_str());
                        return false;
                    }
                    const uint64_t elements = rows * cols;
                    const uint64_t data_bytes = elements / 2;
                    const uint64_t scale_bytes = elements / 32;
                    if (data_bytes > SIZE_MAX || scale_bytes > SIZE_MAX ||
                        data_bytes + scale_bytes != weight_header.data_size) {
                        fprintf(stderr,
                                "Engine: invalid expert-interleaved payload "
                                "for %s\n", wref.c_str());
                        return false;
                    }
                    auto [storage_it, inserted] = packed_weights_.try_emplace(
                        wref + "#expert_native");
                    std::vector<uint8_t>& storage = storage_it->second;
                    if (inserted || storage.empty()) {
                        storage.resize(static_cast<size_t>(
                            data_bytes + scale_bytes));
                        const size_t expert_data =
                            static_cast<size_t>(data_bytes / experts);
                        const size_t expert_scales =
                            static_cast<size_t>(scale_bytes / experts);
                        const size_t expert_stride =
                            expert_data + expert_scales;
                        const uint8_t* source =
                            blob + weight_header.data_offset;
                        uint8_t* data_out = storage.data();
                        uint8_t* scales_out =
                            storage.data() + data_bytes;
                        for (int expert = 0; expert < experts; ++expert) {
                            std::memcpy(
                                data_out + expert * expert_data,
                                source + expert * expert_stride,
                                expert_data);
                            std::memcpy(
                                scales_out + expert * expert_scales,
                                source + expert * expert_stride + expert_data,
                                expert_scales);
                        }
                    }
                    data = storage.data();
                    scales = storage.data() + data_bytes;
                    weight_header.flags &=
                        ~MappedFile::FLAG_EXPERT_INTERLEAVED;
                    weight_header.data_size = data_bytes;
                    weight_header.scales_size = scale_bytes;
                }
                setup_weight(data, precision);
                if (!mollm::detail::configure_weight_metadata(
                        t, weight_header, scales, wref.c_str())) {
                    return false;
                }

                const bool lookup_table =
                    wref.find("embed_tokens") != std::string::npos ||
                    wref.find(
                        "ple_embedding_ngram_embedding_weight") !=
                        std::string::npos ||
                    wref.find("vision_pos_embed.weights") !=
                        std::string::npos;
                // Resident accelerators consume package-native weights or
                // their own prepared layouts. Avoid duplicating CPU sidecars
                // unless the selected backend still needs reference fallback.
                if (build_cpu_weight_sidecars) {
                    prepare_matmul_weight(
                        t, wref, data, packed_weights_, prepared_weights_,
                        !lookup_table,
                        native_fp8_weight_nodes.count(node.id) == 0);
                    if (native_fp8_weight_nodes.count(node.id) != 0)
                        prepare_fp8_bf16_fp16_weight(
                            t, wref, data, packed_weights_);
                }
                // Once a CPU sidecar owns every value needed by the selected
                // kernel, the original package pages are no longer used at
                // inference time. Exclude the whole weight blob from dense
                // warmup/mlock. Gate this on the prepared pointers rather
                // than the architecture or weight name so unsupported
                // platforms retain their raw fallback.
                const bool complete_cpu_sidecar =
                    (t.prec == Precision::FP8_E4M3 &&
                     (t.q8_repack_data || t.fp8_bf16_fp16_data)) ||
                    (t.prec == Precision::INT8 && t.q8_repack_data &&
                     t.scales) ||
                    (t.prec == Precision::FP32 && t.fp32_bf16_data) ||
                    (t.prec == Precision::FP16 && t.is_interleaved &&
                     t.data != data);
                if (complete_cpu_sidecar) {
                    mmap_weight_exclusion_ranges_.push_back(
                        {pit->second.first, pit->second.second});
                }
                // PLE is a 47+ GiB random-access embedding table. Keep its
                // package mapping available for lookup, but do not fault and
                // mlock every page during dense-weight warmup.
                if (wref.find(
                        "ple_embedding_ngram_embedding_weight") !=
                    std::string::npos) {
                    mmap_weight_exclusion_ranges_.push_back(
                        {pit->second.first, pit->second.second});
                }
                finalize_accelerator_weight();
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

        // Lookup tables stay row-major; linear weights, including lm_head,
        // receive their CPU matmul layouts at load time.
        const bool lookup_table =
            t.prec == Precision::INT32 ||
            node.params.str[0].find("embed_tokens") != std::string::npos ||
            node.params.str[0].find("vision_pos_embed.weights") !=
                std::string::npos;
        if (build_cpu_weight_sidecars) {
            prepare_matmul_weight(
                t, wpath, t.data, packed_weights_, prepared_weights_,
                !lookup_table,
                native_fp8_weight_nodes.count(node.id) == 0);
            if (native_fp8_weight_nodes.count(node.id) != 0)
                prepare_fp8_bf16_fp16_weight(
                    t, wpath, t.data, packed_weights_);
        }
        finalize_accelerator_weight();
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
            } else if (wref.find("vision_pos_embed.weights") !=
                       std::string::npos) {
                vision_pos_embed_ = &g.runtime.tensors[node.id];
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

bool LLMEngine::allocate_caches(Graph& g, ExecContext& exec_ctx,
                                std::vector<CachePair>& caches, int n_ctx) {
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

        Tensor& tensor = g.runtime.tensors[node.id];
        initialize_input_tensor(tensor, node);
        if (input.kind == PersistentInputKind::AUX_STATE) {
            auxiliary_states_[input.layer] = &tensor;
            continue;
        }
        // Cache storage is selected by the active execution backend. This
        // keeps device policy out of the loader and lets correctness backends
        // request FP32 while their native attention path is incomplete.
        if (input.kind == PersistentInputKind::KV_KEY ||
            input.kind == PersistentInputKind::KV_VALUE) {
            tensor.prec = exec_ctx.backend->kv_cache_precision(
                tensor.prec);
            tensor.compute_strides();
        }
        if (input.layer >= (int)caches.size())
            caches.resize(input.layer + 1);
        CachePair& cache = caches[input.layer];

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
            // Serialized Qwen3.5 graphs historically label the convolution
            // state FP16, but every current short-convolution kernel
            // reads and updates it as float. Make the physical contract
            // explicit before sizing backend storage.
            tensor.prec = Precision::FP32;
            tensor.compute_strides();
            cache.gdn_conv_groups = (int)node.out_shape[0];
            cache.gdn_conv_kernel = (int)node.out_shape[1] + 1;
            break;
        case PersistentInputKind::GDN_CHECKPOINT:
            cache.gdn_checkpoint = &tensor;
            break;
        case PersistentInputKind::GDN_CONV_CHECKPOINT:
            cache.gdn_conv_checkpoint = &tensor;
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
        case PersistentInputKind::AUX_STATE:
            break; // handled before indexing caches_
        case PersistentInputKind::NONE:
            break;
        }
    }

    // Allocate cache/state data buffers from engine-owned persistent storage
    // (once, at load time). Graph runtime pools are execution-temporary only.
    // Backends may keep only the KV metadata prefix mirrored on the host;
    // Metal Shared storage satisfies all three modes.
    Backend* storage_backend = accelerator_backend_
        ? static_cast<Backend*>(accelerator_backend_.get())
        : exec_ctx.backend;
    if (!storage_backend)
        return false;
    auto alloc_cache_buf = [&](
        Tensor* t, size_t total, PersistentHostAccess host_access,
        size_t host_prefix_bytes = 0) -> bool {
        if (accelerator_backend_) {
            accelerator_backend_->alloc_persistent(
                *t, total, host_access, host_prefix_bytes);
            return t->data && t->device_data;
        }
        void* buf = persistent_pool_.acquire(total);
        t->data = buf;
        t->owner_id = persistent_pool_.id();
        t->storage_id = persistent_pool_.storage_id(buf);
        return buf != nullptr;
    };

    for (auto& cp : caches) {
        // Standard KV cache
        if (cp.k) {
            int hd = cp.k_head_dim;
            int nkv = cp.k_num_heads;
            size_t es = cp.k->element_size();
            size_t data_bytes = (size_t)hd * n_ctx * nkv * es;
            size_t total = CacheMetadata::SIZE + data_bytes;

            cp.k->mem_type = MemoryType::POOLED;
            cp.k->shape[0] = (int64_t)total / (int64_t)es;
            cp.k->shape[1] = 1;
            cp.k->shape[2] = 1;
            cp.k->shape[3] = 1;
            cp.k->compute_strides();
            if (!alloc_cache_buf(
                    cp.k, total,
                    PersistentHostAccess::HOST_AUTHORITATIVE_PREFIX,
                    CacheMetadata::SIZE))
                return false;
            CacheMetadata metadata;
            metadata.max_seq_len = static_cast<uint64_t>(n_ctx);
            metadata.num_kv_heads = static_cast<uint64_t>(nkv);
            metadata.head_dim = static_cast<uint64_t>(hd);
            if (!storage_backend->copy_from_host(
                    &metadata, *cp.k, sizeof(metadata)))
                return false;
        }
        if (cp.v) {
            int vd = cp.v_head_dim;
            int nkv = cp.v_num_heads;
            size_t es = cp.v->element_size();
            size_t data_bytes = (size_t)vd * n_ctx * nkv * es;
            size_t total = CacheMetadata::SIZE + data_bytes;

            cp.v->mem_type = MemoryType::POOLED;
            cp.v->shape[0] = (int64_t)total / (int64_t)es;
            cp.v->shape[1] = 1;
            cp.v->shape[2] = 1;
            cp.v->shape[3] = 1;
            cp.v->compute_strides();
            if (!alloc_cache_buf(
                    cp.v, total,
                    PersistentHostAccess::HOST_AUTHORITATIVE_PREFIX,
                    CacheMetadata::SIZE))
                return false;
            CacheMetadata metadata;
            metadata.max_seq_len = static_cast<uint64_t>(n_ctx);
            metadata.num_kv_heads = static_cast<uint64_t>(nkv);
            metadata.v_head_dim = static_cast<uint64_t>(vd);
            if (!storage_backend->copy_from_host(
                    &metadata, *cp.v, sizeof(metadata)))
                return false;
        }

        // GDN recurrent state: [v_dim, k_dim, num_heads] FP32
        // No CacheMetadata header — GDN state is a plain FP32 buffer.
        if (cp.gdn_state) {
            size_t data_bytes = (size_t)cp.gdn_v_dim * cp.gdn_k_dim *
                                cp.gdn_num_heads * sizeof(float);
            cp.gdn_state->mem_type = MemoryType::POOLED;
            cp.gdn_state->shape[0] = (int64_t)cp.gdn_v_dim;
            cp.gdn_state->shape[1] = (int64_t)cp.gdn_k_dim;
            cp.gdn_state->shape[2] = (int64_t)cp.gdn_num_heads;
            cp.gdn_state->shape[3] = 1;
            cp.gdn_state->compute_strides();
            if (!alloc_cache_buf(
                    cp.gdn_state, data_bytes,
                    PersistentHostAccess::NONE) ||
                !storage_backend->zero_tensor(*cp.gdn_state, data_bytes))
                return false;
        }
        // GDN conv state: [groups, kernel-1] FP32
        if (cp.gdn_conv) {
            int kernel_m1 = cp.gdn_conv_kernel - 1;
            size_t data_bytes =
                (size_t)cp.gdn_conv_groups * kernel_m1 * sizeof(float);
            cp.gdn_conv->mem_type = MemoryType::POOLED;
            cp.gdn_conv->shape[0] = (int64_t)cp.gdn_conv_groups;
            cp.gdn_conv->shape[1] = (int64_t)kernel_m1;
            cp.gdn_conv->shape[2] = 1;
            cp.gdn_conv->shape[3] = 1;
            cp.gdn_conv->compute_strides();
            if (!alloc_cache_buf(
                    cp.gdn_conv, data_bytes,
                    PersistentHostAccess::NONE) ||
                !storage_backend->zero_tensor(*cp.gdn_conv, data_bytes))
                return false;
        }
        if (cp.rwkv_state) {
            size_t bytes = cp.rwkv_state->nbytes();
            cp.rwkv_state->mem_type = MemoryType::POOLED;
            if (!alloc_cache_buf(
                    cp.rwkv_state, bytes, PersistentHostAccess::NONE) ||
                !storage_backend->zero_tensor(*cp.rwkv_state, bytes))
                return false;
        }
        for (Tensor* shift : {cp.rwkv_att_shift, cp.rwkv_ffn_shift})
            if (shift) {
                size_t bytes = shift->nbytes();
                shift->mem_type = MemoryType::POOLED;
                if (!alloc_cache_buf(
                        shift, bytes, PersistentHostAccess::NONE) ||
                    !storage_backend->zero_tensor(*shift, bytes))
                    return false;
            }
    }
    for (const auto& entry : auxiliary_states_) {
        Tensor* state = entry.second;
        if (!state || state->data)
            continue;
        const size_t bytes = state->nbytes();
        state->mem_type = MemoryType::POOLED;
        if (!alloc_cache_buf(
                state, bytes, PersistentHostAccess::NONE) ||
            !storage_backend->zero_tensor(*state, bytes))
            return false;
    }
    return true;
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
    if (cfg_.image_max_pixels < EngineConfig::kMinImageMaxPixels ||
        cfg_.image_max_pixels > EngineConfig::kAbsoluteImageMaxPixels) {
        std::fprintf(
            stderr, "Engine: image_max_pixels must be in [%d, %d]\n",
            EngineConfig::kMinImageMaxPixels,
            EngineConfig::kAbsoluteImageMaxPixels);
        return false;
    }
    std::string sampling_error;
    if (!sampler_.configure(cfg_.sampling, &sampling_error, true)) {
        fprintf(stderr, "Engine: invalid sampling parameters: %s\n",
                sampling_error.c_str());
        return false;
    }
    sampler_.reset();
    if (cfg_.metal_ssd_full &&
        (cfg_.device != Device::METAL || cfg_.moe_ssd_cache_bytes == 0)) {
        fprintf(stderr,
                "Engine: --metal-ssd-full requires --device metal and "
                "--ssd-cache-mb\n");
        return false;
    }
    mollm_trace::start(cfg_.trace_path);
    mollm_trace::set_thread_name("main");
    if (cfg_.moe_ssd_cache_bytes != 0) {
        // Expert tensors must remain unmaterialized; resident package loading
        // would copy the complete aggregate tensors before the cache can help.
        cfg_.weight_loading = WeightLoadingMode::MMAP;
    }
    thread_pool_.resize(cfg_.num_threads);
    exec_ctx_prefill_.thread_pool = &thread_pool_;
    exec_ctx_decode_.thread_pool = &thread_pool_;
    exec_ctx_vision_.thread_pool = &thread_pool_;
    exec_ctx_mtp_.thread_pool = &thread_pool_;
    exec_ctx_mtp_verify_.thread_pool = &thread_pool_;
    exec_ctx_prefill_.trace_label = "prefill";
    exec_ctx_decode_.trace_label = "decode";
    exec_ctx_vision_.trace_label = "vision";
    exec_ctx_mtp_.trace_label = "mtp";
    exec_ctx_mtp_verify_.trace_label = "mtp.verify";
    exec_ctx_prefill_.moe_cross_layer_prefetch = false;
    exec_ctx_prefill_.moe_hash_cross_layer_prefetch = false;
    exec_ctx_mtp_.moe_cross_layer_prefetch = false;
    exec_ctx_mtp_.moe_hash_cross_layer_prefetch = false;
    exec_ctx_decode_.moe_cross_layer_prefetch =
        cfg_.moe_ssd_cache_bytes != 0 &&
        !cfg_.metal_ssd_full && cfg_.moe_ssd_global_cache &&
        cfg_.moe_ssd_cross_layer_prefetch;
    exec_ctx_decode_.moe_hash_cross_layer_prefetch =
        exec_ctx_decode_.moe_cross_layer_prefetch;
    exec_ctx_prefill_.backend = &cpu_backend_;
    exec_ctx_decode_.backend = &cpu_backend_;
    exec_ctx_vision_.backend = &cpu_backend_;
    exec_ctx_mtp_.backend = &cpu_backend_;
    exec_ctx_mtp_verify_.backend = &cpu_backend_;
    exec_ctx_prefill_.moe_backend = nullptr;
    exec_ctx_decode_.moe_backend = nullptr;
    exec_ctx_vision_.moe_backend = nullptr;
    exec_ctx_mtp_.moe_backend = nullptr;
    accelerator_backend_.reset();
    auto fallback_to_cpu = [&](const char* reason) {
        if (cfg_.device_fallback ==
            DeviceFallbackPolicy::REQUIRE_REQUESTED) {
            std::fprintf(
                stderr, "Engine: %s; requested device is required\n",
                reason);
            return false;
        }
        std::fprintf(stderr, "Engine: %s; falling back to CPU\n", reason);
        accelerator_backend_.reset();
        cfg_.device = Device::CPU;
        return true;
    };
    if (cfg_.device == Device::METAL) {
#ifdef MOLLM_METAL
        accelerator_backend_ = std::make_unique<MetalBackend>();
        if (!accelerator_backend_->available()) {
            if (!fallback_to_cpu("Metal backend unavailable"))
                return false;
        } else {
            exec_ctx_prefill_.backend = accelerator_backend_.get();
            // SSD expert compute remains much faster when decode stays wholly
            // on CPU: alternating a GPU dense segment with every CPU expert
            // layer throttles the routed matmuls on UMA. Keep Metal for the
            // large prefill matrices and use the established CPU SSD pipeline
            // (including cross-layer prefetch) for token generation.
            exec_ctx_decode_.backend =
                cfg_.moe_ssd_cache_bytes != 0 && !cfg_.metal_ssd_full
                ? static_cast<Backend*>(&cpu_backend_)
                : accelerator_backend_.get();
            exec_ctx_mtp_.backend = exec_ctx_decode_.backend;
        }
#else
        if (!fallback_to_cpu("built without MOLLM_METAL"))
            return false;
#endif
    } else if (cfg_.device == Device::CUDA) {
#ifdef MOLLM_CUDA
        accelerator_backend_ = std::make_unique<CudaBackend>();
        if (!accelerator_backend_->available()) {
            if (!fallback_to_cpu("CUDA backend unavailable"))
                return false;
        } else {
            exec_ctx_prefill_.backend = accelerator_backend_.get();
            exec_ctx_decode_.backend = accelerator_backend_.get();
            exec_ctx_mtp_.backend = accelerator_backend_.get();
            exec_ctx_vision_.backend = accelerator_backend_.get();
        }
#else
        if (!fallback_to_cpu("built without MOLLM_CUDA"))
            return false;
#endif
    }
    if (accelerator_backend_ &&
        !accelerator_backend_->set_operator_fallback_policy(
            cfg_.operator_fallback)) {
        std::fprintf(
            stderr,
            "Engine: selected accelerator does not support the requested "
            "operator fallback policy\n");
        return false;
    }
    exec_ctx_prefill_.reuse_static_workspace = false;
    exec_ctx_prefill_.reuse_same_shape_workspace = true;
    exec_ctx_decode_.reuse_static_workspace = true;
    exec_ctx_decode_.reuse_same_shape_workspace = false;
    exec_ctx_mtp_.reuse_static_workspace = false;
    exec_ctx_mtp_.reuse_same_shape_workspace = true;
    exec_ctx_mtp_verify_.reuse_static_workspace = false;
    exec_ctx_mtp_verify_.reuse_same_shape_workspace = true;

    // Load the .mollm package (sets up weights mmap, extracts graphs to temp
    // files, parses metadata for weight offset map).
    if (cfg.package_path.empty()) {
        fprintf(stderr,
                "Engine: package_path is required (use .mollm package)\n");
        return false;
    }
    std::string pf_path, dc_path, vi_path, mtp_verify_path, mtp_path;
    {
        std::string tok_tmp, jinja_tmp;
        if (!load_package(cfg.package_path, pf_path, dc_path, vi_path,
                          mtp_verify_path, mtp_path,
                          tok_tmp, jinja_tmp)) {
            return false;
        }
        if (!tok_tmp.empty()) {
            cfg_.tokenizer_path = tok_tmp;
        }
        if (!jinja_tmp.empty()) {
            cfg_.chat_template_path = jinja_tmp;
        }
    }
    if (cfg_.mtp_draft_tokens > 0 && mtp_path.empty()) {
        fprintf(stderr,
                "Engine: --mtp-draft-tokens requires a package with an MTP graph\n");
        return false;
    }
    if (cfg_.mtp_draft_tokens > 0) {
        auto max_drafts = package_metadata_.find("mtp_max_draft_tokens");
        if (max_drafts != package_metadata_.end()) {
            char* end = nullptr;
            const long limit = std::strtol(
                max_drafts->second.c_str(), &end, 10);
            if (!end || *end != '\0' || limit <= 0 ||
                cfg_.mtp_draft_tokens > limit) {
                fprintf(
                    stderr,
                    "Engine: package supports at most %ld MTP draft token(s); "
                    "requested %d\n",
                    limit, cfg_.mtp_draft_tokens);
                return false;
            }
        }
    }

    const auto architecture_it = package_metadata_.find("architecture");
    const std::string architecture =
        architecture_it != package_metadata_.end()
            ? architecture_it->second
            : std::string();
    const bool qwen35_transactional_mtp =
        architecture == "qwen3.5" && cfg_.mtp_draft_tokens > 0;
    if (qwen35_transactional_mtp) {
        const auto transactional =
            package_metadata_.find("mtp_transactional_state");
        if (cfg_.mtp_draft_tokens != 1) {
            fprintf(stderr,
                    "Engine: Qwen3.5 transactional MTP supports exactly one "
                    "draft token\n");
            return false;
        }
        if (!sampler_.uses_plain_argmax()) {
            fprintf(stderr,
                    "Engine: Qwen3.5 transactional MTP requires greedy "
                    "sampling\n");
            return false;
        }
        if (cfg.device == Device::METAL) {
            fprintf(stderr,
                    "Engine: Qwen3.5 transactional MTP is CPU-only\n");
            return false;
        }
        if (transactional == package_metadata_.end() ||
            transactional->second != "1") {
            fprintf(stderr,
                    "Engine: Qwen3.5 MTP package lacks transactional state "
                    "metadata\n");
            return false;
        }
        if (mtp_verify_path.empty()) {
            fprintf(stderr,
                    "Engine: Qwen3.5 MTP package is missing the target "
                    "verification graph\n");
            return false;
        }
        const auto vision = package_metadata_.find("vision");
        if (vision != package_metadata_.end() &&
            (vision->second == "true" || vision->second == "1")) {
            fprintf(stderr,
                    "Engine: Qwen3.5 transactional MTP does not support "
                    "vision packages\n");
            return false;
        }
    }
    if (architecture == "deepseek-v4") {
        const auto context_it = package_metadata_.find("n_ctx");
        if (context_it == package_metadata_.end()) {
            fprintf(stderr,
                    "Engine: DeepSeek-V4 package is missing n_ctx metadata\n");
            return false;
        }
        char* context_end = nullptr;
        const long package_context =
            std::strtol(context_it->second.c_str(), &context_end, 10);
        if (!context_end || *context_end != '\0' || package_context <= 0) {
            fprintf(stderr,
                    "Engine: DeepSeek-V4 package has invalid n_ctx metadata\n");
            return false;
        }
        if (cfg_.n_ctx > package_context) {
            fprintf(
                stderr,
                "Engine: requested n_ctx=%d exceeds this DeepSeek-V4 "
                "package's auxiliary-cache capacity (%ld); reconvert with "
                "a larger --n-ctx or lower --n-ctx\n",
                cfg_.n_ctx, package_context);
            return false;
        }

        // Metal uses a CPU graph, but may delegate routed MXFP4 experts
        // through its dedicated SSD cache.
        const bool unsupported_accelerator = cfg_.device == Device::METAL;
        if (unsupported_accelerator) {
            if (cfg_.metal_ssd_full) {
                fprintf(stderr,
                        "Engine: DeepSeek-V4 does not yet support "
                        "--metal-ssd-full\n");
                return false;
            }
            exec_ctx_prefill_.backend = &cpu_backend_;
            exec_ctx_decode_.backend = &cpu_backend_;
            exec_ctx_vision_.backend = &cpu_backend_;
            exec_ctx_mtp_.backend = &cpu_backend_;
            const bool metal_hybrid =
                cfg_.device == Device::METAL &&
                cfg_.moe_ssd_cache_bytes != 0;
            if (!metal_hybrid) {
                if (!fallback_to_cpu(
                        "DeepSeek-V4 is unsupported by the Metal backend"))
                    return false;
            }
        }
    }

    // Give the accelerator the package region before constants are loaded.
    if (accelerator_backend_ && package_weights_base_ &&
        package_weights_size_) {
        if (moe_ssd_cache_) {
            accelerator_backend_->enable_weight_copy_mode();
            const bool dsv4_hybrid =
                architecture == "deepseek-v4" &&
                cfg_.device == Device::METAL;
            if (cfg_.metal_ssd_full || dsv4_hybrid) {
                if (!accelerator_backend_->configure_moe_ssd_io(
                             cfg_.package_path, cfg_.moe_ssd_cache_bytes,
                             cfg_.moe_ssd_io_workers,
                             cfg_.moe_ssd_cross_layer_prefetch)) {
                    return false;
                }
                if (dsv4_hybrid) {
                    exec_ctx_decode_.moe_backend = accelerator_backend_.get();
                    // The Metal cache owns routed-expert demand traffic on
                    // this path. CPU cross-layer prediction would populate a
                    // second cache with the same tensors and contend for both
                    // SSD bandwidth and UMA pages without serving compute.
                    exec_ctx_decode_.moe_cross_layer_prefetch = false;
                    exec_ctx_decode_.moe_hash_cross_layer_prefetch = false;
                    fprintf(stderr,
                            "Engine: DeepSeek-V4 hybrid SSD decode enabled; "
                            "routed MXFP4 experts use Metal I/O and GPU compute\n");
                } else {
                    fprintf(stderr,
                            "Engine: full-Metal SSD decode enabled; experts "
                            "load directly into Shared Metal buffers\n");
                }
            } else {
                fprintf(
                    stderr,
                    "Engine: Metal/SSD hybrid adaptively selects CPU/Metal "
                    "prefill and uses CPU decode; experts remain mmap-backed\n");
            }
        } else {
            if (!accelerator_backend_->register_weight_region(
                         const_cast<uint8_t*>(package_weights_base_),
                         package_weights_size_)) {
                fprintf(stderr,
                        "Engine: failed to prepare accelerator weight region\n");
            }
        }
    }

    // Load prefill graph first (establishes shared weights)
    const std::string& pf_path_load =
        cfg_.use_decode_as_prefill ? dc_path : pf_path;
    if (!load_graph(graph_prefill_, exec_ctx_prefill_, pf_path_load.c_str())) {
        return false;
    }
    if (!vi_path.empty() &&
        !load_graph(graph_vision_, exec_ctx_vision_, vi_path.c_str())) {
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

    if (!allocate_caches(
            graph_prefill_, exec_ctx_prefill_, caches_, cfg.n_ctx)) {
        fprintf(stderr, "Engine: failed to allocate persistent state\n");
        return false;
    }

    // Load decode graph (reuses shared weights via weight_map_).
    if (!load_graph(graph_decode_, exec_ctx_decode_, dc_path.c_str())) {
        return false;
    }
    if (cfg_.mtp_draft_tokens > 0) {
        if (qwen35_transactional_mtp) {
            if (!load_graph(graph_mtp_verify_, exec_ctx_mtp_verify_,
                            mtp_verify_path.c_str())) {
                return false;
            }

            for (auto& node : graph_mtp_verify_.nodes) {
                if (node.op_type != OpType::INPUT || node.params.str.empty())
                    continue;
                const PersistentInput input =
                    parse_persistent_input(node.params.str[0]);
                if (input.kind == PersistentInputKind::NONE)
                    continue;
                if (input.layer < 0 || input.layer >= (int)caches_.size()) {
                    fprintf(stderr,
                            "Engine: verification graph references invalid "
                            "cache layer %d\n", input.layer);
                    return false;
                }

                Tensor& tensor =
                    graph_mtp_verify_.runtime.tensors[node.id];
                initialize_input_tensor(tensor, node);
                CachePair& cache = caches_[input.layer];
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
                case PersistentInputKind::GDN_CHECKPOINT:
                case PersistentInputKind::GDN_CONV_CHECKPOINT: {
                    if (tensor.prec != Precision::FP32 || tensor.nbytes() == 0) {
                        fprintf(stderr,
                                "Engine: verification checkpoint must be "
                                "non-empty FP32\n");
                        return false;
                    }
                    const size_t bytes = tensor.nbytes();
                    void* buffer = persistent_pool_.acquire(bytes);
                    tensor.data = buffer;
                    tensor.owner_id = persistent_pool_.id();
                    tensor.storage_id = persistent_pool_.storage_id(buffer);
                    tensor.mem_type = MemoryType::POOLED;
                    std::memset(buffer, 0, bytes);
                    if (input.kind == PersistentInputKind::GDN_CHECKPOINT)
                        cache.gdn_checkpoint = &tensor;
                    else
                        cache.gdn_conv_checkpoint = &tensor;
                    break;
                }
                case PersistentInputKind::STATE:
                case PersistentInputKind::ATT_SHIFT:
                case PersistentInputKind::FFN_SHIFT:
                case PersistentInputKind::AUX_STATE:
                case PersistentInputKind::NONE:
                    break;
                }
                if (source) {
                    tensor = *source;
                } else if (input.kind != PersistentInputKind::GDN_CHECKPOINT &&
                           input.kind !=
                               PersistentInputKind::GDN_CONV_CHECKPOINT) {
                    fprintf(stderr,
                            "Engine: verification graph cache input %s has "
                            "no target state\n", node.params.str[0].c_str());
                    return false;
                }
            }
            for (const CachePair& cache : caches_) {
                if (!cache.is_linear_attn)
                    continue;
                if (!cache.gdn_state || !cache.gdn_conv ||
                    !cache.gdn_checkpoint || !cache.gdn_conv_checkpoint ||
                    cache.gdn_state->prec != Precision::FP32 ||
                    cache.gdn_conv->prec != Precision::FP32 ||
                    cache.gdn_checkpoint->prec != Precision::FP32 ||
                    cache.gdn_conv_checkpoint->prec != Precision::FP32 ||
                    cache.gdn_state->nbytes() !=
                        cache.gdn_checkpoint->nbytes() ||
                    cache.gdn_conv->nbytes() !=
                        cache.gdn_conv_checkpoint->nbytes()) {
                    fprintf(stderr,
                            "Engine: invalid Qwen3.5 verification checkpoint "
                            "binding\n");
                    return false;
                }
            }
        }
        if (!load_graph(graph_mtp_, exec_ctx_mtp_, mtp_path.c_str())) {
            return false;
        }
        if (!allocate_caches(
                graph_mtp_, exec_ctx_mtp_, mtp_caches_, cfg.n_ctx)) {
            fprintf(stderr, "Engine: failed to allocate MTP persistent state\n");
            return false;
        }
    }
    // Graph loading builds the CPU sidecars used by FP8 dense weights and
    // records their source blobs as exclusions. Lock only after both graphs
    // have been prepared so those now-unused raw package pages are not pulled
    // into RAM and pinned alongside their sidecars.
    if (cfg_.lock_dense_weights && moe_ssd_cache_) {
        lock_dense_package_weights();
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
        if (input.kind == PersistentInputKind::NONE) {
            continue;
        }

        const Tensor* source = nullptr;
        if (input.kind == PersistentInputKind::AUX_STATE) {
            auto state = auxiliary_states_.find(input.layer);
            source = state != auxiliary_states_.end()
                ? state->second : nullptr;
            if (source)
                graph_decode_.runtime.tensors[node.id] = *source;
            continue;
        }
        if (input.layer >= (int)caches_.size())
            continue;
        const CachePair& cache = caches_[input.layer];
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
        case PersistentInputKind::GDN_CHECKPOINT:
        case PersistentInputKind::GDN_CONV_CHECKPOINT:
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
        case PersistentInputKind::AUX_STATE:
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
