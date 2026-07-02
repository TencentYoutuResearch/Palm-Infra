#include "engine/engine.h"
#include "kernels/matmul.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

namespace {

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && std::strcmp(value, "0") != 0;
}

bool int8_q8dot_repack_supported(const Tensor& t) {
#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    return t.prec == Precision::INT8 && t.group_size > 0 && (t.group_size % 32) == 0;
#else
    (void)t;
    return false;
#endif
}

struct Int8PackingPlan {
    bool build_interleaved = false;
    bool build_q8dot = false;
};

using PackedWeightMap = std::unordered_map<std::string, std::vector<uint8_t>>;

Int8PackingPlan plan_int8_packing(const Tensor& t) {
    Int8PackingPlan plan;
    if (t.prec != Precision::INT8 || !g_matmul_config.use_interleave_pack) {
        return plan;
    }

    bool q8dot_disabled = env_flag_enabled("MOLLM_W8_NO_Q8_DOT");
    bool q8dot_repack_disabled =
        env_flag_enabled("MOLLM_W8_NO_Q8_DOT_REPACK") ||
        env_flag_enabled("MOLLM_W8_NO_Q8_REPACK");
    bool q8dot_gemm_disabled = env_flag_enabled("MOLLM_W8_NO_Q8_DOT_GEMM");
    bool onfly_fp16 = env_flag_enabled("MOLLM_W8_ONFLY_FP16");
    bool keep_interleaved = env_flag_enabled("MOLLM_W8_KEEP_INT8_INTERLEAVED");
    bool can_q8dot = int8_q8dot_repack_supported(t);

    plan.build_q8dot = can_q8dot && !q8dot_disabled && !q8dot_repack_disabled;
    plan.build_interleaved = keep_interleaved || onfly_fp16 || q8dot_disabled ||
                             q8dot_repack_disabled || q8dot_gemm_disabled ||
                             !plan.build_q8dot;
    return plan;
}

void maybe_pack_fp16_weight(Tensor& t, const std::string& key,
                            const void* rowmajor_data,
                            PackedWeightMap& packed_weights) {
    if (t.prec != Precision::FP16 || !g_matmul_config.use_interleave_pack) return;

    auto it = packed_weights.find(key);
    if (it == packed_weights.end()) {
        int N = (int)t.shape[0];
        int K = (int)t.shape[1];
        const __fp16* b_orig = reinterpret_cast<const __fp16*>(rowmajor_data);
        __fp16* b_packed = pack_b_interleaved_full(b_orig, N, K, K);
        size_t buf_size = (size_t)((N + 7) / 8) * 8 * K * sizeof(__fp16);
        std::vector<uint8_t> buf((uint8_t*)b_packed, (uint8_t*)b_packed + buf_size);
        delete[] b_packed;
        it = packed_weights.emplace(key, std::move(buf)).first;
    }
    t.data = it->second.data();
}

void maybe_pack_int8_weight(Tensor& t, const std::string& key,
                            const void* rowmajor_data,
                            PackedWeightMap& packed_weights) {
#if HAS_NEON
    Int8PackingPlan plan = plan_int8_packing(t);
    if (!plan.build_interleaved && !plan.build_q8dot) return;

    int N = (int)t.shape[0];
    int K = (int)t.shape[1];
    const int8_t* b_orig = reinterpret_cast<const int8_t*>(rowmajor_data);

    if (plan.build_interleaved) {
        std::string pack_key = key + "#int8_interleaved";
        auto it = packed_weights.find(pack_key);
        if (it == packed_weights.end()) {
            int8_t* b_packed = pack_b_interleaved_int8_full(b_orig, N, K, K);
            size_t buf_size = (size_t)((N + 7) / 8) * 8 * K * sizeof(int8_t);
            std::vector<uint8_t> buf((uint8_t*)b_packed, (uint8_t*)b_packed + buf_size);
            delete[] b_packed;
            it = packed_weights.emplace(pack_key, std::move(buf)).first;
        }
        t.data = it->second.data();
        t.is_interleaved = true;
    }

    if (plan.build_q8dot) {
        std::string q8_key = key + "#int8_q8dot";
        auto it = packed_weights.find(q8_key);
        if (it == packed_weights.end()) {
            int K_blocks = (K + 31) / 32;
            int8_t* b_q8 = pack_b_q8dot_int8_full(b_orig, N, K, K);
            size_t buf_size = (size_t)((N + 7) / 8) * 8 * K_blocks * 32 * sizeof(int8_t);
            std::vector<uint8_t> buf((uint8_t*)b_q8, (uint8_t*)b_q8 + buf_size);
            delete[] b_q8;
            it = packed_weights.emplace(q8_key, std::move(buf)).first;
        }
        t.q8_repack_data = it->second.data();
    }
#else
    (void)t;
    (void)key;
    (void)rowmajor_data;
    (void)packed_weights;
#endif
}

/// Temperature scaling + softmax in-place.
/// Returns the softmax sum (should be ~1.0).
float softmax_inplace(float* logits, int n, float temperature) {
    float max_val = logits[0];
    for (int i = 1; i < n; i++) max_val = std::max(max_val, logits[i]);

    float inv_t = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        logits[i] = std::exp((logits[i] - max_val) * inv_t);
        sum += logits[i];
    }
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < n; i++) logits[i] *= inv_sum;
    }
    return sum;
}

/// Top-k filtering: zero out probabilities below the k-th largest.
void top_k_filter(float* probs, int n, int top_k) {
    if (top_k <= 0 || top_k >= n) return;

    // Partial sort: find the top_k-th largest element
    std::vector<float> copy(probs, probs + n);
    std::nth_element(copy.begin(), copy.begin() + top_k - 1, copy.end(),
                     std::greater<float>());
    float threshold = copy[top_k - 1];

    for (int i = 0; i < n; i++) {
        if (probs[i] < threshold) probs[i] = 0.0f;
    }
}

/// Top-p (nucleus) filtering: keep smallest set of tokens whose cumulative
/// probability >= top_p.
void top_p_filter(float* probs, int n, float top_p) {
    if (top_p <= 0.0f || top_p >= 1.0f) return;

    // Sort indices by probability descending
    std::vector<int> indices(n);
    for (int i = 0; i < n; i++) indices[i] = i;
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return probs[a] > probs[b]; });

    float cumsum = 0.0f;
    int cutoff = 0;
    for (int i = 0; i < n; i++) {
        cumsum += probs[indices[i]];
        if (cumsum >= top_p) { cutoff = i + 1; break; }
    }

    // Zero out everything below cutoff
    float threshold = probs[indices[std::min(cutoff, n - 1)]];
    for (int i = 0; i < n; i++) {
        if (probs[i] < threshold) probs[i] = 0.0f;
    }
}

/// Multinomial sample from probability distribution.
int sample_multinomial(const float* probs, int n, float random_val) {
    float cumsum = 0.0f;
    for (int i = 0; i < n; i++) {
        cumsum += probs[i];
        if (random_val < cumsum) return i;
    }
    // Fallback: argmax
    int best = 0;
    for (int i = 1; i < n; i++) if (probs[i] > probs[best]) best = i;
    return best;
}

/// Full sample pipeline: logits → temperature → softmax → top-k → top-p → sample.
int sample_token(float* logits, int vocab_size,
                  float temperature, int top_k, float top_p,
                  unsigned int* seed) {
    if (temperature <= 0.0f) {
        // Greedy: just argmax
        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    softmax_inplace(logits, vocab_size, temperature);
    top_k_filter(logits, vocab_size, top_k);
    top_p_filter(logits, vocab_size, top_p);

    // Re-normalize after filtering
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) sum += logits[i];
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < vocab_size; i++) logits[i] *= inv_sum;
    } else {
        // All probs zeroed — fallback to argmax
        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    float r = (float)rand_r(seed) / (float)RAND_MAX;
    return sample_multinomial(logits, vocab_size, r);
}

void release_pool_tensor(BufferPool& pool, Tensor& t) {
    if (t.data && t.mem_type == MemoryType::POOLED && t.nbytes() > 0) {
        if (t.owner_id != 0 && t.owner_id != pool.id()) {
            std::fprintf(stderr,
                         "release_pool_tensor: owner mismatch for %p (tensor owner=%u, pool=%u)\n",
                         t.data, t.owner_id, pool.id());
            assert(false && "release_pool_tensor owner mismatch");
            return;
        }
        pool.release(t.data, t.nbytes());
    }
    t.data = nullptr;
    t.mem_type = MemoryType::NONE;
    t.owner_id = 0;
    t.storage_id = 0;
}

bool is_view_op(OpType op) {
    return op == OpType::PERMUTE || op == OpType::SLICE;
}

bool shares_storage(const Tensor& a, const Tensor& b) {
    if (a.storage_id != 0 && b.storage_id != 0) {
        return a.owner_id == b.owner_id && a.storage_id == b.storage_id;
    }
    return a.data == b.data;
}

void release_graph_temporaries(Graph& graph) {
#ifdef MOLLM_SKIP_GRAPH_TEMP_CLEANUP
    (void)graph;
    return;
#endif
    auto& pool = graph.runtime.pool;
    auto& tensors = graph.runtime.tensors;
    std::vector<uint8_t> borrowed_view(graph.nodes.size(), 0);

    for (auto& node : graph.nodes) {
        if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT) continue;
        Tensor& t = tensors[node.id];
        bool borrowed = is_view_op(node.op_type);
        if (node.op_type == OpType::RESHAPE && !node.inputs.empty()) {
            borrowed = shares_storage(t, tensors[node.inputs[0]]);
        }
        borrowed_view[node.id] = borrowed ? 1 : 0;
    }

    for (auto& node : graph.nodes) {
        if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT) continue;
        if (!borrowed_view[node.id]) continue;
        Tensor& t = tensors[node.id];
        t.data = nullptr;
        t.mem_type = MemoryType::NONE;
        t.owner_id = 0;
        t.storage_id = 0;
    }

    for (auto& node : graph.nodes) {
        if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT) continue;
        if (borrowed_view[node.id]) continue;
        Tensor& t = tensors[node.id];
        release_pool_tensor(pool, t);
    }
}

void clear_graph_borrowed_views(Graph& graph) {
    auto& tensors = graph.runtime.tensors;
    std::vector<uint8_t> borrowed_view(graph.nodes.size(), 0);

    for (auto& node : graph.nodes) {
        if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT) continue;
        Tensor& t = tensors[node.id];
        bool borrowed = is_view_op(node.op_type);
        if (node.op_type == OpType::RESHAPE && !node.inputs.empty()) {
            borrowed = shares_storage(t, tensors[node.inputs[0]]);
        }
        borrowed_view[node.id] = borrowed ? 1 : 0;
    }

    for (auto& node : graph.nodes) {
        if (node.op_type == OpType::INPUT || node.op_type == OpType::CONSTANT) continue;
        if (!borrowed_view[node.id]) continue;
        Tensor& t = tensors[node.id];
        t.data = nullptr;
        t.mem_type = MemoryType::NONE;
        t.owner_id = 0;
        t.storage_id = 0;
    }
}

void invalidate_workspace_key(ExecContext& ctx) {
    ctx.workspace_shape_valid = false;
    ctx.workspace_runtime_seq_len = -1;
    ctx.workspace_runtime_batch = -1;
    ctx.workspace_static_padded = false;
    ctx.workspace_padded_seq_len = -1;
}

void finish_graph_temporaries(Graph& graph, ExecContext& ctx) {
    if (ctx.reuse_static_workspace || ctx.reuse_same_shape_workspace) {
        clear_graph_borrowed_views(graph);
    } else {
        release_graph_temporaries(graph);
        invalidate_workspace_key(ctx);
    }
}

Tensor copy_tensor_contiguous(const Tensor& src, std::vector<uint8_t>& storage) {
    size_t es = src.element_size();
    size_t bytes = (size_t)src.nelements() * es;
    storage.resize(bytes);

    Tensor dst = Tensor::create(src.prec, MemoryType::EXTERNAL,
                                src.shape[0], src.shape[1],
                                src.shape[2], src.shape[3],
                                storage.data());

    if (!src.data || bytes == 0) return dst;

    if (src.is_contiguous()) {
        std::memcpy(dst.data, src.data, bytes);
        return dst;
    }

    char* dp = static_cast<char*>(dst.data);
    const char* sp_base = static_cast<const char*>(src.data);
    size_t flat = 0;
    for (int64_t i3 = 0; i3 < src.shape[3]; i3++) {
        for (int64_t i2 = 0; i2 < src.shape[2]; i2++) {
            for (int64_t i1 = 0; i1 < src.shape[1]; i1++) {
                const char* sp = sp_base
                    + i1 * src.stride[1]
                    + i2 * src.stride[2]
                    + i3 * src.stride[3];
                std::memcpy(dp + flat * es, sp, (size_t)src.shape[0] * es);
                flat += (size_t)src.shape[0];
            }
        }
    }
    return dst;
}

} // namespace

LLMEngine::~LLMEngine() {
    // Tensor data owned by BufferPool or EXTERNAL — no explicit free needed.
    // MappedFiles in shared_weights_ auto-close on destruction.

    // Release .mollm package mmap (can be many GB — qwen35_4b is 8.4 GB).
    if (package_mmap_) {
        munmap(package_mmap_, package_mmap_size_);
        package_mmap_      = nullptr;
        package_mmap_size_ = 0;
    }

    // Remove extracted temp files from /tmp.
    for (const auto& path : temp_files_) {
        std::remove(path.c_str());
    }
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
    if (slash != std::string::npos) graph_dir = graph_dir.substr(0, slash + 1);
    else graph_dir = "./";

    for (auto& node : g.nodes) {
        if (node.op_type != OpType::CONSTANT || node.params.str.empty()) continue;

        std::string wref = node.params.str[0];
        std::string wpath = wref;
        if (wpath[0] != '/' && (wpath.size() < 2 || wpath[1] != ':')) {
            wpath = graph_dir + wpath;
        }

        auto& t = g.runtime.tensors[node.id];

        auto setup_quant_metadata = [&](const void* scales, uint32_t group_size, uint32_t num_groups,
                                        size_t scales_size, const char* label) -> bool {
            t.scales = nullptr;
            t.group_size = 0;
            t.num_groups = 0;
            t.groups_per_row = 0;

            if (t.prec != Precision::INT8) return true;
            int64_t N = t.shape[0];
            int64_t K = t.shape[1];
            if (!scales || group_size == 0 || N <= 0 || K <= 0) {
                fprintf(stderr, "Engine: INT8 weight %s missing quant scales/group metadata\n", label);
                return false;
            }
            uint32_t groups_per_row = (uint32_t)((K + group_size - 1) / group_size);
            uint64_t expected_groups = (uint64_t)N * groups_per_row;
            if (num_groups != expected_groups ||
                scales_size != expected_groups * sizeof(float)) {
                fprintf(stderr,
                        "Engine: INT8 weight %s bad quant metadata (N=%lld K=%lld group=%u groups=%u expected=%llu scales=%zu)\n",
                        label, (long long)N, (long long)K, group_size, num_groups,
                        (unsigned long long)expected_groups, scales_size);
                return false;
            }
            t.scales = static_cast<const float*>(scales);
            t.group_size = group_size;
            t.num_groups = num_groups;
            t.groups_per_row = groups_per_row;
            return true;
        };

        // Helper: set up weight tensor.
        auto setup_weight = [&](void* data, Precision file_prec) {
            t.prec = file_prec;
            int64_t dim0 = node.out_shape[0];  // N
            int64_t dim1 = node.out_shape[1];  // K
            t.shape[0] = dim0;
            t.shape[1] = dim1;
            t.shape[2] = node.out_shape[2];
            t.shape[3] = node.out_shape[3];
            t.compute_strides();
            t.data = data;
            t.mem_type = MemoryType::EXTERNAL;
            t.is_interleaved = false;
            t.q8_repack_data = nullptr;
        };

        // Package mode: resolve weight from package mmap via offset map.
        // The weight path (e.g. "./foo.weights") is looked up in
        // package_weight_map_ to find (offset, size) within the weights region.
        if (package_weights_base_ != nullptr) {
            auto pit = package_weight_map_.find(wref);
            if (pit == package_weight_map_.end()) pit = package_weight_map_.find(wpath);
            if (pit != package_weight_map_.end()) {
                const uint8_t* hdr = package_weights_base_ + pit->second.first;
                // Read data_offset from weight header (byte 48, 8 bytes)
                uint32_t file_prec_u32 = 0;
                uint64_t data_off;
                std::memcpy(&file_prec_u32, hdr + 12, sizeof(file_prec_u32));
                std::memcpy(&data_off, hdr + 48, sizeof(data_off));
                uint64_t scales_off = 0, scales_size = 0;
                uint32_t group_size = 0, num_groups = 0;
                std::memcpy(&scales_off, hdr + 64, sizeof(scales_off));
                std::memcpy(&scales_size, hdr + 72, sizeof(scales_size));
                std::memcpy(&group_size, hdr + 80, sizeof(group_size));
                std::memcpy(&num_groups, hdr + 84, sizeof(num_groups));
                void* data = const_cast<uint8_t*>(hdr + data_off);
                setup_weight(data, static_cast<Precision>(file_prec_u32));
                const void* scales = scales_size ? (hdr + scales_off) : nullptr;
                if (!setup_quant_metadata(scales, group_size, num_groups,
                                          (size_t)scales_size, wref.c_str())) {
                    return false;
                }

                if (wref.find("embed_tokens") == std::string::npos) {
                    maybe_pack_fp16_weight(t, wref, data, packed_weights_);
                }
                maybe_pack_int8_weight(t, wref, data, packed_weights_);
                continue;
            }
        }

        // File mode: load weight from filesystem
        // Check if this weight is already loaded
        auto it = weight_map_.find(wpath);
        if (it != weight_map_.end()) {
            // Reuse existing mmap
            setup_weight(const_cast<void*>(shared_weights_[it->second].data()),
                         static_cast<Precision>(shared_weights_[it->second].header().precision));
        } else {
            // Load new mmap
            MappedFile mf;
            if (!mf.open(wpath.c_str())) {
                fprintf(stderr, "Engine: failed to load weight %s\n", wpath.c_str());
                return false;
            }

            size_t idx = shared_weights_.size();
            weight_map_[wpath] = idx;
            shared_weights_.push_back(std::move(mf));

            setup_weight(const_cast<void*>(shared_weights_[idx].data()),
                         static_cast<Precision>(shared_weights_[idx].header().precision));
            const auto& mapped = shared_weights_[idx];
            if (!setup_quant_metadata(mapped.scales(), mapped.group_size(), mapped.num_groups(),
                                      mapped.scales_size(), wpath.c_str())) {
                return false;
            }
        }

        if (it != weight_map_.end()) {
            const auto& mapped = shared_weights_[it->second];
            if (!setup_quant_metadata(mapped.scales(), mapped.group_size(), mapped.num_groups(),
                                      mapped.scales_size(), wpath.c_str())) {
                return false;
            }
        }

        // embed_tokens stays row-major for lookup; lm_head gets its own packed copy below.
        if (node.params.str[0].find("embed_tokens") == std::string::npos) {
            maybe_pack_fp16_weight(t, wpath, t.data, packed_weights_);
        }
        maybe_pack_int8_weight(t, wpath, t.data, packed_weights_);
    }

    // Find embed_tokens weight and create packed copy for lm_head
    for (auto& node : g.nodes) {
        if (node.op_type == OpType::CONSTANT && !node.params.str.empty()) {
            if (node.params.str[0].find("embed_tokens") != std::string::npos) {
                embed_weight_ = &g.runtime.tensors[node.id];
                // Create packed copy for lm_head matmul (one-time cost)
                if (embed_weight_->prec == Precision::FP16 && g_matmul_config.use_interleave_pack
                    && embed_packed_.empty()) {
                    int N = (int)embed_weight_->shape[0];
                    int K = (int)embed_weight_->shape[1];
                    const __fp16* orig = reinterpret_cast<const __fp16*>(embed_weight_->data);
                    __fp16* packed = pack_b_interleaved_full(orig, N, K, K);
                    size_t buf_size = (size_t)N * K * sizeof(__fp16);
                    embed_packed_.assign((uint8_t*)packed, (uint8_t*)packed + buf_size);
                    delete[] packed;
                }
            }
        }
    }

    exec_ctx.graph = &g;
    exec_ctx.pool  = &g.runtime.pool;
    exec_ctx.thread_pool = &thread_pool_;
    prepare_execution(exec_ctx);

    return true;
}

// ---------------------------------------------------------------------------
// allocate_caches — allocate KV cache buffers with metadata header
// ---------------------------------------------------------------------------

void LLMEngine::allocate_caches(Graph& g, int n_ctx) {
    // Find cache INPUT nodes and initialise their tensor shapes.
    // Supports both KV cache (cache_k/cache_v) and GDN state (gdn_state/gdn_conv).
    for (auto& node : g.nodes) {
        if (node.op_type != OpType::INPUT || node.params.str.empty()) continue;
        const std::string& name = node.params.str[0];

        if (name.find("cache_k") == 0) {
            int layer_idx = std::stoi(name.substr(7));
            if (layer_idx >= (int)caches_.size()) caches_.resize(layer_idx + 1);
            Tensor& t = g.runtime.tensors[node.id];
            t.prec = node.out_prec;
            t.shape[0] = node.out_shape[0];
            t.shape[1] = node.out_shape[1];
            t.shape[2] = node.out_shape[2];
            t.shape[3] = node.out_shape[3];
            t.compute_strides();
            caches_[layer_idx].k = &t;
            caches_[layer_idx].k_head_dim = (int)node.out_shape[0];
            caches_[layer_idx].k_num_heads = (int)node.out_shape[2];
        } else if (name.find("cache_v") == 0) {
            int layer_idx = std::stoi(name.substr(7));
            if (layer_idx >= (int)caches_.size()) caches_.resize(layer_idx + 1);
            Tensor& t = g.runtime.tensors[node.id];
            t.prec = node.out_prec;
            t.shape[0] = node.out_shape[0];
            t.shape[1] = node.out_shape[1];
            t.shape[2] = node.out_shape[2];
            t.shape[3] = node.out_shape[3];
            t.compute_strides();
            caches_[layer_idx].v = &t;
            caches_[layer_idx].v_head_dim = (int)node.out_shape[0];
            caches_[layer_idx].v_num_heads = (int)node.out_shape[2];
        } else if (name.find("gdn_state") == 0) {
            int layer_idx = std::stoi(name.substr(9));
            if (layer_idx >= (int)caches_.size()) caches_.resize(layer_idx + 1);
            Tensor& t = g.runtime.tensors[node.id];
            t.prec = node.out_prec;
            t.shape[0] = node.out_shape[0];
            t.shape[1] = node.out_shape[1];
            t.shape[2] = node.out_shape[2];
            t.shape[3] = node.out_shape[3];
            t.compute_strides();
            caches_[layer_idx].gdn_state = &t;
            caches_[layer_idx].gdn_v_dim = (int)node.out_shape[0];
            caches_[layer_idx].gdn_k_dim = (int)node.out_shape[1];
            caches_[layer_idx].gdn_num_heads = (int)node.out_shape[2];
            caches_[layer_idx].is_linear_attn = true;
        } else if (name.find("gdn_conv") == 0) {
            int layer_idx = std::stoi(name.substr(8));
            if (layer_idx >= (int)caches_.size()) caches_.resize(layer_idx + 1);
            Tensor& t = g.runtime.tensors[node.id];
            t.prec = node.out_prec;
            t.shape[0] = node.out_shape[0];
            t.shape[1] = node.out_shape[1];
            t.shape[2] = node.out_shape[2];
            t.shape[3] = node.out_shape[3];
            t.compute_strides();
            caches_[layer_idx].gdn_conv = &t;
            caches_[layer_idx].gdn_conv_groups = (int)node.out_shape[0];
            caches_[layer_idx].gdn_conv_kernel = (int)node.out_shape[1] + 1;  // kernel-1 stored, kernel = dim+1
        }
    }

    // Allocate cache/state data buffers from engine-owned persistent storage
    // (once, at load time). Graph runtime pools are execution-temporary only.
    for (auto& cp : caches_) {
        // Standard KV cache
        if (cp.k) {
            int hd = cp.k_head_dim;
            int nkv = cp.k_num_heads;
            size_t es = cp.k->element_size();
            size_t data_bytes = (size_t)hd * n_ctx * nkv * es;
            size_t total = CacheMetadata::SIZE + data_bytes;

            void* buf = persistent_pool_.acquire(total);
            std::memset(buf, 0, CacheMetadata::SIZE);

            auto* meta = cache_meta(buf);
            meta->current_seq_len = 0;
            meta->max_seq_len     = (uint64_t)n_ctx;
            meta->num_kv_heads    = (uint64_t)nkv;
            meta->head_dim        = (uint64_t)hd;

            cp.k->data     = buf;
            cp.k->mem_type = MemoryType::POOLED;
            cp.k->owner_id = persistent_pool_.id();
            cp.k->storage_id = persistent_pool_.storage_id(buf);
            cp.k->shape[0] = (int64_t)total / (int64_t)es;
            cp.k->compute_strides();
        }
        if (cp.v) {
            int vd = cp.v_head_dim;
            int nkv = cp.v_num_heads;
            size_t es = cp.v->element_size();
            size_t data_bytes = (size_t)vd * n_ctx * nkv * es;
            size_t total = CacheMetadata::SIZE + data_bytes;

            void* buf = persistent_pool_.acquire(total);
            std::memset(buf, 0, CacheMetadata::SIZE);

            auto* meta = cache_meta(buf);
            meta->current_seq_len = 0;
            meta->max_seq_len     = (uint64_t)n_ctx;
            meta->num_kv_heads    = (uint64_t)nkv;
            meta->v_head_dim      = (uint64_t)vd;

            cp.v->data     = buf;
            cp.v->mem_type = MemoryType::POOLED;
            cp.v->owner_id = persistent_pool_.id();
            cp.v->storage_id = persistent_pool_.storage_id(buf);
            cp.v->shape[0] = (int64_t)total / (int64_t)es;
            cp.v->compute_strides();
        }

        // GDN recurrent state: [v_dim, k_dim, num_heads] FP32
        // No CacheMetadata header — GDN state is a plain FP32 buffer.
        if (cp.gdn_state) {
            size_t data_bytes = (size_t)cp.gdn_v_dim * cp.gdn_k_dim * cp.gdn_num_heads * sizeof(float);
            void* buf = persistent_pool_.acquire(data_bytes);
            std::memset(buf, 0, data_bytes);
            cp.gdn_state->data     = buf;
            cp.gdn_state->mem_type = MemoryType::POOLED;
            cp.gdn_state->owner_id = persistent_pool_.id();
            cp.gdn_state->storage_id = persistent_pool_.storage_id(buf);
            cp.gdn_state->shape[0] = (int64_t)cp.gdn_v_dim;
            cp.gdn_state->shape[1] = (int64_t)cp.gdn_k_dim;
            cp.gdn_state->shape[2] = (int64_t)cp.gdn_num_heads;
            cp.gdn_state->shape[3] = 1;
            cp.gdn_state->compute_strides();
        }
        // GDN conv state: [groups, kernel-1] FP32
        if (cp.gdn_conv) {
            int kernel_m1 = cp.gdn_conv_kernel - 1;
            size_t data_bytes = (size_t)cp.gdn_conv_groups * kernel_m1 * sizeof(float);
            void* buf = persistent_pool_.acquire(data_bytes);
            std::memset(buf, 0, data_bytes);
            cp.gdn_conv->data     = buf;
            cp.gdn_conv->mem_type = MemoryType::POOLED;
            cp.gdn_conv->owner_id = persistent_pool_.id();
            cp.gdn_conv->storage_id = persistent_pool_.storage_id(buf);
            cp.gdn_conv->shape[0] = (int64_t)cp.gdn_conv_groups;
            cp.gdn_conv->shape[1] = (int64_t)kernel_m1;
            cp.gdn_conv->shape[2] = 1;
            cp.gdn_conv->shape[3] = 1;
            cp.gdn_conv->compute_strides();
        }
    }
}

// ---------------------------------------------------------------------------
// load — load both graphs, set up shared weights and caches
// ---------------------------------------------------------------------------

void LLMEngine::set_profile_enabled(bool enabled) {
    exec_ctx_prefill_.profile_enabled = enabled;
    exec_ctx_decode_.profile_enabled = enabled;
}

void LLMEngine::reset_profiles() {
    reset_profile_stats(exec_ctx_prefill_);
    reset_profile_stats(exec_ctx_decode_);
}

bool LLMEngine::load(const EngineConfig& cfg) {
    cfg_ = cfg;
    cfg_.num_threads = std::max(cfg_.num_threads, 1);
    caches_.clear();
    persistent_pool_.clear();
    thread_pool_.resize(cfg_.num_threads);
    exec_ctx_prefill_.thread_pool = &thread_pool_;
    exec_ctx_decode_.thread_pool  = &thread_pool_;
    exec_ctx_prefill_.backend     = &cpu_backend_;
    exec_ctx_decode_.backend      = &cpu_backend_;
    exec_ctx_prefill_.reuse_static_workspace = false;
    exec_ctx_prefill_.reuse_same_shape_workspace = true;
    exec_ctx_decode_.reuse_static_workspace  = true;
    exec_ctx_decode_.reuse_same_shape_workspace = false;

    // Load the .mollm package (sets up weights mmap, extracts graphs to temp
    // files, parses metadata for weight offset map).
    if (cfg.package_path.empty()) {
        fprintf(stderr, "Engine: package_path is required (use .mollm package)\n");
        return false;
    }
    std::string pf_path, dc_path;
    {
        std::string tok_tmp, jin_tmp;
        if (!load_package(cfg.package_path, pf_path, dc_path, tok_tmp, jin_tmp)) {
            return false;
        }
        if (!tok_tmp.empty()) {
            cfg_.tokenizer_path = tok_tmp;
        }
    }

    // Load prefill graph first (establishes shared weights)
    const std::string& pf_path_load = cfg.use_decode_as_prefill ? dc_path : pf_path;
    if (!load_graph(graph_prefill_, exec_ctx_prefill_, pf_path_load.c_str())) {
        return false;
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

    // Migrate cache tensor pointers from prefill graph to decode graph.
    // Done once at load time — the physical cache buffers are shared across
    // both graphs for the engine's entire lifetime.
    for (auto& node : graph_decode_.nodes) {
        if (node.op_type != OpType::INPUT || node.params.str.empty()) continue;
        const std::string& name = node.params.str[0];

        if (name.find("cache_k") == 0) {
            int layer_idx = std::stoi(name.substr(7));
            if (layer_idx < (int)caches_.size() && caches_[layer_idx].k) {
                graph_decode_.runtime.tensors[node.id] = *caches_[layer_idx].k;
            }
        } else if (name.find("cache_v") == 0) {
            int layer_idx = std::stoi(name.substr(7));
            if (layer_idx < (int)caches_.size() && caches_[layer_idx].v) {
                graph_decode_.runtime.tensors[node.id] = *caches_[layer_idx].v;
            }
        } else if (name.find("gdn_state") == 0) {
            int layer_idx = std::stoi(name.substr(9));
            if (layer_idx < (int)caches_.size() && caches_[layer_idx].gdn_state) {
                graph_decode_.runtime.tensors[node.id] = *caches_[layer_idx].gdn_state;
            }
        } else if (name.find("gdn_conv") == 0) {
            int layer_idx = std::stoi(name.substr(8));
            if (layer_idx < (int)caches_.size() && caches_[layer_idx].gdn_conv) {
                graph_decode_.runtime.tensors[node.id] = *caches_[layer_idx].gdn_conv;
            }
        }
    }

    reset();
    return true;
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void LLMEngine::reset() {
    release_graph_temporaries(graph_decode_);
    invalidate_workspace_key(exec_ctx_decode_);
    clear_graph_borrowed_views(graph_prefill_);
    past_len_ = 0;
    // KV cache: only clear metadata header (current_seq_len = 0).
    // GDN state: zero the entire recurrent state buffer (it's small, ~256KB
    // per layer, and GDN reads stale state without causal mask protection).
    // GDN conv state: also zero (short conv uses conv_state for continuity).
    for (auto& cp : caches_) {
        if (cp.k) cache_meta(cp.k->data)->current_seq_len = 0;
        if (cp.v) cache_meta(cp.v->data)->current_seq_len = 0;
        if (cp.gdn_state) {
            size_t sz = (size_t)cp.gdn_v_dim * cp.gdn_k_dim * cp.gdn_num_heads * sizeof(float);
            std::memset(cp.gdn_state->data, 0, sz);
        }
        if (cp.gdn_conv) {
            size_t sz = (size_t)cp.gdn_conv_groups * (cp.gdn_conv_kernel - 1) * sizeof(float);
            std::memset(cp.gdn_conv->data, 0, sz);
        }
    }
}

// ---------------------------------------------------------------------------
// embed
// ---------------------------------------------------------------------------

Tensor LLMEngine::embed(const std::vector<int>& token_ids, int pad_to) {
    int n = (int)token_ids.size();
    int seq_len = (pad_to > n) ? pad_to : n;

    if (embed_weight_ && embed_weight_->data) {
        int vocab_size = (int)embed_weight_->shape[0];
        int hidden_dim = (int)embed_weight_->shape[1];

        // Allocate from the prefill graph's pool so the buffer is tracked and
        // reused. Previously this used `new float[]` with MemoryType::OWNED,
        // but Tensor has no destructor — the buffer leaked every call.
        size_t nbytes = (size_t)hidden_dim * seq_len * sizeof(float);
        void* buf = graph_prefill_.runtime.pool.acquire(nbytes);
        Tensor t = Tensor::create(Precision::FP32, MemoryType::POOLED,
                                  hidden_dim, seq_len, 1, 1, buf);
        t.owner_id = graph_prefill_.runtime.pool.id();
        t.storage_id = graph_prefill_.runtime.pool.storage_id(buf);
        // Zero-init so padding positions (if any) are well-defined zeros.
        std::memset(t.data, 0, nbytes);

        if (embed_weight_->prec == Precision::FP16) {
            // Row-major FP16 — simple direct access
            const __fp16* embed_data = reinterpret_cast<const __fp16*>(embed_weight_->data);
            for (int s = 0; s < n; s++) {
                int tid = token_ids[s];
                if (tid < 0 || tid >= vocab_size) tid = 0;
                float* dst = t.ptr<float>() + s * hidden_dim;
                for (int d = 0; d < hidden_dim; d++) {
                    dst[d] = (float)embed_data[tid * hidden_dim + d];
                }
            }
        } else {
            // Row-major FP32
            const float* embed_data = embed_weight_->ptr<float>();
            for (int s = 0; s < n; s++) {
                int tid = token_ids[s];
                if (tid < 0 || tid >= vocab_size) tid = 0;
                float* dst = t.ptr<float>() + s * hidden_dim;
                for (int d = 0; d < hidden_dim; d++) {
                    dst[d] = embed_data[tid * hidden_dim + d];
                }
            }
        }
        return t;
    }

    int hidden_dim = 2048;
    void* buf = graph_prefill_.runtime.pool.acquire(hidden_dim * seq_len * sizeof(float));
    Tensor t = Tensor::create(Precision::FP32, MemoryType::POOLED, hidden_dim, seq_len, 1, 1, buf);
    t.owner_id = graph_prefill_.runtime.pool.id();
    t.storage_id = graph_prefill_.runtime.pool.storage_id(buf);
    std::memset(t.data, 0, t.nbytes());
    return t;
}

// ---------------------------------------------------------------------------
// run_lmhead
// ---------------------------------------------------------------------------

int LLMEngine::run_lmhead(const Tensor& hidden, int n_tokens) {
    if (!embed_weight_ || !embed_weight_->data) return 0;

    int vocab_size = (int)embed_weight_->shape[0];
    int hidden_dim = (int)embed_weight_->shape[1];
    int seq_len = (int)hidden.shape[1];

    // Read the last real token, not necessarily the last position
    // (which may be padded when graph seq_len > n_tokens).
    int last_pos = (n_tokens > 0 && n_tokens <= seq_len) ? n_tokens - 1 : seq_len - 1;

    // hidden is [hidden_dim, seq_len], we want [hidden_dim, 1] for matmul
    // embed_weight is [vocab_size, hidden_dim] — we use it as weight B
    // output will be [vocab_size, 1] — we take argmax

    // Create a view of the last hidden row as A: [hidden_dim, 1]
    Tensor A = hidden;
    A.shape[1] = 1;
    A.data = static_cast<char*>(hidden.data) + last_pos * hidden_dim * sizeof(float);
    A.compute_strides();

    // Output: [vocab_size, 1]
    void* c_buf = graph_prefill_.runtime.pool.acquire(vocab_size * sizeof(float));
    Tensor C = Tensor::create(Precision::FP32, MemoryType::POOLED, vocab_size, 1, 1, 1, c_buf);
    C.owner_id = graph_prefill_.runtime.pool.id();
    C.storage_id = graph_prefill_.runtime.pool.storage_id(c_buf);

    // Use packed embed copy for fast lm_head matmul (if available)
    if (!embed_packed_.empty()) {
        Tensor B_packed = Tensor::create(Precision::FP16, MemoryType::EXTERNAL,
                                         vocab_size, hidden_dim, 1, 1,
                                         embed_packed_.data());
        kernel_matmul_fp32(A, B_packed, C, exec_ctx_decode_.thread_pool);
    } else {
        kernel_matmul_fp32(A, *embed_weight_, C, exec_ctx_decode_.thread_pool);
    }

    float* scores = C.ptr<float>();
    int token = sample_token(scores, vocab_size,
                              cfg_.temperature, cfg_.top_k, cfg_.top_p,
                              &cfg_.seed);

    release_pool_tensor(graph_prefill_.runtime.pool, C);

    return token;
}

std::vector<float> LLMEngine::run_lmhead_raw(const Tensor& hidden, int n_tokens,
                                               bool all_positions) {
    if (!embed_weight_ || !embed_weight_->data) return {};

    int vocab_size = (int)embed_weight_->shape[0];
    int hidden_dim = (int)embed_weight_->shape[1];
    int seq_len = (int)hidden.shape[1];

    int n_pos = all_positions ? n_tokens : 1;
    std::vector<float> logits(n_pos * vocab_size);

    for (int p = 0; p < n_pos; p++) {
        int pos = all_positions ? p : ((n_tokens > 0 && n_tokens <= seq_len) ? n_tokens - 1 : seq_len - 1);

        Tensor A = hidden;
        A.shape[1] = 1;
        A.data = static_cast<char*>(hidden.data) + pos * hidden_dim * sizeof(float);
        A.compute_strides();

        Tensor C = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                  vocab_size, 1, 1, 1, logits.data() + p * vocab_size);

        if (!embed_packed_.empty()) {
            Tensor B_packed = Tensor::create(Precision::FP16, MemoryType::EXTERNAL,
                                             vocab_size, hidden_dim, 1, 1,
                                             embed_packed_.data());
            kernel_matmul_fp32(A, B_packed, C, exec_ctx_decode_.thread_pool);
        } else {
            kernel_matmul_fp32(A, *embed_weight_, C, exec_ctx_decode_.thread_pool);
        }
    }

    return logits;
}

// ---------------------------------------------------------------------------
// rope / mask helpers
// ---------------------------------------------------------------------------

Tensor LLMEngine::build_causal_mask(int seq_len, int past_len) {
    int total = past_len + seq_len;
    void* buf = graph_prefill_.runtime.pool.acquire(total * seq_len * sizeof(float));
    Tensor mask = Tensor::create(Precision::FP32, MemoryType::POOLED, total, seq_len, 1, 1, buf);
    mask.owner_id = graph_prefill_.runtime.pool.id();
    mask.storage_id = graph_prefill_.runtime.pool.storage_id(buf);
    float* d = mask.ptr<float>();
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < total; j++) {
            d[i * total + j] = (j > past_len + i) ? -1e38f : 0.f;
        }
    }
    return mask;
}

void LLMEngine::generate_rope_cache(int seq_len, int start_pos,
                                    Tensor& cos, Tensor& sin) {
    int half = cfg_.rope_dim / 2;

    void* cb = graph_prefill_.runtime.pool.acquire(half * seq_len * sizeof(float));
    void* sb = graph_prefill_.runtime.pool.acquire(half * seq_len * sizeof(float));
    cos = Tensor::create(Precision::FP32, MemoryType::POOLED, half, seq_len, 1, 1, cb);
    sin = Tensor::create(Precision::FP32, MemoryType::POOLED, half, seq_len, 1, 1, sb);
    cos.owner_id = graph_prefill_.runtime.pool.id();
    sin.owner_id = graph_prefill_.runtime.pool.id();
    cos.storage_id = graph_prefill_.runtime.pool.storage_id(cb);
    sin.storage_id = graph_prefill_.runtime.pool.storage_id(sb);

    for (int n = 0; n < seq_len; n++) {
        int pos = start_pos + n;
        for (int i = 0; i < half; i++) {
            float theta = 1.0f / std::pow(cfg_.rope_theta, 2.0f * i / cfg_.rope_dim);
            float angle = pos * theta;
            cos.ptr<float>()[n * half + i] = std::cos(angle);
            sin.ptr<float>()[n * half + i] = std::sin(angle);
        }
    }
}

// ---------------------------------------------------------------------------
// run_graph — feed inputs and execute a graph
// ---------------------------------------------------------------------------

Tensor LLMEngine::run_graph(Graph& graph, ExecContext& exec_ctx,
                             const Tensor& hidden, const Tensor& mask,
                             const Tensor& cos, const Tensor& sin) {
    auto& tensors = graph.runtime.tensors;

    // Feed graph inputs by borrowing the caller-owned/helper tensors directly.
    // hidden/mask/cos/sin lifetime is managed by the caller; cache/state INPUTs
    // point at engine persistent storage and are set up at load time.
    for (auto& node : graph.nodes) {
        if (node.op_type != OpType::INPUT) continue;
        if (node.params.str.empty()) continue;

        const std::string& name = node.params.str[0];
        Tensor* t = &tensors[node.id];

        if (name == "hidden") {
            *t = hidden;
        } else if (name == "mask") {
            *t = mask;
        } else if (name == "cos") {
            *t = cos;
        } else if (name == "sin") {
            *t = sin;
        }
        // cache_k/cache_v/gdn state are persistent INPUT tensors.
    }

    execute_graph(exec_ctx);

    if (!graph.graph_outputs.empty()) {
        uint32_t out_id = graph.graph_outputs.back();
        return tensors[out_id];
    }

    fprintf(stderr, "Engine: no graph output found\n");
    return Tensor();
}

// ---------------------------------------------------------------------------
// prefill / decode
// ---------------------------------------------------------------------------

void LLMEngine::release_prefill_buffers() {
    release_graph_temporaries(graph_prefill_);
    invalidate_workspace_key(exec_ctx_prefill_);
}

int LLMEngine::prefill(const std::vector<int>& token_ids) {
    int n = (int)token_ids.size();
    if (n == 0) return -1;

    // Determine the graph's expected seq_len from its hidden input shape.
    int graph_seq_len = 1;
    for (auto& node : graph_prefill_.nodes) {
        if (node.op_type == OpType::INPUT && !node.params.str.empty()
            && node.params.str[0] == "hidden") {
            graph_seq_len = (int)node.out_shape[1];
            break;
        }
    }

    // Chunked prefill: split long prompts into graph_seq_len-sized chunks.
    // Each chunk is processed independently, appending to the KV cache.
    // past_len_ can exceed graph_seq_len (cache supports n_ctx); only each
    // chunk's token count must be <= graph_seq_len.
    int offset = 0;
    int last_token = -1;
    while (offset < n) {
        int remaining = n - offset;
        if (past_len_ >= cfg_.n_ctx) {
            fprintf(stderr, "prefill: context full (past=%d >= n_ctx=%d). Use /reset.\n",
                    past_len_, cfg_.n_ctx);
            return -1;
        }
        int chunk_size = std::min(remaining, graph_seq_len);
        std::vector<int> chunk(token_ids.begin() + offset,
                               token_ids.begin() + offset + chunk_size);
        last_token = prefill_chunk(chunk, past_len_);
        if (last_token < 0) return -1;
        offset += chunk_size;
    }
    return last_token;
}

Tensor LLMEngine::prefill_hidden(const std::vector<int>& token_ids) {
    int n = (int)token_ids.size();
    if (n == 0) return Tensor();

    int graph_seq_len = 1;
    for (auto& node : graph_prefill_.nodes) {
        if (node.op_type == OpType::INPUT && !node.params.str.empty()
            && node.params.str[0] == "hidden") {
            graph_seq_len = (int)node.out_shape[1];
            break;
        }
    }

    // Dynamic shape mode: chunked prefill is supported via repeated calls to
    // prefill_hidden (each chunk appends to KV cache). No padding — the
    // graph's SEQ dims are filled with the actual chunk size n at runtime.

    // DYNAMIC mode: no padding, runtime fills SEQ/MUL/ADD dims via DimExpr.
    // STATIC_PADDED mode: pad short prompts to graph_seq_len (A/B comparison).
    const bool use_padding = cfg_.static_padded && n < graph_seq_len;
    Tensor h;
    Tensor cos, sin;
    Tensor mask;

    if (use_padding) {
        exec_ctx_prefill_.runtime_seq_len = n;
        exec_ctx_prefill_.static_padded   = true;
        exec_ctx_prefill_.padded_seq_len  = graph_seq_len;
        inject_runtime_shapes(exec_ctx_prefill_);
        h = embed(token_ids, graph_seq_len);  // zero-padded to graph_seq_len
        h.shape[1] = graph_seq_len;
        h.compute_strides();
        generate_rope_cache(graph_seq_len, past_len_, cos, sin);
        mask = build_causal_mask(graph_seq_len, past_len_);
    } else {
        exec_ctx_prefill_.runtime_seq_len = n;
        exec_ctx_prefill_.static_padded   = false;
        exec_ctx_prefill_.padded_seq_len  = -1;
        inject_runtime_shapes(exec_ctx_prefill_);
        h = embed(token_ids);
        h.shape[1] = n;
        h.compute_strides();
        generate_rope_cache(n, past_len_, cos, sin);
        mask = build_causal_mask(n, past_len_);
    }

    // n_real_tokens injection is done by inject_runtime_shapes() above.

    // Set cache metadata
    for (auto& cp : caches_) {
        if (cp.k) cache_meta(cp.k->data)->current_seq_len = (uint64_t)past_len_;
        if (cp.v) cache_meta(cp.v->data)->current_seq_len = (uint64_t)past_len_;
    }

    mollm_set_matmul_profile_phase("prefill_graph");
    Tensor out = run_graph(graph_prefill_, exec_ctx_prefill_, h, mask, cos, sin);
    mollm_set_matmul_profile_phase("unscoped");
    Tensor copied = copy_tensor_contiguous(out, hidden_output_copy_);
    release_pool_tensor(graph_prefill_.runtime.pool, h);
    release_pool_tensor(graph_prefill_.runtime.pool, mask);
    release_pool_tensor(graph_prefill_.runtime.pool, cos);
    release_pool_tensor(graph_prefill_.runtime.pool, sin);

    past_len_ += n;

    for (auto& cp : caches_) {
        if (cp.k) cache_meta(cp.k->data)->current_seq_len = (uint64_t)past_len_;
        if (cp.v) cache_meta(cp.v->data)->current_seq_len = (uint64_t)past_len_;
    }

    finish_graph_temporaries(graph_prefill_, exec_ctx_prefill_);
    return copied;
}

int LLMEngine::prefill_chunk(const std::vector<int>& token_ids, int past) {
    int n = (int)token_ids.size();
    if (n == 0) return -1;

    // Determine the graph's expected seq_len from its hidden input shape.
    int graph_seq_len = 1;
    for (auto& node : graph_prefill_.nodes) {
        if (node.op_type == OpType::INPUT && !node.params.str.empty()
            && node.params.str[0] == "hidden") {
            graph_seq_len = (int)node.out_shape[1];
            break;
        }
    }

    // Check that this chunk fits in the prefill graph's static seq_len.
    // `past` can be >= graph_seq_len (cache supports n_ctx=4096); only the
    // chunk size n must be <= graph_seq_len.
    if (n > graph_seq_len) {
        fprintf(stderr, "prefill_chunk: n=%d > graph_seq_len=%d\n",
                n, graph_seq_len);
        return -1;
    }

    // DYNAMIC mode: no padding. runtime fills SEQ/MUL/ADD dims via DimExpr
    // evaluation against runtime_seq_len. Symbolic reshape handles N*seq.
    //
    // STATIC_PADDED mode: pad short chunks (n < graph_seq_len) to graph_seq_len.
    // Stateful ops (GDN/SHORTCONV) receive n_real via params to skip padding
    // positions. Full chunks (n == graph_seq_len) skip padding (identical work).
    const bool use_padding = cfg_.static_padded && n < graph_seq_len;
    Tensor h;
    Tensor cos, sin;
    Tensor mask;

    if (use_padding) {
        exec_ctx_prefill_.runtime_seq_len = n;          // real token count
        exec_ctx_prefill_.static_padded   = true;
        exec_ctx_prefill_.padded_seq_len  = graph_seq_len;
        inject_runtime_shapes(exec_ctx_prefill_);
        h = embed(token_ids, graph_seq_len);             // zero-padded to graph_seq_len
        h.shape[1] = graph_seq_len;
        h.compute_strides();
        generate_rope_cache(graph_seq_len, past, cos, sin);
        mask = build_causal_mask(graph_seq_len, past);
    } else {
        exec_ctx_prefill_.runtime_seq_len = n;
        exec_ctx_prefill_.static_padded   = false;
        exec_ctx_prefill_.padded_seq_len  = -1;
        inject_runtime_shapes(exec_ctx_prefill_);
        h = embed(token_ids);
        h.shape[1] = n;
        h.compute_strides();
        generate_rope_cache(n, past, cos, sin);
        mask = build_causal_mask(n, past);
    }

    // n_real_tokens injection is now done by inject_runtime_shapes() above.

    // Set cache metadata so SDPA knows the existing context length.
    for (auto& cp : caches_) {
        if (cp.k) cache_meta(cp.k->data)->current_seq_len = (uint64_t)past;
        if (cp.v) cache_meta(cp.v->data)->current_seq_len = (uint64_t)past;
    }

    mollm_set_matmul_profile_phase("prefill_graph");
    Tensor out = run_graph(graph_prefill_, exec_ctx_prefill_, h, mask, cos, sin);

    past_len_ = past + n;

    // Update cache metadata for decode (past + current prefill tokens)
    for (auto& cp : caches_) {
        if (cp.k) cache_meta(cp.k->data)->current_seq_len = (uint64_t)past_len_;
        if (cp.v) cache_meta(cp.v->data)->current_seq_len = (uint64_t)past_len_;
    }

    // Cache migration (prefill→decode graph) is done once at load time.
    // Both graphs share the same physical cache buffers.

    mollm_set_matmul_profile_phase("prefill_lmhead");
    int token = run_lmhead(out, n);
    mollm_set_matmul_profile_phase("unscoped");
    release_pool_tensor(graph_prefill_.runtime.pool, h);
    release_pool_tensor(graph_prefill_.runtime.pool, mask);
    release_pool_tensor(graph_prefill_.runtime.pool, cos);
    release_pool_tensor(graph_prefill_.runtime.pool, sin);
    finish_graph_temporaries(graph_prefill_, exec_ctx_prefill_);
    return token;
}

int LLMEngine::decode(int token_id) {
    Tensor h = embed({token_id});
    h.shape[1] = 1;
    h.compute_strides();

    Tensor cos, sin;
    generate_rope_cache(1, past_len_, cos, sin);

    Tensor mask = build_causal_mask(1, past_len_);

    mollm_set_matmul_profile_phase("decode_graph");
    Tensor out = run_graph(graph_decode_, exec_ctx_decode_, h, mask, cos, sin);

    past_len_++;

    // Update cache metadata
    for (auto& cp : caches_) {
        if (cp.k) cache_meta(cp.k->data)->current_seq_len = (uint64_t)past_len_;
        if (cp.v) cache_meta(cp.v->data)->current_seq_len = (uint64_t)past_len_;
    }

    mollm_set_matmul_profile_phase("decode_lmhead");
    int token = run_lmhead(out);
    mollm_set_matmul_profile_phase("unscoped");
    release_pool_tensor(graph_prefill_.runtime.pool, h);
    release_pool_tensor(graph_prefill_.runtime.pool, mask);
    release_pool_tensor(graph_prefill_.runtime.pool, cos);
    release_pool_tensor(graph_prefill_.runtime.pool, sin);
    finish_graph_temporaries(graph_decode_, exec_ctx_decode_);
    return token;
}

Tensor LLMEngine::decode_hidden(int token_id) {
    Tensor h = embed({token_id});
    h.shape[1] = 1;
    h.compute_strides();

    // Decode graph is all-STATIC (seq=1); no dynamic shape injection needed.
    exec_ctx_decode_.runtime_seq_len = -1;
    exec_ctx_decode_.static_padded   = false;
    exec_ctx_decode_.padded_seq_len  = -1;

    Tensor cos, sin;
    generate_rope_cache(1, past_len_, cos, sin);

    Tensor mask = build_causal_mask(1, past_len_);

    mollm_set_matmul_profile_phase("decode_graph");
    Tensor out = run_graph(graph_decode_, exec_ctx_decode_, h, mask, cos, sin);
    mollm_set_matmul_profile_phase("unscoped");
    Tensor copied = copy_tensor_contiguous(out, hidden_output_copy_);
    release_pool_tensor(graph_prefill_.runtime.pool, h);
    release_pool_tensor(graph_prefill_.runtime.pool, mask);
    release_pool_tensor(graph_prefill_.runtime.pool, cos);
    release_pool_tensor(graph_prefill_.runtime.pool, sin);

    past_len_++;

    for (auto& cp : caches_) {
        if (cp.k) cache_meta(cp.k->data)->current_seq_len = (uint64_t)past_len_;
        if (cp.v) cache_meta(cp.v->data)->current_seq_len = (uint64_t)past_len_;
    }

    finish_graph_temporaries(graph_decode_, exec_ctx_decode_);
    return copied;
}

// ---------------------------------------------------------------------------
// .mollm package loading
// ---------------------------------------------------------------------------

#include <fstream>
#include <json.hpp>

using json = nlohmann::json;

static constexpr uint32_t PACKAGE_MAGIC_C = 0x4D4C4F4D;  // "MOLM"

bool LLMEngine::load_package(const std::string& path, std::string& pf_path,
                              std::string& dc_path, std::string& tok_path,
                              std::string& jin_path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Engine: failed to open package %s\n", path.c_str());
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return false; }
    size_t file_size = st.st_size;
    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "Engine: mmap failed for %s\n", path.c_str());
        return false;
    }
    package_mmap_ = mapped;
    package_mmap_size_ = file_size;

    const uint8_t* base = static_cast<const uint8_t*>(mapped);

    if (file_size < 128) {
        fprintf(stderr, "Engine: package too small\n");
        return false;
    }

    uint32_t magic, version;
    std::memcpy(&magic, base + 0, 4);
    std::memcpy(&version, base + 4, 4);
    if (magic != PACKAGE_MAGIC_C) {
        fprintf(stderr, "Engine: bad package magic 0x%08x\n", magic);
        return false;
    }

    // Parse header (6 offset/len pairs at bytes 8-103)
    uint64_t meta_off, meta_len, tok_off, tok_len, jin_off, jin_len;
    uint64_t pf_off, pf_len, dc_off, dc_len, w_off, w_len;
    std::memcpy(&meta_off, base + 8, 8);
    std::memcpy(&meta_len, base + 16, 8);
    std::memcpy(&tok_off, base + 24, 8);
    std::memcpy(&tok_len, base + 32, 8);
    std::memcpy(&jin_off, base + 40, 8);
    std::memcpy(&jin_len, base + 48, 8);
    std::memcpy(&pf_off, base + 56, 8);
    std::memcpy(&pf_len, base + 64, 8);
    std::memcpy(&dc_off, base + 72, 8);
    std::memcpy(&dc_len, base + 80, 8);
    std::memcpy(&w_off, base + 88, 8);
    std::memcpy(&w_len, base + 96, 8);

    package_weights_base_ = base + w_off;

    // Parse metadata JSON
    std::string meta_str(reinterpret_cast<const char*>(base + meta_off), meta_len);
    try {
        auto meta = json::parse(meta_str);
        if (meta.contains("prefill_seq_len")) {
            package_prefill_seq_len_ = meta["prefill_seq_len"].get<int>();
        }
        if (meta.contains("weights")) {
            for (auto& [name, info] : meta["weights"].items()) {
                uint64_t off = info[0].get<uint64_t>();
                uint64_t sz = info[1].get<uint64_t>();
                package_weight_map_[name] = {off, sz};
            }
        }
    } catch (std::exception& e) {
        fprintf(stderr, "Engine: failed to parse package metadata: %s\n", e.what());
        return false;
    }

    // Extract graphs + tokenizer + jinja to temp files
    pid_t pid = getpid();
    auto write_tmp = [&](const char* label, uint64_t off, uint64_t len, std::string& out_path) {
        if (len == 0) return;
        out_path = "/tmp/mollm_pkg_" + std::to_string(pid) + "_" + label;
        std::ofstream f(out_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(base + off), len);
        if (f) temp_files_.push_back(out_path);
    };

    write_tmp("prefill.graph", pf_off, pf_len, pf_path);
    write_tmp("decode.graph", dc_off, dc_len, dc_path);
    write_tmp("tokenizer.json", tok_off, tok_len, tok_path);
    write_tmp("chat_template.jinja", jin_off, jin_len, jin_path);

    fprintf(stderr, "Engine: loaded package %s (%.1f MB, %zu weights, prefill_seq=%d)\n",
            path.c_str(), file_size / 1e6, package_weight_map_.size(),
            package_prefill_seq_len_);
    return true;
}

void LLMEngine::dump_prefill_add_outputs(const char* dir) {
    int add_idx = 0;
    for (auto& node : graph_prefill_.nodes) {
        if (node.op_type != OpType::ADD) continue;
        auto& t = graph_prefill_.runtime.tensors[node.id];
        if (!t.data || t.prec != Precision::FP32) continue;
        add_idx++;
        char fname[256];
        snprintf(fname, sizeof(fname), "%s/add_%04d.f32", dir, add_idx);
        FILE* f = fopen(fname, "wb");
        if (!f) continue;
        const float* p = t.ptr<float>();
        int d0 = (int)t.shape[0];
        int d1 = (int)t.shape[1];
        if (d1 > 1) {
            size_t ldx = t.stride[1] / sizeof(float);
            fwrite(p + (d1 - 1) * ldx, sizeof(float), d0, f);
        } else {
            fwrite(p, sizeof(float), d0, f);
        }
        fclose(f);
    }
    fprintf(stderr, "Dumped %d ADD outputs to %s/\n", add_idx, dir);
}
