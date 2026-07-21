#include "engine/engine.h"
#include "kernels/matmul.h"
#ifdef MOLLM_METAL
#include "engine/metal_backend.h"
// Downcast the owned Backend* to MetalBackend* (non-null iff Metal active).
static inline MetalBackend* as_metal(const std::unique_ptr<Backend>& b) {
    return static_cast<MetalBackend*>(b.get());
}
#endif

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

namespace {

volatile uint8_t g_package_warmup_sink = 0;

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && std::strcmp(value, "0") != 0;
}

bool is_2d_linear_weight(const Tensor& t) {
    return t.shape[2] == 1 && t.shape[3] == 1;
}

bool int8_q8dot_repack_supported(const Tensor& t) {
#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    return is_2d_linear_weight(t) &&
           t.prec == Precision::INT8 && t.group_size > 0 && (t.group_size % 32) == 0;
#else
    (void)t;
    return false;
#endif
}

bool int4_q4dot_repack_supported(const Tensor& t) {
#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    return is_2d_linear_weight(t) &&
           t.prec == Precision::INT4 && t.group_size > 0 &&
           (t.group_size % 32) == 0 && t.shape[1] > 0 && (t.shape[1] % 32) == 0;
#else
    (void)t;
    return false;
#endif
}

bool int4_q4dot_kernel_available() {
#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    return true;
#else
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
    if (t.prec != Precision::INT8 || !g_matmul_config.use_interleave_pack ||
        !is_2d_linear_weight(t)) {
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
    if (t.prec != Precision::FP16 || !g_matmul_config.use_interleave_pack ||
        !is_2d_linear_weight(t)) {
        return;
    }

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

void maybe_pack_int4_g128_weight(Tensor& t, const std::string& key,
                                 const void* q4dot_data,
                                 PackedWeightMap& packed_weights) {
#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    if (!is_2d_linear_weight(t)) return;
    if (!env_flag_enabled("MOLLM_W4_PACKED_BG128")) return;
    if (t.prec != Precision::INT4 || !q4dot_data || !t.scales ||
        t.group_size != 128 || t.shape[1] <= 0 || (t.shape[1] % 128) != 0) {
        return;
    }

    int N = (int)t.shape[0];
    int K = (int)t.shape[1];
    std::string g128_key = key + "#int4_q4g128";
    auto it = packed_weights.find(g128_key);
    if (it == packed_weights.end()) {
        uint8_t* b_g128 = pack_b_q4dot_g128_full(
            reinterpret_cast<const uint8_t*>(q4dot_data),
            t.scales, N, K, (int)t.groups_per_row);
        if (!b_g128) return;
        size_t buf_size = pack_b_q4dot_g128_bytes(N, K);
        std::vector<uint8_t> buf(b_g128, b_g128 + buf_size);
        delete[] b_g128;
        it = packed_weights.emplace(g128_key, std::move(buf)).first;
    }
    t.q4_g128_data = it->second.data();
#else
    (void)t;
    (void)key;
    (void)q4dot_data;
    (void)packed_weights;
#endif
}

void maybe_pack_int4_weight(Tensor& t, const std::string& key,
                            const void* weight_data,
                            PackedWeightMap& packed_weights) {
#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    if (!is_2d_linear_weight(t)) return;
    if (t.is_q4_g128_packed) {
        t.q4_g128_data = weight_data;
        return;
    }
    if (t.is_q4_repacked) {
        t.q4_repack_data = weight_data;
        maybe_pack_int4_g128_weight(t, key, weight_data, packed_weights);
        return;
    }
    if (!g_matmul_config.use_interleave_pack || !int4_q4dot_repack_supported(t)) return;
    if (env_flag_enabled("MOLLM_W4_NO_Q8_DOT") ||
        env_flag_enabled("MOLLM_W4_NO_Q4_REPACK") ||
        env_flag_enabled("MOLLM_W4_NO_Q8_REPACK")) {
        return;
    }

    int N = (int)t.shape[0];
    int K = (int)t.shape[1];
    const uint8_t* b_orig = reinterpret_cast<const uint8_t*>(weight_data);

    std::string q4_key = key + "#int4_q4dot";
    auto it = packed_weights.find(q4_key);
    if (it == packed_weights.end()) {
        int K_blocks = (K + 31) / 32;
        uint8_t* b_q4 = pack_b_q4dot_int4_full(b_orig, N, K, K);
        size_t buf_size = (size_t)((N + 7) / 8) * 8 * K_blocks * 16;
        std::vector<uint8_t> buf(b_q4, b_q4 + buf_size);
        delete[] b_q4;
        it = packed_weights.emplace(q4_key, std::move(buf)).first;
    }
    t.q4_repack_data = it->second.data();
    maybe_pack_int4_g128_weight(t, key, t.q4_repack_data, packed_weights);
#else
    (void)t;
    (void)key;
    (void)weight_data;
    (void)packed_weights;
#endif
}

struct SamplerCandidate {
    int id = 0;
    float logit = 0.0f;
    float prob = 0.0f;
};

struct MinLogitHeapCompare {
    bool operator()(const SamplerCandidate& a, const SamplerCandidate& b) const {
        return a.logit > b.logit;
    }
};

struct SamplerScratch {
    std::vector<SamplerCandidate> candidates;
};

int argmax_token(const float* logits, int vocab_size) {
#if HAS_NEON
    if (vocab_size >= 4) {
        static const int32_t kLaneOffsetsData[4] = {0, 1, 2, 3};
        int32x4_t lane_offsets = vld1q_s32(kLaneOffsetsData);
        float32x4_t best_vals = vld1q_f32(logits);
        int32x4_t best_idxs = lane_offsets;

        int i = 4;
        for (; i + 4 <= vocab_size; i += 4) {
            float32x4_t vals = vld1q_f32(logits + i);
            int32x4_t idxs = vaddq_s32(vdupq_n_s32(i), lane_offsets);
            uint32x4_t mask = vcgtq_f32(vals, best_vals);
            best_vals = vbslq_f32(mask, vals, best_vals);
            best_idxs = vbslq_s32(mask, idxs, best_idxs);
        }

        float lane_vals[4];
        int32_t lane_idxs[4];
        vst1q_f32(lane_vals, best_vals);
        vst1q_s32(lane_idxs, best_idxs);

        int best = lane_idxs[0];
        float best_val = lane_vals[0];
        for (int lane = 1; lane < 4; lane++) {
            if (lane_vals[lane] > best_val ||
                (lane_vals[lane] == best_val && lane_idxs[lane] < best)) {
                best = lane_idxs[lane];
                best_val = lane_vals[lane];
            }
        }
        for (; i < vocab_size; i++) {
            if (logits[i] > best_val) {
                best = i;
                best_val = logits[i];
            }
        }
        return best;
    }
#endif
    int best = 0;
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

SamplerScratch& sampler_scratch() {
    static thread_local SamplerScratch scratch;
    return scratch;
}

/// Full sample pipeline: logits -> top-k -> softmax -> top-p -> sample.
int sample_token(float* logits, int vocab_size,
                  float temperature, int top_k, float top_p,
                  unsigned int* seed) {
    if (vocab_size <= 0) return 0;
    if (temperature <= 0.0f || top_k == 1) return argmax_token(logits, vocab_size);

    int k = top_k > 0 ? std::min(top_k, vocab_size) : vocab_size;
    auto& candidates = sampler_scratch().candidates;
    candidates.clear();
    candidates.reserve((size_t)k);

    MinLogitHeapCompare heap_compare;
    for (int i = 0; i < vocab_size; i++) {
        SamplerCandidate cand{i, logits[i], 0.0f};
        if ((int)candidates.size() < k) {
            candidates.push_back(cand);
            std::push_heap(candidates.begin(), candidates.end(), heap_compare);
        } else if (cand.logit > candidates.front().logit) {
            std::pop_heap(candidates.begin(), candidates.end(), heap_compare);
            candidates.back() = cand;
            std::push_heap(candidates.begin(), candidates.end(), heap_compare);
        }
    }
    if (candidates.empty()) return 0;

    std::sort(candidates.begin(), candidates.end(),
              [](const SamplerCandidate& a, const SamplerCandidate& b) {
                  return a.logit > b.logit;
              });

    float max_logit = candidates[0].logit;
    float inv_t = 1.0f / temperature;
    float sum = 0.0f;
    for (auto& cand : candidates) {
        cand.prob = std::exp((cand.logit - max_logit) * inv_t);
        sum += cand.prob;
    }
    if (!(sum > 0.0f) || !std::isfinite(sum)) {
        return candidates[0].id;
    }

    int active = (int)candidates.size();
    if (top_p > 0.0f && top_p < 1.0f) {
        float cutoff_mass = top_p * sum;
        float cumulative = 0.0f;
        for (int i = 0; i < (int)candidates.size(); i++) {
            cumulative += candidates[i].prob;
            if (cumulative >= cutoff_mass) {
                active = i + 1;
                break;
            }
        }
        sum = cumulative;
    }

    float r = (float)rand_r(seed) / (float)RAND_MAX;
    float target = r * sum;
    float cumulative = 0.0f;
    for (int i = 0; i < active; i++) {
        cumulative += candidates[i].prob;
        if (target <= cumulative) return candidates[i].id;
    }
    return candidates[active - 1].id;
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

void release_graph_temporaries(Graph& graph, Backend* backend) {
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
        if (t.data && t.mem_type == MemoryType::POOLED && t.nbytes() > 0) {
            if (backend) backend->free_output(t, &pool);
            else release_pool_tensor(pool, t);
        }
        t.data = nullptr;
        t.device_data = nullptr;
        t.device_offset = 0;
        t.mem_type = MemoryType::NONE;
        t.owner_id = 0;
        t.storage_id = 0;
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
        release_graph_temporaries(graph, ctx.backend);
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
        package_weights_base_ = nullptr;
        package_weights_size_ = 0;
    }
    package_weights_storage_.clear();
    package_weights_resident_ = false;

    // Remove extracted temp files from /tmp.
    for (const auto& path : temp_files_) {
        std::remove(path.c_str());
    }
}

size_t LLMEngine::warmup_package_weights() {
    if (!package_weights_base_ || package_weights_size_ == 0) return 0;
    if (package_weights_resident_) return 0;

    long page_size_long = sysconf(_SC_PAGESIZE);
    size_t page_size = page_size_long > 0 ? (size_t)page_size_long : 4096;

#if defined(MADV_WILLNEED)
    uintptr_t start = reinterpret_cast<uintptr_t>(package_weights_base_);
    uintptr_t aligned_start = (start / page_size) * page_size;
    size_t prefix = (size_t)(start - aligned_start);
    madvise(reinterpret_cast<void*>(aligned_start),
            prefix + package_weights_size_,
            MADV_WILLNEED);
#endif

    const uint8_t* p = package_weights_base_;
    const size_t len = package_weights_size_;

    uint8_t sink = 0;
    for (size_t off = 0; off < len; off += page_size) {
        sink ^= p[off];
    }
    sink ^= p[len - 1];
    g_package_warmup_sink ^= sink;
    return len;
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
                                        size_t scales_size, size_t data_size, uint32_t flags,
                                        const char* label) -> bool {
            t.scales = nullptr;
            t.group_size = 0;
            t.num_groups = 0;
            t.groups_per_row = 0;
            t.is_q4_repacked = false;
            t.is_q4_g128_packed = false;

            bool is_quantized = (t.prec == Precision::INT8 || t.prec == Precision::INT4);
            if (!is_quantized) return true;
            int64_t N = t.shape[0];
            int64_t K = t.shape[1];
            if (!scales || group_size == 0 || N <= 0 || K <= 0) {
                fprintf(stderr, "Engine: quantized weight %s missing scales/group metadata\n", label);
                return false;
            }
            uint32_t groups_per_row = (uint32_t)((K + group_size - 1) / group_size);
            uint64_t expected_groups = (uint64_t)N * groups_per_row;
            bool int4_q4dot_layout =
                t.prec == Precision::INT4 && (flags & MappedFile::FLAG_INT4_Q4DOT);
            bool int4_bg128_layout =
                t.prec == Precision::INT4 && (flags & MappedFile::FLAG_INT4_BG128);
            constexpr uint32_t supported_flags =
                MappedFile::FLAG_INT4_Q4DOT | MappedFile::FLAG_INT4_BG128;
            if (flags & ~supported_flags) {
                fprintf(stderr, "Engine: quantized weight %s has unsupported flags 0x%x\n",
                        label, flags);
                return false;
            }
            if ((flags & supported_flags) && t.prec != Precision::INT4) {
                fprintf(stderr, "Engine: weight %s has INT4 layout flag but precision is not INT4\n",
                        label);
                return false;
            }
            if (int4_q4dot_layout && int4_bg128_layout) {
                fprintf(stderr, "Engine: weight %s has mutually exclusive INT4 layout flags\n",
                        label);
                return false;
            }
            if (int4_q4dot_layout && (K % 32 != 0 || group_size % 32 != 0)) {
                fprintf(stderr,
                        "Engine: INT4 q4dot weight %s requires K/group multiple of 32 (K=%lld group=%u)\n",
                        label, (long long)K, group_size);
                return false;
            }
            if (int4_bg128_layout && (K % 128 != 0 || group_size != 128)) {
                fprintf(stderr,
                        "Engine: INT4 BG128 weight %s requires K multiple of 128 and group=128 (K=%lld group=%u)\n",
                        label, (long long)K, group_size);
                return false;
            }
            if ((int4_q4dot_layout || int4_bg128_layout) && !int4_q4dot_kernel_available()) {
                fprintf(stderr,
                        "Engine: INT4 packed weight %s requires an ARM DOTPROD build\n",
                        label);
                return false;
            }
            uint64_t expected_data_size = (uint64_t)N * (uint64_t)K;
            if (t.prec == Precision::INT4) {
                if (int4_q4dot_layout) {
                    uint64_t n_padded = (uint64_t)(((N + 7) / 8) * 8);
                    uint64_t k_blocks = (uint64_t)((K + 31) / 32);
                    expected_data_size = n_padded * k_blocks * 16;
                } else if (int4_bg128_layout) {
                    expected_data_size =
                        (uint64_t)pack_b_q4dot_g128_bytes((int)N, (int)K);
                } else {
                    expected_data_size = (uint64_t)N * (uint64_t)((K + 1) / 2);
                }
            }
            if (num_groups != expected_groups ||
                scales_size != expected_groups * sizeof(float) ||
                data_size != expected_data_size) {
                fprintf(stderr,
                        "Engine: quantized weight %s bad metadata (N=%lld K=%lld group=%u groups=%u expected=%llu scales=%zu data=%zu expected_data=%llu)\n",
                        label, (long long)N, (long long)K, group_size, num_groups,
                        (unsigned long long)expected_groups, scales_size, data_size,
                        (unsigned long long)expected_data_size);
                return false;
            }
            t.scales = static_cast<const float*>(scales);
            t.group_size = group_size;
            t.num_groups = num_groups;
            t.groups_per_row = groups_per_row;
            t.is_q4_repacked = int4_q4dot_layout;
            t.is_q4_g128_packed = int4_bg128_layout;
            if (int4_bg128_layout) {
                t.q4_g128_data = t.data;
            }
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
            if (metal_backend_) as_metal(metal_backend_)->wrap_weight(t);
#endif
        };

#ifdef MOLLM_METAL
        // Second pass for INT4 g128 weights: once quant metadata + q4_g128_data
        // are populated, decode the CPU Q4B8G128Block layout into a Metal raw
        // nibble+scale device buffer. No-op for non-INT4 weights.
        auto finalize_metal_weight = [&]() {
            if (metal_backend_) {
                bool is_aggregate_expert = wref.find("_experts_") != std::string::npos;
                as_metal(metal_backend_)->wrap_weight_int4_g128(t, is_aggregate_expert);
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
            if (pit == package_weight_map_.end()) pit = package_weight_map_.find(wpath);
            if (pit != package_weight_map_.end()) {
                const uint8_t* hdr = package_weights_base_ + pit->second.first;
                // Read data_offset from weight header (byte 48, 8 bytes)
                uint32_t flags = 0;
                uint32_t file_prec_u32 = 0;
                uint64_t data_off;
                std::memcpy(&flags, hdr + 4, sizeof(flags));
                std::memcpy(&file_prec_u32, hdr + 12, sizeof(file_prec_u32));
                std::memcpy(&data_off, hdr + 48, sizeof(data_off));
                uint64_t scales_off = 0, scales_size = 0;
                uint32_t group_size = 0, num_groups = 0;
                std::memcpy(&scales_off, hdr + 64, sizeof(scales_off));
                std::memcpy(&scales_size, hdr + 72, sizeof(scales_size));
                std::memcpy(&group_size, hdr + 80, sizeof(group_size));
                std::memcpy(&num_groups, hdr + 84, sizeof(num_groups));
                void* data = const_cast<uint8_t*>(hdr + data_off);
                uint64_t data_size = 0;
                std::memcpy(&data_size, hdr + 56, sizeof(data_size));
                setup_weight(data, static_cast<Precision>(file_prec_u32));
                const void* scales = scales_size ? (hdr + scales_off) : nullptr;
                if (!setup_quant_metadata(scales, group_size, num_groups,
                                          (size_t)scales_size, (size_t)data_size, flags,
                                          wref.c_str())) {
                    return false;
                }

                if (wref.find("embed_tokens") == std::string::npos) {
                    maybe_pack_fp16_weight(t, wref, data, packed_weights_);
                }
                maybe_pack_int8_weight(t, wref, data, packed_weights_);
                maybe_pack_int4_weight(t, wref, data, packed_weights_);
                finalize_metal_weight();
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
                                      mapped.scales_size(), mapped.data_size(),
                                      mapped.header().flags, wpath.c_str())) {
                return false;
            }
        }

        if (it != weight_map_.end()) {
            const auto& mapped = shared_weights_[it->second];
            if (!setup_quant_metadata(mapped.scales(), mapped.group_size(), mapped.num_groups(),
                                      mapped.scales_size(), mapped.data_size(),
                                      mapped.header().flags, wpath.c_str())) {
                return false;
            }
        }

        // embed_tokens stays row-major for lookup; all other linear weights,
        // including lm_head, are regular load-time packed weights.
        if (node.params.str[0].find("embed_tokens") == std::string::npos) {
            maybe_pack_fp16_weight(t, wpath, t.data, packed_weights_);
        }
        maybe_pack_int8_weight(t, wpath, t.data, packed_weights_);
        maybe_pack_int4_weight(t, wpath, t.data, packed_weights_);
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
        } else if (name.find("rwkv_state") == 0) {
            int layer_idx = std::stoi(name.substr(10));
            if (layer_idx >= (int)caches_.size()) caches_.resize(layer_idx + 1);
            Tensor& t = g.runtime.tensors[node.id];
            t.prec = node.out_prec;
            for (int d=0; d<4; ++d) t.shape[d]=node.out_shape[d];
            t.compute_strides();
            caches_[layer_idx].rwkv_state=&t;
            caches_[layer_idx].rwkv_head_size=(int)node.out_shape[0];
            caches_[layer_idx].rwkv_num_heads=(int)node.out_shape[2];
        } else if (name.find("rwkv_att_shift") == 0 || name.find("rwkv_ffn_shift") == 0) {
            bool att = name.find("rwkv_att_shift") == 0;
            int layer_idx = std::stoi(name.substr(att ? 14 : 14));
            if (layer_idx >= (int)caches_.size()) caches_.resize(layer_idx + 1);
            Tensor& t = g.runtime.tensors[node.id];
            t.prec = node.out_prec;
            for (int d=0; d<4; ++d) t.shape[d]=node.out_shape[d];
            t.compute_strides();
            if(att) caches_[layer_idx].rwkv_att_shift=&t;
            else caches_[layer_idx].rwkv_ffn_shift=&t;
            caches_[layer_idx].rwkv_hidden_size=(int)node.out_shape[0];
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
            return t->data;  // = [buffer contents], host-visible
        }
#endif
        void* buf = persistent_pool_.acquire(total);
        t->data     = buf;
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
            meta->max_seq_len     = (uint64_t)n_ctx;
            meta->num_kv_heads    = (uint64_t)nkv;
            meta->head_dim        = (uint64_t)hd;

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
            meta->max_seq_len     = (uint64_t)n_ctx;
            meta->num_kv_heads    = (uint64_t)nkv;
            meta->v_head_dim      = (uint64_t)vd;

            cp.v->mem_type = MemoryType::POOLED;
            cp.v->shape[0] = (int64_t)total / (int64_t)es;
            cp.v->compute_strides();
        }

        // GDN recurrent state: [v_dim, k_dim, num_heads] FP32
        // No CacheMetadata header — GDN state is a plain FP32 buffer.
        if (cp.gdn_state) {
            size_t data_bytes = (size_t)cp.gdn_v_dim * cp.gdn_k_dim * cp.gdn_num_heads * sizeof(float);
            void* buf = alloc_cache_buf(cp.gdn_state, data_bytes);  // Metal: sets device_data
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
            size_t data_bytes = (size_t)cp.gdn_conv_groups * kernel_m1 * sizeof(float);
            void* buf = alloc_cache_buf(cp.gdn_conv, data_bytes);  // Metal: sets device_data
            std::memset(buf, 0, data_bytes);
            cp.gdn_conv->mem_type = MemoryType::POOLED;
            cp.gdn_conv->shape[0] = (int64_t)cp.gdn_conv_groups;
            cp.gdn_conv->shape[1] = (int64_t)kernel_m1;
            cp.gdn_conv->shape[2] = 1;
            cp.gdn_conv->shape[3] = 1;
            cp.gdn_conv->compute_strides();
        }
        if (cp.rwkv_state) {
            size_t bytes=cp.rwkv_state->nbytes();
            void* buf=alloc_cache_buf(cp.rwkv_state, bytes); std::memset(buf,0,bytes);
            cp.rwkv_state->mem_type=MemoryType::POOLED;
        }
        for (Tensor* shift : {cp.rwkv_att_shift, cp.rwkv_ffn_shift}) if (shift) {
            size_t bytes=shift->nbytes();
            void* buf=alloc_cache_buf(shift, bytes); std::memset(buf,0,bytes);
            shift->mem_type=MemoryType::POOLED;
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
    embed_weight_ = nullptr;
    lm_head_weight_ = nullptr;
    persistent_pool_.clear();
    thread_pool_.resize(cfg_.num_threads);
    exec_ctx_prefill_.thread_pool = &thread_pool_;
    exec_ctx_decode_.thread_pool  = &thread_pool_;
    exec_ctx_prefill_.backend     = &cpu_backend_;
    exec_ctx_decode_.backend      = &cpu_backend_;
    metal_backend_.reset();
    if (cfg_.device == Device::METAL) {
#ifdef MOLLM_METAL
        metal_backend_.reset(new MetalBackend());
        if (!as_metal(metal_backend_)->available()) {
            fprintf(stderr, "Engine: Metal backend unavailable; falling back to CPU\n");
            metal_backend_.reset();
            cfg_.device = Device::CPU;
        } else {
            exec_ctx_prefill_.backend = metal_backend_.get();
            exec_ctx_decode_.backend  = metal_backend_.get();
            // The Metal backend wraps the weight region via newBufferWithBytesNoCopy.
            // mmap'd file-backed pages are NOT reliably GPU-accessible that way
            // (the GPU reads zeros), so force RESIDENT weights when Metal is active.
            if (cfg_.weight_loading == WeightLoadingMode::MMAP) {
                fprintf(stderr, "Engine: Metal backend requires resident weights; "
                                "ignoring --mmap\n");
                cfg_.weight_loading = WeightLoadingMode::RESIDENT;
            }
        }
#else
        fprintf(stderr, "Engine: built without MOLLM_METAL; using CPU backend\n");
        cfg_.device = Device::CPU;
#endif
    }
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

    // RWKV has recurrent state rather than an attention KV cache. Its generic
    // batched graph cannot yet preserve the FP16 decode trajectory, so use the
    // verified seq=1 graph repeatedly until a numerically compatible batched
    // recurrence lands. prefill() handles the chunking transparently.
    auto package_arch = package_metadata_.find("architecture");
    if (package_arch != package_metadata_.end() && package_arch->second == "rwkv7") {
        cfg_.use_decode_as_prefill = true;
    }

#ifdef MOLLM_METAL
    // Wrap the whole package weight region as one zero-copy MTLBuffer so each
    // weight tensor can alias it via device_offset (set in setup_weight).
    if (metal_backend_ && package_weights_base_ && package_weights_size_) {
        if (!as_metal(metal_backend_)->register_weight_region(
                const_cast<uint8_t*>(package_weights_base_), package_weights_size_)) {
            fprintf(stderr, "Engine: failed to register weight region with Metal\n");
        }
    }
#endif

    // Load prefill graph first (establishes shared weights)
    const std::string& pf_path_load = cfg_.use_decode_as_prefill ? dc_path : pf_path;
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

    if (!embed_weight_ || !embed_weight_->data) {
        fprintf(stderr, "Engine: package missing explicit embed_tokens weight\n");
        return false;
    }
    if (!lm_head_weight_ || !lm_head_weight_->data) {
        fprintf(stderr, "Engine: package missing explicit lm_head weight; reconvert with current converter\n");
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
        } else if (name.find("rwkv_state") == 0) {
            int i=std::stoi(name.substr(10)); if(i<(int)caches_.size()&&caches_[i].rwkv_state) graph_decode_.runtime.tensors[node.id]=*caches_[i].rwkv_state;
        } else if (name.find("rwkv_att_shift") == 0) {
            int i=std::stoi(name.substr(14)); if(i<(int)caches_.size()&&caches_[i].rwkv_att_shift) graph_decode_.runtime.tensors[node.id]=*caches_[i].rwkv_att_shift;
        } else if (name.find("rwkv_ffn_shift") == 0) {
            int i=std::stoi(name.substr(14)); if(i<(int)caches_.size()&&caches_[i].rwkv_ffn_shift) graph_decode_.runtime.tensors[node.id]=*caches_[i].rwkv_ffn_shift;
        }
    }

    reset();
    return true;
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

void LLMEngine::reset() {
    release_graph_temporaries(graph_decode_, exec_ctx_decode_.backend);
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
        if(cp.rwkv_state) std::memset(cp.rwkv_state->data,0,cp.rwkv_state->nbytes());
        if(cp.rwkv_att_shift) std::memset(cp.rwkv_att_shift->data,0,cp.rwkv_att_shift->nbytes());
        if(cp.rwkv_ffn_shift) std::memset(cp.rwkv_ffn_shift->data,0,cp.rwkv_ffn_shift->nbytes());
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
    if (!lm_head_weight_ || !lm_head_weight_->data) return 0;

    int vocab_size = (int)lm_head_weight_->shape[0];
    int hidden_dim = (int)lm_head_weight_->shape[1];
    int seq_len = (int)hidden.shape[1];

    // Read the last real token, not necessarily the last position
    // (which may be padded when graph seq_len > n_tokens).
    int last_pos = (n_tokens > 0 && n_tokens <= seq_len) ? n_tokens - 1 : seq_len - 1;

    // hidden is [hidden_dim, seq_len], we want [hidden_dim, 1] for matmul
    // lm_head_weight is [vocab_size, hidden_dim] — we use it as weight B
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

#ifdef MOLLM_METAL
    if (metal_backend_ && lm_head_weight_->prec == Precision::FP16 &&
        lm_head_weight_->device_data) {
        as_metal(metal_backend_)->lm_head_gemv(
            A.ptr<float>(), *lm_head_weight_, C.ptr<float>(), vocab_size, hidden_dim);
    } else
#endif
    {
        kernel_matmul_fp32(A, *lm_head_weight_, C, exec_ctx_decode_.thread_pool);
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
    if (!lm_head_weight_ || !lm_head_weight_->data) return {};

    int vocab_size = (int)lm_head_weight_->shape[0];
    int hidden_dim = (int)lm_head_weight_->shape[1];
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

#ifdef MOLLM_METAL
        if (metal_backend_ && lm_head_weight_->prec == Precision::FP16 &&
            lm_head_weight_->device_data) {
            as_metal(metal_backend_)->lm_head_gemv(
                A.ptr<float>(), *lm_head_weight_, C.ptr<float>(), vocab_size, hidden_dim);
        } else
#endif
        {
            kernel_matmul_fp32(A, *lm_head_weight_, C, exec_ctx_decode_.thread_pool);
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

        bool is_boundary = false;
        if (name == "hidden") {
            *t = hidden; is_boundary = true;
        } else if (name == "mask") {
            *t = mask; is_boundary = true;
        } else if (name == "cos") {
            *t = cos; is_boundary = true;
        } else if (name == "sin") {
            *t = sin; is_boundary = true;
        }
        // cache_k/cache_v/gdn state are persistent INPUT tensors.

#ifdef MOLLM_METAL
        // Boundary inputs are produced on the host (embed/rope/mask); upload
        // their bytes into a device buffer so GPU kernels can read them.
        // Cache/state INPUTs are already device-resident (allocate_caches).
        if (metal_backend_ && is_boundary && t->data) {
            as_metal(metal_backend_)->upload_input(*t, name, t->data, t->nbytes());
        }
#else
        (void)is_boundary;
#endif
    }

#ifdef MOLLM_METAL
    if (metal_backend_) metal_backend_->begin_graph();
#endif
    execute_graph(exec_ctx);
#ifdef MOLLM_METAL
    if (metal_backend_) metal_backend_->end_graph();
#endif

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
    release_graph_temporaries(graph_prefill_, exec_ctx_prefill_.backend);
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

    const auto arch = package_metadata_.find("architecture");
    const bool rwkv_batched = !cfg_.use_decode_as_prefill &&
        arch != package_metadata_.end() && arch->second == "rwkv7";
    const bool previous_fp32_acc = g_mollm_force_fp32_acc;
    if (rwkv_batched) g_mollm_force_fp32_acc = true;
    mollm_set_matmul_profile_phase("prefill_graph");
    Tensor out = run_graph(graph_prefill_, exec_ctx_prefill_, h, mask, cos, sin);
    mollm_set_matmul_profile_phase("unscoped");
    g_mollm_force_fp32_acc = previous_fp32_acc;
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

namespace {

struct PackageHeaderInfo {
    uint64_t meta_off = 0, meta_len = 0;
    uint64_t tok_off = 0, tok_len = 0;
    uint64_t jin_off = 0, jin_len = 0;
    uint64_t pf_off = 0, pf_len = 0;
    uint64_t dc_off = 0, dc_len = 0;
    uint64_t w_off = 0, w_len = 0;
};

bool read_exact_at(int fd, uint64_t offset, void* dst, size_t len, const char* label) {
    uint8_t* out = static_cast<uint8_t*>(dst);
    size_t done = 0;
    while (done < len) {
        size_t chunk = std::min<size_t>(len - done, 64 * 1024 * 1024);
        ssize_t n = pread(fd, out + done, chunk, static_cast<off_t>(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "Engine: failed to read %s: %s\n", label, strerror(errno));
            return false;
        }
        if (n == 0) {
            fprintf(stderr, "Engine: short read while reading %s\n", label);
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

bool section_in_file(uint64_t off, uint64_t len, size_t file_size, const char* label) {
    uint64_t size = static_cast<uint64_t>(file_size);
    if (off > size || len > size - off) {
        fprintf(stderr, "Engine: package %s section extends beyond file\n", label);
        return false;
    }
    if (len > static_cast<uint64_t>(SIZE_MAX)) {
        fprintf(stderr, "Engine: package %s section is too large for this platform\n", label);
        return false;
    }
    return true;
}

bool parse_package_header(const uint8_t* header, size_t file_size,
                          PackageHeaderInfo& out) {
    if (file_size < 128) {
        fprintf(stderr, "Engine: package too small\n");
        return false;
    }

    uint32_t magic = 0, version = 0;
    std::memcpy(&magic, header + 0, 4);
    std::memcpy(&version, header + 4, 4);
    (void)version;
    if (magic != PACKAGE_MAGIC_C) {
        fprintf(stderr, "Engine: bad package magic 0x%08x\n", magic);
        return false;
    }

    std::memcpy(&out.meta_off, header + 8, 8);
    std::memcpy(&out.meta_len, header + 16, 8);
    std::memcpy(&out.tok_off, header + 24, 8);
    std::memcpy(&out.tok_len, header + 32, 8);
    std::memcpy(&out.jin_off, header + 40, 8);
    std::memcpy(&out.jin_len, header + 48, 8);
    std::memcpy(&out.pf_off, header + 56, 8);
    std::memcpy(&out.pf_len, header + 64, 8);
    std::memcpy(&out.dc_off, header + 72, 8);
    std::memcpy(&out.dc_len, header + 80, 8);
    std::memcpy(&out.w_off, header + 88, 8);
    std::memcpy(&out.w_len, header + 96, 8);

    return section_in_file(out.meta_off, out.meta_len, file_size, "metadata") &&
           section_in_file(out.tok_off, out.tok_len, file_size, "tokenizer") &&
           section_in_file(out.jin_off, out.jin_len, file_size, "chat template") &&
           section_in_file(out.pf_off, out.pf_len, file_size, "prefill graph") &&
           section_in_file(out.dc_off, out.dc_len, file_size, "decode graph") &&
           section_in_file(out.w_off, out.w_len, file_size, "weights");
}

} // namespace

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

    uint8_t header[128];
    if (!read_exact_at(fd, 0, header, sizeof(header), "package header")) {
        close(fd);
        return false;
    }

    PackageHeaderInfo ph;
    if (!parse_package_header(header, file_size, ph)) {
        close(fd);
        return false;
    }

    auto parse_metadata = [&](const std::string& meta_str) -> bool {
        try {
            auto meta = json::parse(meta_str);
            if (meta.contains("prefill_seq_len")) {
                package_prefill_seq_len_ = meta["prefill_seq_len"].get<int>();
            }
            if (meta.contains("weights")) {
                for (auto& [name, info] : meta["weights"].items()) {
                    if (!info.is_array() || info.size() < 2) {
                        fprintf(stderr, "Engine: bad package metadata for weight %s\n",
                                name.c_str());
                        return false;
                    }
                    uint64_t off = info[0].get<uint64_t>();
                    uint64_t sz = info[1].get<uint64_t>();
                    if (off > ph.w_len || sz > ph.w_len - off) {
                        fprintf(stderr, "Engine: package weight %s extends beyond weights section\n",
                                name.c_str());
                        return false;
                    }
                    package_weight_map_[name] = {off, sz};
                }
            }
            // Retain all top-level scalar fields for CLI banner / display.
            // Skip "weights" (object) and any non-scalar. ints/bools are stringified.
            for (auto it = meta.begin(); it != meta.end(); ++it) {
                const std::string& key = it.key();
                if (key == "weights") continue;
                if (it->is_string()) {
                    package_metadata_[key] = it->get<std::string>();
                } else if (it->is_number_integer()) {
                    package_metadata_[key] = std::to_string(it->get<int64_t>());
                } else if (it->is_number_unsigned()) {
                    package_metadata_[key] = std::to_string(it->get<uint64_t>());
                } else if (it->is_boolean()) {
                    package_metadata_[key] = it->get<bool>() ? "true" : "false";
                }
                // arrays / objects / layer_types list are skipped (not needed for banner)
            }
        } catch (std::exception& e) {
            fprintf(stderr, "Engine: failed to parse package metadata: %s\n", e.what());
            return false;
        }
        return true;
    };

    // Extract graphs + tokenizer + jinja to temp files
    pid_t pid = getpid();
    auto write_tmp_from_memory = [&](const uint8_t* base, const char* label,
                                     uint64_t off, uint64_t len,
                                     std::string& out_path) -> bool {
        if (len == 0) return true;
        out_path = "/tmp/mollm_pkg_" + std::to_string(pid) + "_" + label;
        std::ofstream f(out_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(base + off), static_cast<std::streamsize>(len));
        if (!f) {
            fprintf(stderr, "Engine: failed to write extracted package section %s\n", label);
            return false;
        }
        temp_files_.push_back(out_path);
        return true;
    };
    auto write_tmp_from_fd = [&](const char* label, uint64_t off, uint64_t len,
                                 std::string& out_path) -> bool {
        if (len == 0) return true;
        out_path = "/tmp/mollm_pkg_" + std::to_string(pid) + "_" + label;
        std::ofstream f(out_path, std::ios::binary);
        if (!f) {
            fprintf(stderr, "Engine: failed to create extracted package section %s\n", label);
            return false;
        }
        std::vector<uint8_t> buf(1024 * 1024);
        uint64_t done = 0;
        while (done < len) {
            size_t chunk = static_cast<size_t>(
                std::min<uint64_t>(buf.size(), len - done));
            if (!read_exact_at(fd, off + done, buf.data(), chunk, label)) {
                return false;
            }
            f.write(reinterpret_cast<const char*>(buf.data()), chunk);
            if (!f) {
                fprintf(stderr, "Engine: failed to write extracted package section %s\n", label);
                return false;
            }
            done += chunk;
        }
        temp_files_.push_back(out_path);
        return true;
    };

    if (cfg_.weight_loading == WeightLoadingMode::RESIDENT) {
        std::string meta_str(static_cast<size_t>(ph.meta_len), '\0');
        if (!read_exact_at(fd, ph.meta_off, meta_str.data(), meta_str.size(),
                           "package metadata")) {
            close(fd);
            return false;
        }
        if (!parse_metadata(meta_str)) {
            close(fd);
            return false;
        }

        try {
            package_weights_storage_.resize(static_cast<size_t>(ph.w_len));
        } catch (const std::bad_alloc&) {
            fprintf(stderr,
                    "Engine: failed to allocate %.1f MB for resident package weights; try --mmap\n",
                    ph.w_len / 1e6);
            close(fd);
            return false;
        }
        if (ph.w_len > 0 &&
            !read_exact_at(fd, ph.w_off, package_weights_storage_.data(),
                           package_weights_storage_.size(), "package weights")) {
            close(fd);
            return false;
        }
        package_weights_base_ = package_weights_storage_.empty()
                                    ? nullptr
                                    : package_weights_storage_.data();
        package_weights_size_ = package_weights_storage_.size();
        package_weights_resident_ = true;

        bool ok = write_tmp_from_fd("prefill.graph", ph.pf_off, ph.pf_len, pf_path) &&
                  write_tmp_from_fd("decode.graph", ph.dc_off, ph.dc_len, dc_path) &&
                  write_tmp_from_fd("tokenizer.json", ph.tok_off, ph.tok_len, tok_path) &&
                  write_tmp_from_fd("chat_template.jinja", ph.jin_off, ph.jin_len, jin_path);
        close(fd);
        if (!ok) return false;

        fprintf(stderr,
                "Engine: loaded package %s (%.1f MB, %zu weights, prefill_seq=%d, weights=resident)\n",
                path.c_str(), file_size / 1e6, package_weight_map_.size(),
                package_prefill_seq_len_);
        return true;
    }

    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "Engine: mmap failed for %s\n", path.c_str());
        return false;
    }
    package_mmap_ = mapped;
    package_mmap_size_ = file_size;

    const uint8_t* base = static_cast<const uint8_t*>(mapped);
    package_weights_base_ = base + ph.w_off;
    package_weights_size_ = static_cast<size_t>(ph.w_len);
    package_weights_resident_ = false;

    std::string meta_str(reinterpret_cast<const char*>(base + ph.meta_off),
                         static_cast<size_t>(ph.meta_len));
    if (!parse_metadata(meta_str)) {
        return false;
    }

    if (!write_tmp_from_memory(base, "prefill.graph", ph.pf_off, ph.pf_len, pf_path) ||
        !write_tmp_from_memory(base, "decode.graph", ph.dc_off, ph.dc_len, dc_path) ||
        !write_tmp_from_memory(base, "tokenizer.json", ph.tok_off, ph.tok_len, tok_path) ||
        !write_tmp_from_memory(base, "chat_template.jinja", ph.jin_off, ph.jin_len, jin_path)) {
        return false;
    }

    fprintf(stderr, "Engine: loaded package %s (%.1f MB, %zu weights, prefill_seq=%d, weights=mmap)\n",
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
