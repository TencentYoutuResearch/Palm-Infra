#pragma once

#include "kernels/tensor.h"
#include "graph/buffer_pool.h"
#include "graph/mmap_file.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// mollm — Graph definition
//
// A Graph is a linear (topologically-sorted) list of nodes.  Execution walks
// the list once, dispatching each node to its kernel.  All graph construction
// and optimisation happens in Python; the C++ side only loads and executes.
//
// Dynamic shape support
// ---------------------
// Each GraphNode carries a `dim_expr[4]` array of DimExpr structs, one per
// output dimension.  When a dim depends on runtime seq_len, its size is
// expressed as a symbolic expression (SEQ, N*SEQ, N+SEQ, SEQ/N) evaluated at
// runtime.
// STATIC dims use out_shape[i] verbatim.  The transpiler does full symbolic
// shape propagation (ONNX style): expr information flows from INPUT nodes
// through every op, so every tensor's dim_expr[] is baked into the graph.
// Runtime does NO shape inference — it just reads dim_expr[] and evaluates.
//
// BATCH is reserved for future batch dim support; not used currently.
// ---------------------------------------------------------------------------

// Per-dimension symbolic expression.
//   CONST: value = out_shape[i] (no runtime dependence)
//   SEQ:   value = runtime_seq_len
//   MUL:   value = coeff * runtime_seq_len    (covers N * SEQ)
//   ADD:   value = coeff + runtime_seq_len    (covers N + SEQ, rare)
//   BATCH: value = runtime_batch_size          (reserved)
//   DIV:   value = runtime_seq_len / coeff     (exact integer division)
//
// Serialized as 8 bytes (kind + coeff + padding). The coeff field is only
// used by MUL, ADD, and DIV; CONST/SEQ/BATCH ignore it.
struct DimExpr {
    int8_t  kind  = 0;   // 0=CONST, 1=SEQ, 2=MUL, 3=ADD, 4=BATCH, 5=DIV
    int32_t coeff = 0;   // multiplier/addend/divisor

    bool is_static() const { return kind == 0; }
    bool is_dynamic() const { return kind != 0; }
};

// DimExpr kinds (matching the int8_t values above)
enum : int8_t {
    DIM_CONST = 0,
    DIM_SEQ   = 1,
    DIM_MUL   = 2,
    DIM_ADD   = 3,
    DIM_BATCH = 4,
    DIM_DIV   = 5,
};

// ---- op types ----
enum class OpType : uint32_t {
    // meta
    INPUT  = 0,
    CONSTANT = 1,

    // linear
    MATMUL  = 10,
    MATMUL_BATCH = 11,

    // normalisation
    RMS_NORM   = 20,
    LAYER_NORM = 21,
    ADD_RMS_NORM = 22,  // residual += update; out = rms_norm(residual)
    RMS_NORM_ROPE = 23, // Q/K RMSNorm + materialize + RoPE
    QK_RMS_NORM_ROPE = 24, // fused Q+K RMSNorm/materialize/RoPE
    GROUP_RMS_NORM = 25, // RMSNorm independent consecutive feature groups

    // activations
    SILU = 30,
    GELU = 31,
    TANH = 32,

    // position encoding
    ROTARY_EMBED = 40,

    // attention
    SDPA     = 50,
    SDPA_MLA = 51,

    // shape (zero-copy views)
    RESHAPE  = 60,
    PERMUTE  = 61,
    CONCAT   = 62,
    SLICE    = 63,
    TILE     = 64,
    CONTIGUOUS = 65,  // materialize to row-major contiguous

    // element-wise
    ADD = 70,
    MUL = 71,
    SIGMOID  = 72,   // 1 / (1 + exp(-x))
    EXP      = 73,   // exp(x)
    SOFTPLUS = 74,   // log(1 + exp(x))
    // Scalar IEEE sigmoid for numerically sensitive recurrent graphs.
    // SIGMOID keeps the vector approximation used by transformer paths.
    SWIGLU   = 75,   // silu(gate) * up over a merged [2I,...] tensor (gate|up halves)
    SIGMOID_EXACT = 76,
    EXP_EXACT     = 77,
    GEMV_SPARSE_A = 78,
    SIGMOID_MUL   = 79,   // value * sigmoid(gate)

    // KV cache
    QUANTIZE_KV   = 80,
    DEQUANTIZE_KV = 81,

    // ---- Phase 2+ (reserved) ----
    // FLASH_ATTN         = 100,
    GATED_DELTANET_DECODE  = 110,
    GATED_DELTANET_PREFILL = 111,
    GATED_DELTANET_CONV_DECODE = 112,
    GATED_DELTANET_CONV_VERIFY = 113,
    MOE                 = 120,
    HC_PRE               = 130,
    HC_POST              = 131,
    HC_HEAD              = 132,
    GR_REDUCE            = 133,
    GR_INJECT            = 134,
    PLE_LOOKUP           = 135,
    PLE_GATE             = 136,
    PLE_DILATED_CONV     = 137,
    DSV4_COMPRESSOR      = 160,
    DSV4_INDEXER         = 161,
    DSV4_SPARSE_ATTN     = 162,
    DSV4_GROUPED_LINEAR  = 163,
    SHORTCONV      = 140,
    RWKV7           = 150,
    RWKV_TOKEN_SHIFT = 151,
    RWKV_MIX         = 152,
    RWKV_L2_NORM     = 153,
    RWKV_POST        = 157,
};

inline const char* op_type_name(OpType op) {
    switch (op) {
    case OpType::INPUT: return "INPUT";
    case OpType::CONSTANT: return "CONSTANT";
    case OpType::MATMUL: return "MATMUL";
    case OpType::MATMUL_BATCH: return "MATMUL_BATCH";
    case OpType::RMS_NORM: return "RMS_NORM";
    case OpType::LAYER_NORM: return "LAYER_NORM";
    case OpType::ADD_RMS_NORM: return "ADD_RMS_NORM";
    case OpType::RMS_NORM_ROPE: return "RMS_NORM_ROPE";
    case OpType::QK_RMS_NORM_ROPE: return "QK_RMS_NORM_ROPE";
    case OpType::GROUP_RMS_NORM: return "GROUP_RMS_NORM";
    case OpType::SILU: return "SILU";
    case OpType::GELU: return "GELU";
    case OpType::TANH: return "TANH";
    case OpType::ROTARY_EMBED: return "ROTARY_EMBED";
    case OpType::SDPA: return "SDPA";
    case OpType::SDPA_MLA: return "SDPA_MLA";
    case OpType::RESHAPE: return "RESHAPE";
    case OpType::PERMUTE: return "PERMUTE";
    case OpType::CONCAT: return "CONCAT";
    case OpType::SLICE: return "SLICE";
    case OpType::TILE: return "TILE";
    case OpType::CONTIGUOUS: return "CONTIGUOUS";
    case OpType::ADD: return "ADD";
    case OpType::MUL: return "MUL";
    case OpType::SIGMOID: return "SIGMOID";
    case OpType::EXP: return "EXP";
    case OpType::SOFTPLUS: return "SOFTPLUS";
    case OpType::SIGMOID_EXACT: return "SIGMOID_EXACT";
    case OpType::EXP_EXACT: return "EXP_EXACT";
    case OpType::GEMV_SPARSE_A: return "GEMV_SPARSE_A";
    case OpType::SWIGLU: return "SWIGLU";
    case OpType::SIGMOID_MUL: return "SIGMOID_MUL";
    case OpType::QUANTIZE_KV: return "QUANTIZE_KV";
    case OpType::DEQUANTIZE_KV: return "DEQUANTIZE_KV";
    case OpType::GATED_DELTANET_DECODE: return "GATED_DELTANET_DECODE";
    case OpType::GATED_DELTANET_PREFILL: return "GATED_DELTANET_PREFILL";
    case OpType::GATED_DELTANET_CONV_DECODE: return "GATED_DELTANET_CONV_DECODE";
    case OpType::GATED_DELTANET_CONV_VERIFY: return "GATED_DELTANET_CONV_VERIFY";
    case OpType::MOE: return "MOE";
    case OpType::HC_PRE: return "HC_PRE";
    case OpType::HC_POST: return "HC_POST";
    case OpType::HC_HEAD: return "HC_HEAD";
    case OpType::GR_REDUCE: return "GR_REDUCE";
    case OpType::GR_INJECT: return "GR_INJECT";
    case OpType::PLE_LOOKUP: return "PLE_LOOKUP";
    case OpType::PLE_GATE: return "PLE_GATE";
    case OpType::PLE_DILATED_CONV: return "PLE_DILATED_CONV";
    case OpType::DSV4_COMPRESSOR: return "DSV4_COMPRESSOR";
    case OpType::DSV4_INDEXER: return "DSV4_INDEXER";
    case OpType::DSV4_SPARSE_ATTN: return "DSV4_SPARSE_ATTN";
    case OpType::DSV4_GROUPED_LINEAR: return "DSV4_GROUPED_LINEAR";
    case OpType::SHORTCONV: return "SHORTCONV";
    case OpType::RWKV7: return "RWKV7";
    case OpType::RWKV_TOKEN_SHIFT: return "RWKV_TOKEN_SHIFT";
    case OpType::RWKV_MIX: return "RWKV_MIX";
    case OpType::RWKV_L2_NORM: return "RWKV_L2_NORM";
    case OpType::RWKV_POST: return "RWKV_POST";
    }
    return "UNKNOWN";
}

// ---- op parameters ----
//
// Stored as flat arrays so serialisation is trivial.  The Python transpiler
// is responsible for packing/unpacking; the C++ kernels know their own
// expected layout.
//
struct OpParams {
    std::vector<int32_t>  i32;
    std::vector<float>    f32;
    std::vector<std::string> str;

    bool empty() const { return i32.empty() && f32.empty() && str.empty(); }
};

// ---- one node in the graph ----
struct GraphNode {
    uint32_t   id       = 0;
    OpType     op_type  = OpType::INPUT;
    std::vector<uint32_t> inputs;   // IDs of source nodes
    OpParams   params;

    // output shape as a flat list of 4 int64s (always 4 elements)
    int64_t    out_shape[4] = {0, 1, 1, 1};
    // per-dim symbolic expression.  Default CONST (out_shape[i] is the size).
    // When SEQ/MUL/ADD, runtime evaluates against runtime_seq_len.
    DimExpr    dim_expr[4] = {};
    Precision  out_prec     = Precision::FP32;
};

// ---- the full graph ----
struct Graph {
    std::vector<GraphNode> nodes;

    // Indices into nodes[] for graph-level inputs / outputs.
    // These are set by the Python transpiler and used by the engine
    // to know which tensors to feed / extract.
    std::vector<uint32_t> graph_inputs;
    std::vector<uint32_t> graph_outputs;

    // ---- graph metadata (serialised in header, after magic/version) ----
    // Model-specific config that the engine needs to set up RoPE, caches, etc.
    // Stored as key=value string pairs for extensibility.
    std::unordered_map<std::string, std::string> metadata;

    // ---- runtime state (not serialised) ----
    struct Runtime {
        std::vector<Tensor>     tensors;  // indexed by node ID
        std::vector<MappedFile> weights;  // mmap'd weight files
        BufferPool              pool;
    };
    Runtime runtime;

    // ---- helpers ----
    const GraphNode* input_node(uint32_t idx) const {
        return idx < graph_inputs.size() ? &nodes[graph_inputs[idx]] : nullptr;
    }
    const GraphNode* output_node(uint32_t idx) const {
        return idx < graph_outputs.size() ? &nodes[graph_outputs[idx]] : nullptr;
    }
};

// ---- per-node parameter access helpers (used by kernels) ----
namespace graph_params {

inline int32_t  get_i32(const OpParams& p, size_t idx, int32_t def = 0) {
    return idx < p.i32.size() ? p.i32[idx] : def;
}
inline float    get_f32(const OpParams& p, size_t idx, float def = 0.f) {
    return idx < p.f32.size() ? p.f32[idx] : def;
}
inline const std::string& get_str(const OpParams& p, size_t idx) {
    static const std::string empty;
    return idx < p.str.size() ? p.str[idx] : empty;
}

} // namespace graph_params

// ---------------------------------------------------------------------------
// graph I/O — declared here, implemented in io.cpp
// ---------------------------------------------------------------------------

/// Load a graph from a binary file.  Returns true on success.
bool graph_load(Graph& g, const char* path);

/// Save a graph to a binary file (used by Python transpiler).
bool graph_save(const Graph& g, const char* path);
