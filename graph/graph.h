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
// expressed as a symbolic expression (SEQ, N*SEQ, N+SEQ) evaluated at runtime.
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
//
// Serialized as 8 bytes (kind + coeff + padding). The coeff field is only
// used by MUL and ADD; CONST/SEQ/BATCH ignore it.
struct DimExpr {
    int8_t  kind  = 0;   // 0=CONST, 1=SEQ, 2=MUL, 3=ADD, 4=BATCH
    int32_t coeff = 0;   // multiplier (MUL) or constant term (ADD)

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
};

// ---- op types ----
enum class OpType : uint32_t {
    // meta
    INPUT  = 0,
    CONSTANT = 1,

    // linear
    MATMUL  = 10,

    // normalisation
    RMS_NORM   = 20,
    LAYER_NORM = 21,

    // activations
    SILU = 30,
    GELU = 31,

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
    SWIGLU   = 75,   // silu(gate) * up over a merged [2I,...] tensor (gate|up halves)

    // KV cache
    QUANTIZE_KV   = 80,
    DEQUANTIZE_KV = 81,

    // ---- Phase 2+ (reserved) ----
    // FLASH_ATTN         = 100,
    GATED_DELTANET_DECODE  = 110,
    GATED_DELTANET_PREFILL = 111,
    MOE                 = 120,
    // MOE_COMBINE        = 121,
    // GATED_ATTENTION    = 130,
    SHORTCONV      = 140,
};

inline const char* op_type_name(OpType op) {
    switch (op) {
    case OpType::INPUT: return "INPUT";
    case OpType::CONSTANT: return "CONSTANT";
    case OpType::MATMUL: return "MATMUL";
    case OpType::RMS_NORM: return "RMS_NORM";
    case OpType::LAYER_NORM: return "LAYER_NORM";
    case OpType::SILU: return "SILU";
    case OpType::GELU: return "GELU";
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
    case OpType::SWIGLU: return "SWIGLU";
    case OpType::QUANTIZE_KV: return "QUANTIZE_KV";
    case OpType::DEQUANTIZE_KV: return "DEQUANTIZE_KV";
    case OpType::GATED_DELTANET_DECODE: return "GATED_DELTANET_DECODE";
    case OpType::GATED_DELTANET_PREFILL: return "GATED_DELTANET_PREFILL";
    case OpType::MOE: return "MOE";
    case OpType::SHORTCONV: return "SHORTCONV";
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
