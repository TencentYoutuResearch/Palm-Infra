#pragma once

#include "kernels/tensor.h"
#include "graph/buffer_pool.h"
#include "graph/mmap_file.h"

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// PROJECT_NAME — Graph definition
//
// A Graph is a linear (topologically-sorted) list of nodes.  Execution walks
// the list once, dispatching each node to its kernel.  All graph construction
// and optimisation happens in Python; the C++ side only loads and executes.
// ---------------------------------------------------------------------------

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

    // element-wise
    ADD = 70,
    MUL = 71,
    SIGMOID  = 72,   // 1 / (1 + exp(-x))
    EXP      = 73,   // exp(x)
    SOFTPLUS = 74,   // log(1 + exp(x))

    // KV cache
    QUANTIZE_KV   = 80,
    DEQUANTIZE_KV = 81,

    // ---- Phase 2+ (reserved) ----
    // FLASH_ATTN         = 100,
    GATED_DELTANET_DECODE  = 110,
    GATED_DELTANET_PREFILL = 111,
    // MOE_DISPATCH       = 120,
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
    case OpType::ADD: return "ADD";
    case OpType::MUL: return "MUL";
    case OpType::SIGMOID: return "SIGMOID";
    case OpType::EXP: return "EXP";
    case OpType::SOFTPLUS: return "SOFTPLUS";
    case OpType::QUANTIZE_KV: return "QUANTIZE_KV";
    case OpType::DEQUANTIZE_KV: return "DEQUANTIZE_KV";
    case OpType::GATED_DELTANET_DECODE: return "GATED_DELTANET_DECODE";
    case OpType::GATED_DELTANET_PREFILL: return "GATED_DELTANET_PREFILL";
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
