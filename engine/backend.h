#pragma once

#include "graph/graph.h"
#include "kernels/tensor.h"

class ThreadPool;

// ---------------------------------------------------------------------------
// mollm — Backend abstraction
//
// A Backend is responsible for dispatching graph ops to target-specific
// kernels. The graph itself is backend-agnostic (no NPU-specific ops, no
// PAD nodes baked in); the same .mollm package can run on any backend.
//
// Shape mode negotiation:
//   CPUBackend prefers DYNAMIC (no padding, runtime fills actual seq_len).
//   Future NPUBackend prefers STATIC_PADDED (pads to compile-time seq_len,
//   emits fixed-shape instructions).
//
// The engine picks a backend at load time based on target / config.
// ---------------------------------------------------------------------------

enum class ShapeMode {
    DYNAMIC,         // actual seq_len at runtime, no padding (CPU)
    STATIC_PADDED,   // pad input to compile-time seq_len (NPU future)
};

class Backend {
public:
    virtual ~Backend() = default;

    /// Desired shape mode for this backend.
    /// Engine reads this to decide how to set ExecContext fields before
    /// calling inject_runtime_shapes().
    virtual ShapeMode shape_mode() const = 0;

    /// For STATIC_PADDED backends: the compile-time seq_len to pad to.
    /// DYNAMIC backends return -1 (unused).
    virtual int padded_seq_len() const { return -1; }

    /// Dispatch a single op. Called by execute_graph() for each node.
    /// `node.dim_expr[d]` tells the dispatcher which output dims are
    /// runtime-dynamic; dispatchers should preserve output->shape[d]
    /// for those dims instead of overwriting from params.
    virtual void dispatch(const GraphNode& node,
                          const std::vector<const Tensor*>& inputs,
                          Tensor* output, ThreadPool* thread_pool) = 0;
};

// ---------------------------------------------------------------------------
// CPUBackend — Apple Silicon ARM NEON FP16FML kernels.
//
// Shape mode: DYNAMIC (no padding). Runtime injects actual seq_len into
// SEQ-tagged dims via inject_runtime_shapes().
//
// dispatch() routes to dispatch_kernel() in graph/execute.cpp (the existing
// kernel dispatcher). All current CPU kernels remain unchanged.
// ---------------------------------------------------------------------------

class CPUBackend : public Backend {
public:
    ShapeMode shape_mode() const override { return ShapeMode::DYNAMIC; }

    void dispatch(const GraphNode& node,
                  const std::vector<const Tensor*>& inputs,
                  Tensor* output, ThreadPool* thread_pool) override;
};
