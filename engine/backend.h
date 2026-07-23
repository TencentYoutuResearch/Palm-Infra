#pragma once

#include "graph/graph.h"
#include "kernels/tensor.h"
#include "graph/buffer_pool.h"

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

    // -----------------------------------------------------------------------
    // Storage allocation hooks.
    //
    // execute_graph() calls these instead of touching BufferPool directly, so
    // a device-resident backend (Metal) can allocate device storage rather
    // than host memory. The default implementation is the exact host BufferPool
    // behaviour the CPU executor used before, so CPU semantics are byte-identical.
    // -----------------------------------------------------------------------

    /// Allocate storage for a node output of `nbytes`. Sets out.data (host
    /// pointer, possibly a device buffer's shared contents), out.mem_type,
    /// out.owner_id, out.storage_id. Returns out.data (non-null on success).
    virtual void* alloc_output(Tensor& out, size_t nbytes, BufferPool* pool) {
        void* buf = pool->acquire(nbytes);
        if (!buf) return nullptr;
        out.data     = buf;
        out.mem_type = MemoryType::POOLED;
        out.owner_id = pool->id();
        out.storage_id = pool->storage_id(buf);
        return buf;
    }

    /// Release storage previously allocated via alloc_output(). Only called for
    /// tensors with mem_type==POOLED. Does not null the tensor fields — the
    /// executor does that after this returns.
    virtual void free_output(Tensor& t, BufferPool* pool) {
        pool->release(t.data, t.nbytes());
    }

    /// True when intermediates live in device (GPU) buffers rather than host
    /// BufferPool memory. The executor uses this to classify borrowed views by
    /// op type instead of host-pointer equality.
    virtual bool is_device_resident() const { return false; }

    /// Make preceding device writes visible through Tensor::data for a
    /// host-side consumer such as SSD route prediction. CPU is already
    /// coherent; device backends may submit and wait for queued work.
    virtual void synchronize_for_host_read() {}

    /// Called by run_graph() before/after a full execute_graph() pass so a
    /// device backend can open/commit a command buffer around the whole graph.
    virtual void begin_graph() {}
    virtual void end_graph() {}
};

// ---------------------------------------------------------------------------
// CPUBackend — Apple Silicon ARM NEON FP16FML kernels.
//
// Shape mode: DYNAMIC (no padding). Runtime injects actual seq_len into
// SEQ-tagged dims via inject_runtime_shapes().
//
// dispatch() is implemented in graph/cpu_backend.cpp and routes each graph op
// to the corresponding CPU kernel.
// ---------------------------------------------------------------------------

class CPUBackend : public Backend {
public:
    ShapeMode shape_mode() const override { return ShapeMode::DYNAMIC; }

    void dispatch(const GraphNode& node,
                  const std::vector<const Tensor*>& inputs,
                  Tensor* output, ThreadPool* thread_pool) override;
};
