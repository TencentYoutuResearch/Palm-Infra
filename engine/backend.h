#pragma once

#include "graph/graph.h"
#include "kernels/tensor.h"
#include "graph/buffer_pool.h"

#include <cstdint>
#include <cstring>

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

    /// Reset/query a recoverable dispatch failure. Most kernels are legacy
    /// void functions, so this side channel lets checked operators stop graph
    /// execution without changing every backend dispatch signature at once.
    virtual void clear_dispatch_error() {}
    virtual bool dispatch_failed() const { return false; }

    // -----------------------------------------------------------------------
    // Storage allocation hooks.
    //
    // execute_graph() calls these instead of touching BufferPool directly, so
    // a device-resident backend (Metal) can allocate device storage rather
    // than host memory. The default implementation is the exact host BufferPool
    // behaviour the CPU executor used before, so CPU semantics are byte-identical.
    // -----------------------------------------------------------------------

    /// Allocate storage for a node output of `nbytes`. `out.data` is a
    /// non-null storage handle on success, but device backends may place an
    /// opaque, non-host-accessible address there. Host access must use the
    /// explicit transfer methods below.
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

    /// Explicit tensor transfer boundary. Device backends override these;
    /// the CPU implementation copies from ordinary host storage. Offsets are
    /// relative to the tensor view, after any backend-specific device offset.
    virtual bool copy_to_host(const Tensor& source, void* destination,
                              size_t nbytes, size_t source_offset = 0) {
        if (!source.data || !destination ||
            source_offset > source.view_span_bytes() ||
            nbytes > source.view_span_bytes() - source_offset)
            return false;
        std::memcpy(
            destination,
            static_cast<const uint8_t*>(source.data) + source_offset,
            nbytes);
        return true;
    }
    virtual bool copy_from_host(const void* source, Tensor& destination,
                                size_t nbytes,
                                size_t destination_offset = 0) {
        if (!source || !destination.data ||
            destination_offset > destination.view_span_bytes() ||
            nbytes > destination.view_span_bytes() - destination_offset)
            return false;
        std::memcpy(
            static_cast<uint8_t*>(destination.data) + destination_offset,
            source, nbytes);
        return true;
    }

    /// Apply an FP32 -> BF16-rounded FP32 activation boundary in place.
    /// CPUBackend inherits the host implementation; device backends launch
    /// their own kernel so the executor never dereferences device memory.
    virtual bool round_to_bf16(Tensor& tensor) {
        if (tensor.prec != Precision::FP32 || !tensor.data)
            return false;
        float* values = tensor.ptr<float>();
        const size_t count = static_cast<size_t>(tensor.nelements());
        for (size_t i = 0; i < count; ++i) {
            uint32_t bits = 0;
            std::memcpy(&bits, &values[i], sizeof(bits));
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits += 0x7fffu + ((bits >> 16) & 1u);
            bits &= 0xffff0000u;
            std::memcpy(&values[i], &bits, sizeof(values[i]));
        }
        return true;
    }

    /// Select the physical precision for persistent KV storage. Graph files
    /// describe the preferred format; a backend may request a correctness
    /// fallback when its attention implementation cannot consume it.
    virtual Precision kv_cache_precision(Precision requested) const {
        return requested;
    }

    /// Complete preceding device writes before a host-side consumer such as
    /// SSD route prediction performs an explicit transfer. CPU is already
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

    void clear_dispatch_error() override { dispatch_failed_ = false; }
    bool dispatch_failed() const override { return dispatch_failed_; }
    Precision kv_cache_precision(Precision requested) const override {
        if (requested == Precision::FP16 &&
            !mollm::cpu::capabilities().fp16_kv_cache)
            return Precision::FP32;
        return requested;
    }

private:
    bool dispatch_failed_ = false;
};
