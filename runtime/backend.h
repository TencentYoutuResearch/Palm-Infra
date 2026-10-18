#pragma once

#include "core/tensor.h"
#include "runtime/host_buffer_pool.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

class ThreadPool;
struct GraphNode;
struct Graph;

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
// The engine picks a backend at load time based on target / config. Concrete
// implementations live under backends/; this contract is shared by CPU,
// Metal, CUDA, and future backends.
// ---------------------------------------------------------------------------

enum class ShapeMode {
    DYNAMIC,         // actual seq_len at runtime, no padding (CPU)
    STATIC_PADDED,   // pad input to compile-time seq_len (NPU future)
};

// Backend-owned immutable byte snapshot of plain recurrent state. KV control
// metadata is not included. Optional hooks keep unsupported backends on the
// existing host transfer path.
class BackendStateSnapshot {
public:
    virtual ~BackendStateSnapshot() = default;
    virtual bool restore() const = 0;
    virtual size_t bytes() const = 0;
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

    /// Optional host-resident MoE delegation. A hybrid device backend may
    /// consume CPU tensors, execute only routed experts on the device, and
    /// write the final FP32 output back into host storage. Returns true when
    /// the node was handled; `success` distinguishes a completed dispatch from
    /// a handled runtime failure. Unsupported nodes return false so the normal
    /// graph backend can execute them.
    virtual bool dispatch_host_moe(
        const GraphNode&, const std::vector<const Tensor*>&, Tensor*,
        ThreadPool*, bool& success) {
        success = false;
        return false;
    }

    virtual uint64_t completed_host_moe_batches() const { return 0; }

    /// Reset/query a recoverable dispatch failure. Most kernels are legacy
    /// void functions, so this side channel lets checked operators stop graph
    /// execution without changing every backend dispatch signature at once.
    virtual void clear_dispatch_error() {}
    virtual bool dispatch_failed() const { return false; }

    /// Start a new execute_graph() pass, including partial/repeated passes.
    /// Invalidate backend execution caches even when tensor addresses and
    /// workspace storage are reused. This does not open a device command
    /// buffer: begin_graph()/end_graph() remain owned by the caller.
    virtual void begin_execution() {}
    // Phase-scoped admission; does not override engine memory/length gates.
    virtual bool set_logical_prefill_batch(bool enabled) { return !enabled; }
    virtual bool logical_prefill_batch_enabled() const { return false; }

    /// Optional execution-plan inspection. Graph/tensor storage stays owned by
    /// the caller; no serialized graph or weight representation is changed.
    virtual void prepare_graph(const Graph&) {}
    // Drop pointer-bearing plans before a temporary graph/runtime is destroyed.
    virtual void release_graph_plan(const Graph&) {}

    // Optional stateless same-layer MoE aggregation. Outputs are independently
    // allocated by the executor. False must leave both outputs/inputs untouched;
    // true may signal a failed dispatch via dispatch_failed(). Default fallback
    // executes each window normally. Caller owns begin_graph/end_graph.
    virtual bool dispatch_moe_pair(const GraphNode&,
        const std::vector<const Tensor*>&, const std::vector<const Tensor*>&,
        Tensor*, Tensor*) { return false; }

    // Opt-in stateful lifetime contract. -1 means unknown: retain everything.
    // A nonnegative index names the ONLY mutated/persistent input. The output
    // must have independent storage; every other input and the output are not
    // retained by the backend beyond dispatch's ordered stream work. Returning
    // buffers to the pool must preserve that stream ordering, including fallback.
    // Planner still retains graph outputs/view owners and the mutable state's
    // owner. This is not a promise that graph capture or model use is validated.
    virtual int independent_stateful_mutable_input(const GraphNode&) const { return -1; }

    // Multi-state version of the same contract (e.g. SDPA mutates both K/V).
    // UINT64_MAX means unknown; zero explicitly declares no mutated inputs.
    // Existing single-state backends keep their behavior; all set bits must
    // identify valid input indices (at most63 inputs for this contract).
    virtual uint64_t independent_stateful_mutable_inputs(const GraphNode& node) const {
        const int index = independent_stateful_mutable_input(node);
        return index >= 0 && index < 63 ? uint64_t(1) << index : UINT64_MAX;
    }

    // Optional adjacent SIGMOID -> GR_REDUCE fusion. The executor verifies
    // exclusive consumption and preserves the original path for diagnostics.
    // A false dispatch result must leave inputs/output untouched for fallback.
    virtual bool supports_sigmoid_gr_reduce_fusion() const { return false; }
    // Optional adjacent HC up -> sigmoid -> reduce. Inputs are normalized
    // carry, low-rank activation, up weight. False must leave graph tensors
    // untouched; executor retains up inputs and materializes original chain.
    virtual bool supports_hc_up_reduce_fusion() const { return false; }
    virtual int hc_up_reduce_max_rows() const { return 1024; }
    virtual bool dispatch_hc_up_reduce(const GraphNode&,const GraphNode&,
        const std::vector<const Tensor*>&,Tensor*,ThreadPool*) { return false; }
    virtual bool dispatch_sigmoid_gr_reduce(
        const GraphNode&, const std::vector<const Tensor*>&,
        Tensor*, ThreadPool*) { return false; }

    // Optional dual-output GR_INJECT -> GROUP_RMS_NORM fusion. Both the raw
    // carry and normalized output remain materialized, including multi-users.
    // Inputs are branch, residual, gates, norm weight. False means untouched.
    virtual bool supports_gr_inject_rms_fusion() const { return false; }
    virtual bool dispatch_gr_inject_rms(
        const GraphNode&, const GraphNode&, const std::vector<const Tensor*>&,
        Tensor*, Tensor*, ThreadPool*) { return false; }
    // Executor-proven single-use, unexposed MOE branch followed by inject/RMS.
    // Inputs remain live through RMS; false must leave every output untouched.
    // Success need not materialize branch; carry and normalized remain logical
    // outputs (device backends may use their established deferred contracts).
    virtual bool supports_moe_gr_inject_rms_fusion() const { return false; }
    virtual bool dispatch_moe_gr_inject_rms(const GraphNode&,const GraphNode&,
        const GraphNode&,const std::vector<const Tensor*>&,
        const std::vector<const Tensor*>&,Tensor*,Tensor*,Tensor*,ThreadPool*) { return false; }

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
    // Optional exact, address-preserving reshape of a strided input. Must be
    // pure and agree with dispatch: executor allocation and borrowed-owner
    // reset/release classification all use this same predicate. Default keeps
    // every other backend's existing materialization semantics unchanged.
    virtual bool reshape_view_strides(const Tensor&, const Tensor&, size_t[4]) const {
        return false;
    }
    virtual bool supports_state_snapshot(const std::vector<Tensor*>&) const {
        return false;
    }
    // Optional post-token recurrent snapshots for speculative verification.
    // restore must validate complete coverage before changing any state.
    virtual bool begin_state_prefixes(const std::vector<Tensor*>&, int) { return false; }
    virtual bool restore_state_prefix(int) { return false; }
    virtual void end_state_prefixes() {}
    virtual std::unique_ptr<BackendStateSnapshot> capture_state_snapshot(
        const std::vector<Tensor*>&) { return nullptr; }

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

    /// Zero a byte range in tensor storage. Device backends override this to
    /// avoid making persistent state host-addressable solely for reset.
    virtual bool zero_tensor(Tensor& tensor, size_t nbytes,
                             size_t destination_offset = 0) {
        if (!tensor.data ||
            destination_offset > tensor.view_span_bytes() ||
            nbytes > tensor.view_span_bytes() - destination_offset)
            return false;
        std::memset(
            static_cast<uint8_t*>(tensor.data) + destination_offset,
            0, nbytes);
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
