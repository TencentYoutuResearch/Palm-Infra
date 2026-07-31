#pragma once

#include "engine/accelerator_backend.h"
#include <memory>
#include <string>

// ---------------------------------------------------------------------------
// mollm — MetalBackend
//
// Whole-graph-resident Metal (Apple GPU) backend. Intermediates live in
// MTLBuffers (via MetalBufferPool); weights are wrapped zero-copy over the
// resident package weight region; the KV cache is device-resident. Only
// input/output boundary tensors are copied host<->device.
//
// This header is plain C++ (opaque void* handles, PIMPL) so it can be included
// by pure-C++ translation units. The implementation is in metal_backend.mm.
//
// Only compiled when MOLLM_METAL is defined.
// ---------------------------------------------------------------------------

class MetalBufferPool;

class MetalBackend : public AcceleratorBackend {
public:
    // metallib_path: path to the compiled default.metallib. If empty, the
    // backend looks next to the executable / uses a compile-time define.
    explicit MetalBackend(const std::string& metallib_path = "");
    ~MetalBackend() override;

    MetalBackend(const MetalBackend&) = delete;
    MetalBackend& operator=(const MetalBackend&) = delete;

    /// True if a Metal device was found and the metallib loaded.
    bool available() const override;

    // --- Backend interface ---
    ShapeMode shape_mode() const override { return ShapeMode::DYNAMIC; }

    void dispatch(const GraphNode& node,
                  const std::vector<const Tensor*>& inputs,
                  Tensor* output, ThreadPool* thread_pool) override;

    void* alloc_output(Tensor& out, size_t nbytes, BufferPool* pool) override;
    void  free_output(Tensor& t, BufferPool* pool) override;
    bool  is_device_resident() const override { return true; }
    void  synchronize_for_host_read() override;

    void begin_graph() override;
    void end_graph() override;

    // --- Metal-specific helpers ---

    /// True if the Metal 4 tensor API (int8/fp16 matmul2d) is usable on this GPU
    /// (M5 / A19+). Gates the fast tensor GEMM and W8A8 int8-MMA paths.
    bool has_tensor_path() const;

    /// Register the whole package weight region as one shared MTLBuffer wrapping
    /// the mmap (newBufferWithBytesNoCopy). Individual weight tensors then carry
    /// device_offset = (weight ptr - base). Returns false if wrapping failed.
    bool register_weight_region(void* base, size_t size) override;

    /// Enable per-weight copies for hybrid SSD-MoE. The package remains mmap'd
    /// so expert aggregates stay unmaterialized; dense constants are copied
    /// into individual Shared Metal buffers by wrap_weight().
    void enable_weight_copy_mode() override;

    /// True when hybrid copy mode currently owns at least one dense weight.
    bool has_weight_copies() const override;

    /// Configure the full-Metal SSD expert cache. Metal I/O loads package
    /// ranges directly into Shared buffers consumed by routed-expert kernels.
    bool configure_moe_ssd_io(const std::string& package_path,
                              size_t capacity_bytes, int max_commands_in_flight,
                              bool cross_layer_prefetch) override;

    /// After register_weight_region, point a weight/constant tensor at the
    /// shared weight buffer with the correct device_offset (from t.data).
    void wrap_weight(Tensor& t) override;

    /// Second pass for packed INT4 weights (call after quant metadata is set):
    /// decode ordinary BG32/BG128 linear weights into a Metal-friendly raw
    /// nibble + scale device buffer. Aggregate expert tensors stay in their
    /// package-native layout and are consumed directly by MoE kernels.
    void wrap_weight_int4(Tensor& t,
                          bool keep_native_experts = false) override;

    /// Allocate a device-resident buffer of nbytes and point t at it (used for
    /// KV cache and recurrent state). Metal Shared storage satisfies every
    /// requested host-access mode, so t.data remains directly host-visible.
    /// Persistent buffers are owned for the lifetime of the backend.
    void alloc_persistent(
        Tensor& t, size_t nbytes,
        PersistentHostAccess host_access = PersistentHostAccess::FULL,
        size_t host_prefix_bytes = 0) override;

    /// Upload host bytes into a REUSABLE device buffer identified by `key`
    /// (e.g. graph INPUT node name like "hidden"/"cos"). The buffer is owned by
    /// the backend and reused across graph runs, growing only when a larger
    /// size is needed. Sets t.device_data / t.device_offset (t.data unchanged so
    /// host reads still work). Used for boundary inputs (hidden/mask/cos/sin).
    void upload_input(Tensor& t, const std::string& key,
                      const void* host_src, size_t nbytes) override;

    bool supports_lm_head(const Tensor& weight) const override {
        return weight.device_data &&
            (weight.prec == Precision::FP16 ||
             weight.prec == Precision::INT8 ||
             weight.prec == Precision::INT4);
    }

    /// Run a single GEMV on the GPU: out[N] = a[K] (fp32) * W[N,K] (fp16),
    /// used for the lm_head projection during decode. `a_host` is a host FP32
    /// vector of length K; `weight` is a device-resident FP16 weight tensor
    /// (W.device_data set, shape[0]=N, shape[1]=K). Results written to
    /// `out_host` (length N). Runs its own command buffer (commit+wait).
    void lm_head_gemv(const float* a_host, const Tensor& weight,
                      float* out_host, int N, int K,
                      int activation = 0) override;

    /// Append lm_head GEMV to the currently open graph command stream, reading
    /// the graph output directly from its device buffer. This closes and waits
    /// for the graph, then copies logits to `out_host` for sampling.
    void lm_head_gemv_device_and_end_graph(
        const Tensor& a, size_t a_element_offset, const Tensor& weight,
        float* out_host, int N, int K, int activation = 0) override;

private:
    void lm_head_gemv_impl(void* a_device, size_t a_byte_offset,
                           const Tensor& weight, float* out_host,
                           int N, int K, int activation,
                           bool finish_open_graph);

    /// Debug: flush the current command buffer (commit+wait) so intermediate
    /// device buffers become host-readable. No-op unless MOLLM_METAL_SYNC_EACH.
    void sync_point();

    /// Print the accumulated per-op-type GPU time table (MOLLM_METAL_PROFILE)
    /// to stderr and reset the counters. No-op if profiling is disabled.
    void dump_profile();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};
