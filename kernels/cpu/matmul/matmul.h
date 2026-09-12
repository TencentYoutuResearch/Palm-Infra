#pragma once

#include "kernels/cpu/activations.h" // for Activation enum
#include "backends/cpu/platform.h"
#include "kernels/tensor.h"

#include <string>
#include <unordered_map>
#include <vector>

class ThreadPool;

// Runtime-configurable matmul parameters (for benchmarking).
struct MatmulConfig {
    int k_block = 2048;              // 0 = disable K-blocking (sweep: 512→2048 = +4-8%)
    int gemv_chunk_size = 64;        // chunk size for M==1 or N==1 shapes
    bool use_interleave_pack = true; // B interleaved packing for FP16
    bool use_fp16_accumulate =
        true; // FP16 accumulate (2x throughput, may lose precision)
};

extern MatmulConfig g_matmul_config;
extern bool g_mollm_force_fp32_acc; // debug: force FP32 accumulation

// Low-precision floating-point helpers. MXFP4 uses the OCP-defined E2M1
// element encoding and E8M0 block scale; FP8 dense weights use E4M3FN.
float decode_e8m0(uint8_t value);
float decode_fp8_e4m3fn(uint8_t value);
uint8_t encode_fp8_e4m3fn(float value);
float decode_mxfp4_e2m1(uint8_t nibble);

// Build the load-time Q8-dot sidecar used to execute native FP8 weights on
// ARM CPUs without retaining a dequantized FP16/FP32 copy.
size_t pack_fp8_e4m3_q8dot_bytes(int N, int K);
bool pack_fp8_e4m3_q8dot(const uint8_t* source,
                         const uint8_t* e8m0_scales,
                         int N, int K,
                         int8_t* packed,
                         float* q8_scales);

// Exact F32-activation reference path used by validation and as the portable
// fallback for MXFP4 GEMM. Production M=1 dispatch uses the Q8/SDOT path.
void kernel_matmul_mxfp4_reference(const Tensor& A, const Tensor& B, Tensor& C,
                                   ThreadPool* thread_pool = nullptr);

// Multiply ordinary floating-point activations by a native block-scaled FP8
// weight without quantizing the activation. DeepSeek-V4's grouped wo_a
// projection is defined this way: the official runtime dequantizes that
// checkpoint tensor to BF16 and executes a regular einsum.
bool kernel_matmul_fp8_weight_f32_activation(
    const Tensor& A, const Tensor& B, Tensor& C,
    ThreadPool* thread_pool = nullptr);

// Engine-lifetime buffers that own load-time matmul repacks. The key is the
// package/file weight path plus a layout suffix where one source weight needs
// multiple layouts.
using PackedWeightMap = std::unordered_map<std::string, std::vector<uint8_t>>;

// Whether this build can consume the packed INT4 Q4-dot/BG32/BG128 layouts
// emitted by the converter.
bool matmul_int4_q4dot_kernel_available();

// Prepare load-time layouts consumed by the CPU matmul kernels. `weight_data`
// must point at the source row-major (or prepacked INT4) bytes for `weight`.
// Embedding tables pass false for `pack_fp16` because lookup requires their
// original row-major layout.
void prepare_matmul_weight(Tensor& weight, const std::string& key,
                           const void* weight_data,
                           PackedWeightMap& packed_weights,
                           PreparedWeightMap& prepared_weights,
                           bool pack_fp16 = true,
                           bool pack_fp8 = true);

// Build the exact-value interleaved FP16 layout for a block-scaled FP8 tensor
// whose reference semantics first round dequantized weights to BF16.
bool prepare_fp8_bf16_fp16_weight(
    Tensor& weight, const std::string& key, const void* weight_data,
    PackedWeightMap& packed_weights);

extern "C" {
int mollm_matmul_shape_profile_enabled();
void mollm_set_matmul_profile_phase(const char* phase);
void mollm_reset_matmul_shape_profile();
void mollm_print_matmul_shape_profile(const char* title, int top_n);
}

// ---------------------------------------------------------------------------
// mollm — Matmul kernels
//
// C[M,N] = A[M,K] * B[K,N]
//   A: M×K, row-major (stride[1] = row stride)
//   B: K×N, row-major
//   C: M×N, row-major
//
// `act` is an optional fused activation applied to C at writeback time
// (avoids a separate SILU/GELU op + memory round-trip). When act != NONE,
// only output columns in [act_n_begin, act_n_begin + act_n_len) get the
// activation applied; the rest are written raw. Set act_n_len = -1 for
// "apply to whole N" (fast path, no per-column check).
// `force_fp32_acc` selects FP32 accumulation for this call without changing
// the process-wide matmul configuration.
//
// kernel_matmul_fp32 dispatches to the best available implementation
// (NEON SIMD for ARM, scalar fallback otherwise).
// ---------------------------------------------------------------------------

void kernel_matmul_fp32(const Tensor& A, const Tensor& B, Tensor& C,
                        ThreadPool* thread_pool = nullptr,
                        Activation act = Activation::NONE, int act_n_begin = 0,
                        int act_n_len = -1,
                        bool force_fp32_acc = false);

// Execute several M=1 INT4 GEMVs in one thread-pool dispatch. Every GEMV
// still uses all workers (each worker owns the same output-row shard across
// the batch), avoiding the loss of per-GEMV parallelism while amortizing
// dispatch barriers. Returns false when a tensor/layout is unsupported.
bool kernel_matmul_int4_gemv_batch(const std::vector<Tensor>& inputs,
                                   const std::vector<Tensor>& weights,
                                   std::vector<Tensor>& outputs,
                                   ThreadPool* thread_pool);
// Execute several Q8-dot GEMVs backed by either native INT8 weights or the
// load-time Q8 sidecar of FP8_E4M3 weights. Identical input pointers share one
// activation quantization.
bool kernel_matmul_int8_gemv_batch(const std::vector<Tensor>& inputs,
                                   const std::vector<Tensor>& weights,
                                   std::vector<Tensor>& outputs,
                                   ThreadPool* thread_pool);
bool kernel_matmul_mxfp4_gemv_batch(const std::vector<Tensor>& inputs,
                                    const std::vector<Tensor>& weights,
                                    std::vector<Tensor>& outputs,
                                    ThreadPool* thread_pool);
bool kernel_matmul_nvfp4_gemv_batch(const std::vector<Tensor>& inputs,
                                    const std::vector<Tensor>& weights,
                                    std::vector<Tensor>& outputs,
                                    ThreadPool* thread_pool);
// Reorder row-major packed NVFP4 values into four-row tiles. Within each
// K-block, two adjacent rows occupy one 16-byte vector so the Q8-dot kernel can
// unpack both with a single load. Source and destination sizes are N*K/2.
bool pack_nvfp4_q8_pairs(const uint8_t* source, uint8_t* destination,
                         int N, int K);
// Execute several M=1 FP32 GEMVs in one thread-pool dispatch. This preserves
// each projection's FP32 accumulation order while amortizing worker barriers.
bool kernel_matmul_fp32_gemv_batch(const std::vector<Tensor>& inputs,
                                   const std::vector<Tensor>& weights,
                                   std::vector<Tensor>& outputs,
                                   ThreadPool* thread_pool);

// Execute independent, same-shaped [A, weight] pairs into consecutive slices
// of one output tensor. Decode can share a single worker-pool dispatch.
void kernel_matmul_batch(const std::vector<const Tensor*>& pairs,
                         Tensor& output, ThreadPool* thread_pool);

// Invalidate decode-local activation sidecars. Repeated FP8 linears fed by
// the same Tensor can share an exact FP8->Q8 representation within one graph
// execution, but never across executions where its storage may be overwritten.
void matmul_reset_activation_cache();

void kernel_gemv_sparse_a(const Tensor& A, const Tensor& B, Tensor& C,
                          ThreadPool* thread_pool = nullptr);

// Pack full B [N, K] row-major → interleaved [N/8, K, 8] layout.
// For each N-tile of 8 rows, transpose so that for fixed k,
// B_packed[tile_base + k*8 + 0..7] are 8 consecutive FP16 values.
// Enables vld1q_f16 contiguous load instead of strided gather.
// Returns newly allocated buffer (caller owns, must delete[]).
// K_weight is the stride between consecutive k rows in B_original
// (typically == K for row-major).
__fp16* pack_b_interleaved_full(const __fp16* B_original, int N, int K,
                                int K_weight);

// Pack full int8 B [N, K] row-major -> interleaved [N/8, K, 8] layout.
// Same layout as FP16 B packing, but with int8 elements. Padding rows are zero.
int8_t* pack_b_interleaved_int8_full(const int8_t* B_original, int N, int K,
                                     int K_weight);

// Pack full int8 B [N, K] row-major -> Q8-dot layout [N/8, K/32, 8, 32].
// Padding output rows and K tail are zero.
int8_t* pack_b_q8dot_int8_full(const int8_t* B_original, int N, int K,
                               int K_weight);

// Pack full int4 B [N, ceil(K/2)] row-major -> Q4-dot layout
// [N/8, K/32, 8, 16 packed bytes]. Padding output rows and K tail are zero.
uint8_t* pack_b_q4dot_int4_full(const uint8_t* B_original, int N, int K,
                                int K_weight);

// Pack Q4-dot B plus W4G32 scales -> [N/8, K/32] blocks.
// Each block stores float scales[8] then q4dot q[8][16].
size_t pack_b_q4dot_g32_bytes(int N, int K);
uint8_t* pack_b_q4dot_g32_full(const uint8_t* B_q4dot, const float* scales,
                               int N, int K, int groups_per_row);
size_t pack_b_q4_vnni_bytes(int N, int K);
uint8_t* pack_b_q4_vnni_full(const void* B_q4_g32, int N, int K);

// Pack Q4-dot B plus W4G128 scales -> [N/8, K/128] blocks.
// Each block stores float scales[8] then q4dot q[4][8][16].
size_t pack_b_q4dot_g128_bytes(int N, int K);
uint8_t* pack_b_q4dot_g128_full(const uint8_t* B_q4dot, const float* scales,
                                int N, int K, int groups_per_row);

// Expand G128 nibbles into the [N/8,K,8] signed-byte layout consumed by
// sparse-A GEMV. Exposed for sparse kernel validation.
int8_t* pack_b_sparse_int4_g128_full(const void* B_g128, int N, int K);
