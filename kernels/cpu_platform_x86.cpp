#include "kernels/cpu_platform.h"
#include "kernels/matmul.h"
#include "kernels/matmul_internal.h"
#include "kernels/threading.h"
#include "kernels/x86_avx2.h"
#include "kernels/x86_avx512.h"
#include "kernels/x86_vnni.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <vector>

namespace mollm::cpu {

namespace {

using DenseFp32Fn = void (*)(const float*, const float*, float*, int, int, int,
                             int, int, int, int);
using DenseFp16Fn = void (*)(const float*, const fp16_t*, float*, int, int, int,
                             int, int, int, int);
using DenseFp16M2Fn = void (*)(const float*, const fp16_t*, float*, int, int,
                               int, int, int, int, int);
using Int4Fn = void (*)(const Tensor&, const Tensor&, Tensor&, int, int, int,
                        int, int, int);
using Int8Fn = void (*)(const float*, const int8_t*, const float*, float*, int,
                        int, int, int, int, int, int, int, int, int, int);
using Int4VnniFn = void (*)(const int8_t*, const float*, const int16_t*,
                            const void*, const Tensor&, Tensor&, int, int, int,
                            int, int, int);
using QuantizeVnniFn = void (*)(const float*, int8_t*, float*, int16_t*, int,
                                int, int, int);

struct X86Dispatch {
    Capabilities caps;
    const char* name = "x86-scalar";
    DenseFp32Fn fp32 = nullptr;
    DenseFp16Fn fp16 = nullptr;
    DenseFp16M2Fn fp16_m2 = nullptr;
    Int4Fn int4 = nullptr;
    Int4VnniFn int4_vnni = nullptr;
    QuantizeVnniFn quantize_vnni = nullptr;
    Int8Fn int8 = nullptr;
};

X86Dispatch detect_dispatch() {
    X86Dispatch dispatch;
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    const bool has_avx2 = __builtin_cpu_supports("avx2");
    const bool has_fma = __builtin_cpu_supports("fma");
    const bool has_f16c = __builtin_cpu_supports("f16c");
    const bool has_avx512 = __builtin_cpu_supports("avx512f");
    const bool has_avx512_bw = __builtin_cpu_supports("avx512bw");
    const bool has_avx512_vl = __builtin_cpu_supports("avx512vl");
    const bool has_avx512_vnni = __builtin_cpu_supports("avx512vnni");

    const char* requested = std::getenv("MOLLM_X86_ISA");
    const bool force_scalar =
        std::getenv("MOLLM_X86_DISABLE_AVX2") != nullptr ||
        (requested && std::strcmp(requested, "scalar") == 0);
    const bool cap_at_avx2 =
        requested && std::strcmp(requested, "avx2") == 0;
    const bool allow_avx512 =
        !cap_at_avx2 &&
        (!requested || std::strcmp(requested, "auto") == 0 ||
         std::strcmp(requested, "avx512") == 0);

    if (!force_scalar && allow_avx512 && has_avx512 && has_fma) {
        dispatch.caps.x86_avx2 = has_avx2;
        dispatch.caps.x86_fma = true;
        dispatch.caps.x86_f16c = has_f16c;
        dispatch.caps.x86_avx512 = true;
        dispatch.caps.x86_avx512_vnni =
            has_avx512_bw && has_avx512_vl && has_avx512_vnni;
        dispatch.caps.x86_isa = X86Isa::AVX512;
        dispatch.name = "x86-avx512";
        dispatch.fp32 = x86::matmul_fp32_avx512_range;
        if (has_f16c) {
            dispatch.fp16 = x86::matmul_fp16_avx512_range;
            dispatch.fp16_m2 = x86::matmul_fp16_m2_avx512_range_n;
        }
        dispatch.caps.fp16_m2_shared_weight = dispatch.fp16_m2 != nullptr;
        dispatch.int4 = x86::matmul_int4_bg_avx512_range;
        if (dispatch.caps.x86_avx512_vnni)
            dispatch.int4_vnni = x86::matmul_int4_bg32_vnni_range;
        if (dispatch.caps.x86_avx512_vnni)
            dispatch.quantize_vnni = x86::quantize_q8_vnni_range;
        dispatch.int8 = x86::matmul_int8_avx512_range;
        return dispatch;
    }
    if (!force_scalar && has_avx2 && has_fma) {
        dispatch.caps.x86_avx2 = true;
        dispatch.caps.x86_fma = true;
        dispatch.caps.x86_f16c = has_f16c;
        dispatch.caps.x86_isa = X86Isa::AVX2;
        dispatch.name = "x86-avx2";
        dispatch.fp32 = x86::matmul_fp32_avx2_range;
        if (has_f16c) {
            dispatch.fp16 = x86::matmul_fp16_avx2_range;
            dispatch.fp16_m2 = x86::matmul_fp16_m2_avx2_range_n;
        }
        dispatch.caps.fp16_m2_shared_weight = dispatch.fp16_m2 != nullptr;
        dispatch.int4 = x86::matmul_int4_bg_avx2_range;
        dispatch.int8 = x86::matmul_int8_avx2_range;
    }
#endif
    return dispatch;
}

const X86Dispatch& dispatch() {
    static const X86Dispatch value = detect_dispatch();
    return value;
}

int8_t unpack_int4_signed(uint8_t byte, bool high_nibble) {
    int value = high_nibble ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
    return static_cast<int8_t>(value >= 8 ? value - 16 : value);
}

}  // namespace

const Capabilities& capabilities() {
    return dispatch().caps;
}

const char* isa_name() {
    return dispatch().name;
}

void relax() {
    _mm_pause();
}

bool matmul_int4_packed(const Tensor& A, const Tensor& B, Tensor& C, int lda,
                        int ldc, ThreadPool* thread_pool) {
    const int M = static_cast<int>(A.shape[1]);
    const int K = static_cast<int>(A.shape[0]);
    const int N = static_cast<int>(B.shape[0]);
    const int groups = static_cast<int>(B.groups_per_row);
    const auto* q4dot = static_cast<const uint8_t*>(B.q4_repack_data);
    const auto* bg32 = static_cast<const Q4B8G32Block*>(B.q4_g32_data);
    const auto* bg128 = static_cast<const Q4B8G128Block*>(B.q4_g128_data);
    const bool is_q4dot = B.is_q4_repacked && q4dot && B.scales;
    const bool is_bg32 = B.is_q4_g32_packed && bg32 && B.group_size == 32;
    const bool is_bg128 = B.is_q4_g128_packed && bg128 && B.group_size == 128;
    if ((!is_q4dot && !is_bg32 && !is_bg128) || groups <= 0)
        return false;

    const auto int4_vnni = dispatch().int4_vnni;
    const auto quantize_vnni = dispatch().quantize_vnni;
    const void* vnni_data = nullptr;
    if (B.prepared_weight && (B.prepared_weight_row_offset % 8) == 0) {
        const void* base = B.prepared_weight->data(
            WeightLayout::X86_VNNI_Q4_G32);
        if (base) {
            vnni_data = static_cast<const uint8_t*>(base) +
                (B.prepared_weight_row_offset / 8) *
                    pack_b_q4_vnni_bytes(8, K);
        }
    }
    const bool use_vnni =
        int4_vnni && quantize_vnni && is_bg32 && vnni_data && M >= 4 &&
        K % 32 == 0;
    if (use_vnni) {
        struct VnniScratch {
            std::vector<int8_t> qA;
            std::vector<float> scales;
            std::vector<int16_t> sums;
        };
        thread_local VnniScratch scratch;
        scratch.qA.resize(static_cast<size_t>(M) * K);
        scratch.scales.resize(static_cast<size_t>(M) * groups);
        scratch.sums.resize(static_cast<size_t>(M) * groups);
        int8_t* const quantized_a = scratch.qA.data();
        float* const activation_scales = scratch.scales.data();
        int16_t* const activation_sums = scratch.sums.data();
        const auto quant_begin = std::chrono::steady_clock::now();
        quantize_vnni(A.ptr<float>(), quantized_a, activation_scales,
                      activation_sums, K, lda, 0, M);
        const auto quant_end = std::chrono::steady_clock::now();
        matmul_record_q8_quant_a(
            std::chrono::duration<double, std::milli>(
                quant_end - quant_begin).count());
        const int n_threads = thread_pool ? thread_pool->num_threads() : 1;
        if (n_threads > 1) {
            thread_pool->parallel_for(
                0, M, 4, [&](int, int begin, int end) {
                    int4_vnni(quantized_a, activation_scales,
                              activation_sums, vnni_data, B, C, K, ldc,
                              begin, end, 0, N);
                });
        } else {
            int4_vnni(quantized_a, activation_scales, activation_sums,
                      vnni_data, B, C, K, ldc, 0, M, 0, N);
        }
        return true;
    }

    const auto int4_kernel = dispatch().int4;
    const bool use_simd =
        int4_kernel && (is_bg32 || is_bg128) &&
        (K % static_cast<int>(B.group_size) == 0);
    auto run_range = [&](int m_begin, int m_end, int n_begin, int n_end) {
        if (use_simd) {
            int4_kernel(A, B, C, lda, ldc, m_begin, m_end, n_begin, n_end);
            return;
        }
        for (int m = m_begin; m < m_end; ++m) {
            float* out = C.ptr<float>() + static_cast<size_t>(m) * ldc;
            for (int n = n_begin; n < n_end; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    int8_t value = 0;
                    float scale = 0.0f;
                    if (is_bg128) {
                        const int group = k / 128;
                        const auto& block = bg128[
                            static_cast<size_t>(n / 8) * groups + group];
                        const int local = k & 127;
                        const uint8_t byte =
                            block.q[local / 32][n & 7][(local & 31) >> 1];
                        value =
                            unpack_int4_signed(byte, (local & 1) != 0);
                        scale = block.scales[n & 7];
                    } else if (is_bg32) {
                        const int group = k / 32;
                        const auto& block = bg32[
                            static_cast<size_t>(n / 8) * groups + group];
                        const int local = k & 31;
                        const uint8_t byte =
                            block.q[n & 7][local >> 1];
                        value =
                            unpack_int4_signed(byte, (local & 1) != 0);
                        scale = block.scales[n & 7];
                    } else {
                        const int block_k = k / 32;
                        const size_t byte_index =
                            ((static_cast<size_t>(n / 8) *
                                  ((K + 31) / 32) +
                              block_k) *
                                 8 +
                             (n & 7)) *
                                16 +
                            ((k & 31) >> 1);
                        value = unpack_int4_signed(
                            q4dot[byte_index], (k & 1) != 0);
                        scale =
                            B.scales[static_cast<size_t>(n) * groups +
                                     k / static_cast<int>(B.group_size)];
                    }
                    sum +=
                        A.ptr<float>()[static_cast<size_t>(m) * lda + k] *
                        static_cast<float>(value) * scale;
                }
                out[n] = sum;
            }
        }
    };

    const int n_threads = thread_pool ? thread_pool->num_threads() : 1;
    if (n_threads > 1 && M == 1 && N >= 64) {
        const int chunk = std::max(64, (N + n_threads - 1) / n_threads);
        thread_pool->parallel_for(
            0, N, chunk, [&](int, int n_begin, int n_end) {
                run_range(0, 1, n_begin, n_end);
            });
    } else if (n_threads > 1 && M > 1) {
        thread_pool->parallel_for(
            0, M, 1, [&](int, int m_begin, int m_end) {
                run_range(m_begin, m_end, 0, N);
            });
    } else {
        run_range(0, M, 0, N);
    }
    return true;
}

bool matmul_dense_fp32_range(const float* A, const float* B, float* C, int N,
                             int K, int lda, int K_weight, int ldc,
                             int m_begin, int m_end) {
    const auto kernel = dispatch().fp32;
    if (!kernel)
        return false;
    kernel(A, B, C, N, K, lda, K_weight, ldc, m_begin, m_end);
    return true;
}

bool matmul_dense_fp16_range(const float* A, const fp16_t* B, float* C, int N,
                             int K, int lda, int K_weight, int ldc,
                             int m_begin, int m_end, bool interleaved) {
    const auto kernel = dispatch().fp16;
    if (interleaved || !kernel)
        return false;
    kernel(A, B, C, N, K, lda, K_weight, ldc, m_begin, m_end);
    return true;
}

bool matmul_dense_fp16_m2_range_n(
    const float* A, const fp16_t* B, float* C,
    int N, int K, int lda, int K_weight, int ldc,
    int n_begin, int n_end) {
    const auto kernel = dispatch().fp16_m2;
    if (!kernel)
        return false;
    kernel(A, B, C, N, K, lda, K_weight, ldc, n_begin, n_end);
    return true;
}

bool matmul_int8_range(const float* A, const int8_t* B, const float* scales,
                       float* C, int N, int K, int group_size,
                       int groups_per_row, int lda, int K_weight, int ldc,
                       int m_begin, int m_end, int n_begin, int n_end,
                       bool interleaved) {
    const auto kernel = dispatch().int8;
    if (interleaved || !kernel)
        return false;
    kernel(A, B, scales, C, N, K, group_size, groups_per_row, lda, K_weight,
           ldc, m_begin, m_end, n_begin, n_end);
    return true;
}

}  // namespace mollm::cpu

// x86 keeps dense FP16 weights row-major, but direct packing tests still use
// this architecture-independent layout helper.
__fp16* pack_b_interleaved_full(const __fp16* source, int rows, int cols,
                                int source_stride) {
    if (!source || rows < 0 || cols < 0 || source_stride < cols)
        return nullptr;
    const int padded_rows = ((rows + 7) / 8) * 8;
    auto* packed = new __fp16[static_cast<size_t>(padded_rows) * cols];
    for (int row_tile = 0; row_tile < padded_rows; row_tile += 8) {
        const int valid_rows = std::max(0, std::min(8, rows - row_tile));
        for (int col = 0; col < cols; ++col) {
            for (int lane = 0; lane < valid_rows; ++lane) {
                packed[row_tile * cols + col * 8 + lane] =
                    source[(row_tile + lane) * source_stride + col];
            }
            for (int lane = valid_rows; lane < 8; ++lane)
                packed[row_tile * cols + col * 8 + lane] = (__fp16)0.0f;
        }
    }
    return packed;
}
