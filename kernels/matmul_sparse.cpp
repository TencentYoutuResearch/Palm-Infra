#include "kernels/matmul_internal.h"
#include "kernels/threading.h"

#include <algorithm>
#include <cstdint>
#include <vector>

int8_t* pack_b_sparse_int4_g128_full(const void* source, int N, int K) {
    if (!source || K % 128 != 0)
        return nullptr;
    int N_padded = ((N + 7) / 8) * 8;
    int groups_per_row = K / 128;
    int8_t* dst = new int8_t[(size_t)N_padded * K];
    std::fill(dst, dst + (size_t)N_padded * K, 0);
    const auto* blocks = reinterpret_cast<const Q4B8G128Block*>(source);
    for (int n = 0; n < N_padded; n += 8) {
        const Q4B8G128Block* tile = blocks + (size_t)(n / 8) * groups_per_row;
        int valid = std::min(8, N - n);
        for (int g = 0; g < groups_per_row; ++g) {
            for (int p = 0; p < 128; ++p) {
                int qgi = p >> 5;
                int pos = p & 31;
                int k = g * 128 + p;
                for (int j = 0; j < valid; ++j) {
                    uint8_t packed = tile[g].q[qgi][j][pos >> 1];
                    int value = (pos & 1) ? (packed >> 4) : (packed & 15);
                    dst[(size_t)n * K + (size_t)k * 8 + j] =
                        (int8_t)(value >= 8 ? value - 16 : value);
                }
            }
        }
    }
    return dst;
}

void kernel_gemv_sparse_a(const Tensor& A, const Tensor& B, Tensor& C,
                          ThreadPool* thread_pool) {
    const int M = (int)A.shape[1];
    const int K = (int)A.shape[0];
    const int N = (int)B.shape[0];
    if (M != 1 || (int)B.shape[1] != K) {
        kernel_matmul_fp32(A, B, C, thread_pool);
        return;
    }
    const float* a = A.ptr<float>();
#if HAS_NEON
    const bool fp16 = B.prec == Precision::FP16;
    const bool w8 = B.prec == Precision::INT8 && B.sparse_data && B.scales;
    const bool w4 = B.prec == Precision::INT4 && B.sparse_data && B.scales &&
                    B.group_size == 128 && (K % 128) == 0;
    if (!fp16 && !w8 && !w4) {
        kernel_matmul_fp32(A, B, C, thread_pool);
        return;
    }
#else
    kernel_matmul_fp32(A, B, C, thread_pool);
    return;
#endif

    size_t density_limit =
        B.prec == Precision::FP16
            ? (size_t)K * 4 / 5
            : (B.prec == Precision::INT8 ? (size_t)K * 9 / 100 : 0);
    if (density_limit == 0) {
        kernel_matmul_fp32(A, B, C, thread_pool);
        return;
    }
    // Decode calls this once per FFN layer. Reuse the index buffer so sparse
    // dispatch does not allocate and free heap storage on every token.
    static thread_local std::vector<int> nonzero;
    nonzero.clear();
    nonzero.reserve(K / 2);
    for (int k = 0; k < K; ++k) {
        if (a[k] != 0.0f) {
            nonzero.push_back(k);
            if (nonzero.size() >= density_limit) {
                kernel_matmul_fp32(A, B, C, thread_pool);
                return;
            }
        }
    }

#if HAS_NEON
    float* c = C.ptr<float>();
    const __fp16* b16 =
        fp16 ? reinterpret_cast<const __fp16*>(B.data) : nullptr;
    const int8_t* b8 =
        w8 ? reinterpret_cast<const int8_t*>(B.sparse_data) : nullptr;
    const int8_t* b4 =
        w4 ? reinterpret_cast<const int8_t*>(B.sparse_data) : nullptr;
    const int gpr = (int)B.groups_per_row;
    // `nonzero` is thread_local so repeated decode calls can reuse its
    // capacity.  Do not access that TLS variable from worker threads: each
    // worker would see its own empty vector.  Capture the main thread's
    // immutable storage explicitly for the duration of this dispatch.
    const int* active_indices = nonzero.data();
    const int active_count = (int)nonzero.size();
    auto for_each_active = [&](auto&& fn) {
        for (int i = 0; i < active_count; ++i)
            fn(active_indices[i]);
    };

    auto run_range = [&](int n_begin, int n_end) {
        for (int n = n_begin; n < n_end; n += 8) {
            int valid = std::min(8, N - n);
            const __fp16* tile16 = fp16 ? b16 + (size_t)(n & ~7) * K : nullptr;
            const int8_t* tile8 =
                !fp16 ? (w8 ? b8 : b4) + (size_t)(n & ~7) * K : nullptr;
            float16x8_t acc[4] = {
                vdupq_n_f16((__fp16)0), vdupq_n_f16((__fp16)0),
                vdupq_n_f16((__fp16)0), vdupq_n_f16((__fp16)0)};
            // FP16 value projections have thousands of active inputs.  Half
            // accumulation is not accurate enough across that many terms and
            // caused the sparse RWKV decode graph to diverge badly from its
            // dense/prefill counterpart.  Keep four independent FP32 chains
            // to hide FMA latency without sacrificing accumulation precision.
            float32x4_t acc32_lo[4] = {
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
            float32x4_t acc32_hi[4] = {
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
            int slot = 0;
            int last_group = -1;
            float16x8_t scale = vdupq_n_f16((__fp16)0);
            for_each_active([&](int k) {
                float16x8_t weight;
                if (fp16) {
                    weight = vld1q_f16(tile16 + (size_t)k * 8);
                    acc32_lo[slot] =
                        vfmaq_n_f32(acc32_lo[slot],
                                    vcvt_f32_f16(vget_low_f16(weight)), a[k]);
                    acc32_hi[slot] =
                        vfmaq_n_f32(acc32_hi[slot],
                                    vcvt_f32_f16(vget_high_f16(weight)), a[k]);
                } else {
                    int group = std::min(k / (int)B.group_size, gpr - 1);
                    if (group != last_group) {
                        if (w4 && B.q4_g128_data) {
                            const auto* blocks =
                                reinterpret_cast<const Q4B8G128Block*>(
                                    B.q4_g128_data);
                            const float* st =
                                blocks[(size_t)(n / 8) * gpr + group].scales;
                            scale =
                                vcombine_f16(vcvt_f16_f32(vld1q_f32(st)),
                                             vcvt_f16_f32(vld1q_f32(st + 4)));
                        } else if (gpr == 1 && valid == 8) {
                            const float* st = B.scales + n;
                            scale =
                                vcombine_f16(vcvt_f16_f32(vld1q_f32(st)),
                                             vcvt_f16_f32(vld1q_f32(st + 4)));
                        } else {
                            float st[8] = {};
                            for (int j = 0; j < valid; ++j)
                                st[j] = B.scales[(size_t)(n + j) * gpr + group];
                            scale =
                                vcombine_f16(vcvt_f16_f32(vld1q_f32(st)),
                                             vcvt_f16_f32(vld1q_f32(st + 4)));
                        }
                        last_group = group;
                    }
                    weight =
                        vcvtq_f16_s16(vmovl_s8(vld1_s8(tile8 + (size_t)k * 8)));
                    weight = vmulq_f16(weight, scale);
                    acc[slot] = vfmaq_n_f16(acc[slot], weight, (__fp16)a[k]);
                }
                slot = (slot + 1) & 3;
            });
            float tmp[8];
            if (fp16) {
                float32x4_t total_lo =
                    vaddq_f32(vaddq_f32(acc32_lo[0], acc32_lo[1]),
                              vaddq_f32(acc32_lo[2], acc32_lo[3]));
                float32x4_t total_hi =
                    vaddq_f32(vaddq_f32(acc32_hi[0], acc32_hi[1]),
                              vaddq_f32(acc32_hi[2], acc32_hi[3]));
                vst1q_f32(tmp, total_lo);
                vst1q_f32(tmp + 4, total_hi);
            } else {
                float16x8_t total =
                    vaddq_f16(vaddq_f16(acc[0], acc[1]),
                              vaddq_f16(acc[2], acc[3]));
                vst1q_f32(tmp, vcvt_f32_f16(vget_low_f16(total)));
                vst1q_f32(tmp + 4, vcvt_f32_f16(vget_high_f16(total)));
            }
            for (int j = 0; j < valid; ++j)
                c[n + j] = tmp[j];
        }
    };

    int threads = thread_pool ? thread_pool->num_threads() : 1;
    if (threads > 1 && N >= 64) {
        int chunk = std::max(64, ((N + threads * 8 - 1) / (threads * 8)) * 8);
        thread_pool->parallel_for(0, N, chunk, [&](int, int begin, int end) {
            run_range(begin, end);
        });
    } else {
        run_range(0, N);
    }
#else
    kernel_matmul_fp32(A, B, C, thread_pool);
#endif
}
