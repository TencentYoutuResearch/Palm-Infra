#include "kernels/x86_avx2.h"
#include "kernels/matmul_internal.h"

#include <algorithm>
#include <immintrin.h>

namespace mollm::cpu::x86 {
namespace {

inline float horizontal_sum(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

struct Q4Vectors {
    __m256 values[4];
};

inline Q4Vectors unpack_q4_block32(const uint8_t* packed) {
    const __m128i bytes =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i sign_bit = _mm_set1_epi8(0x08);
    const __m128i low = _mm_and_si128(bytes, nibble_mask);
    const __m128i high = _mm_and_si128(
        _mm_srli_epi16(bytes, 4), nibble_mask);
    __m128i q0 = _mm_unpacklo_epi8(low, high);
    __m128i q1 = _mm_unpackhi_epi8(low, high);
    q0 = _mm_sub_epi8(_mm_xor_si128(q0, sign_bit), sign_bit);
    q1 = _mm_sub_epi8(_mm_xor_si128(q1, sign_bit), sign_bit);

    Q4Vectors result;
    result.values[0] =
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0));
    result.values[1] = _mm256_cvtepi32_ps(
        _mm256_cvtepi8_epi32(_mm_srli_si128(q0, 8)));
    result.values[2] =
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1));
    result.values[3] = _mm256_cvtepi32_ps(
        _mm256_cvtepi8_epi32(_mm_srli_si128(q1, 8)));
    return result;
}

inline __m256 dot_q4_block32(const float* activation,
                             const Q4Vectors& weights) {
    __m256 acc = _mm256_mul_ps(_mm256_loadu_ps(activation),
                               weights.values[0]);
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(activation + 8),
                           weights.values[1], acc);
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(activation + 16),
                           weights.values[2], acc);
    return _mm256_fmadd_ps(_mm256_loadu_ps(activation + 24),
                            weights.values[3], acc);
}

}  // namespace

void matmul_fp32_avx2_range(const float* A, const float* B, float* C, int N,
                            int K, int lda, int K_weight, int ldc,
                            int m_begin, int m_end) {
    for (int m = m_begin; m < m_end; ++m) {
        const float* a_row = A + static_cast<size_t>(m) * lda;
        float* c_row = C + static_cast<size_t>(m) * ldc;
        for (int n = 0; n < N; ++n) {
            const float* b_row = B + static_cast<size_t>(n) * K_weight;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            int k = 0;
            for (; k + 15 < K; k += 16) {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + k),
                                       _mm256_loadu_ps(b_row + k), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + k + 8),
                                       _mm256_loadu_ps(b_row + k + 8), acc1);
            }
            __m256 acc = _mm256_add_ps(acc0, acc1);
            for (; k + 7 < K; k += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + k),
                                      _mm256_loadu_ps(b_row + k), acc);
            }
            float sum = horizontal_sum(acc);
            for (; k < K; ++k)
                sum += a_row[k] * b_row[k];
            c_row[n] = sum;
        }
    }
}

void matmul_fp16_avx2_range(const float* A, const fp16_t* B, float* C, int N,
                            int K, int lda, int K_weight, int ldc,
                            int m_begin, int m_end) {
    for (int m = m_begin; m < m_end; ++m) {
        const float* a_row = A + static_cast<size_t>(m) * lda;
        float* c_row = C + static_cast<size_t>(m) * ldc;
        for (int n = 0; n < N; ++n) {
            const fp16_t* b_row = B + static_cast<size_t>(n) * K_weight;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            int k = 0;
            for (; k + 15 < K; k += 16) {
                const __m128i b0 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(b_row + k));
                const __m128i b1 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(b_row + k + 8));
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + k),
                                       _mm256_cvtph_ps(b0), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + k + 8),
                                       _mm256_cvtph_ps(b1), acc1);
            }
            __m256 acc = _mm256_add_ps(acc0, acc1);
            for (; k + 7 < K; k += 8) {
                const __m128i packed = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(b_row + k));
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + k),
                                      _mm256_cvtph_ps(packed), acc);
            }
            float sum = horizontal_sum(acc);
            for (; k < K; ++k)
                sum += a_row[k] * static_cast<float>(b_row[k]);
            c_row[n] = sum;
        }
    }
}

void matmul_fp16_m2_avx2_range_n(const float* A, const fp16_t* B, float* C,
                                  int N, int K, int lda, int K_weight, int ldc,
                                  int n_begin, int n_end) {
    (void)N;
    const float* a_row0 = A;
    const float* a_row1 = A + lda;
    float* c_row0 = C;
    float* c_row1 = C + ldc;
    for (int n = n_begin; n < n_end; ++n) {
        const fp16_t* b_row = B + static_cast<size_t>(n) * K_weight;
        __m256 acc00 = _mm256_setzero_ps();
        __m256 acc01 = _mm256_setzero_ps();
        __m256 acc10 = _mm256_setzero_ps();
        __m256 acc11 = _mm256_setzero_ps();
        int k = 0;
        for (; k + 15 < K; k += 16) {
            const __m128i b0 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(b_row + k));
            const __m128i b1 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(b_row + k + 8));
            const __m256 weights0 = _mm256_cvtph_ps(b0);
            const __m256 weights1 = _mm256_cvtph_ps(b1);
            acc00 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row0 + k), weights0,
                                    acc00);
            acc01 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row0 + k + 8), weights1,
                                    acc01);
            acc10 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row1 + k), weights0,
                                    acc10);
            acc11 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row1 + k + 8), weights1,
                                    acc11);
        }
        __m256 acc0 = _mm256_add_ps(acc00, acc01);
        __m256 acc1 = _mm256_add_ps(acc10, acc11);
        for (; k + 7 < K; k += 8) {
            const __m128i packed = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(b_row + k));
            const __m256 weights = _mm256_cvtph_ps(packed);
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row0 + k), weights, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row1 + k), weights, acc1);
        }
        float sum0 = horizontal_sum(acc0);
        float sum1 = horizontal_sum(acc1);
        for (; k < K; ++k) {
            const float weight = static_cast<float>(b_row[k]);
            sum0 += a_row0[k] * weight;
            sum1 += a_row1[k] * weight;
        }
        c_row0[n] = sum0;
        c_row1[n] = sum1;
    }
}

void matmul_int4_bg_avx2_range(const Tensor& A, const Tensor& B, Tensor& C,
                               int lda, int ldc, int m_begin, int m_end,
                               int n_begin, int n_end) {
    const int K = static_cast<int>(A.shape[0]);
    const int groups = static_cast<int>(B.groups_per_row);
    const auto* bg32 = static_cast<const Q4B8G32Block*>(B.q4_g32_data);
    const auto* bg128 = static_cast<const Q4B8G128Block*>(B.q4_g128_data);
    const bool is_bg32 =
        B.is_q4_g32_packed && bg32 && B.group_size == 32;
    const bool is_bg128 =
        B.is_q4_g128_packed && bg128 && B.group_size == 128;

    if (m_end - m_begin == 1) {
        const float* a_row =
            A.ptr<float>() + static_cast<size_t>(m_begin) * lda;
        float* out =
            C.ptr<float>() + static_cast<size_t>(m_begin) * ldc;
        for (int n = n_begin; n < n_end; ++n) {
            float sum = 0.0f;
            if (is_bg32) {
                for (int group = 0; group < groups; ++group) {
                    const auto& block = bg32[
                        static_cast<size_t>(n / 8) * groups + group];
                    sum += horizontal_sum(dot_q4_block32(
                               a_row + static_cast<size_t>(group) * 32,
                               unpack_q4_block32(block.q[n & 7]))) *
                           block.scales[n & 7];
                }
            } else {
                for (int group = 0; group < groups; ++group) {
                    const auto& block = bg128[
                        static_cast<size_t>(n / 8) * groups + group];
                    __m256 acc = _mm256_setzero_ps();
                    for (int sub = 0; sub < 4; ++sub) {
                        acc = _mm256_add_ps(
                            acc, dot_q4_block32(
                                     a_row +
                                         static_cast<size_t>(group) * 128 +
                                         sub * 32,
                                     unpack_q4_block32(
                                         block.q[sub][n & 7])));
                    }
                    sum += horizontal_sum(acc) * block.scales[n & 7];
                }
            }
            out[n] = sum;
        }
        return;
    }

    constexpr int M_TILE = 4;
    for (int m_base = m_begin; m_base < m_end; m_base += M_TILE) {
        const int m_count = std::min(M_TILE, m_end - m_base);
        for (int n = n_begin; n < n_end; ++n) {
            float sums[M_TILE] = {};
            if (is_bg32) {
                for (int group = 0; group < groups; ++group) {
                    const auto& block = bg32[
                        static_cast<size_t>(n / 8) * groups + group];
                    const Q4Vectors weights =
                        unpack_q4_block32(block.q[n & 7]);
                    const float scale = block.scales[n & 7];
                    for (int mi = 0; mi < m_count; ++mi) {
                        const float* activation =
                            A.ptr<float>() +
                            static_cast<size_t>(m_base + mi) * lda +
                            static_cast<size_t>(group) * 32;
                        sums[mi] +=
                            horizontal_sum(
                                dot_q4_block32(activation, weights)) *
                            scale;
                    }
                }
            } else if (is_bg128) {
                for (int group = 0; group < groups; ++group) {
                    const auto& block = bg128[
                        static_cast<size_t>(n / 8) * groups + group];
                    __m256 acc[M_TILE] = {
                        _mm256_setzero_ps(), _mm256_setzero_ps(),
                        _mm256_setzero_ps(), _mm256_setzero_ps()};
                    for (int sub = 0; sub < 4; ++sub) {
                        const Q4Vectors weights =
                            unpack_q4_block32(block.q[sub][n & 7]);
                        for (int mi = 0; mi < m_count; ++mi) {
                            const float* activation =
                                A.ptr<float>() +
                                static_cast<size_t>(m_base + mi) * lda +
                                static_cast<size_t>(group) * 128 + sub * 32;
                            acc[mi] = _mm256_add_ps(
                                acc[mi],
                                dot_q4_block32(activation, weights));
                        }
                    }
                    for (int mi = 0; mi < m_count; ++mi)
                        sums[mi] += horizontal_sum(acc[mi]) *
                                    block.scales[n & 7];
                }
            }
            for (int mi = 0; mi < m_count; ++mi) {
                C.ptr<float>()[static_cast<size_t>(m_base + mi) * ldc + n] =
                    sums[mi];
            }
        }
    }
}

void matmul_int8_avx2_range(const float* A, const int8_t* B,
                            const float* scales, float* C, int N, int K,
                            int group_size, int groups_per_row, int lda,
                            int K_weight, int ldc, int m_begin, int m_end,
                            int n_begin, int n_end) {
    (void)N;
    for (int m = m_begin; m < m_end; ++m) {
        const float* a_row = A + static_cast<size_t>(m) * lda;
        float* c_row = C + static_cast<size_t>(m) * ldc;
        for (int n = n_begin; n < n_end; ++n) {
            const int8_t* b_row = B + static_cast<size_t>(n) * K_weight;
            const float* scale_row =
                scales + static_cast<size_t>(n) * groups_per_row;
            float sum = 0.0f;
            for (int group = 0; group < groups_per_row; ++group) {
                const int k_begin = group * group_size;
                const int k_end = std::min(k_begin + group_size, K);
                __m256 acc = _mm256_setzero_ps();
                int k = k_begin;
                for (; k + 7 < k_end; k += 8) {
                    const __m128i q = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(b_row + k));
                    const __m256 qf = _mm256_cvtepi32_ps(
                        _mm256_cvtepi8_epi32(q));
                    acc = _mm256_fmadd_ps(
                        _mm256_loadu_ps(a_row + k), qf, acc);
                }
                float group_sum = horizontal_sum(acc);
                for (; k < k_end; ++k)
                    group_sum += a_row[k] * static_cast<float>(b_row[k]);
                sum += group_sum * scale_row[group];
            }
            c_row[n] = sum;
        }
    }
}

}  // namespace mollm::cpu::x86
