#include "kernels/cpu/x86/x86_vnni.h"

#include "kernels/cpu/matmul/matmul_internal.h"

#include <algorithm>
#include <cstring>
#include <immintrin.h>

namespace mollm::cpu::x86 {
namespace {

inline int horizontal_sum_i8(__m256i values) {
    const __m256i pair_sums =
        _mm256_maddubs_epi16(_mm256_set1_epi8(1), values);
    const __m256i quad_sums =
        _mm256_madd_epi16(pair_sums, _mm256_set1_epi16(1));
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(quad_sums),
                                _mm256_extracti128_si256(quad_sums, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

inline __m256i unpack_q4_plus8(const uint8_t* packed) {
    const __m128i bytes =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i low = _mm_and_si128(bytes, mask);
    const __m128i high =
        _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
    return _mm256_set_m128i(_mm_unpackhi_epi8(low, high),
                            _mm_unpacklo_epi8(low, high));
}

}  // namespace

void quantize_q8_vnni_range(const float* A, int8_t* qA, float* scales,
                            int16_t* sums, int K, int lda, int m_begin,
                            int m_end) {
    const int groups = K / 32;
    const __m512i abs_mask = _mm512_set1_epi32(0x7fffffff);
    for (int m = m_begin; m < m_end; ++m) {
        const float* input = A + static_cast<size_t>(m) * lda;
        int8_t* output = qA + static_cast<size_t>(m) * K;
        float* row_scales = scales + static_cast<size_t>(m) * groups;
        int16_t* row_sums = sums + static_cast<size_t>(m) * groups;
        for (int group = 0; group < groups; ++group) {
            const float* block = input + group * 32;
            const __m512 first = _mm512_loadu_ps(block);
            const __m512 second = _mm512_loadu_ps(block + 16);
            const __m512 abs_first = _mm512_castsi512_ps(_mm512_and_epi32(
                _mm512_castps_si512(first), abs_mask));
            const __m512 abs_second = _mm512_castsi512_ps(_mm512_and_epi32(
                _mm512_castps_si512(second), abs_mask));
            const float amax =
                _mm512_reduce_max_ps(_mm512_max_ps(abs_first, abs_second));
            const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
            const float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
            row_scales[group] = scale;
            const __m512 multiplier = _mm512_set1_ps(inverse);
            const __m128i q_first = _mm512_cvtsepi32_epi8(
                _mm512_cvtps_epi32(_mm512_mul_ps(first, multiplier)));
            const __m128i q_second = _mm512_cvtsepi32_epi8(
                _mm512_cvtps_epi32(_mm512_mul_ps(second, multiplier)));
            const __m256i quantized =
                _mm256_set_m128i(q_second, q_first);
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(output + group * 32), quantized);
            row_sums[group] =
                static_cast<int16_t>(horizontal_sum_i8(quantized));
        }
    }
}

void matmul_int4_bg32_vnni_range(
    const int8_t* qA, const float* a_scales, const int16_t* a_sums,
    const void* vnni_data, const Tensor& B, Tensor& C, int K, int ldc,
    int m_begin, int m_end, int n_begin, int n_end) {
    const int groups = K / 32;
    const auto* source_blocks =
        static_cast<const Q4B8G32Block*>(B.q4_g32_data);
    const auto* vnni_blocks =
        static_cast<const Q4B8G32VnniBlock*>(vnni_data);
    constexpr int M_TILE = 4;

    for (int m_base = m_begin; m_base < m_end; m_base += M_TILE) {
        const int m_count = std::min(M_TILE, m_end - m_base);
        for (int n_base = n_begin; n_base < n_end; n_base += 8) {
            const int n_count = std::min(8, n_end - n_base);
            __m256 sums[M_TILE] = {
                _mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps(), _mm256_setzero_ps()};

            for (int group = 0; group < groups; ++group) {
                const size_t block_index =
                    static_cast<size_t>(n_base / 8) * groups + group;
                const auto& source = source_blocks[block_index];
                const auto& packed = vnni_blocks[block_index];
                __m256i weights[8];
                for (int chunk = 0; chunk < 8; ++chunk)
                    weights[chunk] =
                        unpack_q4_plus8(packed.q[chunk]);

                for (int mi = 0; mi < m_count; ++mi) {
                    const int m = m_base + mi;
                    const size_t scale_index =
                        static_cast<size_t>(m) * groups + group;
                    const int8_t* activations =
                        qA + static_cast<size_t>(m) * K + group * 32;
                    __m256i dots =
                        _mm256_set1_epi32(-8 * a_sums[scale_index]);
                    for (int chunk = 0; chunk < 8; ++chunk) {
                        int32_t activation_word;
                        std::memcpy(&activation_word,
                                    activations + chunk * 4,
                                    sizeof(activation_word));
                        dots = _mm256_dpbusd_epi32(
                            dots, weights[chunk],
                            _mm256_set1_epi32(activation_word));
                    }
                    const __m256 values = _mm256_cvtepi32_ps(dots);
                    const __m256 scales = _mm256_mul_ps(
                        _mm256_loadu_ps(source.scales),
                        _mm256_set1_ps(a_scales[scale_index]));
                    sums[mi] =
                        _mm256_fmadd_ps(values, scales, sums[mi]);
                }
            }

            for (int mi = 0; mi < m_count; ++mi) {
                float* output =
                    C.ptr<float>() +
                    static_cast<size_t>(m_base + mi) * ldc + n_base;
                if (n_count == 8) {
                    _mm256_storeu_ps(output, sums[mi]);
                } else {
                    alignas(32) float tail[8];
                    _mm256_store_ps(tail, sums[mi]);
                    std::copy_n(tail, n_count, output);
                }
            }
        }
    }
}

}  // namespace mollm::cpu::x86
