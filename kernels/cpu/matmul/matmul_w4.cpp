#include "kernels/cpu/matmul/matmul_internal.h"
#include "runtime/threading.h"

#include <algorithm>
#include <cstdint>
#include <vector>

static inline int8_t unpack_int4_signed(uint8_t byte, bool high_nibble) {
    int v = high_nibble ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
    if (v >= 8)
        v -= 16;
    return (int8_t)v;
}

#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
static inline void load_int4x32_signed_scaled16(const uint8_t* src,
                                                int8x16_t& even_scaled,
                                                int8x16_t& odd_scaled) {
    uint8x16_t packed = vld1q_u8(src);
    even_scaled = vreinterpretq_s8_u8(vshlq_n_u8(packed, 4));
    odd_scaled = vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0xF0)));
}

static inline int32x4_t q4_q8_dot32_acc(int32x4_t d,
                                        int8x16_t q4_even,
                                        int8x16_t q4_odd,
                                        int8x16_t qa_even,
                                        int8x16_t qa_odd) {
#if defined(__aarch64__)
    // Keep the M=1 path on SDOT even when the translation unit enables i8mm.
    // Clang otherwise recognizes the eight independent column dots as a
    // matrix multiply and rewrites them to SMMLA, whose operand duplication
    // and lane rearrangement make GEMV substantially slower.
    __asm__("sdot %0.4s, %1.16b, %2.16b"
            : "+w"(d)
            : "w"(q4_even), "w"(qa_even));
    __asm__("sdot %0.4s, %1.16b, %2.16b"
            : "+w"(d)
            : "w"(q4_odd), "w"(qa_odd));
#else
    d = vdotq_s32(d, q4_even, qa_even);
    d = vdotq_s32(d, q4_odd, qa_odd);
#endif
    return d;
}

static inline int32x4_t q4_q8_dot32(int8x16_t q4_even, int8x16_t q4_odd,
                                    int8x16_t qa_even, int8x16_t qa_odd) {
    return q4_q8_dot32_acc(vdupq_n_s32(0), q4_even, q4_odd, qa_even, qa_odd);
}

static inline float32x4_t q4_scaled16_dot_to_f32(int32x4_t dots) {
#if defined(__aarch64__)
    return vcvtq_n_f32_s32(dots, 4);
#else
    return vmulq_n_f32(vcvtq_f32_s32(dots), 1.0f / 16.0f);
#endif
}

static void matmul_int4_q8dot_neon_gemv_range(
    const int8_t* qA, const int8_t* qA_even_pre, const int8_t* qA_odd_pre,
    const float* a_scales, const uint8_t* B, const uint8_t* B_repack,
    const float* scales, int group_size, int groups_per_row, float* C, int K,
    int K_weight, int n_begin, int n_end) {
    if (group_size <= 0)
        group_size = K;
    int row_stride = (K_weight + 1) / 2;
    int blocks_per_row = K / MATMUL_Q8_BLOCK;
    constexpr int bytes_per_block = MATMUL_Q8_BLOCK / 2;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int n = n_begin; n < n_end; n += 8) {
        int n_tile_end = std::min(n + 8, n_end);
        int c_valid = n_tile_end - n;
        bool full_n_tile = (c_valid == 8);
        float32x4_t bscale_lo_pc = vdupq_n_f32(0.f);
        float32x4_t bscale_hi_pc = vdupq_n_f32(0.f);
        if (scale_per_channel) {
            load_w8_b_scales8(scales, n, c_valid, groups_per_row, 0,
                              bscale_lo_pc, bscale_hi_pc);
        }

        float32x4_t acc_lo = vdupq_n_f32(0.f);
        float32x4_t acc_hi = vdupq_n_f32(0.f);

        auto run_qblock = [&](int qb, float32x4_t bscale_lo,
                              float32x4_t bscale_hi) {
            int byte_off = qb * (MATMUL_Q8_BLOCK / 2);
            const uint8_t* b_repack_block =
                B_repack ? B_repack + ((size_t)(n / 8) * blocks_per_row + qb) *
                                          8 * bytes_per_block
                         : nullptr;

            int8x16_t qa_even;
            int8x16_t qa_odd;
            if (qA_even_pre && qA_odd_pre) {
                qa_even = vld1q_s8(qA_even_pre + (size_t)qb * 16);
                qa_odd = vld1q_s8(qA_odd_pre + (size_t)qb * 16);
            } else {
                int8x16_t qa0 = vld1q_s8(qA + (size_t)qb * MATMUL_Q8_BLOCK);
                int8x16_t qa1 =
                    vld1q_s8(qA + (size_t)qb * MATMUL_Q8_BLOCK + 16);
                qa_even = vuzp1q_s8(qa0, qa1);
                qa_odd = vuzp2q_s8(qa0, qa1);
            }

            int32x4_t d0 = vdupq_n_s32(0);
            int32x4_t d1 = vdupq_n_s32(0);
            int32x4_t d2 = vdupq_n_s32(0);
            int32x4_t d3 = vdupq_n_s32(0);
            int32x4_t d4 = vdupq_n_s32(0);
            int32x4_t d5 = vdupq_n_s32(0);
            int32x4_t d6 = vdupq_n_s32(0);
            int32x4_t d7 = vdupq_n_s32(0);

            auto dot_col = [&](int c, int32x4_t& d) {
                int8x16_t q4_even, q4_odd;
                const uint8_t* src =
                    b_repack_block
                        ? b_repack_block + (size_t)c * bytes_per_block
                        : B + (size_t)(n + c) * row_stride + byte_off;
                load_int4x32_signed_scaled16(src, q4_even, q4_odd);
                d = q4_q8_dot32(q4_even, q4_odd, qa_even, qa_odd);
            };
            if (full_n_tile) {
                dot_col(0, d0);
                dot_col(1, d1);
                dot_col(2, d2);
                dot_col(3, d3);
                dot_col(4, d4);
                dot_col(5, d5);
                dot_col(6, d6);
                dot_col(7, d7);
            } else {
                if (c_valid > 0)
                    dot_col(0, d0);
                if (c_valid > 1)
                    dot_col(1, d1);
                if (c_valid > 2)
                    dot_col(2, d2);
                if (c_valid > 3)
                    dot_col(3, d3);
                if (c_valid > 4)
                    dot_col(4, d4);
                if (c_valid > 5)
                    dot_col(5, d5);
                if (c_valid > 6)
                    dot_col(6, d6);
                if (c_valid > 7)
                    dot_col(7, d7);
            }

            int32x4_t p01 = vpaddq_s32(d0, d1);
            int32x4_t p23 = vpaddq_s32(d2, d3);
            int32x4_t p45 = vpaddq_s32(d4, d5);
            int32x4_t p67 = vpaddq_s32(d6, d7);
            int32x4_t dots_lo = vpaddq_s32(p01, p23);
            int32x4_t dots_hi = vpaddq_s32(p45, p67);

            float a_scale = a_scales[qb];
            acc_lo = vfmaq_f32(acc_lo, q4_scaled16_dot_to_f32(dots_lo),
                               vmulq_n_f32(bscale_lo, a_scale));
            acc_hi = vfmaq_f32(acc_hi, q4_scaled16_dot_to_f32(dots_hi),
                               vmulq_n_f32(bscale_hi, a_scale));
        };

        if (scale_per_channel) {
            for (int qb = 0; qb < blocks_per_row; qb++) {
                run_qblock(qb, bscale_lo_pc, bscale_hi_pc);
            }
        } else if (scale_mode == W8ScaleMode::PerGroup) {
            int qblocks_per_group = std::max(1, group_size / MATMUL_Q8_BLOCK);
            for (int group = 0; group < groups_per_row; group++) {
                float32x4_t bscale_lo;
                float32x4_t bscale_hi;
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, group,
                                  bscale_lo, bscale_hi);
                int qb_begin = group * qblocks_per_group;
                int qb_end =
                    std::min(qb_begin + qblocks_per_group, blocks_per_row);
                for (int qb = qb_begin; qb < qb_end; qb++) {
                    run_qblock(qb, bscale_lo, bscale_hi);
                }
            }
        } else {
            for (int qb = 0; qb < blocks_per_row; qb++) {
                float32x4_t bscale_lo;
                float32x4_t bscale_hi;
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, qb,
                                  bscale_lo, bscale_hi);
                run_qblock(qb, bscale_lo, bscale_hi);
            }
        }

        if (full_n_tile) {
            vst1q_f32(C + n, acc_lo);
            vst1q_f32(C + n + 4, acc_hi);
        } else {
            float tmp[4];
            vst1q_f32(tmp, acc_lo);
            for (int c = 0; c < 4 && c < c_valid; c++)
                C[n + c] = tmp[c];
            vst1q_f32(tmp, acc_hi);
            for (int c = 0; c < 4 && c + 4 < c_valid; c++)
                C[n + 4 + c] = tmp[c];
        }
    }
}

static void matmul_int4_q8dot_neon_gemv_g128_range(
    const int8_t* qA_even_pre, const int8_t* qA_odd_pre, const float* a_scales,
    const Q4B8G128Block* B_g128, float* C, int K, int n_begin, int n_end) {
    int g128_per_row = K / 128;

    for (int n = n_begin; n < n_end; n += 8) {
        int n_tile_end = std::min(n + 8, n_end);
        int c_valid = n_tile_end - n;

        float32x4_t acc_lo = vdupq_n_f32(0.f);
        float32x4_t acc_hi = vdupq_n_f32(0.f);
        const Q4B8G128Block* b_tile = B_g128 + (size_t)(n / 8) * g128_per_row;

        for (int g = 0; g < g128_per_row; g++) {
            const Q4B8G128Block& b_group = b_tile[g];
            float32x4_t bscale_lo = vld1q_f32(b_group.scales);
            float32x4_t bscale_hi = vld1q_f32(b_group.scales + 4);

            // A and B now share one scale for the complete 128-value group.
            // Accumulate its four dot blocks in integer space so horizontal
            // reduction, conversion, and scale application happen only once.
            int32x4_t d0 = vdupq_n_s32(0);
            int32x4_t d1 = vdupq_n_s32(0);
            int32x4_t d2 = vdupq_n_s32(0);
            int32x4_t d3 = vdupq_n_s32(0);
            int32x4_t d4 = vdupq_n_s32(0);
            int32x4_t d5 = vdupq_n_s32(0);
            int32x4_t d6 = vdupq_n_s32(0);
            int32x4_t d7 = vdupq_n_s32(0);

            for (int qgi = 0; qgi < 4; qgi++) {
                int qb = g * 4 + qgi;
                int8x16_t qa_even = vld1q_s8(qA_even_pre + (size_t)qb * 16);
                int8x16_t qa_odd = vld1q_s8(qA_odd_pre + (size_t)qb * 16);

                auto dot_col = [&](int c, int32x4_t& d) {
                    int8x16_t q4_even;
                    int8x16_t q4_odd;
                    load_int4x32_signed_scaled16(b_group.q[qgi][c], q4_even,
                                                 q4_odd);
                    d = q4_q8_dot32_acc(d, q4_even, q4_odd, qa_even, qa_odd);
                };

                if (c_valid > 0)
                    dot_col(0, d0);
                if (c_valid > 1)
                    dot_col(1, d1);
                if (c_valid > 2)
                    dot_col(2, d2);
                if (c_valid > 3)
                    dot_col(3, d3);
                if (c_valid > 4)
                    dot_col(4, d4);
                if (c_valid > 5)
                    dot_col(5, d5);
                if (c_valid > 6)
                    dot_col(6, d6);
                if (c_valid > 7)
                    dot_col(7, d7);

            }

            int32x4_t p01 = vpaddq_s32(d0, d1);
            int32x4_t p23 = vpaddq_s32(d2, d3);
            int32x4_t p45 = vpaddq_s32(d4, d5);
            int32x4_t p67 = vpaddq_s32(d6, d7);
            int32x4_t dots_lo = vpaddq_s32(p01, p23);
            int32x4_t dots_hi = vpaddq_s32(p45, p67);

            float a_scale = a_scales[g];
            acc_lo = vfmaq_f32(acc_lo, q4_scaled16_dot_to_f32(dots_lo),
                               vmulq_n_f32(bscale_lo, a_scale));
            acc_hi = vfmaq_f32(acc_hi, q4_scaled16_dot_to_f32(dots_hi),
                               vmulq_n_f32(bscale_hi, a_scale));
        }

        float tmp[4];
        vst1q_f32(tmp, acc_lo);
        for (int c = 0; c < 4 && c < c_valid; c++)
            C[n + c] = tmp[c];
        vst1q_f32(tmp, acc_hi);
        for (int c = 0; c < 4 && c + 4 < c_valid; c++)
            C[n + 4 + c] = tmp[c];
    }
}

static void matmul_int4_q8dot_neon_gemv_g32_range(
    const int8_t* qA_even_pre, const int8_t* qA_odd_pre,
    const float* a_scales, const Q4B8G32Block* B_g32, float* C, int K,
    int n_begin, int n_end) {
    int blocks_per_row = K / MATMUL_Q8_BLOCK;

    for (int n = n_begin; n < n_end; n += 8) {
        int c_valid = std::min(n + 8, n_end) - n;
        const Q4B8G32Block* b_tile =
            B_g32 + (size_t)(n / 8) * blocks_per_row;
        float32x4_t acc_lo = vdupq_n_f32(0.f);
        float32x4_t acc_hi = vdupq_n_f32(0.f);

        for (int qb = 0; qb < blocks_per_row; qb++) {
            const Q4B8G32Block& block = b_tile[qb];
            int8x16_t qa_even =
                vld1q_s8(qA_even_pre + (size_t)qb * 16);
            int8x16_t qa_odd =
                vld1q_s8(qA_odd_pre + (size_t)qb * 16);
            auto dot_col = [&](int c) {
                int8x16_t q4_even;
                int8x16_t q4_odd;
                load_int4x32_signed_scaled16(block.q[c], q4_even, q4_odd);
                return q4_q8_dot32(q4_even, q4_odd, qa_even, qa_odd);
            };
            int32x4_t d0 = vdupq_n_s32(0);
            int32x4_t d1 = vdupq_n_s32(0);
            int32x4_t d2 = vdupq_n_s32(0);
            int32x4_t d3 = vdupq_n_s32(0);
            int32x4_t d4 = vdupq_n_s32(0);
            int32x4_t d5 = vdupq_n_s32(0);
            int32x4_t d6 = vdupq_n_s32(0);
            int32x4_t d7 = vdupq_n_s32(0);
            if (c_valid == 8) {
                d0 = dot_col(0);
                d1 = dot_col(1);
                d2 = dot_col(2);
                d3 = dot_col(3);
                d4 = dot_col(4);
                d5 = dot_col(5);
                d6 = dot_col(6);
                d7 = dot_col(7);
            } else {
                if (c_valid > 0) d0 = dot_col(0);
                if (c_valid > 1) d1 = dot_col(1);
                if (c_valid > 2) d2 = dot_col(2);
                if (c_valid > 3) d3 = dot_col(3);
                if (c_valid > 4) d4 = dot_col(4);
                if (c_valid > 5) d5 = dot_col(5);
                if (c_valid > 6) d6 = dot_col(6);
            }

            int32x4_t p01 = vpaddq_s32(d0, d1);
            int32x4_t p23 = vpaddq_s32(d2, d3);
            int32x4_t p45 = vpaddq_s32(d4, d5);
            int32x4_t p67 = vpaddq_s32(d6, d7);
            int32x4_t dots_lo = vpaddq_s32(p01, p23);
            int32x4_t dots_hi = vpaddq_s32(p45, p67);
            float32x4_t bscale_lo = vld1q_f32(block.scales);
            float32x4_t bscale_hi = vld1q_f32(block.scales + 4);
            float a_scale = a_scales[qb];
            acc_lo = vfmaq_f32(acc_lo,
                               q4_scaled16_dot_to_f32(dots_lo),
                               vmulq_n_f32(bscale_lo, a_scale));
            acc_hi = vfmaq_f32(acc_hi,
                               q4_scaled16_dot_to_f32(dots_hi),
                               vmulq_n_f32(bscale_hi, a_scale));
        }

        if (c_valid == 8) {
            vst1q_f32(C + n, acc_lo);
            vst1q_f32(C + n + 4, acc_hi);
        } else {
            float tmp[4];
            vst1q_f32(tmp, acc_lo);
            for (int c = 0; c < 4 && c < c_valid; c++)
                C[n + c] = tmp[c];
            vst1q_f32(tmp, acc_hi);
            for (int c = 0; c < 4 && c + 4 < c_valid; c++)
                C[n + 4 + c] = tmp[c];
        }
    }
}

bool kernel_matmul_int4_gemv_batch(const std::vector<Tensor>& inputs,
                                   const std::vector<Tensor>& weights,
                                   std::vector<Tensor>& outputs,
                                   ThreadPool* thread_pool) {
    const size_t batch = inputs.size();
    if (batch < 2 || weights.size() != batch || outputs.size() != batch ||
        !thread_pool || thread_pool->num_threads() < 2) {
        return false;
    }

    const int K = (int)inputs[0].shape[0];
    const int N = (int)weights[0].shape[0];
    if (K <= 0 || N <= 0 || K % 32 != 0) return false;

    struct BatchScratch {
        std::vector<Q4GemvScratch> quantized_inputs;
        std::vector<size_t> input_indices;
        std::vector<const Q4B8G32Block*> packed_g32_weights;
        std::vector<const Q4B8G128Block*> packed_g128_weights;
        std::vector<float*> output_data;
    };
    static thread_local BatchScratch batch_scratch;
    batch_scratch.quantized_inputs.resize(batch);
    batch_scratch.input_indices.resize(batch);
    batch_scratch.packed_g32_weights.resize(batch);
    batch_scratch.packed_g128_weights.resize(batch);
    batch_scratch.output_data.resize(batch);
    const int group_size = (int)weights[0].group_size;
    if (group_size != 32 && group_size != 128)
        return false;
    for (size_t i = 0; i < batch; ++i) {
        const Tensor& input = inputs[i];
        const Tensor& weight = weights[i];
        Tensor& output = outputs[i];
        if (input.prec != Precision::FP32 || input.shape[0] != K ||
            input.shape[1] != 1 || weight.prec != Precision::INT4 ||
            weight.shape[0] != N || weight.shape[1] != K ||
            (int)weight.group_size != group_size ||
            (group_size == 32 && !weight.q4_g32_data) ||
            (group_size == 128 &&
             (!weight.is_q4_g128_packed || !weight.q4_g128_data)) ||
            output.prec != Precision::FP32 ||
            output.shape[0] != N || output.shape[1] != 1) {
            return false;
        }
        batch_scratch.packed_g32_weights[i] =
            reinterpret_cast<const Q4B8G32Block*>(weight.q4_g32_data);
        batch_scratch.packed_g128_weights[i] =
            reinterpret_cast<const Q4B8G128Block*>(weight.q4_g128_data);
        batch_scratch.output_data[i] = output.ptr<float>();
    }

    MatmulTimer timer;
    timer.set_shape(group_size == 128 ? "q4dot_gemv_bg128_batch"
                                      : "q4dot_gemv_bg32_batch",
                    (int)batch, N, K, group_size, K / group_size, true, false,
                    thread_pool->num_threads());
    for (size_t i = 0; i < batch; ++i) {
        const Tensor& input = inputs[i];
        size_t input_index = i;
        for (size_t previous = 0; previous < i; ++previous) {
            if (inputs[previous].data == input.data &&
                inputs[previous].stride[1] == input.stride[1]) {
                input_index = batch_scratch.input_indices[previous];
                break;
            }
        }
        batch_scratch.input_indices[i] = input_index;
        if (input_index == i) {
            Q4GemvScratch& quantized =
                batch_scratch.quantized_inputs[input_index];
            if (group_size == 128) {
                quantize_a_q8_g128_even_odd(
                    input.ptr<float>(), K, quantized.qA_even,
                    quantized.qA_odd, quantized.a_scales);
            } else {
                quantize_a_q8_blocks_even_odd(
                    input.ptr<float>(), K, quantized.qA_even,
                    quantized.qA_odd, quantized.a_scales);
            }
        }
    }

    int n_chunk = std::max(N / (thread_pool->num_threads() * 8), 64);
    n_chunk = ((n_chunk + 7) / 8) * 8;
    const Q4GemvScratch* quantized_inputs =
        batch_scratch.quantized_inputs.data();
    const size_t* input_indices = batch_scratch.input_indices.data();
    const Q4B8G32Block* const* packed_g32_weights =
        batch_scratch.packed_g32_weights.data();
    const Q4B8G128Block* const* packed_g128_weights =
        batch_scratch.packed_g128_weights.data();
    float* const* output_data = batch_scratch.output_data.data();
    thread_pool->parallel_for(
        0, N, n_chunk, [&](int, int n_begin, int n_end) {
            for (size_t i = 0; i < batch; ++i) {
                const Q4GemvScratch& quantized =
                    quantized_inputs[input_indices[i]];
                if (group_size == 128) {
                    matmul_int4_q8dot_neon_gemv_g128_range(
                        quantized.qA_even.data(), quantized.qA_odd.data(),
                        quantized.a_scales.data(), packed_g128_weights[i],
                        output_data[i], K, n_begin, n_end);
                } else {
                    matmul_int4_q8dot_neon_gemv_g32_range(
                        quantized.qA_even.data(), quantized.qA_odd.data(),
                        quantized.a_scales.data(), packed_g32_weights[i],
                        output_data[i], K, n_begin, n_end);
                }
            }
        });
    return true;
}

static void matmul_int4_q8dot_neon_4x8_range(
    const int8_t* qA, const float* a_scales, const uint8_t* B,
    const uint8_t* B_repack, const float* scales, int group_size,
    int groups_per_row, float* C, int M, int N, int K, int K_padded,
    int K_weight, int ldc, int m_begin, int m_end) {
    (void)M;
    if (group_size <= 0)
        group_size = K;
    int row_stride = (K_weight + 1) / 2;
    int blocks_per_row = K / MATMUL_Q8_BLOCK;
    constexpr int bytes_per_block = MATMUL_Q8_BLOCK / 2;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int m = m_begin; m < m_end; m += 4) {
        int m_tile_end = std::min(m + 4, m_end);
        int r_valid = m_tile_end - m;

        for (int n = 0; n < N; n += 8) {
            int n_tile_end = std::min(n + 8, N);
            int c_valid = n_tile_end - n;
            bool full_n_tile = (c_valid == 8);
            float32x4_t bscale_lo_pc = vdupq_n_f32(0.f);
            float32x4_t bscale_hi_pc = vdupq_n_f32(0.f);
            if (scale_per_channel) {
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, 0,
                                  bscale_lo_pc, bscale_hi_pc);
            }

            float acc[4][8] = {};

            for (int qb = 0; qb < blocks_per_row; qb++) {
                int group = w8_scale_group(scale_mode, qb, group_size);
                int byte_off = qb * (MATMUL_Q8_BLOCK / 2);
                const uint8_t* b_repack_block =
                    B_repack
                        ? B_repack + ((size_t)(n / 8) * blocks_per_row + qb) *
                                         8 * bytes_per_block
                        : nullptr;

                int8x16_t q4_even[8];
                int8x16_t q4_odd[8];
                int c_load_end = full_n_tile ? 8 : c_valid;
                for (int c = 0; c < c_load_end; c++) {
                    const uint8_t* src =
                        b_repack_block
                            ? b_repack_block + (size_t)c * bytes_per_block
                            : B + (size_t)(n + c) * row_stride + byte_off;
                    load_int4x32_signed_scaled16(src, q4_even[c], q4_odd[c]);
                }

                int32_t dots[4][8] = {};
                for (int r = 0; r < r_valid; r++) {
                    const int8_t* qa =
                        qA + (size_t)(m + r) * K_padded + qb * MATMUL_Q8_BLOCK;
                    int8x16_t qa0 = vld1q_s8(qa);
                    int8x16_t qa1 = vld1q_s8(qa + 16);
                    int8x16_t qa_even = vuzp1q_s8(qa0, qa1);
                    int8x16_t qa_odd = vuzp2q_s8(qa0, qa1);
                    for (int c = 0; c < c_valid; c++) {
                        int32x4_t d =
                            q4_q8_dot32(q4_even[c], q4_odd[c], qa_even, qa_odd);
                        dots[r][c] = vaddvq_s32(d);
                    }
                }

                float32x4_t bscale_lo = bscale_lo_pc;
                float32x4_t bscale_hi = bscale_hi_pc;
                if (!scale_per_channel) {
                    load_w8_b_scales8(scales, n, c_valid, groups_per_row, group,
                                      bscale_lo, bscale_hi);
                }

                for (int r = 0; r < r_valid; r++) {
                    float a_scale =
                        a_scales[(size_t)(m + r) * blocks_per_row + qb];
                    float32x4_t acc_lo = vld1q_f32(acc[r]);
                    float32x4_t acc_hi = vld1q_f32(acc[r] + 4);
                    acc_lo = vfmaq_f32(
                        acc_lo, q4_scaled16_dot_to_f32(vld1q_s32(dots[r])),
                        vmulq_n_f32(bscale_lo, a_scale));
                    acc_hi = vfmaq_f32(
                        acc_hi, q4_scaled16_dot_to_f32(vld1q_s32(dots[r] + 4)),
                        vmulq_n_f32(bscale_hi, a_scale));
                    vst1q_f32(acc[r], acc_lo);
                    vst1q_f32(acc[r] + 4, acc_hi);
                }
            }

            for (int r = 0; r < r_valid; r++) {
                float* c_row = C + (m + r) * ldc;
                for (int c = 0; c < c_valid; c++) {
                    c_row[n + c] = acc[r][c];
                }
            }
        }
    }
}

template <bool PackedA4>
static void matmul_int4_q8dot_neon_8x8_range(
    const int8_t* qA, const float* a_scales, const uint8_t* B,
    const uint8_t* B_repack, const float* scales, int group_size,
    int groups_per_row, float* C, int M, int N, int K, int K_padded,
    int K_weight, int ldc, int m_begin, int m_end, int n_begin, int n_end,
    const Q8A4Block* qA4) {
    (void)M;
    (void)N;
    if (group_size <= 0)
        group_size = K;
    int row_stride = (K_weight + 1) / 2;
    int blocks_per_row = K / MATMUL_Q8_BLOCK;
    constexpr int bytes_per_block = MATMUL_Q8_BLOCK / 2;
    W8ScaleMode scale_mode = w8_scale_mode(group_size, groups_per_row);
    bool scale_per_channel = scale_mode == W8ScaleMode::PerChannel;

    for (int m = m_begin; m < m_end; m += 8) {
        int m_tile_end = std::min(m + 8, m_end);
        int r_valid = m_tile_end - m;

        for (int n = n_begin; n < n_end; n += 8) {
            int n_tile_end = std::min(n + 8, n_end);
            int c_valid = n_tile_end - n;
            bool full_n_tile = (c_valid == 8);
            float32x4_t bscale_lo_pc = vdupq_n_f32(0.f);
            float32x4_t bscale_hi_pc = vdupq_n_f32(0.f);
            if (scale_per_channel) {
                load_w8_b_scales8(scales, n, c_valid, groups_per_row, 0,
                                  bscale_lo_pc, bscale_hi_pc);
            }

            float32x4_t acc_lo[8];
            float32x4_t acc_hi[8];
            for (int r = 0; r < r_valid; r++) {
                acc_lo[r] = vdupq_n_f32(0.f);
                acc_hi[r] = vdupq_n_f32(0.f);
            }

            int qblocks_per_group =
                (scale_mode == W8ScaleMode::PerGroup)
                    ? std::max(1, group_size / MATMUL_Q8_BLOCK)
                    : 1;
            int cached_group = -1;
            float32x4_t cached_bscale_lo = vdupq_n_f32(0.f);
            float32x4_t cached_bscale_hi = vdupq_n_f32(0.f);

            for (int qb = 0; qb < blocks_per_row; qb++) {
                int byte_off = qb * (MATMUL_Q8_BLOCK / 2);
                const uint8_t* b_repack_block =
                    B_repack
                        ? B_repack + ((size_t)(n / 8) * blocks_per_row + qb) *
                                         8 * bytes_per_block
                        : nullptr;

                int8x16_t q4_even[8];
                int8x16_t q4_odd[8];
                int c_load_end = full_n_tile ? 8 : c_valid;
                for (int c = 0; c < c_load_end; c++) {
                    const uint8_t* src =
                        b_repack_block
                            ? b_repack_block + (size_t)c * bytes_per_block
                            : B + (size_t)(n + c) * row_stride + byte_off;
                    load_int4x32_signed_scaled16(src, q4_even[c], q4_odd[c]);
                }

                float32x4_t bscale_lo = bscale_lo_pc;
                float32x4_t bscale_hi = bscale_hi_pc;
                if (!scale_per_channel) {
                    int group = (scale_mode == W8ScaleMode::PerGroup)
                                    ? qb / qblocks_per_group
                                    : qb;
                    if (group != cached_group) {
                        load_w8_b_scales8(scales, n, c_valid, groups_per_row,
                                          group, cached_bscale_lo,
                                          cached_bscale_hi);
                        cached_group = group;
                    }
                    bscale_lo = cached_bscale_lo;
                    bscale_hi = cached_bscale_hi;
                }

                for (int r = 0; r < r_valid; r++) {
                    int8x16_t qa_even;
                    int8x16_t qa_odd;
                    float a_scale;
                    if constexpr (PackedA4) {
                        int row = m + r;
                        const Q8A4Block& a_block =
                            qA4[(size_t)(row / 4) * blocks_per_row + qb];
                        int ar = row & 3;
                        qa_even = vld1q_s8(a_block.even[ar]);
                        qa_odd = vld1q_s8(a_block.odd[ar]);
                        a_scale = a_block.scales[ar];
                    } else {
                        const int8_t* qa = qA + (size_t)(m + r) * K_padded +
                                           qb * MATMUL_Q8_BLOCK;
                        int8x16_t qa0 = vld1q_s8(qa);
                        int8x16_t qa1 = vld1q_s8(qa + 16);
                        qa_even = vuzp1q_s8(qa0, qa1);
                        qa_odd = vuzp2q_s8(qa0, qa1);
                        a_scale =
                            a_scales[(size_t)(m + r) * blocks_per_row + qb];
                    }
                    int32x4_t d0 = vdupq_n_s32(0);
                    int32x4_t d1 = vdupq_n_s32(0);
                    int32x4_t d2 = vdupq_n_s32(0);
                    int32x4_t d3 = vdupq_n_s32(0);
                    int32x4_t d4 = vdupq_n_s32(0);
                    int32x4_t d5 = vdupq_n_s32(0);
                    int32x4_t d6 = vdupq_n_s32(0);
                    int32x4_t d7 = vdupq_n_s32(0);
                    if (full_n_tile) {
                        d0 =
                            q4_q8_dot32(q4_even[0], q4_odd[0], qa_even, qa_odd);
                        d1 =
                            q4_q8_dot32(q4_even[1], q4_odd[1], qa_even, qa_odd);
                        d2 =
                            q4_q8_dot32(q4_even[2], q4_odd[2], qa_even, qa_odd);
                        d3 =
                            q4_q8_dot32(q4_even[3], q4_odd[3], qa_even, qa_odd);
                        d4 =
                            q4_q8_dot32(q4_even[4], q4_odd[4], qa_even, qa_odd);
                        d5 =
                            q4_q8_dot32(q4_even[5], q4_odd[5], qa_even, qa_odd);
                        d6 =
                            q4_q8_dot32(q4_even[6], q4_odd[6], qa_even, qa_odd);
                        d7 =
                            q4_q8_dot32(q4_even[7], q4_odd[7], qa_even, qa_odd);
                    } else {
                        if (c_valid > 0)
                            d0 = q4_q8_dot32(q4_even[0], q4_odd[0], qa_even,
                                             qa_odd);
                        if (c_valid > 1)
                            d1 = q4_q8_dot32(q4_even[1], q4_odd[1], qa_even,
                                             qa_odd);
                        if (c_valid > 2)
                            d2 = q4_q8_dot32(q4_even[2], q4_odd[2], qa_even,
                                             qa_odd);
                        if (c_valid > 3)
                            d3 = q4_q8_dot32(q4_even[3], q4_odd[3], qa_even,
                                             qa_odd);
                        if (c_valid > 4)
                            d4 = q4_q8_dot32(q4_even[4], q4_odd[4], qa_even,
                                             qa_odd);
                        if (c_valid > 5)
                            d5 = q4_q8_dot32(q4_even[5], q4_odd[5], qa_even,
                                             qa_odd);
                        if (c_valid > 6)
                            d6 = q4_q8_dot32(q4_even[6], q4_odd[6], qa_even,
                                             qa_odd);
                        if (c_valid > 7)
                            d7 = q4_q8_dot32(q4_even[7], q4_odd[7], qa_even,
                                             qa_odd);
                    }

                    int32x4_t p01 = vpaddq_s32(d0, d1);
                    int32x4_t p23 = vpaddq_s32(d2, d3);
                    int32x4_t p45 = vpaddq_s32(d4, d5);
                    int32x4_t p67 = vpaddq_s32(d6, d7);
                    int32x4_t dots_lo = vpaddq_s32(p01, p23);
                    int32x4_t dots_hi = vpaddq_s32(p45, p67);

                    acc_lo[r] =
                        vfmaq_f32(acc_lo[r], q4_scaled16_dot_to_f32(dots_lo),
                                  vmulq_n_f32(bscale_lo, a_scale));
                    acc_hi[r] =
                        vfmaq_f32(acc_hi[r], q4_scaled16_dot_to_f32(dots_hi),
                                  vmulq_n_f32(bscale_hi, a_scale));
                }
            }

            for (int r = 0; r < r_valid; r++) {
                float* c_row = C + (m + r) * ldc;
                if (full_n_tile) {
                    vst1q_f32(c_row + n, acc_lo[r]);
                    vst1q_f32(c_row + n + 4, acc_hi[r]);
                } else {
                    float tmp[4];
                    vst1q_f32(tmp, acc_lo[r]);
                    for (int c = 0; c < 4 && c < c_valid; c++)
                        c_row[n + c] = tmp[c];
                    vst1q_f32(tmp, acc_hi[r]);
                    for (int c = 0; c < 4 && c + 4 < c_valid; c++)
                        c_row[n + 4 + c] = tmp[c];
                }
            }
        }
    }
}

template <bool PackedA4>
static void matmul_int4_q8dot_neon_8x8_g128packed_range(
    const int8_t* qA, const float* a_scales, const Q4B8G128Block* B_g128,
    float* C, int M, int N, int K, int K_padded, int ldc, int m_begin,
    int m_end, int n_begin, int n_end, const Q8A4Block* qA4) {
    (void)M;
    (void)N;
    int blocks_per_row = K / MATMUL_Q8_BLOCK;
    int g128_per_row = K / 128;

    for (int m = m_begin; m < m_end; m += 8) {
        int m_tile_end = std::min(m + 8, m_end);
        int r_valid = m_tile_end - m;

        for (int n = n_begin; n < n_end; n += 8) {
            int n_tile_end = std::min(n + 8, n_end);
            int c_valid = n_tile_end - n;
            bool full_n_tile = (c_valid == 8);

            float32x4_t acc_lo[8];
            float32x4_t acc_hi[8];
            for (int r = 0; r < r_valid; r++) {
                acc_lo[r] = vdupq_n_f32(0.f);
                acc_hi[r] = vdupq_n_f32(0.f);
            }

            const Q4B8G128Block* b_tile =
                B_g128 + (size_t)(n / 8) * g128_per_row;
            for (int g = 0; g < g128_per_row; g++) {
                const Q4B8G128Block& b_group = b_tile[g];
                float32x4_t bscale_lo = vld1q_f32(b_group.scales);
                float32x4_t bscale_hi = vld1q_f32(b_group.scales + 4);

                for (int qgi = 0; qgi < 4; qgi++) {
                    int qb = g * 4 + qgi;
                    int8x16_t q4_even[8];
                    int8x16_t q4_odd[8];
                    for (int c = 0; c < 8; c++) {
                        load_int4x32_signed_scaled16(b_group.q[qgi][c],
                                                     q4_even[c], q4_odd[c]);
                    }

                    for (int r = 0; r < r_valid; r++) {
                        int8x16_t qa_even;
                        int8x16_t qa_odd;
                        float a_scale;
                        if constexpr (PackedA4) {
                            int row = m + r;
                            const Q8A4Block& a_block =
                                qA4[(size_t)(row / 4) * blocks_per_row + qb];
                            int ar = row & 3;
                            qa_even = vld1q_s8(a_block.even[ar]);
                            qa_odd = vld1q_s8(a_block.odd[ar]);
                            a_scale = a_block.scales[ar];
                        } else {
                            const int8_t* qa = qA + (size_t)(m + r) * K_padded +
                                               qb * MATMUL_Q8_BLOCK;
                            int8x16_t qa0 = vld1q_s8(qa);
                            int8x16_t qa1 = vld1q_s8(qa + 16);
                            qa_even = vuzp1q_s8(qa0, qa1);
                            qa_odd = vuzp2q_s8(qa0, qa1);
                            a_scale =
                                a_scales[(size_t)(m + r) * blocks_per_row + qb];
                        }

                        int32x4_t d0 =
                            q4_q8_dot32(q4_even[0], q4_odd[0], qa_even, qa_odd);
                        int32x4_t d1 =
                            q4_q8_dot32(q4_even[1], q4_odd[1], qa_even, qa_odd);
                        int32x4_t d2 =
                            q4_q8_dot32(q4_even[2], q4_odd[2], qa_even, qa_odd);
                        int32x4_t d3 =
                            q4_q8_dot32(q4_even[3], q4_odd[3], qa_even, qa_odd);
                        int32x4_t d4 =
                            q4_q8_dot32(q4_even[4], q4_odd[4], qa_even, qa_odd);
                        int32x4_t d5 =
                            q4_q8_dot32(q4_even[5], q4_odd[5], qa_even, qa_odd);
                        int32x4_t d6 =
                            q4_q8_dot32(q4_even[6], q4_odd[6], qa_even, qa_odd);
                        int32x4_t d7 =
                            q4_q8_dot32(q4_even[7], q4_odd[7], qa_even, qa_odd);

                        int32x4_t p01 = vpaddq_s32(d0, d1);
                        int32x4_t p23 = vpaddq_s32(d2, d3);
                        int32x4_t p45 = vpaddq_s32(d4, d5);
                        int32x4_t p67 = vpaddq_s32(d6, d7);
                        int32x4_t dots_lo = vpaddq_s32(p01, p23);
                        int32x4_t dots_hi = vpaddq_s32(p45, p67);

                        acc_lo[r] = vfmaq_f32(acc_lo[r],
                                              q4_scaled16_dot_to_f32(dots_lo),
                                              vmulq_n_f32(bscale_lo, a_scale));
                        acc_hi[r] = vfmaq_f32(acc_hi[r],
                                              q4_scaled16_dot_to_f32(dots_hi),
                                              vmulq_n_f32(bscale_hi, a_scale));
                    }
                }
            }

            for (int r = 0; r < r_valid; r++) {
                float* c_row = C + (m + r) * ldc;
                if (full_n_tile) {
                    vst1q_f32(c_row + n, acc_lo[r]);
                    vst1q_f32(c_row + n + 4, acc_hi[r]);
                } else {
                    float tmp[4];
                    vst1q_f32(tmp, acc_lo[r]);
                    for (int c = 0; c < 4 && c < c_valid; c++)
                        c_row[n + c] = tmp[c];
                    vst1q_f32(tmp, acc_hi[r]);
                    for (int c = 0; c < 4 && c + 4 < c_valid; c++)
                        c_row[n + 4 + c] = tmp[c];
                }
            }
        }
    }
}

#endif

#if !(HAS_NEON && defined(__ARM_FEATURE_DOTPROD))
bool kernel_matmul_int4_gemv_batch(const std::vector<Tensor>&,
                                   const std::vector<Tensor>&,
                                   std::vector<Tensor>&,
                                   ThreadPool*) {
    return false;
}
#endif

static void matmul_int4_scalar_range(const float* A, const uint8_t* B,
                                     const float* scales, int group_size,
                                     int groups_per_row, float* C, int M, int N,
                                     int K, int lda, int K_weight, int ldc,
                                     int m_begin, int m_end, int n_begin,
                                     int n_end) {
    (void)M;
    (void)N;
    if (group_size <= 0)
        group_size = K;
    if (groups_per_row <= 0)
        groups_per_row = 1;
    int row_stride = (K_weight + 1) / 2;

    for (int m = m_begin; m < m_end; m++) {
        float* c_row = C + m * ldc;
        for (int n = n_begin; n < n_end; n++) {
            const uint8_t* b_row = B + (size_t)n * row_stride;
            const float* s_row = scales + n * groups_per_row;
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                uint8_t byte = b_row[k >> 1];
                int8_t q = unpack_int4_signed(byte, (k & 1) != 0);
                int g = k / group_size;
                sum += A[k + m * lda] * ((float)q * s_row[g]);
            }
            c_row[n] = sum;
        }
    }
}

void matmul_dispatch_int4(const Tensor& A, const Tensor& B, Tensor& C,
                          ThreadPool* thread_pool, Activation act,
                          int act_n_begin, int act_n_len, MatmulTimer& timer) {
    const int M = (int)A.shape[1];
    const int K = (int)A.shape[0];
    const int N = (int)B.shape[0];
    const int lda = (int)(A.stride[1] / sizeof(float));
    const int ldc = (int)(C.stride[1] / sizeof(float));
    const int K_weight = (int)B.shape[1];
    const float* a_ptr = A.ptr<float>();
    float* c_ptr = C.ptr<float>();
    const uint8_t* b_int4 = reinterpret_cast<const uint8_t*>(B.data);

    const float* scales = B.scales;
    int group_size = (int)B.group_size;
    int groups_per_row = (int)B.groups_per_row;
    const uint8_t* b_q4_repack =
        reinterpret_cast<const uint8_t*>(B.q4_repack_data);
    const auto* b_q4_g32 =
        reinterpret_cast<const Q4B8G32Block*>(B.q4_g32_data);
    const auto* b_q4_g128 =
        reinterpret_cast<const Q4B8G128Block*>(B.q4_g128_data);
    int n_threads = thread_pool ? thread_pool->num_threads() : 1;
    const bool has_embedded_bg128_scales =
        B.is_q4_g128_packed && b_q4_g128 && group_size == 128 &&
        K % 128 == 0;
    const bool has_embedded_bg32_scales =
        B.is_q4_g32_packed && b_q4_g32 && group_size == 32 &&
        K % 32 == 0;
    if ((!scales && !has_embedded_bg32_scales &&
         !has_embedded_bg128_scales) ||
        group_size <= 0 ||
        groups_per_row <= 0) {
        timer.set_shape("int4_invalid_scales", M, N, K, group_size,
                        groups_per_row, false, false, n_threads);
        return;
    }

    if (mollm::cpu::matmul_int4_packed(
            A, B, C, lda, ldc, thread_pool)) {
        timer.set_shape(mollm::cpu::capabilities().x86_avx512
                            ? "int4_packed_avx512"
                        : mollm::cpu::capabilities().x86_avx2
                            ? "int4_packed_avx2"
                            : "int4_packed_scalar",
                        M, N, K, group_size,
                        groups_per_row, true, false, n_threads);
        if (act != Activation::NONE && act_n_len != 0)
            matmul_apply_activation(c_ptr, M, N, ldc, 0, M, act,
                                    act_n_begin, act_n_len);
        return;
    }

    constexpr int tile_m = HAS_NEON ? 8 : 1;
    bool shard_by_n = (N > M * 8 && M == 1);
    int chunk_size =
        (M == 1 || N == 1) ? g_matmul_config.gemv_chunk_size : tile_m;
    int total_dim = shard_by_n ? N : M;
    int n_chunks = (total_dim + chunk_size - 1) / chunk_size;
    bool use_parallel = n_threads > 1 && n_chunks > 1;

#if HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
    bool has_direct_q4_g32 = B.is_q4_g32_packed && b_q4_g32;
    bool has_direct_q4_g128 = B.is_q4_g128_packed && b_q4_g128;
    bool can_use_q4_dot =
        (B.is_q4_repacked || has_direct_q4_g32 || has_direct_q4_g128 ||
         b_int4 != nullptr) &&
        (K % MATMUL_Q8_BLOCK == 0) && (group_size % MATMUL_Q8_BLOCK == 0);
    bool use_q4_repack = can_use_q4_dot && b_q4_repack;
    bool can_use_q4_bg32 =
        can_use_q4_dot && b_q4_g32 && group_size == 32 && (K % 32 == 0);
    bool can_use_q4_bg128 =
        can_use_q4_dot && b_q4_g128 && group_size == 128 && (K % 128 == 0);
#if MOLLM_ARM_I8MM_KERNELS
    bool can_use_q4_i8mm =
        mollm::cpu::capabilities().arm_i8mm && M > 1 &&
        (can_use_q4_bg32 || can_use_q4_bg128);
#else
    bool can_use_q4_i8mm = false;
#endif
    // Direct BG32 packages do not retain the original row-major INT4 bytes
    // or a separate scale array.  Without i8mm, run the existing DOTPROD
    // GEMV kernel once per input row instead of falling through to the
    // generic GEMM path with null B/scales pointers.
    if (M > 1 && B.is_q4_g32_packed && can_use_q4_bg32 &&
        !can_use_q4_i8mm) {
        timer.set_shape("q4dot_gemm_bg32_rowwise", M, N, K, group_size,
                        groups_per_row, true, false, n_threads);
        auto run_rows = [&](int m_begin, int m_end) {
            static thread_local Q4GemvScratch scratch;
            for (int m = m_begin; m < m_end; ++m) {
                quantize_a_q8_blocks_even_odd(
                    a_ptr + static_cast<size_t>(m) * lda, K, scratch.qA_even,
                    scratch.qA_odd, scratch.a_scales);
                matmul_int4_q8dot_neon_gemv_g32_range(
                    scratch.qA_even.data(), scratch.qA_odd.data(),
                    scratch.a_scales.data(), b_q4_g32,
                    c_ptr + static_cast<size_t>(m) * ldc, K, 0, N);
            }
        };
        if (use_parallel) {
            thread_pool->parallel_for(
                0, M, 1, [&](int, int m_begin, int m_end) {
                    run_rows(m_begin, m_end);
                });
        } else {
            run_rows(0, M);
        }
        if (act != Activation::NONE && act_n_len != 0)
            matmul_apply_activation(c_ptr, M, N, ldc, 0, M, act,
                                    act_n_begin, act_n_len);
        return;
    }
    if (M == 1 && can_use_q4_dot) {
        bool use_q4_gemv_bg128 = can_use_q4_bg128;
        bool use_q4_gemv_bg32 = can_use_q4_bg32;
        const char* path =
            use_q4_gemv_bg128
                ? "q4dot_gemv_bg128"
                : (use_q4_gemv_bg32
                       ? "q4dot_gemv_bg32"
                       : (use_q4_repack ? "q4dot_gemv_repack"
                                        : "q4dot_gemv"));
        timer.set_shape(path, M, N, K, group_size, groups_per_row,
                        use_q4_gemv_bg128 || use_q4_gemv_bg32 ||
                            use_q4_repack,
                        false, n_threads);
        static thread_local Q4GemvScratch scratch;
        if (use_q4_gemv_bg128) {
            quantize_a_q8_g128_even_odd(a_ptr, K, scratch.qA_even,
                                       scratch.qA_odd, scratch.a_scales);
        } else {
            quantize_a_q8_blocks_even_odd(a_ptr, K, scratch.qA_even,
                                          scratch.qA_odd, scratch.a_scales);
        }
        const int8_t* qA_even_data = scratch.qA_even.data();
        const int8_t* qA_odd_data = scratch.qA_odd.data();
        const float* a_scales_data = scratch.a_scales.data();
        if (!use_parallel) {
            if (use_q4_gemv_bg128) {
                matmul_int4_q8dot_neon_gemv_g128_range(
                    qA_even_data, qA_odd_data, a_scales_data, b_q4_g128, c_ptr,
                    K, 0, N);
            } else if (use_q4_gemv_bg32) {
                matmul_int4_q8dot_neon_gemv_g32_range(
                    qA_even_data, qA_odd_data, a_scales_data, b_q4_g32, c_ptr,
                    K, 0, N);
            } else {
                matmul_int4_q8dot_neon_gemv_range(
                    nullptr, qA_even_data, qA_odd_data, a_scales_data, b_int4,
                    use_q4_repack ? b_q4_repack : nullptr, scales, group_size,
                    groups_per_row, c_ptr, K, K_weight, 0, N);
            }
        } else {
            int n_chunk = std::max(N / (n_threads * 8), 64);
            n_chunk = ((n_chunk + 7) / 8) * 8;
            thread_pool->parallel_for(
                0, N, n_chunk, [&](int, int n_begin, int n_end) {
                    if (use_q4_gemv_bg128) {
                        matmul_int4_q8dot_neon_gemv_g128_range(
                            qA_even_data, qA_odd_data, a_scales_data, b_q4_g128,
                            c_ptr, K, n_begin, n_end);
                    } else if (use_q4_gemv_bg32) {
                        matmul_int4_q8dot_neon_gemv_g32_range(
                            qA_even_data, qA_odd_data, a_scales_data,
                            b_q4_g32, c_ptr, K, n_begin, n_end);
                    } else {
                        matmul_int4_q8dot_neon_gemv_range(
                            nullptr, qA_even_data, qA_odd_data, a_scales_data,
                            b_int4, use_q4_repack ? b_q4_repack : nullptr,
                            scales, group_size, groups_per_row, c_ptr, K,
                            K_weight, n_begin, n_end);
                    }
                });
        }
        if (act != Activation::NONE && act_n_len != 0) {
            matmul_apply_activation_gemv(c_ptr, N, act, act_n_begin, act_n_len);
        }
        return;
    }
    if (can_use_q4_dot) {
        constexpr bool force_4x8 = false;
        constexpr bool use_q4_dot_gemm_a4 = true;
        bool use_q4_dot_gemm_bg128 = can_use_q4_bg128 && !force_4x8;
        const char* path =
            can_use_q4_i8mm
                ? (can_use_q4_bg128 ? "q4_i8mm_g128_a8"
                                    : "q4_i8mm_g32_a8")
                : (use_q4_dot_gemm_bg128
                       ? (use_q4_dot_gemm_a4 ? "q4dot_gemm_bg128_a4"
                                             : "q4dot_gemm_bg128")
                       : (use_q4_dot_gemm_a4
                              ? (use_q4_repack ? "q4dot_gemm_repack_a4"
                                               : "q4dot_gemm_a4")
                              : (use_q4_repack ? "q4dot_gemm_repack"
                                               : "q4dot_gemm")));
        timer.set_shape(path, M, N, K, group_size, groups_per_row,
                        use_q4_repack || use_q4_dot_gemm_bg128, false,
                        n_threads);
        std::vector<float> a_scales;
        std::vector<int8_t> qA;
        std::vector<Q8A4Block> qA4;
        std::vector<Q8A8I8MMBlock> qA8;
        if (can_use_q4_i8mm) {
            quantize_a_q8_blocks_i8mm_a8(a_ptr, M, K, lda, qA8);
        } else if (use_q4_dot_gemm_a4) {
            quantize_a_q8_blocks_a4(a_ptr, M, K, lda, qA4);
        } else {
            quantize_a_q8_blocks(a_ptr, M, K, lda, K, qA, a_scales);
        }
        const int8_t* qA_data = qA.data();
        const float* a_scales_data = a_scales.data();
        const Q8A4Block* qA4_data = qA4.data();

        auto run_q4_gemm = [&](int m_begin, int m_end, int n_begin, int n_end) {
#if MOLLM_ARM_I8MM_KERNELS
            if (can_use_q4_i8mm) {
                if (can_use_q4_bg128) {
                    matmul_int4_i8mm_g128(
                        qA8.data(), b_q4_g128, c_ptr, M, N, K, ldc, m_begin,
                        m_end, n_begin, n_end);
                } else {
                    matmul_int4_i8mm_g32(
                        qA8.data(), b_q4_g32, c_ptr, M, N, K, ldc, m_begin,
                        m_end, n_begin, n_end);
                }
                return;
            } else
#endif
                if (force_4x8) {
                matmul_int4_q8dot_neon_4x8_range(
                    qA_data, a_scales_data, b_int4,
                    use_q4_repack ? b_q4_repack : nullptr, scales, group_size,
                    groups_per_row, c_ptr, M, N, K, K, K_weight, ldc, m_begin,
                    m_end);
            } else if (use_q4_dot_gemm_bg128) {
                if (use_q4_dot_gemm_a4) {
                    matmul_int4_q8dot_neon_8x8_g128packed_range<true>(
                        nullptr, nullptr, b_q4_g128, c_ptr, M, N, K, K, ldc,
                        m_begin, m_end, n_begin, n_end, qA4_data);
                } else {
                    matmul_int4_q8dot_neon_8x8_g128packed_range<false>(
                        qA_data, a_scales_data, b_q4_g128, c_ptr, M, N, K, K,
                        ldc, m_begin, m_end, n_begin, n_end, nullptr);
                }
            } else if (use_q4_dot_gemm_a4) {
                matmul_int4_q8dot_neon_8x8_range<true>(
                    nullptr, nullptr, b_int4,
                    use_q4_repack ? b_q4_repack : nullptr, scales, group_size,
                    groups_per_row, c_ptr, M, N, K, K, K_weight, ldc, m_begin,
                    m_end, n_begin, n_end, qA4_data);
            } else {
                matmul_int4_q8dot_neon_8x8_range<false>(
                    qA_data, a_scales_data, b_int4,
                    use_q4_repack ? b_q4_repack : nullptr, scales, group_size,
                    groups_per_row, c_ptr, M, N, K, K, K_weight, ldc, m_begin,
                    m_end, n_begin, n_end, nullptr);
            }
        };

        if (can_use_q4_i8mm && n_threads > 1 && N > 8) {
            int n_chunk = ((N + n_threads - 1) / n_threads + 7) / 8 * 8;
            thread_pool->parallel_for(
                0, N, n_chunk, [&](int, int n_begin, int n_end) {
                    run_q4_gemm(0, M, n_begin, n_end);
                });
        } else if (!use_parallel) {
            run_q4_gemm(0, M, 0, N);
        } else {
            thread_pool->parallel_for(0, M, tile_m,
                                      [&](int, int m_begin, int m_end) {
                                          run_q4_gemm(m_begin, m_end, 0, N);
                                      });
        }
        if (act != Activation::NONE && act_n_len != 0) {
            matmul_apply_activation(c_ptr, M, N, ldc, 0, M, act, act_n_begin,
                                    act_n_len);
        }
        return;
    }
#endif

    timer.set_shape("int4_scalar", M, N, K, group_size, groups_per_row, false,
                    false, n_threads);
    if (!use_parallel) {
        matmul_int4_scalar_range(a_ptr, b_int4, scales, group_size,
                                 groups_per_row, c_ptr, M, N, K, lda, K_weight,
                                 ldc, 0, M, 0, N);
    } else if (shard_by_n) {
        thread_pool->parallel_for(
            0, N, chunk_size, [&](int, int n_begin, int n_end) {
                matmul_int4_scalar_range(a_ptr, b_int4, scales, group_size,
                                         groups_per_row, c_ptr, M, N, K, lda,
                                         K_weight, ldc, 0, M, n_begin, n_end);
            });
    } else {
        thread_pool->parallel_for(
            0, M, chunk_size, [&](int, int m_begin, int m_end) {
                matmul_int4_scalar_range(a_ptr, b_int4, scales, group_size,
                                         groups_per_row, c_ptr, M, N, K, lda,
                                         K_weight, ldc, m_begin, m_end, 0, N);
            });
    }
    if (act != Activation::NONE && act_n_len != 0) {
        matmul_apply_activation(c_ptr, M, N, ldc, 0, M, act, act_n_begin,
                                act_n_len);
    }
    return;
}
