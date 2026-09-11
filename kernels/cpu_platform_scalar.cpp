#include "kernels/cpu_platform.h"
#include "kernels/matmul.h"
#include "kernels/matmul_internal.h"

#include <algorithm>
#include <thread>

namespace mollm::cpu {

const Capabilities& capabilities() {
    static const Capabilities value{};
    return value;
}

const char* isa_name() {
    return "scalar";
}

void relax() {
    // Correctness-first backend: yielding avoids embedding an architecture
    // instruction in common worker-pool code.  An x86 microkernel provider
    // can replace this with pause later without changing callers.
    std::this_thread::yield();
}

namespace {

int8_t unpack_int4_signed(uint8_t byte, bool high_nibble) {
    int value = high_nibble ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
    return static_cast<int8_t>(value >= 8 ? value - 16 : value);
}

}  // namespace

bool matmul_int4_packed(const Tensor& A, const Tensor& B, Tensor& C, int lda,
                        int ldc, ThreadPool*) {
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

    for (int m = 0; m < M; ++m) {
        float* out = C.ptr<float>() + static_cast<size_t>(m) * ldc;
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                int8_t value = 0;
                float scale = 0.0f;
                if (is_bg128) {
                    const int group = k / 128;
                    const auto& block = bg128[
                        static_cast<size_t>(n / 8) * groups + group];
                    const int local = k & 127;
                    const uint8_t byte = block.q[local / 32][n & 7]
                                                   [(local & 31) >> 1];
                    value = unpack_int4_signed(byte, (local & 1) != 0);
                    scale = block.scales[n & 7];
                } else if (is_bg32) {
                    const int group = k / 32;
                    const auto& block = bg32[
                        static_cast<size_t>(n / 8) * groups + group];
                    const int local = k & 31;
                    const uint8_t byte = block.q[n & 7][local >> 1];
                    value = unpack_int4_signed(byte, (local & 1) != 0);
                    scale = block.scales[n & 7];
                } else {
                    const int block_k = k / 32;
                    const size_t byte_index =
                        ((static_cast<size_t>(n / 8) * ((K + 31) / 32) +
                          block_k) * 8 + (n & 7)) * 16 +
                        ((k & 31) >> 1);
                    value = unpack_int4_signed(q4dot[byte_index],
                                                (k & 1) != 0);
                    scale = B.scales[static_cast<size_t>(n) * groups +
                                     k / static_cast<int>(B.group_size)];
                }
                sum += A.ptr<float>()[static_cast<size_t>(m) * lda + k] *
                       static_cast<float>(value) * scale;
            }
            out[n] = sum;
        }
    }
    return true;
}

bool matmul_dense_fp32_range(const float*, const float*, float*, int, int, int,
                             int, int, int, int) {
    return false;
}

bool matmul_dense_fp16_range(const float*, const fp16_t*, float*, int, int, int,
                             int, int, int, int, bool) {
    return false;
}

bool matmul_dense_fp16_m2_range_n(
    const float*, const fp16_t*, float*, int, int, int, int, int, int, int) {
    return false;
}

bool matmul_int8_range(const float*, const int8_t*, const float*, float*, int,
                       int, int, int, int, int, int, int, int, int, int, bool) {
    return false;
}

}  // namespace mollm::cpu

// The ARM provider owns the production interleaved FP16 packing routine.
// Keep a byte-identical scalar implementation here for API completeness and
// for direct layout tests; the scalar capability provider never selects this
// layout for inference.
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
