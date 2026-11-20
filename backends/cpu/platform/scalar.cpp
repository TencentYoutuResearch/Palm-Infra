#include "backends/cpu/platform.h"
#include "runtime/threading.h"
#include "kernels/cpu/matmul/matmul.h"
#include "kernels/cpu/matmul/matmul_internal.h"

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

}  // namespace mollm::cpu

// Implemented at global scope: the contract is declared by runtime/threading.h,
// which ThreadPool uses and which must not depend on a CPU-backend namespace.
void worker_relax() {
    // Correctness-first provider: yielding avoids embedding an architecture
    // instruction in common worker-pool code. The x86 provider overrides this
    // with pause without changing callers.
    std::this_thread::yield();
}

namespace mollm::cpu {

bool matmul_int4_packed(const Tensor& A, const Tensor& B, Tensor& C, int lda,
                        int ldc, ThreadPool*) {
    const int M = static_cast<int>(A.shape[1]);
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

    matmul_int4_packed_scalar_range(A, B, C, lda, ldc, 0, M, 0, N);
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

bool matmul_int8_range(const float*, const int8_t*, const float*, float*, int,
                       int, int, int, int, int, int, int, int, int, int, bool) {
    return false;
}

}  // namespace mollm::cpu
