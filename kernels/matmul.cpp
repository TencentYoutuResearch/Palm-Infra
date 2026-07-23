#include "kernels/matmul.h"
#include "kernels/matmul_internal.h"
#include "kernels/matmul_profile.h"

MatmulConfig g_matmul_config;

// Debug override used by precision-comparison tests and tools.
bool g_mollm_force_fp32_acc = false;

void kernel_matmul_fp32(const Tensor& A, const Tensor& B, Tensor& C,
                        ThreadPool* thread_pool, Activation act,
                        int act_n_begin, int act_n_len) {
    MatmulTimer timer;

    switch (B.prec) {
    case Precision::INT4:
        matmul_dispatch_int4(A, B, C, thread_pool, act, act_n_begin, act_n_len,
                             timer);
        return;
    case Precision::INT8:
        matmul_dispatch_int8(A, B, C, thread_pool, act, act_n_begin, act_n_len,
                             timer);
        return;
    default:
        matmul_dispatch_dense(A, B, C, thread_pool, act, act_n_begin, act_n_len,
                              timer);
        return;
    }
}
