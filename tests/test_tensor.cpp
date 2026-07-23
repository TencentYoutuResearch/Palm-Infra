#include "kernels/tensor.h"
#include <cstdio>
#include <cstdlib>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
    else { printf("  PASS: %s\n", msg); } \
} while(0)

int main() {
    // ---- creation ----
    Tensor a = Tensor::create(Precision::FP16, MemoryType::OWNED, 4, 3);
    CHECK(a.shape[0] == 4 && a.shape[1] == 3, "create 4x3 FP16");
    CHECK(a.element_size() == 2, "FP16 element size");
    CHECK(a.nelements() == 12, "nelements 4*3=12");
    CHECK(a.nbytes() == 24, "nbytes 4*3*2=24");
    CHECK(a.is_contiguous(), "contiguous after create");

    // ---- strides ----
    CHECK(a.stride[0] == 2, "stride[0] == 2 bytes");
    CHECK(a.stride[1] == 8, "stride[1] == 2*4 = 8 bytes");

    // ---- view_1d ----
    Tensor v1 = a.view_1d(6);
    CHECK(v1.shape[0] == 6 && v1.shape[1] == 1, "view_1d shape");
    CHECK(v1.nelements() == 6, "view_1d nelements");

    // ---- view_2d ----
    Tensor v2 = a.view_2d(2, 3);
    CHECK(v2.shape[0] == 2 && v2.shape[1] == 3, "view_2d shape 2x3");
    CHECK(v2.data == a.data, "view_2d shares data");

    // ---- storage identity ----
    float storage[4] = {};
    Tensor pooled =
        Tensor::create(Precision::FP32, MemoryType::POOLED, 4, 1, 1, 1, storage);
    pooled.owner_id = 7;
    pooled.storage_id = 42;
    Tensor pooled_view = pooled.view_1d(3, sizeof(float));
    CHECK(pooled_view.data != pooled.data, "offset view has a distinct pointer");
    CHECK(pooled_view.shares_storage_with(pooled),
          "offset view retains pooled storage identity");
    Tensor other_pool = pooled_view;
    other_pool.owner_id = 8;
    CHECK(!other_pool.shares_storage_with(pooled),
          "storage ids are scoped by pool owner");
    Tensor external_a =
        Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 4, 1, 1, 1, storage);
    Tensor external_b = external_a;
    CHECK(external_a.shares_storage_with(external_b),
          "external tensors fall back to pointer identity");

    // ---- reshape ----
    Tensor r = a.reshape(6, 2);
    CHECK(r.shape[0] == 6 && r.shape[1] == 2, "reshape 6x2");
    CHECK(r.nelements() == 12, "reshape nelements");

    // ---- permute ----
    Tensor p = a.permute(1, 0, 2, 3);
    CHECK(p.shape[0] == 3 && p.shape[1] == 4, "permute 3x4 shape");
    CHECK(p.stride[0] == 8 && p.stride[1] == 2, "permute strides swapped");
    CHECK(p.data == a.data, "permute shares data");

    // ---- at() access ----
    float* fd = new float[6]{1, 2, 3, 4, 5, 6};
    Tensor f = Tensor::create(Precision::FP32, MemoryType::OWNED, 3, 2, 1, 1, fd);
    CHECK(f.at<float>(0, 0) == 1.0f, "at(0,0)");
    CHECK(f.at<float>(2, 1) == 6.0f, "at(2,1)");
    CHECK(f.at<float>(1, 0) == 2.0f, "at(1,0)");

    // row pointer
    CHECK(f.row<float>(1)[0] == 4.0f, "row(1)[0] == 4");
    CHECK(f.row<float>(1)[2] == 6.0f, "row(1)[2] == 6");

    // ---- 1-D tensor ----
    Tensor t1d = Tensor::create(Precision::FP32, MemoryType::NONE, 10);
    CHECK(t1d.shape[0] == 10, "1d shape[0]");
    CHECK(t1d.shape[1] == 1,  "1d shape[1]==1");

    // ---- Precision enum ----
    CHECK(precision_size(Precision::FP32) == 4, "FP32=4 bytes");
    CHECK(precision_size(Precision::FP16) == 2, "FP16=2 bytes");
    CHECK(precision_size(Precision::INT8) == 1, "INT8=1 byte");
    CHECK(precision_size(Precision::INT4) == 1, "INT4 packed storage byte");

    // ---- is_contiguous after permute ----
    CHECK(!p.is_contiguous(), "permuted tensor not contiguous");
    CHECK(a.is_contiguous(), "original still contiguous");

    delete[] fd;

    if (failures == 0) {
        printf("\nAll tensor tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
