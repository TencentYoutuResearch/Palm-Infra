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

    // ---- is_contiguous_n ----
    // a is 4x3 contiguous; after permute(1,0), dim 0 is not contiguous
    // but dims >=1 are
    CHECK(!p.is_contiguous_n(0), "permute: dim0 not contiguous");
    // p = permute(1,0): shape[1]=4, stride[1]=2.  stride[0]*shape[0]=8*3=24.
    // Since 2 != 24, dim1 is also not contiguous.  Correct.
    CHECK(!p.is_contiguous_n(1), "permute: dim1 also not contiguous (stride 2 vs expected 24)");

    delete[] fd;

    if (failures == 0) {
        printf("\nAll tensor tests passed!\n");
    } else {
        printf("\n%d test(s) FAILED\n", failures);
    }
    return failures;
}
