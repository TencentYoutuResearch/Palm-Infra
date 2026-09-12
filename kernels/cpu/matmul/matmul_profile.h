#pragma once

#include <chrono>

struct MatmulShapeProfileRow {
    const char* phase = "unscoped";
    const char* path = "unknown";
    int M = 0;
    int N = 0;
    int K = 0;
    int group_size = 0;
    int groups_per_row = 0;
    int threads = 1;
    bool has_q8_repack = false;
    bool b_interleaved = false;
    long long calls = 0;
    double total_ms = 0.0;
};

class MatmulTimer {
  public:
    MatmulTimer();
    ~MatmulTimer();

    void set_shape(const char* path, int M, int N, int K, int group_size = 0,
                   int groups_per_row = 0, bool has_q8_repack = false,
                   bool b_interleaved = false, int threads = 1);

  private:
    std::chrono::steady_clock::time_point t0_;
    MatmulShapeProfileRow shape_;
    bool shape_valid_ = false;
};

void matmul_record_pack_a(double elapsed_ms);
void matmul_record_q8_quant_a(double elapsed_ms);
