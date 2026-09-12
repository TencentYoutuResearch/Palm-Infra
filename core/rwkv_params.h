#pragma once

struct RwkvTokenShiftParams {
    int hidden = 0;
    int seq_len = 1;
    int real_tokens = 1;
};

struct RwkvL2NormParams {
    int num_heads = 0;
    int head_dim = 0;
    float epsilon = 1e-12f;
};

struct RwkvPostParams {
    int num_heads = 0;
    int head_dim = 0;
    float epsilon = 64e-5f;
};

struct Rwkv7Params {
    int num_heads = 0;
    int head_dim = 0;
    int seq_len = 1;
    int real_tokens = 1;
};
