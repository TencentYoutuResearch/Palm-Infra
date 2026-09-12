#pragma once

// Runtime SDPA configuration, independent of graph serialization.
struct SdpaParams {
    int kv_cache = 2;
    int causal = 1;
    int num_heads = 16;
    int num_kv_heads = 16;
    int head_dim = 192;
    int v_head_dim = 128;
    float scale = 0.f;  // Zero selects 1/sqrt(head_dim).
};
