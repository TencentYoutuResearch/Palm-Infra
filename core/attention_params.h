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

    // Optional per-head attention sink (Hy4 / gpt-oss style), or nullptr.
    //
    // The sink is one extra logit per head, appended to the score vector before
    // softmax. It has no value vector, so it inflates the softmax denominator
    // and shrinks every real attention weight, but contributes nothing to the
    // output numerator. Point here to an array of `num_heads` floats.
    //
    // Kept last so existing aggregate initialisation still compiles.
    const float* sink = nullptr;
};
