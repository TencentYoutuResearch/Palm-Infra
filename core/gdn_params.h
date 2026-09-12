#pragma once

// Resolved configuration shared by scalar, NEON and x86 GDN implementations.
struct GdnParams {
    int num_heads = 16;
    int k_head_dim = 128;
    int v_head_dim = 128;
    int seq_len = 4;
    int flags = 1;  // Bit 0: q/k L2 normalization; bit 1: sigmoid output gate.
    int conv_kernel = 4;
    int real_tokens = 4;
    int num_v_heads = 16;
    float rms_eps = 1e-6f;
    float l2norm_eps = 1e-6f;
    float scale = 0.f;  // Zero selects 1/sqrt(k_head_dim).
};
