#pragma once

struct ShortConvParams {
    int kernel_size = 4;
    // Positive values smaller than the input sequence exclude trailing padding.
    // Other values process the full input sequence.
    int real_tokens = 0;
};
