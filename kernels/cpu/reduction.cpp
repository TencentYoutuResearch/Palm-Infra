#include "kernels/cpu/reduction.h"

#include "backends/cpu/platform.h"

#include <cstdint>

int kernel_argmax(const float* logits, int vocab_size) {
#if HAS_NEON
    if (vocab_size >= 4) {
        static const int32_t kLaneOffsetsData[4] = {0, 1, 2, 3};
        int32x4_t lane_offsets = vld1q_s32(kLaneOffsetsData);
        float32x4_t best_vals = vld1q_f32(logits);
        int32x4_t best_idxs = lane_offsets;

        int i = 4;
        for (; i + 4 <= vocab_size; i += 4) {
            float32x4_t vals = vld1q_f32(logits + i);
            int32x4_t idxs = vaddq_s32(vdupq_n_s32(i), lane_offsets);
            uint32x4_t mask = vcgtq_f32(vals, best_vals);
            best_vals = vbslq_f32(mask, vals, best_vals);
            best_idxs = vbslq_s32(mask, idxs, best_idxs);
        }

        float lane_vals[4];
        int32_t lane_idxs[4];
        vst1q_f32(lane_vals, best_vals);
        vst1q_s32(lane_idxs, best_idxs);

        int best = lane_idxs[0];
        float best_val = lane_vals[0];
        for (int lane = 1; lane < 4; lane++) {
            if (lane_vals[lane] > best_val ||
                (lane_vals[lane] == best_val && lane_idxs[lane] < best)) {
                best = lane_idxs[lane];
                best_val = lane_vals[lane];
            }
        }
        for (; i < vocab_size; i++) {
            if (logits[i] > best_val) {
                best = i;
                best_val = logits[i];
            }
        }
        return best;
    }
#endif

    int best = 0;
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > logits[best])
            best = i;
    }
    return best;
}
