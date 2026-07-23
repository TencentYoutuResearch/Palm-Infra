#include "engine/sampler.h"

#include "kernels/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

struct SamplerCandidate {
    int id = 0;
    float logit = 0.0f;
    float prob = 0.0f;
};

struct MinLogitHeapCompare {
    bool operator()(const SamplerCandidate& a,
                    const SamplerCandidate& b) const {
        return a.logit > b.logit;
    }
};

struct SamplerScratch {
    std::vector<SamplerCandidate> candidates;
};

int argmax_token(const float* logits, int vocab_size) {
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

SamplerScratch& sampler_scratch() {
    static thread_local SamplerScratch scratch;
    return scratch;
}

} // namespace

int sample_token(float* logits, int vocab_size, float temperature, int top_k,
                 float top_p, unsigned int* seed) {
    if (vocab_size <= 0)
        return 0;
    if (temperature <= 0.0f || top_k == 1)
        return argmax_token(logits, vocab_size);

    const int k = top_k > 0 ? std::min(top_k, vocab_size) : vocab_size;
    auto& candidates = sampler_scratch().candidates;
    candidates.clear();
    candidates.reserve((size_t)k);

    MinLogitHeapCompare heap_compare;
    for (int i = 0; i < vocab_size; i++) {
        SamplerCandidate cand{i, logits[i], 0.0f};
        if ((int)candidates.size() < k) {
            candidates.push_back(cand);
            std::push_heap(candidates.begin(), candidates.end(), heap_compare);
        } else if (cand.logit > candidates.front().logit) {
            std::pop_heap(candidates.begin(), candidates.end(), heap_compare);
            candidates.back() = cand;
            std::push_heap(candidates.begin(), candidates.end(), heap_compare);
        }
    }
    if (candidates.empty())
        return 0;

    std::sort(candidates.begin(), candidates.end(),
              [](const SamplerCandidate& a, const SamplerCandidate& b) {
                  return a.logit > b.logit;
              });

    const float max_logit = candidates[0].logit;
    const float inv_t = 1.0f / temperature;
    float sum = 0.0f;
    for (auto& cand : candidates) {
        cand.prob = std::exp((cand.logit - max_logit) * inv_t);
        sum += cand.prob;
    }
    if (!(sum > 0.0f) || !std::isfinite(sum))
        return candidates[0].id;

    int active = (int)candidates.size();
    if (top_p > 0.0f && top_p < 1.0f) {
        const float cutoff_mass = top_p * sum;
        float cumulative = 0.0f;
        for (int i = 0; i < (int)candidates.size(); i++) {
            cumulative += candidates[i].prob;
            if (cumulative >= cutoff_mass) {
                active = i + 1;
                break;
            }
        }
        sum = cumulative;
    }

    const float r = (float)rand_r(seed) / (float)RAND_MAX;
    const float target = r * sum;
    float cumulative = 0.0f;
    for (int i = 0; i < active; i++) {
        cumulative += candidates[i].prob;
        if (target <= cumulative)
            return candidates[i].id;
    }
    return candidates[active - 1].id;
}
