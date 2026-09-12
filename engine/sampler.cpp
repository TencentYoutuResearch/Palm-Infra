#include "engine/sampler.h"

#include "kernels/cpu_platform.h"
#include "kernels/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <unordered_map>
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
        if (a.logit != b.logit) return a.logit > b.logit;
        return a.id < b.id;
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

struct PreparedCandidates {
    int active = 0;
    float sum = 0.0f;
};

PreparedCandidates prepare_candidates(const float* logits, int vocab_size,
                                      float temperature, int top_k,
                                      float top_p, float min_p) {
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
        } else if (cand.logit > candidates.front().logit ||
                   (cand.logit == candidates.front().logit &&
                    cand.id < candidates.front().id)) {
            std::pop_heap(candidates.begin(), candidates.end(), heap_compare);
            candidates.back() = cand;
            std::push_heap(candidates.begin(), candidates.end(), heap_compare);
        }
    }
    if (candidates.empty())
        return {};

    std::sort(candidates.begin(), candidates.end(),
              [](const SamplerCandidate& a, const SamplerCandidate& b) {
                  if (a.logit != b.logit) return a.logit > b.logit;
                  return a.id < b.id;
              });

    const float max_logit = candidates[0].logit;
    const float inv_t = 1.0f / temperature;
    for (auto& cand : candidates)
        cand.prob = std::exp((cand.logit - max_logit) * inv_t);

    int active = (int)candidates.size();
    if (min_p > 0.0f) {
        const float threshold = min_p * candidates[0].prob;
        active = 1;
        while (active < (int)candidates.size() &&
               candidates[active].prob >= threshold)
            ++active;
    }

    float sum = 0.0f;
    for (int i = 0; i < active; ++i)
        sum += candidates[i].prob;
    if (!(sum > 0.0f) || !std::isfinite(sum)) {
        candidates[0].prob = 1.0f;
        return {1, 1.0f};
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        const float cutoff_mass = top_p * sum;
        float cumulative = 0.0f;
        for (int i = 0; i < active; i++) {
            cumulative += candidates[i].prob;
            if (cumulative >= cutoff_mass) {
                active = i + 1;
                sum = cumulative;
                break;
            }
        }
    }

    return {active, sum};
}

float next_uniform(unsigned int* seed) {
    unsigned int fallback_seed = 42;
    if (!seed) seed = &fallback_seed;
    return static_cast<float>(rand_r(seed)) /
           static_cast<float>(RAND_MAX);
}

int sample_token_impl(float* logits, int vocab_size, float temperature,
                      int top_k, float top_p, float min_p,
                      unsigned int* seed) {
    if (vocab_size <= 0)
        return 0;
    if (temperature <= 0.0f || top_k == 1)
        return argmax_token(logits, vocab_size);

    const PreparedCandidates prepared = prepare_candidates(
        logits, vocab_size, temperature, top_k, top_p, min_p);
    auto& candidates = sampler_scratch().candidates;
    if (prepared.active <= 0 || candidates.empty())
        return 0;

    const float target = next_uniform(seed) * prepared.sum;
    float cumulative = 0.0f;
    for (int i = 0; i < prepared.active; i++) {
        cumulative += candidates[i].prob;
        if (target <= cumulative)
            return candidates[i].id;
    }
    return candidates[prepared.active - 1].id;
}

void probability_distribution_impl(const float* logits, int vocab_size,
                                   float temperature, int top_k,
                                   float top_p, float min_p,
                                   std::vector<float>& output) {
    output.assign(static_cast<size_t>(std::max(0, vocab_size)), 0.0f);
    if (vocab_size <= 0)
        return;
    if (temperature <= 0.0f || top_k == 1) {
        output[static_cast<size_t>(argmax_token(logits, vocab_size))] = 1.0f;
        return;
    }

    const PreparedCandidates prepared = prepare_candidates(
        logits, vocab_size, temperature, top_k, top_p, min_p);
    const auto& candidates = sampler_scratch().candidates;
    if (prepared.active <= 0 || candidates.empty() ||
        !(prepared.sum > 0.0f)) {
        output[0] = 1.0f;
        return;
    }
    const float inv_sum = 1.0f / prepared.sum;
    for (int i = 0; i < prepared.active; ++i)
        output[static_cast<size_t>(candidates[i].id)] =
            candidates[i].prob * inv_sum;
}

void set_error(std::string* error, const char* message) {
    if (error) *error = message;
}

} // namespace

bool validate_sampling_params(const SamplingParams& p, std::string* error) {
    if (!std::isfinite(p.temperature) || p.temperature < 0.0f ||
        p.temperature > 2.0f) {
        set_error(error, "temperature must be between 0 and 2");
        return false;
    }
    if (p.top_k < 0) {
        set_error(error, "top_k must be non-negative");
        return false;
    }
    if (!std::isfinite(p.top_p) || p.top_p < 0.0f || p.top_p > 1.0f) {
        set_error(error, "top_p must be between 0 and 1");
        return false;
    }
    if (!std::isfinite(p.min_p) || p.min_p < 0.0f || p.min_p > 1.0f) {
        set_error(error, "min_p must be between 0 and 1");
        return false;
    }
    if (!std::isfinite(p.repeat_penalty) || p.repeat_penalty <= 0.0f) {
        set_error(error, "repeat_penalty must be greater than 0");
        return false;
    }
    if (p.repeat_last_n < -1) {
        set_error(error, "repeat_last_n must be -1 or non-negative");
        return false;
    }
    if (!std::isfinite(p.presence_penalty) ||
        p.presence_penalty < -2.0f || p.presence_penalty > 2.0f) {
        set_error(error, "presence_penalty must be between -2 and 2");
        return false;
    }
    if (!std::isfinite(p.frequency_penalty) ||
        p.frequency_penalty < -2.0f || p.frequency_penalty > 2.0f) {
        set_error(error, "frequency_penalty must be between -2 and 2");
        return false;
    }
    if (error) error->clear();
    return true;
}

Sampler::Sampler(const SamplingParams& params) {
    configure(params, nullptr, true);
}

bool Sampler::configure(const SamplingParams& params, std::string* error,
                        bool reset_seed) {
    if (!validate_sampling_params(params, error))
        return false;
    params_ = params;
    if (reset_seed)
        random_state_ = params.seed;
    return true;
}

void Sampler::reset() {
    history_.clear();
    saved_logits_.clear();
    adjusted_logits_.clear();
    correction_probabilities_.clear();
    random_state_ = params_.seed;
}

void Sampler::accept(int token_id) {
    if (token_id < 0) return;
    history_.push_back(token_id);
}

void Sampler::accept(const std::vector<int>& token_ids) {
    for (int token_id : token_ids)
        accept(token_id);
}

bool Sampler::uses_plain_argmax() const {
    const bool penalties_enabled =
        params_.repeat_last_n != 0 &&
        (params_.repeat_penalty != 1.0f ||
         params_.presence_penalty != 0.0f ||
         params_.frequency_penalty != 0.0f);
    return !penalties_enabled &&
        (params_.temperature <= 0.0f || params_.top_k == 1);
}

int Sampler::sample(float* logits, int vocab_size) {
    const bool penalties_enabled =
        params_.repeat_last_n != 0 &&
        (params_.repeat_penalty != 1.0f ||
         params_.presence_penalty != 0.0f ||
         params_.frequency_penalty != 0.0f);
    if (!penalties_enabled) {
        return sample_token_impl(logits, vocab_size, params_.temperature,
                                 params_.top_k, params_.top_p, params_.min_p,
                                 &random_state_);
    }

    saved_logits_.clear();
    const size_t window =
        params_.repeat_last_n < 0
            ? history_.size()
            : std::min(history_.size(),
                       static_cast<size_t>(params_.repeat_last_n));
    std::unordered_map<int, int> counts;
    counts.reserve(window);
    const size_t begin = history_.size() - window;
    for (size_t i = begin; i < history_.size(); ++i) {
        const int token_id = history_[i];
        if (token_id >= 0 && token_id < vocab_size)
            ++counts[token_id];
    }

    saved_logits_.reserve(counts.size());
    for (const auto& [token_id, count] : counts) {
        saved_logits_.push_back({token_id, logits[token_id]});
        float& logit = logits[token_id];
        if (params_.repeat_penalty != 1.0f) {
            logit = logit > 0.0f ? logit / params_.repeat_penalty
                                 : logit * params_.repeat_penalty;
        }
        logit -= params_.presence_penalty;
        logit -= params_.frequency_penalty * count;
    }

    const int token =
        sample_token_impl(logits, vocab_size, params_.temperature,
                          params_.top_k, params_.top_p, params_.min_p,
                          &random_state_);
    for (const auto& [token_id, logit] : saved_logits_)
        logits[token_id] = logit;
    return token;
}

void Sampler::probabilities(const float* logits, int vocab_size,
                            const std::vector<int>& extra_history,
                            std::vector<float>& output) {
    if (!logits || vocab_size <= 0) {
        output.clear();
        return;
    }

    const bool penalties_enabled =
        params_.repeat_last_n != 0 &&
        (params_.repeat_penalty != 1.0f ||
         params_.presence_penalty != 0.0f ||
         params_.frequency_penalty != 0.0f);
    const float* filtered_logits = logits;
    if (penalties_enabled) {
        adjusted_logits_.assign(logits, logits + vocab_size);
        const size_t combined_size = history_.size() + extra_history.size();
        const size_t window = params_.repeat_last_n < 0
            ? combined_size
            : std::min(combined_size,
                       static_cast<size_t>(params_.repeat_last_n));
        const size_t begin = combined_size - window;
        std::unordered_map<int, int> counts;
        counts.reserve(window);
        for (size_t i = begin; i < combined_size; ++i) {
            const int token_id = i < history_.size()
                ? history_[i]
                : extra_history[i - history_.size()];
            if (token_id >= 0 && token_id < vocab_size)
                ++counts[token_id];
        }
        for (const auto& [token_id, count] : counts) {
            float& logit = adjusted_logits_[static_cast<size_t>(token_id)];
            if (params_.repeat_penalty != 1.0f) {
                logit = logit > 0.0f ? logit / params_.repeat_penalty
                                     : logit * params_.repeat_penalty;
            }
            logit -= params_.presence_penalty;
            logit -= params_.frequency_penalty * count;
        }
        filtered_logits = adjusted_logits_.data();
    }

    probability_distribution_impl(
        filtered_logits, vocab_size, params_.temperature, params_.top_k,
        params_.top_p, params_.min_p, output);
}

int Sampler::sample_probabilities(
    const std::vector<float>& probabilities) {
    double sum = 0.0;
    for (float probability : probabilities) {
        if (probability > 0.0f && std::isfinite(probability))
            sum += probability;
    }
    if (!(sum > 0.0) || !std::isfinite(sum))
        return 0;

    const double target = static_cast<double>(random_uniform()) * sum;
    double cumulative = 0.0;
    int last = 0;
    for (size_t i = 0; i < probabilities.size(); ++i) {
        const float probability = probabilities[i];
        if (!(probability > 0.0f) || !std::isfinite(probability))
            continue;
        cumulative += probability;
        last = static_cast<int>(i);
        if (target <= cumulative)
            return last;
    }
    return last;
}

int Sampler::speculative_sample(
    const std::vector<float>& target,
    const std::vector<float>& proposal,
    int proposed_token, bool* accepted) {
    if (accepted) *accepted = false;
    if (target.size() != proposal.size() || target.empty() ||
        proposed_token < 0 ||
        static_cast<size_t>(proposed_token) >= target.size()) {
        return sample_probabilities(target);
    }

    const float p = target[static_cast<size_t>(proposed_token)];
    const float q = proposal[static_cast<size_t>(proposed_token)];
    const float acceptance = q > 0.0f
        ? std::min(1.0f, p / q)
        : 0.0f;
    if (random_uniform() <= acceptance) {
        if (accepted) *accepted = true;
        return proposed_token;
    }

    correction_probabilities_.resize(target.size());
    for (size_t token = 0; token < target.size(); ++token) {
        correction_probabilities_[token] =
            std::max(0.0f, target[token] - proposal[token]);
    }
    return sample_probabilities(correction_probabilities_);
}

float Sampler::random_uniform() {
    return next_uniform(&random_state_);
}

int sample_token(float* logits, int vocab_size, float temperature, int top_k,
                 float top_p, unsigned int* seed) {
    return sample_token_impl(logits, vocab_size, temperature, top_k, top_p,
                             0.0f, seed);
}
