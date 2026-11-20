#pragma once

#include <cmath>

namespace mollm::detail {

// Proposal policy only; target verification remains authoritative. The first
// draft is always proposed. Missing confidence preserves the fixed-depth path.
// A higher continuation floor can stop after proposing the current draft;
// it never changes the proposal decision for that computed probability prefix.
// Default greedy CUDA schedule: stop proposing once the product of draft
// probabilities drops below 0.4, and stop drafting further below 0.5.
inline constexpr double mtp_draft_confidence_floor = 0.4;
inline constexpr double mtp_draft_continuation_floor = 0.5;

class MtpDraftPolicy {
public:
    struct Decision { bool propose; bool more; };
    explicit MtpDraftPolicy(double floor, double continuation_floor = 0.0)
        : floor_(floor),
          continuation_floor_(std::isfinite(continuation_floor) && continuation_floor >= floor &&
                                      continuation_floor <= 1.0
                                  ? continuation_floor
                                  : floor),
          enabled_(std::isfinite(floor) && floor > 0.0 && floor <= 1.0) {}
    bool needs_probability() const { return enabled_; }
    Decision take(int depth, float probability) {
        if (!enabled_) return {true, true};
        if (!std::isfinite(probability) || probability <= 0.0f || probability > 1.0f) {
            enabled_ = false;
            return {true, true};
        }
        product_ *= probability;
        const bool low = product_ < floor_;
        return {depth == 0 || !low, product_ >= continuation_floor_};
    }
private:
    double floor_, continuation_floor_, product_ = 1.0;
    bool enabled_;
};

}  // namespace mollm::detail
