#include "engine/mtp_draft_policy.h"
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>

static void require(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "MTP policy: %s\n", message); std::exit(1); }
}
int main() {
    using mollm::detail::MtpDraftPolicy;
    MtpDraftPolicy product(.25);
    auto first=product.take(0,.5f);
    require(first.propose&&first.more,"first draft above floor");
    auto boundary=product.take(1,.5f);
    require(boundary.propose&&boundary.more,"exact product boundary is included");
    auto low=product.take(2,.5f);
    require(!low.propose&&!low.more,"individual confidence above floor cannot hide low product");
    MtpDraftPolicy first_low(.5);
    auto forced=first_low.take(0,.25f);
    require(forced.propose&&!forced.more,"first low-confidence draft remains proposed");
    MtpDraftPolicy new_round(.25);
    require(new_round.take(0,.5f).more,"new round resets running product");
    for(float unavailable:{-1.f,0.f,2.f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}) {
        MtpDraftPolicy fallback(.5);
        auto decision=fallback.take(0,unavailable);
        require(decision.propose&&decision.more&&!fallback.needs_probability(),"unknown confidence disables policy for round");
        require(fallback.take(1,.01f).more,"fallback preserves later fixed-depth drafts");
    }
    for(double disabled:{0.0,-.1,1.1,std::numeric_limits<double>::infinity()}) {
        MtpDraftPolicy fixed(disabled);
        require(!fixed.needs_probability(),"invalid or disabled floor makes no probability request");
        auto decision=fixed.take(3,.01f);
        require(decision.propose&&decision.more,"disabled policy preserves fixed depth");
    }
    MtpDraftPolicy continuation(.25, .5);
    require(continuation.take(0, .5f).more, "continuation equality stays enabled");
    auto stop_after_current = continuation.take(1, .5f);
    require(stop_after_current.propose && !stop_after_current.more,
            "proposal equality retained while higher continuation floor stops");
    MtpDraftPolicy continuation_first(.5, .75);
    auto first_below_both = continuation_first.take(0, .1f);
    require(first_below_both.propose && !first_below_both.more,
            "continuation preserves forced first proposal");
    for (double more : {0., -.1, .1, .25, 2., std::numeric_limits<double>::quiet_NaN()}) {
        MtpDraftPolicy original(.25), compatible(.25, more);
        for (float p : {.5f, .5f, .5f, -1.f, .1f}) {
            auto a=original.take(1,p), b=compatible.take(1,p);
            require(a.propose==b.propose && a.more==b.more,
                    "default, lower and invalid continuation floors preserve legacy behavior");
        }
    }
    for (float unknown : {-1.f, 0.f, 2.f, std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity()}) {
        MtpDraftPolicy fallback(.25, .5);
        auto decision=fallback.take(0,unknown);
        require(decision.propose && decision.more && !fallback.needs_probability(),
                "continuation preserves unavailable-confidence fallback");
        require(fallback.take(1,.01f).more,"fixed-depth continuation after fallback");
    }
    MtpDraftPolicy disabled_more(0., .5);
    require(!disabled_more.needs_probability() && disabled_more.take(2,.01f).more,
            "continuation alone cannot enable confidence policy");
    for (unsigned seed=1; seed<=200; ++seed) {
        MtpDraftPolicy original(.25), earlier(.25,.5);
        unsigned state=seed;
        for (int depth=0; depth<16; ++depth) {
            state=1664525u*state+1013904223u;
            float probability=float(state>>8)/16777216.f;
            auto a=original.take(depth,probability), b=earlier.take(depth,probability);
            require(a.propose==b.propose,"same-prefix proposal decisions remain identical");
            require(!b.more || a.more,"continuation only stops earlier");
        }
    }
    std::puts("MTP policy: product/boundary/first-draft/reset/unavailable/default/continuation behavior passed");
}
