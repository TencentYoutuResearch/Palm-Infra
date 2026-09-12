#include "runtime/expert_provider.h"
#include "runtime/threading.h"
#include "kernels/cpu/moe/moe.h"

#include <array>
#include <cmath>
#include <cstdio>

namespace {
int failures = 0;
void check(bool ok, const char* label) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); ++failures; }
}

// Deliberately has no SSD descriptors, package files, or cache implementation.
class MemoryExperts final : public ExpertProvider {
public:
    ExpertSource gate_source, down_source;
    std::array<int, 2> pins{};
    int released = 0;
    bool lose_readiness = false;
    bool fail_second = false;
    float gate_weights[2][2] = {{1, 2}, {2, 1}};
    float down_weights[2] = {1, 3};

    MemoryExperts() {
        gate_source.provider = down_source.provider = this;
        gate_source.layer = down_source.layer = 0;
    }
    bool borrow(const ExpertSource* gate, const ExpertSource* down, int expert,
                ExpertLease& lease, bool wait) override {
        lease.reset();
        if (gate != &gate_source || down != &down_source || expert < 0 || expert > 1)
            return false;
        if (expert == 1 && (fail_second || (!wait && lose_readiness))) return false;
        ++pins[expert];
        Tensor gu = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                    2, 1, 1, 1, gate_weights[expert]);
        Tensor dw = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                    1, 1, 1, 1, &down_weights[expert]);
        lease = ExpertLease(*this, &pins[expert], gu, dw);
        return true;
    }
    bool request_many(const ExpertSource*, const ExpertSource*,
                      const std::vector<int>&) override { return true; }
    bool retain_for_next_forward(const ExpertSource*, const ExpertSource*,
                                 const std::vector<int>&, bool) override { return true; }
    size_t resident_count(const ExpertSource*, const ExpertSource*,
                          const std::vector<int>&) const override { return 2; }
    bool contains(const ExpertSource*, const ExpertSource*, int) const override { return true; }
    bool is_ready(const ExpertSource*, const ExpertSource*, int) const override { return true; }
    bool evict(const ExpertSource*, const ExpertSource*, int) override { return false; }
private:
    void release_lease(void* token) noexcept override {
        --*static_cast<int*>(token);
        ++released;
    }
};

void run_case(bool threaded, bool readiness_changed, bool read_failed) {
    MemoryExperts provider;
    provider.lose_readiness = readiness_changed;
    provider.fail_second = read_failed;
    float x = 1, router_values[2] = {0, 0}, result = 0;
    Tensor input = Tensor::create(Precision::FP32, MemoryType::EXTERNAL, 1, 1, 1, 1, &x);
    Tensor router = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                   2, 1, 1, 1, router_values);
    Tensor gate = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                 2, 2, 1, 1, nullptr);
    Tensor down = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                 2, 1, 1, 1, nullptr);
    gate.moe_ssd_source = &provider.gate_source;
    down.moe_ssd_source = &provider.down_source;
    Tensor output = Tensor::create(Precision::FP32, MemoryType::EXTERNAL,
                                   1, 1, 1, 1, &result);
    ThreadPool pool(2);
    bool ok = kernel_qwen3_moe({&input, &router, &gate, &down}, output,
                               threaded ? &pool : nullptr,
                               1, 2, 2, 1, 0, 0, true, false);
    check(ok == !read_failed, "MoE propagates provider success or read failure");
    if (ok) {
        const float expected = 1.f / (1.f + std::exp(-1.f)) +
                               3.f / (1.f + std::exp(-2.f));
        check(std::fabs(result - expected) < 2e-3f,
              "memory provider matches analytical two-expert output");
    }
    check(provider.pins[0] == 0 && provider.pins[1] == 0,
          "success, partial batch and read failure all release their leases");
    if (threaded && readiness_changed && !read_failed)
        check(provider.released == 3,
              "partial ready batch releases its pin before sequential fallback");
}
}  // namespace

int main() {
    run_case(false, false, false);
    run_case(true, false, false);
    run_case(true, true, false);
    run_case(true, true, true);
    if (!failures) std::puts("All expert provider tests passed!");
    return failures ? 1 : 0;
}
