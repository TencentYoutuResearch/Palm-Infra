#include "engine/engine.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

static bool run(const char* path, Device device, const char* companion = nullptr) {
    EngineConfig config;
    config.package_path = path;
    config.device = device;
    config.device_fallback = DeviceFallbackPolicy::REQUIRE_REQUESTED;
    config.operator_fallback = OperatorFallbackPolicy::REQUIRE_NATIVE;
    config.num_threads = 1;
    config.n_ctx = 64;
    config.sampling.temperature = 0;
    LLMEngine reference, speculative;
    if (!reference.load(config)) return false;
    config.mtp_draft_tokens = 3;
    if (companion) config.mtp_companion_dir = companion;
    if (!speculative.load(config)) return false;
    for (int repeat = 0; repeat < 2; ++repeat) {
        reference.reset();
        speculative.reset();
        int current = reference.prefill({1, 2, 3});
        if (current < 0 || speculative.prefill({1, 2, 3}) != current)
            return false;
        // Ordinary decode must synchronize the same retained state contract
        // before switching back to speculative generation.
        int first = reference.decode(current);
        if (first < 0 || speculative.decode(current) != first) return false;
        current = first;
        for (int step = 0; step < 8; ++step) {
            std::vector<int> outputs;
            if (!speculative.speculative_decode(current, outputs, 4) ||
                outputs.empty()) return false;
            std::vector<int> retained{current};
            retained.insert(retained.end(), outputs.begin(), outputs.end() - 1);
            std::vector<float> logits;
            Tensor reference_hidden = reference.prefill_hidden(retained, &logits);
            if (reference_hidden.data && logits.empty())
                logits = reference.run_lmhead_raw(reference_hidden, retained.size(), true);
            if (!reference_hidden.data || logits.size() != outputs.size() * 32)
                return false;
            for (size_t i = 0; i < outputs.size(); ++i) {
                int output = outputs[i];
                const auto start = logits.begin() + i * 32;
                int expected = std::max_element(start, start + 32) - start;
                if (expected != output) {
                    std::fprintf(stderr, "MTP output mismatch: %d != %d\n", output, expected);
                    return false;
                }
                current = output;
            }
            if (reference.past_len() != speculative.past_len()) return false;
        }
        const auto stats = speculative.mtp_stats();
        const char* snapshot_option=std::getenv("MOLLM_CUDA_MTP_DEVICE_CHECKPOINT");
        if (device==Device::CUDA && snapshot_option && !std::strcmp(snapshot_option,"1") &&
            stats.recurrent_device_checkpoint_bytes!=stats.recurrent_checkpoint_bytes) {
            std::fprintf(stderr,"CUDA MTP did not use device checkpoints for every state\n");
            return false;
        }
        if (!stats.recurrent_rollbacks || !stats.recurrent_replay_tokens ||
            !stats.recurrent_checkpoint_bytes) return false;
        // The next hidden state tests more than matching greedy IDs: a
        // corrupted recurrent state may leave argmax unchanged for a while.
        Tensor expected = reference.decode_hidden(current);
        Tensor actual = speculative.decode_hidden(current);
        if (!expected.data || !actual.data || expected.nelements() != actual.nelements())
            return false;
        float max_error = 0;
        for (size_t i = 0; i < expected.nelements(); ++i) {
            float a = actual.ptr<float>()[i], b = expected.ptr<float>()[i];
            if (!std::isfinite(a) || !std::isfinite(b)) return false;
            max_error = std::max(max_error, std::fabs(a - b));
        }
        if (max_error > 2e-4f) {
            std::fprintf(stderr, "MTP subsequent hidden max error: %g\n", max_error);
            return false;
        }
        std::printf("device=%d repeat=%d rollbacks=%llu replay=%llu hidden_error=%g\n",
                    static_cast<int>(device), repeat,
                    static_cast<unsigned long long>(stats.recurrent_rollbacks),
                    static_cast<unsigned long long>(stats.recurrent_replay_tokens), max_error);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc == 4 && std::strcmp(argv[3], "--reject-companion") == 0) {
        EngineConfig config;
        config.package_path = argv[1]; config.n_ctx = 64; config.num_threads = 1;
        config.mtp_draft_tokens = 3; config.mtp_companion_dir = argv[2];
        LLMEngine engine;
        if (engine.load(config)) return 1;
        // Failed companion loads must release everything and allow reuse.
        config.mtp_draft_tokens = 0; config.mtp_companion_dir.clear();
        return engine.load(config) ? 0 : 1;
    }
    if (argc != 2 && argc != 3) return 1;
    const char* companion = argc == 3 ? argv[2] : nullptr;
    if (!run(argv[1], Device::CPU, companion)) return 1;
#ifdef MOLLM_CUDA
    if (!run(argv[1], Device::CUDA, companion)) return 1;
#endif
    return 0;
}
