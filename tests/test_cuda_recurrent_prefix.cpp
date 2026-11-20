#include "kernels/cuda/recurrent.h"

#include "cuda_test_support.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using mollm_test::check;
using mollm_test::GuardedBuffer;
using mollm_test::require;

// Prefix and output buffers carry `tail` extra elements preset to `sentinel`.
constexpr float sentinel = -123.f;
constexpr size_t tail = 8;

void copy_to_device(float* device, const float* host, size_t count) {
    check(cudaMemcpy(device, host, count * sizeof(float), cudaMemcpyHostToDevice), "host-to-device copy");
}

void copy_to_host(float* host, const float* device, size_t count) {
    check(cudaMemcpy(host, device, count * sizeof(float), cudaMemcpyDeviceToHost), "device-to-host copy");
}

void copy_on_device(float* destination, const float* source, size_t count) {
    check(cudaMemcpy(destination, source, count * sizeof(float), cudaMemcpyDeviceToDevice),
          "device-to-device copy");
}

bool same_floats(const float* a, const float* b, size_t count) {
    return !std::memcmp(a, b, count * sizeof(float));
}

bool tail_intact(const std::vector<float>& values, size_t count) {
    for (size_t i = count; i < values.size(); ++i) {
        if (values[i] != sentinel) return false;
    }
    return true;
}

bool shortconv_fixture(int groups, int sequence, int kernel, int real) {
    const size_t state_count = size_t(groups) * (kernel - 1);
    std::vector<float> input(size_t(groups) * sequence);
    std::vector<float> weight(size_t(groups) * kernel);
    std::vector<float> initial(state_count);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = float(int(i * 13 % 101) - 50) / 137.f;
    for (size_t i = 0; i < weight.size(); ++i)
        weight[i] = float(int(i * 7 % 31) - 15) / 53.f;
    for (size_t i = 0; i < initial.size(); ++i)
        initial[i] = float(int(i * 3 % 17) - 8) / 29.f;

    GuardedBuffer x_buffer(input.size() * sizeof(float));
    GuardedBuffer w_buffer(weight.size() * sizeof(float));
    GuardedBuffer state_buffer(state_count * sizeof(float));
    GuardedBuffer output_buffer((input.size() + tail) * sizeof(float));
    GuardedBuffer prefix_buffer((state_count * real + tail) * sizeof(float));
    float* x = x_buffer.as<float>();
    float* w = w_buffer.as<float>();
    float* state = state_buffer.as<float>();
    float* output = output_buffer.as<float>();
    float* prefix = prefix_buffer.as<float>();
    x_buffer.upload(input.data());
    w_buffer.upload(weight.data());

    std::vector<float> expected(state_count * real);
    std::vector<float> expected_output(input.size(), 0.f);
    std::vector<float> one(groups);
    // Consumes one token from the device state.
    const auto step = [&](int token) {
        return mollm_cuda::launch_shortconv(x + size_t(token) * groups, w, state, output, groups, 1, kernel,
                                            1, 1, groups);
    };
    // Serial single-token launches supply the reference states and outputs.
    const auto serial_reference = [&] {
        copy_to_device(state, initial.data(), state_count);
        for (int t = 0; t < real; ++t) {
            if (!step(t)) return false;
            copy_to_host(expected.data() + size_t(t) * state_count, state, state_count);
            copy_to_host(one.data(), output, groups);
            for (int g = 0; g < groups; ++g)
                expected_output[size_t(g) * sequence + t] = one[g];
        }
        return true;
    };
    if (!serial_reference()) return false;

    std::vector<float> got(expected.size() + tail, sentinel);
    std::vector<float> got_output(input.size() + tail, sentinel);
    std::vector<float> final_state(state_count);
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal), "begin capture");
    if (!mollm_cuda::launch_shortconv_prefixes(x, w, state, output, prefix, groups, sequence, kernel, real, 1,
                                               groups))
        return false;
    check(cudaStreamEndCapture(cudaStreamPerThread, &graph), "end capture");
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "graph instantiate");
    for (int replay = 0; replay < 2; ++replay) {
        if (replay) {
            for (float& value : input)
                value = -value;
            x_buffer.upload(input.data());
            if (!serial_reference()) return false;
        }
        copy_to_device(state, initial.data(), state_count);
        std::fill(got.begin(), got.end(), sentinel);
        std::fill(got_output.begin(), got_output.end(), sentinel);
        copy_to_device(prefix, got.data(), got.size());
        copy_to_device(output, got_output.data(), got_output.size());
        check(cudaGraphLaunch(executable, cudaStreamPerThread), "graph launch");
        check(cudaStreamSynchronize(cudaStreamPerThread), "graph");
        copy_to_host(got.data(), prefix, got.size());
        copy_to_host(got_output.data(), output, got_output.size());
        copy_to_host(final_state.data(), state, state_count);
        if (!same_floats(got.data(), expected.data(), expected.size()) ||
            !same_floats(got_output.data(), expected_output.data(), expected_output.size()) ||
            !same_floats(final_state.data(), expected.data() + size_t(real - 1) * state_count, state_count)) {
            std::fprintf(stderr, "prefix mismatch groups=%d seq=%d kernel=%d real=%d\n", groups, sequence,
                         kernel, real);
            return false;
        }
        if (!tail_intact(got, expected.size()) || !tail_intact(got_output, expected_output.size()))
            return false;
        // Restore each accepted prefix, consume the next token and compare
        // against the serial reference.
        for (int kept = 1; kept < real; ++kept) {
            copy_on_device(state, prefix + size_t(kept - 1) * state_count, state_count);
            if (!step(kept)) return false;
            copy_to_host(final_state.data(), state, state_count);
            copy_to_host(one.data(), output, groups);
            if (!same_floats(final_state.data(), expected.data() + size_t(kept) * state_count, state_count))
                return false;
            for (int group = 0; group < groups; ++group) {
                if (!same_floats(&one[group], &expected_output[size_t(group) * sequence + kept], 1))
                    return false;
            }
        }
    }
    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    return true;
}

bool gdn_fixture(int sequence, int real, int flags, int heads, int values) {
    const int width = 2 * heads * 128 + values * 128;
    const int output_width = values * 128;
    const size_t state_count = size_t(values) * 128 * 128;
    std::vector<float> qkv(size_t(sequence) * width);
    std::vector<float> a(sequence * values);
    std::vector<float> b(a.size());
    std::vector<float> z(sequence * output_width);
    std::vector<float> alog(values, -1.f);
    std::vector<float> bias(values, .1f);
    std::vector<float> norm(128, 1.f);
    std::vector<float> initial(state_count);
    for (size_t i = 0; i < qkv.size(); ++i)
        qkv[i] = float(int(i * 7 % 61) - 30) / 83.f;
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = float(int(i % 7) - 3) / 11.f;
        b[i] = float(int(i % 11) - 5) / 13.f;
    }
    for (size_t i = 0; i < z.size(); ++i)
        z[i] = float(int(i % 13) - 6) / 17.f;
    for (size_t i = 0; i < initial.size(); ++i)
        initial[i] = float(int(i % 17) - 8) / 101.f;
    std::vector<float> one_qkv(width);

    GuardedBuffer qkv_buffer(qkv.size() * sizeof(float));
    GuardedBuffer single_qkv_buffer(one_qkv.size() * sizeof(float));
    GuardedBuffer a_buffer(a.size() * sizeof(float));
    GuardedBuffer b_buffer(b.size() * sizeof(float));
    GuardedBuffer z_buffer(z.size() * sizeof(float));
    GuardedBuffer alog_buffer(alog.size() * sizeof(float));
    GuardedBuffer bias_buffer(bias.size() * sizeof(float));
    GuardedBuffer norm_buffer(norm.size() * sizeof(float));
    GuardedBuffer state_buffer(initial.size() * sizeof(float));
    GuardedBuffer output_buffer((size_t(sequence) * output_width + tail) * sizeof(float));
    GuardedBuffer prefix_buffer((size_t(real) * state_count + tail) * sizeof(float));
    qkv_buffer.upload(qkv.data());
    single_qkv_buffer.upload(one_qkv.data());
    a_buffer.upload(a.data());
    b_buffer.upload(b.data());
    z_buffer.upload(z.data());
    alog_buffer.upload(alog.data());
    bias_buffer.upload(bias.data());
    norm_buffer.upload(norm.data());
    state_buffer.upload(initial.data());
    float* q = qkv_buffer.as<float>();
    float* single_q = single_qkv_buffer.as<float>();
    float* da = a_buffer.as<float>();
    float* db = b_buffer.as<float>();
    float* dz = z_buffer.as<float>();
    float* dl = alog_buffer.as<float>();
    float* dt = bias_buffer.as<float>();
    float* dn = norm_buffer.as<float>();
    float* state = state_buffer.as<float>();
    float* output = output_buffer.as<float>();
    float* prefix = prefix_buffer.as<float>();

    // GDN consumes convolution's feature-major QKV, not token-major input.
    const auto upload_token = [&](int token) {
        for (int feature = 0; feature < width; ++feature)
            one_qkv[feature] = qkv[size_t(feature) * sequence + token];
        copy_to_device(single_q, one_qkv.data(), width);
    };
    const auto decode_token = [&](int token) {
        return mollm_cuda::launch_gdn(single_q, da + token * values, db + token * values,
                                      dz + token * output_width, dl, dt, dn, state, output, heads, values,
                                      128, 128, 1, 1, flags, 1e-6f, 1e-6f, .125f, values, values,
                                      output_width);
    };
    const auto launch_prefixes = [&] {
        return mollm_cuda::launch_gdn_prefixes(q, da, db, dz, dl, dt, dn, state, output, prefix, heads,
                                               values, sequence, real, flags, 1e-6f, 1e-6f, .125f, values,
                                               values, output_width);
    };
    std::vector<float> expected(size_t(real) * state_count);
    std::vector<float> expected_output(size_t(sequence) * output_width, 0.f);
    // Serial single-token decode supplies the reference states and outputs.
    const auto serial_reference = [&] {
        copy_to_device(state, initial.data(), state_count);
        for (int token = 0; token < real; ++token) {
            upload_token(token);
            if (!decode_token(token)) return false;
            copy_to_host(expected.data() + size_t(token) * state_count, state, state_count);
            copy_to_host(expected_output.data() + size_t(token) * output_width, output, output_width);
        }
        return true;
    };
    std::vector<float> got(expected.size() + tail, sentinel);
    std::vector<float> actual(expected_output.size() + tail, sentinel);
    std::vector<float> final_state(state_count);
    const auto matches_reference = [&] {
        return same_floats(expected.data(), got.data(), expected.size()) &&
               same_floats(expected_output.data(), actual.data(), expected_output.size()) &&
               same_floats(final_state.data(), expected.data() + size_t(real - 1) * state_count, state_count);
    };
    const auto read_results = [&] {
        copy_to_host(got.data(), prefix, got.size());
        copy_to_host(actual.data(), output, actual.size());
        copy_to_host(final_state.data(), state, state_count);
    };

    // Eager launch.
    if (!serial_reference()) return false;
    copy_to_device(state, initial.data(), state_count);
    copy_to_device(prefix, got.data(), got.size());
    copy_to_device(output, actual.data(), actual.size());
    if (!launch_prefixes()) return false;
    read_results();
    if (!matches_reference()) {
        std::fprintf(stderr, "GDN prefix mismatch seq=%d real=%d flags=%d heads=%d values=%d\n", sequence,
                     real, flags, heads, values);
        return false;
    }
    if (!tail_intact(got, expected.size()) || !tail_intact(actual, expected_output.size())) return false;

    // Captured launch, replayed with the original and then a changed input.
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal), "begin capture");
    if (!launch_prefixes()) return false;
    check(cudaStreamEndCapture(cudaStreamPerThread, &graph), "end capture");
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "graph instantiate");
    for (int replay = 0; replay < 2; ++replay) {
        if (replay) {
            for (float& value : qkv)
                value = -value;
            qkv_buffer.upload(qkv.data());
            if (!serial_reference()) return false;
        }
        copy_to_device(state, initial.data(), state_count);
        std::fill(got.begin(), got.end(), sentinel);
        std::fill(actual.begin(), actual.end(), sentinel);
        copy_to_device(prefix, got.data(), got.size());
        copy_to_device(output, actual.data(), actual.size());
        check(cudaGraphLaunch(executable, cudaStreamPerThread), "graph launch");
        check(cudaStreamSynchronize(cudaStreamPerThread), "graph");
        read_results();
        if (!matches_reference()) {
            std::fprintf(stderr, "GDN prefix graph mismatch replay=%d\n", replay);
            return false;
        }
        if (!tail_intact(got, expected.size()) || !tail_intact(actual, expected_output.size())) return false;
        // Restore each accepted prefix, then consume the next token. Compare
        // against the independently generated serial post-prefix state.
        for (int kept = 1; kept < real; ++kept) {
            upload_token(kept);
            copy_to_device(state, expected.data() + size_t(kept - 1) * state_count, state_count);
            if (!decode_token(kept)) return false;
            std::vector<float> next_state(state_count);
            std::vector<float> next_output(output_width);
            std::vector<float> restored_output(output_width);
            copy_to_host(next_state.data(), state, state_count);
            copy_to_host(next_output.data(), output, output_width);
            copy_on_device(state, prefix + size_t(kept - 1) * state_count, state_count);
            if (!decode_token(kept)) return false;
            copy_to_host(final_state.data(), state, state_count);
            copy_to_host(restored_output.data(), output, output_width);
            if (!same_floats(next_state.data(), final_state.data(), state_count) ||
                !same_floats(next_output.data(), restored_output.data(), output_width))
                return false;
        }
    }
    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    // --gdn-sanitizer skips shortconv and runs one wide GDN sequence, which
    // keeps compute-sanitizer runs short.
    const bool sanitizer_subset = argc == 2 && std::strcmp(argv[1], "--gdn-sanitizer") == 0;
    if (argc != 1 && !sanitizer_subset) {
        std::fprintf(stderr, "usage: %s [--gdn-sanitizer]\n", argv[0]);
        return 2;
    }
    if (!mollm_test::cuda_device_available()) return mollm_test::skip_status;
    require(!mollm_cuda::launch_shortconv_prefixes(nullptr, nullptr, nullptr, nullptr, nullptr, 7, 2, 4, 2, 1,
                                                   7),
            "null shortconv prefix arguments accepted");

    int shortconv_fixtures = 0;
    if (!sanitizer_subset) {
        for (int groups : {7, 257}) {
            for (int sequence : {2, 3, 4, 5, 6, 7, 8}) {
                for (int kernel : {2, 4, 8}) {
                    for (int real : {sequence - 1, sequence}) {
                        if (!shortconv_fixture(groups, sequence, kernel, real)) return 1;
                        ++shortconv_fixtures;
                    }
                }
            }
        }
    }

    int gdn_fixtures = 0;
    for (int sequence : {2, 3, 4, 5, 6, 7, 8}) {
        for (int real : {sequence - 1, sequence}) {
            for (int flags : {0, 1, 2, 3}) {
                for (int heads : {1, 2}) {
                    if (!gdn_fixture(sequence, real, flags, heads, heads * 2)) return 1;
                    ++gdn_fixtures;
                }
            }
        }
    }
    const std::vector<int> wide_sequences =
        sanitizer_subset ? std::vector<int>{3} : std::vector<int>{2, 3, 4, 8};
    for (int sequence : wide_sequences) {
        for (int real : {sequence - 1, sequence}) {
            for (int flags : {1, 3}) {
                if (!gdn_fixture(sequence, real, flags, 16, 48)) return 1;
                ++gdn_fixtures;
            }
        }
    }
    std::printf("%d shortconv and %d GDN prefix-state/output/guard fixtures passed; "
                "GDN graph replay/changed input exact\n",
                shortconv_fixtures, gdn_fixtures);
    return 0;
}
