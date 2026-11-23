// Load-time gate views must decode exactly like the canonical payload.
//
// launch_repack_nvfp4_gate_codes turns canonical ROW/PAIR gate codes into the
// K64-major code view and launch_repack_nvfp4_gate_scales turns row-major block
// scales into the K-major scale view. Every consumer of those views must be
// byte-exact with the canonical payload: plain and pipelined SIMT, the subgroup
// gate entry and the MMA path (1/2/8 split-K parts, pre-quantized inputs).
// Each run checks both views: the code view against the canonical layout, and
// the scale view against row-major scales with both arms on the code view.
// The shared gate launcher fixture lives in test_cuda_nvfp4_gate_projections.cpp.
//
// Usage: test_cuda_nvfp4_gate_code_decode [--routes N[,N...]] [--timing]
//   N is 10..60 in steps of 10 (default 10). --timing also prints pipelined
//   SIMT timings for the canonical and view arms (developer measurement).

#include "kernels/cuda/moe.h"

#include "cuda_test_support.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using Job = mollm_cuda::Nvfp4Projection;
using mollm_test::check;
using mollm_test::require;
using mollm_test::same_bits;

constexpr int rows = 1280;
constexpr int inner = 2560;
// Two graphs over disjoint expert sets; each replay rotates the routed experts.
constexpr int sets = 2;
constexpr int variants = 8;
constexpr int split_row = 640;
constexpr size_t code_bytes = size_t(rows) * inner / 2;
constexpr size_t scale_bytes = size_t(rows) * inner / 16;
// One canonical expert payload: codes, row-major block scales, FP32 row scales.
constexpr size_t expert_stride = code_bytes + scale_bytes + rows * sizeof(float);

// Typed, stream-ordered view of a GuardedBuffer. The test stream is
// non-blocking, so every copy and fill is issued on it explicitly.
template<class T> class DeviceArray {
public:
    explicit DeviceArray(size_t count) : buffer_(count * sizeof(T)), count_(count) {}

    T* data() { return buffer_.as<T>(); }
    const T* data() const { return buffer_.as<T>(); }
    size_t size() const { return count_; }

    void upload(const std::vector<T>& host, cudaStream_t stream) {
        require(host.size() == count_, "upload size mismatch");
        check(cudaMemcpyAsync(data(), host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice, stream),
              "upload");
        check(cudaStreamSynchronize(stream), "upload");
    }
    std::vector<T> download(cudaStream_t stream) const {
        std::vector<T> host(count_);
        check(cudaMemcpyAsync(host.data(), data(), count_ * sizeof(T), cudaMemcpyDeviceToHost, stream),
              "download");
        check(cudaStreamSynchronize(stream), "download");
        return host;
    }
    // Fills the payload with the guard pattern so unwritten outputs differ.
    void poison(cudaStream_t stream) {
        check(cudaMemsetAsync(data(), mollm_test::GuardedBuffer::guard_pattern, count_ * sizeof(T), stream),
              "poison");
    }
    void check_guards() const { buffer_.check_guards(); }

private:
    mollm_test::GuardedBuffer buffer_;
    size_t count_;
};

enum class Mode { simt, pipelined, mma1, mma2, mma8, mma_prepared, subgroup };

const Mode modes[] = {Mode::simt, Mode::pipelined, Mode::mma1, Mode::mma2,
                      Mode::mma8, Mode::mma_prepared, Mode::subgroup};

const char* mode_name(Mode mode) {
    switch (mode) {
    case Mode::simt: return "simt";
    case Mode::pipelined: return "pipelined";
    case Mode::mma1: return "mma1";
    case Mode::mma2: return "mma2";
    case Mode::mma8: return "mma8";
    case Mode::mma_prepared: return "mma_prepared";
    case Mode::subgroup: return "subgroup";
    }
    return "unknown";
}

bool is_mma(Mode mode) {
    return mode == Mode::mma1 || mode == Mode::mma2 || mode == Mode::mma8 || mode == Mode::mma_prepared;
}

std::vector<uint8_t> make_payload(bool pair, int experts) {
    const uint8_t scale_table[] = {0, 1, 7, 8, 48, 56, 64, 126};
    std::vector<uint8_t> payload(size_t(experts) * expert_stride);
    for (int expert = 0; expert < experts; ++expert) {
        uint8_t* p = payload.data() + size_t(expert) * expert_stride;
        for (int row = 0; row < rows; ++row) {
            const float row_scale = float(1 + (row + expert) % 7) / 64.f;
            std::memcpy(p + code_bytes + scale_bytes + size_t(row) * sizeof(float), &row_scale,
                        sizeof(float));
            for (int group = 0; group < inner / 16; ++group) {
                p[code_bytes + size_t(row) * (inner / 16) + group] = scale_table[(row + group + expert) % 8];
                for (int byte = 0; byte < 8; ++byte) {
                    const size_t offset = pair
                        ? size_t(row / 4) * (inner / 16) * 32 + group * 32 + (row % 4) * 8 + byte
                        : size_t(row) * (inner / 2) + group * 8 + byte;
                    p[offset] = uint8_t((row * 29 + group * 17 + byte * 13 + expert * 37) & 255);
                }
            }
        }
    }
    return payload;
}

class ViewCheck {
public:
    ViewCheck(int routes, cudaStream_t stream)
        : routes_(routes),
          stream_(stream),
          weights_(size_t(sets) * routes * expert_stride),
          code_view_(size_t(sets) * routes * code_bytes),
          kmajor_scales_(size_t(sets) * routes * scale_bytes),
          input_(size_t(sets) * routes * inner),
          baseline_(size_t(routes) * rows),
          candidate_(size_t(routes) * rows),
          partial_base_(size_t(routes) * rows * 8),
          partial_code_(size_t(routes) * rows * 8),
          prepared_(size_t(routes) * 2 * inner / 2),
          prepared_scales_(size_t(routes) * 2 * inner / 16),
          base_jobs_(size_t(sets) * routes),
          code_jobs_(size_t(sets) * routes) {}

    // Returns the number of checked graph replays.
    int run(bool scale_view, bool timing) {
        scale_view_enabled_ = scale_view;
        int replays = 0;
        for (bool pair : {false, true}) {
            const std::vector<uint8_t> payload = make_payload(pair, sets * routes_);
            weights_.upload(payload, stream_);
            repack(pair);
            const std::vector<uint8_t> code_view = code_view_.download(stream_);
            const std::vector<uint8_t> kmajor_scale_bytes = kmajor_scales_.download(stream_);
            for (Mode mode : modes) {
                replays += run_mode(pair, mode);
                if (timing && mode == Mode::pipelined) time_pipelined(pair);
            }
            fail_unless(same_bits(weights_.download(stream_), payload), pair, "canonical weights modified");
            fail_unless(same_bits(code_view_.download(stream_), code_view), pair, "code view modified");
            fail_unless(same_bits(kmajor_scales_.download(stream_), kmajor_scale_bytes), pair,
                        "scale view modified");
        }
        return replays;
    }

private:
    void fail_unless(bool ok, bool pair, const char* what, Mode mode = Mode::simt, int variant = -1,
                     int set = -1) const {
        if (ok) return;
        char message[256];
        std::snprintf(message, sizeof(message),
                      "%s (routes=%d scale_view=%d pair=%d mode=%s variant=%d set=%d)", what, routes_,
                      int(scale_view_enabled_), int(pair), mode_name(mode), variant, set);
        mollm_test::fail(message);
    }

    void repack(bool pair) {
        code_view_.poison(stream_);
        kmajor_scales_.poison(stream_);
        for (int expert = 0; expert < sets * routes_; ++expert) {
            Job source{};
            source.weight = weights_.data() + size_t(expert) * expert_stride;
            source.pair_layout = pair;
            check(mollm_cuda::launch_repack_nvfp4_gate_codes(
                      source, code_view_.data() + size_t(expert) * code_bytes, rows, inner, stream_),
                  "code repack");
            if (scale_view_enabled_) {
                source.block_scales = weights_.data() + size_t(expert) * expert_stride + code_bytes;
                check(mollm_cuda::launch_repack_nvfp4_gate_scales(
                          source, kmajor_scales_.data() + size_t(expert) * scale_bytes, rows, inner, stream_),
                      "scale repack");
            }
        }
        check(cudaStreamSynchronize(stream_), "repack");
        code_view_.check_guards();
        kmajor_scales_.check_guards();
    }

    // Baseline jobs read the canonical payload; candidate jobs read the views.
    void describe(bool pair, int variant) {
        std::vector<float> values(input_.size());
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = float(int((i * 17 + size_t(variant) * 31) % 257) - 128) / 128.f;
        input_.upload(values, stream_);
        std::vector<Job> base(size_t(sets) * routes_);
        std::vector<Job> code(size_t(sets) * routes_);
        for (int set = 0; set < sets; ++set) {
            for (int route = 0; route < routes_; ++route) {
                const int index = set * routes_ + route;
                const int expert = set * routes_ + (route * 3 + variant) % routes_;
                const uint8_t* payload = weights_.data() + size_t(expert) * expert_stride;
                const uint8_t* view = code_view_.data() + size_t(expert) * code_bytes;
                Job job{};
                job.weight = payload;
                job.block_scales = payload + code_bytes;
                job.row_scales = reinterpret_cast<const float*>(payload + code_bytes + scale_bytes);
                job.input = input_.data() + size_t(index) * inner;
                job.output = baseline_.data() + size_t(route) * rows;
                job.pair_layout = pair;
                job.input_global = float(1 + route % 3) / 16.f;
                job.input_global_second = float(1 + route % 5) / 16.f;
                job.output_scale = float(1 + route % 7) / 8.f;
                // Isolate the scale-layout change: with the scale view both
                // arms read the already-qualified code view.
                if (scale_view_enabled_) job.weight = view;
                base[index] = job;
                job.weight = view;
                if (scale_view_enabled_) {
                    job.block_scales = kmajor_scales_.data() + size_t(expert) * scale_bytes;
                    job.gate_scales_kmajor = true;
                }
                job.output = candidate_.data() + size_t(route) * rows;
                code[index] = job;
            }
        }
        base_jobs_.upload(base, stream_);
        code_jobs_.upload(code, stream_);
    }

    void launch(bool code, Mode mode, int set) {
        const Job* jobs = (code ? code_jobs_.data() : base_jobs_.data()) + set * routes_;
        // Candidate jobs always read the code view; baseline jobs do with the scale view.
        const bool view = code || scale_view_enabled_;
        if (is_mma(mode)) {
            const int parts = mode == Mode::mma2 ? 2 : mode == Mode::mma8 ? 8 : 1;
            const bool prepare = mode == Mode::mma_prepared;
            check(mollm_cuda::launch_nvfp4_mma_projections(
                      jobs, routes_, rows, inner, split_row, stream_,
                      code ? partial_code_.data() : partial_base_.data(), parts, true,
                      prepare ? prepared_.data() : nullptr, prepare ? prepared_scales_.data() : nullptr,
                      view),
                  "MMA launch");
        } else if (view && mode == Mode::subgroup) {
            check(mollm_cuda::launch_nvfp4_gate_code_subgroup_projections(jobs, routes_, rows, inner,
                                                                          stream_),
                  "subgroup launch");
        } else if (view) {
            check(mollm_cuda::launch_nvfp4_gate_code_projections(jobs, routes_, rows, inner,
                                                                 mode == Mode::pipelined, stream_),
                  "gate code launch");
        } else if (mode == Mode::pipelined) {
            check(mollm_cuda::launch_nvfp4_pipelined_projections(jobs, routes_, rows, inner, stream_),
                  "pipelined launch");
        } else {
            mollm_cuda::launch_nvfp4_projections(jobs, routes_, rows, inner, stream_);
            check(cudaGetLastError(), "SIMT launch");
        }
    }

    int run_mode(bool pair, Mode mode) {
        describe(pair, 0);
        cudaGraph_t graphs[sets];
        cudaGraphExec_t executables[sets];
        for (int set = 0; set < sets; ++set) {
            check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "begin capture");
            launch(false, mode, set);
            launch(true, mode, set);
            check(cudaStreamEndCapture(stream_, &graphs[set]), "end capture");
            check(cudaGraphInstantiate(&executables[set], graphs[set], nullptr, nullptr, 0),
                  "graph instantiate");
        }
        int replays = 0;
        for (int variant = 0; variant < variants; ++variant) {
            describe(pair, variant);
            for (int set = 0; set < sets; ++set) {
                baseline_.poison(stream_);
                candidate_.poison(stream_);
                check(cudaGraphLaunch(executables[set], stream_), "graph launch");
                const std::vector<float> expected = baseline_.download(stream_);
                fail_unless(same_bits(candidate_.download(stream_), expected), pair,
                            "view output differs from the canonical output", mode, variant, set);
                baseline_.check_guards();
                candidate_.check_guards();
                ++replays;
            }
        }
        for (int set = 0; set < sets; ++set) {
            check(cudaGraphExecDestroy(executables[set]), "graph exec destroy");
            check(cudaGraphDestroy(graphs[set]), "graph destroy");
        }
        return replays;
    }

    void time_pipelined(bool pair) {
        constexpr int graph_repeats = 16;
        constexpr int batches = 16;
        cudaGraph_t graphs[2];
        cudaGraphExec_t executables[2];
        for (int code = 0; code < 2; ++code) {
            check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "begin capture");
            for (int repeat = 0; repeat < graph_repeats; ++repeat) {
                for (int set = 0; set < sets; ++set) launch(code == 1, Mode::pipelined, set);
            }
            check(cudaStreamEndCapture(stream_, &graphs[code]), "end capture");
            check(cudaGraphInstantiate(&executables[code], graphs[code], nullptr, nullptr, 0),
                  "graph instantiate");
        }
        cudaEvent_t start;
        cudaEvent_t end;
        check(cudaEventCreate(&start), "event create");
        check(cudaEventCreate(&end), "event create");
        for (int repeat = 0; repeat < 4; ++repeat) {
            for (int code = 0; code < 2; ++code) check(cudaGraphLaunch(executables[code], stream_), "warmup");
        }
        check(cudaStreamSynchronize(stream_), "warmup");
        for (int repeat = 0; repeat < 7; ++repeat) {
            float us[2];
            for (int slot = 0; slot < 2; ++slot) {
                const int code = (repeat + slot) & 1;
                check(cudaEventRecord(start, stream_), "event record");
                for (int batch = 0; batch < batches; ++batch)
                    check(cudaGraphLaunch(executables[code], stream_), "graph launch");
                check(cudaEventRecord(end, stream_), "event record");
                check(cudaEventSynchronize(end), "event synchronize");
                float ms = 0.f;
                check(cudaEventElapsedTime(&ms, start, end), "event elapsed");
                us[code] = ms * 1000.f / float(batches * graph_repeats * sets);
            }
            std::printf("gate_code_decode_timing routes=%d scale_view=%d pair=%d repeat=%d baseline_us=%.6f "
                        "kmajor_us=%.6f scope=two_rotating_synthetic_sets_gate_only_not_model\n",
                        routes_, int(scale_view_enabled_), int(pair), repeat, us[0], us[1]);
        }
        check(cudaEventDestroy(start), "event destroy");
        check(cudaEventDestroy(end), "event destroy");
        for (int code = 0; code < 2; ++code) {
            check(cudaGraphExecDestroy(executables[code]), "graph exec destroy");
            check(cudaGraphDestroy(graphs[code]), "graph destroy");
        }
    }

    int routes_;
    cudaStream_t stream_;
    bool scale_view_enabled_ = false;
    DeviceArray<uint8_t> weights_;
    DeviceArray<uint8_t> code_view_;
    DeviceArray<uint8_t> kmajor_scales_;
    DeviceArray<float> input_;
    DeviceArray<float> baseline_;
    DeviceArray<float> candidate_;
    DeviceArray<float> partial_base_;
    DeviceArray<float> partial_code_;
    DeviceArray<uint8_t> prepared_;
    DeviceArray<uint8_t> prepared_scales_;
    DeviceArray<Job> base_jobs_;
    DeviceArray<Job> code_jobs_;
};

// The K64-major MMA entry rejects invalid arguments before enqueue: the
// rejected calls run inside a stream capture that has to stay empty.
void check_mma_admission(cudaStream_t stream) {
    constexpr int routes = 10;
    DeviceArray<Job> jobs(routes);
    struct Case {
        const char* name;
        const Job* jobs;
        int rows;
        int inner;
        bool canonical;
    };
    const Case cases[] = {
        {"null descriptors", nullptr, rows, inner, true},
        {"N 1024", jobs.data(), 1024, inner, true},
        {"K 1280", jobs.data(), rows, 1280, true},
        {"native weights", jobs.data(), rows, inner, false},
    };
    cudaGraph_t graph;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    for (const Case& c : cases) {
        const cudaError_t status = mollm_cuda::launch_nvfp4_mma_projections(
            c.jobs, routes, c.rows, c.inner, split_row, stream, nullptr, 1, c.canonical, nullptr, nullptr,
            true);
        if (status != cudaErrorInvalidValue) {
            char message[128];
            std::snprintf(message, sizeof(message), "K64-major MMA: %s admitted", c.name);
            mollm_test::fail(message);
        }
    }
    check(cudaStreamEndCapture(stream, &graph), "end capture");
    size_t nodes = 0;
    check(cudaGraphGetNodes(graph, nullptr, &nodes), "graph nodes");
    check(cudaGraphDestroy(graph), "graph destroy");
    require(nodes == 0, "rejected launches enqueued work");
}

bool parse_routes(const char* text, std::vector<int>& routes) {
    const char* cursor = text;
    while (true) {
        char* end = nullptr;
        const long value = std::strtol(cursor, &end, 10);
        if (end == cursor || value < 10 || value > 60 || value % 10) return false;
        routes.push_back(int(value));
        if (*end == '\0') return true;
        if (*end != ',') return false;
        cursor = end + 1;
    }
}

bool parse_options(int argc, char** argv, std::vector<int>& routes, bool& timing) {
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--timing")) {
            timing = true;
        } else if (!std::strcmp(argv[i], "--routes") && i + 1 < argc) {
            if (!parse_routes(argv[++i], routes)) return false;
        } else {
            return false;
        }
    }
    if (routes.empty()) routes.push_back(10);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> routes;
    bool timing = false;
    if (!parse_options(argc, argv, routes, timing)) {
        std::fprintf(stderr, "usage: %s [--routes N[,N...]] [--timing]   (N: 10..60, multiple of 10)\n",
                     argv[0]);
        return 2;
    }
    if (!mollm_test::cuda_device_available()) return mollm_test::skip_status;
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "device properties");
    // The MMA consumers need the SM 12.0/12.1 FP4 tensor-core path.
    if (prop.major != 12 || prop.minor > 1) return mollm_test::skip_status;
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");
    check_mma_admission(stream);
    for (int count : routes) {
        ViewCheck view_check(count, stream);
        for (bool scale_view : {false, true}) {
            const int replays = view_check.run(scale_view, timing);
            std::printf("PASS gate_code_decode routes=%d scale_view=%d replays=%d layouts=ROW,PAIR "
                        "modes=simt,pipelined,mma1,mma2,mma8,mma_prepared,subgroup split_row=640 "
                        "bitwise=1 guards=1 immutable=1\n",
                        count, int(scale_view), replays);
        }
    }
    check(cudaStreamDestroy(stream), "stream destroy");
    return 0;
}
