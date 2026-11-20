// Fixed 1280x2560 NVFP4 gate projections over the K64-major code view.
//
// Every gate launcher must reproduce, bit for bit, the general non-pipelined
// 32-lane gate-code consumer (launch_nvfp4_gate_code_projections with
// pipeline_inputs=false). One fixture serves all launchers: three experts with
// ROW/PAIR codes, row-major/K-major block scales, odd code addresses and
// float-aligned inputs, two block-scale tables, eleven input variants (signed
// zeros, subnormals, NaN, infinities, NaN scale bytes) and descriptor changes
// between eager runs and graph replays. Output guards, immutable inputs and the
// rejection of invalid arguments before enqueue are checked as well.
//
// Usage: test_cuda_nvfp4_gate_projections --launcher NAME --routes N[,N...]
//   NAME: code (pipelined general entry), subgroup, m1_proven, paired, paired8.

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
using mollm_test::from_bits;
using mollm_test::require;
using mollm_test::same_bits;

constexpr int rows = 1280;
constexpr int inner = 2560;
constexpr int experts = 3;
constexpr int input_variants = 11;
constexpr uint32_t seed = 181203;
constexpr size_t code_bytes = size_t(rows) * inner / 2;
constexpr size_t scale_bytes = size_t(rows) * inner / 16;
// Spare bytes per expert let the odd-address fixtures shift the codes by one.
constexpr size_t code_stride = code_bytes + 16;
constexpr size_t scale_stride = scale_bytes + 16;

enum class Launcher { code, subgroup, m1_proven, paired, paired8 };

struct LauncherInfo {
    const char* name;
    Launcher id;
    int max_routes;
    // m1_proven does not verify its precondition (PAIR codes, K-major block
    // scales, aligned pointers), so it only gets qualified descriptors.
    bool qualified_only;
};

const LauncherInfo launchers[] = {
    {"code", Launcher::code, 65535, false},
    {"subgroup", Launcher::subgroup, 65535, false},
    {"m1_proven", Launcher::m1_proven, 50, true},
    {"paired", Launcher::paired, 50, false},
    {"paired8", Launcher::paired8, 80, false},
};

cudaError_t launch_candidate(Launcher launcher, const Job* jobs, int count, int n, int k,
                             cudaStream_t stream) {
    switch (launcher) {
    case Launcher::code:
        return mollm_cuda::launch_nvfp4_gate_code_projections(jobs, count, n, k, true, stream);
    case Launcher::subgroup:
        return mollm_cuda::launch_nvfp4_gate_code_subgroup_projections(jobs, count, n, k, stream);
    case Launcher::m1_proven:
        return mollm_cuda::launch_nvfp4_gate_code_m1_proven_projections(jobs, count, n, k, stream);
    case Launcher::paired:
        return mollm_cuda::launch_nvfp4_gate_paired_projections(jobs, count, n, k, stream);
    case Launcher::paired8:
        return mollm_cuda::launch_nvfp4_gate_paired8_projections(jobs, count, n, k, stream);
    }
    return cudaErrorInvalidValue;
}

// Independent oracle: the general entry without input pipelining.
cudaError_t launch_reference(const Job* jobs, int count, cudaStream_t stream) {
    return mollm_cuda::launch_nvfp4_gate_code_projections(jobs, count, rows, inner, false, stream);
}

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

struct Buffers {
    explicit Buffers(int routes)
        : codes(experts * code_stride),
          scales(experts * scale_stride),
          row_scales(size_t(experts) * rows),
          input(size_t(routes) * inner + 4),
          output(size_t(routes) * rows),
          reference(size_t(routes) * rows),
          jobs(routes),
          reference_jobs(routes) {}

    void check_guards() const {
        codes.check_guards();
        scales.check_guards();
        row_scales.check_guards();
        input.check_guards();
        output.check_guards();
        reference.check_guards();
        jobs.check_guards();
        reference_jobs.check_guards();
    }

    DeviceArray<uint8_t> codes;
    DeviceArray<uint8_t> scales;
    DeviceArray<float> row_scales;
    DeviceArray<float> input;
    DeviceArray<float> output;
    DeviceArray<float> reference;
    DeviceArray<Job> jobs;
    DeviceArray<Job> reference_jobs;
};

struct Fixture {
    bool pair;
    bool scale_major;
    int code_shift;   // 1: odd code addresses
    int input_shift;  // 1: float-aligned inputs that cannot be loaded as float4
    int scale_set;    // 0: twelve-entry table, 1: every byte value
};

bool qualified(const Fixture& fixture) {
    return fixture.pair && fixture.scale_major && fixture.code_shift == 0 && fixture.input_shift == 0;
}

struct Context {
    const char* launcher;
    int routes;
    Fixture fixture;
    int variant;
};

void expect(bool ok, const Context& context, const char* what) {
    if (ok) return;
    char message[320];
    std::snprintf(message, sizeof(message),
                  "%s (launcher=%s routes=%d pair=%d kmajor_scales=%d code_shift=%d input_shift=%d "
                  "scale_set=%d variant=%d)",
                  what, context.launcher, context.routes, int(context.fixture.pair),
                  int(context.fixture.scale_major), context.fixture.code_shift,
                  context.fixture.input_shift, context.fixture.scale_set, context.variant);
    mollm_test::fail(message);
}

class Rng {
public:
    uint32_t next() {
        state_ ^= state_ << 13;
        state_ ^= state_ >> 17;
        state_ ^= state_ << 5;
        return state_;
    }

private:
    uint32_t state_ = seed;
};

// K64-major code view: [K/64][rows][32] bytes for ROW codes, with PAIR codes
// interleaving four rows in 8-byte groups inside each 128-byte block.
size_t code_offset(bool pair, int row, int col) {
    const size_t tile = size_t(col / 64) * rows * 32;
    if (pair)
        return tile + size_t(row / 4) * 128 + (col % 64 / 16) * 32 + (row % 4) * 8 + col % 16 / 2;
    return tile + size_t(row) * 32 + col % 64 / 2;
}

// Block scales: row-major [rows][K/16] or K-major [K/64][rows][4].
size_t scale_offset(bool scale_major, int row, int group) {
    if (scale_major)
        return size_t(group / 4) * rows * 4 + size_t(row) * 4 + group % 4;
    return size_t(row) * (inner / 16) + group;
}

struct HostWeights {
    std::vector<uint8_t> codes;
    std::vector<uint8_t> scales;
    std::vector<float> row_scales;
};

HostWeights make_weights(const Fixture& fixture, Rng& rng) {
    const uint8_t scale_table[] = {0, 1, 7, 8, 48, 56, 64, 126, 128, 129, 184, 254};
    HostWeights host;
    // Padding keeps 0x5a so that stray reads past an expert stay deterministic.
    host.codes.assign(experts * code_stride, 0x5a);
    host.scales.assign(experts * scale_stride, 0x5a);
    host.row_scales.resize(size_t(experts) * rows);
    for (int expert = 0; expert < experts; ++expert) {
        for (int row = 0; row < rows; ++row) {
            host.row_scales[size_t(expert) * rows + row] = float((row + expert) % 7 + 1) / 64.f;
            for (int group = 0; group < inner / 16; ++group) {
                const int index = group + row + expert;
                const uint8_t value = fixture.scale_set ? uint8_t(index) : scale_table[index % 12];
                host.scales[expert * scale_stride + scale_offset(fixture.scale_major, row, group)] = value;
            }
            for (int col = 0; col < inner; col += 2) {
                const size_t offset = expert * code_stride + fixture.code_shift +
                                      code_offset(fixture.pair, row, col);
                host.codes[offset] = uint8_t(rng.next());
            }
        }
    }
    return host;
}

// Variant 0 is plain random input; the others stress sign handling, rounding,
// zeros, negative zeros, subnormals, a NaN payload and infinities.
std::vector<float> make_input(Rng& rng, int variant, size_t count) {
    std::vector<float> input(count);
    for (size_t i = 0; i < count; ++i) {
        float value = float(int(rng.next() % 2001) - 1000) / 8192.f;
        if (variant == 1)
            value = -value;
        else if (variant == 2)
            value = value * .5f + float(int(i % 13) - 6) * .00013f;
        else if (variant == 3)
            value = 0.f;
        else if (variant == 4)
            value = from_bits(0x80000000u);
        else if (variant == 5)
            value = i % 4 ? 0.f : from_bits(2);
        input[i] = value;
    }
    if (variant == 6) input[13] = from_bits(0x7fc12345u);
    if (variant == 7 || variant == 8) input[17] = from_bits(0x7f800000u);
    if (variant == 8) input[23] = from_bits(0xff800000u);
    return input;
}

// Odd variants remap routes onto experts, so graph replays must read the
// current descriptors and the pairing launchers see changing equal-expert
// masks. Where per-descriptor layouts are allowed, variants 1 and 2 also flip
// the scale/code layout of some routes: unequal layouts then share one code
// address and must not be paired.
void describe(Buffers& buffers, const Fixture& fixture, int variant, bool mixed_layouts,
              std::vector<Job>& jobs, std::vector<Job>& reference_jobs) {
    const int routes = int(jobs.size());
    for (int route = 0; route < routes; ++route) {
        const int expert = variant & 1 ? int((size_t(route) * (route + 3) + variant) % experts)
                                       : route % experts;
        Job job{};
        job.weight = buffers.codes.data() + expert * code_stride + fixture.code_shift;
        job.block_scales = buffers.scales.data() + expert * scale_stride;
        job.row_scales = buffers.row_scales.data() + size_t(expert) * rows;
        job.input = buffers.input.data() + size_t(route) * inner + fixture.input_shift;
        job.output = buffers.output.data() + size_t(route) * rows;
        const bool flip_codes = mixed_layouts && variant == 2 && route % 5 == 0;
        const bool flip_scales = mixed_layouts && variant == 1 && route % 7 == 0;
        job.pair_layout = fixture.pair != flip_codes;
        job.gate_scales_kmajor = fixture.scale_major != flip_scales;
        jobs[route] = job;
        job.output = buffers.reference.data() + size_t(route) * rows;
        reference_jobs[route] = job;
    }
}

// Returns the number of checked input variants.
int run_fixture(const LauncherInfo& launcher, Buffers& buffers, const Fixture& fixture,
                cudaStream_t stream) {
    const int routes = int(buffers.jobs.size());
    const bool mixed_layouts = !launcher.qualified_only;
    Rng rng;
    const HostWeights weights = make_weights(fixture, rng);
    buffers.codes.upload(weights.codes, stream);
    buffers.scales.upload(weights.scales, stream);
    buffers.row_scales.upload(weights.row_scales, stream);

    std::vector<Job> jobs(routes);
    std::vector<Job> reference_jobs(routes);
    describe(buffers, fixture, 0, mixed_layouts, jobs, reference_jobs);
    buffers.jobs.upload(jobs, stream);
    buffers.reference_jobs.upload(reference_jobs, stream);

    const auto launch_both = [&]() {
        check(launch_reference(buffers.reference_jobs.data(), routes, stream), "reference launch");
        check(launch_candidate(launcher.id, buffers.jobs.data(), routes, rows, inner, stream),
              "candidate launch");
    };
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    launch_both();
    check(cudaStreamEndCapture(stream, &graph), "end capture");
    check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "graph instantiate");

    for (int variant = 0; variant < input_variants; ++variant) {
        const Context context{launcher.name, routes, fixture, variant};
        const std::vector<float> input = make_input(rng, variant, buffers.input.size());
        // Variants 9 and 10 turn one block scale into an E4M3 NaN encoding.
        std::vector<uint8_t> scales = weights.scales;
        if (variant >= 9) scales[3] = variant == 9 ? 127 : 255;
        describe(buffers, fixture, variant, mixed_layouts, jobs, reference_jobs);
        buffers.jobs.upload(jobs, stream);
        buffers.reference_jobs.upload(reference_jobs, stream);
        buffers.input.upload(input, stream);
        buffers.scales.upload(scales, stream);

        buffers.output.poison(stream);
        buffers.reference.poison(stream);
        launch_both();
        const std::vector<float> expected = buffers.reference.download(stream);
        expect(same_bits(buffers.output.download(stream), expected), context,
               "eager output differs from the reference");

        buffers.output.poison(stream);
        buffers.reference.poison(stream);
        check(cudaGraphLaunch(executable, stream), "graph launch");
        expect(same_bits(buffers.output.download(stream), expected), context,
               "graph replay output differs from the reference");
        expect(same_bits(buffers.reference.download(stream), expected), context,
               "graph replay reference differs from the eager reference");

        buffers.check_guards();
        expect(same_bits(buffers.codes.download(stream), weights.codes), context, "codes modified");
        expect(same_bits(buffers.scales.download(stream), scales), context, "block scales modified");
        expect(same_bits(buffers.row_scales.download(stream), weights.row_scales), context,
               "row scales modified");
        expect(same_bits(buffers.input.download(stream), input), context, "inputs modified");
        expect(same_bits(buffers.jobs.download(stream), jobs), context, "descriptors modified");
        expect(same_bits(buffers.reference_jobs.download(stream), reference_jobs), context,
               "reference descriptors modified");
    }
    check(cudaGraphExecDestroy(executable), "graph exec destroy");
    check(cudaGraphDestroy(graph), "graph destroy");
    return input_variants;
}

// Invalid arguments must be rejected before anything is enqueued: the
// rejected calls run inside a stream capture that has to stay empty.
void check_admission(const LauncherInfo& launcher, cudaStream_t stream) {
    DeviceArray<Job> jobs(1);
    const Job* valid = jobs.data();
    const Job* unaligned = reinterpret_cast<const Job*>(reinterpret_cast<const uint8_t*>(valid) + 1);
    struct Case {
        const char* name;
        const Job* jobs;
        int count;
        int n;
        int k;
    };
    const Case cases[] = {
        {"null descriptors", nullptr, 1, rows, inner},
        {"unaligned descriptors", unaligned, 1, rows, inner},
        {"negative count", valid, -1, rows, inner},
        {"zero count", valid, 0, rows, inner},
        {"count above the limit", valid, launcher.max_routes + 1, rows, inner},
        {"count 65536", valid, 65536, rows, inner},
        {"N 1279", valid, 1, rows - 1, inner},
        {"N 640", valid, 1, 640, inner},
        {"K 2559", valid, 1, rows, inner - 1},
        {"K 640", valid, 1, rows, 640},
    };
    cudaGraph_t graph;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    for (const Case& c : cases) {
        bool rejected = launch_candidate(launcher.id, c.jobs, c.count, c.n, c.k, stream) ==
                        cudaErrorInvalidValue;
        // The oracle's non-pipelined form is covered alongside the general entry.
        if (launcher.id == Launcher::code)
            rejected = rejected && mollm_cuda::launch_nvfp4_gate_code_projections(
                                       c.jobs, c.count, c.n, c.k, false, stream) == cudaErrorInvalidValue;
        if (!rejected) {
            char message[160];
            std::snprintf(message, sizeof(message), "%s: %s admitted", launcher.name, c.name);
            mollm_test::fail(message);
        }
    }
    check(cudaStreamEndCapture(stream, &graph), "end capture");
    size_t nodes = 0;
    check(cudaGraphGetNodes(graph, nullptr, &nodes), "graph nodes");
    check(cudaGraphDestroy(graph), "graph destroy");
    require(nodes == 0, "rejected launches enqueued work");
}

void run(const LauncherInfo& launcher, int routes, cudaStream_t stream) {
    Buffers buffers(routes);
    int fixtures = 0;
    int variants = 0;
    for (bool pair : {false, true}) {
        for (bool scale_major : {false, true}) {
            for (int alignment = 0; alignment < 4; ++alignment) {
                for (int scale_set = 0; scale_set < 2; ++scale_set) {
                    const Fixture fixture{pair, scale_major, alignment & 1, alignment >> 1, scale_set};
                    if (launcher.qualified_only && !qualified(fixture)) continue;
                    variants += run_fixture(launcher, buffers, fixture, stream);
                    ++fixtures;
                }
            }
        }
    }
    std::printf("PASS gate_projections launcher=%s routes=%d fixtures=%d variants=%d "
                "bitwise=1 eager_graph=1 mutable_descriptors=1 mixed_layouts=%d guards=1 immutable=1\n",
                launcher.name, routes, fixtures, variants, int(!launcher.qualified_only));
}

const LauncherInfo* find_launcher(const char* name) {
    for (const LauncherInfo& launcher : launchers) {
        if (!std::strcmp(launcher.name, name)) return &launcher;
    }
    return nullptr;
}

bool parse_routes(const char* text, int max_routes, std::vector<int>& routes) {
    const char* cursor = text;
    while (true) {
        char* end = nullptr;
        const long value = std::strtol(cursor, &end, 10);
        if (end == cursor || value < 1 || value > max_routes) return false;
        routes.push_back(int(value));
        if (*end == '\0') return true;
        if (*end != ',') return false;
        cursor = end + 1;
    }
}

bool parse_options(int argc, char** argv, const LauncherInfo*& launcher, std::vector<int>& routes) {
    const char* routes_text = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) return false;
        if (!std::strcmp(argv[i], "--launcher")) {
            launcher = find_launcher(argv[++i]);
            if (!launcher) return false;
        } else if (!std::strcmp(argv[i], "--routes")) {
            routes_text = argv[++i];
        } else {
            return false;
        }
    }
    return launcher && routes_text && parse_routes(routes_text, launcher->max_routes, routes);
}

}  // namespace

int main(int argc, char** argv) {
    const LauncherInfo* launcher = nullptr;
    std::vector<int> routes;
    if (!parse_options(argc, argv, launcher, routes)) {
        std::fprintf(stderr,
                     "usage: %s --launcher code|subgroup|m1_proven|paired|paired8 --routes N[,N...]\n"
                     "  N must not exceed the launcher limit (code/subgroup 65535, m1_proven/paired 50, "
                     "paired8 80)\n",
                     argv[0]);
        return 2;
    }
    if (!mollm_test::cuda_device_available()) return mollm_test::skip_status;
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");
    check_admission(*launcher, stream);
    for (int count : routes) run(*launcher, count, stream);
    check(cudaStreamDestroy(stream), "stream destroy");
    return 0;
}
