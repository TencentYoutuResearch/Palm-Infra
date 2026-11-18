#include "kernels/cuda/moe.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status)); std::exit(1);
    }
}
template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
    ~Buffer() { cudaFree(p); }
};
template<class T> void copy(T* ptr, const std::vector<T>& data) {
    check(cudaMemcpy(ptr, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template<class T> std::vector<T> read(const T* ptr, size_t count) {
    std::vector<T> result(count);
    check(cudaMemcpy(result.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost)); return result;
}
void equal(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size()*sizeof(float))) {
        std::fprintf(stderr, "decode byte mismatch\n"); std::exit(2);
    }
}
// Keep the general 32-lane reduction as an independent reference for the
// subgroup's alignment dispatch and peeled final packet loads.
void check_subgroup_packets(int routes, cudaStream_t stream, bool paired = false) {
    constexpr int rows = 1280, inner = 2560, experts = 3;
    constexpr size_t code_bytes = size_t(rows) * inner / 2;
    constexpr size_t scale_bytes = size_t(rows) * inner / 16;
    constexpr size_t code_stride = code_bytes + 16, scale_stride = scale_bytes + 16;
    const size_t output_count = size_t(routes) * rows + 32;
    Buffer<uint8_t> codes(experts * code_stride), scales(experts * scale_stride);
    Buffer<float> row_scales(experts * rows), input(size_t(routes) * inner + 4);
    Buffer<float> reference(output_count), result(output_count);
    Buffer<mollm_cuda::Nvfp4Projection> reference_jobs(routes), candidate_jobs(routes);
    const auto from_bits = [](uint32_t bits) {
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    };
    const auto bytes_equal = [](const auto& a, const auto& b) {
        return a.size() == b.size() &&
            !std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0]));
    };
    int cases = 0;
    for (bool pair : {false, true}) for (bool scale_major : {false, true}) {
        for (int alignment = 0; alignment < 4; ++alignment) {
            // Deliberately support odd code addresses and float-aligned inputs
            // that cannot be loaded as float4. Padding also acts as a sentinel.
            const int code_shift = alignment & 1, input_shift = (alignment >> 1) & 1;
            std::vector<uint8_t> host_codes(experts * code_stride, 0x5a);
            std::vector<uint8_t> host_scales(experts * scale_stride, 0x5a);
            std::vector<float> host_rows(experts * rows);
            unsigned random = 181203;
            const auto next = [&]() {
                random ^= random << 13; random ^= random >> 17; random ^= random << 5;
                return random;
            };
            for (int expert = 0; expert < experts; ++expert) {
                for (int row = 0; row < rows; ++row) {
                    host_rows[expert * rows + row] = float((row + expert) % 7 + 1) / 64.f;
                    for (int group = 0; group < inner / 16; ++group) {
                        const uint8_t values[] = {0,1,7,8,48,56,64,126,128,129,184,254};
                        const size_t offset = scale_major
                            ? size_t(group / 4) * rows * 4 + row * 4 + group % 4
                            : size_t(row) * (inner / 16) + group;
                        host_scales[expert * scale_stride + offset] =
                            values[(group + row + expert) % 12];
                    }
                    for (int col = 0; col < inner; col += 2) {
                        const size_t offset = size_t(col / 64) * rows * 32 + (pair
                            ? size_t(row / 4) * 128 + (col % 64 / 16) * 32 +
                                (row % 4) * 8 + (col % 16) / 2
                            : size_t(row) * 32 + (col % 64) / 2);
                        host_codes[expert * code_stride + code_shift + offset] = next();
                    }
                }
            }
            copy(codes.p, host_codes); copy(row_scales.p, host_rows);
            std::vector<mollm_cuda::Nvfp4Projection> jobs(routes), refs(routes);
            for (int route = 0; route < routes; ++route) {
                const int expert = route % experts;
                jobs[route] = {codes.p + expert * code_stride + code_shift,
                    scales.p + expert * scale_stride, row_scales.p + expert * rows,
                    input.p + size_t(route) * inner + input_shift,
                    result.p + size_t(route) * rows, pair};
                jobs[route].gate_scales_kmajor = scale_major;
                refs[route] = jobs[route];
                refs[route].output = reference.p + size_t(route) * rows;
            }
            copy(candidate_jobs.p, jobs); copy(reference_jobs.p, refs);
            const auto launch = [&]() {
                check(mollm_cuda::launch_nvfp4_gate_code_projections(
                    reference_jobs.p, routes, rows, inner, false, stream));
                check(paired
                    ? mollm_cuda::launch_nvfp4_gate_paired_projections(
                        candidate_jobs.p, routes, rows, inner, stream)
                    : mollm_cuda::launch_nvfp4_gate_code_subgroup_projections(
                        candidate_jobs.p, routes, rows, inner, stream));
            };
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            launch();
            check(cudaStreamEndCapture(stream, &graph));
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int variant = 0; variant < 11; ++variant) {
                std::vector<float> host_input(size_t(routes) * inner + 4);
                for (size_t i = 0; i < host_input.size(); ++i) {
                    float value = float(int(next() % 2001) - 1000) / 8192.f;
                    if (variant == 1) value = -value;
                    if (variant == 2) value = value * .5f + float(int(i % 13) - 6) * .00013f;
                    if (variant == 3) value = 0.f;
                    if (variant == 4) value = from_bits(0x80000000u);
                    if (variant == 5) value = i % 4 ? 0.f : from_bits(2);
                    host_input[i] = value;
                }
                if (variant == 6) host_input[13] = from_bits(0x7fc12345u);
                if (variant == 7 || variant == 8) host_input[17] = from_bits(0x7f800000u);
                if (variant == 8) host_input[23] = from_bits(0xff800000u);
                auto changed_scales = host_scales;
                if (variant >= 9) changed_scales[3] = variant == 9 ? 127 : 255;
                if (paired) {
                    // Graph replays must match the current descriptor contents,
                    // including unequal layouts sharing the same code address.
                    for (int route = 0; route < routes; ++route) {
                        const int expert = variant & 1
                            ? (route * (route + 3) + variant) % experts : route % experts;
                        jobs[route].weight = codes.p + expert * code_stride + code_shift;
                        jobs[route].block_scales = scales.p + expert * scale_stride;
                        jobs[route].row_scales = row_scales.p + expert * rows;
                        jobs[route].pair_layout = pair ^ (variant == 2 && route % 5 == 0);
                        jobs[route].gate_scales_kmajor = scale_major ^ (variant == 1 && route % 7 == 0);
                        refs[route] = jobs[route];
                        refs[route].output = reference.p + size_t(route) * rows;
                    }
                    copy(candidate_jobs.p, jobs); copy(reference_jobs.p, refs);
                }
                copy(input.p, host_input); copy(scales.p, changed_scales);
                for (bool captured : {false, true}) {
                    check(cudaMemsetAsync(reference.p, 0x5a, output_count * sizeof(float), stream));
                    check(cudaMemsetAsync(result.p, 0x5a, output_count * sizeof(float), stream));
                    if (captured) check(cudaGraphLaunch(executable, stream));
                    else launch();
                    check(cudaStreamSynchronize(stream));
                    const auto expected = read(reference.p, output_count);
                    const auto actual = read(result.p, output_count);
                    equal(expected, actual);
                    const auto* tail = reinterpret_cast<const uint8_t*>(actual.data() + size_t(routes) * rows);
                    for (size_t i = 0; i < 32 * sizeof(float); ++i) if (tail[i] != 0x5a) std::exit(6);
                }
                if (read(codes.p, host_codes.size()) != host_codes ||
                    read(scales.p, changed_scales.size()) != changed_scales ||
                    !bytes_equal(read(input.p, host_input.size()), host_input) ||
                    !bytes_equal(read(row_scales.p, host_rows.size()), host_rows)) std::exit(7);
                ++cases;
            }
            check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
        }
    }
    std::printf("subgroup_packets cases=%d routes=%d aligned_unaligned=1 nonfinite=1 eager_graph=1 bitwise=1 guards=1 immutable=1 paired=%d\n", cases, routes, paired);
}
}

int main(int argc, char** argv) {
    const bool timing = argc == 2 && !std::strcmp(argv[1], "--timing");
    if (argc > 1 && !timing) return 64;
    const bool scale_view=std::getenv("MOLLM_TEST_NVFP4_RESIDENT_GATE_SCALE_VIEW")&&
        !std::strcmp(std::getenv("MOLLM_TEST_NVFP4_RESIDENT_GATE_SCALE_VIEW"),"1");
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop,0));
    if (prop.major != 12 || prop.minor > 1) return 77;
    constexpr int rows = 1280, inner = 2560, sets = 2;
    int routes=10;
    if(const char* option=std::getenv("MOLLM_TEST_NVFP4_GATE_CODE_ROUTES")) {
        char* end=nullptr;const long value=std::strtol(option,&end,10);
        if(end==option || *end || value<10 || value>60 || value%10)return 64;
        routes=int(value);
    }
    const char* subgroup_option=std::getenv("MOLLM_TEST_NVFP4_GATE_CODE_SUBGROUP");
    const bool subgroup=subgroup_option && !std::strcmp(subgroup_option,"1");
    constexpr size_t codes = size_t(rows)*inner/2, scales = size_t(rows)*inner/16;
    constexpr size_t stride = codes + scales + rows*sizeof(float);
    const size_t outputs = size_t(routes)*rows;
    Buffer<uint8_t> weights(sets*routes*stride), view(sets*routes*codes+16);
    Buffer<uint8_t> scale_major(sets*routes*scales+16);
    Buffer<float> input(size_t(sets)*routes*inner), baseline(outputs+8), candidate(outputs+8);
    Buffer<float> partial_base(outputs*8), partial_code(outputs*8);
    Buffer<uint8_t> prepared(routes*2*inner/2), prepared_scales(routes*2*inner/16);
    Buffer<mollm_cuda::Nvfp4Projection> base_jobs(sets*routes), code_jobs(sets*routes);
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto launch = [&](bool code, int mode, int set) {
        if (mode >= 2) {
            const int parts = mode == 3 ? 2 : mode == 4 ? 8 : 1;
            check(mollm_cuda::launch_nvfp4_mma_projections(
                (code?code_jobs.p:base_jobs.p)+set*routes,routes,rows,inner,640,stream,
                code?partial_code.p:partial_base.p,parts,true,
                mode==5?prepared.p:nullptr,mode==5?prepared_scales.p:nullptr,code||scale_view));
        } else if (code||scale_view) {
            if(subgroup && mode==0)check(mollm_cuda::launch_nvfp4_gate_code_subgroup_projections(
                (code?code_jobs.p:base_jobs.p)+set*routes,routes,rows,inner,stream));
            else check(mollm_cuda::launch_nvfp4_gate_code_projections(
                (code?code_jobs.p:base_jobs.p)+set*routes,routes,rows,inner,mode==1,stream));
        }
        else if (mode==1) check(mollm_cuda::launch_nvfp4_pipelined_projections(
            base_jobs.p+set*routes, routes, rows, inner, true, stream));
        else {
            mollm_cuda::launch_nvfp4_projections(base_jobs.p+set*routes, routes, rows, inner, stream);
            check(cudaGetLastError());
        }
    };
    check(cudaMemset(candidate.p, 0x5a, (outputs+8)*sizeof(float)));
    check(cudaDeviceSynchronize());
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    int rejects = 0;
    for (int id = 0; id < 7; ++id) {
        const auto* jobs = code_jobs.p;
        int n = routes, r = rows, k = inner;
        switch (id) {
            case 0: jobs = nullptr; break;
            case 1: n = 0; break;
            case 2: n = -1; break;
            case 3: n = 65536; break;
            case 4: r = 640; break;
            case 5: k = 640; break;
            case 6: jobs = reinterpret_cast<const mollm_cuda::Nvfp4Projection*>(
                reinterpret_cast<const uint8_t*>(jobs)+1); break;
        }
        if (mollm_cuda::launch_nvfp4_gate_code_projections(jobs,n,r,k,true,stream)
            != cudaErrorInvalidValue) return 3;
        ++rejects;
        if(mollm_cuda::launch_nvfp4_gate_code_subgroup_projections(jobs,n,r,k,stream)
            !=cudaErrorInvalidValue)return 3;
        ++rejects;
        if(mollm_cuda::launch_nvfp4_gate_paired_projections(jobs,n,r,k,stream)
            !=cudaErrorInvalidValue)return 3;
        ++rejects;
    }
    if(mollm_cuda::launch_nvfp4_gate_paired_projections(code_jobs.p,51,rows,inner,stream)
        !=cudaErrorInvalidValue)return 3;
    ++rejects;
    for (int id = 0; id < 4; ++id) {
        const auto* jobs = code_jobs.p;
        int r = rows, k = inner;
        bool canonical = true;
        switch (id) {
            case 0: jobs = nullptr; break;
            case 1: r = 1024; break;
            case 2: k = 1280; break;
            case 3: canonical = false; break;
        }
        if (mollm_cuda::launch_nvfp4_mma_projections(jobs,routes,r,k,640,stream,
            nullptr,1,canonical,nullptr,nullptr,true) != cudaErrorInvalidValue) return 3;
        ++rejects;
    }
    cudaGraph_t empty;
    check(cudaStreamEndCapture(stream, &empty));
    size_t nodes = 0;
    check(cudaGraphGetNodes(empty, nullptr, &nodes));
    if (nodes) return 4;
    check(cudaGraphDestroy(empty));
    const auto sentinel = read(reinterpret_cast<const uint8_t*>(candidate.p), (outputs+8)*sizeof(float));
    for (uint8_t b : sentinel) if (b != 0x5a) return 5;
    int changed = 0;
    for (bool pair : {false, true}) {
        std::vector<uint8_t> payload(sets*routes*stride);
        for (int expert = 0; expert < sets*routes; ++expert) {
            auto* p = payload.data()+size_t(expert)*stride;
            for (int row = 0; row < rows; ++row) {
                const float global = (1+(row+expert)%7)/64.f;
                std::memcpy(p+codes+scales+row*sizeof(float), &global, sizeof(float));
                for (int group = 0; group < inner/16; ++group) {
                    const uint8_t sf[] = {0,1,7,8,48,56,64,126};
                    p[codes+size_t(row)*(inner/16)+group] = sf[(row+group+expert)%8];
                    for (int byte = 0; byte < 8; ++byte) {
                        const size_t offset = pair ? size_t(row/4)*(inner/16)*32+group*32+(row%4)*8+byte
                            : size_t(row)*(inner/2)+group*8+byte;
                        p[offset] = (row*29+group*17+byte*13+expert*37)&255;
                    }
                }
            }
        }
        copy(weights.p, payload);
        check(cudaMemset(view.p, 0x5a, sets*routes*codes+16));
        for (int expert = 0; expert < sets*routes; ++expert) {
            mollm_cuda::Nvfp4Projection source{};
            source.weight = weights.p+size_t(expert)*stride; source.pair_layout = pair;
            check(mollm_cuda::launch_repack_nvfp4_gate_codes(source, view.p+size_t(expert)*codes,
                rows,inner,stream));
            if(scale_view) {
                source.block_scales=weights.p+size_t(expert)*stride+codes;
                check(mollm_cuda::launch_repack_nvfp4_gate_scales(source,
                    scale_major.p+size_t(expert)*scales,rows,inner,stream));
            }
        }
        check(cudaStreamSynchronize(stream));
        const auto original_view = read(view.p, sets*routes*codes+16);
        for (int mode = 0; mode < 6; ++mode) {
            std::vector<mollm_cuda::Nvfp4Projection> a(sets*routes), b(sets*routes);
            const auto update = [&](int variant) {
                std::vector<float> values(size_t(sets)*routes*inner);
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] = (int((i*17+variant*31)%257)-128)/128.f;
                copy(input.p, values);
                for (int set = 0; set < sets; ++set) for (int route = 0; route < routes; ++route) {
                    const int expert = set*routes+(route*3+variant)%routes;
                    const auto* ptr = weights.p+size_t(expert)*stride;
                    a[set*routes+route] = {ptr,ptr+codes,reinterpret_cast<const float*>(ptr+codes+scales),
                        input.p+size_t(set*routes+route)*inner,baseline.p+size_t(route)*rows,pair};
                    a[set*routes+route].input_global = (1+route%3)/16.f;
                    a[set*routes+route].input_global_second = (1+route%5)/16.f;
                    a[set*routes+route].output_scale = (1+route%7)/8.f;
                    // Isolate the scale-layout change: both arms use the
                    // already-qualified K-major code payload in this mode.
                    if(scale_view)a[set*routes+route].weight=view.p+size_t(expert)*codes;
                    b[set*routes+route] = a[set*routes+route];
                    b[set*routes+route].weight = view.p+size_t(expert)*codes;
                    if(scale_view) {
                        b[set*routes+route].block_scales=scale_major.p+size_t(expert)*scales;
                        b[set*routes+route].gate_scales_kmajor=true;
                    }
                    b[set*routes+route].output = candidate.p+size_t(route)*rows;
                }
                copy(base_jobs.p,a); copy(code_jobs.p,b);
            };
            update(0);
            cudaGraph_t graphs[sets]; cudaGraphExec_t execs[sets];
            for (int set = 0; set < sets; ++set) {
                check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
                launch(false,mode,set); launch(true,mode,set);
                check(cudaStreamEndCapture(stream,&graphs[set]));
                check(cudaGraphInstantiate(&execs[set],graphs[set],nullptr,nullptr,0));
            }
            for (int variant = 0; variant < 8; ++variant) {
                update(variant);
                for (int set = 0; set < sets; ++set) {
                    check(cudaMemsetAsync(baseline.p,0x5a,(outputs+8)*sizeof(float),stream));
                    check(cudaMemsetAsync(candidate.p,0x5a,(outputs+8)*sizeof(float),stream));
                    check(cudaGraphLaunch(execs[set],stream)); check(cudaStreamSynchronize(stream));
                    const auto ref = read(baseline.p,outputs+8), result = read(candidate.p,outputs+8);
                    equal(ref,result);
                    const auto* tail = reinterpret_cast<const uint8_t*>(result.data()+outputs);
                    for (int i = 0; i < 8*sizeof(float); ++i) if (tail[i] != 0x5a) return 6;
                    ++changed;
                }
            }
            for (int set = 0; set < sets; ++set) {
                check(cudaGraphExecDestroy(execs[set])); check(cudaGraphDestroy(graphs[set]));
            }
            if (timing && mode==1) {
                cudaGraph_t graph[2]; cudaGraphExec_t exec[2];
                for (int code = 0; code < 2; ++code) {
                    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
                    for (int repeat = 0; repeat < 16; ++repeat)
                        for (int set = 0; set < sets; ++set) launch(code,mode,set);
                    check(cudaStreamEndCapture(stream,&graph[code]));
                    check(cudaGraphInstantiate(&exec[code],graph[code],nullptr,nullptr,0));
                }
                cudaEvent_t start,end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
                for (int repeat = 0; repeat < 4; ++repeat)
                    for (int code = 0; code < 2; ++code) check(cudaGraphLaunch(exec[code],stream));
                check(cudaStreamSynchronize(stream));
                for (int repeat = 0; repeat < 7; ++repeat) {
                    float us[2];
                    for (int slot = 0; slot < 2; ++slot) {
                        const int code = (repeat+slot)&1;
                        check(cudaEventRecord(start,stream));
                        for (int batch = 0; batch < 16; ++batch) check(cudaGraphLaunch(exec[code],stream));
                        check(cudaEventRecord(end,stream)); check(cudaEventSynchronize(end));
                        float ms; check(cudaEventElapsedTime(&ms,start,end)); us[code]=ms*1000/(16*32);
                    }
                    std::printf("gate_code_decode_timing pair=%d repeat=%d baseline_us=%.6f kmajor_us=%.6f scale_view=%d scope=ten_routes_two_rotating_synthetic_sets_gate_only_not_model\n",
                        pair,repeat,us[0],us[1],scale_view);
                }
                check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
                for (int code = 0; code < 2; ++code) {
                    check(cudaGraphExecDestroy(exec[code])); check(cudaGraphDestroy(graph[code]));
                }
            }
        }
        if (read(view.p,original_view.size()) != original_view || read(weights.p,payload.size()) != payload) return 7;
    }
    if (subgroup) {
        check_subgroup_packets(routes, stream);
        if (routes <= 50) check_subgroup_packets(routes, stream, true);
    }
    check(cudaStreamDestroy(stream));
    std::printf("gate_code_decode rejected=%d changed_replays=%d rows=1280 inner=2560 routes=%d sets=2 layouts=ROW,PAIR simt_pipeline=0,1 mma_parts=1,2,8 mma_prequantized=1 split_row=640 byte_exact=1 weights_immutable=1 guards=1\n",rejects,changed,routes);
}
