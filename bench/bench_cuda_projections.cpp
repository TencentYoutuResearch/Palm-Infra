#include "kernels/cuda/matmul.h"
#include "kernels/cuda/moe.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <sys/mman.h>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}
template<class T> T* allocate(size_t count) {
    T* ptr = nullptr;
    check(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}
template<class F> double measure(F launch, int iterations) {
    for (int i = 0; i < 512; ++i) launch(i);
    check(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    check(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) launch(i);
    check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start, stop));
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    return ms * 1000.0 / iterations;
}
template<class F> double measure_graph(F launch, int iterations) {
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < iterations; ++i) launch(i, stream);
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiateWithFlags(&executable, graph, 0));
    for (int i = 0; i < 2; ++i) check(cudaGraphLaunch(executable, stream));
    check(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    check(cudaEventRecord(start, stream));
    for (int i = 0; i < 4; ++i) check(cudaGraphLaunch(executable, stream));
    check(cudaEventRecord(stop, stream)); check(cudaEventSynchronize(stop));
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start, stop));
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
    check(cudaStreamDestroy(stream));
    return ms * 1000.0 / (4 * iterations);
}
void q8_probe() {
    constexpr int rows = 248320, inner = 2560;
    const size_t count = static_cast<size_t>(rows) * inner;
    std::vector<int8_t> host(count);
    for (size_t i = 0; i < count; ++i) host[i] = static_cast<int8_t>(static_cast<int>((i * 37 + i / 19) % 255) - 127);
    std::vector<float> x(inner), scales(rows), reference(rows), actual(rows);
    for (int i = 0; i < inner; ++i) x[i] = (i % 41 - 20) / 37.f;
    for (int i = 0; i < rows; ++i) scales[i] = (1 + i % 17) / 37.f;
    auto* w = allocate<int8_t>(count);
    auto* a = allocate<float>(inner);
    auto* s = allocate<float>(rows);
    auto* y = allocate<float>(rows);
    check(cudaMemcpy(w, host.data(), count, cudaMemcpyHostToDevice));
    check(cudaMemcpy(a, x.data(), inner * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(s, scales.data(), rows * sizeof(float), cudaMemcpyHostToDevice));
    auto launch = [&]() {
        mollm_cuda::launch_q8_dense_gemv(a, w, s, inner, 1, y, rows, inner);
    };
    launch();
    check(cudaMemcpy(reference.data(), y, rows * sizeof(float), cudaMemcpyDeviceToHost));
    if (std::getenv("MOLLM_Q8_EXTREME_CTA_PROBE")) {
        for (int warps : {1, 2, 32}) {
            auto candidate = [&](cudaStream_t stream) {
                check(mollm_cuda::launch_q8_tuned_gemv(a,w,s,y,rows,inner,warps,false,stream));
            };
            candidate(nullptr);
            check(cudaMemcpy(actual.data(),y,rows*sizeof(float),cudaMemcpyDeviceToHost));
            if (std::memcmp(reference.data(),actual.data(),rows*sizeof(float))) {
                std::fprintf(stderr,"Q8 extreme CTA mismatch warps=%d\n",warps);
                std::exit(1);
            }
            auto timing = [&](int cta_warps) {
                const double us = measure_graph([&](int,cudaStream_t stream) {
                    check(mollm_cuda::launch_q8_tuned_gemv(a,w,s,y,rows,inner,cta_warps,false,stream));
                },32);
                check(cudaMemcpy(actual.data(),y,rows*sizeof(float),cudaMemcpyDeviceToHost));
                if (std::memcmp(reference.data(),actual.data(),rows*sizeof(float))) {
                    std::fprintf(stderr,"Q8 graph CTA mismatch warps=%d\n",cta_warps);
                    std::exit(1);
                }
                return us;
            };
            for (int repeat=0;repeat<3;++repeat) {
                double control_us,candidate_us;
                if (repeat%2) {candidate_us=timing(warps);control_us=timing(8);}
                else {control_us=timing(8);candidate_us=timing(warps);}
                check(cudaMemcpy(actual.data(),y,rows*sizeof(float),cudaMemcpyDeviceToHost));
                if(std::memcmp(reference.data(),actual.data(),rows*sizeof(float))) std::exit(1);
                std::printf("q8-extreme-cta warps=%d repeat=%d us=%.3f control_us=%.3f speedup=%.3f bit_exact=true\n",
                    warps,repeat,candidate_us,control_us,control_us/candidate_us);
            }
        }
    } else if (std::getenv("MOLLM_Q8_TUNING_PROBE")) {
        for (int variant : {4, 8, 16, -8}) {
            auto tuned = [&]() {
                check(mollm_cuda::launch_q8_tuned_gemv(a, w, s, y, rows, inner,
                    std::abs(variant), variant < 0));
            };
            tuned();
            check(cudaMemcpy(actual.data(), y, rows * sizeof(float), cudaMemcpyDeviceToHost));
            if (std::memcmp(reference.data(), actual.data(), rows * sizeof(float))) {
                std::fprintf(stderr, "Q8 tuning mismatch variant=%d\n", variant); std::exit(1);
            }
            const double candidate = measure([&](int) { tuned(); }, 128);
            const double control = measure([&](int) { launch(); }, 128);
            std::printf("q8-tuning warps=%d shared_activation=%d us=%.3f control_us=%.3f speedup=%.3f bit_exact=true\n",
                std::abs(variant), variant < 0, candidate, control, control / candidate);
        }
    }
    check(cudaFree(w)); check(cudaFree(a)); check(cudaFree(s)); check(cudaFree(y));
}

void q4(int rows, int inner) {
    const size_t blocks = static_cast<size_t>((rows + 7) / 8) * (inner / 32);
    const size_t bytes = blocks * sizeof(Q4B8G32Block);
    const bool compact_probe = std::getenv("MOLLM_Q4_COMPACT_PROBE") != nullptr;
    const size_t smaller_payload = compact_probe ? blocks * sizeof(mollm_cuda::Q4CompactG32) : bytes;
    // Strictly exceed the target even for the smaller representation.
    const int sets = std::max<size_t>(8, (128ull << 20) / smaller_payload + 1);
    std::vector<Q4B8G32Block> host(blocks * sets);
    size_t block_index = 0;
    for (auto& block : host) {
        for (int row = 0; row < 8; ++row) {
            block.scales[row] = (1 + (block_index + row) % 17) / 37.f;
            if (compact_probe) {
                const uint32_t bits = ((110u + (block_index / 128) % 20) << 23) |
                    (static_cast<uint32_t>(block_index + row) % 128 << 16);
                float maximum;
                std::memcpy(&maximum, &bits, 4);
                block.scales[row] = block_index % 97 == 0 ? 0.f : maximum / 7.f;
            }
            for (int p = 0; p < 16; ++p)
                block.q[row][p] = (p * 13 + row * 7 + block_index * 19) & 255;
        }
        ++block_index;
    }
    auto* weight = allocate<Q4B8G32Block>(host.size());
    auto* input = allocate<float>(inner + 4);
    auto* output = allocate<float>(rows);
    std::vector<float> x(inner);
    for (int i = 0; i < inner; ++i) x[i] = (i % 41 - 20) / 37.f;
    check(cudaMemcpy(weight, host.data(), host.size() * sizeof(host[0]), cudaMemcpyHostToDevice));
    check(cudaMemcpy(input, x.data(), inner * sizeof(float), cudaMemcpyHostToDevice));
    const double us = measure([&](int i) {
        mollm_cuda::launch_q4_g32_dense_gemv(input, weight + (i % sets) * blocks, output, rows, inner);
    }, std::max(sets, 256));
    std::printf("q4 rows=%d inner=%d sets=%d us=%.3f GBps=%.2f\n", rows, inner, sets, us, bytes / us / 1000.);
    if (std::getenv("MOLLM_Q4_UNEQUAL_PROBE")) {
        const int second_rows = std::min(rows, 4);
        const int pairs = sets / 2;
        auto* second_output = allocate<float>(second_rows);
        std::vector<float> first_reference(rows), second_reference(second_rows), actual(rows);
        const auto serial = [&](int i, cudaStream_t stream = nullptr) {
            const int set = 2 * (i % pairs);
            check(mollm_cuda::launch_q4_g32_pipelined_narrow_gemv(input,
                weight + set * blocks, output, rows, inner, 1, true, stream));
            mollm_cuda::launch_q4_g32_dense_gemv(input, weight + (set + 1) * blocks,
                second_output, second_rows, inner, stream);
        };
        const auto fused = [&](int i, cudaStream_t stream = nullptr) {
            const int set = 2 * (i % pairs);
            check(mollm_cuda::launch_q4_g32_unequal_projections(input,
                weight + set * blocks, weight + (set + 1) * blocks,
                output, second_output, rows, second_rows, inner, stream));
        };
        for (int i = 0; i < pairs; ++i) {
            serial(i);
            check(cudaMemcpy(first_reference.data(), output, rows * sizeof(float), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(second_reference.data(), second_output, second_rows * sizeof(float), cudaMemcpyDeviceToHost));
            fused(i);
            check(cudaMemcpy(actual.data(), output, rows * sizeof(float), cudaMemcpyDeviceToHost));
            if (std::memcmp(first_reference.data(), actual.data(), rows * sizeof(float))) {
                std::fprintf(stderr, "unequal first projection mismatch\n"); std::exit(1);
            }
            check(cudaMemcpy(actual.data(), second_output, second_rows * sizeof(float), cudaMemcpyDeviceToHost));
            if (std::memcmp(second_reference.data(), actual.data(), second_rows * sizeof(float))) {
                std::fprintf(stderr, "unequal second projection mismatch\n"); std::exit(1);
            }
        }
        for (int repeat = 0; repeat < 3; ++repeat) {
            const double candidate = measure(fused, std::max(pairs, 256));
            const double control = measure(serial, std::max(pairs, 256));
            std::printf("q4-unequal rows=%d second_rows=%d inner=%d pairs=%d repeat=%d us=%.3f control_us=%.3f bit_exact=true\n",
                rows, second_rows, inner, pairs, repeat, candidate, control);
        }
        for (int repeat = 0; repeat < 3; ++repeat) {
            double candidate, control;
            if (repeat % 2) {
                control = measure_graph(serial, std::max(pairs, 256));
                candidate = measure_graph(fused, std::max(pairs, 256));
            } else {
                candidate = measure_graph(fused, std::max(pairs, 256));
                control = measure_graph(serial, std::max(pairs, 256));
            }
            std::printf("q4-unequal-graph rows=%d second_rows=%d inner=%d pairs=%d repeat=%d us=%.3f control_us=%.3f bit_exact=true\n",
                rows, second_rows, inner, pairs, repeat, candidate, control);
        }
        check(cudaFree(second_output)); check(cudaFree(weight)); check(cudaFree(input)); check(cudaFree(output)); return;
    }
    if (compact_probe) {
        std::vector<mollm_cuda::Q4CompactG32> compact_host(host.size());
        for (size_t i = 0; i < host.size(); ++i) {
            std::memcpy(compact_host[i].q, host[i].q, 128);
            for (int row = 0; row < 8; ++row) {
                const float maximum = host[i].scales[row] * 7.f;
                uint32_t bits;
                std::memcpy(&bits, &maximum, 4);
                const uint16_t code = (bits + 32767u + ((bits >> 16) & 1)) >> 16;
                compact_host[i].scales[row] = code;
                bits = static_cast<uint32_t>(code) << 16;
                float restored;
                std::memcpy(&restored, &bits, 4);
                restored /= 7.f;
                if (std::memcmp(&restored, &host[i].scales[row], 4)) {
                    std::fprintf(stderr, "nonencodable compact scale\n"); std::exit(1);
                }
            }
        }
        auto* compact_weight = allocate<mollm_cuda::Q4CompactG32>(compact_host.size());
        check(cudaMemcpy(compact_weight, compact_host.data(), compact_host.size() * sizeof(compact_host[0]), cudaMemcpyHostToDevice));
        // Verify the prefill bridge too: expanding either layout must yield
        // identical FP16 weights and activation views, without a second
        // persistent canonical device-weight allocation in production.
        const size_t expanded_count = static_cast<size_t>(rows) * inner;
        auto* expanded = allocate<__half>(expanded_count);
        auto* converted = allocate<__half>(inner + 1);
        std::vector<__half> reference_half(expanded_count), candidate_half(expanded_count);
        std::vector<__half> reference_activation(inner + 1), candidate_activation(inner + 1);
        mollm_cuda::launch_prepare_q4_g32_dense_gemm(weight,
            reinterpret_cast<__half2*>(expanded), input, converted, inner + 1, rows, inner / 32);
        check(cudaMemcpy(reference_half.data(), expanded, expanded_count * sizeof(__half), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(reference_activation.data(), converted, (inner + 1) * sizeof(__half), cudaMemcpyDeviceToHost));
        mollm_cuda::launch_prepare_q4_compact_g32_dense_gemm(compact_weight,
            reinterpret_cast<__half2*>(expanded), input, converted, inner + 1, rows, inner / 32);
        check(cudaMemcpy(candidate_half.data(), expanded, expanded_count * sizeof(__half), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(candidate_activation.data(), converted, (inner + 1) * sizeof(__half), cudaMemcpyDeviceToHost));
        if (std::memcmp(reference_half.data(), candidate_half.data(), expanded_count * sizeof(__half)) ||
            std::memcmp(reference_activation.data(), candidate_activation.data(), (inner + 1) * sizeof(__half))) {
            std::fprintf(stderr, "compact W4 prefill preparation mismatch\n"); std::exit(1);
        }
        check(cudaFree(expanded)); check(cudaFree(converted));
        std::printf("q4-compact prefill_prepare_bit_exact=true\n");
        std::vector<float> reference(rows), actual(rows);
        for (int set = 0; set < std::min(sets, 8); ++set) {
            mollm_cuda::launch_q4_g32_dense_gemv(input, weight + set * blocks, output, rows, inner);
            check(cudaMemcpy(reference.data(), output, rows * sizeof(float), cudaMemcpyDeviceToHost));
            mollm_cuda::launch_q4_g32_compact_gemv(input, compact_weight + set * blocks, output, rows, inner);
            check(cudaMemcpy(actual.data(), output, rows * sizeof(float), cudaMemcpyDeviceToHost));
            if (std::memcmp(reference.data(), actual.data(), rows * sizeof(float))) {
                std::fprintf(stderr, "compact W4 output mismatch\n"); std::exit(1);
            }
        }
        for (int repeat = 0; repeat < 3; ++repeat) {
            const double candidate = measure([&](int i) {
                mollm_cuda::launch_q4_g32_compact_gemv(input, compact_weight + (i % sets) * blocks, output, rows, inner);
            }, std::max(sets, 256));
            const double control = measure([&](int i) {
                mollm_cuda::launch_q4_g32_dense_gemv(input, weight + (i % sets) * blocks, output, rows, inner);
            }, std::max(sets, 256));
            std::printf("q4-compact rows=%d inner=%d repeat=%d us=%.3f control_us=%.3f bit_exact=true\n",
                rows, inner, repeat, candidate, control);
        }
        if (std::getenv("MOLLM_Q4_COMPACT_INTEGER")) {
            for (int set = 0; set < sets; ++set) {
                mollm_cuda::launch_q4_g32_compact_gemv(input, compact_weight + set * blocks, output, rows, inner);
                check(cudaMemcpy(reference.data(), output, rows * sizeof(float), cudaMemcpyDeviceToHost));
                check(mollm_cuda::launch_q4_compact_integer_gemv(input, compact_weight + set * blocks, output, rows, inner));
                check(cudaMemcpy(actual.data(), output, rows * sizeof(float), cudaMemcpyDeviceToHost));
                if (std::memcmp(actual.data(), reference.data(), rows * sizeof(float))) {
                    std::fprintf(stderr, "compact integer decoder mismatch\n"); std::exit(1);
                }
            }
            for (int repeat = 0; repeat < 3; ++repeat) {
                auto direct = [&] {
                    const int iterations = std::max(sets, 256);
                    const double us = measure_graph([&](int i, cudaStream_t stream) {
                        check(mollm_cuda::launch_q4_compact_integer_gemv(input, compact_weight + (i % sets) * blocks, output, rows, inner, stream));
                    }, iterations);
                    check(cudaMemcpy(actual.data(), output, rows * sizeof(float), cudaMemcpyDeviceToHost));
                    mollm_cuda::launch_q4_g32_compact_gemv(input, compact_weight + ((iterations - 1) % sets) * blocks, output, rows, inner);
                    check(cudaMemcpy(reference.data(), output, rows * sizeof(float), cudaMemcpyDeviceToHost));
                    if (std::memcmp(actual.data(), reference.data(), rows * sizeof(float))) {
                        std::fprintf(stderr, "compact integer graph mismatch\n"); std::exit(1);
                    }
                    return us;
                };
                auto table = [&] { return measure_graph([&](int i, cudaStream_t stream) {
                    mollm_cuda::launch_q4_g32_compact_gemv(input, compact_weight + (i % sets) * blocks, output, rows, inner, stream);
                }, std::max(sets, 256)); };
                double direct_us, table_us;
                if (repeat % 2) { direct_us = direct(); table_us = table(); }
                else { table_us = table(); direct_us = direct(); }
                std::printf("compact-integer rows=%d K=%d repeat=%d integer_us=%.3f table_us=%.3f bit_exact=true\n", rows, inner, repeat, direct_us, table_us);
            }
        }
        if (std::getenv("MOLLM_Q4_COMPACT_CACHE_PROBE")) {
            const bool integer_scale = rows == 13312;
            const bool init_barrier = std::getenv("MOLLM_Q4_COMPACT_INIT_BARRIER_PROBE") != nullptr;
            const bool cta_probe = std::getenv("MOLLM_Q4_COMPACT_CTA_PROBE") != nullptr;
            const bool lookahead_probe = std::getenv("MOLLM_Q4_COMPACT_LOOKAHEAD_PROBE") != nullptr;
            const int table_hint=std::getenv("MOLLM_Q4_COMPACT_CONSTANT_TABLE_PROBE")?9:8;
            const bool global_table_probe = std::getenv("MOLLM_Q4_COMPACT_GLOBAL_TABLE_PROBE") != nullptr || table_hint==9;
            if ((init_barrier || cta_probe || lookahead_probe || global_table_probe) && integer_scale) { std::fprintf(stderr,"probe requires table scale geometry\n"); std::exit(1); }
            auto launch = [&](const mollm_cuda::Q4CompactG32* selected, int mode, cudaStream_t stream) {
                if (mode) check(mollm_cuda::launch_q4_compact_cache_probe(input,selected,output,rows,inner,integer_scale,mode,stream));
                else if (integer_scale) check(mollm_cuda::launch_q4_compact_integer_gemv(input,selected,output,rows,inner,stream));
                else mollm_cuda::launch_q4_g32_compact_gemv(input,selected,output,rows,inner,stream);
            };
            for (int set=0; set<sets; ++set) {
                launch(compact_weight+set*blocks,0,nullptr);
                check(cudaMemcpy(reference.data(),output,rows*sizeof(float),cudaMemcpyDeviceToHost));
                for (int mode : (global_table_probe ? std::vector<int>{table_hint} : lookahead_probe ? std::vector<int>{6,7} : cta_probe ? std::vector<int>{4,5} : init_barrier ? std::vector<int>{3} : std::vector<int>{1,2})) {
                    launch(compact_weight+set*blocks,mode,nullptr);
                    check(cudaMemcpy(actual.data(),output,rows*sizeof(float),cudaMemcpyDeviceToHost));
                    if (std::memcmp(actual.data(),reference.data(),rows*sizeof(float))) { std::fprintf(stderr,"compact cache mismatch\n"); std::exit(1); }
                }
            }
            auto time = [&](int mode) {
                const int iterations=std::max(sets,256);
                const double us=measure_graph([&](int i,cudaStream_t stream){launch(compact_weight+(i%sets)*blocks,mode,stream);},iterations);
                check(cudaMemcpy(actual.data(),output,rows*sizeof(float),cudaMemcpyDeviceToHost));
                launch(compact_weight+((iterations-1)%sets)*blocks,0,nullptr);
                check(cudaMemcpy(reference.data(),output,rows*sizeof(float),cudaMemcpyDeviceToHost));
                if (std::memcmp(actual.data(),reference.data(),rows*sizeof(float))) { std::fprintf(stderr,"compact cache graph mismatch\n"); std::exit(1); }
                return us;
            };
            for (int repeat=0; repeat<3; ++repeat) {
                if(global_table_probe){
                    double baseline,candidate;
                    if(repeat%2){candidate=time(table_hint);baseline=time(0);}
                    else{baseline=time(0);candidate=time(table_hint);}
                    std::printf("compact-table-storage hint=%d rows=%d K=%d repeat=%d baseline_us=%.3f candidate_us=%.3f bit_exact=true\n",table_hint,rows,inner,repeat,baseline,candidate);
                    continue;
                }
                if (lookahead_probe) {
                    double baseline,candidate,no_prefetch;
                    if(repeat%2){no_prefetch=time(7);candidate=time(6);baseline=time(0);}
                    else{baseline=time(0);candidate=time(6);no_prefetch=time(7);}
                    std::printf("compact-lookahead rows=%d K=%d repeat=%d baseline_us=%.3f candidate_us=%.3f no_prefetch_us=%.3f bit_exact=true\n",rows,inner,repeat,baseline,candidate,no_prefetch);
                    continue;
                }
                if (cta_probe) {
                    double baseline,two,eight;
                    if(repeat%2){eight=time(5);two=time(4);baseline=time(0);}
                    else{baseline=time(0);two=time(4);eight=time(5);}
                    std::printf("compact-cta rows=%d K=%d repeat=%d four_us=%.3f two_us=%.3f eight_us=%.3f bit_exact=true\n",rows,inner,repeat,baseline,two,eight);
                    continue;
                }
                if (init_barrier) {
                    double baseline,candidate;
                    if(repeat%2){candidate=time(3);baseline=time(0);}
                    else{baseline=time(0);candidate=time(3);}
                    std::printf("compact-init-barrier rows=%d K=%d repeat=%d baseline_us=%.3f candidate_us=%.3f bit_exact=true\n",rows,inner,repeat,baseline,candidate);
                    continue;
                }
                double baseline,cg,cs;
                if (repeat%2) { cs=time(2); cg=time(1); baseline=time(0); }
                else { baseline=time(0); cg=time(1); cs=time(2); }
                std::printf("compact-cache rows=%d K=%d integer=%d repeat=%d baseline_us=%.3f cg_us=%.3f cs_us=%.3f bit_exact=true\n",rows,inner,integer_scale,repeat,baseline,cg,cs);
            }
        }
        check(cudaFree(compact_weight)); check(cudaFree(weight)); check(cudaFree(input)); check(cudaFree(output));
        return;
    }
    if (rows <= 4096 && inner >= 2048) {
        std::vector<float> reference(rows), actual(rows);
        if(std::getenv("MOLLM_Q4_PIPELINE_SINGLE_BARRIER_PROBE")) {
            auto launch=[&](int i,cudaStream_t stream,bool single) {
                check(mollm_cuda::launch_q4_g32_pipelined_narrow_gemv(input,
                    weight+(i%sets)*blocks,output,rows,inner,1,true,stream,single));
            };
            for(int set=0;set<sets;++set) {
                launch(set,nullptr,false);
                check(cudaMemcpy(reference.data(),output,rows*sizeof(float),cudaMemcpyDeviceToHost));
                launch(set,nullptr,true);
                check(cudaMemcpy(actual.data(),output,rows*sizeof(float),cudaMemcpyDeviceToHost));
                if(std::memcmp(reference.data(),actual.data(),rows*sizeof(float)))std::exit(1);
            }
            const int iterations=std::max(sets,256);
            auto timing=[&](bool single) {
                const double us=measure_graph([&](int i,cudaStream_t stream){launch(i,stream,single);},iterations);
                check(cudaMemcpy(actual.data(),output,rows*sizeof(float),cudaMemcpyDeviceToHost));
                launch(iterations-1,nullptr,false);
                check(cudaMemcpy(reference.data(),output,rows*sizeof(float),cudaMemcpyDeviceToHost));
                if(std::memcmp(reference.data(),actual.data(),rows*sizeof(float)))std::exit(1);
                return us;
            };
            int probe_repeats=3;
            if(const char* value=std::getenv("MOLLM_Q4_SINGLE_BARRIER_REPEATS")) {
                char* end=nullptr;
                const long parsed=std::strtol(value,&end,10);
                if(end==value || *end || parsed<3 || parsed>100) {
                    std::fprintf(stderr,"MOLLM_Q4_SINGLE_BARRIER_REPEATS must be 3..100\n");
                    std::exit(1);
                }
                probe_repeats=static_cast<int>(parsed);
            }
            for(int repeat=0;repeat<probe_repeats;++repeat) {
                double candidate,control;
                if(repeat%2){candidate=timing(true);control=timing(false);}
                else{control=timing(false);candidate=timing(true);}
                std::printf("q4-pipeline-single-barrier rows=%d K=%d repeat=%d us=%.3f control_us=%.3f bit_exact=true\n",
                    rows,inner,repeat,candidate,control);
            }
            cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;
            check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            launch(sets-1,stream,true);
            check(cudaStreamEndCapture(stream,&graph));
            check(cudaGraphInstantiateWithFlags(&executable,graph,0));
            for(int replay=0;replay<2;++replay) {
                for(float& value:x)value=-value;
                check(cudaMemcpyAsync(input,x.data(),inner*sizeof(float),cudaMemcpyHostToDevice,stream));
                check(cudaGraphLaunch(executable,stream));
                check(cudaMemcpyAsync(actual.data(),output,rows*sizeof(float),cudaMemcpyDeviceToHost,stream));
                launch(sets-1,stream,false);
                check(cudaMemcpyAsync(reference.data(),output,rows*sizeof(float),cudaMemcpyDeviceToHost,stream));
                check(cudaStreamSynchronize(stream));
                if(std::memcmp(reference.data(),actual.data(),rows*sizeof(float)))std::exit(1);
            }
            check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
            std::puts("q4-pipeline-single-barrier changed_input_graph_bit_exact=true");
            check(cudaFree(weight));check(cudaFree(input));check(cudaFree(output));return;
        }
        if (std::getenv("MOLLM_Q4_PIPELINE_PROBE")) {
            const bool graph_control=std::getenv("MOLLM_Q4_PIPELINE_GRAPH_CONTROL")!=nullptr;
            for (bool inputs : {false, true}) for (int depth : {1, 2, 4}) {
                const auto launch = [&](const Q4B8G32Block* selected) {
                    check(mollm_cuda::launch_q4_g32_pipelined_narrow_gemv(input, selected, output, rows, inner, depth, inputs));
                };
                for (int set = 0; set < sets; ++set) {
                    mollm_cuda::launch_q4_g32_dense_gemv(input, weight + set * blocks, output, rows, inner);
                    check(cudaMemcpy(reference.data(), output, rows * sizeof(float), cudaMemcpyDeviceToHost));
                    launch(weight + set * blocks);
                    check(cudaMemcpy(actual.data(), output, rows * sizeof(float), cudaMemcpyDeviceToHost));
                    if (std::memcmp(reference.data(), actual.data(), rows * sizeof(float))) {
                        std::fprintf(stderr, "narrow pipeline mismatch depth=%d inputs=%d set=%d\n", depth, inputs, set); std::exit(1);
                    }
                }
                for (int repeat = 0; repeat < 3; ++repeat) {
                    const int iterations = std::max(sets, 256);
                    if(graph_control) {
                        auto timing=[&](bool candidate) {
                            const double us=measure_graph([&](int i,cudaStream_t stream){
                                check(mollm_cuda::launch_q4_g32_pipelined_narrow_gemv(input,weight+(i%sets)*blocks,output,rows,inner,candidate?depth:1,candidate?inputs:true,stream));
                            },iterations);
                            check(cudaMemcpy(actual.data(),output,rows*sizeof(float),cudaMemcpyDeviceToHost));
                            check(mollm_cuda::launch_q4_g32_pipelined_narrow_gemv(input,weight+((iterations-1)%sets)*blocks,output,rows,inner,1,true));
                            check(cudaMemcpy(reference.data(),output,rows*sizeof(float),cudaMemcpyDeviceToHost));
                            if(std::memcmp(reference.data(),actual.data(),rows*sizeof(float))) {std::fprintf(stderr,"pipeline final graph mismatch\n");std::exit(1);}
                            return us;
                        };
                        double candidate,control;
                        if(repeat%2){control=timing(false);candidate=timing(true);}
                        else{candidate=timing(true);control=timing(false);}
                        std::printf("q4-pipeline-graph rows=%d inner=%d sets=%d depth=%d inputs=%d repeat=%d candidate_us=%.3f depth1_us=%.3f bit_exact=true\n",rows,inner,sets,depth,inputs,repeat,candidate,control);
                        continue;
                    }
                    const double candidate = measure([&](int i) { launch(weight + (i % sets) * blocks); }, iterations);
                    const double control = measure([&](int i) {
                        mollm_cuda::launch_q4_g32_dense_gemv(input, weight + (i % sets) * blocks, output, rows, inner);
                    }, iterations);
                    std::printf("q4-pipeline rows=%d inner=%d depth=%d inputs=%d repeat=%d us=%.3f control_us=%.3f bit_exact=true\n",
                        rows, inner, depth, inputs, repeat, candidate, control);
                }
            }
            check(cudaFree(weight)); check(cudaFree(input)); check(cudaFree(output)); return;
        }
        const size_t row_blocks = static_cast<size_t>(rows) * (inner / 32);
        std::vector<mollm_cuda::Q4RowG32> row_host(row_blocks * sets);
        for (int set = 0; set < sets; ++set)
            for (int row = 0; row < rows; ++row)
                for (int group = 0; group < inner / 32; ++group) {
                    const auto& source = host[set * blocks + (row / 8) * (inner / 32) + group];
                    auto& target = row_host[set * row_blocks + row * (inner / 32) + group];
                    target.scale = source.scales[row % 8];
                    std::memcpy(target.q, source.q[row % 8], 16);
                }
        auto* row_weight = allocate<mollm_cuda::Q4RowG32>(row_host.size());
        check(cudaMemcpy(row_weight, row_host.data(), row_host.size() * sizeof(row_host[0]), cudaMemcpyHostToDevice));
        for (int set = 0; set < sets; ++set) {
            mollm_cuda::launch_q4_g32_dense_gemv(input, weight + set * blocks, output, rows, inner);
            check(cudaMemcpy(reference.data(), output, rows * sizeof(float), cudaMemcpyDeviceToHost));
            mollm_cuda::launch_q4_row_g32_gemv(input, row_weight + set * row_blocks, output, rows, inner);
            check(cudaMemcpy(actual.data(), output, rows * sizeof(float), cudaMemcpyDeviceToHost));
            if (std::memcmp(reference.data(), actual.data(), rows * sizeof(float))) {
                std::fprintf(stderr, "row-major W4 is not bit-exact\n"); std::exit(1);
            }
        }
        const double row_time = measure([&](int i) {
            mollm_cuda::launch_q4_row_g32_gemv(input, row_weight + (i % sets) * row_blocks, output, rows, inner);
        }, std::max(sets, 256));
        const double row_control = measure([&](int i) {
            mollm_cuda::launch_q4_g32_dense_gemv(input, weight + (i % sets) * blocks, output, rows, inner);
        }, std::max(sets, 256));
        std::printf("q4-row-major rows=%d inner=%d us=%.3f control_us=%.3f speedup=%.3f bit_exact=true\n", rows, inner, row_time, row_control, row_control / row_time);
        check(cudaFree(row_weight));
    }
    check(cudaFree(weight)); check(cudaFree(input)); check(cudaFree(output));
}
void nvfp4(int rows, int inner, bool pair_layout) {
    constexpr int experts = 10;
    const size_t data_bytes = static_cast<size_t>(rows) * inner / 2;
    const size_t scale_bytes = static_cast<size_t>(rows) * inner / 16;
    const size_t per_set = experts * (data_bytes + scale_bytes + rows * sizeof(float));
    const int sets = std::max<size_t>(8, (128ull << 20) / per_set);
    std::vector<uint8_t> packed(data_bytes * experts * sets, 0x63);
    std::vector<uint8_t> scales(scale_bytes * experts * sets, 56);
    std::vector<float> row_scales(static_cast<size_t>(rows) * experts * sets, 1.f/64);
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = (i * 37 + i / 19) & 255;
    for (size_t i = 0; i < scales.size(); ++i) scales[i] = 48 + i % 24;
    if (mlock(packed.data(), packed.size()) || mlock(scales.data(), scales.size()) ||
        mlock(row_scales.data(), row_scales.size() * sizeof(float))) {
        std::perror("mlock NVFP4 benchmark"); std::exit(1);
    }
    auto* input = allocate<float>(inner);
    auto* output = allocate<float>(rows * experts);
    std::vector<float> x(inner, .125f);
    check(cudaMemcpy(input, x.data(), inner * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<mollm_cuda::Nvfp4Projection> jobs(experts * sets);
    for (size_t i = 0; i < jobs.size(); ++i)
        jobs[i] = {packed.data() + i * data_bytes, scales.data() + i * scale_bytes,
                   row_scales.data() + i * rows, input, output + (i % experts) * rows, pair_layout};
    const double us = measure([&](int i) {
        mollm_cuda::launch_nvfp4_projections(jobs.data() + (i % sets) * experts, experts, rows, inner);
    }, 256);
    std::printf("nvfp4 layout=%s rows=%d inner=%d experts=%d sets=%d us=%.3f GBps=%.2f\n",
                pair_layout ? "pair" : "row", rows, inner, experts, sets, us, per_set / us / 1000.);
    check(cudaFree(input)); check(cudaFree(output));
    munlock(packed.data(), packed.size()); munlock(scales.data(), scales.size());
    munlock(row_scales.data(), row_scales.size() * sizeof(float));
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--q8") == 0) { q8_probe(); return 0; }
    if (argc != 1) {
        if (argc != 3) {
            std::fprintf(stderr, "usage: bench_cuda_projections [rows inner]\n");
            return 1;
        }
        char* rows_end = nullptr;
        char* inner_end = nullptr;
        const long rows = std::strtol(argv[1], &rows_end, 10);
        const long inner = std::strtol(argv[2], &inner_end, 10);
        if (!*argv[1] || !*argv[2] || *rows_end || *inner_end ||
            rows <= 0 || rows > 1048576 || inner <= 0 ||
            inner > 1048576 || inner % 32) {
            std::fprintf(stderr, "invalid Q4 shape (inner must be a multiple of 32)\n");
            return 1;
        }
        q4(static_cast<int>(rows), static_cast<int>(inner));
        return 0;
    }
    for (auto shape : {std::pair<int,int>{4,10240}, {320,10240}, {2560,6144},
                       {16480,2560}, {13312,2560}, {10240,320}})
        q4(shape.first, shape.second);
    int coherent = 0, host_tables = 0;
    check(cudaDeviceGetAttribute(&coherent, cudaDevAttrPageableMemoryAccess, 0));
    check(cudaDeviceGetAttribute(&host_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0));
    if (coherent && host_tables) {
        nvfp4(1280,2560,true); nvfp4(2560,640,true);
        nvfp4(1280,2560,false); nvfp4(2560,640,false);
    }
    return 0;
}
