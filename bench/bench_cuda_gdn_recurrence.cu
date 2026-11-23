#include "kernels/cuda/recurrent.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

static void check(cudaError_t status) {
    if (status != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(status)); std::exit(1); }
}
struct Buffer {
    float* data = nullptr;
    explicit Buffer(size_t n) { check(cudaMalloc(&data, n * 4)); }
    ~Buffer() { cudaFree(data); }
};
static bool butterfly_key = false;
static bool broadcast_gates = false;
static bool paired_values = false;
static bool quad_values = false;
static bool run(int sequence, int real, int heads, int values, bool timing, int candidate_warps = 2) {
    const size_t qkv_count = size_t(sequence) * (2 * heads + values) * 128;
    const size_t qk_count = size_t(sequence) * heads * 128, dot_count = size_t(sequence) * heads;
    const size_t gate_count = size_t(sequence) * values, state_count = size_t(values) * 128 * 128;
    const size_t out_count = gate_count * 128;
    const size_t output_offset=quad_values?1:0;
    Buffer qkv(qkv_count), q(qk_count), k(qk_count), dot(dot_count), decay(gate_count), beta(gate_count);
    Buffer old_state(state_count + 8), new_state(state_count + 8), old_output(out_count + output_offset + 8), new_output(out_count + output_offset + 8);
    std::vector<float> host(std::max({qkv_count, state_count + 8, out_count + output_offset + 8}));
    auto fill = [&](Buffer& buffer, size_t count, int seed, float factor, float bias) {
        for (size_t i = 0; i < count; ++i) host[i] = (int((i * 7 + seed) % 31) - 15) * factor + bias;
        check(cudaMemcpy(buffer.data, host.data(), count * 4, cudaMemcpyHostToDevice));
    };
    auto launch = [&](bool pipeline) {
        if (!mollm_cuda::launch_gdn_recurrence_probe(qkv.data, q.data, k.data, dot.data,
                decay.data, beta.data, pipeline ? new_state.data : old_state.data,
                (pipeline ? new_output.data : old_output.data)+output_offset, heads, values, sequence, real, .125f,
                pipeline && candidate_warps == 2 && !butterfly_key && !broadcast_gates && !paired_values && !quad_values, pipeline ? candidate_warps : 2,
                pipeline && butterfly_key, pipeline && broadcast_gates,
                (pipeline && paired_values) || (!pipeline && quad_values),
                pipeline && quad_values)) std::exit(2);
    };
    if(quad_values) {
        const auto rejected=[&](bool prefetch,int warps,bool butterfly,bool broadcast,bool paired,int offset) {
            return !mollm_cuda::launch_gdn_recurrence_probe(qkv.data,q.data,k.data,dot.data,
                decay.data,beta.data,new_state.data+offset,new_output.data,
                heads,values,sequence,real,.125f,prefetch,warps,butterfly,broadcast,paired,true);
        };
        if(!rejected(true,2,false,false,false,0) || !rejected(false,4,false,false,false,0) ||
            !rejected(false,2,true,false,false,0) || !rejected(false,2,false,true,false,0) ||
            !rejected(false,2,false,false,true,0) || !rejected(false,2,false,false,false,1) ||
            !rejected(false,2,false,false,false,2) || !rejected(false,2,false,false,false,3) ||
            !rejected(false,8,false,false,false,0))return false;
        for(int fault=0;fault<21;++fault) {
            if(mollm_cuda::launch_gdn_recurrence_probe(
                fault==0?nullptr:qkv.data,fault==1?nullptr:q.data,fault==2?nullptr:k.data,
                fault==3?nullptr:dot.data,fault==4?nullptr:decay.data,fault==5?nullptr:beta.data,
                fault==6?nullptr:new_state.data,fault==7?nullptr:new_output.data,
                fault==8?0:fault==9?-1:fault==10?8388608:fault==14?values+1:heads,
                fault==11?0:fault==12?-1:fault==13?65536:values,
                fault==15?0:fault==16?-1:fault==17?2147483647:sequence,
                fault==18?-1:fault==19?sequence+1:real,.125f,
                false,fault==20?1:2,false,false,false,true))return false;
        }
        std::printf("quad_invalid_arguments_rejected=30 seq=%d real=%d\n",sequence,real);
    }
    if (paired_values) {
        const auto rejected = [&](bool prefetch, int warps, bool butterfly, bool broadcast,
                                  int state_offset, int output_offset) {
            return !mollm_cuda::launch_gdn_recurrence_probe(qkv.data,q.data,k.data,dot.data,
                decay.data,beta.data,new_state.data + state_offset,new_output.data + output_offset,
                heads,values,sequence,real,.125f,prefetch,warps,butterfly,broadcast,true);
        };
        if (!rejected(true,2,false,false,0,0) || !rejected(false,4,false,false,0,0) ||
            !rejected(false,2,true,false,0,0) || !rejected(false,2,false,true,0,0) ||
            !rejected(false,2,false,false,1,0)) return false;
    }
    if (butterfly_key &&
        (mollm_cuda::launch_gdn_recurrence_probe(qkv.data,q.data,k.data,dot.data,decay.data,beta.data,
            new_state.data,new_output.data,heads,values,sequence,real,.125f,true,2,true) ||
         mollm_cuda::launch_gdn_recurrence_probe(qkv.data,q.data,k.data,dot.data,decay.data,beta.data,
            new_state.data,new_output.data,heads,values,sequence,real,.125f,false,4,true))) return false;
    if (broadcast_gates &&
        (mollm_cuda::launch_gdn_recurrence_probe(qkv.data,q.data,k.data,dot.data,decay.data,beta.data,
            new_state.data,new_output.data,heads,values,sequence,real,.125f,true,2,false,true) ||
         mollm_cuda::launch_gdn_recurrence_probe(qkv.data,q.data,k.data,dot.data,decay.data,beta.data,
            new_state.data,new_output.data,heads,values,sequence,real,.125f,false,4,false,true) ||
         mollm_cuda::launch_gdn_recurrence_probe(qkv.data,q.data,k.data,dot.data,decay.data,beta.data,
            new_state.data,new_output.data,heads,values,sequence,real,.125f,false,2,true,true))) return false;
    cudaGraph_t graph; cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)); launch(true);
    check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    std::vector<float> expected(std::max(state_count, out_count+output_offset) + 8), actual(expected.size());
    const int replays=quad_values?7:3;
    for (int replay = 0; replay < replays; ++replay) {
        fill(qkv, qkv_count, replay, .01f, 0.f); fill(q, qk_count, replay, .002f, 0.f);
        fill(k, qk_count, replay + 3, .002f, 0.f); fill(dot, dot_count, replay, .0001f, 0.f);
        fill(decay, gate_count, replay, .0001f, .99f); fill(beta, gate_count, replay, .001f, .1f);
        fill(old_state, state_count, replay, .002f, 0.f);
        if(quad_values && replay>=3) {
            const std::pair<Buffer*,size_t> buffers[]={{&qkv,qkv_count},{&q,qk_count},{&k,qk_count},
                                                    {&dot,dot_count},{&old_state,state_count}};
            for(auto item:buffers) {
                for(size_t i=0;i<item.second;++i)host[i]=replay==3?((i&1)?-0.f:0.f):
                    replay==4?((i&1)?-1e-38f:1e-38f):replay==5?((i&1)?-1e20f:1e20f):
                    i%3==0?NAN:i%3==1?INFINITY:-INFINITY;
                check(cudaMemcpy(item.first->data,host.data(),item.second*4,cudaMemcpyHostToDevice));
            }
        }
        check(cudaMemcpy(new_state.data, old_state.data, state_count * 4, cudaMemcpyDeviceToDevice));
        for (Buffer* buffer : {&old_output, &new_output}) check(cudaMemset(buffer->data, 0x7f, (out_count + output_offset + 8) * 4));
        check(cudaMemset(old_state.data + state_count, 0x7f, 32)); check(cudaMemset(new_state.data + state_count, 0x7f, 32));
        launch(false); check(cudaDeviceSynchronize());
        check(cudaGraphLaunch(exec, cudaStreamPerThread)); check(cudaStreamSynchronize(cudaStreamPerThread));
        for (int state = 0; state < 2; ++state) {
            const size_t count = (state ? state_count : out_count+output_offset) + 8;
            check(cudaMemcpy(expected.data(), state ? old_state.data : old_output.data, count * 4, cudaMemcpyDeviceToHost));
            check(cudaMemcpy(actual.data(), state ? new_state.data : new_output.data, count * 4, cudaMemcpyDeviceToHost));
            if (std::memcmp(expected.data(), actual.data(), count * 4)) {
                std::fprintf(stderr, "not exact seq=%d real=%d heads=%d values=%d state=%d replay=%d\n", sequence, real, heads, values, state, replay);
                return false;
            }
            for(size_t i=count-8;i<count;++i) {
                unsigned a,b;std::memcpy(&a,&expected[i],4);std::memcpy(&b,&actual[i],4);
                if(a!=0x7f7f7f7fu || b!=0x7f7f7f7fu)return false;
            }
            if(!state && output_offset) {
                unsigned a,b;std::memcpy(&a,&expected[0],4);std::memcpy(&b,&actual[0],4);
                if(a!=0x7f7f7f7fu || b!=0x7f7f7f7fu)return false;
            }
        }
    }
    check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
    std::printf("bit_exact=true seq=%d real=%d heads=%d values=%d candidate_warps=%d replays=%d output_offset=%zu independent_guards=true changed_input_graph=true\n", sequence, real, heads, values, candidate_warps,replays,output_offset);
    if (timing && !std::getenv("MOLLM_GDN_RECURRENCE_CHECK_ONLY")) {
        fill(qkv,qkv_count,0,.01f,0.f);fill(q,qk_count,0,.002f,0.f);fill(k,qk_count,3,.002f,0.f);
        fill(dot,dot_count,0,.0001f,0.f);fill(decay,gate_count,0,.0001f,.99f);fill(beta,gate_count,0,.001f,.1f);
        cudaEvent_t start, end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
        for (int iteration = 0; iteration < 7; ++iteration) for (int order = 0; order < 2; ++order) {
            const bool pipeline = (iteration + order) % 2;
            check(cudaMemset(old_state.data, 0, state_count * 4)); check(cudaMemset(new_state.data, 0, state_count * 4));
            launch(pipeline); check(cudaDeviceSynchronize()); check(cudaEventRecord(start));
            for (int repeat = 0; repeat < 8; ++repeat) launch(pipeline);
            check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
            float ms; check(cudaEventElapsedTime(&ms, start, end));
            std::printf("timing seq=%d iteration=%d pipeline=%d us=%.3f\n", sequence, iteration, pipeline, ms * 1000 / 8);
        }
        check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
    }
    return true;
}
static bool shortconv_probe(int groups, int sequence, int kernel, int real, bool feature_major, bool timing, int mapping = 1, bool padded = false) {
    const size_t features = feature_major ? sequence + (padded ? 3 : 0) : 1;
    const size_t rows = feature_major ? 1 : groups + (padded ? 3 : 0);
    const size_t state_count = size_t(groups) * std::max(1, kernel - 1);
    const size_t count = size_t(groups) * sequence;
    const size_t input_count = feature_major ? size_t(groups) * features : size_t(sequence) * rows;
    Buffer input(input_count), weight(size_t(groups) * kernel), a(state_count + 8), b(state_count + 8), output(count + 8);
    std::vector<float> host(std::max({input_count, state_count + 8, size_t(groups) * kernel}));
    auto fill = [&](Buffer& buffer, size_t n, int seed) {
        for (size_t i = 0; i < n; ++i) host[i] = (int((i * 13 + seed) % 37) - 18) * .02f;
        check(cudaMemcpy(buffer.data, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    };
    auto launch = [&](bool tiled) {
        if (!mollm_cuda::launch_shortconv(input.data, weight.data, tiled ? b.data : a.data, output.data,
                groups, sequence, kernel, real, features, rows, tiled ? mapping : 0)) std::exit(2);
        check(cudaGetLastError());
    };
    cudaGraph_t graph; cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)); launch(true);
    check(cudaStreamEndCapture(cudaStreamPerThread, &graph)); check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    std::vector<float> expected(count + 8), actual(count + 8), expected_state(state_count + 8), actual_state(state_count + 8);
    for (int replay = 0; replay < 4; ++replay) {
        fill(input, input_count, replay); fill(weight, size_t(groups) * kernel, replay + 2); fill(a, state_count, replay + 4);
        check(cudaMemset(a.data + state_count, 0x7f, 32));
        check(cudaMemcpy(b.data, a.data, (state_count + 8) * sizeof(float), cudaMemcpyDeviceToDevice));
        check(cudaMemset(output.data, 0x7f, (count + 8) * sizeof(float))); launch(false); check(cudaDeviceSynchronize());
        check(cudaMemcpy(expected.data(), output.data, expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(expected_state.data(), a.data, expected_state.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemset(output.data, 0x7f, (count + 8) * sizeof(float)));
        check(cudaGraphLaunch(executable, cudaStreamPerThread)); check(cudaStreamSynchronize(cudaStreamPerThread));
        check(cudaMemcpy(actual.data(), output.data, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(actual_state.data(), b.data, actual_state.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) ||
            std::memcmp(actual_state.data(), expected_state.data(), actual_state.size() * sizeof(float))) return false;
    }
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
    std::printf("shortconv_exact groups=%d seq=%d kernel=%d real=%d feature_major=%d padded=%d mapping=%d\n", groups, sequence, kernel, real, feature_major, padded, mapping);
    if (timing) {
        cudaEvent_t start, end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
        for (int iteration = 0; iteration < 7; ++iteration) for (int order = 0; order < 2; ++order) {
            bool tiled = (iteration + order) % 2; launch(tiled); check(cudaDeviceSynchronize()); check(cudaEventRecord(start));
            for (int repeat = 0; repeat < 16; ++repeat) launch(tiled);
            check(cudaEventRecord(end)); check(cudaEventSynchronize(end)); float ms; check(cudaEventElapsedTime(&ms, start, end));
            std::printf("shortconv_timing seq=%d feature_major=%d tiled=%d us=%.3f\n", sequence, feature_major, tiled, ms * 1000 / 16);
        }
        check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
    }
    return true;
}
int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    if (argc == 2 && (!std::strcmp(argv[1], "--shortconv") || !std::strcmp(argv[1], "--shortconv-transpose"))) {
        const int mapping = !std::strcmp(argv[1], "--shortconv-transpose") ? 2 : 1;
        Buffer stub(8);
        auto rejected = [&](int groups, int sequence, int selected) {
            return !mollm_cuda::launch_shortconv(stub.data, stub.data, stub.data, stub.data,
                groups, sequence, 4, 1, 1, groups, selected);
        };
        if (!rejected(19, 17, -1) || !rejected(19, 17, 3) ||
            !rejected(2147483647, 17, 0) || !rejected(19, 2147483647, 2) ||
            !rejected(65536, 17, 1) || !rejected(19, 65536, 0) ||
            !rejected(19, 524281, 2)) return 1;
        for (bool padded : {false, true}) for (bool feature : {false, true}) for (int sequence : {1, 3, 17, 256, 857})
            for (int kernel : {1, 2, 4, 8}) for (int real : {0, 1, sequence - 1, sequence})
                if (!shortconv_probe(19, sequence, kernel, real, feature, false, mapping, padded)) return 1;
        for (bool feature : {false, true})
            if (!shortconv_probe(10240, 857, 4, 857, feature, true, mapping)) return 1;
        return 0;
    }
    int candidate_warps = 2;
    if (argc == 2 && !std::strcmp(argv[1],"--key-butterfly")) butterfly_key = true;
    else if (argc == 2 && !std::strcmp(argv[1],"--gate-broadcast")) broadcast_gates = true;
    else if (argc == 2 && !std::strcmp(argv[1],"--paired-values")) paired_values = true;
    else if (argc == 2 && !std::strcmp(argv[1],"--quad-values")) quad_values = true;
    else if (argc == 2 && (!std::strcmp(argv[1], "4") || !std::strcmp(argv[1], "8"))) candidate_warps = std::atoi(argv[1]);
    else if (argc != 1) return 2;
    for (int sequence : {1, 3, 17, 256, 857}) for (int real : {0, 1, sequence - 1, sequence})
        if (!run(sequence, real, 2, 6, false, candidate_warps)) return 1;
    return run(857, 857, 16, 48, true, candidate_warps) ? 0 : 1;
}
