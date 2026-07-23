#include "kernels/gdn.h"
#include "kernels/rwkv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using Clock = std::chrono::steady_clock;

static Tensor external(Precision p, int64_t d0, int64_t d1, void* data) {
    return Tensor::create(p, MemoryType::EXTERNAL, d0, d1, 1, 1, data);
}

static void fill(std::vector<float>& x, float scale = 0.1f) {
    uint32_t s = 1;
    for (float& v : x) {
        s = s * 1664525u + 1013904223u;
        v = scale * ((float)(s >> 8) / 8388608.f - 1.f);
    }
}

template<class Fn, class Reset>
static double median_ms(int warmup, int repeats, Fn&& fn, Reset&& reset) {
    for (int i = 0; i < warmup; ++i) { reset(); fn(); }
    std::vector<double> samples;
    samples.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        reset();
        auto begin = Clock::now();
        fn();
        auto end = Clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

static double bench_rwkv(int heads, int d, int seq, ThreadPool& pool) {
    const int hidden = heads * d;
    const size_t state_n = (size_t)heads * d * d;
    std::vector<float> r((size_t)seq * hidden), w(r.size()), k(r.size()),
                       v(r.size()), a(r.size()), b(r.size()), state(state_n),
                       initial(state_n), out(r.size());
    fill(r); fill(k); fill(v); fill(a); fill(b); fill(initial, 0.01f);
    std::fill(w.begin(), w.end(), 0.995f);
    Tensor tr=external(Precision::FP32,hidden,seq,r.data());
    Tensor tw=external(Precision::FP32,hidden,seq,w.data());
    Tensor tk=external(Precision::FP32,hidden,seq,k.data());
    Tensor tv=external(Precision::FP32,hidden,seq,v.data());
    Tensor ta=external(Precision::FP32,hidden,seq,a.data());
    Tensor tb=external(Precision::FP32,hidden,seq,b.data());
    Tensor ts=external(Precision::FP32,(int64_t)state_n,1,state.data());
    Tensor to=external(Precision::FP32,hidden,seq,out.data());
    std::vector<const Tensor*> inputs={&tr,&tw,&tk,&tv,&ta,&tb,&ts};
    OpParams p; p.i32={heads,d,seq,seq};
    int repeats = seq == 1 ? 101 : 15;
    return median_ms(5,repeats,
        [&]{ kernel_rwkv7(p,inputs,to,&pool); },
        [&]{ std::memcpy(state.data(),initial.data(),state_n*sizeof(float)); });
}

static double bench_gdn(int heads, int d, int seq, ThreadPool& pool) {
    const int hidden = heads * d;
    const int qkv_dim = 3 * hidden;
    const size_t state_n = (size_t)heads * d * d;
    std::vector<float> qkv((size_t)seq*qkv_dim), a((size_t)seq*heads),
                       b(a.size()), z((size_t)seq*hidden), A(heads,-2.f),
                       dt(heads,0.f), norm(d,1.f), state(state_n),
                       initial(state_n), out(z.size());
    fill(qkv); fill(a); fill(b); fill(z); fill(initial,0.01f);
    Tensor tq=external(Precision::FP32,qkv_dim,seq,qkv.data());
    Tensor ta=external(Precision::FP32,heads,seq,a.data());
    Tensor tb=external(Precision::FP32,heads,seq,b.data());
    Tensor tz=external(Precision::FP32,hidden,seq,z.data());
    Tensor tA=external(Precision::FP32,heads,1,A.data());
    Tensor td=external(Precision::FP32,heads,1,dt.data());
    Tensor tn=external(Precision::FP32,d,1,norm.data());
    Tensor ts=external(Precision::FP32,(int64_t)state_n,1,state.data());
    Tensor to=external(Precision::FP32,hidden,seq,out.data());
    std::vector<const Tensor*> inputs={&tq,&ta,&tb,&tz,&tA,&td,&tn,&ts};
    std::vector<Tensor*> outputs={&to};
    OpParams p; p.i32={heads,d,d,seq,1,4,seq,heads};
    p.f32={1e-6f,1e-6f,1.f/std::sqrt((float)d)};
    int repeats = seq == 1 ? 101 : 15;
    return median_ms(5,repeats,
        [&]{
            if(seq==1) kernel_gdn_decode(p,inputs,outputs,&pool);
            else kernel_gdn_prefill(p,inputs,outputs,&pool);
        },
        [&]{ std::memcpy(state.data(),initial.data(),state_n*sizeof(float)); });
}

int main() {
    constexpr int heads=32, d=64, threads=4;
    ThreadPool pool(threads);
    std::printf("same-scale recurrent kernels: heads=%d d=%d state=%.2f MiB threads=%d\n",
                heads,d,(double)heads*d*d*4/(1024.*1024.),threads);
    std::printf("boundary: WKV=recurrence only; GDN=L2+gates+recurrence+RMSNormGated\n");
    for(int seq : {1,256}) {
        double w=bench_rwkv(heads,d,seq,pool);
        double g=bench_gdn(heads,d,seq,pool);
        std::printf("seq=%-3d WKV %.3f ms (%7.3f us/token)  GDN %.3f ms (%7.3f us/token)  WKV/GDN %.3fx\n",
                    seq,w,w*1000/seq,g,g*1000/seq,w/g);
    }
    return 0;
}
