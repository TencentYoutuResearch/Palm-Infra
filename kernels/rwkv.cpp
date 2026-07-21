#include "kernels/rwkv.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

void kernel_rwkv_token_shift(const OpParams& p,
                             const std::vector<const Tensor*>& in, Tensor& out) {
    if (in.size() < 2) return;
    const int hidden = graph_params::get_i32(p, 0, 0);
    const int seq = graph_params::get_i32(p, 1, 1);
    int real = graph_params::get_i32(p, 2, seq);
    if (real <= 0 || real > seq) real = seq;
    const float* x = in[0]->ptr<float>();
    const bool state_fp16 = in[1]->prec == Precision::FP16;
    __fp16* state16 = state_fp16 ? reinterpret_cast<__fp16*>(in[1]->data) : nullptr;
    float* state32 = state_fp16 ? nullptr : reinterpret_cast<float*>(in[1]->data);
    float* y = out.ptr<float>();
    for (int t = 0; t < real; ++t) {
        const float* row = x + (size_t)t * hidden;
        for (int d = 0; d < hidden; ++d)
            y[(size_t)t * hidden + d] = (state_fp16 ? (float)state16[d] : state32[d]) - row[d];
        if (state_fp16) {
            for (int d = 0; d < hidden; ++d) state16[d] = (__fp16)row[d];
        } else {
            std::memcpy(state32, row, (size_t)hidden * sizeof(float));
        }
    }
    if (real < seq)
        std::memset(y + (size_t)real * hidden, 0,
                    (size_t)(seq - real) * hidden * sizeof(float));
}

void kernel_rwkv7(const OpParams& p, const std::vector<const Tensor*>& in,
                  Tensor& out, ThreadPool*) {
    if (in.size() < 17) return;
    const int heads = graph_params::get_i32(p, 0, 0);
    const int dhead = graph_params::get_i32(p, 1, 0);
    const int seq = graph_params::get_i32(p, 2, 1);
    const bool first_layer = graph_params::get_i32(p, 4, 0) != 0;
    int real = graph_params::get_i32(p, 3, seq);
    if (real <= 0 || real > seq) real = seq;
    const int hidden = heads * dhead;
    const float eps = graph_params::get_f32(p, 0, 64e-5f);
    const float *r=in[0]->ptr<float>(), *wd=in[1]->ptr<float>();
    const float *k0=in[2]->ptr<float>(), *v0p=in[3]->ptr<float>();
    const float *ad=in[4]->ptr<float>(), *gd=in[5]->ptr<float>();
    const float *vd=in[6]->ptr<float>(), *vf=in[7]->ptr<float>();
    const float *w0=in[8]->ptr<float>(), *a0=in[9]->ptr<float>();
    const float *v0=in[10]->ptr<float>(), *kkw=in[11]->ptr<float>();
    const float *ka=in[12]->ptr<float>(), *rk=in[13]->ptr<float>();
    const float *nw=in[14]->ptr<float>(), *nb=in[15]->ptr<float>();
    const bool state_fp16 = in[16]->prec == Precision::FP16;
    __fp16* state16 = state_fp16 ? reinterpret_cast<__fp16*>(in[16]->data) : nullptr;
    float* state32 = state_fp16 ? nullptr : reinterpret_cast<float*>(in[16]->data);
    float* dst = out.ptr<float>();
    std::vector<float> raw(dhead), sa(dhead), kval(dhead), vval(dhead), kk(dhead), aval(dhead), decay(dhead), gate(dhead), statef((size_t)dhead*dhead);
    for (int t=0; t<real; ++t) for (int h=0; h<heads; ++h) {
        const size_t base=(size_t)t*hidden+h*dhead;
        const size_t state_base=(size_t)h*dhead*dhead;
        float* s=statef.data();
        if (state_fp16) {
            for (int i=0; i<dhead*dhead; ++i) s[i]=(float)state16[state_base+i];
        } else {
            std::memcpy(s, state32+state_base, (size_t)dhead*dhead*sizeof(float));
        }
        float knorm=0.f;
        for(int j=0;j<dhead;j++){ kk[j]=k0[base+j]*kkw[h*dhead+j]; knorm+=kk[j]*kk[j]; }
        knorm=1.f/(std::sqrt(knorm)+1e-6f);
        for(int j=0;j<dhead;j++){
            kk[j]*=knorm;
            aval[j]=1.f/(1.f+std::exp(-(a0[h*dhead+j]+ad[base+j])));
            decay[j]=std::exp(-0.606531f/(1.f+std::exp(-(w0[h*dhead+j]+wd[base+j]))));
            // gate = g2(sigmoid(g1(xg))) is fully formed by the graph.
            gate[j]=gd[base+j];
            kval[j]=k0[base+j]*(1.f+(aval[j]-1.f)*ka[h*dhead+j]);
            if (first_layer) vval[j]=v0p[base+j];
            else {
                float vmix=1.f/(1.f+std::exp(-(v0[h*dhead+j]+vd[base+j])));
                vval[j]=v0p[base+j]+(vf[base+j]-v0p[base+j])*vmix;
            }
        }
        for (int i=0;i<dhead;i++) {
            float q=0.f;
            for (int j=0;j<dhead;j++) q += s[(size_t)i*dhead+j] * (-kk[j]);
            sa[i]=q;
        }
        for (int i=0;i<dhead;i++) for (int j=0;j<dhead;j++)
            s[(size_t)i*dhead+j] = decay[j]*s[(size_t)i*dhead+j]
                + vval[i]*kval[j] + sa[i]*(kk[j]*aval[j]);
        for (int i=0;i<dhead;i++) {
            float z=0.f;
            for (int j=0;j<dhead;j++) z += s[(size_t)i*dhead+j]*r[base+j];
            raw[i]=z;
        }
        float mean=0.f; for(float z:raw) mean+=z; mean/=dhead;
        float var=0.f; for(float z:raw){float q=z-mean;var+=q*q;} var/=dhead;
        float inv=1.f/std::sqrt(var+eps);
        float bonus=0.f;
        for(int j=0;j<dhead;j++) bonus += r[base+j]*kval[j]*rk[h*dhead+j];
        for(int i=0;i<dhead;i++)
            dst[base+i]=(((raw[i]-mean)*inv*nw[h*dhead+i]+nb[h*dhead+i])
                         +bonus*vval[i])*gate[i];
        if (state_fp16) {
            for (int i=0; i<dhead*dhead; ++i) state16[state_base+i]=(__fp16)s[i];
        } else {
            std::memcpy(state32+state_base, s, (size_t)dhead*dhead*sizeof(float));
        }
    }
    if(real<seq) std::memset(dst+(size_t)real*hidden,0,
                             (size_t)(seq-real)*hidden*sizeof(float));
}
