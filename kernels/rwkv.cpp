#include "kernels/rwkv.h"
#include "kernels/threading.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if HAS_NEON
#include <arm_neon.h>

static inline float rwkv_dot_neon(const float* a, const float* b, int n) {
    float32x4_t sum = vdupq_n_f32(0.f);
    for (int i = 0; i < n; i += 4)
        sum = vfmaq_f32(sum, vld1q_f32(a + i), vld1q_f32(b + i));
    return vaddvq_f32(sum);
}

static inline void rwkv_load_state_fp16(const __fp16* src, float* dst, int n) {
    for (int i = 0; i < n; i += 4)
        vst1q_f32(dst + i, vcvt_f32_f16(vld1_f16(src + i)));
}

static inline void rwkv_store_state_fp16(const float* src, __fp16* dst, int n) {
    for (int i = 0; i < n; i += 4)
        vst1_f16(dst + i, vcvt_f16_f32(vld1q_f32(src + i)));
}
#endif

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
                  Tensor& out, ThreadPool* thread_pool) {
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
#if HAS_NEON
    const bool use_neon = (dhead & 3) == 0;
#endif
    auto process_heads = [&](int, int h_begin, int h_end) {
      std::vector<float> raw(dhead), sa(dhead), kval(dhead), vval(dhead),
          kk(dhead), aval(dhead), decay(dhead), gate(dhead),
          statef((size_t)dhead*dhead);
      for (int h=h_begin; h<h_end; ++h) for (int t=0; t<real; ++t) {
        const size_t base=(size_t)t*hidden+(size_t)h*dhead;
        const size_t state_base=(size_t)h*dhead*dhead;
        float* s=statef.data();
        if (state_fp16) {
#if HAS_NEON
            if (use_neon) rwkv_load_state_fp16(state16+state_base, s, dhead*dhead);
            else
#endif
            for (int i=0; i<dhead*dhead; ++i) s[i]=(float)state16[state_base+i];
        } else {
            std::memcpy(s, state32+state_base, (size_t)dhead*dhead*sizeof(float));
        }
        float knorm=0.f;
#if HAS_NEON
        if (use_neon) {
            float32x4_t sum = vdupq_n_f32(0.f);
            for (int j=0; j<dhead; j+=4) {
                float32x4_t kv = vmulq_f32(vld1q_f32(k0+base+j),
                                            vld1q_f32(kkw+(size_t)h*dhead+j));
                vst1q_f32(kk.data()+j, kv);
                sum = vfmaq_f32(sum, kv, kv);
            }
            knorm = vaddvq_f32(sum);
        } else
#endif
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
#if HAS_NEON
            if (use_neon) {
                sa[i] = -rwkv_dot_neon(s+(size_t)i*dhead, kk.data(), dhead);
                continue;
            }
#endif
            float q=0.f;
            for (int j=0;j<dhead;j++) q += s[(size_t)i*dhead+j] * (-kk[j]);
            sa[i]=q;
        }
        for (int i=0;i<dhead;i++) {
#if HAS_NEON
            if (use_neon) {
                const float32x4_t vv = vdupq_n_f32(vval[i]);
                const float32x4_t sav = vdupq_n_f32(sa[i]);
                float* row = s+(size_t)i*dhead;
                for (int j=0;j<dhead;j+=4) {
                    float32x4_t sv = vmulq_f32(vld1q_f32(decay.data()+j),
                                               vld1q_f32(row+j));
                    sv = vfmaq_f32(sv, vv, vld1q_f32(kval.data()+j));
                    float32x4_t kaa = vmulq_f32(vld1q_f32(kk.data()+j),
                                                vld1q_f32(aval.data()+j));
                    sv = vfmaq_f32(sv, sav, kaa);
                    vst1q_f32(row+j, sv);
                }
                continue;
            }
#endif
            for (int j=0;j<dhead;j++)
                s[(size_t)i*dhead+j] = decay[j]*s[(size_t)i*dhead+j]
                    + vval[i]*kval[j] + sa[i]*(kk[j]*aval[j]);
        }
        for (int i=0;i<dhead;i++) {
#if HAS_NEON
            if (use_neon) {
                raw[i] = rwkv_dot_neon(s+(size_t)i*dhead, r+base, dhead);
                continue;
            }
#endif
            float z=0.f;
            for (int j=0;j<dhead;j++) z += s[(size_t)i*dhead+j]*r[base+j];
            raw[i]=z;
        }
        float mean=0.f, var=0.f, bonus=0.f;
#if HAS_NEON
        if (use_neon) {
            float32x4_t sum = vdupq_n_f32(0.f);
            for(int i=0;i<dhead;i+=4) sum=vaddq_f32(sum,vld1q_f32(raw.data()+i));
            mean=vaddvq_f32(sum)/dhead;
            const float32x4_t mv=vdupq_n_f32(mean);
            sum=vdupq_n_f32(0.f);
            for(int i=0;i<dhead;i+=4) {
                float32x4_t q=vsubq_f32(vld1q_f32(raw.data()+i),mv);
                sum=vfmaq_f32(sum,q,q);
            }
            var=vaddvq_f32(sum)/dhead;
            sum=vdupq_n_f32(0.f);
            for(int j=0;j<dhead;j+=4) {
                float32x4_t q=vmulq_f32(vld1q_f32(r+base+j),
                                        vld1q_f32(kval.data()+j));
                sum=vfmaq_f32(sum,q,vld1q_f32(rk+(size_t)h*dhead+j));
            }
            bonus=vaddvq_f32(sum);
        } else
#endif
        {
            for(float z:raw) mean+=z; mean/=dhead;
            for(float z:raw){float q=z-mean;var+=q*q;} var/=dhead;
            for(int j=0;j<dhead;j++) bonus += r[base+j]*kval[j]*rk[h*dhead+j];
        }
        float inv=1.f/std::sqrt(var+eps);
#if HAS_NEON
        if (use_neon) {
            const float32x4_t mv=vdupq_n_f32(mean);
            const float32x4_t iv=vdupq_n_f32(inv);
            const float32x4_t bv=vdupq_n_f32(bonus);
            for(int i=0;i<dhead;i+=4) {
                float32x4_t q=vmulq_f32(vsubq_f32(vld1q_f32(raw.data()+i),mv),iv);
                q=vfmaq_f32(vld1q_f32(nb+(size_t)h*dhead+i),q,
                            vld1q_f32(nw+(size_t)h*dhead+i));
                q=vfmaq_f32(q,bv,vld1q_f32(vval.data()+i));
                vst1q_f32(dst+base+i,vmulq_f32(q,vld1q_f32(gate.data()+i)));
            }
        } else
#endif
        for(int i=0;i<dhead;i++)
            dst[base+i]=(((raw[i]-mean)*inv*nw[h*dhead+i]+nb[h*dhead+i])
                         +bonus*vval[i])*gate[i];
        if (state_fp16) {
#if HAS_NEON
            if (use_neon) rwkv_store_state_fp16(s, state16+state_base, dhead*dhead);
            else
#endif
            for (int i=0; i<dhead*dhead; ++i) state16[state_base+i]=(__fp16)s[i];
        } else {
            std::memcpy(state32+state_base, s, (size_t)dhead*dhead*sizeof(float));
        }
      }
    };
    if (thread_pool && heads >= 4) {
        thread_pool->parallel_for(0, heads, 1, process_heads);
    } else {
        process_heads(0, 0, heads);
    }
    if(real<seq) std::memset(dst+(size_t)real*hidden,0,
                             (size_t)(seq-real)*hidden*sizeof(float));
}
