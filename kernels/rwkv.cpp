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

void kernel_rwkv_mix(const OpParams&, const std::vector<const Tensor*>& in,
                     Tensor& out) {
    if (in.size() < 3) return;
    const float* x = in[0]->ptr<float>();
    const float* shift = in[1]->ptr<float>();
    const float* mix = in[2]->ptr<float>();
    float* dst = out.ptr<float>();
    const int hidden = (int)in[2]->nelements();
    const int64_t total = in[0]->nelements();
    if (hidden <= 0 || total % hidden != 0) return;
    const int rows = (int)(total / hidden);
    for (int row = 0; row < rows; ++row) {
        const size_t base = (size_t)row * hidden;
        int d = 0;
#if HAS_NEON
        for (; d + 7 < hidden; d += 8) {
            vst1q_f32(dst+base+d,
                      vfmaq_f32(vld1q_f32(x+base+d),
                                 vld1q_f32(shift+base+d),vld1q_f32(mix+d)));
            vst1q_f32(dst+base+d+4,
                      vfmaq_f32(vld1q_f32(x+base+d+4),
                                 vld1q_f32(shift+base+d+4),vld1q_f32(mix+d+4)));
        }
#endif
        for (; d < hidden; ++d)
            dst[base+d] = x[base+d] + shift[base+d] * mix[d];
    }
}

void kernel_rwkv_l2_norm(const OpParams& p,
                         const std::vector<const Tensor*>& in, Tensor& out) {
    if (in.empty()) return;
    const int heads=graph_params::get_i32(p,0,0);
    const int dhead=graph_params::get_i32(p,1,0);
    const float eps=graph_params::get_f32(p,0,1e-12f);
    const int hidden=heads*dhead;
    const int tokens=(int)(in[0]->nelements()/hidden);
    const float* src=in[0]->ptr<float>();
    float* dst=out.ptr<float>();
    for(int t=0;t<tokens;++t) for(int h=0;h<heads;++h) {
        const size_t base=(size_t)t*hidden+(size_t)h*dhead;
        float sum=0.f;
#if HAS_NEON
        if((dhead&3)==0) {
            float32x4_t s=vdupq_n_f32(0.f);
            for(int j=0;j<dhead;j+=4) {
                float32x4_t x=vld1q_f32(src+base+j);
                s=vfmaq_f32(s,x,x);
            }
            sum=vaddvq_f32(s);
        } else
#endif
        for(int j=0;j<dhead;++j) sum+=src[base+j]*src[base+j];
        const float inv=1.f/(std::sqrt(sum)+eps);
        int j=0;
#if HAS_NEON
        for(;j+3<dhead;j+=4)
            vst1q_f32(dst+base+j,vmulq_n_f32(vld1q_f32(src+base+j),inv));
#endif
        for(;j<dhead;++j) dst[base+j]=src[base+j]*inv;
    }
}

void kernel_rwkv_group_norm(const OpParams& p,
                            const std::vector<const Tensor*>& in, Tensor& out) {
    if(in.size()<3) return;
    const int heads=graph_params::get_i32(p,0,0);
    const int dhead=graph_params::get_i32(p,1,0);
    const float eps=graph_params::get_f32(p,0,64e-5f);
    const int hidden=heads*dhead;
    const int tokens=(int)(in[0]->nelements()/hidden);
    const float *src=in[0]->ptr<float>(),*weight=in[1]->ptr<float>(),
                *bias=in[2]->ptr<float>();
    float* dst=out.ptr<float>();
    for(int t=0;t<tokens;++t) for(int h=0;h<heads;++h) {
        const size_t base=(size_t)t*hidden+(size_t)h*dhead;
        float mean=0.f,var=0.f;
#if HAS_NEON
        if((dhead&3)==0) {
            float32x4_t sum=vdupq_n_f32(0.f);
            for(int j=0;j<dhead;j+=4) sum=vaddq_f32(sum,vld1q_f32(src+base+j));
            mean=vaddvq_f32(sum)/dhead;
            sum=vdupq_n_f32(0.f);
            const float32x4_t mv=vdupq_n_f32(mean);
            for(int j=0;j<dhead;j+=4) {
                float32x4_t q=vsubq_f32(vld1q_f32(src+base+j),mv);
                sum=vfmaq_f32(sum,q,q);
            }
            var=vaddvq_f32(sum)/dhead;
        } else
#endif
        {
            for(int j=0;j<dhead;++j) mean+=src[base+j]; mean/=dhead;
            for(int j=0;j<dhead;++j){float q=src[base+j]-mean;var+=q*q;} var/=dhead;
        }
        const float inv=1.f/std::sqrt(var+eps);
        int j=0;
#if HAS_NEON
        const float32x4_t mv=vdupq_n_f32(mean),iv=vdupq_n_f32(inv);
        for(;j+3<dhead;j+=4) {
            float32x4_t q=vmulq_f32(vsubq_f32(vld1q_f32(src+base+j),mv),iv);
            q=vfmaq_f32(vld1q_f32(bias+(size_t)h*dhead+j),q,
                        vld1q_f32(weight+(size_t)h*dhead+j));
            vst1q_f32(dst+base+j,q);
        }
#endif
        for(;j<dhead;++j)
            dst[base+j]=(src[base+j]-mean)*inv*weight[h*dhead+j]+bias[h*dhead+j];
    }
}

void kernel_rwkv_bonus(const OpParams& p,
                       const std::vector<const Tensor*>& in, Tensor& out) {
    if(in.size()<4) return;
    const int heads=graph_params::get_i32(p,0,0);
    const int dhead=graph_params::get_i32(p,1,0);
    const int hidden=heads*dhead;
    const int tokens=(int)(in[0]->nelements()/hidden);
    const float *r=in[0]->ptr<float>(),*k=in[1]->ptr<float>(),
                *v=in[2]->ptr<float>(),*rk=in[3]->ptr<float>();
    float* dst=out.ptr<float>();
    for(int t=0;t<tokens;++t) for(int h=0;h<heads;++h) {
        const size_t base=(size_t)t*hidden+(size_t)h*dhead;
        float bonus=0.f;
#if HAS_NEON
        if((dhead&3)==0) {
            float32x4_t sum=vdupq_n_f32(0.f);
            for(int j=0;j<dhead;j+=4) {
                float32x4_t q=vmulq_f32(vld1q_f32(r+base+j),vld1q_f32(k+base+j));
                sum=vfmaq_f32(sum,q,vld1q_f32(rk+(size_t)h*dhead+j));
            }
            bonus=vaddvq_f32(sum);
        } else
#endif
        for(int j=0;j<dhead;++j) bonus+=r[base+j]*k[base+j]*rk[h*dhead+j];
        int j=0;
#if HAS_NEON
        for(;j+3<dhead;j+=4)
            vst1q_f32(dst+base+j,vmulq_n_f32(vld1q_f32(v+base+j),bonus));
#endif
        for(;j<dhead;++j) dst[base+j]=v[base+j]*bonus;
    }
}

static void kernel_rwkv7_core(const OpParams& p,
                              const std::vector<const Tensor*>& in, Tensor& out,
                              ThreadPool* thread_pool) {
    const int heads=graph_params::get_i32(p,0,0);
    const int dhead=graph_params::get_i32(p,1,0);
    const int seq=graph_params::get_i32(p,2,1);
    int real=graph_params::get_i32(p,3,seq);
    if(real<=0||real>seq) real=seq;
    const int hidden=heads*dhead;
    const float *r=in[0]->ptr<float>(),*w=in[1]->ptr<float>(),
                *k=in[2]->ptr<float>(),*v=in[3]->ptr<float>(),
                *a=in[4]->ptr<float>(),*b=in[5]->ptr<float>();
    const bool state_fp16=in[6]->prec==Precision::FP16;
    __fp16* state16=state_fp16?reinterpret_cast<__fp16*>(in[6]->data):nullptr;
    float* state32=state_fp16?nullptr:reinterpret_cast<float*>(in[6]->data);
    float* dst=out.ptr<float>();
    auto process_heads=[&](int,int h_begin,int h_end) {
        std::vector<float> statef((size_t)dhead*dhead);
        for(int h=h_begin;h<h_end;++h) {
            const size_t sb=(size_t)h*dhead*dhead;
            const bool direct_state=!state_fp16&&real==1;
            float* s=direct_state?state32+sb:statef.data();
            if(!state_fp16&&!direct_state)
                std::memcpy(s,state32+sb,(size_t)dhead*dhead*sizeof(float));
            for(int t=0;t<real;++t) {
                const size_t base=(size_t)t*hidden+(size_t)h*dhead;
                if(state_fp16) {
#if HAS_NEON
                    if((dhead&3)==0) rwkv_load_state_fp16(state16+sb,s,dhead*dhead);
                    else
#endif
                    for(int q=0;q<dhead*dhead;++q) s[q]=(float)state16[sb+q];
                }
                int row=0;
#if HAS_NEON
            // Four-row register block: a/w/k/b/r are shared by the rows, so
            // load each vector once instead of four times.
            if((dhead&3)==0) for(;row+3<dhead;row+=4) {
                float32x4_t sa0=vdupq_n_f32(0.f),sa1=sa0,sa2=sa0,sa3=sa0;
                for(int j=0;j<dhead;j+=4) {
                    const float32x4_t av=vld1q_f32(a+base+j);
                    sa0=vfmaq_f32(sa0,vld1q_f32(s+(size_t)(row+0)*dhead+j),av);
                    sa1=vfmaq_f32(sa1,vld1q_f32(s+(size_t)(row+1)*dhead+j),av);
                    sa2=vfmaq_f32(sa2,vld1q_f32(s+(size_t)(row+2)*dhead+j),av);
                    sa3=vfmaq_f32(sa3,vld1q_f32(s+(size_t)(row+3)*dhead+j),av);
                }
                const float sas[4]={vaddvq_f32(sa0),vaddvq_f32(sa1),
                                    vaddvq_f32(sa2),vaddvq_f32(sa3)};
                float32x4_t re0=vdupq_n_f32(0.f),re1=re0,re2=re0,re3=re0;
                const float32x4_t vv0=vdupq_n_f32(v[base+row+0]);
                const float32x4_t vv1=vdupq_n_f32(v[base+row+1]);
                const float32x4_t vv2=vdupq_n_f32(v[base+row+2]);
                const float32x4_t vv3=vdupq_n_f32(v[base+row+3]);
                const float32x4_t sv0=vdupq_n_f32(sas[0]);
                const float32x4_t sv1=vdupq_n_f32(sas[1]);
                const float32x4_t sv2=vdupq_n_f32(sas[2]);
                const float32x4_t sv3=vdupq_n_f32(sas[3]);
                for(int j=0;j<dhead;j+=4) {
                    const float32x4_t wv=vld1q_f32(w+base+j);
                    const float32x4_t kv=vld1q_f32(k+base+j);
                    const float32x4_t bv=vld1q_f32(b+base+j);
                    const float32x4_t rv=vld1q_f32(r+base+j);
#define RWKV_ROW_STEP(N) \
                    float32x4_t x##N=vmulq_f32(vld1q_f32(s+(size_t)(row+N)*dhead+j),wv); \
                    x##N=vfmaq_f32(x##N,vv##N,kv); \
                    x##N=vfmaq_f32(x##N,sv##N,bv); \
                    vst1q_f32(s+(size_t)(row+N)*dhead+j,x##N); \
                    re##N=vfmaq_f32(re##N,x##N,rv)
                    RWKV_ROW_STEP(0); RWKV_ROW_STEP(1);
                    RWKV_ROW_STEP(2); RWKV_ROW_STEP(3);
#undef RWKV_ROW_STEP
                }
                dst[base+row+0]=vaddvq_f32(re0);
                dst[base+row+1]=vaddvq_f32(re1);
                dst[base+row+2]=vaddvq_f32(re2);
                dst[base+row+3]=vaddvq_f32(re3);
            }
#endif
            for(int i=row;i<dhead;++i) {
                float sa=0.f,result=0.f;
#if HAS_NEON
                if((dhead&3)==0) sa=rwkv_dot_neon(s+(size_t)i*dhead,a+base,dhead);
                else
#endif
                for(int j=0;j<dhead;++j) sa+=s[(size_t)i*dhead+j]*a[base+j];
                int j=0;
#if HAS_NEON
                float32x4_t result4=vdupq_n_f32(0.f);
                const float32x4_t vv=vdupq_n_f32(v[base+i]),sav=vdupq_n_f32(sa);
                for(;j+3<dhead;j+=4) {
                    float32x4_t sv=vmulq_f32(vld1q_f32(s+(size_t)i*dhead+j),
                                             vld1q_f32(w+base+j));
                    sv=vfmaq_f32(sv,vv,vld1q_f32(k+base+j));
                    sv=vfmaq_f32(sv,sav,vld1q_f32(b+base+j));
                    vst1q_f32(s+(size_t)i*dhead+j,sv);
                    result4=vfmaq_f32(result4,sv,vld1q_f32(r+base+j));
                }
                result=vaddvq_f32(result4);
#endif
                for(;j<dhead;++j) {
                    float& sv=s[(size_t)i*dhead+j];
                    sv=sv*w[base+j]+v[base+i]*k[base+j]+sa*b[base+j];
                    result+=sv*r[base+j];
                }
                dst[base+i]=result;
            }
                if(state_fp16) {
#if HAS_NEON
                    if((dhead&3)==0) rwkv_store_state_fp16(s,state16+sb,dhead*dhead);
                    else
#endif
                    for(int q=0;q<dhead*dhead;++q) state16[sb+q]=(__fp16)s[q];
                }
            }
            if(!state_fp16&&!direct_state)
                std::memcpy(state32+sb,s,(size_t)dhead*dhead*sizeof(float));
        }
    };
    if(thread_pool&&heads>=4) thread_pool->parallel_for(0,heads,1,process_heads);
    else process_heads(0,0,heads);
    if(real<seq) std::memset(dst+(size_t)real*hidden,0,(size_t)(seq-real)*hidden*sizeof(float));
}

void kernel_rwkv7(const OpParams& p, const std::vector<const Tensor*>& in,
                  Tensor& out, ThreadPool* thread_pool) {
    if(in.size()==7) {
        kernel_rwkv7_core(p,in,out,thread_pool);
        return;
    }
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
