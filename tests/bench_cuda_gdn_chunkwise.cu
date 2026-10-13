// Standalone block-algebra prototypes. No backend or production selector.
#include "kernels/cuda/recurrent.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <utility>
#include <vector>

namespace {
constexpr int C=32,D=128,U=0,W=C*D,A=2*C*D,P=A+C*C,G=P+C,S=G+C;
bool tensor_core=false;
int tensor_columns=16;
bool tensor_padding=false;
bool tensor_prepare=false;
bool tensor_coeff=false;
bool column_solve=false;
bool coalesced_values=false;
bool dual_projection=false;
bool shared_grams=false;
bool staged_coefficients=false;
bool tensor_coeff_combo=false;
bool shared_update=false;
bool half_products=false;
int apply_threads=128;
int timing_replays=16;
constexpr int HalfSharedBytes=(D*32+2*C*32)*sizeof(float)+4*C*D*sizeof(__half);
__device__ void gemm_tf32_residual(const float*,int,const float*,int,float*,int,int,int,int,int=0,int=0);
// Q/K heads are shared by several value heads. These two unweighted Gram
// matrices are independent of value-head decay/beta and may be computed once.
__global__ void prepare_grams(const float* query,const float* key,float* grams,
    int seq,int real,int chunks) {
    const int kh=blockIdx.y,first=blockIdx.x*C,n=min(C,real-first),tid=threadIdx.x;
    __shared__ __align__(32) float k[C][D],q[C][D],kt[D][C];
    for(int i=tid;i<C*D;i+=blockDim.x) {
        const int t=i/D,r=i%D;
        k[t][r]=t<n?key[(size_t(kh)*seq+first+t)*D+r]:0;
        q[t][r]=t<n?query[(size_t(kh)*seq+first+t)*D+r]:0;
    }
    __syncthreads();
    for(int i=tid;i<D*C;i+=blockDim.x)kt[i/C][i%C]=k[i%C][i/C];
    __syncthreads();
    float* dst=grams+(size_t(kh)*chunks+blockIdx.x)*2*C*C;
    if(tid<128)gemm_tf32_residual(&k[0][0],D,&kt[0][0],C,dst,C,C,C,D,0,4);
    else gemm_tf32_residual(&q[0][0],D,&kt[0][0],C,dst+C*C,C,C,C,D,4,4);
}
template<bool Tensor=false,bool Coeff=false,bool ColumnSolve=false,bool CoalescedValues=false,bool SharedGrams=false>
__global__ void prepare(const float* qkv,const float* query,const float* key,
    const float* decay,const float* beta,float* work,int heads,int values,int seq,int real,int chunks,const float* grams=nullptr) {
    static_assert(!SharedGrams||(Tensor&&!ColumnSolve));
    const int vh=blockIdx.y,kh=vh/(values/heads),first=blockIdx.x*C;
    const int n=min(C,real-first),tid=threadIdx.x;
    constexpr int Vpitch=D+(CoalescedValues?4:0); // WMMA-compatible float leading dimension
    __shared__ __align__(32) float k[C][D],v[C][Vpitch],lower[C][C],inverse[C][C],gamma[C][C],d[C],b[C],p[C];
    float* dst=work+(size_t(vh)*chunks+blockIdx.x)*S;
    for(int i=tid;i<C*D;i+=blockDim.x) {
        const int t=i/D,j=i%D;
        k[t][j]=t<n?key[(size_t(kh)*seq+first+t)*D+j]:0;
        if constexpr(!Tensor)v[t][j]=t<n?qkv[(size_t(2*heads*D)+vh*D+j)*seq+first+t]:0;
    }
    if(tid<C) {
        d[tid]=tid<n?decay[(first+tid)*values+vh]:1;
        b[tid]=tid<n?beta[(first+tid)*values+vh]:0;
    }
    __syncthreads();
    if constexpr(Tensor&&!SharedGrams) {
        for(int i=tid;i<C*D;i+=blockDim.x) {
            const int r=i/C,t=i%C;
            reinterpret_cast<float*>(v)[i]=k[t][r];
        }
        __syncthreads();
        gemm_tf32_residual(&k[0][0],D,&v[0][0],C,&lower[0][0],C,C,C,D);
        __syncthreads();
    }
    if(tid<C) {
        float product=1;
        for(int j=0;j<=tid;++j)product*=d[j];
        p[tid]=product;dst[P+tid]=product;
    }
    for(int i=tid;i<C*C;i+=blockDim.x) {
        const int t=i/C,j=i%C;
        float product=j<=t?1.f:0.f;
        for(int s=j+1;s<=t;++s)product*=d[s];
        gamma[t][j]=product;
        float dot=0;
        if constexpr(SharedGrams)dot=grams[(size_t(kh)*chunks+blockIdx.x)*2*C*C+i];
        else if constexpr(Tensor)dot=lower[t][j];
        else if(t>j)for(int r=0;r<D;++r)dot=fmaf(k[t][r],k[j][r],dot);
        lower[t][j]=t>j?(b[t]*product)*dot:0;
        inverse[t][j]=t==j?1:0;
    }
    __syncthreads();
    for(int t=1;t<C;++t) {
        if(tid<t) {
            float sum=0;
            for(int j=0;j<t;++j)sum=fmaf(lower[t][j],inverse[j][tid],sum);
            inverse[t][tid]=-sum;
        }
        if constexpr(!ColumnSolve)__syncthreads();
    }
    if constexpr(ColumnSolve)__syncthreads(); // columns independent until consumers below
    if constexpr(Tensor&&!Coeff) {
        for(int i=tid;i<C*D;i+=blockDim.x) {
            const int t=CoalescedValues?i%C:i/D,r=CoalescedValues?i/C:i%D;
            v[t][r]=t<n?qkv[(size_t(2*heads*D)+vh*D+r)*seq+first+t]:0;
        }
        __syncthreads();
    }
    if constexpr(Coeff) {
        for(int i=tid;i<C*D;i+=blockDim.x) {
            const int t=CoalescedValues?i%C:i/D,r=CoalescedValues?i/C:i%D;
            v[t][r]=t<n?b[t]*qkv[(size_t(2*heads*D)+vh*D+r)*seq+first+t]:0;
        }
        __syncthreads();
        gemm_tf32_residual(&inverse[0][0],C,&v[0][0],Vpitch,dst+U,D,C,D,C);
        __syncthreads();
        for(int i=tid;i<C*D;i+=blockDim.x) {
            const int t=i/D,r=i%D;
            v[t][r]=(b[t]*p[t])*k[t][r];
        }
        __syncthreads();
        gemm_tf32_residual(&inverse[0][0],C,&v[0][0],Vpitch,dst+W,D,C,D,C);
    } else for(int i=tid;i<C*D;i+=blockDim.x) {
        const int t=i/D,r=i%D;
        float u=0,w=0;
        for(int j=0;j<=t;++j) {
            u=fmaf(inverse[t][j],b[j]*v[j][r],u);
            w=fmaf(inverse[t][j],(b[j]*p[j])*k[j][r],w);
        }
        dst[U+i]=u;dst[W+i]=w;
    }
    if constexpr(Tensor&&!SharedGrams) {
        __syncthreads();
        float transpose[16];
        for(int j=0;j<16;++j) {
            const int i=tid+j*256,r=i/C,t=i%C;
            transpose[j]=k[t][r];
        }
        __syncthreads();
        for(int j=0;j<16;++j)reinterpret_cast<float*>(k)[tid+j*256]=transpose[j];
        for(int i=tid;i<C*D;i+=blockDim.x) {
            const int t=i/D,r=i%D;
            v[t][r]=t<n?query[(size_t(kh)*seq+first+t)*D+r]:0;
        }
        __syncthreads();
        gemm_tf32_residual(&v[0][0],Vpitch,&k[0][0],C,&lower[0][0],C,C,C,D);
        __syncthreads();
    }
    for(int i=tid;i<C*C;i+=blockDim.x) {
        const int t=i/C,j=i%C;
        float dot=0;
        if constexpr(SharedGrams)dot=j<=t&&t<n?grams[(size_t(kh)*chunks+blockIdx.x)*2*C*C+C*C+i]:0;
        else if constexpr(Tensor)dot=j<=t&&t<n?lower[t][j]:0;
        else if(j<=t&&t<n)for(int r=0;r<D;++r)
            dot=fmaf(query[(size_t(kh)*seq+first+t)*D+r],k[j][r],dot);
        dst[A+i]=dot*gamma[t][j];
    }
    if(tid<C)dst[G+tid]=gamma[C-1][tid];
}

void launch_prepare(const float* qkv,const float* q,const float* k,const float* d,const float* b,
    float* work,int heads,int values,int seq,int real,int chunks,float* grams=nullptr) {
    if(!chunks)return;
    if(shared_grams) {
        prepare_grams<<<dim3(chunks,heads),256>>>(q,k,grams,seq,real,chunks);
        if(tensor_coeff_combo)prepare<true,true,false,true,true><<<dim3(chunks,values),256>>>(qkv,q,k,d,b,work,heads,values,seq,real,chunks,grams);
        else if(coalesced_values)prepare<true,false,false,true,true><<<dim3(chunks,values),256>>>(qkv,q,k,d,b,work,heads,values,seq,real,chunks,grams);
        else prepare<true,false,false,false,true><<<dim3(chunks,values),256>>>(qkv,q,k,d,b,work,heads,values,seq,real,chunks,grams);
    }
    else if(coalesced_values)prepare<true,false,false,true><<<dim3(chunks,values),256>>>(qkv,q,k,d,b,work,heads,values,seq,real,chunks);
    else if(column_solve)prepare<true,false,true><<<dim3(chunks,values),256>>>(qkv,q,k,d,b,work,heads,values,seq,real,chunks);
    else if(tensor_coeff)prepare<true,true><<<dim3(chunks,values),256>>>(qkv,q,k,d,b,work,heads,values,seq,real,chunks);
    else if(tensor_prepare)prepare<true><<<dim3(chunks,values),256>>>(qkv,q,k,d,b,work,heads,values,seq,real,chunks);
    else prepare<false><<<dim3(chunks,values),256>>>(qkv,q,k,d,b,work,heads,values,seq,real,chunks);
}

__global__ void apply(const float* query,const float* key,const float* work,
    float* state,float* output,int heads,int values,int seq,int real,int chunks,float scale) {
    const int lane=threadIdx.x&31,warp=threadIdx.x/32;
    const int value=(blockIdx.x*2+warp)*4,vh=blockIdx.y,kh=vh/(values/heads);
    float local[4][4];
    float* base=state+size_t(vh)*D*D;
    for(int r=0;r<4;++r) {
        const float4 x=*reinterpret_cast<const float4*>(base+size_t(lane+r*32)*D+value);
        local[0][r]=x.x;local[1][r]=x.y;local[2][r]=x.z;local[3][r]=x.w;
    }
    for(int block=0;block<chunks;++block) {
        const int first=block*C,n=min(C,real-first);
        const float* src=work+(size_t(vh)*chunks+block)*S;
        float h[4]={};
        for(int t=0;t<C;++t) {
            float dot[4]={};
            for(int r=0;r<4;++r)for(int col=0;col<4;++col)
                dot[col]=fmaf(local[col][r],src[W+t*D+lane+r*32],dot[col]);
            for(int offset=16;offset;offset/=2)for(int col=0;col<4;++col)
                dot[col]+=__shfl_down_sync(0xffffffffu,dot[col],offset);
            for(int col=0;col<4;++col) {
                const float total=__shfl_sync(0xffffffffu,dot[col],0);
                if(lane==t)h[col]=src[U+t*D+value+col]-total;
            }
        }
        for(int t=0;t<n;++t) {
            float dot[4]={},attention[4];
            for(int r=0;r<4;++r) {
                const float q=query[(size_t(kh)*seq+first+t)*D+lane+r*32];
                for(int col=0;col<4;++col)dot[col]=fmaf(local[col][r],q,dot[col]);
            }
            for(int col=0;col<4;++col)attention[col]=src[A+t*C+lane]*h[col];
            for(int offset=16;offset;offset/=2)for(int col=0;col<4;++col) {
                dot[col]+=__shfl_down_sync(0xffffffffu,dot[col],offset);
                attention[col]+=__shfl_down_sync(0xffffffffu,attention[col],offset);
            }
            if(lane==0)for(int col=0;col<4;++col)
                output[(size_t(first+t)*values+vh)*D+value+col]=
                    (dot[col]*src[P+t]+attention[col])*scale;
        }
        for(int r=0;r<4;++r)for(int col=0;col<4;++col)local[col][r]*=src[P+C-1];
        for(int t=0;t<C;++t) {
            float delta[4];
            for(int col=0;col<4;++col)delta[col]=__shfl_sync(0xffffffffu,h[col],t);
            for(int r=0;r<4;++r) {
                // Padding contributes exactly zero; never read past real.
                const float k=t<n?key[(size_t(kh)*seq+first+t)*D+lane+r*32]*src[G+t]:0;
                for(int col=0;col<4;++col)local[col][r]=fmaf(k,delta[col],local[col][r]);
            }
        }
    }
    for(int r=0;r<4;++r)*reinterpret_cast<float4*>(base+size_t(lane+r*32)*D+value)=
        make_float4(local[0][r],local[1][r],local[2][r],local[3][r]);
    if(lane==0)for(int t=real;t<seq;++t)for(int col=0;col<4;++col)
        output[(size_t(t)*values+vh)*D+value+col]=0;
}

// Three TF32 products retain the residuals of BOTH operands. This is an
// independent precision experiment, not permission to change model tolerances.
// Each warp owns complete 16x16 output tiles; callers synchronize shared data.
__device__ void gemm_tf32_residual(const float* a,int lda,const float* b,int ldb,
    float* out,int ldc,int rows,int columns,int depth,int warp_offset,int warp_stride) {
#if __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    const int warp=threadIdx.x/32-warp_offset,tiles=(rows/16)*(columns/16);
    const int stride=warp_stride?warp_stride:blockDim.x/32;
    for(int tile=warp;tile<tiles;tile+=stride) {
        const int row=(tile/(columns/16))*16,col=(tile%(columns/16))*16;
        wmma::fragment<wmma::accumulator,16,16,8,float> acc;
        wmma::fill_fragment(acc,0.f);
        for(int inner=0;inner<depth;inner+=8) {
            wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> ah,al;
            wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bh,bl;
            wmma::load_matrix_sync(ah,a+row*lda+inner,lda);
            wmma::load_matrix_sync(bh,b+inner*ldb+col,ldb);
            for(int i=0;i<ah.num_elements;++i) {
                const float x=ah.x[i];ah.x[i]=wmma::__float_to_tf32(x);al.x[i]=wmma::__float_to_tf32(x-ah.x[i]);
            }
            for(int i=0;i<bh.num_elements;++i) {
                const float x=bh.x[i];bh.x[i]=wmma::__float_to_tf32(x);bl.x[i]=wmma::__float_to_tf32(x-bh.x[i]);
            }
            wmma::mma_sync(acc,al,bh,acc);
            wmma::mma_sync(acc,ah,bl,acc);
            wmma::mma_sync(acc,ah,bh,acc);
        }
        wmma::store_matrix_sync(out+row*ldc+col,acc,ldc,wmma::mem_row_major);
    }
#endif
}

// Each warp owns all output columns of a 16-row tile. It consumes that entire
// weighted-K tile before overwriting it; other warps never read those rows.
// Ordinary column-distributed GEMM would race if its output aliased K.
template<int Columns>
__device__ void update_inplace(float* weighted_key,int key_pitch,const float* h,int h_pitch) {
#if __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    const int warp=threadIdx.x/32;
    for(int row=warp*16;row<D;row+=(blockDim.x/32)*16) {
        wmma::fragment<wmma::accumulator,16,16,8,float> acc[Columns/16];
        for(int col=0;col<Columns/16;++col)wmma::fill_fragment(acc[col],0.f);
        for(int inner=0;inner<C;inner+=8) {
            wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> ah,al;
            wmma::load_matrix_sync(ah,weighted_key+row*key_pitch+inner,key_pitch);
            for(int i=0;i<ah.num_elements;++i) {
                const float x=ah.x[i];ah.x[i]=wmma::__float_to_tf32(x);al.x[i]=wmma::__float_to_tf32(x-ah.x[i]);
            }
            for(int col=0;col<Columns/16;++col) {
                wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> bh,bl;
                wmma::load_matrix_sync(bh,h+inner*h_pitch+col*16,h_pitch);
                for(int i=0;i<bh.num_elements;++i) {
                    const float x=bh.x[i];bh.x[i]=wmma::__float_to_tf32(x);bl.x[i]=wmma::__float_to_tf32(x-bh.x[i]);
                }
                wmma::mma_sync(acc[col],al,bh,acc[col]);
                wmma::mma_sync(acc[col],ah,bl,acc[col]);
                wmma::mma_sync(acc[col],ah,bh,acc[col]);
            }
        }
        // mma_sync is not a shared-memory ordering barrier. Ensure every
        // lane finished its K loads before any lane overwrites the tile.
        __syncwarp();
        for(int col=0;col<Columns/16;++col)
            wmma::store_matrix_sync(weighted_key+row*key_pitch+col*16,acc[col],key_pitch,wmma::mem_row_major);
        __syncwarp();
    }
#endif
}

template<int Columns,int Padding=0,bool SharedUpdate=false,bool DualProjection=false,bool StagedCoefficients=false>
__global__ void apply_tensor_core(const float* query,const float* key,const float* work,
    float* state,float* output,int heads,int values,int seq,int real,int chunks,float scale) {
    const int vh=blockIdx.y,kh=vh/(values/heads),column=blockIdx.x*Columns,tid=threadIdx.x;
    constexpr int Pitch=Columns+Padding,QPitch=D+Padding,KPitch=C+Padding;
    constexpr int OperandSize=D*KPitch>C*QPitch?D*KPitch:C*QPitch;
    static_assert(!StagedCoefficients||(Columns==32&&Padding==0&&SharedUpdate&&!DualProjection));
    __shared__ __align__(32) float st[D*Pitch],h[C*Pitch],temp[C*Pitch],operand[OperandSize];
    float* base=state+size_t(vh)*D*D;
    for(int i=tid;i<D*Columns;i+=blockDim.x)st[(i/Columns)*Pitch+i%Columns]=base[(i/Columns)*D+column+i%Columns];
    __syncthreads();
    for(int block=0;block<chunks;++block) {
        const int first=block*C,n=min(C,real-first);
        const float* src=work+(size_t(vh)*chunks+block)*S;
        if constexpr(DualProjection) {
            static_assert(Columns==32&&Padding==0&&SharedUpdate);
            // Two independent 32x32 projections share the read-only carry.
            // Each owns four complete output tiles, split across eight warps.
            for(int i=tid;i<C*D;i+=blockDim.x)
                operand[(i/D)*QPitch+i%D]=i/D<n?query[(size_t(kh)*seq+first+i/D)*D+i%D]:0;
            __syncthreads();
            if(tid<128)gemm_tf32_residual(src+W,D,st,Pitch,h,Pitch,C,Columns,D,0,4);
            else gemm_tf32_residual(operand,QPitch,st,Pitch,temp,Pitch,C,Columns,D,4,4);
            __syncthreads();
            for(int i=tid;i<C*Columns;i+=blockDim.x) {
                const int j=(i/Columns)*Pitch+i%Columns;
                h[j]=src[U+(i/Columns)*D+column+i%Columns]-h[j];
            }
        } else {
            // Reuse the query/weighted-K scratch before either is live.
            // Copy unchanged FP32 coefficients cooperatively instead of
            // loading W directly from global memory in each WMMA tile.
            if constexpr(StagedCoefficients) {
                for(int i=tid;i<C*D;i+=blockDim.x)operand[i]=src[W+i];
                __syncthreads();
            }
            gemm_tf32_residual(StagedCoefficients?operand:src+W,D,st,Pitch,h,Pitch,C,Columns,D);
            __syncthreads();
            for(int i=tid;i<C*Columns;i+=blockDim.x) {
                const int j=(i/Columns)*Pitch+i%Columns;
                h[j]=src[U+(i/Columns)*D+column+i%Columns]-h[j];
            }
            for(int i=tid;i<C*D;i+=blockDim.x)
                operand[(i/D)*QPitch+i%D]=i/D<n?query[(size_t(kh)*seq+first+i/D)*D+i%D]:0;
            __syncthreads();
            gemm_tf32_residual(operand,QPitch,st,Pitch,temp,Pitch,C,Columns,D);
            __syncthreads();
        }
        // Reuse the query tile to retain the initial-state contribution.
        // H and the old state stay intact until the state update.
        for(int i=tid;i<C*Columns;i+=blockDim.x)operand[i]=temp[(i/Columns)*Pitch+i%Columns]*src[P+i/Columns];
        // The remaining scratch has room for A beside the live contribution.
        // Its cooperative copy shares the already-required consumer barrier.
        if constexpr(StagedCoefficients)
            for(int i=tid;i<C*C;i+=blockDim.x)operand[C*Columns+i]=src[A+i];
        __syncthreads();
        gemm_tf32_residual(StagedCoefficients?operand+C*Columns:src+A,C,h,Pitch,temp,Pitch,C,Columns,C);
        __syncthreads();
        for(int i=tid;i<n*Columns;i+=blockDim.x)
            output[(size_t(first+i/Columns)*values+vh)*D+column+i%Columns]=(operand[i]+temp[(i/Columns)*Pitch+i%Columns])*scale;
        __syncthreads();
        // Transpose weighted K into row-major [128,32] for the outer update.
        for(int i=tid;i<D*C;i+=blockDim.x) {
            const int r=i/C,t=i%C;
            operand[r*KPitch+t]=t<n?key[(size_t(kh)*seq+first+t)*D+r]*src[G+t]:0;
        }
        __syncthreads();
        // Accumulate into the CTA's disjoint global state columns, then reload
        // the carry. No other CTA reads or writes these columns.
        if constexpr(SharedUpdate)update_inplace<Columns>(operand,KPitch,h,Pitch);
        else gemm_tf32_residual(operand,KPitch,h,Pitch,base+column,D,D,Columns,C);
        __syncthreads();
        for(int i=tid;i<D*Columns;i+=blockDim.x) {
            const int offset=(i/Columns)*D+column+i%Columns;
            const int j=(i/Columns)*Pitch+i%Columns;
            const float delta=SharedUpdate?operand[(i/Columns)*KPitch+i%Columns]:base[offset];
            st[j]=fmaf(st[j],src[P+C-1],delta);
        }
        __syncthreads();
    }
    for(int i=tid;i<D*Columns;i+=blockDim.x)base[(i/Columns)*D+column+i%Columns]=st[(i/Columns)*Pitch+i%Columns];
    for(int i=tid;i<(seq-real)*Columns;i+=blockDim.x)
        output[(size_t(real+i/Columns)*values+vh)*D+column+i%Columns]=0;
}

__device__ void split_half(float x,__half* hi,__half* lo) {
    *hi=__float2half_rn(x);
    *lo=__float2half_rn((x-__half2float(*hi))*4096.f);
}

template<bool AddState=false>
__device__ void gemm_half_residual(const __half* a,const __half* al,int lda,
    const __half* b,const __half* bl,int ldb,float* out,int ldc,int rows,int depth,float decay=0) {
#if __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    const int warp=threadIdx.x/32;
    for(int tile=warp;tile<(rows/16)*2;tile+=blockDim.x/32) {
        const int row=(tile/2)*16,col=(tile%2)*16;
        wmma::fragment<wmma::accumulator,16,16,16,float> high,low;
        wmma::fill_fragment(high,0.f);wmma::fill_fragment(low,0.f);
        for(int inner=0;inner<depth;inner+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> ah,ar;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> bh,br;
            wmma::load_matrix_sync(ah,a+row*lda+inner,lda);
            wmma::load_matrix_sync(ar,al+row*lda+inner,lda);
            wmma::load_matrix_sync(bh,b+inner*ldb+col,ldb);
            wmma::load_matrix_sync(br,bl+inner*ldb+col,ldb);
            wmma::mma_sync(high,ah,bh,high);
            wmma::mma_sync(low,ar,bh,low);
            wmma::mma_sync(low,ah,br,low);
        }
        for(int i=0;i<high.num_elements;++i)high.x[i]+=low.x[i]*(1.f/4096);
        if constexpr(AddState) {
            wmma::fragment<wmma::accumulator,16,16,16,float> old;
            wmma::load_matrix_sync(old,out+row*ldc+col,ldc,wmma::mem_row_major);
            for(int i=0;i<high.num_elements;++i)high.x[i]=fmaf(old.x[i],decay,high.x[i]);
        }
        wmma::store_matrix_sync(out+row*ldc+col,high,ldc,wmma::mem_row_major);
    }
#endif
}

// FP32 state carry; FP16 is only a split representation of GEMM operands.
// The low operand is scaled by4096 to avoid losing its useful bits to half
// subnormal rounding. high*high and the two residual products accumulate in
// separate FP32 fragments. This experimental fixture is NOT a model gate.
__global__ void apply_half_products(const float* query,const float* key,const float* work,
    float* state,float* output,int heads,int values,int seq,int real,int chunks,float scale) {
    extern __shared__ __align__(32) unsigned char storage[];
    float* st=reinterpret_cast<float*>(storage);
    float* h=st+D*32;
    float* temp=h+C*32;
    __half* a=reinterpret_cast<__half*>(temp+C*32);
    __half* al=a+C*D;
    __half* b=al+C*D;
    __half* bl=b+C*D;
    const int vh=blockIdx.y,kh=vh/(values/heads),column=blockIdx.x*32,tid=threadIdx.x;
    float* base=state+size_t(vh)*D*D;
    for(int i=tid;i<D*32;i+=blockDim.x)st[i]=base[(i/32)*D+column+i%32];
    __syncthreads();
    for(int block=0;block<chunks;++block) {
        const int first=block*C,n=min(C,real-first);
        const float* src=work+(size_t(vh)*chunks+block)*S;
        for(int i=tid;i<D*32;i+=blockDim.x)split_half(st[i],b+i,bl+i);
        for(int i=tid;i<C*D;i+=blockDim.x)split_half(src[W+i],a+i,al+i);
        __syncthreads();
        gemm_half_residual(a,al,D,b,bl,32,h,32,C,D);
        __syncthreads();
        for(int i=tid;i<C*32;i+=blockDim.x)h[i]=src[U+(i/32)*D+column+i%32]-h[i];
        for(int i=tid;i<C*D;i+=blockDim.x) {
            const int t=i/D,r=i%D;
            split_half(t<n?query[(size_t(kh)*seq+first+t)*D+r]:0,a+i,al+i);
        }
        __syncthreads();
        gemm_half_residual(a,al,D,b,bl,32,temp,32,C,D);
        __syncthreads();
        // Preserve H as a split operand, then reuse its FP32 storage for the
        // initial-state output. No extra per-chunk global temporary is needed.
        for(int i=tid;i<C*32;i+=blockDim.x) {
            split_half(h[i],b+i,bl+i);
            h[i]=temp[i]*src[P+i/32];
        }
        for(int i=tid;i<C*C;i+=blockDim.x)split_half(src[A+i],a+i,al+i);
        __syncthreads();
        gemm_half_residual(a,al,C,b,bl,32,temp,32,C,C);
        __syncthreads();
        for(int i=tid;i<n*32;i+=blockDim.x)
            output[(size_t(first+i/32)*values+vh)*D+column+i%32]=(h[i]+temp[i])*scale;
        for(int i=tid;i<D*C;i+=blockDim.x) {
            const int r=i/C,t=i%C;
            split_half(t<n?key[(size_t(kh)*seq+first+t)*D+r]*src[G+t]:0,a+i,al+i);
        }
        __syncthreads();
        gemm_half_residual<true>(a,al,C,b,bl,32,st,32,D,C,src[P+C-1]);
        __syncthreads();
    }
    for(int i=tid;i<D*32;i+=blockDim.x)base[(i/32)*D+column+i%32]=st[i];
    for(int i=tid;i<(seq-real)*32;i+=blockDim.x)
        output[(size_t(real+i/32)*values+vh)*D+column+i%32]=0;
}

void launch_apply(const float* q,const float* k,const float* work,float* state,float* out,
    int heads,int values,int seq,int real,int chunks,float scale) {
    if(staged_coefficients)apply_tensor_core<32,0,true,false,true><<<dim3(4,values),256>>>(q,k,work,state,out,heads,values,seq,real,chunks,scale);
    else if(dual_projection)apply_tensor_core<32,0,true,true><<<dim3(4,values),256>>>(q,k,work,state,out,heads,values,seq,real,chunks,scale);
    else if(half_products)apply_half_products<<<dim3(4,values),apply_threads,HalfSharedBytes>>>(q,k,work,state,out,heads,values,seq,real,chunks,scale);
    else if(tensor_core&&shared_update)apply_tensor_core<32,0,true><<<dim3(4,values),apply_threads>>>(q,k,work,state,out,heads,values,seq,real,chunks,scale);
    else if(tensor_core&&tensor_padding)apply_tensor_core<32,8><<<dim3(4,values),apply_threads>>>(q,k,work,state,out,heads,values,seq,real,chunks,scale);
    else if(tensor_core&&tensor_columns==32)apply_tensor_core<32><<<dim3(4,values),apply_threads>>>(q,k,work,state,out,heads,values,seq,real,chunks,scale);
    else if(tensor_core)apply_tensor_core<16><<<dim3(8,values),apply_threads>>>(q,k,work,state,out,heads,values,seq,real,chunks,scale);
    else apply<<<dim3(16,values),64>>>(q,k,work,state,out,heads,values,seq,real,chunks,scale);
}

bool launch(const float* qkv,const float* q,const float* k,const float* d,const float* b,
    float* state,float* out,float* work,size_t work_bytes,int heads,int values,int seq,int real,float scale,float* grams=nullptr,size_t grams_bytes=0) {
    if(heads<1||heads>16||values<1||values>48||values%heads||seq<1||seq>2048||real<0||real>seq||
       !qkv||!q||!k||!d||!b||!state||!out||!std::isfinite(scale))return false;
    const int chunks=(real+C-1)/C;
    const size_t required=size_t(values)*chunks*S*4;
    const size_t grams_required=shared_grams?size_t(heads)*chunks*2*C*C*4:0;
    if(grams_required&&(!grams||grams_required>grams_bytes||(reinterpret_cast<uintptr_t>(grams)&31)))return false;
    if(required>work_bytes||(required&&!work)||(reinterpret_cast<uintptr_t>(state)&15))return false;
    if(tensor_core&&((reinterpret_cast<uintptr_t>(state)&31)||
        (required&&(reinterpret_cast<uintptr_t>(work)&31))))return false;
    const void* ptrs[]={qkv,q,k,d,b,state,out,work,grams};
    const size_t lengths[]={size_t(seq)*(2*heads+values)*D*4,size_t(heads)*seq*D*4,
        size_t(heads)*seq*D*4,size_t(seq)*values*4,size_t(seq)*values*4,size_t(values)*D*D*4,
        size_t(seq)*values*D*4,required,grams_required};
    for(int i=0;i<9;++i) {
        const uintptr_t a=reinterpret_cast<uintptr_t>(ptrs[i]);
        if((a&3)||lengths[i]>UINTPTR_MAX-a)return false;
        for(int j=0;j<i;++j) {
            const uintptr_t previous=reinterpret_cast<uintptr_t>(ptrs[j]);
            if((i>=5||j>=5)&&lengths[i]&&lengths[j]&&a<previous+lengths[j]&&previous<a+lengths[i])return false;
        }
    }
    launch_prepare(qkv,q,k,d,b,work,heads,values,seq,real,chunks,grams);
    if(cudaGetLastError()!=cudaSuccess)return false;
    launch_apply(q,k,work,state,out,heads,values,seq,real,chunks,scale);
    return cudaGetLastError()==cudaSuccess;
}

void check(cudaError_t s){if(s!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(s));std::exit(1);}}
template<class Kernel>
void occupancy(const char* name,Kernel kernel,int threads,size_t dynamic,const cudaDeviceProp& prop) {
    cudaFuncAttributes attributes;check(cudaFuncGetAttributes(&attributes,kernel));
    int blocks=0;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,threads,dynamic));
    std::printf("resource=%s registers=%d static_shared=%zu dynamic_shared=%zu blocks_per_sm_limit=%d warps_per_sm_limit=%d max_warps_per_sm=%d scope=theoretical_not_measured_utilization\n",
        name,attributes.numRegs,attributes.sharedSizeBytes,dynamic,blocks,blocks*(threads/32),prop.maxThreadsPerMultiProcessor/32);
}
void require(bool b){if(!b){std::fprintf(stderr,"chunkwise prototype check failed\n");std::exit(2);}}
struct Buffer {
    float* p;size_t n;
    explicit Buffer(size_t count):n(count){check(cudaMalloc(&p,(n+8)*4));check(cudaMemset(p,0x5a,(n+8)*4));}
    ~Buffer(){cudaFree(p);}
};
void upload(Buffer& b,const std::vector<float>& x){require(x.size()==b.n);check(cudaMemcpy(b.p,x.data(),x.size()*4,cudaMemcpyHostToDevice));}
std::vector<float> read(const Buffer& b){std::vector<float> x(b.n+8);check(cudaMemcpy(x.data(),b.p,x.size()*4,cudaMemcpyDeviceToHost));return x;}
void guard(const Buffer& b){const auto x=read(b);const auto* tail=reinterpret_cast<const unsigned char*>(x.data()+b.n);for(int i=0;i<32;++i)require(tail[i]==0x5a);}
double error(const Buffer& a,const Buffer& b) {
    const auto x=read(a),y=read(b);double maximum=0;
    for(size_t i=0;i<a.n;++i){require(std::isfinite(x[i])&&std::isfinite(y[i]));maximum=std::max(maximum,double(std::fabs(x[i]-y[i])));}
    guard(a);guard(b);return maximum;
}
void compare_block_control(const Buffer& actual,const std::vector<float>& expected,bool workspace) {
    const auto values=read(actual);
    require(values.size()==expected.size());
    if(!tensor_coeff_combo) {
        require(!std::memcmp(values.data(),expected.data(),values.size()*4));
        return;
    }
    // ONLY this explicit arithmetic-order experiment permits differences in
    // U/W or their downstream output/state. Other workspace fields and all
    // guards stay byte-exact. This is not a model quality acceptance.
    double maximum=0;size_t changed=0;
    for(size_t i=0;i<values.size();++i) {
        const bool approximate=i<actual.n&&(!workspace||i%S<A);
        if(!approximate)require(!std::memcmp(&values[i],&expected[i],4));
        else {
            require(std::isfinite(values[i])&&std::isfinite(expected[i]));
            maximum=std::max(maximum,double(std::fabs(values[i]-expected[i])));
            changed+=std::memcmp(&values[i],&expected[i],4)!=0;
        }
    }
    require(maximum<=1e-5);
    std::printf("tensor_coeff_combo_control workspace=%d elements=%zu changed=%zu max_abs=%.9g other_fields_and_guards_byte_exact=true tolerance=1e-5_not_model_quality\n",
        int(workspace),actual.n,changed,maximum);
}
double timed(cudaGraphExec_t graph) {
    cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));
    check(cudaGraphLaunch(graph,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
    check(cudaEventRecord(a,cudaStreamPerThread));
    for(int i=0;i<timing_replays;++i)check(cudaGraphLaunch(graph,cudaStreamPerThread));
    check(cudaEventRecord(b,cudaStreamPerThread));check(cudaEventSynchronize(b));float ms;check(cudaEventElapsedTime(&ms,a,b));
    check(cudaEventDestroy(a));check(cudaEventDestroy(b));return ms*1000/timing_replays;
}
void run(int seq,int real,int heads,int values,bool timing) {
    const bool prepare_comparison=column_solve||coalesced_values||dual_projection||shared_grams||staged_coefficients||tensor_coeff_combo;
    const char* comparison_name=tensor_coeff_combo?"tensor_coeff_combo":staged_coefficients?"staged_coefficients":shared_grams?(dual_projection&&coalesced_values?"shared_grams_combo":"shared_grams"):dual_projection?"dual_projection":coalesced_values?"coalesced_values":"column_solve";
    const size_t qkn=size_t(heads)*seq*D,gn=size_t(values)*seq,sn=size_t(values)*D*D,on=gn*D;
    Buffer qkv(size_t(2*heads+values)*seq*D),q(qkn),k(qkn),dot(size_t(heads)*seq),d(gn),b(gn);
    Buffer initial(sn),ref(sn),candidate(sn),out0(on),out1(on),work(size_t(values)*((real+C-1)/C)*S);
    Buffer grams(shared_grams?size_t(heads)*((real+C-1)/C)*2*C*C:0);
    std::vector<float> hq(qkn),hk(qkn),hv(qkv.n),hd(gn),hb(gn),hs(sn),h_dot(dot.n);
    auto fill=[&](int variant) {
        for(int head=0;head<heads;++head)for(int t=0;t<seq;++t) {
            float qsum=0,ksum=0;
            for(int r=0;r<D;++r) {
                const size_t i=(size_t(head)*seq+t)*D+r;
                hq[i]=(int((r*7+t*3+head+variant)%37)-18)/37.f;
                hk[i]=(int((r*11+(variant==3?0:t)*5+head+variant)%41)-20)/41.f;
                qsum=fmaf(hq[i],hq[i],qsum);ksum=fmaf(hk[i],hk[i],ksum);
            }
            float partial[32]={};
            for(int r=0;r<D;++r) {
                const size_t i=(size_t(head)*seq+t)*D+r;
                hq[i]/=std::sqrt(qsum+1e-6f);hk[i]/=std::sqrt(ksum+1e-6f);
            }
            for(int lane=0;lane<32;++lane)for(int r=0;r<4;++r) {
                const size_t i=(size_t(head)*seq+t)*D+lane+r*32;
                partial[lane]=fmaf(hq[i],hk[i],partial[lane]);
            }
            for(int offset=16;offset;offset/=2)for(int lane=0;lane<32-offset;++lane)partial[lane]+=partial[lane+offset];
            h_dot[size_t(head)*seq+t]=partial[0];
        }
        for(size_t i=0;i<hv.size();++i)hv[i]=(int((i*13+variant)%47)-23)/97.f;
        for(size_t i=0;i<hs.size();++i)hs[i]=(int((i*5+variant)%31)-15)/1024.f;
        for(int t=0;t<seq;++t)for(int h=0;h<values;++h) {
            const int i=t*values+h;
            hd[i]=variant==2?1e-4f:variant==3?1.f:.95f+((t+h)%7)*.007f;
            hb[i]=variant==3?1.f:.2f+((t*3+h)%7)*.1f;
            if(variant==1){if(t%17==0)hd[i]=0;if(t%19==1)hd[i]=1;if(t%11==0)hb[i]=0;if(t%13==1)hb[i]=1;}
        }
        upload(q,hq);upload(k,hk);upload(qkv,hv);upload(dot,h_dot);upload(initial,hs);upload(d,hd);upload(b,hb);
    };
    auto call=[&](int mode) {
        if(mode!=2)check(cudaMemcpyAsync(mode?candidate.p:ref.p,initial.p,sn*4,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
        if(mode==1)require(launch(qkv.p,q.p,k.p,d.p,b.p,candidate.p,out1.p,work.p,work.n*4,heads,values,seq,real,.125f,grams.p,grams.n*4));
        else if(mode==2) {
            launch_prepare(qkv.p,q.p,k.p,d.p,b.p,work.p,heads,values,seq,real,(real+C-1)/C,grams.p);
            check(cudaGetLastError());
        } else if(mode==3) {
            launch_apply(q.p,k.p,work.p,candidate.p,out1.p,heads,values,seq,real,(real+C-1)/C,.125f);
            check(cudaGetLastError());
        } else require(mollm_cuda::launch_gdn_recurrence_probe(qkv.p,q.p,k.p,dot.p,d.p,b.p,ref.p,out0.p,
            heads,values,seq,real,.125f,false,2,false,false,false,true));
    };
    fill(0);
    require(!launch(qkv.p,q.p,k.p,d.p,b.p,candidate.p,out1.p,work.p,work.n*4,0,values,seq,real,.125f));
    require(!launch(qkv.p,q.p,k.p,d.p,b.p,candidate.p,out1.p,work.p,work.n*4,heads,values,seq,seq+1,.125f));
    require(!launch(qkv.p,q.p,k.p,d.p,b.p,candidate.p,candidate.p,work.p,work.n*4,heads,values,seq,real,.125f));
    if(real)require(!launch(qkv.p,q.p,k.p,d.p,b.p,candidate.p,out1.p,work.p,work.n*4-1,heads,values,seq,real,.125f));
    if(tensor_core) {
        require(!launch(qkv.p,q.p,k.p,d.p,b.p,candidate.p+4,out1.p,work.p,work.n*4,heads,values,seq,real,.125f));
        if(real)require(!launch(qkv.p,q.p,k.p,d.p,b.p,candidate.p,out1.p,work.p+1,work.n*4,heads,values,seq,real,.125f));
    }
    if(shared_grams&&real) {
        auto gram_launch=[&](float* ptr,size_t bytes) {return launch(qkv.p,q.p,k.p,d.p,b.p,candidate.p,out1.p,work.p,work.n*4,heads,values,seq,real,.125f,ptr,bytes);};
        require(!gram_launch(nullptr,grams.n*4));
        require(!gram_launch(grams.p,grams.n*4-1));
        require(!gram_launch(grams.p+1,grams.n*4));
        require(!gram_launch(work.p,grams.n*4));
        require(!gram_launch(q.p,grams.n*4));
    }
    for(int variant=0;variant<4;++variant) {
        fill(variant);call(0);
        std::vector<float> original_work,original_out,original_state;
        if(prepare_comparison) {
            const bool saved_column=column_solve,saved_values=coalesced_values,saved_dual=dual_projection,saved_grams=shared_grams,saved_stage=staged_coefficients,saved_combo=tensor_coeff_combo,saved_coeff=tensor_coeff;
            // Isolate ONLY Gram reuse when selected. Combined-mode control
            // retains coalesced V and concurrent projections in both graphs.
            if(!saved_stage&&!saved_combo) {
                if(!saved_grams){column_solve=false;coalesced_values=false;dual_projection=false;}
                shared_grams=false;
            }
            staged_coefficients=false;tensor_coeff_combo=false;
            if(saved_combo)tensor_coeff=false;
            call(1);
            column_solve=saved_column;coalesced_values=saved_values;dual_projection=saved_dual;shared_grams=saved_grams;staged_coefficients=saved_stage;tensor_coeff_combo=saved_combo;tensor_coeff=saved_coeff;
            original_work=read(work);original_out=read(out1);original_state=read(candidate);
        }
        call(1);check(cudaDeviceSynchronize());
        if(prepare_comparison) {
            for(auto item:{std::pair<const Buffer*,const std::vector<float>*>{&work,&original_work},
                {&out1,&original_out},{&candidate,&original_state}}) {
                compare_block_control(*item.first,*item.second,item.first==&work);
            }
        }
        const double oe=error(out0,out1),se=error(ref,candidate);
        std::printf("seq=%d real=%d heads=%d values=%d variant=%d output_max_abs=%.9g state_max_abs=%.9g workspace_bytes=%zu\n",
            seq,real,heads,values,variant,oe,se,work.n*4);
        require(oe<=1e-5&&se<=1e-5);guard(work);guard(grams);
    }
    cudaGraphExec_t graphs[6]{};
    const int graph_count=prepare_comparison?6:4;
    for(int mode=0;mode<graph_count;++mode) {
        const bool saved=column_solve,saved_values=coalesced_values,saved_dual=dual_projection,saved_grams=shared_grams,saved_stage=staged_coefficients,saved_combo=tensor_coeff_combo,saved_coeff=tensor_coeff;
        if(mode>=4) {
            if(!saved_stage&&!saved_combo) {
                if(!saved_grams){column_solve=false;coalesced_values=false;dual_projection=false;}
                shared_grams=false;
            }
            staged_coefficients=false;
            tensor_coeff_combo=false;
            if(saved_combo)tensor_coeff=false;
        }
        cudaGraph_t graph;check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));call(mode==4?1:mode==5?2:mode);
        check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiateWithFlags(&graphs[mode],graph,0));check(cudaGraphDestroy(graph));
        column_solve=saved;
        coalesced_values=saved_values;
        dual_projection=saved_dual;
        shared_grams=saved_grams;
        staged_coefficients=saved_stage;
        tensor_coeff_combo=saved_combo;tensor_coeff=saved_coeff;
    }
    for(int variant:{1,2,0}) {
        fill(variant);
        std::vector<float> original_work,original_out,original_state;
        if(prepare_comparison) {
            check(cudaGraphLaunch(graphs[4],cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
            original_work=read(work);original_out=read(out1);original_state=read(candidate);
        }
        for(int mode=0;mode<2;++mode)check(cudaGraphLaunch(graphs[mode],cudaStreamPerThread));
        check(cudaStreamSynchronize(cudaStreamPerThread));require(error(out0,out1)<=1e-5&&error(ref,candidate)<=1e-5);guard(work);guard(grams);
        if(prepare_comparison)for(auto item:{std::pair<const Buffer*,const std::vector<float>*>{&work,&original_work},
            {&out1,&original_out},{&candidate,&original_state}}) {
            compare_block_control(*item.first,*item.second,item.first==&work);
        }
        check(cudaGraphLaunch(graphs[2],cudaStreamPerThread));
        check(cudaGraphLaunch(graphs[3],cudaStreamPerThread));
        check(cudaStreamSynchronize(cudaStreamPerThread));require(error(out0,out1)<=1e-5&&error(ref,candidate)<=1e-5);guard(work);guard(grams);
    }
    if(timing)for(int sample=0;sample<7;++sample) {
        double times[2];for(int j=0;j<2;++j){int mode=(sample+j)%2;times[mode]=timed(graphs[mode]);}
        std::printf("sample=%d seq=%d real=%d native_us=%.6f block_total_us=%.6f scope=includes_prepare_and_same_state_reset\n",sample,seq,real,times[0],times[1]);
        std::printf("phase_sample=%d seq=%d prepare_us=%.6f apply_reset_us=%.6f scope=component_timings_not_model_gain\n",
            sample,seq,timed(graphs[2]),timed(graphs[3]));
        if(prepare_comparison) {
            double block[2],prep[2];for(int j=0;j<2;++j){const int candidate=(sample+j)%2;
                block[candidate]=timed(graphs[candidate?1:4]);prep[candidate]=timed(graphs[candidate?2:5]);}
            std::printf("%s_sample=%d seq=%d baseline_block_us=%.6f candidate_block_us=%.6f baseline_prepare_us=%.6f candidate_prepare_us=%.6f scope=same_algorithm_reset_and_payload_not_model\n",
                comparison_name,sample,seq,block[0],block[1],prep[0],prep[1]);
        }
    }
    if(prepare_comparison)std::printf("%s_exact_workspace_output_state=%s seq=%d real=%d direct_variants=4 capture_variants=3\n",comparison_name,tensor_coeff_combo?"false_explicit_1e-5_experiment":"true",seq,real);
    for(auto graph:graphs)if(graph)check(cudaGraphExecDestroy(graph));
}
}
int main(int argc,char** argv) {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    bool timing=false,small=false,profile_shape=false,logical_shapes=false;
    for(int i=1;i<argc;++i){if(!std::strcmp(argv[i],"--bench"))timing=true;
        else if(!std::strcmp(argv[i],"--tensor-core"))tensor_core=true;
        else if(!std::strcmp(argv[i],"--value-tile32")){tensor_core=true;tensor_columns=32;}
        else if(!std::strcmp(argv[i],"--padded")){tensor_core=true;tensor_columns=32;tensor_padding=true;}
        else if(!std::strcmp(argv[i],"--tensor-prepare")){tensor_core=true;tensor_columns=32;tensor_prepare=true;}
        else if(!std::strcmp(argv[i],"--shared-update")){tensor_core=true;tensor_columns=32;tensor_prepare=true;shared_update=true;}
        else if(!std::strcmp(argv[i],"--half-products")){tensor_core=true;tensor_columns=32;tensor_prepare=true;half_products=true;}
        else if(!std::strcmp(argv[i],"--eight-warps"))apply_threads=256;
        else if(!std::strcmp(argv[i],"--tensor-coeff")){tensor_core=true;tensor_columns=32;tensor_prepare=true;tensor_coeff=true;shared_update=true;apply_threads=256;}
        else if(!std::strcmp(argv[i],"--column-solve")){tensor_core=true;tensor_columns=32;tensor_prepare=true;shared_update=true;apply_threads=256;column_solve=true;}
        else if(!std::strcmp(argv[i],"--coalesced-values")){tensor_core=true;tensor_columns=32;tensor_prepare=true;shared_update=true;apply_threads=256;coalesced_values=true;}
        else if(!std::strcmp(argv[i],"--dual-projection")){tensor_core=true;tensor_columns=32;tensor_prepare=true;shared_update=true;apply_threads=256;dual_projection=true;}
        else if(!std::strcmp(argv[i],"--shared-grams")){tensor_core=true;tensor_columns=32;tensor_prepare=true;shared_update=true;apply_threads=256;shared_grams=true;}
        else if(!std::strcmp(argv[i],"--staged-coefficients")){tensor_core=true;tensor_columns=32;tensor_prepare=true;shared_update=true;apply_threads=256;staged_coefficients=true;}
        else if(!std::strcmp(argv[i],"--tensor-coeff-combo")){tensor_core=true;tensor_columns=32;tensor_prepare=true;shared_update=true;apply_threads=256;shared_grams=true;coalesced_values=true;dual_projection=true;tensor_coeff=true;tensor_coeff_combo=true;}
        else if(!std::strcmp(argv[i],"--timing-replays")) {
            if(i+1==argc)return 2;
            const char* count=argv[++i];
            if(std::strcmp(count,"16")&&std::strcmp(count,"64"))return 2;
            timing_replays=std::atoi(count);
        }
        else if(!std::strcmp(argv[i],"--small-only"))small=true;
        else if(!std::strcmp(argv[i],"--profile-shape"))profile_shape=true;
        else if(!std::strcmp(argv[i],"--logical-shapes"))logical_shapes=true;
        else if(std::strcmp(argv[i],"--check-only"))return 2;}
    if(logical_shapes&&(small||profile_shape))return 2;
    cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,0));if(tensor_core&&prop.major<8)return 77;
    if(column_solve&&(tensor_coeff||half_products||tensor_padding||apply_threads!=256))return 2;
    if(coalesced_values&&(column_solve||(tensor_coeff&&!tensor_coeff_combo)||half_products||tensor_padding||apply_threads!=256))return 2;
    if(dual_projection&&(column_solve||(coalesced_values&&!shared_grams)||(tensor_coeff&&!tensor_coeff_combo)||half_products||tensor_padding||apply_threads!=256))return 2;
    if(shared_grams&&(column_solve||(tensor_coeff&&!tensor_coeff_combo)||half_products||tensor_padding||apply_threads!=256))return 2;
    if(staged_coefficients&&(column_solve||coalesced_values||dual_projection||shared_grams||tensor_coeff||half_products||tensor_padding||apply_threads!=256))return 2;
    if(tensor_coeff_combo&&(!shared_grams||!coalesced_values||!dual_projection||column_solve||half_products||tensor_padding||staged_coefficients||apply_threads!=256))return 2;
    if(half_products) {
        if(prop.sharedMemPerBlockOptin<HalfSharedBytes)return 77;
        check(cudaFuncSetAttribute(apply_half_products,cudaFuncAttributeMaxDynamicSharedMemorySize,HalfSharedBytes));
    }
    std::printf("apply_mode=%s value_tile=%d padding=%d tensor_prepare=%d tensor_coeff=%d shared_update=%d apply_threads=%d\n",half_products?"half_three_products":tensor_core?"tf32_three_products":"scalar_fp32",tensor_columns,tensor_padding?8:0,tensor_prepare,tensor_coeff,shared_update,apply_threads);
    std::printf("device=%s sms=%d shared_per_sm=%zu\n",prop.name,prop.multiProcessorCount,prop.sharedMemPerMultiprocessor);
    std::printf("internal_chunk_rows=%d prefill_input_shape_unchanged=true\n",C);
    std::printf("timing_replays=%d\n",timing_replays);
    std::printf("shared_grams=%d coalesced_values=%d dual_projection=%d gram_control_retains_other_flags=%d\n",shared_grams,coalesced_values,dual_projection,shared_grams);
    if(staged_coefficients){occupancy("staged_coefficients",apply_tensor_core<32,0,true,false,true>,256,0,prop);occupancy("original_apply",apply_tensor_core<32,0,true>,256,0,prop);}
    else if(dual_projection){occupancy("dual_projection",apply_tensor_core<32,0,true,true>,256,0,prop);occupancy("original_apply",apply_tensor_core<32,0,true>,256,0,prop);}
    else if(half_products)occupancy("half_products",apply_half_products,apply_threads,HalfSharedBytes,prop);
    else if(shared_update)occupancy("tf32_shared_update",apply_tensor_core<32,0,true>,apply_threads,0,prop);
    else if(tensor_padding)occupancy("tf32_padded",apply_tensor_core<32,8>,apply_threads,0,prop);
    else if(tensor_core&&tensor_columns==32)occupancy("tf32_tile32",apply_tensor_core<32>,apply_threads,0,prop);
    else if(tensor_core)occupancy("tf32_tile16",apply_tensor_core<16>,apply_threads,0,prop);
    else occupancy("scalar_block",apply,64,0,prop);
    if(shared_grams){occupancy("prepare_grams",prepare_grams,256,0,prop);
        if(tensor_coeff_combo){occupancy("tensor_coeff_combo",prepare<true,true,false,true,true>,256,0,prop);occupancy("original_prepare",prepare<true,false,false,true,true>,256,0,prop);}
        else if(coalesced_values)occupancy("shared_grams_coalesced",prepare<true,false,false,true,true>,256,0,prop);
        else occupancy("shared_grams",prepare<true,false,false,false,true>,256,0,prop);}
    else if(coalesced_values){occupancy("coalesced_values",prepare<true,false,false,true>,256,0,prop);occupancy("original_prepare",prepare<true>,256,0,prop);}
    else if(column_solve){occupancy("column_solve",prepare<true,false,true>,256,0,prop);occupancy("original_prepare",prepare<true>,256,0,prop);}
    else if(tensor_coeff)occupancy("tensor_coeff",prepare<true,true>,256,0,prop);
    else if(tensor_prepare)occupancy("tensor_prepare",prepare<true>,256,0,prop);
    else occupancy("scalar_prepare",prepare<false>,256,0,prop);
    if(logical_shapes) {
        std::puts("scope=current-logical-prefill_Q2048_and_Q957_H16_VH48_native_quad_vs_block_total_not_model_quality");
        run(2048,2048,16,48,timing);run(957,957,16,48,timing);
    }
    else if(profile_shape)run(1024,1024,16,48,timing);
    else {
        for(auto pair:std::vector<std::pair<int,int>>{{1,0},{1,1},{17,16},{33,33},{65,63}})run(pair.first,pair.second,2,6,false);
        if(!small){run(719,719,16,48,timing);run(1024,1024,16,48,timing);}
    }
    std::puts("GDN chunkwise native prototype passed; NOT byte-exact/model-quality/TPS proof");
}
