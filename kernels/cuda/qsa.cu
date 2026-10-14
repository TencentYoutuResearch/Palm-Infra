#include "kernels/cuda/qsa.h"
#include "kernels/cuda/reduction.cuh"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cmath>
#include <limits>
#include <cub/block/block_radix_sort.cuh>

namespace {
__device__ void indexer_norm_rotate(float value,const float* weight,
    const float* frequency,float* output,int dim,int rope_dim,int position,
    float eps,float* normalized,float* reduction) {
    const int d=threadIdx.x;
    const float square=d<dim?__fmul_rn(value,value):0.f;
    const float variance=mollm_cuda::detail::block_reduce_sum<256>(square,reduction)/dim;
    const float inverse=rsqrtf(variance+eps);
    if(d<dim)normalized[d]=__fmul_rn(__fmul_rn(value,inverse),__fadd_rn(1.f,weight[d]));
    __syncthreads();
    if(d<dim) {
        float x=normalized[d];
        if(d<rope_dim) {
            const int half=rope_dim/2,offset=d%half;
            float sine,cosine;
            sincosf(float(position)*frequency[offset],&sine,&cosine);
            const float rotated=d<half?-normalized[d+half]:normalized[d-half];
            x=__fadd_rn(__fmul_rn(x,cosine),__fmul_rn(rotated,sine));
        }
        output[d]=x;
    }
}

__global__ void qsa_append_query_raw(const float* projection,const float* weight,
    const float* frequency,float* raw,float* query,mollm_cuda::QsaIndexerParams p) {
    const int row=blockIdx.x,head=blockIdx.y,d=threadIdx.x;
    const float* source=projection+size_t(row)*p.projection_stride+head*p.dimension;
    if(head==p.heads) {
        if(d<p.dimension)raw[size_t(p.past+row)*p.dimension+d]=source[d];
        return;
    }
    __shared__ float normalized[256],reduction[256];
    indexer_norm_rotate(d<p.dimension?source[d]:0.f,weight,frequency,
        query+(size_t(row)*p.heads+head)*p.dimension,p.dimension,p.rope_dim,
        p.position_offset+p.past+row,p.eps,normalized,reduction);
}

__global__ void qsa_pool_complete(const float* raw,const float* weight,
    const float* frequency,float* pooled,mollm_cuda::QsaIndexerParams p) {
    const int group=p.past/p.ratio+blockIdx.x,d=threadIdx.x;
    float value=0;
    if(d<p.dimension) {
        for(int i=0;i<p.ratio;++i)value=__fadd_rn(value,raw[size_t(group*p.ratio+i)*p.dimension+d]);
        value/=p.ratio;
    }
    __shared__ float normalized[256],reduction[256];
    indexer_norm_rotate(value,weight,frequency,pooled+size_t(group)*p.dimension,
        p.dimension,p.rope_dim,p.position_offset+group*p.ratio,p.eps,normalized,reduction);
}

__global__ void qsa_index_scores(const float* query,const float* pooled,float* scores,
    mollm_cuda::QsaIndexerParams p) {
    const int row=blockIdx.x,group=blockIdx.y*8+threadIdx.x/32,lane=threadIdx.x%32;
    const int total_groups=(p.past+p.queries)/p.ratio;
    if(group>=total_groups)return; // uniform within each warp
    const bool visible=group<(p.past+row+1)/p.ratio;
    float score=0;
    if(visible) {
        const float* key=pooled+size_t(group)*p.dimension;
        for(int h=0;h<p.heads;++h) {
            const float* q=query+(size_t(row)*p.heads+h)*p.dimension;
            float dot=0;
            for(int d=lane;d<p.dimension;d+=32)dot=fmaf(q[d],key[d],dot);
            dot=mollm_cuda::detail::warp_reduce_sum(dot);
            if(lane==0)score+=fmaxf(0.f,dot);
        }
    }
    if(lane==0)scores[size_t(row)*p.score_stride+group]=visible?
        score/sqrtf(float(p.dimension)):-CUDART_INF_F;
}

template<class Cache>
__global__ void selected_attention(const float* query, const Cache* key,
    const Cache* value, const int32_t* indices, float* output,
    mollm_cuda::QsaSelectedAttentionParams p) {
    const int row=blockIdx.x, head=blockIdx.y;
    const int kv=head/(p.heads/p.kv_heads);
    const int lane=threadIdx.x%32, warp=threadIdx.x/32;
    const float* q=query+size_t(row)*p.query_position_stride+size_t(head)*p.query_head_stride;
    const int32_t* ids=indices+size_t(row)*p.index_position_stride;
    extern __shared__ float shared[];
    float* probabilities=shared;
    float* reduction=shared+p.slots;
    for(int slot=warp;slot<p.slots;slot+=8) {
        const int id=ids[slot];
        const bool active=id>=0&&id<p.keys&&id<=p.past+row;
        float dot=0;
        if(active) {
            const Cache* k=key+size_t(kv)*p.key_head_stride+size_t(id)*p.key_position_stride;
            for(int d=lane;d<p.key_dim;d+=32)
                dot=fmaf(q[d],float(k[d]),dot);
        }
        dot=mollm_cuda::detail::warp_reduce_sum(dot);
        if(lane==0)probabilities[slot]=active?dot*p.scale:-CUDART_INF_F;
    }
    __syncthreads();
    float maximum=-CUDART_INF_F;
    for(int slot=threadIdx.x;slot<p.slots;slot+=256)
        maximum=fmaxf(maximum,probabilities[slot]);
    maximum=mollm_cuda::detail::block_reduce_max<256>(maximum,reduction);
    float total=0;
    for(int slot=threadIdx.x;slot<p.slots;slot+=256) {
        // Avoid -inf - -inf on all-masked rows.
        const float score=probabilities[slot];
        const float weight=score==-CUDART_INF_F?0.f:expf(score-maximum);
        probabilities[slot]=weight;
        total+=weight;
    }
    total=mollm_cuda::detail::block_reduce_sum<256>(total,reduction);
    for(int slot=threadIdx.x;slot<p.slots;slot+=256)
        probabilities[slot]=total>0?probabilities[slot]/total:0.f;
    __syncthreads();
    float* out=output+size_t(row)*p.output_position_stride+size_t(head)*p.output_head_stride;
    for(int d=threadIdx.x;d<p.value_dim;d+=256) {
        float sum=0;
        for(int slot=0;slot<p.slots;++slot) {
            const int id=ids[slot];
            // Skip by validity, not just zero weight: do not read unselected
            // or future values, including poison values beyond actual keys.
            if(id>=0&&id<p.keys&&id<=p.past+row)
                sum=fmaf(probabilities[slot],float(value[size_t(kv)*p.value_head_stride+
                              size_t(id)*p.value_position_stride+d]),sum);
        }
        out[d]=sum;
    }
}

template<int Items>
__global__ void qsa_block_topk(const float* scores,int32_t* output,int keys,
    int past,int budget,int ratio,size_t score_stride,size_t index_stride) {
    using ScoreSort=cub::BlockRadixSort<float,256,Items,int>;
    using TokenSort=cub::BlockRadixSort<int,256,16>;
    union Storage {
        typename ScoreSort::TempStorage score;
        typename TokenSort::TempStorage token;
        int selected[4096];
    };
    static_assert(sizeof(Storage)<=48*1024,"QSA sort exceeds portable shared-memory budget");
    __shared__ Storage storage;
    const int row=blockIdx.x;
    const int visible=min(keys,past+row+1);
    const int complete=visible/ratio;
    const int top=min(budget/ratio,complete);
    const int count=top*ratio+visible%ratio;
    float values[Items];int groups[Items];
    for(int i=0;i<Items;++i) {
        const int group=threadIdx.x*Items+i;
        values[i]=group<complete?scores[size_t(row)*score_stride+group]:-CUDART_INF_F;
        groups[i]=group;
    }
    ScoreSort(storage.score).SortDescending(values,groups);
    __syncthreads();
    int* selected=storage.selected;
    for(int i=threadIdx.x;i<4096;i+=256)selected[i]=INT_MAX;
    __syncthreads();
    for(int i=0;i<Items;++i) {
        const int rank=threadIdx.x*Items+i;
        if(rank<top)for(int j=0;j<ratio;++j)selected[rank*ratio+j]=groups[i]*ratio+j;
    }
    for(int i=threadIdx.x;i<visible%ratio;i+=256)
        selected[top*ratio+i]=complete*ratio+i;
    __syncthreads();
    int tokens[16];
    for(int i=0;i<16;++i)tokens[i]=selected[threadIdx.x*16+i];
    __syncthreads();
    TokenSort(storage.token).Sort(tokens);
    for(int i=0;i<16;++i) {
        const int slot=threadIdx.x*16+i;
        if(slot<budget+ratio-1)
            output[size_t(row)*index_stride+slot]=slot<count?tokens[i]:-1;
    }
}
}

cudaError_t mollm_cuda::launch_qsa_indexer_update(const float* projection,const float* q_norm,
    const float* k_norm,const float* frequency,float* raw,float* pooled,float* query,
    float* scores,const QsaIndexerParams& p,cudaStream_t stream) {
    if(!projection||!q_norm||!k_norm||!frequency||!raw||!pooled||!query||!scores||
       p.heads<1||p.heads>16||p.dimension<1||p.dimension>256||p.rope_dim<2||
       p.rope_dim>p.dimension||p.rope_dim%2||p.ratio<1||p.ratio>16||
       p.queries<1||p.queries>65535||p.past<0||p.capacity<1||p.past>p.capacity||
       p.queries>p.capacity-p.past||p.position_offset<0||
       p.position_offset>INT_MAX-p.capacity||!std::isfinite(p.eps)||p.eps<=0||
       (p.past+p.queries)/p.ratio>8192||
       p.projection_stride<size_t((p.heads+1)*p.dimension)||
       p.score_stride<size_t((p.past+p.queries)/p.ratio))return cudaErrorInvalidValue;
    const size_t limit=std::numeric_limits<size_t>::max()/sizeof(float);
    if(p.projection_stride>limit/size_t(p.queries)||p.score_stride>limit/size_t(p.queries)||
       size_t(p.capacity)>limit/size_t(p.dimension)||
       size_t(p.queries)>limit/size_t(p.heads*p.dimension))return cudaErrorInvalidValue;
    qsa_append_query_raw<<<dim3(p.queries,p.heads+1),256,0,stream>>>(projection,q_norm,frequency,raw,query,p);
    auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    const int complete=(p.past+p.queries)/p.ratio,new_groups=complete-p.past/p.ratio;
    if(new_groups) {
        qsa_pool_complete<<<new_groups,256,0,stream>>>(raw,k_norm,frequency,pooled,p);
        error=cudaGetLastError();if(error!=cudaSuccess)return error;
    }
    if(complete) {
        qsa_index_scores<<<dim3(p.queries,(complete+7)/8),256,0,stream>>>(query,pooled,scores,p);
        error=cudaGetLastError();
    }
    return error;
}

cudaError_t mollm_cuda::launch_qsa_causal_block_topk(const float* scores,int32_t* ids,
    int queries,int keys,int past,int budget,int ratio,size_t score_stride,
    size_t index_stride,cudaStream_t stream) {
    if(!scores||!ids||queries<1||queries>65535||keys<1||past<0||past>keys||
       queries>keys-past||ratio<1||ratio>4096||budget<1||budget>4096||budget%ratio||
       budget>4096-ratio+1||keys/ratio>8192||score_stride<size_t(keys/ratio)||
       index_stride<size_t(budget+ratio-1))return cudaErrorInvalidValue;
    const size_t limit=std::numeric_limits<size_t>::max()/sizeof(float);
    if(score_stride>limit/size_t(queries)||index_stride>limit/size_t(queries))
        return cudaErrorInvalidValue;
    const int groups=keys/ratio;
    // Static shared storage remains below the standard48KiB launch budget for
    // these geometries; no device attribute mutation during graph capture.
    if(groups<=2048)qsa_block_topk<8><<<queries,256,0,stream>>>(scores,ids,keys,past,budget,ratio,score_stride,index_stride);
    else if(groups<=4096)qsa_block_topk<16><<<queries,256,0,stream>>>(scores,ids,keys,past,budget,ratio,score_stride,index_stride);
    else qsa_block_topk<32><<<queries,256,0,stream>>>(scores,ids,keys,past,budget,ratio,score_stride,index_stride);
    return cudaGetLastError();
}

cudaError_t mollm_cuda::launch_qsa_selected_attention(const float* query,
    const void* key,const void* value,bool half,const int32_t* ids,float* output,
    const QsaSelectedAttentionParams& p,cudaStream_t stream) {
    if(!query||!key||!value||!ids||!output||p.heads<1||p.kv_heads<1||
       p.heads%p.kv_heads||p.queries<1||p.queries>65535||p.heads>65535||
       p.keys<1||p.past<0||p.past>p.keys||p.queries>p.keys-p.past||
       p.key_dim<1||p.value_dim<1||p.key_dim>4096||p.value_dim>4096||
       p.slots<1||p.slots>4096||!std::isfinite(p.scale)||p.scale<=0||
       p.query_position_stride<size_t(p.key_dim)||p.query_head_stride<size_t(p.key_dim)||
       p.key_position_stride<size_t(p.key_dim)||p.key_head_stride<size_t(p.key_dim)||
       p.value_position_stride<size_t(p.value_dim)||p.value_head_stride<size_t(p.value_dim)||
       p.output_position_stride<size_t(p.value_dim)||p.output_head_stride<size_t(p.value_dim)||
       p.index_position_stride<size_t(p.slots))return cudaErrorInvalidValue;
    // Bound pointer offset arithmetic before launching; buffer capacities and
    // non-overlap remain the caller's explicit allocation contract.
    const size_t limit=std::numeric_limits<size_t>::max()/sizeof(float);
    const auto fits=[&](int rows,int heads,size_t row_stride,size_t head_stride,int dim) {
        return row_stride<=limit/size_t(rows)&&head_stride<=limit/size_t(heads)&&
            size_t(rows-1)*row_stride<=limit-size_t(dim)&&
            size_t(heads-1)*head_stride<=limit-size_t(dim)-size_t(rows-1)*row_stride;
    };
    if(!fits(p.queries,p.heads,p.query_position_stride,p.query_head_stride,p.key_dim)||
       !fits(p.keys,p.kv_heads,p.key_position_stride,p.key_head_stride,p.key_dim)||
       !fits(p.keys,p.kv_heads,p.value_position_stride,p.value_head_stride,p.value_dim)||
       !fits(p.queries,p.heads,p.output_position_stride,p.output_head_stride,p.value_dim)||
       p.index_position_stride>limit/size_t(p.queries))return cudaErrorInvalidValue;
    const size_t shared=size_t(p.slots+256)*sizeof(float);
    const dim3 grid(p.queries,p.heads);
    if(half)selected_attention<<<grid,256,shared,stream>>>(query,
        static_cast<const __half*>(key),static_cast<const __half*>(value),ids,output,p);
    else selected_attention<<<grid,256,shared,stream>>>(query,
        static_cast<const float*>(key),static_cast<const float*>(value),ids,output,p);
    return cudaGetLastError();
}
