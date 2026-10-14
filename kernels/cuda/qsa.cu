#include "kernels/cuda/qsa.h"
#include "kernels/cuda/reduction.cuh"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cmath>
#include <limits>
#include <cub/block/block_radix_sort.cuh>

namespace {
__global__ void validate_causal_mask(const float* mask,uint32_t* invalid,
    int queries,int keys,int past,size_t cs,size_t rs) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=size_t(queries)*keys)return;
    const int row=index/keys,col=index%keys;
    const float value=mask[size_t(row)*rs+size_t(col)*cs];
    // Future NaN/+inf would not be a valid additive attention mask even
    // though mask==0 treats it as invisible. Reject before mutating state.
    if(col<=past+row?value!=0.f:!(value<0.f))atomicOr(invalid,1u);
}
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
    const Cache* value, const int32_t* indices, float* output,const float* mask,
    mollm_cuda::QsaSelectedAttentionParams p) {
    const int row=blockIdx.x, head=blockIdx.y;
    const int partition_size=p.slots,begin=0,slots=p.slots;
    const int kv=head/(p.heads/p.kv_heads);
    const int lane=threadIdx.x%32, warp=threadIdx.x/32;
    const float* q=query+size_t(row)*p.query_position_stride+size_t(head)*p.query_head_stride;
    const int32_t* ids=indices+size_t(row)*p.index_position_stride+begin;
    extern __shared__ float shared[];
    float* probabilities=shared;
    float* reduction=shared+partition_size;
    for(int slot=warp;slot<slots;slot+=8) {
        const int id=ids[slot];
        const bool active=id>=0&&id<p.keys&&id<=p.past+row&&
            (!mask||mask[size_t(row)*p.mask_row_stride+size_t(id)*p.mask_column_stride]==0.f);
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
    for(int slot=threadIdx.x;slot<slots;slot+=256)
        maximum=fmaxf(maximum,probabilities[slot]);
    maximum=mollm_cuda::detail::block_reduce_max<256>(maximum,reduction);
    float total=0;
    for(int slot=threadIdx.x;slot<slots;slot+=256) {
        // Avoid -inf - -inf on all-masked rows.
        const float score=probabilities[slot];
        const float weight=score==-CUDART_INF_F?0.f:expf(score-maximum);
        probabilities[slot]=weight;
        total+=weight;
    }
    total=mollm_cuda::detail::block_reduce_sum<256>(total,reduction);
    for(int slot=threadIdx.x;slot<slots;slot+=256)
        probabilities[slot]=total>0?probabilities[slot]/total:0.f;
    __syncthreads();
    float* out=output+size_t(row)*p.output_position_stride+size_t(head)*p.output_head_stride;
    for(int d=threadIdx.x;d<p.value_dim;d+=256) {
        float sum=0;
        for(int slot=0;slot<slots;++slot) {
            const int id=ids[slot];
            // Skip by validity, not just zero weight: do not read unselected
            // or future values, including poison values beyond actual keys.
            if(id>=0&&id<p.keys&&id<=p.past+row&&
               (!mask||mask[size_t(row)*p.mask_row_stride+size_t(id)*p.mask_column_stride]==0.f))
                sum=fmaf(probabilities[slot],float(value[size_t(kv)*p.value_head_stride+
                              size_t(id)*p.value_position_stride+d]),sum);
        }
        out[d]=sum;
    }
}

template<class Cache>
__global__ void selected_scores_partitioned(const float* query,const Cache* key,
    const int32_t* indices,const float* mask,mollm_cuda::QsaSelectedAttentionParams p) {
    const int row=blockIdx.x,head=blockIdx.y,partition=blockIdx.z;
    const int width=(p.slots+p.partitions-1)/p.partitions;
    const int begin=min(p.slots,partition*width),count=min(width,p.slots-begin);
    const int kv=head/(p.heads/p.kv_heads),lane=threadIdx.x%32,warp=threadIdx.x/32;
    const float* q=query+size_t(row)*p.query_position_stride+size_t(head)*p.query_head_stride;
    const int32_t* ids=indices+size_t(row)*p.index_position_stride+begin;
    float* scores=p.selected_scores+(size_t(row)*p.heads+head)*p.slots+begin;
    for(int slot=warp;slot<count;slot+=8) {
        const int id=ids[slot];
        const bool active=id>=0&&id<p.keys&&id<=p.past+row&&
            (!mask||mask[size_t(row)*p.mask_row_stride+size_t(id)*p.mask_column_stride]==0.f);
        float dot=0;
        if(active) {
            const Cache* k=key+size_t(kv)*p.key_head_stride+size_t(id)*p.key_position_stride;
            for(int d=lane;d<p.key_dim;d+=32)dot=fmaf(q[d],float(k[d]),dot);
        }
        dot=mollm_cuda::detail::warp_reduce_sum(dot);
        if(lane==0)scores[slot]=active?dot*p.scale:-CUDART_INF_F;
    }
}

template<class Cache>
__global__ void selected_scores_gqa(const float* query,const Cache* key,
    const int32_t* indices,const float* mask,mollm_cuda::QsaSelectedAttentionParams p) {
    constexpr int Tile=16,Dim=256;
    const int row=blockIdx.x,kv=blockIdx.y,lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int head=kv*(p.heads/p.kv_heads)+warp;
    __shared__ float staged[Tile*Dim],scores[16*Tile];
    __shared__ int active[Tile];
    const float* q=query+size_t(row)*p.query_position_stride+size_t(head)*p.query_head_stride;
    float qr[8];
    #pragma unroll
    for(int j=0;j<8;++j)qr[j]=q[lane+32*j];
    const int32_t* ids=indices+size_t(row)*p.index_position_stride;
    for(int base=0;base<p.slots;base+=Tile) {
        if(threadIdx.x<Tile) {
            const int slot=base+threadIdx.x,id=slot<p.slots?ids[slot]:-1;
            active[threadIdx.x]=id>=0&&id<p.keys&&id<=p.past+row&&
                (!mask||mask[size_t(row)*p.mask_row_stride+size_t(id)*p.mask_column_stride]==0.f)?id:-1;
        }
        __syncthreads();
        for(int i=threadIdx.x;i<Tile*Dim;i+=blockDim.x) {
            const int id=active[i/Dim];
            staged[i]=id>=0?float(key[size_t(kv)*p.key_head_stride+size_t(id)*p.key_position_stride+i%Dim]):0.f;
        }
        __syncthreads();
        for(int token=0;token<Tile;++token) {
            float dot=0;
            #pragma unroll
            for(int j=0;j<8;++j)dot=fmaf(qr[j],staged[token*Dim+lane+32*j],dot);
            dot=mollm_cuda::detail::warp_reduce_sum(dot);
            if(lane==0)scores[warp*Tile+token]=active[token]>=0?dot*p.scale:-CUDART_INF_F;
        }
        __syncwarp();
        if(lane<Tile&&base+lane<p.slots)
            p.selected_scores[(size_t(row)*p.heads+head)*p.slots+base+lane]=scores[warp*Tile+lane];
        __syncthreads();
    }
}

__global__ void normalize_selected_scores(mollm_cuda::QsaSelectedAttentionParams p) {
    float* scores=p.selected_scores+(size_t(blockIdx.x)*p.heads+blockIdx.y)*p.slots;
    __shared__ float reduction[256];
    float maximum=-CUDART_INF_F;
    for(int s=threadIdx.x;s<p.slots;s+=256)maximum=fmaxf(maximum,scores[s]);
    maximum=mollm_cuda::detail::block_reduce_max<256>(maximum,reduction);
    float total=0;
    for(int s=threadIdx.x;s<p.slots;s+=256) {
        const float weight=scores[s]==-CUDART_INF_F?0.f:expf(scores[s]-maximum);
        scores[s]=weight;total+=weight;
    }
    total=mollm_cuda::detail::block_reduce_sum<256>(total,reduction);
    for(int s=threadIdx.x;s<p.slots;s+=256)scores[s]=total>0?scores[s]/total:0.f;
}

template<class Cache>
__global__ void selected_values_gqa(const Cache* value,const int32_t* indices,float* output,
    const float* mask,mollm_cuda::QsaSelectedAttentionParams p) {
    constexpr int Tile=16,Dim=256;
    const int row=blockIdx.x,kv=blockIdx.y,lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int head=kv*(p.heads/p.kv_heads)+warp;
    __shared__ float staged[Tile*Dim],weights[16*Tile];
    __shared__ int active[Tile];
    const int32_t* ids=indices+size_t(row)*p.index_position_stride;
    const float* scores=p.selected_scores+(size_t(row)*p.heads+head)*p.slots;
    float acc[8]={};
    for(int base=0;base<p.slots;base+=Tile) {
        if(threadIdx.x<Tile) {
            const int slot=base+threadIdx.x,id=slot<p.slots?ids[slot]:-1;
            active[threadIdx.x]=id>=0&&id<p.keys&&id<=p.past+row&&
                (!mask||mask[size_t(row)*p.mask_row_stride+size_t(id)*p.mask_column_stride]==0.f)?id:-1;
        }
        if(lane<Tile)weights[warp*Tile+lane]=base+lane<p.slots?scores[base+lane]:0.f;
        __syncthreads();
        for(int i=threadIdx.x;i<Tile*Dim;i+=blockDim.x) {
            const int id=active[i/Dim];
            staged[i]=id>=0?float(value[size_t(kv)*p.value_head_stride+size_t(id)*p.value_position_stride+i%Dim]):0.f;
        }
        __syncthreads();
        for(int token=0;token<Tile;++token)if(active[token]>=0) {
            const float weight=weights[warp*Tile+token];
            #pragma unroll
            for(int j=0;j<8;++j)acc[j]=fmaf(weight,staged[token*Dim+lane+32*j],acc[j]);
        }
        __syncthreads();
    }
    float* out=output+size_t(row)*p.output_position_stride+size_t(head)*p.output_head_stride;
    #pragma unroll
    for(int j=0;j<8;++j)out[lane+32*j]=acc[j];
}

template<class Cache>
__global__ void selected_values_feature_tiles(const Cache* value,const int32_t* indices,float* output,
    const float* mask,mollm_cuda::QsaSelectedAttentionParams p) {
    const int row=blockIdx.x,head=blockIdx.y,d=blockIdx.z*64+threadIdx.x;
    if(d>=p.value_dim)return;
    const int kv=head/(p.heads/p.kv_heads);
    const int32_t* ids=indices+size_t(row)*p.index_position_stride;
    const float* scores=p.selected_scores+(size_t(row)*p.heads+head)*p.slots;
    float sum=0;
    for(int s=0;s<p.slots;++s) {
        const int id=ids[s];
        if(id>=0&&id<p.keys&&id<=p.past+row&&
            (!mask||mask[size_t(row)*p.mask_row_stride+size_t(id)*p.mask_column_stride]==0.f))
            sum=fmaf(scores[s],float(value[size_t(kv)*p.value_head_stride+size_t(id)*p.value_position_stride+d]),sum);
    }
    output[size_t(row)*p.output_position_stride+size_t(head)*p.output_head_stride+d]=sum;
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
    const QsaSelectedAttentionParams& p,cudaStream_t stream,const float* mask) {
    if(!query||!key||!value||!ids||!output||p.heads<1||p.kv_heads<1||
       p.heads%p.kv_heads||p.queries<1||p.queries>65535||p.heads>65535||
       p.keys<1||p.past<0||p.past>p.keys||p.queries>p.keys-p.past||
       p.key_dim<1||p.value_dim<1||p.key_dim>4096||p.value_dim>4096||
       p.slots<1||p.slots>4096||p.partitions<1||p.partitions>32||
       ((p.partitions>1||p.reuse_gqa)&&!p.selected_scores)||!std::isfinite(p.scale)||p.scale<=0||
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
    if(mask&&(!p.mask_column_stride||!p.mask_row_stride||
        !fits(p.queries,p.keys,p.mask_row_stride,p.mask_column_stride,1)))return cudaErrorInvalidValue;
    const int group=p.heads/p.kv_heads;
    if(p.partitions>1||p.reuse_gqa) {
        if(size_t(p.slots)>limit/size_t(p.heads)/size_t(p.queries))return cudaErrorInvalidValue;
        const bool grouped=p.reuse_gqa&&p.key_dim==256&&p.value_dim==256&&group>=2&&group<=16;
        if(grouped) {
            const dim3 grid(p.queries,p.kv_heads);
            if(half)selected_scores_gqa<<<grid,group*32,0,stream>>>(query,static_cast<const __half*>(key),ids,mask,p);
            else selected_scores_gqa<<<grid,group*32,0,stream>>>(query,static_cast<const float*>(key),ids,mask,p);
        } else {
            const dim3 grid(p.queries,p.heads,p.partitions);
            if(half)selected_scores_partitioned<<<grid,256,0,stream>>>(query,static_cast<const __half*>(key),ids,mask,p);
            else selected_scores_partitioned<<<grid,256,0,stream>>>(query,static_cast<const float*>(key),ids,mask,p);
        }
        auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
        normalize_selected_scores<<<dim3(p.queries,p.heads),256,0,stream>>>(p);
        error=cudaGetLastError();if(error!=cudaSuccess)return error;
        if(grouped) {
            const dim3 grid(p.queries,p.kv_heads);
            if(half)selected_values_gqa<<<grid,group*32,0,stream>>>(static_cast<const __half*>(value),ids,output,mask,p);
            else selected_values_gqa<<<grid,group*32,0,stream>>>(static_cast<const float*>(value),ids,output,mask,p);
        } else {
            const dim3 grid(p.queries,p.heads,(p.value_dim+63)/64);
            if(half)selected_values_feature_tiles<<<grid,64,0,stream>>>(static_cast<const __half*>(value),ids,output,mask,p);
            else selected_values_feature_tiles<<<grid,64,0,stream>>>(static_cast<const float*>(value),ids,output,mask,p);
        }
        return cudaGetLastError();
    }
    const size_t shared=size_t(p.slots+256)*sizeof(float);
    const dim3 grid(p.queries,p.heads);
    if(half)selected_attention<<<grid,256,shared,stream>>>(query,
        static_cast<const __half*>(key),static_cast<const __half*>(value),ids,output,mask,p);
    else selected_attention<<<grid,256,shared,stream>>>(query,
        static_cast<const float*>(key),static_cast<const float*>(value),ids,output,mask,p);
    return cudaGetLastError();
}

cudaError_t mollm_cuda::launch_qsa_validate_causal_mask(const float* mask,uint32_t* invalid,
    int queries,int keys,int past,size_t cs,size_t rs,cudaStream_t stream) {
    const size_t limit=std::numeric_limits<size_t>::max()/sizeof(float);
    if(!mask||!invalid||queries<1||queries>65535||keys<1||keys>32768||past<0||past>keys||
       queries>keys-past||!cs||!rs||cs>limit/size_t(keys)||rs>limit/size_t(queries)||
       size_t(keys-1)*cs>limit-size_t(queries-1)*rs)return cudaErrorInvalidValue;
    auto error=cudaMemsetAsync(invalid,0,sizeof(uint32_t),stream);
    if(error!=cudaSuccess)return error;
    const size_t count=size_t(queries)*keys;
    validate_causal_mask<<<(count+255)/256,256,0,stream>>>(mask,invalid,queries,keys,past,cs,rs);
    return cudaGetLastError();
}
