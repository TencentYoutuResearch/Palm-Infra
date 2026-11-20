#include "kernels/cuda/qsa.h"
#include "kernels/cuda/reduction.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
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

// Supports any whole-warp GQA block size (2..16 warps), including 12.
// No sorted/unique-ID assumption: find the last potentially contributing slot.
__device__ int selected_extent(const int32_t* ids, int slots, int keys, int position,
    float* warp_max) {
    float last = 0;
    for (int s = threadIdx.x; s < slots; s += blockDim.x) {
        const int id = ids[s];
        if (id >= 0 && id < keys && id <= position) last = float(s + 1);
    }
    last = mollm_cuda::detail::warp_reduce_max(last);
    if (threadIdx.x % 32 == 0) warp_max[threadIdx.x / 32] = last;
    __syncthreads();
    if (threadIdx.x == 0) {
        float end = 0;
        for (int w = 0; w < int(blockDim.x) / 32; ++w) end = fmaxf(end, warp_max[w]);
        warp_max[0] = end;
    }
    __syncthreads();
    return int(warp_max[0]);
}

// Copy full 16-byte packets when the caller's cache layout is aligned.
// Arbitrary strides retain scalar loads; every packet stays within one row.
template<int Tile, int Dim, class Cache>
__device__ void stage_selected_cache(const Cache* cache, Cache* staged,
    const int* active, size_t head_offset, size_t position_stride) {
    constexpr int Elements = 16 / sizeof(Cache);
    for (int packet = threadIdx.x; packet < Tile * Dim / Elements; packet += blockDim.x) {
        const int first = packet * Elements;
        const int id = active[first / Dim];
        uint4 bits = {0, 0, 0, 0};
        if (id >= 0) {
            const Cache* source = cache + head_offset + size_t(id) * position_stride + first % Dim;
            if ((reinterpret_cast<uintptr_t>(source) & 15) == 0) {
                bits = *reinterpret_cast<const uint4*>(source);
            } else {
                #pragma unroll
                for (int j = 0; j < Elements; ++j) staged[first + j] = source[j];
                continue;
            }
        }
        *reinterpret_cast<uint4*>(staged + first) = bits;
    }
}

template<class Cache>
__global__ void selected_scores_gqa(const float* query,const Cache* key,
    const int32_t* indices,const float* mask,mollm_cuda::QsaSelectedAttentionParams p) {
    constexpr int Tile=16,Dim=256;
    const int row=blockIdx.x,kv=blockIdx.y,lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int head=kv*(p.heads/p.kv_heads)+warp;
    __shared__ __align__(16) Cache staged[Tile*Dim];
    __shared__ float scores[16*Tile];
    __shared__ int active[Tile];
    const float* q=query+size_t(row)*p.query_position_stride+size_t(head)*p.query_head_stride;
    float qr[8];
    #pragma unroll
    for(int j=0;j<8;++j)qr[j]=q[lane+32*j];
    const int32_t* ids=indices+size_t(row)*p.index_position_stride;
    __shared__ float warp_max[16];
    const int end=selected_extent(ids,p.slots,p.keys,p.past+row,warp_max);
    float* out=p.selected_scores+(size_t(row)*p.heads+head)*p.slots;
    for(int s=end+lane;s<p.slots;s+=32)out[s]=-CUDART_INF_F;
    for(int base=0;base<end;base+=Tile) {
        if(threadIdx.x<Tile) {
            const int slot=base+threadIdx.x,id=slot<p.slots?ids[slot]:-1;
            active[threadIdx.x]=id>=0&&id<p.keys&&id<=p.past+row&&
                (!mask||mask[size_t(row)*p.mask_row_stride+size_t(id)*p.mask_column_stride]==0.f)?id:-1;
        }
        __syncthreads();
        stage_selected_cache<Tile, Dim>(key, staged, active,
            size_t(kv) * p.key_head_stride, p.key_position_stride);
        __syncthreads();
        for(int token=0;token<Tile;++token) {
            float dot=0;
            #pragma unroll
            for(int j=0;j<8;++j)dot=fmaf(qr[j],float(staged[token*Dim+lane+32*j]),dot);
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
    __shared__ __align__(16) Cache staged[Tile*Dim];
    __shared__ float weights[16*Tile];
    __shared__ int active[Tile];
    const int32_t* ids=indices+size_t(row)*p.index_position_stride;
    const float* scores=p.selected_scores+(size_t(row)*p.heads+head)*p.slots;
    float acc[8]={};
    __shared__ float warp_max[16];
    const int end=selected_extent(ids,p.slots,p.keys,p.past+row,warp_max);
    for(int base=0;base<end;base+=Tile) {
        if(threadIdx.x<Tile) {
            const int slot=base+threadIdx.x,id=slot<p.slots?ids[slot]:-1;
            active[threadIdx.x]=id>=0&&id<p.keys&&id<=p.past+row&&
                (!mask||mask[size_t(row)*p.mask_row_stride+size_t(id)*p.mask_column_stride]==0.f)?id:-1;
        }
        if(lane<Tile)weights[warp*Tile+lane]=base+lane<p.slots?scores[base+lane]:0.f;
        __syncthreads();
        stage_selected_cache<Tile, Dim>(value, staged, active,
            size_t(kv) * p.value_head_stride, p.value_position_stride);
        __syncthreads();
        for(int token=0;token<Tile;++token)if(active[token]>=0) {
            const float weight=weights[warp*Tile+token];
            #pragma unroll
            for(int j=0;j<8;++j)acc[j]=fmaf(weight,float(staged[token*Dim+lane+32*j]),acc[j]);
        }
        __syncthreads();
    }
    float* out=output+size_t(row)*p.output_position_stride+size_t(head)*p.output_head_stride;
    #pragma unroll
    for(int j=0;j<8;++j)out[lane+32*j]=acc[j];
}

// Materialize selected V once per KV head with a larger, independent grid.
// The following PV kernel retains the exact ascending-slot FMA order.
__global__ void gather_selected_values_fp16(const __half* value, const int32_t* indices,
    const float* mask, __half* gathered, int32_t* active,
    mollm_cuda::QsaSelectedAttentionParams p) {
    const size_t flat = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(p.kv_heads) * p.slots * p.value_dim;
    if (flat >= total) return;
    const int row = blockIdx.y;
    const int d = flat % p.value_dim;
    const int slot = (flat / p.value_dim) % p.slots;
    const int kv = flat / (size_t(p.slots) * p.value_dim);
    const int id = indices[size_t(row) * p.index_position_stride + slot];
    const bool valid = id >= 0 && id < p.keys && id <= p.past + row &&
        (!mask || mask[size_t(row) * p.mask_row_stride + size_t(id) * p.mask_column_stride] == 0.f);
    // Validity is independent of KV head/feature: one writer per row/slot.
    if (kv == 0 && d == 0) active[size_t(row) * p.slots + slot] = valid;
    gathered[size_t(row) * total + flat] = valid
        ? value[size_t(kv) * p.value_head_stride + size_t(id) * p.value_position_stride + d]
        : __half{};
}

__global__ void selected_values_gathered_fp16(const __half* gathered, const int32_t* active,
    float* output, mollm_cuda::QsaSelectedAttentionParams p) {
    const int row = blockIdx.x, head = blockIdx.y, d = blockIdx.z * 64 + threadIdx.x;
    // Prove a contiguous active prefix once per CTA. This removes the
    // active-flag dependency from each value load while preserving ordered FMA.
    // Arbitrary masks/indices with holes retain the per-slot validity checks.
    int first_invalid = p.slots, last_valid = -1;
    for (int slot = threadIdx.x; slot < p.slots; slot += 64) {
        if (active[size_t(row) * p.slots + slot])
            last_valid = slot;
        else
            first_invalid = min(first_invalid, slot);
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    first_invalid = __reduce_min_sync(0xffffffffu, first_invalid);
    last_valid = __reduce_max_sync(0xffffffffu, last_valid);
#else
    for (int offset = 16; offset; offset /= 2) {
        first_invalid = min(first_invalid, __shfl_down_sync(0xffffffffu, first_invalid, offset));
        last_valid = max(last_valid, __shfl_down_sync(0xffffffffu, last_valid, offset));
    }
#endif
    __shared__ int first[2], last[2];
    if ((threadIdx.x & 31) == 0) {
        first[threadIdx.x / 32] = first_invalid;
        last[threadIdx.x / 32] = last_valid;
    }
    __syncthreads();
    const bool prefix = min(first[0], first[1]) > max(last[0], last[1]);
    const int limit = max(last[0], last[1]) + 1;
    // Every lane participates in proof/barrier, including odd feature tails.
    if (d >= p.value_dim) return;
    const int kv = head / (p.heads / p.kv_heads);
    const float* scores = p.selected_scores + (size_t(row) * p.heads + head) * p.slots;
    const __half* selected = gathered + (size_t(row) * p.kv_heads + kv) * p.slots * p.value_dim;
    float sum = 0;
    if (prefix) {
        int base = 0;
        for (; base + 8 <= limit; base += 8) {
            float weights[8], values[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                weights[j] = scores[base + j];
                values[j] = float(selected[size_t(base + j) * p.value_dim + d]);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) sum = fmaf(weights[j], values[j], sum);
        }
        for (int slot = base; slot < limit; ++slot)
            sum = fmaf(scores[slot], float(selected[size_t(slot) * p.value_dim + d]), sum);
    } else {
        for (int base = 0; base < p.slots; base += 8) {
            float weights[8], values[8];
            bool use[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int slot = base + j;
                use[j] = slot < p.slots && active[size_t(row) * p.slots + slot];
                weights[j] = use[j] ? scores[slot] : 0.f;
                values[j] = use[j] ? float(selected[size_t(slot) * p.value_dim + d]) : 0.f;
            }
            // Zero weights must still participate: 0*NaN is not an inactive slot.
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (use[j]) sum = fmaf(weights[j], values[j], sum);
        }
    }
    output[size_t(row) * p.output_position_stride + size_t(head) * p.output_head_stride + d] = sum;
}

template<class Cache>
__global__ void selected_values_prefetched(const Cache* value, const int32_t* indices,
    float* output, const float* mask, mollm_cuda::QsaSelectedAttentionParams p) {
    const int row = blockIdx.x, head = blockIdx.y, d = blockIdx.z * 64 + threadIdx.x;
    indices += size_t(row) * p.index_position_stride;
    if (mask) mask += size_t(row) * p.mask_row_stride;
    // Find a conservative last valid slot, independent of selection ordering
    // or uniqueness. Padding, out-of-range and future IDs cannot contribute.
    // All lanes must participate even for an odd feature tail.
    float last = 0;
    for (int s = threadIdx.x; s < p.slots; s += 64) {
        const int id = indices[s];
        if (id >= 0 && id < p.keys && id <= p.past + row) last = float(s + 1);
    }
    __shared__ float reduction[64];
    const int end = int(mollm_cuda::detail::block_reduce_max<64>(last, reduction));
    if (d >= p.value_dim) return;
    const int kv = head / (p.heads / p.kv_heads);
    const float* scores = p.selected_scores + (size_t(row) * p.heads + head) * p.slots;
    float sum = 0;
    // Stage independent loads before consuming them. The original scalar loop
    // serializes id -> mask -> V -> FMA at every slot, with few resident warps
    // during one-token decode. Keep the exact ascending-slot FMA order.
    for (int base = 0; base < end; base += 8) {
        float weights[8], values[8];
        bool active[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int slot = base + j;
            const int id = slot < end ? indices[slot] : -1;
            active[j] = id >= 0 && id < p.keys && id <= p.past + row &&
                (!mask || mask[size_t(id) * p.mask_column_stride] == 0.f);
            weights[j] = active[j] ? scores[slot] : 0.f;
            values[j] = active[j] ? float(value[size_t(kv) * p.value_head_stride +
                size_t(id) * p.value_position_stride + d]) : 0.f;
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            if (active[j]) sum = fmaf(weights[j], values[j], sum);
    }
    output[size_t(row) * p.output_position_stride + size_t(head) * p.output_head_stride + d] = sum;
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

__global__ void qsa_all_visible_indices(int32_t* output, int keys, int past,
    int slots, size_t stride) {
    const int row = blockIdx.x, visible = min(keys, past + row + 1);
    for (int s = threadIdx.x; s < slots; s += blockDim.x)
        output[size_t(row) * stride + s] = s < visible ? s : -1;
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

template<int MergeItems>
__global__ void qsa_streaming_topk(const float* scores,int32_t* output,float* candidates,
    int32_t* candidate_ids,int keys,int past,int budget,int ratio,size_t ss,size_t is) {
    using TileSort=cub::BlockRadixSort<float,256,32,int>;
    using MergeSort=cub::BlockRadixSort<float,256,MergeItems,int>;
    using TokenSort=cub::BlockRadixSort<int,256,16>;
    union Storage {
        typename TileSort::TempStorage tile;
        typename MergeSort::TempStorage merge;
        typename TokenSort::TempStorage token;
        int tokens[4096];
    };
    static_assert(sizeof(Storage)<=48*1024,"QSA streaming sort exceeds shared budget");
    __shared__ Storage storage;
    const int row=blockIdx.x,visible=min(keys,past+row+1),complete=visible/ratio;
    const int keep=budget/ratio,top=min(keep,complete),count=top*ratio+visible%ratio;
    float* values=candidates+size_t(row)*2*keep;
    int32_t* groups=candidate_ids+size_t(row)*2*keep;
    for(int i=threadIdx.x;i<keep;i+=256){values[i]=-CUDART_INF_F;groups[i]=-1;}
    __syncthreads();
    for(int base=0;base<complete;base+=8192) {
        float tile_values[32];int tile_ids[32];
        #pragma unroll
        for(int i=0;i<32;++i) {
            const int id=base+threadIdx.x*32+i;
            tile_values[i]=id<complete?scores[size_t(row)*ss+id]:-CUDART_INF_F;
            tile_ids[i]=id;
        }
        TileSort(storage.tile).SortDescending(tile_values,tile_ids);
        __syncthreads();
        #pragma unroll
        for(int i=0;i<32;++i) {
            const int rank=threadIdx.x*32+i;
            if(rank<keep){values[keep+rank]=tile_values[i];groups[keep+rank]=tile_ids[i];}
        }
        __syncthreads();
        float merged[MergeItems];int ids[MergeItems];
        #pragma unroll
        for(int i=0;i<MergeItems;++i) {
            const int rank=threadIdx.x*MergeItems+i;
            merged[i]=rank<2*keep?values[rank]:-CUDART_INF_F;
            ids[i]=rank<2*keep?groups[rank]:-1;
        }
        // Prior chunks occupy the first half. Stable ties thus retain smaller
        // original group IDs, just like the monolithic stable radix sort.
        MergeSort(storage.merge).SortDescending(merged,ids);
        __syncthreads();
        #pragma unroll
        for(int i=0;i<MergeItems;++i) {
            const int rank=threadIdx.x*MergeItems+i;
            if(rank<keep){values[rank]=merged[i];groups[rank]=ids[i];}
        }
        __syncthreads();
    }
    for(int i=threadIdx.x;i<4096;i+=256)storage.tokens[i]=INT_MAX;
    __syncthreads();
    for(int i=threadIdx.x;i<top;i+=256)
        for(int j=0;j<ratio;++j)storage.tokens[i*ratio+j]=groups[i]*ratio+j;
    for(int i=threadIdx.x;i<visible%ratio;i+=256)storage.tokens[top*ratio+i]=complete*ratio+i;
    __syncthreads();
    int tokens[16];
    #pragma unroll
    for(int i=0;i<16;++i)tokens[i]=storage.tokens[threadIdx.x*16+i];
    __syncthreads();
    TokenSort(storage.token).Sort(tokens);
    #pragma unroll
    for(int i=0;i<16;++i) {
        const int rank=threadIdx.x*16+i;
        if(rank<budget+ratio-1)output[size_t(row)*is+rank]=rank<count?tokens[i]:-1;
    }
}

cudaError_t mollm_cuda::launch_qsa_indexer_update(const float* projection,const float* q_norm,
    const float* k_norm,const float* frequency,float* raw,float* pooled,float* query,
    float* scores,const QsaIndexerParams& p,cudaStream_t stream) {
    if(!projection||!q_norm||!k_norm||!frequency||!raw||!pooled||!query||!scores||
       p.heads<1||p.heads>16||p.dimension<1||p.dimension>256||p.rope_dim<2||
       p.rope_dim>p.dimension||p.rope_dim%2||p.ratio<1||p.ratio>16||
       p.queries<1||p.queries>65535||p.past<0||p.capacity<1||p.capacity>mollm_qsa::max_context||p.past>p.capacity||
       p.queries>p.capacity-p.past||p.position_offset<0||
       p.position_offset>INT_MAX-p.capacity||!std::isfinite(p.eps)||p.eps<=0||
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
    size_t index_stride,cudaStream_t stream,float* candidates,int32_t* candidate_ids,bool force_streaming) {
    if(!scores||!ids||queries<1||queries>65535||keys<1||past<0||past>keys||
       queries>keys-past||ratio<1||ratio>4096||budget<1||budget>4096||budget%ratio||
       budget>4096-ratio+1||keys>mollm_qsa::max_context||score_stride<size_t(keys/ratio)||
       index_stride<size_t(budget+ratio-1))return cudaErrorInvalidValue;
    const size_t limit=std::numeric_limits<size_t>::max()/sizeof(float);
    if(score_stride>limit/size_t(queries)||index_stride>limit/size_t(queries))
        return cudaErrorInvalidValue;
    if (keys <= budget && !force_streaming) {
        qsa_all_visible_indices<<<queries, 256, 0, stream>>>(
            ids, keys, past, budget + ratio - 1, index_stride);
        return cudaGetLastError();
    }
    const int groups=keys/ratio;
    if(groups>mollm_qsa::sort_tile_groups||force_streaming) {
        if(!candidates||!candidate_ids||size_t(2*(budget/ratio))>limit/size_t(queries))return cudaErrorInvalidValue;
        const int keep=budget/ratio;
        if(keep<=512)qsa_streaming_topk<4><<<queries,256,0,stream>>>(scores,ids,candidates,candidate_ids,keys,past,budget,ratio,score_stride,index_stride);
        else if(keep<=1024)qsa_streaming_topk<8><<<queries,256,0,stream>>>(scores,ids,candidates,candidate_ids,keys,past,budget,ratio,score_stride,index_stride);
        else if(keep<=2048)qsa_streaming_topk<16><<<queries,256,0,stream>>>(scores,ids,candidates,candidate_ids,keys,past,budget,ratio,score_stride,index_stride);
        else qsa_streaming_topk<32><<<queries,256,0,stream>>>(scores,ids,candidates,candidate_ids,keys,past,budget,ratio,score_stride,index_stride);
        return cudaGetLastError();
    }
    // Static shared storage remains below the standard48KiB launch budget for
    // these geometries; no device attribute mutation during graph capture.
    if(groups<=2048)qsa_block_topk<8><<<queries,256,0,stream>>>(scores,ids,keys,past,budget,ratio,score_stride,index_stride);
    else if(groups<=4096)qsa_block_topk<16><<<queries,256,0,stream>>>(scores,ids,keys,past,budget,ratio,score_stride,index_stride);
    else qsa_block_topk<32><<<queries,256,0,stream>>>(scores,ids,keys,past,budget,ratio,score_stride,index_stride);
    return cudaGetLastError();
}

bool mollm_cuda::qsa_gathered_fp16_workspace_bytes(const QsaSelectedAttentionParams& p,
    size_t& values_bytes, size_t& active_bytes) {
    values_bytes = active_bytes = 0;
    if (p.queries < 1 || p.queries > 8 || p.kv_heads < 1 || p.kv_heads > 65535 ||
        p.slots < 1 || p.slots > 4096 || p.value_dim < 1 || p.value_dim > 4096)
        return false;
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(float);
    if (size_t(p.slots) > limit / size_t(p.queries) / size_t(p.kv_heads) / size_t(p.value_dim))
        return false;
    const size_t elements = size_t(p.kv_heads) * p.slots * p.value_dim;
    // CUDA grid.x is bounded by 2^31-1.
    if ((elements + 255) / 256 > 0x7fffffffULL) return false;
    values_bytes = size_t(p.queries) * elements * sizeof(__half);
    active_bytes = size_t(p.queries) * p.slots * sizeof(int32_t);
    return true;
}

namespace {

cudaError_t launch_qsa_selected_impl(const float* query,
    const void* key, const void* value, bool half, const int32_t* ids, float* output,
    const mollm_cuda::QsaSelectedAttentionParams& p, cudaStream_t stream, const float* mask,
    void* gathered, size_t gathered_bytes, int32_t* active, size_t active_bytes) {
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
    if (gathered || active) {
        size_t required_values = 0, required_active = 0;
        if (!half || !gathered || !active || p.partitions <= 1 || p.reuse_gqa ||
            !mollm_cuda::qsa_gathered_fp16_workspace_bytes(p, required_values, required_active) ||
            gathered_bytes < required_values || active_bytes < required_active ||
            reinterpret_cast<uintptr_t>(gathered) % alignof(__half) ||
            reinterpret_cast<uintptr_t>(active) % alignof(int32_t))
            return cudaErrorInvalidValue;
    }
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
            if (p.queries <= 8) {
                if (gathered) {
                    const size_t elements = size_t(p.kv_heads) * p.slots * p.value_dim;
                    const dim3 gather_grid((elements + 255) / 256, p.queries);
                    gather_selected_values_fp16<<<gather_grid, 256, 0, stream>>>(
                        static_cast<const __half*>(value), ids, mask,
                        static_cast<__half*>(gathered), active, p);
                    error = cudaGetLastError();
                    if (error != cudaSuccess) return error;
                    selected_values_gathered_fp16<<<grid, 64, 0, stream>>>(
                        static_cast<const __half*>(gathered), active, output, p);
                } else if (half) {
                    selected_values_prefetched<<<grid, 64, 0, stream>>>(
                        static_cast<const __half*>(value), ids, output, mask, p);
                } else {
                    selected_values_prefetched<<<grid, 64, 0, stream>>>(
                        static_cast<const float*>(value), ids, output, mask, p);
                }
            } else {
                if (half)
                    selected_values_feature_tiles<<<grid, 64, 0, stream>>>(
                        static_cast<const __half*>(value), ids, output, mask, p);
                else
                    selected_values_feature_tiles<<<grid, 64, 0, stream>>>(
                        static_cast<const float*>(value), ids, output, mask, p);
            }
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

}  // namespace

cudaError_t mollm_cuda::launch_qsa_selected_attention(const float* query,
    const void* key, const void* value, bool half, const int32_t* ids, float* output,
    const QsaSelectedAttentionParams& p, cudaStream_t stream, const float* mask) {
    return launch_qsa_selected_impl(query, key, value, half, ids, output, p, stream, mask,
        nullptr, 0, nullptr, 0);
}

cudaError_t mollm_cuda::launch_qsa_selected_attention_gathered_fp16(const float* query,
    const void* key, const void* value, const int32_t* ids, float* output,
    const QsaSelectedAttentionParams& p, const QsaSelectedValueWorkspace& workspace,
    cudaStream_t stream, const float* mask) {
    if (!workspace.values || !workspace.active) return cudaErrorInvalidValue;
    return launch_qsa_selected_impl(query, key, value, true, ids, output, p, stream, mask,
        workspace.values, workspace.values_bytes, workspace.active, workspace.active_bytes);
}

cudaError_t mollm_cuda::launch_qsa_validate_causal_mask(const float* mask,uint32_t* invalid,
    int queries,int keys,int past,size_t cs,size_t rs,cudaStream_t stream) {
    const size_t limit=std::numeric_limits<size_t>::max()/sizeof(float);
    if(!mask||!invalid||queries<1||queries>65535||keys<1||keys>mollm_qsa::max_context||past<0||past>keys||
       queries>keys-past||!cs||!rs||cs>limit/size_t(keys)||rs>limit/size_t(queries)||
       size_t(keys-1)*cs>limit-size_t(queries-1)*rs)return cudaErrorInvalidValue;
    auto error=cudaMemsetAsync(invalid,0,sizeof(uint32_t),stream);
    if(error!=cudaSuccess)return error;
    const size_t count=size_t(queries)*keys;
    validate_causal_mask<<<(count+255)/256,256,0,stream>>>(mask,invalid,queries,keys,past,cs,rs);
    return cudaGetLastError();
}
