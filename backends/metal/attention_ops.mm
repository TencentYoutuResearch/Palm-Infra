#include "backends/metal/attention_ops.h"

#include "backends/metal/buffer_pool.h"
#include "backends/metal/command_context.h"
#include "backends/metal/pipeline_cache.h"
#include "kernels/metal/metal_common.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace {

id<MTLBuffer> buffer_of(const Tensor* tensor) {
    return tensor && tensor->device.buffer
        ? (__bridge id<MTLBuffer>)tensor->device.buffer
        : nil;
}

int element_stride(const Tensor& tensor, int dimension) {
    return static_cast<int>(tensor.stride[dimension] / tensor.element_size());
}

uint element_offset(const Tensor& tensor) {
    return static_cast<uint>(tensor.device.offset / tensor.element_size());
}

}  // namespace

MetalAttentionOps::MetalAttentionOps(
        MetalPipelineCache* pipelines, MetalBufferPool* pool,
        MetalCommandContext* commands)
    : pipelines_(pipelines), pool_(pool), commands_(commands) {}

bool MetalAttentionOps::dispatch(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output, id<MTLComputeCommandEncoder> encoder,
        std::string& profile_label) const {
    const OpParams& params = node.params;
    id<MTLComputeCommandEncoder> enc = encoder;

    switch (node.op_type) {
    case OpType::SDPA:
    case OpType::SDPA_MLA: {
        // inputs = {Q, K_cur, V_cur, mask?, K_cache?, V_cache?}
        const Tensor& Q     = *inputs[0];
        const Tensor& K_cur = *inputs[1];
        const Tensor& V_cur = *inputs[2];
        const Tensor* mask    = (inputs.size()>3 && inputs[3] && inputs[3]->data) ? inputs[3] : nullptr;
        const Tensor* K_cache = (inputs.size()>4 && inputs[4] && inputs[4]->data) ? inputs[4] : nullptr;
        const Tensor* V_cache = (inputs.size()>5 && inputs[5] && inputs[5]->data) ? inputs[5] : nullptr;
        Tensor& out = *output;

        int kv_cache   = params.i32.size()>0 ? params.i32[0] : 2;
        int causal     = params.i32.size()>1 ? params.i32[1] : 1;
        int num_heads  = params.i32.size()>2 ? params.i32[2] : (int)Q.shape[2];
        int num_kv     = params.i32.size()>3 ? params.i32[3] : (int)K_cur.shape[2];
        int head_dim   = params.i32.size()>4 ? params.i32[4] : (int)Q.shape[0];
        int v_head_dim = params.i32.size()>5 ? params.i32[5] : (int)V_cur.shape[0];
        float scale    = params.f32.size()>0 ? params.f32[0] : 0.f;
        if (scale == 0.f) scale = 1.f / std::sqrt((float)head_dim);

        int src_seqlen = (int)Q.shape[1];
        int cur_seqlen = (int)K_cur.shape[1];
        // Cache metadata lives in the Shared buffer's host-visible header.
        int past = 0, max_seq = 0;
        if (kv_cache == 2 && K_cache && K_cache->data) {
            const auto* meta = reinterpret_cast<const uint64_t*>(K_cache->data);
            past    = (int)meta[0];  // current_seq_len
            max_seq = (int)meta[1];  // max_seq_len
        }
        std::string sdpa_profile_suffix;
        if (commands_->profile) {
            sdpa_profile_suffix =
                "[S=" + std::to_string(src_seqlen) +
                ",P=" + std::to_string(past) +
                ",H=" + std::to_string(num_heads) +
                ",HKV=" + std::to_string(num_kv) +
                ",DK=" + std::to_string(head_dim) +
                ",DV=" + std::to_string(v_head_dim) + "]";
            profile_label += sdpa_profile_suffix;
        }

        auto profile_sdpa_stage = [&](const char* label) {
            if (!commands_->profile) return;
            if (commands_->enc) {
                [commands_->enc endEncoding];
                commands_->enc = nil;
            }
            if (commands_->cmd) {
                [commands_->cmd commit];
                [commands_->cmd waitUntilCompleted];
                const double gpu_ms =
                    (commands_->cmd.GPUEndTime -
                     commands_->cmd.GPUStartTime) * 1000.0;
                auto& stat =
                    commands_->op_stats[std::string(label) + sdpa_profile_suffix];
                stat.gpu_ms += gpu_ms;
                stat.calls += 1;
            }
            commands_->cmd = [commands_->queue commandBuffer];
            commands_->enc = [commands_->cmd computeCommandEncoder];
            enc = commands_->enc;
        };

        int dst_seqlen = past + cur_seqlen;
        // Cache data begins 64 bytes past the buffer base (CacheMetadata header).
        // FP16 cache: element offset = (device.offset + 64) / 2.
        const size_t CACHE_HDR = 64;
        uint k_cache_eoff = (uint)((K_cache ? K_cache->device.offset : 0) + CACHE_HDR) / 2;
        uint v_cache_eoff = (uint)((V_cache ? V_cache->device.offset : 0) + CACHE_HDR) / 2;

        // 1) Append K_cur/V_cur (FP32) into the FP16 cache at position past+s.
        SdpaAppendKvParams ap{};
        if (kv_cache == 2 && K_cache && V_cache) {
            ap.num_kv_heads = num_kv;
            ap.cur_seqlen = cur_seqlen;
            ap.past_seqlen = past;
            ap.max_seq_len = max_seq;
            ap.k_dim = head_dim;
            ap.v_dim = v_head_dim;
            ap.k_cur_offset = element_offset(K_cur);
            ap.k_stride_head = element_stride(K_cur, 2);
            ap.k_stride_pos = element_stride(K_cur, 1);
            ap.k_cache_offset = k_cache_eoff;
            ap.v_cur_offset = element_offset(V_cur);
            ap.v_stride_head = element_stride(V_cur, 2);
            ap.v_stride_pos = element_stride(V_cur, 1);
            ap.v_cache_offset = v_cache_eoff;
        }
        if (kv_cache == 2 && K_cache && V_cache) {
            id<MTLComputePipelineState> aps =
                pipelines_->pipeline("sdpa_append_kv_f32_to_f16");
            [enc setComputePipelineState:aps];
            [enc setBuffer:buffer_of(&K_cur) offset:0 atIndex:0];
            [enc setBuffer:buffer_of(&V_cur) offset:0 atIndex:1];
            [enc setBuffer:buffer_of(K_cache) offset:0 atIndex:2];
            [enc setBytes:&ap length:sizeof(ap) atIndex:3];
            [enc setBuffer:buffer_of(V_cache) offset:0 atIndex:4];
            const NSUInteger tx = cur_seqlen == 1 ? 32 : 8;
            const NSUInteger ty = cur_seqlen == 1 ? 1 : 8;
            const NSUInteger tz = 4;
            [enc dispatchThreadgroups:
                     MTLSizeMake(
                         ((NSUInteger)std::max(head_dim, v_head_dim) + tx - 1) / tx,
                         ((NSUInteger)cur_seqlen + ty - 1) / ty,
                         ((NSUInteger)num_kv + tz - 1) / tz)
                threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
            // Attention immediately reads the cache regions written by the two
            // cache writes in this same compute encoder.
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            profile_sdpa_stage("SDPA.append");
        }

        // 2) Attention compute.
        SdpaParams sp{};
        sp.num_heads = num_heads;
        sp.num_kv_heads = num_kv;
        sp.head_dim = head_dim;
        sp.v_head_dim = v_head_dim;
        sp.src_seqlen = src_seqlen;
        sp.dst_seqlen = dst_seqlen;
        sp.past_seqlen = past;
        sp.max_seq_len = max_seq;
        sp.causal = causal;
        sp.scale = scale;
        sp.q_offset = element_offset(Q);
        sp.q_stride_pos = element_stride(Q, 1);
        sp.q_stride_head = element_stride(Q, 2);
        sp.k_cache_offset = k_cache_eoff;
        sp.v_cache_offset = v_cache_eoff;
        sp.o_offset = element_offset(out);
        sp.o_stride_pos = element_stride(out, 1);
        sp.o_stride_head = element_stride(out, 2);
        sp.has_mask = mask ? 1 : 0;
        sp.mask_offset = mask ? element_offset(*mask) : 0;
        sp.mask_stride_row = mask ? element_stride(*mask, 1) : 0;

        bool decode_path = (src_seqlen == 1);
        // Prefill SDPA routing:
        //   default -> sdpa_prefill_fa2_f32 (flash attention: query-split,
        //              direct-global K/V MMA, threadgroup-O elementwise rescale).
        // FA2 requires DK/DV % 8 and <= 256. C=64 and PV is padded to
        // 64, so both 8x8 QK and PV stripes divide evenly across NSG=8.
        constexpr bool use_simple = false;
        const int FA2_NSG = 8;
        const int FA2_Q =
            src_seqlen >= 16 &&
                    head_dim <= 192 &&
                    v_head_dim <= 128
                ? 16
                : 8;
        // fa2 declares function constants without defaults, so it can ONLY be built
        // via the specialized (pipeline_fa2) path — never the plain name-keyed
        // pipeline. Probe the specialized pipeline directly as the guard.
        id<MTLComputePipelineState> fa2_ps = nil;
        bool fa2_pre = !decode_path && !use_simple
            && (head_dim % 8 == 0) && (v_head_dim % 8 == 0)
            && head_dim <= 256 && v_head_dim <= 256;
        if (fa2_pre)
            fa2_ps = pipelines_->fa2(
                head_dim, v_head_dim, FA2_NSG, FA2_Q);
        bool fa2_ok = (fa2_ps != nil);

        const bool decode_192_128 = decode_path && head_dim == 192 && v_head_dim == 128
            && std::getenv("MOLLM_METAL_SDPA_DECODE_GENERIC") == nullptr;
        const bool decode_192_128_multi = decode_192_128 && dst_seqlen >= 768;
        const bool decode_128_128 = decode_path
            && head_dim == 128 && v_head_dim == 128
            && std::getenv("MOLLM_METAL_SDPA_DECODE_GENERIC") == nullptr;
        const bool decode_256_256 = decode_path
            && head_dim == 256 && v_head_dim == 256
            && std::getenv("MOLLM_METAL_SDPA_DECODE_GENERIC") == nullptr;
        const char* kname = decode_192_128_multi ? "sdpa_decode_192_128_partial_f32"
                           : decode_192_128 ? "sdpa_decode_192_128_f32"
                           : decode_128_128 ? "sdpa_decode_128_128_f32"
                           : decode_256_256 ? "sdpa_decode_256_256_f32"
                           : decode_path ? "sdpa_decode_f32"
                           : "sdpa_prefill_f32";
        // FA2 uses its DK/DV/NSG-specialized pipeline; all other paths use
        // name-keyed pipelines.
        id<MTLComputePipelineState> ps =
            fa2_ok ? fa2_ps : pipelines_->pipeline(kname);
        [enc setComputePipelineState:ps];
        [enc setBuffer:buffer_of(&Q) offset:0 atIndex:0];
        [enc setBuffer:(K_cache?buffer_of(K_cache):buffer_of(&K_cur)) offset:0 atIndex:1];
        [enc setBuffer:(V_cache?buffer_of(V_cache):buffer_of(&V_cur)) offset:0 atIndex:2];
        [enc setBuffer:buffer_of(&out) offset:0 atIndex:4];
        // Buffer index 5 (mask) must always be bound; use Q as a dummy if no mask.
        [enc setBuffer:(mask?buffer_of(mask):buffer_of(&Q)) offset:0 atIndex:5];
        [enc setBytes:&sp length:sizeof(sp) atIndex:3];
        if (decode_192_128_multi) {
            constexpr int nparts = 32;
            const size_t partial_bytes =
                (size_t)num_heads * nparts * (128 + 2) * sizeof(float);
            void* partial_h = pool_->acquire(partial_bytes);
            id<MTLBuffer> partial = (__bridge id<MTLBuffer>)partial_h;
            commands_->pending_free.push_back({partial_h, partial_bytes});
            [enc setBuffer:partial offset:0 atIndex:7];
            [enc setBytes:&nparts length:sizeof(nparts) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(nparts,(NSUInteger)num_heads,1)
                threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:
                pipelines_->pipeline("sdpa_decode_192_128_reduce_f32")];
            [enc setBuffer:partial offset:0 atIndex:7];
            [enc setBuffer:buffer_of(&out) offset:0 atIndex:4];
            [enc setBytes:&sp length:sizeof(sp) atIndex:3];
            [enc setBytes:&nparts length:sizeof(nparts) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)num_heads,1,1)
                threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        } else if (decode_path) {
            // The fused 192/128 path uses eight SIMD groups; the generic path
            // uses 256 threads to split the score and output loops.
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)num_heads,1,1)
                threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        } else if (fa2_ok) {
            // Flash attention: one threadgroup (NSG simdgroups, 32*NSG threads) per
            // (query-tile=Q, head). Threadgroup memory:
            //   sq[Q*DK] half + so[Q*PV] float + ss[Q*SH] float,
            //   C=64, SH=C+40 (bank-conflict padding), and PV=PAD(DV,64).
            const int PV = ((v_head_dim + 63) / 64) * 64;
            const int SH = 64 + 40;
            size_t tg_bytes = (size_t)FA2_Q * head_dim * 2   // sq (half)
                            + (size_t)FA2_Q * PV * 4         // so (float)
                            + (size_t)FA2_Q * SH * 4;        // ss (float)
            [enc setThreadgroupMemoryLength:tg_bytes atIndex:0];
            NSUInteger q_tiles = ((NSUInteger)src_seqlen + FA2_Q - 1) / FA2_Q;
            [enc dispatchThreadgroups:MTLSizeMake(q_tiles,(NSUInteger)num_heads,1)
                threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)FA2_NSG, 1)];
        } else {
            // One SIMD group (32 lanes) per (query pos, head); 4 groups/tg.
            const NSUInteger sg_per_tg = 4;
            NSUInteger nq = (NSUInteger)num_heads * (NSUInteger)src_seqlen;
            NSUInteger tgc = (nq + sg_per_tg - 1) / sg_per_tg;
            [enc dispatchThreadgroups:MTLSizeMake(tgc,1,1)
                threadsPerThreadgroup:MTLSizeMake(sg_per_tg*32,1,1)];
        }
        return true;
    }

    default:
        return false;
    }
}
