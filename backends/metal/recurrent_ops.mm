#include "backends/metal/recurrent_ops.h"

#include "backends/metal/buffer_pool.h"
#include "backends/metal/command_context.h"
#include "backends/metal/pipeline_cache.h"
#include "kernels/metal/metal_common.h"

#include <algorithm>
#include <cmath>

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

void encode_1d(id<MTLComputeCommandEncoder> encoder,
               id<MTLComputePipelineState> pipeline, int elements) {
    [encoder setComputePipelineState:pipeline];
    NSUInteger threads = pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > 256)
        threads = 256;
    const MTLSize group_size = MTLSizeMake(threads, 1, 1);
    const MTLSize group_count = MTLSizeMake(
        (static_cast<NSUInteger>(elements) + threads - 1) / threads, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
}

void dispatch_grid_1d(id<MTLComputeCommandEncoder> encoder, int elements) {
    constexpr NSUInteger threads = 256;
    const MTLSize group_size = MTLSizeMake(threads, 1, 1);
    const MTLSize group_count = MTLSizeMake(
        (static_cast<NSUInteger>(elements) + threads - 1) / threads, 1, 1);
    [encoder dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
}

}  // namespace

MetalRecurrentOps::MetalRecurrentOps(
        MetalPipelineCache* pipelines, MetalBufferPool* pool,
        MetalCommandContext* commands)
    : pipelines_(pipelines), pool_(pool), commands_(commands) {}

bool MetalRecurrentOps::dispatch(
        const GraphNode& node, const std::vector<const Tensor*>& inputs,
        Tensor* output, id<MTLComputeCommandEncoder> encoder,
        std::string& profile_label) const {
    const OpParams& params = node.params;
    const OpType op = node.op_type;
    id<MTLComputeCommandEncoder> enc = encoder;
    auto dispatch_1d = [&](id<MTLComputePipelineState> pipeline, int elements) {
        encode_1d(encoder, pipeline, elements);
    };
    auto grid1d = [&](int elements) {
        dispatch_grid_1d(encoder, elements);
    };

    switch (op) {
    case OpType::RWKV_TOKEN_SHIFT: {
        const Tensor& X = *inputs[0];
        const Tensor& STATE = *inputs[1];
        Tensor& O = *output;
        RwkvTokenShiftParams p{};
        p.hidden = params.i32.size() > 0 ? params.i32[0] : (int)X.shape[0];
        p.seq = params.i32.size() > 1 ? params.i32[1] : (int)X.shape[1];
        p.real = params.i32.size() > 2 ? params.i32[2] : p.seq;
        if (p.real <= 0 || p.real > p.seq) p.real = p.seq;
        p.state_fp16 = STATE.prec == Precision::FP16;
        p.x_offset = element_offset(X);
        p.state_offset = element_offset(STATE);
        p.out_offset = element_offset(O);
        id<MTLComputePipelineState> ps =
            pipelines_->pipeline("rwkv_token_shift_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buffer_of(&X) offset:0 atIndex:0];
        [enc setBuffer:buffer_of(&STATE) offset:0 atIndex:1];
        [enc setBuffer:buffer_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        dispatch_1d(ps, p.hidden);
        return true;
    }

    case OpType::RWKV_MIX: {
        const Tensor& X = *inputs[0];
        const Tensor& SHIFT = *inputs[1];
        const Tensor& MIX = *inputs[2];
        Tensor& O = *output;
        RwkvMixParams p{};
        p.hidden = (int)MIX.nelements();
        p.total = (int)X.nelements();
        p.x_offset = element_offset(X);
        p.shift_offset = element_offset(SHIFT);
        p.mix_offset = element_offset(MIX);
        p.out_offset = element_offset(O);
        id<MTLComputePipelineState> ps = pipelines_->pipeline("rwkv_mix_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buffer_of(&X) offset:0 atIndex:0];
        [enc setBuffer:buffer_of(&SHIFT) offset:0 atIndex:1];
        [enc setBuffer:buffer_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc setBuffer:buffer_of(&MIX) offset:0 atIndex:4];
        dispatch_1d(ps, p.total);
        return true;
    }

    case OpType::RWKV_L2_NORM: {
        const Tensor& X = *inputs[0];
        Tensor& O = *output;
        RwkvL2NormParams p{};
        p.heads = params.i32.size() > 0 ? params.i32[0] : 0;
        p.head_size = params.i32.size() > 1 ? params.i32[1] : 0;
        p.groups = p.head_size > 0
            ? (int)(X.nelements() / p.head_size) : 0;
        p.x_offset = element_offset(X);
        p.out_offset = element_offset(O);
        p.eps = params.f32.size() > 0 ? params.f32[0] : 1e-12f;
        id<MTLComputePipelineState> ps =
            pipelines_->pipeline("rwkv_l2_norm_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buffer_of(&X) offset:0 atIndex:0];
        [enc setBuffer:buffer_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        return true;
    }

    case OpType::RWKV_POST: {
        const Tensor& RAW = *inputs[0];
        const Tensor& R = *inputs[1];
        const Tensor& K = *inputs[2];
        const Tensor& V = *inputs[3];
        const Tensor& RK = *inputs[4];
        const Tensor& W = *inputs[5];
        const Tensor& BIAS = *inputs[6];
        const Tensor& GATE = *inputs[7];
        Tensor& O = *output;
        RwkvPostParams p{};
        p.heads = params.i32.size() > 0 ? params.i32[0] : 0;
        p.head_size = params.i32.size() > 1 ? params.i32[1] : 0;
        p.groups = p.head_size > 0
            ? (int)(RAW.nelements() / p.head_size) : 0;
        p.raw_offset = element_offset(RAW);
        p.r_offset = element_offset(R);
        p.k_offset = element_offset(K);
        p.v_offset = element_offset(V);
        p.rk_offset = element_offset(RK);
        p.weight_offset = element_offset(W);
        p.bias_offset = element_offset(BIAS);
        p.gate_offset = element_offset(GATE);
        p.out_offset = element_offset(O);
        p.eps = params.f32.size() > 0 ? params.f32[0] : 64e-5f;
        id<MTLComputePipelineState> ps = pipelines_->pipeline("rwkv_post_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buffer_of(&RAW) offset:0 atIndex:0];
        [enc setBuffer:buffer_of(&R) offset:0 atIndex:1];
        [enc setBuffer:buffer_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc setBuffer:buffer_of(&K) offset:0 atIndex:4];
        [enc setBuffer:buffer_of(&V) offset:0 atIndex:5];
        [enc setBuffer:buffer_of(&RK) offset:0 atIndex:6];
        [enc setBuffer:buffer_of(&W) offset:0 atIndex:7];
        [enc setBuffer:buffer_of(&BIAS) offset:0 atIndex:8];
        [enc setBuffer:buffer_of(&GATE) offset:0 atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        return true;
    }

    case OpType::RWKV7: {
        const Tensor& R = *inputs[0];
        const Tensor& DECAY = *inputs[1];
        const Tensor& K = *inputs[2];
        const Tensor& V = *inputs[3];
        const Tensor& A = *inputs[4];
        const Tensor& B = *inputs[5];
        const Tensor& STATE = *inputs[6];
        Tensor& O = *output;
        Rwkv7Params p{};
        p.heads = params.i32.size() > 0 ? params.i32[0] : 0;
        p.head_size = params.i32.size() > 1 ? params.i32[1] : 0;
        p.seq = params.i32.size() > 2 ? params.i32[2] : (int)R.shape[1];
        p.real = params.i32.size() > 3 ? params.i32[3] : p.seq;
        if (p.real <= 0 || p.real > p.seq) p.real = p.seq;
        p.state_fp16 = STATE.prec == Precision::FP16;
        p.r_offset = element_offset(R);
        p.decay_offset = element_offset(DECAY);
        p.k_offset = element_offset(K);
        p.v_offset = element_offset(V);
        p.a_offset = element_offset(A);
        p.b_offset = element_offset(B);
        p.state_offset = element_offset(STATE);
        p.out_offset = element_offset(O);
        const bool use_h64_tgstate =
            !p.state_fp16 && p.head_size == 64 && p.real > 1;
        id<MTLComputePipelineState> ps = pipelines_->pipeline(
            use_h64_tgstate ? "rwkv7_h64_tgstate_fp32" : "rwkv7_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buffer_of(&R) offset:0 atIndex:0];
        [enc setBuffer:buffer_of(&DECAY) offset:0 atIndex:1];
        [enc setBuffer:buffer_of(&O) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc setBuffer:buffer_of(&K) offset:0 atIndex:4];
        [enc setBuffer:buffer_of(&V) offset:0 atIndex:5];
        [enc setBuffer:buffer_of(&A) offset:0 atIndex:6];
        [enc setBuffer:buffer_of(&B) offset:0 atIndex:7];
        [enc setBuffer:buffer_of(&STATE) offset:0 atIndex:8];
        if (use_h64_tgstate) {
            profile_label = "RWKV7_H64_TGSTATE";
            [enc dispatchThreadgroups:
                    MTLSizeMake((NSUInteger)p.heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        } else {
            const NSUInteger rows_per_group = 32;
            const NSUInteger groups_per_head =
                ((NSUInteger)p.head_size + rows_per_group - 1) / rows_per_group;
            [enc dispatchThreadgroups:
                    MTLSizeMake((NSUInteger)p.heads * groups_per_head, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(rows_per_group, 1, 1)];
        }
        return true;
    }

    case OpType::GATED_DELTANET_CONV_DECODE: {
        // Decode-only fusion: raw qkv + conv weight/state are appended to the
        // regular GDN inputs. One threadgroup owns one key head and all of its
        // value heads, so shared q/k convolution state is updated exactly once.
        const Tensor& QKV = *inputs[0]; const Tensor& Aa = *inputs[1];
        const Tensor& Bb  = *inputs[2]; const Tensor& Zz = *inputs[3];
        const Tensor& ALG = *inputs[4]; const Tensor& DTB = *inputs[5];
        const Tensor& NRM = *inputs[6]; const Tensor& ST  = *inputs[7];
        const Tensor& CW  = *inputs[8]; const Tensor& CS  = *inputs[9];
        Tensor& O = *output;
        GdnParams p{};
        p.num_heads   = params.i32.size()>0 ? params.i32[0] : 16;
        p.k_dim       = params.i32.size()>1 ? params.i32[1] : 128;
        p.v_dim       = params.i32.size()>2 ? params.i32[2] : 128;
        p.seq_len     = 1;
        const int gdn_flags = params.i32.size()>4 ? params.i32[4] : 1;
        p.use_qk_l2norm = gdn_flags & 1;
        p.output_gate_type = (gdn_flags & 2) != 0;
        p.conv_kernel = params.i32.size()>5 ? params.i32[5] : 4;
        p.n_real      = 1;
        p.num_v_heads = (params.i32.size()>7 && params.i32[7]>0)
                            ? params.i32[7] : p.num_heads;
        p.rms_eps     = params.f32.size()>0 ? params.f32[0] : 1e-6f;
        p.l2_eps      = params.f32.size()>1 ? params.f32[1] : 1e-6f;
        p.scale       = params.f32.size()>2 ? params.f32[2] : 0.f;
        if (p.scale == 0.f) p.scale = 1.f / std::sqrt((float)p.k_dim);
        p.qkv_offset = element_offset(QKV); p.a_offset = element_offset(Aa);
        p.b_offset = element_offset(Bb); p.z_offset = element_offset(Zz);
        p.Alog_offset = element_offset(ALG); p.dtb_offset = element_offset(DTB);
        p.norm_offset = element_offset(NRM); p.state_offset = element_offset(ST);
        p.out_offset = element_offset(O);
        p.a_row_stride = element_stride(Aa, 1);
        p.b_row_stride = element_stride(Bb, 1);
        p.z_row_stride = element_stride(Zz, 1);
        p.conv_weight_offset = element_offset(CW);
        p.conv_state_offset = element_offset(CS);

        const int repeat = p.num_v_heads / p.num_heads;
        const NSUInteger threads = (NSUInteger)(repeat * p.v_dim);
        const NSUInteger qk_nsg = ((NSUInteger)p.k_dim + 31) / 32;
        const NSUInteger v_nsg = ((NSUInteger)p.v_dim + 31) / 32;
        const NSUInteger smem_floats =
            (NSUInteger)(2*p.k_dim) + 2*qk_nsg +
            (NSUInteger)repeat*v_nsg;
        id<MTLComputePipelineState> ps =
            pipelines_->pipeline("gdn_conv_decode_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buffer_of(&QKV) offset:0 atIndex:0];
        [enc setBuffer:buffer_of(&Aa)  offset:0 atIndex:1];
        [enc setBuffer:buffer_of(&Bb)  offset:0 atIndex:2];
        [enc setBuffer:buffer_of(&O)   offset:0 atIndex:4];
        [enc setBuffer:buffer_of(&Zz)  offset:0 atIndex:5];
        [enc setBuffer:buffer_of(&ALG) offset:0 atIndex:6];
        [enc setBuffer:buffer_of(&DTB) offset:0 atIndex:7];
        [enc setBuffer:buffer_of(&NRM) offset:0 atIndex:8];
        [enc setBuffer:buffer_of(&ST)  offset:0 atIndex:9];
        [enc setBuffer:buffer_of(&CW)  offset:0 atIndex:10];
        [enc setBuffer:buffer_of(&CS)  offset:0 atIndex:11];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc setThreadgroupMemoryLength:
                smem_floats*sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:
                MTLSizeMake((NSUInteger)p.num_heads,1,1)
            threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
        profile_label = "GDN_CONV_DECODE";
        return true;
    }

    case OpType::GATED_DELTANET_DECODE:
    case OpType::GATED_DELTANET_PREFILL: {
        // Fused Gated Delta Rule + RMSNormGated. inputs (gdn.h contract):
        // [0]qkv [1]a [2]b [3]z [4]A_log [5]dt_bias [6]norm_w [7]state; out[0].
        // One threadgroup per value head, v_dim threads. decode=seq1; prefill
        // loops seq serially. qkv layout: decode [seq,dim](seq=1); prefill [dim,seq].
        const Tensor& QKV = *inputs[0]; const Tensor& Aa = *inputs[1];
        const Tensor& Bb  = *inputs[2]; const Tensor& Zz = *inputs[3];
        const Tensor& ALG = *inputs[4]; const Tensor& DTB = *inputs[5];
        const Tensor& NRM = *inputs[6]; const Tensor& ST  = *inputs[7];
        Tensor& O = *output;
        GdnParams p{};
        p.num_heads   = params.i32.size()>0 ? params.i32[0] : 16;
        p.k_dim       = params.i32.size()>1 ? params.i32[1] : 128;
        p.v_dim       = params.i32.size()>2 ? params.i32[2] : 128;
        p.seq_len     = params.i32.size()>3 ? params.i32[3] : 1;
        const int gdn_flags = params.i32.size()>4 ? params.i32[4] : 1;
        p.use_qk_l2norm = gdn_flags & 1;
        p.output_gate_type = (gdn_flags & 2) != 0;
        p.n_real      = params.i32.size()>6 ? params.i32[6] : 0;
        p.num_v_heads = (params.i32.size()>7 && params.i32[7]>0) ? params.i32[7] : p.num_heads;
        p.rms_eps     = params.f32.size()>0 ? params.f32[0] : 1e-6f;
        p.l2_eps      = params.f32.size()>1 ? params.f32[1] : 1e-6f;
        p.scale       = params.f32.size()>2 ? params.f32[2] : 0.f;
        if (p.scale == 0.f) p.scale = 1.f / std::sqrt((float)p.k_dim);
        p.qkv_offset = element_offset(QKV); p.a_offset = element_offset(Aa); p.b_offset = element_offset(Bb);
        p.z_offset = element_offset(Zz); p.Alog_offset = element_offset(ALG); p.dtb_offset = element_offset(DTB);
        p.norm_offset = element_offset(NRM); p.state_offset = element_offset(ST); p.out_offset = element_offset(O);
        p.a_row_stride = element_stride(Aa, 1);
        p.b_row_stride = element_stride(Bb, 1);
        p.z_row_stride = element_stride(Zz, 1);
        const bool prefill =
            op == OpType::GATED_DELTANET_PREFILL;
        const bool row_recurrence =
            prefill && p.k_dim == 128 &&
            p.v_dim > 0 && p.v_dim <= 1024;
        if (row_recurrence) {
            const size_t qk_bytes =
                (size_t)p.seq_len * (size_t)p.num_heads *
                (size_t)p.k_dim * sizeof(float);
            const size_t gate_bytes =
                (size_t)p.seq_len * (size_t)p.num_v_heads *
                sizeof(float);
            const size_t raw_bytes =
                (size_t)p.seq_len * (size_t)p.num_v_heads *
                (size_t)p.v_dim * sizeof(float);
            void* qn_h = pool_->acquire(qk_bytes);
            void* kn_h = pool_->acquire(qk_bytes);
            void* ge_h = pool_->acquire(gate_bytes);
            void* be_h = pool_->acquire(gate_bytes);
            void* raw_h = pool_->acquire(raw_bytes);
            id<MTLBuffer> qn = (__bridge id<MTLBuffer>)qn_h;
            id<MTLBuffer> kn = (__bridge id<MTLBuffer>)kn_h;
            id<MTLBuffer> ge = (__bridge id<MTLBuffer>)ge_h;
            id<MTLBuffer> be = (__bridge id<MTLBuffer>)be_h;
            id<MTLBuffer> raw = (__bridge id<MTLBuffer>)raw_h;
            commands_->pending_free.push_back({qn_h, qk_bytes});
            commands_->pending_free.push_back({kn_h, qk_bytes});
            commands_->pending_free.push_back({ge_h, gate_bytes});
            commands_->pending_free.push_back({be_h, gate_bytes});
            commands_->pending_free.push_back({raw_h, raw_bytes});

            id<MTLComputePipelineState> prep =
                pipelines_->pipeline("gdn_prepare_qk_f32");
            [enc setComputePipelineState:prep];
            [enc setBuffer:buffer_of(&QKV) offset:0 atIndex:0];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc setBuffer:qn offset:0 atIndex:10];
            [enc setBuffer:kn offset:0 atIndex:11];
            [enc setThreadgroupMemoryLength:8*sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:
                    MTLSizeMake((NSUInteger)p.seq_len *
                                    (NSUInteger)p.num_heads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128,1,1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            id<MTLComputePipelineState> gates =
                pipelines_->pipeline("gdn_prepare_gates_f32");
            [enc setComputePipelineState:gates];
            [enc setBuffer:buffer_of(&Aa) offset:0 atIndex:1];
            [enc setBuffer:buffer_of(&Bb) offset:0 atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc setBuffer:buffer_of(&ALG) offset:0 atIndex:6];
            [enc setBuffer:buffer_of(&DTB) offset:0 atIndex:7];
            [enc setBuffer:ge offset:0 atIndex:12];
            [enc setBuffer:be offset:0 atIndex:13];
            grid1d(p.seq_len * p.num_v_heads);

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            id<MTLComputePipelineState> recur =
                pipelines_->pipeline("gdn_recurrence_rows_f32");
            [enc setComputePipelineState:recur];
            [enc setBuffer:buffer_of(&QKV) offset:0 atIndex:0];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc setBuffer:buffer_of(&ST) offset:0 atIndex:9];
            [enc setBuffer:qn offset:0 atIndex:10];
            [enc setBuffer:kn offset:0 atIndex:11];
            [enc setBuffer:ge offset:0 atIndex:12];
            [enc setBuffer:be offset:0 atIndex:13];
            [enc setBuffer:raw offset:0 atIndex:14];
            [enc dispatchThreadgroups:
                    MTLSizeMake(((NSUInteger)p.v_dim + 3)/4,
                                (NSUInteger)p.num_v_heads, 1)
                threadsPerThreadgroup:MTLSizeMake(32,4,1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            id<MTLComputePipelineState> post =
                pipelines_->pipeline("gdn_post_f32");
            [enc setComputePipelineState:post];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc setBuffer:buffer_of(&O) offset:0 atIndex:4];
            [enc setBuffer:buffer_of(&Zz) offset:0 atIndex:5];
            [enc setBuffer:buffer_of(&NRM) offset:0 atIndex:8];
            [enc setBuffer:raw offset:0 atIndex:14];
            const NSUInteger post_threads = (NSUInteger)p.v_dim;
            const NSUInteger post_nsg = (post_threads + 31) / 32;
            [enc setThreadgroupMemoryLength:
                    post_nsg*sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:
                    MTLSizeMake((NSUInteger)p.seq_len *
                                    (NSUInteger)p.num_v_heads, 1, 1)
                threadsPerThreadgroup:
                    MTLSizeMake(post_threads,1,1)];
            return true;
        }
        const bool kparallel =
            prefill && p.v_dim > 0 && 4*p.v_dim <= 1024;
        const char* gk =
            kparallel
                ? "gdn_prefill_kparallel_f32"
                : (prefill ? "gdn_prefill_f32" : "gdn_decode_f32");
        id<MTLComputePipelineState> ps = pipelines_->pipeline(gk);
        [enc setComputePipelineState:ps];
        [enc setBuffer:buffer_of(&QKV) offset:0 atIndex:0];
        [enc setBuffer:buffer_of(&Aa)  offset:0 atIndex:1];
        [enc setBuffer:buffer_of(&Bb)  offset:0 atIndex:2];
        [enc setBuffer:buffer_of(&O)   offset:0 atIndex:4];
        [enc setBuffer:buffer_of(&Zz)  offset:0 atIndex:5];
        [enc setBuffer:buffer_of(&ALG) offset:0 atIndex:6];
        [enc setBuffer:buffer_of(&DTB) offset:0 atIndex:7];
        [enc setBuffer:buffer_of(&NRM) offset:0 atIndex:8];
        [enc setBuffer:buffer_of(&ST)  offset:0 atIndex:9];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        const NSUInteger threads =
            kparallel ? (NSUInteger)(4*p.v_dim)
                      : (NSUInteger)p.v_dim;
        // 4-way prefill: q/k + three SIMD reductions + four partial pairs,
        // delta, attn, and two gate scalars. Decode retains q/k + red[V].
        const NSUInteger nsg = (threads + 31) / 32;
        NSUInteger smem =
            (kparallel
                 ? (NSUInteger)(2*p.k_dim + 3*nsg + 10*p.v_dim + 2)
                 : (NSUInteger)(2*p.k_dim + p.v_dim)) *
            sizeof(float);
        [enc setThreadgroupMemoryLength:smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.num_v_heads,1,1)
            threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
        return true;
    }

    case OpType::SHORTCONV: {
        // Depth-wise causal conv1d + silu. inputs = {x, w, conv_state}; output.
        // conv_state is a persistent device buffer, read+written in-place. One
        // thread per group (groups is large: 6144/8192).
        const Tensor& X = *inputs[0];
        const Tensor& W = *inputs[1];
        const Tensor& STATE = *inputs[2];   // GPU buffer written in-place by kernel
        Tensor& O = *output;
        ShortConvParams p{};
        p.kernel_size = params.i32.size()>0 ? params.i32[0] : 4;
        p.groups = (int)X.shape[0];
        p.seq = (int)X.shape[1];
        p.n_real = params.i32.size()>1 ? params.i32[1] : p.seq;
        p.x_offset = element_offset(X);
        p.x_row_stride = element_stride(X, 1);
        p.w_offset = element_offset(W);
        p.state_offset = element_offset(STATE);
        p.out_offset = element_offset(O);
        id<MTLComputePipelineState> ps = pipelines_->pipeline("shortconv_f32");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buffer_of(&X) offset:0 atIndex:0];
        [enc setBuffer:buffer_of(&W) offset:0 atIndex:1];
        [enc setBuffer:buffer_of(&STATE) offset:0 atIndex:2];
        [enc setBuffer:buffer_of(&O) offset:0 atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        // One thread per group; bounds-checked threadgroups (M5 dispatchThreads bug).
        NSUInteger tg = 64;
        MTLSize tgc = MTLSizeMake(((NSUInteger)p.groups + tg - 1)/tg, 1, 1);
        [enc dispatchThreadgroups:tgc threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        return true;
    }

    default:
        return false;
    }
}
