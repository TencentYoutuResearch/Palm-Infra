#include "backends/metal/matmul_ops.h"

#include "backends/metal/buffer_pool.h"
#include "backends/metal/command_context.h"
#include "backends/metal/dispatch_tuning.h"
#include "backends/metal/pipeline_cache.h"
#include "backends/metal/resource_store.h"
#include "kernels/metal/metal_common.h"

#include <algorithm>
#include <cassert>
#include <cstdio>

namespace {

id<MTLBuffer> buffer_of(const Tensor *tensor) {
  return tensor && tensor->device.buffer
             ? (__bridge id<MTLBuffer>)tensor->device.buffer
             : nil;
}

id<MTLBuffer> scales_buffer_of(const Tensor *tensor) {
  return tensor && tensor->device.scales_buffer
             ? (__bridge id<MTLBuffer>)tensor->device.scales_buffer
             : nil;
}

int element_stride(const Tensor &tensor, int dimension) {
  return static_cast<int>(tensor.stride[dimension] / tensor.element_size());
}

uint element_offset(const Tensor &tensor) {
  return static_cast<uint>(tensor.device.offset / tensor.element_size());
}

} // namespace

MetalMatmulOps::MetalMatmulOps(MetalPipelineCache *pipelines,
                               MetalBufferPool *pool,
                               MetalCommandContext *commands,
                               MetalResourceStore *resources)
    : pipelines_(pipelines), pool_(pool), commands_(commands),
      resources_(resources) {}

bool MetalMatmulOps::dispatch(const GraphNode &node,
                              const std::vector<const Tensor *> &inputs,
                              Tensor *output,
                              id<MTLComputeCommandEncoder> encoder,
                              bool has_tensor,
                              std::string &profile_label) const {
  const OpParams &params = node.params;
  id<MTLComputeCommandEncoder> enc = encoder;
  auto grid1d = [&](int n) {
    constexpr NSUInteger threads = 256;
    const MTLSize group_size = MTLSizeMake(threads, 1, 1);
    const MTLSize group_count =
        MTLSizeMake((static_cast<NSUInteger>(n) + threads - 1) / threads, 1, 1);
    [enc dispatchThreadgroups:group_count threadsPerThreadgroup:group_size];
  };

  switch (node.op_type) {
  case OpType::MATMUL:
  case OpType::GEMV_SPARSE_A: {
    const Tensor &A = *inputs[0];
    const Tensor &B = *inputs[1];
    Tensor &C = *output;
    // Mirror kernel_matmul_fp32 exactly: A is [K(inner), M], B is the weight
    // stored logically [N, K] with shape[0]=N, shape[1]=K and K contiguous
    // (row stride = K elements), C is [N(inner), M].
    MatmulParams p{};
    p.M = (int)A.shape[1];
    p.K = (int)A.shape[0];
    p.N = (int)B.shape[0];
    p.a_offset = element_offset(A);
    p.b_offset = element_offset(B);
    p.c_offset = element_offset(C);
    p.a_row_stride = element_stride(A, 1); // elements between rows of A (>= K)
    p.b_row_stride = (int)B.shape[1];      // K: elements between weight rows
    p.c_row_stride = element_stride(C, 1); // elements between rows of C (>= N)
    // fused activation: params.i32[0]=Activation (0 NONE, 1 SILU).
    int act = params.i32.size() > 0 ? params.i32[0] : 0;
    p.activation = (act >= 0 && act <= 4) ? act : 0;
    p.act_n_begin = params.i32.size() > 1 ? params.i32[1] : 0;
    p.act_n_len = params.i32.size() > 2 ? params.i32[2] : -1;
    auto cast_activation_to_f16 = [&]() -> id<MTLBuffer> {
      const size_t bytes = (size_t)p.M * (size_t)p.K * sizeof(uint16_t);
      void *handle = pool_->acquire(bytes);
      id<MTLBuffer> result = (__bridge id<MTLBuffer>)handle;
      commands_->pending_free.push_back({handle, bytes});
      id<MTLComputePipelineState> ps =
          pipelines_->pipeline("matmul_cast_f32_to_f16");
      [enc setComputePipelineState:ps];
      [enc setBuffer:buffer_of(&A) offset:0 atIndex:0];
      [enc setBuffer:result offset:0 atIndex:2];
      [enc setBytes:&p length:sizeof(p) atIndex:3];
      grid1d((p.M * p.K + 3) / 4);
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      return result;
    };
    // Decode graphs use M=1 throughout. Enable prefix submission only
    // after observing that invariant; large-M prefill benefits from one
    // command buffer and does not need CPU/GPU encoding overlap.
    if (p.M == 1)
      commands_->chunk_graph = true;

    if (p.M >= 2 && p.M <= 4 && p.N <= 512 && B.prec == Precision::FP16) {
      profile_label = "FP16_SMALL_M[M=" + std::to_string(p.M) +
                      ",N=" + std::to_string(p.N) +
                      ",K=" + std::to_string(p.K) + "]";
      MatmulParams small = p;
      small.b_offset = 0;
      const int nsg = std::min(mollm::metal::gemv_nsg_cap(), (p.K + 127) / 128);
      [enc setComputePipelineState:pipelines_->small_m(
                                       "gemv_small_m_f32a_f16b_f32c", p.M)];
      [enc setBuffer:buffer_of(&A) offset:0 atIndex:0];
      [enc setBuffer:buffer_of(&B) offset:B.device.offset atIndex:1];
      [enc setBuffer:buffer_of(&C) offset:0 atIndex:2];
      [enc setBytes:&small length:sizeof(small) atIndex:3];
      const NSUInteger groups = ((NSUInteger)p.N + nsg - 1) / nsg;
      [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32 * nsg, 1, 1)];
      break;
    }

    if (p.M >= 2 && p.M <= 4 && B.prec == Precision::INT8) {
      profile_label = "W8_SMALL_M[M=" + std::to_string(p.M) +
                      ",N=" + std::to_string(p.N) +
                      ",K=" + std::to_string(p.K) + "]";
      MatmulW8Params w{};
      w.M = p.M;
      w.N = p.N;
      w.K = p.K;
      w.a_offset = element_offset(A);
      w.c_offset = element_offset(C);
      w.a_row_stride = p.a_row_stride;
      w.c_row_stride = p.c_row_stride;
      w.activation = p.activation;
      w.act_n_begin = p.act_n_begin;
      w.act_n_len = p.act_n_len;
      w.group_size = (int)B.group_size;
      w.groups_per_row = (int)B.groups_per_row;
      const int nsg = std::min(mollm::metal::gemv_nsg_cap(), (p.K + 127) / 128);
      [enc setComputePipelineState:pipelines_->small_m(
                                       "gemv_w8_small_m_f32a_i8b_f32c", p.M)];
      [enc setBuffer:buffer_of(&A) offset:0 atIndex:0];
      [enc setBuffer:buffer_of(&B) offset:B.device.offset atIndex:1];
      [enc setBuffer:buffer_of(&C) offset:0 atIndex:2];
      [enc setBuffer:scales_buffer_of(&B)
              offset:B.device.scales_offset
             atIndex:4];
      [enc setBytes:&w length:sizeof(w) atIndex:3];
      const NSUInteger groups = ((NSUInteger)p.N + nsg - 1) / nsg;
      [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32 * nsg, 1, 1)];
      break;
    }

    if (p.M == 1 && B.prec == Precision::INT8) {
      profile_label = "W8_GEMV[N=" + std::to_string(p.N) +
                      ",K=" + std::to_string(p.K) + "]";
      // W8 decode: int8 weight x float activation, per-group weight scale.
      // Weight int8 + fp32 scales both live in the weight region; bind each
      // at its byte offset (scales offset relative to weight_base).
      MatmulW8Params w{};
      w.M = p.M;
      w.N = p.N;
      w.K = p.K;
      w.a_offset = element_offset(A);
      w.c_offset = element_offset(C);
      w.a_row_stride = p.a_row_stride;
      w.c_row_stride = p.c_row_stride;
      w.activation = p.activation;
      w.act_n_begin = p.act_n_begin;
      w.act_n_len = p.act_n_len;
      w.group_size = (int)B.group_size;
      w.groups_per_row = (int)B.groups_per_row;
      size_t scales_boff = B.device.scales_offset;
      const int NR0 = 2;
      const int NSG = std::min(mollm::metal::gemv_nsg_cap(), (p.K + 127) / 128);
      id<MTLComputePipelineState> ps =
          pipelines_->pipeline("gemv_w8_f32a_i8b_f32c");
      [enc setComputePipelineState:ps];
      [enc setBuffer:buffer_of(&A) offset:0 atIndex:0];
      [enc setBuffer:buffer_of(&B) offset:B.device.offset atIndex:1];
      [enc setBuffer:buffer_of(&C) offset:0 atIndex:2];
      [enc setBuffer:scales_buffer_of(&B) offset:scales_boff atIndex:4];
      [enc setBytes:&w length:sizeof(w) atIndex:3];
      [enc setThreadgroupMemoryLength:(NSUInteger)(NR0 * 32 * sizeof(float))
                              atIndex:0];
      NSUInteger tgc = ((NSUInteger)p.N + NR0 - 1) / NR0;
      [enc dispatchThreadgroups:MTLSizeMake(tgc, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)NSG, 1)];
      break;
    }
    if (p.M >= 2 && p.M <= 4 && B.prec == Precision::INT4) {
      profile_label = "W4_SMALL_M[M=" + std::to_string(p.M) +
                      ",N=" + std::to_string(p.N) +
                      ",K=" + std::to_string(p.K) + "]";
      MatmulW8Params w{};
      w.M = p.M;
      w.N = p.N;
      w.K = p.K;
      w.a_offset = element_offset(A);
      w.c_offset = element_offset(C);
      w.a_row_stride = p.a_row_stride;
      w.c_row_stride = p.c_row_stride;
      w.activation = p.activation;
      w.act_n_begin = p.act_n_begin;
      w.act_n_len = p.act_n_len;
      w.group_size = (int)B.group_size;
      w.groups_per_row = (int)B.groups_per_row;
      const size_t scales_boff = (size_t)p.N * (p.K / 2);
      const int NSG =
          std::min(mollm::metal::gemv_w4_nsg_cap(), (p.K / 2 + 63) / 64);
      id<MTLComputePipelineState> ps =
          pipelines_->small_m("gemv_w4_small_m_f32a_i4b_f32c", p.M);
      [enc setComputePipelineState:ps];
      [enc setBuffer:buffer_of(&A) offset:0 atIndex:0];
      [enc setBuffer:buffer_of(&B) offset:B.device.offset atIndex:1];
      [enc setBuffer:buffer_of(&C) offset:0 atIndex:2];
      [enc setBuffer:buffer_of(&B) offset:scales_boff atIndex:4];
      [enc setBytes:&w length:sizeof(w) atIndex:3];
      const NSUInteger rows_per_tg = (NSUInteger)std::max(1, NSG);
      const NSUInteger tgc = ((NSUInteger)p.N + rows_per_tg - 1) / rows_per_tg;
      [enc dispatchThreadgroups:MTLSizeMake(tgc, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32 * (NSUInteger)std::max(1, NSG),
                                            1, 1)];
      break;
    }
    if (p.M == 1 && B.prec == Precision::INT4) {
      profile_label = "W4_GEMV[N=" + std::to_string(p.N) +
                      ",K=" + std::to_string(p.K) + "]";
      // W4 decode: per-group symmetric int4 weight x float activation.
      MatmulW8Params w{};
      w.M = p.M;
      w.N = p.N;
      w.K = p.K;
      w.a_offset = element_offset(A);
      w.c_offset = element_offset(C);
      w.a_row_stride = p.a_row_stride;
      w.c_row_stride = p.c_row_stride;
      w.activation = p.activation;
      w.act_n_begin = p.act_n_begin;
      w.act_n_len = p.act_n_len;
      w.group_size = (int)B.group_size;
      w.groups_per_row = (int)B.groups_per_row;
      // Decoded W4 buffer layout: [ nibbles (N*K/2) | scales (N*gpr f32) ].
      size_t scales_boff = (size_t)p.N * (p.K / 2);
      const int NR0 = mollm::metal::gemv_w4_nr0(p.N, p.K);
      const int NSG =
          std::min(mollm::metal::gemv_w4_nsg_cap(), (p.K / 2 + 63) / 64);
      id<MTLComputePipelineState> ps = pipelines_->gemv_w4(NR0);
      [enc setComputePipelineState:ps];
      [enc setBuffer:buffer_of(&A) offset:0 atIndex:0];
      [enc setBuffer:buffer_of(&B) offset:B.device.offset atIndex:1];
      [enc setBuffer:buffer_of(&C) offset:0 atIndex:2];
      [enc setBuffer:buffer_of(&B) offset:scales_boff atIndex:4];
      [enc setBytes:&w length:sizeof(w) atIndex:3];
      [enc setThreadgroupMemoryLength:(NSUInteger)(NR0 * 32 * sizeof(float))
                              atIndex:0];
      NSUInteger rows_per_tg = (NSUInteger)NR0 * (NSUInteger)std::max(1, NSG);
      NSUInteger tgc = ((NSUInteger)p.N + rows_per_tg - 1) / rows_per_tg;
      [enc dispatchThreadgroups:MTLSizeMake(tgc, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(32 * (NSUInteger)std::max(1, NSG),
                                            1, 1)];
      break;
    }
    if (p.M == 1) {
      profile_label = "MATMUL_FP16_GEMV";
      // GEMV v2: each threadgroup owns NR0=2 output rows; NSG simdgroups
      // split K and reduce via shmem, so large-K matmuls (down_proj K=9728)
      // get up to 4x32 lanes.
      constexpr bool gemv_old = false;
      constexpr int NR0 = 2;
      // Bind weight at BYTE offset (64-bit) to avoid uint32 element-offset
      // overflow for late weights in the 8.8GB region (esp. lm_head).
      MatmulParams pv = p;
      pv.b_offset = 0;
      [enc setBuffer:buffer_of(&A) offset:0 atIndex:0];
      [enc setBuffer:buffer_of(&B) offset:B.device.offset atIndex:1];
      [enc setBuffer:buffer_of(&C) offset:0 atIndex:2];
      [enc setBytes:&pv length:sizeof(pv) atIndex:3];
      id<MTLComputePipelineState> gemv2_ps =
          gemv_old ? nil : pipelines_->gemv2(NR0);
      if (gemv2_ps) {
        [enc setComputePipelineState:gemv2_ps];
        // Eight SIMD groups give the best cross-model decode throughput
        // on M5 Pro. The environment override keeps this tunable for
        // future GPU families.
        int nsg = std::min(mollm::metal::gemv_nsg_cap(), (p.K + 127) / 128);
        if (nsg < 1)
          nsg = 1;
        [enc setThreadgroupMemoryLength:(NSUInteger)(NR0 * 32 * sizeof(float))
                                atIndex:0];
        NSUInteger tgcount = ((NSUInteger)p.N + NR0 - 1) / NR0;
        [enc dispatchThreadgroups:MTLSizeMake(tgcount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
      } else {
        id<MTLComputePipelineState> ps =
            pipelines_->pipeline("gemv_f32a_f16b_f32c");
        const NSUInteger rows_per_tg = 8; // 8*32 = 256 threads/tg
        [enc setComputePipelineState:ps];
        NSUInteger tgcount = ((NSUInteger)p.N + rows_per_tg - 1) / rows_per_tg;
        [enc dispatchThreadgroups:MTLSizeMake(tgcount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(rows_per_tg * 32, 1, 1)];
      }
    } else if (B.prec == Precision::INT8) {
      // W8 prefill GEMM. Two paths:
      //  - W8A8 (opt-in, per-channel weights only): quantize activations
      //    per-token to int8, run int8xint8->int32 MMA, dequant at store.
      //  - W8A16 (default): cast the activation matrix once, dequant int8
      //    weight->half during staging, and use FP16-input tensor MMA with
      //    FP32 accumulation. Requires the tensor path.
#ifdef MOLLM_METAL_TENSOR
      static const bool w8a8 = (getenv("MOLLM_METAL_W8A8") != nullptr);
      if (has_tensor && w8a8 && B.groups_per_row == 1) {
        profile_label = "MATMUL_W8A8_GEMM";
        // --- W8A8: quantize A -> int8 scratch, then int8 MMA ----------
        size_t a_i8_bytes = (size_t)p.M * (size_t)p.K; // [M,K] contiguous
        size_t sa_bytes = (size_t)p.M * sizeof(float); // scale_a[M]
        void *a_i8_h = pool_->acquire(a_i8_bytes);
        void *sa_h = pool_->acquire(sa_bytes);
        id<MTLBuffer> a_i8 = (__bridge id<MTLBuffer>)a_i8_h;
        id<MTLBuffer> sa = (__bridge id<MTLBuffer>)sa_h;
        commands_->pending_free.push_back({a_i8_h, a_i8_bytes});
        commands_->pending_free.push_back({sa_h, sa_bytes});

        // 1) per-token activation quantization.
        {
          QuantActParams q{};
          q.M = p.M;
          q.K = p.K;
          q.a_offset = A.device.offset / sizeof(float); // A bound at 0
          q.a_row_stride = p.a_row_stride;
          id<MTLComputePipelineState> qps =
              pipelines_->pipeline("quantize_act_i8");
          [enc setComputePipelineState:qps];
          [enc setBuffer:buffer_of(&A) offset:0 atIndex:0];
          [enc setBuffer:a_i8 offset:0 atIndex:2];
          [enc setBuffer:sa offset:0 atIndex:4];
          [enc setBytes:&q length:sizeof(q) atIndex:3];
          const NSUInteger nsg = 8; // 8*32 = 256 threads/row
          [enc setThreadgroupMemoryLength:nsg * sizeof(float) atIndex:0];
          [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.M, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
        }
        // 2) int8xint8->int32 GEMM with dequant at store.
        {
          MatmulW8A8Params w{};
          w.M = p.M;
          w.N = p.N;
          w.K = p.K;
          w.c_offset = element_offset(C);
          w.c_row_stride = p.c_row_stride;
          w.activation = p.activation;
          w.act_n_begin = p.act_n_begin;
          w.act_n_len = p.act_n_len;
          size_t scales_boff = B.device.scales_offset;
          id<MTLComputePipelineState> ps =
              pipelines_->pipeline("gemm_w8a8_i8a_i8b_f32c");
          [enc setComputePipelineState:ps];
          [enc setBuffer:a_i8 offset:0 atIndex:0];
          [enc setBuffer:buffer_of(&B) offset:B.device.offset atIndex:1];
          [enc setBuffer:buffer_of(&C) offset:0 atIndex:2];
          [enc setBytes:&w length:sizeof(w) atIndex:3];
          [enc setBuffer:sa offset:0 atIndex:4];
          [enc setBuffer:scales_buffer_of(&B) offset:scales_boff atIndex:5];
          // NRB=64 (M) x NRA=64 (N) tile / threadgroup, 128 threads;
          // int32 accumulators staged in 64*64*4 = 16KB threadgroup.
          [enc setThreadgroupMemoryLength:64 * 64 * sizeof(int32_t) atIndex:0];
          MTLSize tgc = MTLSizeMake(((NSUInteger)p.M + 63) / 64,
                                    ((NSUInteger)p.N + 63) / 64, 1);
          [enc dispatchThreadgroups:tgc
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        break;
      }
      if (has_tensor) {
        profile_label = "MATMUL_W8A16_GEMM";
        MatmulW8Params w{};
        w.M = p.M;
        w.N = p.N;
        w.K = p.K;
        w.a_offset = 0;
        w.c_offset = element_offset(C);
        w.a_row_stride = p.a_row_stride;
        w.c_row_stride = p.c_row_stride;
        w.activation = p.activation;
        w.act_n_begin = p.act_n_begin;
        w.act_n_len = p.act_n_len;
        w.group_size = (int)B.group_size;
        w.groups_per_row = (int)B.groups_per_row;
        size_t scales_boff = B.device.scales_offset;
        id<MTLBuffer> ah = cast_activation_to_f16();
        w.a_row_stride = p.K;
        // M64 improves occupancy for the smaller projection shapes.
        // Very large gate/up or down projections amortize the larger
        // M128 cooperative accumulator and issue fewer threadgroups.
        const bool use_m128 = p.K >= 2560 && (p.N >= 8192 || p.K >= 8192);
        id<MTLComputePipelineState> ps =
            pipelines_->pipeline(use_m128 ? "gemm_tensor_w8_f16a_i8b_f32c"
                                          : "gemm_tensor_w8_f16a_i8b_f32c_m64");
        [enc setComputePipelineState:ps];
        [enc setBuffer:ah offset:0 atIndex:0];
        [enc setBuffer:buffer_of(&B) offset:B.device.offset atIndex:1];
        [enc setBuffer:buffer_of(&C) offset:0 atIndex:2];
        [enc setBuffer:scales_buffer_of(&B) offset:scales_boff atIndex:4];
        [enc setBytes:&w length:sizeof(w) atIndex:3];
        [enc setThreadgroupMemoryLength:64 * 32 * sizeof(uint16_t) atIndex:0];
        const NSUInteger m_tile = use_m128 ? 128 : 64;
        MTLSize tgc = MTLSizeMake(((NSUInteger)p.M + m_tile - 1) / m_tile,
                                  ((NSUInteger)p.N + 63) / 64, 1);
        [enc dispatchThreadgroups:tgc
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        if (w.activation != 0 && w.act_n_len != 0) {
          [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
          id<MTLComputePipelineState> aps =
              pipelines_->pipeline("matmul_w8_activation_range_f32");
          [enc setComputePipelineState:aps];
          [enc setBuffer:buffer_of(&C) offset:0 atIndex:2];
          [enc setBytes:&w length:sizeof(w) atIndex:3];
          grid1d(p.M * p.N);
        }
      } else
#endif
      {
        fprintf(stderr,
                "MetalBackend: W8 GEMM requires tensor path (M5/A19+)\n");
        assert(false && "W8 GEMM needs tensor path");
      }
    } else if (B.prec == Precision::INT4) {
      // W4 prefill GEMM. W4A16 keeps activations in FP32 and unpacks
      // weights to half while staging for better numerical parity.
      // W4A8 quantizes activations and remains the throughput baseline.
#ifdef MOLLM_METAL_TENSOR
      if (has_tensor) {
        // Default balanced path: unactivated projections use half
        // tensor GEMM; fused gate/up projections use K64 activation
        // quantization for the entire output. K64 is closer to the CPU
        // reference than half staging while combining two K32 tensor
        // operations before each FP32 dequantization.
        const char *w4_mode = std::getenv("MOLLM_METAL_W4_PREFILL_MODE");
        const bool fast = w4_mode && std::strcmp(w4_mode, "fast") == 0;
        const bool accurate = w4_mode && std::strcmp(w4_mode, "accurate") == 0;
        const bool w4a16 = fast || (!accurate && p.activation == 0);
        if (w4a16) {
          profile_label = "MATMUL_W4A16_GEMM";
          if (commands_->profile) {
            profile_label += "[M=" + std::to_string(p.M) +
                             ",N=" + std::to_string(p.N) +
                             ",K=" + std::to_string(p.K) + "]";
          }
          MatmulW8Params w{};
          w.M = p.M;
          w.N = p.N;
          w.K = p.K;
          w.a_offset = 0;
          w.c_offset = element_offset(C);
          w.a_row_stride = p.a_row_stride;
          w.c_row_stride = p.c_row_stride;
          w.activation = p.activation;
          w.act_n_begin = p.act_n_begin;
          w.act_n_len = p.act_n_len;
          w.group_size = (int)B.group_size;
          w.groups_per_row = (int)B.groups_per_row;
          size_t scales_boff = (size_t)p.N * (p.K / 2);
          const bool use_m128 =
              std::min(p.N, p.K) >= 2560 && std::max(p.N, p.K) >= 4096;
          const bool specialize_g128 = w.group_size == 128 && p.K % 128 == 0;
          id<MTLComputePipelineState> ps =
              pipelines_->w4a16(use_m128, specialize_g128);
          [enc setComputePipelineState:ps];
          [enc setBuffer:buffer_of(&A) offset:A.device.offset atIndex:0];
          [enc setBuffer:buffer_of(&B) offset:B.device.offset atIndex:1];
          [enc setBuffer:buffer_of(&C) offset:0 atIndex:2];
          [enc setBytes:&w length:sizeof(w) atIndex:3];
          [enc setBuffer:buffer_of(&B) offset:scales_boff atIndex:4];
          [enc setThreadgroupMemoryLength:64 * 128 * sizeof(uint16_t)
                                  atIndex:0];
          MTLSize tgc = MTLSizeMake(((NSUInteger)p.M + (use_m128 ? 127 : 63)) /
                                        (use_m128 ? 128 : 64),
                                    ((NSUInteger)p.N + 63) / 64, 1);
          [enc dispatchThreadgroups:tgc
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
          if (w.activation != 0 && w.act_n_len != 0) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            id<MTLComputePipelineState> aps =
                pipelines_->pipeline("matmul_w8_activation_range_f32");
            [enc setComputePipelineState:aps];
            [enc setBuffer:buffer_of(&C) offset:0 atIndex:2];
            [enc setBytes:&w length:sizeof(w) atIndex:3];
            grid1d(p.M * p.N);
          }
          break;
        }
        size_t a_i8_bytes = (size_t)p.M * (size_t)p.K;
        const bool block32 = accurate;
        const bool block64 = !accurate && B.is_q4_g128_packed;
        profile_label = block32 ? "MATMUL_W4_BLOCK_GEMM"
                                : (block64 ? "MATMUL_W4_BLOCK64_GEMM"
                                           : "MATMUL_W4_GROUP128_GEMM");
        const int a_blocks =
            block32 ? (p.K + 31) / 32
                    : (block64 ? (p.K + 63) / 64 : (int)B.groups_per_row);
        size_t sa_bytes = (size_t)p.M * (size_t)a_blocks * sizeof(float);
        void *a_i8_h = pool_->acquire(a_i8_bytes);
        void *sa_h = pool_->acquire(sa_bytes);
        id<MTLBuffer> a_i8 = (__bridge id<MTLBuffer>)a_i8_h;
        id<MTLBuffer> sa = (__bridge id<MTLBuffer>)sa_h;
        commands_->pending_free.push_back({a_i8_h, a_i8_bytes});
        commands_->pending_free.push_back({sa_h, sa_bytes});

        // 1) per-token activation quantization -> int8 [M,K] + scale_a[M].
        {
          QuantActParams q{};
          q.M = p.M;
          q.K = p.K;
          q.a_offset = A.device.offset / sizeof(float);
          q.a_row_stride = p.a_row_stride;
          q.block_size = block32 ? 32 : (block64 ? 64 : (int)B.group_size);
          id<MTLComputePipelineState> qps = pipelines_->pipeline(
              block32 ? "quantize_act_i8_block32"
                      : (block64 ? "quantize_act_i8_block64"
                                 : "quantize_act_i8_blocks"));
          [enc setComputePipelineState:qps];
          [enc setBuffer:buffer_of(&A) offset:0 atIndex:0];
          [enc setBuffer:a_i8 offset:0 atIndex:2];
          [enc setBuffer:sa offset:0 atIndex:4];
          [enc setBytes:&q length:sizeof(q) atIndex:3];
          if (block32 || block64) {
            constexpr NSUInteger nsg = 4;
            const NSUInteger block_groups =
                ((NSUInteger)a_blocks + nsg - 1) / nsg;
            [enc dispatchThreadgroups:MTLSizeMake(
                                          (NSUInteger)p.M * block_groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
          } else {
            const NSUInteger nsg = 4;
            [enc setThreadgroupMemoryLength:nsg * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.M *
                                                      (NSUInteger)a_blocks,
                                                  1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
          }
        }
        // 2) int8 x per-group int4 GEMM with per-group dequant.
        {
          MatmulW4A8Params w{};
          w.M = p.M;
          w.N = p.N;
          w.K = p.K;
          w.c_offset = element_offset(C);
          w.c_row_stride = p.c_row_stride;
          w.activation = p.activation;
          w.act_n_begin = p.act_n_begin;
          w.act_n_len = p.act_n_len;
          w.group_size = (int)B.group_size;
          w.groups_per_row = (int)B.groups_per_row;
          char *native_ptr = (char *)B.q4_g128_data;
          void *native_buffer_handle = nullptr;
          size_t native_buffer_offset = 0;
          const bool native_bg128 =
              B.is_q4_g128_packed && native_ptr &&
              resources_->locate_weight(native_ptr, 1, native_buffer_handle,
                                        native_buffer_offset);
          // Decoded W4 buffer: [ nibbles (N*K/2) | scales (N*gpr f32) ].
          size_t scales_boff = (size_t)p.N * (p.K / 2);
          id<MTLComputePipelineState> ps = pipelines_->pipeline(
              block64 && native_bg128
                  ? (p.K <= 1024 ? "gemm_w4a8_block64_bg128_smallk_i8a_i4b_f32c"
                                 : "gemm_w4a8_block64_bg128_i8a_i4b_f32c")
                  : (block32 ? (native_bg128
                                    ? "gemm_w4a8_block32_bg128_i8a_i4b_f32c"
                                    : "gemm_w4a8_block32_i8a_i4b_f32c")
                             : (native_bg128 ? "gemm_w4a8_bg128_i8a_i4b_f32c"
                                             : "gemm_w4a8_i8a_i4b_f32c")));
          [enc setComputePipelineState:ps];
          [enc setBuffer:a_i8 offset:0 atIndex:0];
          if (native_bg128) {
            [enc setBuffer:(__bridge id<MTLBuffer>)native_buffer_handle
                    offset:native_buffer_offset
                   atIndex:1];
          } else {
            [enc setBuffer:buffer_of(&B) offset:B.device.offset atIndex:1];
          }
          [enc setBuffer:buffer_of(&C) offset:0 atIndex:2];
          [enc setBytes:&w length:sizeof(w) atIndex:3];
          [enc setBuffer:sa offset:0 atIndex:4];
          [enc setBuffer:buffer_of(&B) offset:scales_boff atIndex:5];
          const NSUInteger tile_m = 64;
          const NSUInteger tile_n = 16;
          const NSUInteger tg_mem =
              block64
                  ? (p.K <= 1024
                         ? 2 * tile_m * tile_n * sizeof(int32_t) + tile_n * 64
                         : tile_n * 64 / 2 + (tile_n + tile_m) * sizeof(float))
                  : 2 * tile_m * tile_n * sizeof(int32_t) +
                        (block32 ? tile_n * 32 : 0);
          [enc setThreadgroupMemoryLength:tg_mem atIndex:0];
          MTLSize tgc = MTLSizeMake(((NSUInteger)p.M + tile_m - 1) / tile_m,
                                    ((NSUInteger)w.N + tile_n - 1) / tile_n, 1);
          [enc dispatchThreadgroups:tgc
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
      } else
#endif
      {
        fprintf(stderr,
                "MetalBackend: W4 GEMM requires tensor path (M5/A19+)\n");
        assert(false && "W4 GEMM needs tensor path");
      }
    } else {
      // Tiled/tensor GEMM. Apply an optional fused graph activation as a
      // lightweight post-pass so activation-bearing FP16 nodes retain
      // the same high-throughput matrix path.
      // Weight buffer B bound at its 64-bit BYTE offset with
      // in-shader b_offset=0 to avoid uint32 element-offset overflow for
      // the 8.8GB weight region (incl. lm_head).
      // Default: 32x32 half-staged tile (acc[4]/sg). This is the OCCUPANCY
      // sweet spot on M5 Pro: larger tiles / more accumulators per
      // simdgroup (tested 64x32 -> 219 t/s, 32x64 acc[8] -> 226 t/s, TK=32
      // -> 357) all LOWER perf by reducing resident threadgroup count.
      // Apple GPUs favor many small threadgroups over deep register
      // blocking (opposite of NVIDIA).
      // Weight bound at 64-bit byte offset, b_offset=0.
      MatmulParams pt = p;
      pt.b_offset = 0;
#ifdef MOLLM_METAL_TENSOR
      if (has_tensor) {
        profile_label = "MATMUL_FP16_TENSOR";
        // Metal 4 tensor-API GEMM (fast path on M5/A19+): large-K
        // weights are staged in a 64-row tile and activations cast once
        // to an FP16 device tensor. Medium K uses direct device tensors.
        // grid: tgpig.y = N/64, tgpig.x = M/128.
        const bool direct_weights = p.K >= 512 && p.K <= 1024;
        id<MTLComputePipelineState> ps = pipelines_->pipeline(
            direct_weights ? "gemm_tensor_direct_f32a_f16b_f32c"
                           : "gemm_tensor_direct_f16a_f16b_f32c");
        // Bind A (activations) and B (weights) at their 64-bit byte
        // offsets; zero the in-shader element offsets accordingly.
        MatmulParams ptt = pt;
        ptt.a_offset = 0;
        ptt.b_offset = 0;
        id<MTLBuffer> activation_buffer = buffer_of(&A);
        NSUInteger activation_offset = A.device.offset;
        if (!direct_weights) {
          activation_buffer = cast_activation_to_f16();
          activation_offset = 0;
          ptt.a_row_stride = p.K;
        }
        [enc setComputePipelineState:ps];
        [enc setBuffer:activation_buffer offset:activation_offset atIndex:0];
        [enc setBuffer:buffer_of(&B) offset:B.device.offset atIndex:1];
        [enc setBuffer:buffer_of(&C) offset:0 atIndex:2];
        [enc setBytes:&ptt length:sizeof(ptt) atIndex:3];
        MTLSize tgc = MTLSizeMake(((NSUInteger)p.M + 127) / 128,
                                  ((NSUInteger)p.N + 63) / 64, 1);
        [enc dispatchThreadgroups:tgc
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      } else
#endif
      {
        profile_label = "MATMUL_FP16_TILED";
        id<MTLComputePipelineState> ps =
            pipelines_->pipeline("gemm_tiled_f32a_f16b_f32c");
        [enc setComputePipelineState:ps];
        [enc setBuffer:buffer_of(&A) offset:0 atIndex:0];
        [enc setBuffer:buffer_of(&B) offset:B.device.offset atIndex:1];
        [enc setBuffer:buffer_of(&C) offset:0 atIndex:2];
        [enc setBytes:&pt length:sizeof(pt) atIndex:3];
        // Half-staged tiles, TK=8: (32*8 + 32*8) halves = 1KB; the FP32
        // edge store scratch (256 floats = 1KB) reuses the region.
        [enc setThreadgroupMemoryLength:1024 atIndex:0];
        MTLSize tgc = MTLSizeMake(((NSUInteger)p.N + 31) / 32,
                                  ((NSUInteger)p.M + 31) / 32, 1);
        [enc dispatchThreadgroups:tgc
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      }
      if (p.activation != 0 && p.act_n_len != 0) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        id<MTLComputePipelineState> aps =
            pipelines_->pipeline("matmul_activation_range_f32");
        [enc setComputePipelineState:aps];
        [enc setBuffer:buffer_of(&C) offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        grid1d(p.M * p.N);
      }
    }
    if (commands_->profile) {
      profile_label += "[M=" + std::to_string(p.M) +
                       ",N=" + std::to_string(p.N) +
                       ",K=" + std::to_string(p.K) + "]";
    }
    return true;
  }

  default:
    return false;
  }
