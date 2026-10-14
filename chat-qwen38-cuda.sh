#!/usr/bin/env bash
# Opt-in, recorded GB10 Qwen3.8 resident+PLE preset. No benchmark-script dependency.
# Arguments are forwarded to mollm_chat; later CLI arguments override defaults.
set -euo pipefail
preset_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$preset_root"
if [[ ! -x build_cuda/mollm_chat ]]; then
  echo 'Build first: cmake --build build_cuda --target mollm_chat -j 2' >&2
  exit 1
fi
preset_help=0
preset_qsa=0
for preset_arg in "$@"; do
  [[ "$preset_arg" != --help && "$preset_arg" != -h ]] || preset_help=1
  [[ "$preset_arg" != --qsa-companion ]] || preset_qsa=1
done
if (( ! preset_help )); then
  if (( preset_qsa )); then
    echo 'Experimental QSA companion requested: CUDA text, contiguous causal history, context<=32768; checkpoint quality/performance validation is ongoing.' >&2
  else
  echo 'WARNING: this bring-up package omits QSA selection; histories beyond its2048-token budget use experimental dense attention, not the original long-context model.' >&2
  fi
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo 'GPU is in use; stop the other workload before starting this resident model.' >&2
    exit 1
  fi
  if ! awk '/MemAvailable:/{seen=1; if($2<104857600)exit 1} END{if(!seen)exit 1}' /proc/meminfo; then
    echo 'This preset requires at least100 GiB MemAvailable at startup; this is not a memory reservation.' >&2
    exit 1
  fi
fi
exec env \
  -u MOLLM_TEST_CUDA_WINDOW_DUMP \
  -u MOLLM_TEST_CUDA_PREFILL_LIVENESS \
  -u MOLLM_TEST_CUDA_PREFILL_LIVENESS_MAX_SEQ \
  -u MOLLM_TEST_CUDA_PREFILL_STATEFUL_MAX_SEQ \
  -u MOLLM_TEST_CUDA_PREFILL_WINDOW_SCHEDULE \
  MOLLM_RESIDENT_EXPERT_ARENA=cuda-device \
  MOLLM_CUDA_MOE_SHARED_OVERLAP=1 \
  MOLLM_CUDA_Q4_NARROW_PIPELINE=1 \
  MOLLM_CUDA_HC_PROJECTION_PAIR=1 \
  MOLLM_CUDA_NVFP4_SIMT_PIPELINE=1 \
  MOLLM_CUDA_Q4_COMPACT=1 \
  MOLLM_CUDA_REGISTER_ROUTES=0 \
  MOLLM_CUDA_GROUP_RMS_CACHE=1 \
  MOLLM_CUDA_Q4_COMPACT_13312=1 \
  MOLLM_CUDA_GDN_REGISTER_STATE=1 \
  MOLLM_CUDA_NVFP4_MMA_PREFILL_ONLY=1 \
  MOLLM_CUDA_NVFP4_MMA_PREFILL_BATCH=1 \
  MOLLM_CUDA_NVFP4_MMA_PREFILL_BATCH_SMALL=1 \
  MOLLM_CUDA_NVFP4_MMA_GATE_MIN_BATCH=1 \
  MOLLM_CUDA_NVFP4_MMA_DOWN_MIN_BATCH=1 \
  MOLLM_CUDA_NVFP4_MMA_BATCH_WARPS=4 \
  MOLLM_CUDA_PREFILL_FP16_CACHE_MB=8192 \
  MOLLM_CUDA_NVFP4_PREFILL_TILED_GROUP=1 \
  MOLLM_CUDA_NVFP4_MMA_COMPACT_BATCHES=0 \
  MOLLM_CUDA_NVFP4_PREFILL_RADIX_GROUP=0 \
  MOLLM_CUDA_NVFP4_MMA_BATCH_ROUTES=8 \
  MOLLM_CUDA_NVFP4_MMA_PREPARED_SPECIALIZED=0 \
  MOLLM_CUDA_SIGMOID_GR_REDUCE_FUSION=1 \
  MOLLM_CUDA_SHORTCONV_TRANSPOSE=1 \
  MOLLM_CUDA_NVFP4_MMA_GATE_BATCH_ROUTES=32 \
  MOLLM_CUDA_PREFILL_FP16_CACHE_HC_DOWN=1 \
  MOLLM_CUDA_GDN_PREFILL_BUTTERFLY_KEY=0 \
  MOLLM_CUDA_HC_PREFILL_ACTIVATION_REUSE=1 \
  MOLLM_CUDA_HC_PREFILL_RMS_HALF=1 \
  MOLLM_CUDA_NVFP4_ACTIVATION_PREPARE_FUSION=2 \
  MOLLM_CUDA_SDPA_PREFILL_SHARED_SCORE_LIMIT=1024 \
  MOLLM_CUDA_GDN_PREFILL_PAIRED_VALUES=1 \
  MOLLM_CUDA_GDN_PREFILL_QUAD_VALUES=1 \
  MOLLM_CUDA_SDPA_PREFILL_PV_UNROLL=1 \
  MOLLM_CUDA_GR_INJECT_RMS_FUSION=1 \
  MOLLM_CUDA_NVFP4_N32_ASYNC_WEIGHTS=1 \
  MOLLM_CUDA_HOST_HEADROOM_MB=24576 \
  MOLLM_CUDA_MEMORY_AUDIT=1 \
  MOLLM_CUDA_OUTPUT_POOL_MB=4096 \
  MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1 \
  MOLLM_TEST_CUDA_PREFILL_STATEFUL_LAST_USE=1 \
  MOLLM_CUDA_HC_PREFILL_DEFERRED_RMS=1 \
  MOLLM_CUDA_NVFP4_PREFILL_BF16_DOWN_OUTPUT=1 \
  MOLLM_CUDA_NVFP4_PREFILL_BF16_GATE_OUTPUT=1 \
  MOLLM_CUDA_PREFILL_FP16_CACHE_NARROW=1 \
  MOLLM_CUDA_NVFP4_PREFILL_EARLY_INPUTS=1 \
  MOLLM_CUDA_PREFILL_LT_GEMM=1 \
  MOLLM_CUDA_NVFP4_PREFILL_PARALLEL_PLAN=1 \
  MOLLM_CUDA_NVFP4_PREFILL_VECTOR_QUANTIZE=1 \
  MOLLM_CUDA_NVFP4_PREFILL_MMA_ACTIVATION_FUSED=1 \
  MOLLM_CUDA_HC_PREFILL_UP_REDUCE_FUSION=1 \
  MOLLM_TEST_CUDA_MOE_WINDOW_MERGE=1 \
  MOLLM_TEST_CUDA_WINDOW_SEGMENT_FUSIONS=1 \
  MOLLM_TEST_CUDA_WINDOW_SELECTIVE_KV_METADATA=1 \
  MOLLM_TEST_CUDA_HC_STANDALONE_DEFERRED_RMS=1 \
  MOLLM_TEST_CUDA_SDPA_PREFIX_QUERY_GROUP=4 \
  MOLLM_TEST_CUDA_WINDOW_SINGLE_GRAPH=0 \
  MOLLM_CUDA_HC_PREFILL_UP_INPUT_PITCH=336 \
  MOLLM_CUDA_HC_PREFILL_UP_TILE_COLUMNS=32 \
  MOLLM_TEST_CUDA_WINDOW_DEFER_MOE_INJECT=1 \
  MOLLM_CUDA_HC_COMPACT_CARRY=1 \
  MOLLM_CUDA_SDPA_PREFILL_HEAD_GROUP=4 \
  MOLLM_RESIDENT_NVFP4_GATE_CODES_KMAJOR=1 \
  MOLLM_CUDA_MOE_GR_INJECT_RMS_FUSION=1 \
  MOLLM_TEST_CUDA_HC_GATE_ROW_TILE=1024 \
  MOLLM_TEST_CUDA_HC_LOW_RANK_ROW_TILE=1024 \
  MOLLM_TEST_CUDA_DENSE_WIDE_FULL_LT=1 \
  MOLLM_TEST_CUDA_DENSE_WIDE_ROW_TILE=0 \
  MOLLM_TEST_CUDA_DENSE_WIDE_TILE=23 \
  MOLLM_CUDA_STRIDED_RESHAPE_VIEWS=1 \
  MOLLM_CUDA_PREFILL_FP16_CACHE_SHARED=1 \
  MOLLM_TEST_CUDA_SDPA_GEMM_SHADOW=0 \
  build_cuda/mollm_chat \
  --package ../qwen38_flash_next_nvfp4_calibrated.mollm \
  --device cuda --offload-ple --cuda-prefill-reclaim --cuda-prefill-logical-batch \
  --n-ctx 16384 --prefill-chunk-size 2048 --threads 8 \
  --max-new-tokens 512 --temperature 0.6 "$@"
