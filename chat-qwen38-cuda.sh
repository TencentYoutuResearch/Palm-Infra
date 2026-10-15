#!/usr/bin/env bash
# Recorded GB10 Qwen3.8 resident+PLE+QSA preset. No benchmark-script dependency.
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
preset_dense=0
preset_package=../qwen38_flash_next_nvfp4_calibrated.mollm
preset_default_package=$preset_package
preset_companion_args=()
preset_forward_args=()
preset_context=16384
preset_args=("$@")
for ((preset_i=0;preset_i<${#preset_args[@]};++preset_i)); do
  preset_arg=${preset_args[preset_i]}
  if [[ $preset_arg == --dense-bringup ]]; then preset_dense=1; continue; fi
  preset_forward_args+=("$preset_arg")
  case "$preset_arg" in
    --help|-h) preset_help=1 ;;
    --package|--qsa-companion|--n-ctx|--prompt|--prompt-file|--image|--image-max-pixels|--prompt-tokens|--max-new-tokens|--rope-dim|--rope-theta|--threads|--prefill-chunk-size|--device|--ssd-cache-mb|--ssd-io-workers|--ssd-shallow-cache-layers|--mtp-draft-tokens|--mtp-companion|--warmup|--repeat|--temperature|--top-k|--top-p|--min-p|--repeat-penalty|--repeat-last-n|--presence-penalty|--frequency-penalty|--seed|--output|--trace)
      # Consume option values, never reinterpret prompt/path text as flags.
      if (( preset_i+1<${#preset_args[@]} )); then
        ((preset_i+=1))
        preset_value=${preset_args[preset_i]}
        preset_forward_args+=("$preset_value")
        if [[ $preset_arg == --package ]]; then preset_package=$preset_value; fi
        if [[ $preset_arg == --qsa-companion ]]; then preset_qsa=1; fi
        if [[ $preset_arg == --n-ctx && $preset_value =~ ^\+?[0-9]+$ ]]; then
          preset_value=${preset_value#+}
          preset_value=${preset_value#"${preset_value%%[!0]*}"}
          preset_value=${preset_value:-0}
          if (( ${#preset_value}<=6 )); then preset_context=$((10#$preset_value)); fi
        fi
      fi ;;
  esac
done
if (( preset_dense && preset_qsa )); then
  echo '--dense-bringup conflicts with --qsa-companion.' >&2; exit 1
fi
# Existing files can be compared by identity (including symlinks). For help
# before downloading weights, compare the basename and physical parent path.
# Failed directory resolution must never identify a custom package as default.
preset_is_default_package() {
  [[ $preset_package == "$preset_default_package" ||
     $preset_package -ef $preset_default_package ]] && return 0
  [[ ${preset_package##*/} == "${preset_default_package##*/}" ]] || return 1
  local package_parent default_parent
  package_parent=$(cd -P -- "$(dirname -- "$preset_package")" 2>/dev/null && pwd -P) || return 1
  default_parent=$(cd -P -- "$(dirname -- "$preset_default_package")" 2>/dev/null && pwd -P) || return 1
  [[ -n $package_parent && $package_parent == "$default_parent" ]]
}
preset_default=0
if preset_is_default_package; then preset_default=1; fi
if (( !preset_dense && !preset_qsa && preset_default )); then
  preset_qsa=1
  preset_companion_args=(--qsa-companion ../qwen38_qsa_bundle)
  if (( !preset_help )) && [[ ! -f ../qwen38_qsa_bundle/manifest.json ]]; then
    echo 'Missing QSA bundle: ../qwen38_qsa_bundle. Supply --qsa-companion <dir>; --dense-bringup is diagnostic only.' >&2
    exit 1
  fi
fi
preset_chunk=2048
preset_fp16_cache=8192
if (( preset_qsa && preset_context>65536 )); then
  preset_chunk=512
  preset_fp16_cache=4096
fi
# Preserve explicit cache overrides; final CLI arguments likewise override the
# chosen chunk. This is a conservative preset, not a memory reservation.
preset_fp16_cache=${MOLLM_CUDA_PREFILL_FP16_CACHE_MB:-$preset_fp16_cache}
if (( preset_help )); then
  echo 'Launcher: QSA is default for its Qwen3.8 package; --dense-bringup skips automatic QSA for diagnostics. Other options are forwarded.' >&2
fi
if (( ! preset_help )); then
  if (( preset_qsa )); then
    echo 'QSA enabled: CUDA text, contiguous causal history, context<=262144 (memory-dependent); package uses dense W4/LM W8, not original BF16 precision.' >&2
  else
    if (( preset_default )); then
      echo 'WARNING: dense bring-up omits QSA; beyond2048 tokens this is a diagnostic model, not original long-context attention.' >&2
    else
      echo 'Explicit package override: no automatic QSA companion attached; verify the package attention/precision yourself.' >&2
    fi
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
  -u MOLLM_TEST_CUDA_QSA_DUMP \
  -u MOLLM_TEST_CUDA_QSA_COMPARE \
  -u MOLLM_TEST_CUDA_QSA_INDEXER_BF16 \
  -u MOLLM_TEST_CUDA_QSA_FORCE_STREAMING_TOPK \
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
  MOLLM_CUDA_PREFILL_FP16_CACHE_MB="$preset_fp16_cache" \
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
  "${preset_companion_args[@]}" \
  --device cuda --offload-ple --cuda-prefill-reclaim --cuda-prefill-logical-batch \
  --n-ctx 16384 --prefill-chunk-size "$preset_chunk" --threads 8 \
  --max-new-tokens 512 --temperature 0.6 "${preset_forward_args[@]}"
