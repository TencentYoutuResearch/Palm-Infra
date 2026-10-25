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
# The guarded array expansions below also support Bash 3.2 + nounset, where
# expanding an empty array directly raises an unbound-variable error.
exec env \
  MOLLM_RESIDENT_EXPERT_ARENA=cuda-device \
  MOLLM_CUDA_PREFILL_FP16_CACHE_MB="$preset_fp16_cache" \
  MOLLM_CUDA_HOST_HEADROOM_MB=24576 \
  MOLLM_CUDA_OUTPUT_POOL_MB=4096 \
  MOLLM_CUDA_DISABLE_DECODE_SEGMENTS=1 \
  build_cuda/mollm_chat \
  --package ../qwen38_flash_next_nvfp4_calibrated.mollm \
  ${preset_companion_args[@]+"${preset_companion_args[@]}"} \
  --device cuda --offload-ple --cuda-prefill-reclaim --cuda-prefill-logical-batch \
  --n-ctx 16384 --prefill-chunk-size "$preset_chunk" --threads 8 \
  --max-new-tokens 512 --temperature 0.6 ${preset_forward_args[@]+"${preset_forward_args[@]}"}
