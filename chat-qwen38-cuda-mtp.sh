#!/usr/bin/env bash
# Opt-in QSA + MTP preset for long contexts.
set -euo pipefail
preset_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$preset_root"
if [[ ! -x build_cuda/mollm_chat ]]; then
  echo 'Build first: cmake --build build_cuda --target mollm_chat -j 2' >&2
  exit 1
fi
if [[ $# == 1 && ( $1 == --help || $1 == -h ) ]]; then
  echo 'Experimental Qwen3.8 CUDA QSA + MTP: default context32768, draft depth2; verification replay is automatic for eligible greedy batches. Speed depends on draft acceptance; short prompts can be slower. Options are forwarded to mollm_chat.' >&2
  exec build_cuda/mollm_chat --help
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo 'GPU is in use; stop the other workload before starting this resident model.' >&2
  exit 1
fi
if ! awk '/MemAvailable:/{seen=1; if($2<104857600)exit 1} END{if(!seen)exit 1}' /proc/meminfo; then
  echo 'This preset requires at least100 GiB MemAvailable at startup.' >&2
  exit 1
fi
echo 'Experimental QSA + MTP: default context32768, draft depth2; use --mtp-draft-tokens 1 to compare. The regular QSA launcher remains the default.' >&2
exec env \
  MOLLM_RESIDENT_EXPERT_ARENA=cuda-device \
  MOLLM_CUDA_PREFILL_FP16_CACHE_MB="${MOLLM_CUDA_PREFILL_FP16_CACHE_MB:-2048}" \
  MOLLM_CUDA_HOST_HEADROOM_MB=24576 \
  MOLLM_CUDA_OUTPUT_POOL_MB="${MOLLM_CUDA_OUTPUT_POOL_MB:-3072}" \
  MOLLM_CUDA_Q4_SMALL_BATCH_EXACT=1 \
  MOLLM_CUDA_Q4_SMALL_BATCH_EXACT_REUSE=1 \
  MOLLM_CUDA_Q4_COMPACT_BATCH_REUSE=1 \
  MOLLM_CUDA_GDN_SMALL_BATCH_SERIAL=1 \
  MOLLM_CUDA_MTP_DEVICE_ARGMAX="${MOLLM_CUDA_MTP_DEVICE_ARGMAX:-1}" \
  MOLLM_CUDA_MTP_DEVICE_CHECKPOINT=1 \
  MOLLM_CUDA_MTP_STATE_PREFIXES=1 \
  MOLLM_CUDA_MTP_VERIFY_Q8_BATCH=1 \
  build_cuda/mollm_chat \
  --package ../qwen38_flash_next_nvfp4_calibrated.mollm \
  --qsa-companion ../qwen38_qsa_bundle \
  --mtp-companion ../qwen38_mtp_qsa_bundle --mtp-draft-tokens 2 \
  --device cuda --offload-ple --n-ctx 32768 --threads 8 \
  --cuda-prefill-reclaim --cuda-prefill-logical-batch \
  --prefill-chunk-size 2048 --max-new-tokens 512 --temperature 0 "$@"
