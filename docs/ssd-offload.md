# Running 122B MoE models with SSD offload

mollm runs Qwen3.5-122B-A10B W4 on a 48GB Apple Silicon Mac without keeping the
complete model resident. Always-used dense weights stay locked in RAM, while
asynchronous `pread` workers fetch only the routed MoE expert pairs from the
package. A bounded RAM cache, configured with `--ssd-cache-mb`, retains recently
used expert pairs.

## Cache policy

The default uses one shared cache instead of fixed per-layer quotas. Least-Stale
eviction protects experts needed by current and future layers. A next-layer
router prediction submits low-priority reads before the following MoE layer
needs them.

On macOS, the expert cache is pinned by default. Without pinning, the VM
compressor may compress cold anonymous expert buffers and later decompress them
inside matmul. This can make a larger cache slower despite a higher hit rate.
Use `--no-lock-expert-cache` when system-memory headroom matters more than stable
decode throughput.

Real chat prompt, four CPU threads, 16 prompt tokens, 128 generated tokens,
`warmup=1`, and five independent process runs:

| 16 GiB cache policy | Decode | Expert-cache hit rate | SSD reads |
|---|---:|---:|---:|
| Legacy equal per-layer cache, no prediction | 10.89 t/s | 74.5% | 69.4 GB |
| **Shared cache + cross-layer prefetch** | **12.70 t/s** | **89.3%** | 75.7 GB |

The default row locks both dense weights and the expert cache. The prediction
uses somewhat more SSD bandwidth but hides enough latency to improve
interactive decode. With a longer 256-token context, the strict `pp256 + tg64`,
`warmup=3` five-process median is 37.99 pp / 8.46 tg.

## Cache-size sweep

This sweep was rerun on 2026-07-24 using the prompt “给我讲一个故事”, greedy
decoding, 16 prompt tokens, 256 generated tokens, `warmup=0`, and three
independent processes per cache size.

| Expert RAM cache | Decode | Peak RSS | Expert-cache hit rate | SSD reads |
|---:|---:|---:|---:|---:|
| **1 GiB** | 9.84 t/s | **5.90 GiB** | 0.0% | 587.1 GB |
| **10 GiB** | 13.21 t/s | 14.69 GiB | 83.3% | 206.1 GB |
| **16 GiB** | **13.50 t/s** | 20.61 GiB | **88.6%** | **141.4 GB** |

Ten GiB retains most of the 16 GiB throughput while using about 5.9 GiB less
peak RSS. The 1 GiB configuration demonstrates that the 122B package can run
at under 6 GiB peak RSS, at the cost of a zero cache-hit rate and much more SSD
I/O.

Cache capacity must leave room for dense weights, KV cache, runtime buffers, and
other applications.

## Trace SSD overlap

Both `mollm_chat` and `mollm_bench` accept `--trace <path.json>` and write a
Chrome Trace / Perfetto timeline. It includes prefill/decode, per-layer routing
and expert compute, cache requests and acquisition, merged worker `pread`
operations, and flow arrows from queued reads to their workers.

```bash
./build/mollm_bench \
  --package /path/to/qwen35_122b_a10b_w4g128_ssd.mollm \
  --ssd-cache-mb 16384 --ssd-io-workers 8 --threads 4 \
  --prompt "Give me a short story" --max-new-tokens 128 --warmup 1 \
  --trace /tmp/mollm_122b_trace.json
```

Open the resulting JSON in [Perfetto](https://ui.perfetto.dev/).

![Chrome Trace / Perfetto view of decode, MoE execution, and SSD I/O workers](../assets/ssd_io_trace.png)

Future work includes cache-aware prefetch admission and issuing routed-expert
reads during attention, before the MoE layer begins.
