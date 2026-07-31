# palm-infra

Palm Team 的 AI Infra 项目，目前包含 `mollm`。完整英文说明见 [README.md](README.md)。

## mollm

mobile-oriented LLM inference engine.
```
                 _ _
 _ __ ___   ___ | | |_ __ ___
| '_ ` _ \ / _ \| | | '_ ` _ \
| | | | | | (_) | | | | | | | |
|_| |_| |_|\___/|_|_|_| |_| |_|
```

`mollm` 是面向 ARM 和 x86 CPU 的轻量 C++ LLM 推理引擎，并提供实验性的 Apple Metal 支持。它将已支持的 Hugging Face 模型目录转换成单个 `.mollm` 文件，其中包含计算图、权重、tokenizer 与对话模板，并可直接运行。

项目当前聚焦 Apple Silicon 和其他现代 ARM 处理器上的高性能本地推理：FP16 使用 NEON FP16FML kernel；CPU 量化模型使用针对 ARM dot-product 指令优化的 weight-only int8/int4 kernel。Linux x86_64 将 scalar、AVX2/FMA/F16C 与 AVX-512 provider 编译为相互隔离的单元，覆盖 FP32、FP16、W8 以及 packed W4G32/W4G128 matmul；启动时通过 CPUID 选择当前 CPU 支持的最宽指令集，旧 CPU 不会执行新指令。设置 `MOLLM_X86_ISA=scalar|avx2|avx512|auto` 可以限制分派层级，便于正确性验证或针对具体机器调优；原有的 `MOLLM_X86_DISABLE_AVX2=1` scalar override 仍然可用。

## 现在，48GB Mac 也能运行 122B 模型

`mollm` 可以在 48GB Apple Silicon Mac 上运行 W4 量化的
Qwen3.5-122B-A10B：稠密权重保留在 RAM 中，仅从 SSD 读取被路由到的 MoE
expert。在当前 256-token cache sweep 中，通过有界的 16 GiB 共享 expert
cache 和跨层预取，decode 达到 16.53 t/s。

cache 容量可配置，并不要求把模型完整常驻内存。在下面的真实 prompt sweep
中，1 GiB expert cache 配置的峰值 RSS 仅 5.90 GiB；增大 cache 可以用更多内存
换取更少的 SSD 读取和更高吞吐。

| Expert RAM cache | Decode | Peak RSS | Cache 命中率 | 每个生成 token 的平均 SSD 读取量 |
|---:|---:|---:|---:|---:|
| **1 GiB** | 12.38 t/s | **5.90 GiB** | 47.9% | 1.72 GB/token |
| **10 GiB** | 16.19 t/s | 14.64 GiB | 83.5% | 0.75 GB/token |
| **16 GiB** | **16.53 t/s** | 20.60 GiB | **88.6%** | **0.51 GB/token** |

该 sweep 使用 16-token 真实 prompt、生成 256 tokens、greedy decoding、
`warmup=0`，每档 cache 取三个独立进程的中位数，并于 2026-07-29 重新运行。
10 GiB cache 的 decode 吞吐与 16 GiB 相差约 2.1%，同时峰值 RSS 少约
6 GiB；1 GiB 则展示了低内存运行能力。SSD 读取量统计 demand 或 prefetch
实际装入的路由 expert 逻辑字节，并除以生成 token 数；该摊销值包含 prompt
prefill，但不包含稠密权重和 CPU sidecar 的加载。

cache 策略、内存/吞吐 sweep、I/O 行为和 Perfetto trace 详见
[122B MoE SSD offload](docs/ssd-offload.md)。

## 实验性 DeepSeek-V4-Flash 支持

`mollm` 可以直接转换 DeepSeek-V4-Flash 原生 checkpoint，无需先展开成 FP16。
其中量化的稠密权重和共享 expert 权重保留原生 FP8 E4M3 数据，路由 expert
则保留 OCP MXFP4 格式，即 packed E2M1 数值和每 32 个数值一个 E8M0 scale。
生成的约 157 GB 模型包沿用大 MoE 模型的有界异步 SSD expert offload 路径。

Apple M5 Pro 上当前的纯 CPU 实验结果：

| 标准吞吐 | Expert RAM cache | 结果 |
|---|---:|---:|
| `pp256 + tg64` | 16 GiB | **9.35 pp / 4.74 tg** |

该标准样本使用 4 个 CPU 线程和 `warmup=3`。目前只完成了一个独立进程，
并非通常采用的五进程中位数，因此暂作 provisional 数据。

| Expert RAM cache | Decode | Peak RSS | 每个生成 token 的平均 SSD 读取量 |
|---:|---:|---:|---:|
| **1 GiB** | 4.37 t/s | **19.09 GiB** | 3.73 GB/token |
| **10 GiB** | **4.73 t/s** | **24.32 GiB** | 1.89 GB/token |
| **16 GiB** | 4.67 t/s | 26.91 GiB | **1.55 GB/token** |

该实验使用 19-token 中文真实 prompt、生成 64 tokens、greedy decoding、
6 个 CPU 线程、`warmup=0`，每档 cache 取三个独立进程的中位数。由于测试
协议不同，这组真实 prompt 数据不与标准 `pp256 + tg64` 性能图混排。SSD
读取量采用与上方 122B 表格相同的摊销逻辑 expert 字节定义。

## 已支持的模型

| 模型系列 | 状态 |
|---|---|
| Qwen3 dense text models | FP16、W8、W4 |
| Qwen3-30B-A3B MoE | 仅文本 W4 路径 |
| Qwen3.6-35B-A3B MoE | 仅文本 W4 路径 |
| Qwen3.5-122B-A10B MoE | CPU 与实验性 CUDA W4，支持 SSD expert offload |
| Qwen3.5-0.8B / Qwen3.5-4B | FP16、W8、W4、混合 W4；实验性单图视觉输入 |
| Youtu-LLM-2B | FP16、W8、W4、混合 W4 |
| RWKV7 | 官方 `.pth` 转换；FP16、W8、混合 W4；循环式 CPU/CUDA prefill/decode |
| DeepSeek-V4-Flash | 实验性 resident 与 SSD-offloaded CPU/CUDA 推理，支持原生 FP8/MXFP4 |

当前测试最充分的运行路径是 `w4g128`：它占用内存最少，且具有 mollm 中最快的
decode 速度。所有基于 `config.json` 的 converter 也支持 `w4g32` 和
`w4mixg32`。更小的 group 能改善 W4 质量，代价是更多 scale 和潜在的吞吐下降；
mixed 模式沿用各模型 `w4mixg128` 的 policy，决定哪些敏感 tensor 保留为 W8。

## 性能

### CPU

除非另有说明，Apple M5 Pro 数据使用 4 CPU 线程、`pp256 + tg64`、
`warmup=3` 和独立进程中位数。

当前 W4 中位数在 Youtu-LLM-2B 和 Qwen3.5-4B 上分别达到 140.77 和
69.78 decode tokens/s，是对应 llama.cpp Q4_0 CPU 结果的 1.47 倍和
1.73 倍。W4A8 decode 将激活与权重统一为 128-value 分组，先在整数寄存器
中累计四个 dot-product block，再统一应用 scale。图中的所有 dense 与
recurrent W4 行均使用当前模型包与二进制重新测试。

![CPU 吞吐量对比](assets/performance_cpu.svg)

### Metal（实验性）

Apple M5 Pro Metal 数据采用相同的 `pp256 + tg64`、`warmup=3` 和独立
进程中位数协议。Decode 从已有 256-token 上下文开始，两个运行时都将
模型权重保留在 Metal 上。

![Metal 吞吐量对比](assets/performance_metal.svg)

[测试协议、完整 CPU/Metal 性能表、长上下文 scaling 与正确性门禁](docs/performance.md)

## 快速开始

```bash
cmake -G Ninja -B build_i8mm -DCMAKE_BUILD_TYPE=Release
cmake --build build_i8mm -j

# W4 转换需要此工具。
cmake --build build_i8mm --target mollm-quantize

# 转换 Hugging Face 模型目录。
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4g128.mollm w4g128

# 从单个包文件启动对话。
./build_i8mm/mollm_chat --package qwen35_4b_w4g128.mollm --threads 4
```

交互命令：

```text
/reset   清空对话上下文
/quit    退出
```

## 构建

依赖：

- macOS/Apple Silicon、ARM Linux 或 Linux x86_64
- CMake 与 Ninja 或 Make
- Python 3
- 转换所需的 Python 包，主要是 `numpy` 与 `safetensors`

推荐构建方式：

```bash
cmake -G Ninja -B build_i8mm -DCMAKE_BUILD_TYPE=Release
cmake --build build_i8mm -j
```

在 ARM64 上，支持 i8mm 的编译器会把优化 kernel 作为独立 object 嵌入；运行时通过 HWCAP/sysctl 仅在兼容 CPU 上选用，库的其余部分仍保持 NEON/DOTPROD baseline。`MOLLM_ARM_I8MM=OFF` 会完全移除 i8mm object，`MOLLM_ARM_ISA=neon` 可在测试时强制走 fallback。普通 `build/` 目录也可以使用；将示例中的 `build_i8mm` 替换为对应目录即可。

## 转换模型

转换器会从 `config.json` 自动识别模型类型。

```bash
# 默认 FP16 包。
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_fp16.mollm

# W8 int8 基线。
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w8pc.mollm w8pc

# W4 性能包。
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4g128.mollm w4g128

# 混合 W4 质量包。
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4mixg128.mollm w4mixg128

# 更小 group 的纯 W4，以及沿用同一模型 mixed policy 的版本。
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4g32.mollm w4g32
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4mixg32.mollm w4mixg32

# RWKV7
python3 models/rwkv7.py /path/to/rwkv7-g1h-1.5b.pth rwkv7_1.5b_w4mixg32.mollm --tokenizer /path/to/tokenizer.txt --quant w4mixg32
```

MoE 示例：

```bash
python3 models/converter.py \
    /path/to/Qwen3-30B-A3B \
    qwen3_30b_a3b_w4g128.mollm \
    w4g128
```

DeepSeek-V4-Flash 的实验性转换会保留 checkpoint 原生 FP8 和 MXFP4 权重：

```bash
python3 models/deepseek_v4.py \
    /path/to/DeepSeek-V4-Flash \
    deepseek_v4_flash_native.mollm

./build_i8mm/mollm_chat \
    --package deepseek_v4_flash_native.mollm \
    --ssd-cache-mb 10240 --ssd-io-workers 8 --threads 6
```

CUDA 下还可以显式启用有界的 device expert LRU。cache miss 时 expert pair
从 host cache 复制到 GPU 一次，后续命中不会再读取 SSD 或进行 H2D 传输；当前
correctness-first kernel 仍会通过 D2D 复制组装 compact device scratch：

```bash
./build_cuda/mollm_chat --device cuda \
    --package deepseek_v4_flash_native.mollm \
    --ssd-cache-mb 10240 --device-moe-cache-mb 4096
```

`--device-moe-cache-mb` 默认关闭，并且必须和 `--ssd-cache-mb` 一起使用。

| `model_type` | 支持的模型 |
|---|---|
| `qwen3` | Qwen3 dense text models |
| `qwen3_moe` | Qwen3 MoE text models |
| `qwen3_5` | Qwen3.5 dense text 与单图视觉模型 |
| `qwen3_5_moe` | Qwen3.5/3.6 MoE text models |
| `youtu` | Youtu-LLM MLA models |
| `deepseek_v4` | 直接使用实验性的 `models/deepseek_v4.py` converter。 |

| 模式 | 适用场景 |
|---|---|
| `fp16` | 最简单的基线，且内存充足。 |
| `w8pc` | 需要 weight-only int8 量化，允许轻微质量偏移。 |
| `w4g128` | 需要最小包大小和最快 decode；通常是性能首选。 |
| `w4mixg128` | 纯 W4 质量不足，可使用更多内存保留部分 W8 tensor。 |
| `w4g32` | 使用更小的 32-value group 改善 W4 质量，并接受更多 scale 和潜在吞吐损失。 |
| `w4mixg32` | 在 W4G32 基础上，沿用该模型 `w4mixg128` 的 W8 tensor promotion policy。 |

DeepSeek-V4-Flash 不使用上述转换量化模式：其中量化的稠密/共享 expert
权重采用 checkpoint 原生 FP8 E4M3 block-128，路由 expert 采用
MXFP4 E2M1/E8M0 group-32。

W4 转换需要 C++ 构建的 `mollm-quantize` 工具；FP16 和 W8 不需要。prefill
图内部以 256 token 为分块大小，但 CPU runtime 使用 dynamic prefill；除非
显式指定 `--static-padded`，短 prompt 不会 padding 到 256。

## 运行对话

```bash
./build_i8mm/mollm_chat --package qwen35_4b_w4g128.mollm --threads 4
```

一次性、确定性的 smoke test：

```bash
./build_i8mm/mollm_chat \
    --package qwen35_4b_w4g128.mollm \
    --prompt "请只输出一句话，不要解释：杭州有什么特点？" \
    --max-new-tokens 64 \
    --threads 4 \
    --temperature 0
```

实验性的 Qwen3.5 单图对话（macOS，CPU vision encoder）：

```bash
./build_i8mm/mollm_chat \
    --package qwen35_0.8b_w4g128.mollm \
    --image photo.png \
    --prompt "描述一下这张图片。" \
    --max-new-tokens 128
```

converter 会将 checkpoint 的 FP16 vision tower 与量化后的文本模型一起写入
`.mollm` 包。当前首版支持 single-shot chat 中的一张静态图片；暂未支持多图、
视频、server 接口及 Metal vision encoder。图片默认按 262,144 像素预算缩放
（约 512x512）；可用 `--image-max-pixels` 在速度和细节间调整，最高
1,048,576 像素。如果不需要视觉能力，转换时可传 `--text-only`，避免 vision
tower 带来的包体与常驻内存开销。

采样生成：

```bash
./build_i8mm/mollm_chat \
    --package qwen35_4b_w4g128.mollm \
    --temperature 0.8 --top-p 0.95 --top-k 40 --min-p 0.05 \
    --repeat-penalty 1.05 --repeat-last-n 128 --seed 42
```

同时支持 OpenAI 风格的 `--presence-penalty` 与
`--frequency-penalty`。参数范围和默认值可通过 `mollm_chat --help` 查看。

默认情况下，`mollm_chat` 以 resident 模式加载包内权重。若需 mmap A/B 测试，传入 `--mmap`；默认 mmap 页面 warmup 已启用，可搭配 `--no-load-warmup` 关闭。

## 基准测试

```bash
./build_i8mm/mollm_bench \
    --package qwen35_4b_w4g128.mollm \
    --prompt-tokens 256 \
    --max-new-tokens 64 \
    --warmup 3 \
    --threads 4
```

## 本地 HTTP 服务

```bash
./build_i8mm/mollm_server \
    --package qwen35_4b_w4g128.mollm \
    --host 127.0.0.1 --port 8080 --threads 4
```

初始 server 提供 `GET /v1/models` 和 OpenAI 兼容的 `POST /v1/chat/completions`（含 SSE streaming）。它在串行请求间保留一个精确 token-prefix KV cache。每个请求均可指定采样参数；默认仍为确定性的 `temperature=0`。详见 [SERVER.md](SERVER.md) 的字段、示例与限制。

## 项目结构

```text
mollm/
├── kernels/    matmul、attention、MoE、norm、rope 的 ARM kernel
├── graph/      计算图格式、执行器、mmap 包加载、BufferPool
├── engine/     LLMEngine、tokenizer、对话/生成生命周期
├── models/     Python 转换器与计算图构建器
├── examples/   mollm_chat、mollm_server、mollm_bench、mollm_ppl
└── tests/      单元、压力与端到端测试
```

## 路线图

- 优化 prefill 性能，特别是 W8/W4 稠密模型的 prompt 处理。
- 提升实验性 Metal 性能，重点是量化 prefill、MoE prefill 与 CPU/GPU 同步开销。
- 基于当前单用户 REPL cache，为 serving 工作负载实现完整的 prefix cache。
- 扩展 accelerator 覆盖范围，同时保持 CPU runtime 作为可移植基线。
- 增加更多模型系列，并将当前 Qwen3.5 单图路径扩展到多图、视频、serving 与
  Metal vision encoder。
- 为 DeepSeek-V4 支持投机解码，先实现原生 MTP，并探索
  [DSpark](https://arxiv.org/abs/2607.05147) 风格的 confidence-scheduled
  semi-autoregressive drafting。
- 继续优化 SSD offload。

## 许可证

Copyright 2026 Tencent。根据 Apache License 2.0 发布；详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。捆绑依赖的声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 致谢

- [Cider](https://github.com/Mininglamp-AI/cider)：其在 Apple Silicon 上使用 Metal 4 INT8 TensorOps 实现 W8A8/W4A8 推理的工作，为 mollm 实验性的量化 Metal 路径提供了启发。
- Fang 等人的 [Fate](https://arxiv.org/abs/2502.12224)：其跨层 gate 预测思路为 mollm 实验性的 expert 预取路径提供了启发。
