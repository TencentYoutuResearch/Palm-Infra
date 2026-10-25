# palm-infra

PalmAI Team 的 AI Infra 项目。

[English](README.md)

## mollm

**一个小巧、可移植的 C++ 推理引擎，让大模型跑在你已有的硬件上。**

```text
                 _ _
 _ __ ___   ___ | | |_ __ ___
| '_ ` _ \ / _ \| | | '_ ` _ \
| | | | | | (_) | | | | | | | |
|_| |_| |_|\___/|_|_|_| |_| |_|
```

[![CI](https://github.com/TencentYoutuResearch/palm-infra/actions/workflows/ci.yml/badge.svg)](https://github.com/TencentYoutuResearch/palm-infra/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-ARM64%20%7C%20x86--64%20%7C%20Metal%20%7C%20CUDA-informational)](#构建)

`mollm` 将受支持的 Hugging Face checkpoint 转换成单个 `.mollm` 文件，其中
包含计算图、权重、tokenizer 与对话模板，然后直接运行这个包。推理时不需要
Python runtime、不需要 serving 框架、也不存在逐模型的粘合代码。

项目聚焦 Apple Silicon、ARM/x86 CPU 上的高性能本地推理，并提供可选的
Apple Metal 与 NVIDIA CUDA 后端。Decode kernel 针对 NEON/i8mm、
AVX2/FMA/F16C 与 AVX-512 手工优化；量化模型使用为直接从包内执行而设计的
weight-only W8/W4 布局。

### 亮点

- **每个模型一个文件。** 计算图、权重、tokenizer 和模板都在一个可 mmap 的
  包中。
- **大模型，一台机器。** SSD expert offload 将稠密权重保留在 RAM，只把命中
  的路由 expert 通过有界共享 cache 流式读入——100B+ MoE 模型可以在笔记本或
  单台工作站上运行。
- **Decode 优先。** Packed W4/W8 布局、AArch64 GEMV kernel、静态 decode
  workspace，以及运行时 ISA dispatch。
- **可移植优先。** CPU 后端是基线，Metal 与 CUDA 是可选加速器。CUDA 未实现的
  算子会显式回退，`--require-native` 可将任何回退变成错误。
- **CUDA 长上下文。** Qwen3.8-Flash-Next 使用原生 QSA 稀疏注意力与
  selected-key indexing，支持最多 262,144 tokens。
- **开箱即用。** 交互式 chat、benchmark、perplexity，以及 OpenAI 兼容的
  HTTP server。

> `mollm` 是研究项目，接口和包格式可能变化；实验性路径会在下文和链接文档中
> 标注。

## 快速开始

```bash
# 构建 CPU runtime。
cmake -G Ninja -B build_i8mm -DCMAKE_BUILD_TYPE=Release
cmake --build build_i8mm -j

# W4 转换所需的 helper。
cmake --build build_i8mm --target mollm-quantize

# 将 Hugging Face checkpoint 转换为单个包。
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4g128.mollm w4g128

# 开始对话。
./build_i8mm/mollm_chat --package qwen35_4b_w4g128.mollm --threads 4
```

```text
                 _ _
 _ __ ___   ___ | | |_ __ ___
| '_ ` _ \ / _ \| | | '_ ` _ \
| | | | | | (_) | | | | | | | |
|_| |_| |_|\___/|_|_|_| |_| |_|

 model     : Qwen3-30B-A3B
 arch      : qwen3-moe
 layers    : 48
 hidden    : 2048
 quant     : w4g128
 ctx       : 16384
 threads   : 4

 /reset   clear context
 /quit    exit


>
```

## 支持的模型

| 模型系列 | 状态 |
|---|---|
| Qwen3 稠密文本模型 | FP16、W8、W4 |
| Qwen3-30B-A3B MoE | text-only W4 路径 |
| Qwen3.6-35B-A3B MoE | text-only W4 路径 |
| Qwen3.5-122B-A10B MoE | CPU W4 + SSD expert offload |
| Qwen3.8-Flash-Next | CPU NVFP4 expert SSD offload；实验性 CUDA 原生 QSA |
| Tencent Hy-MT2-30B-A3B | text-only W4 MoE；CPU 与 Metal |
| Tencent Hy3-295B-A21B | text-only W4 MoE + CPU SSD expert offload |
| Hy4-preview（770B MoE） | 实验性 CPU GGUF/FP16 + SSD expert paging |
| Qwen3.5-0.8B / Qwen3.5-4B | FP16、W8、W4、mixed W4；实验性单图 vision |
| Youtu-LLM-2B | FP16、W8、W4、mixed W4 |
| RWKV7 | FP16、W8、mixed W4；recurrent CPU prefill/decode |
| DeepSeek-V4-Flash | 实验性 CPU 推理，原生 FP8/MXFP4 + SSD expert offload |

当前测试最充分的运行时路径是 `w4g128`：内存占用最低、decode 速度最快。所有
基于 config 的转换器也接受 `w4g32` 与 `w4mixg32`。更小的 group 以更多 scale
和潜在吞吐损失换取更好的 W4 质量；mixed 模式复用每个模型自己的 `w4mixg128`
策略来决定哪些敏感 tensor 保持 W8。

## 性能

### 大规模 MoE 与 SSD offload

稠密权重保留在 RAM；只有路由 expert 会从 SSD 读入有界共享 cache。同一套异步
路径支持 W4、checkpoint 原生 FP8/MXFP4 与 NVFP4 expert。

| 模型 | Expert cache | Prefill | Decode |
|---|---:|---:|---:|
| Qwen3.5-122B-A10B W4 | 16 GiB | **51.06 t/s** | **16.53 t/s** |
| Qwen3.8-Flash-Next NVFP4 | 20 GiB | **16.2 t/s** | **21.8 t/s** † |
| DeepSeek-V4-Flash | 16 GiB | **9.52 t/s** | **5.71 t/s** |
| Hy3-295B-A21B W4G128 | 16 GiB | **10.57 t/s** | **3.41 t/s** |

† 真实 prompt 下观察到的最好结果，使用 8 CPU threads 和 20 GiB expert
cache；其他行使用各自注明的 4-thread / 16-GiB 协议。

[Cache sweep、I/O 行为与 tracing](docs/ssd-offload.md)

### CPU

Apple M5 Pro 结果使用 4 CPU threads、`pp256 + tg64`、`warmup=3`，除非特别
说明，均为独立进程的中位数。

当前 W4 中位数在 Youtu-LLM-2B 上达到 140.77 decode tokens/s，在
Qwen3.5-4B 上达到 69.78 tokens/s，分别是同条件 llama.cpp Q4_0 CPU 结果的
1.47 倍和 1.73 倍。图中的稠密与 recurrent W4 行均使用当前包和二进制。

![CPU throughput comparison](assets/performance_cpu.svg)

### Metal（实验性）

Apple M5 Pro Metal 结果使用相同的 `pp256 + tg64`、`warmup=3` 与独立进程
中位数协议。Decode 从已有的 256-token context 开始，两个 runtime 都将模型
权重保留在 Metal 上。

![Metal throughput comparison](assets/performance_metal.svg)

[测试协议、完整表格、context scaling、CUDA 测量与正确性门禁](docs/performance.md)

### Decode 为什么快

- 针对 FP16、W8、W4 decode 深度优化的 AArch64 GEMV kernel。
- 对 decode 友好的 packed 权重布局，包括直接执行的 W4G128 包布局。
- W4A8 decode 以 128 个值为一组对齐 activation 与权重，在应用 scale 前于
  整数寄存器中累加四个 dot-product block。
- 静态可复用的 decode workspace，避免逐 token 分配抖动。
- 稠密模型的 prefill 仍是主要优化目标。

## 构建

要求：

- macOS/Apple Silicon、ARM Linux 或 Linux x86_64
- CMake + Ninja 或 Make
- Python 3，以及用于转换的 `numpy`、`safetensors`

推荐的本地构建：

```bash
cmake -G Ninja -B build_i8mm -DCMAKE_BUILD_TYPE=Release
cmake --build build_i8mm -j
```

在 ARM64 上，支持 i8mm 的编译器会将优化 kernel 编译为独立对象。运行时通过
HWCAP/sysctl 探测，只在兼容 CPU 上选择这些 kernel；库的其余部分保持
NEON/DOTPROD 基线。`MOLLM_ARM_I8MM=OFF` 会省略 i8mm 对象，
`MOLLM_ARM_ISA=neon` 可强制走运行时回退路径以便测试。普通 `build/` 目录也
可用；将示例中的 `build_i8mm` 替换为你的构建目录即可。

在 x86_64 上，scalar、AVX2/FMA/F16C 与 AVX-512 provider 分别编译，启动时由
CPUID 选择。设置 `MOLLM_X86_ISA=scalar|avx2|avx512|auto` 可限制选择层级；
原有的 `MOLLM_X86_DISABLE_AVX2=1` override 仍然可用。

可选后端：

```bash
# Apple Metal
cmake -G Ninja -B build_metal -DCMAKE_BUILD_TYPE=Release -DMOLLM_METAL=ON

# NVIDIA CUDA
cmake -G Ninja -B build_cuda -DCMAKE_BUILD_TYPE=Release -DMOLLM_CUDA=ON
```

## 转换模型

转换器会从 `config.json` 自动识别模型类型。

```bash
# 默认 FP16 包。
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_fp16.mollm

# W8 int8 基线。
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w8pc.mollm w8pc

# W4 性能包。
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4g128.mollm w4g128

# Mixed W4 质量包。
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4mixg128.mollm w4mixg128

# 更小 group 的 W4，纯 W4 或相同的模型特定 mixed 策略。
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

实验性 DeepSeek-V4-Flash 转换保留 checkpoint 原生的 FP8 与 MXFP4 权重：

```bash
python3 models/deepseek_v4.py \
    /path/to/DeepSeek-V4-Flash \
    deepseek_v4_flash_native.mollm

./build_i8mm/mollm_chat \
    --package deepseek_v4_flash_native.mollm \
    --ssd-cache-mb 10240 --ssd-io-workers 8 --threads 6
```

Qwen3.8-Flash-Next 转换保留 checkpoint 的 NVFP4 路由 expert，将稠密权重
量化为 W4G32，并把 `lm_head` 存为 per-channel W8：

```bash
python3 models/converter.py \
    /path/to/Qwen3.8-Flash-Next-NVFP4 \
    qwen38_flash_next_nvfp4.mollm

./build_i8mm/mollm_chat \
    --package qwen38_flash_next_nvfp4.mollm \
    --ssd-cache-mb 16384 --ssd-io-workers 8 --threads 4
```

支持的 `config.json` 模型类型：

| `model_type` | 支持的模型 |
|---|---|
| `qwen3` | Qwen3 稠密文本模型 |
| `qwen3_moe` | Qwen3 MoE 文本模型 |
| `qwen3_5` | Qwen3.5 稠密文本与单图 vision 模型 |
| `qwen3_5_moe` | Qwen3.5/3.6 MoE 文本模型 |
| `qwen4_exp` | Qwen3.8-Flash-Next 文本模型，原生 NVFP4 expert |
| `hy_v3` | Tencent HY-V3 / Hy-MT2 MoE 文本模型 |
| `hy_v4` | Hy4-preview MoE（原生 GGUF 或 FP16） |
| `youtu` | Youtu-LLM MLA 模型 |
| RWKV7 `.pth` | 直接使用 `models/rwkv7.py`。 |
| `deepseek_v4` | 直接使用实验性的 `models/deepseek_v4.py`。 |

量化模式选择：

| 模式 | 适用场景 |
|---|---|
| `fp16` | 想要最简单的基线，且内存充足。 |
| `w8pc` | 想要 int8 weight-only 量化与较小质量漂移。 |
| `w4g128` | 想要最小包与最快 decode。通常的性能选择。 |
| `w4mixg128` | 纯 W4 质量不够，且愿意为部分 W8 tensor 花费更多内存。 |
| `w4g32` | 想要更小 32-value group 带来的更好 W4 质量，并接受额外 scale 与可能的吞吐损失。 |
| `w4mixg32` | W4G32 加上与 `w4mixg128` 相同的模型特定 W8 tensor 提升策略。 |

说明：

- DeepSeek-V4-Flash 使用 checkpoint 原生 FP8 E4M3 block-128 量化的
  稠密/shared-expert 权重与 MXFP4 E2M1/E8M0 group-32 路由 expert，而不是
  上面的转换模式。
- Qwen3.8-Flash-Next 使用 checkpoint 原生 NVFP4 路由 expert、离线 W4G32
  稠密权重与 per-channel W8 `lm_head`。
- W4 转换需要 C++ 构建的 `mollm-quantize` helper。
- FP16 与 W8 转换不需要该 helper。
- Prefill 图内部以 256 token 为 chunk 大小，但 CPU runtime prefill 是动态的：
  除非显式传入 `--static-padded`，短 prompt 不会被 padding 到 256。
  `--prefill-chunk-size N` 可覆盖 runtime chunk 上限（`0` 保留包默认值）；
  改变 batching 可能改变数值结果，也不一定更快。
- 转换后的包会保存 Hugging Face config 或模型目录中的用户可见模型名，因此
  chat 会显示诸如 `Qwen3-30B-A3B` 的名字。

## 运行对话

```bash
./build_i8mm/mollm_chat --package qwen35_4b_w4g128.mollm --threads 4
```

一次性确定性 smoke test：

```bash
./build_i8mm/mollm_chat \
    --package qwen35_4b_w4g128.mollm \
    --prompt "请只输出一句话，不要解释：杭州有什么特点？" \
    --max-new-tokens 64 \
    --threads 4 \
    --temperature 0
```

采样生成：

```bash
./build_i8mm/mollm_chat \
    --package qwen35_4b_w4g128.mollm \
    --temperature 0.8 --top-p 0.95 --top-k 40 --min-p 0.05 \
    --repeat-penalty 1.05 --repeat-last-n 128 --seed 42
```

也支持 OpenAI 风格的 `--presence-penalty` 与 `--frequency-penalty`。范围与
默认值见 `mollm_chat --help`。

Qwen3.5 单图对话（macOS，CPU vision encoder）：

```bash
./build_i8mm/mollm_chat \
    --package qwen35_0.8b_w4g128.mollm \
    --image photo.png \
    --prompt "Describe this image." \
    --max-new-tokens 128
```

转换器会将 checkpoint 的 FP16 vision tower 与量化文本模型一起打包。当前实现
只支持单张静态图片的单次对话；多图/视频输入、server 集成与 Metal vision
执行尚未启用。图片默认使用 262,144 像素的 resize 预算（约 512x512）；
`--image-max-pixels` 可在速度与细节之间权衡，最高 1,048,576 像素。若不需要
vision tower，可使用转换选项 `--text-only`，避免其包体积与常驻内存开销。

MoE 对话：

```bash
./build_i8mm/mollm_chat --package qwen3_30b_a3b_w4g128.mollm --threads 4
```

默认情况下，`mollm_chat` 将包权重加载到常驻内存。若要 A/B 测试 mmap，传入
`--mmap`；除非同时传入 `--no-load-warmup`，否则会启用 mmap page warmup。

## 基准测试

```bash
./build_i8mm/mollm_bench \
    --package qwen35_4b_w4g128.mollm \
    --prompt-tokens 256 \
    --max-new-tokens 64 \
    --warmup 3 \
    --threads 4
```

## CUDA 后端（实验性）

CUDA 可独立于 CPU provider 构建：

```bash
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DMOLLM_CUDA=ON
cmake --build build_cuda -j
./build_cuda/mollm_chat --device cuda --package model.mollm
```

CMake 会检测可见 GPU 并检查编译器支持，在 GB10 上自动选择 `121a`。没有可用
GPU 或交叉编译时使用通用目标 `75;80;86`。只有需要覆盖自动选择（例如为其他
机器构建）时，才设置 `-DCMAKE_CUDA_ARCHITECTURES=...` 或 `CUDAARCHS`。
检测到 GPU 但工具链不支持时会在配置阶段报错；GB10 需要支持 `121a` 的工具链，
例如已验证的 CUDA 13。

选择 CUDA 时必须有可用的 CUDA 后端，不会静默回退到 CPU。传入
`--require-native` 可将任何使用 CPU reference 实现的算子变成错误。设置
`MOLLM_CUDA_PROFILE=1` 可打印 native/fallback 算子计数。

**Qwen3.8-Flash-Next** 已有验证过的原生路径：常驻 NVFP4 expert、QSA indexer
与 selected-key attention、grouped NVFP4 prefill、GDN / short-convolution，
以及 MTP state 处理。在 GB10 级统一内存系统上，可用 launcher 启动交互式长
上下文预设。先从同一个 checkpoint 准备模型包和 companion，以下命令在仓库
根目录执行：

```bash
python3 models/converter.py /path/to/Qwen3.8-Flash-Next-NVFP4 \
  ../qwen38_flash_next_nvfp4_calibrated.mollm
python3 tools/export_qwen4_qsa_weights.py /path/to/Qwen3.8-Flash-Next-NVFP4 \
  ../qwen38_qsa_bundle
```

转换器会保留 checkpoint 中已有的 activation calibration；输出文件名本身不
代表已具备校准数据。Companion 应导出到新建或空目录，其中仅包含 indexer
权重，不能单独推理。构建和 launcher 都不会自动下载这些文件。准备好后启动：

```bash
./chat-qwen38-cuda.sh
```

该预设使用 16K context 与 2048-token 分块 prefill。对于默认包，它会自动挂载
本地 `../qwen38_qsa_bundle` companion。companion 提供 QSA indexer 权重；没有
`--qsa-companion` 时，长上下文使用 dense attention，无法复现原始模型语义。
`--dense-bringup` 会跳过 QSA 以进行诊断性 A/B，`--qsa-companion <dir>` 可指定
其他 bundle。CLI 参数可覆盖预设，例如 `--prefill-chunk-size 512`。

QSA 支持范围：仅文本、连续因果历史最多 262,144 tokens（受内存限制），不支持
MTP、static padding、vision 或非规范 mask。不支持的配置会直接拒绝，而不是
静默使用 dense attention。companion 必须与所选 checkpoint 匹配。Launcher
会在启动时检查 GPU 空闲与 host 内存；admission check 不是内存预留。

对于 Qwen3.8-Flash-Next 之外的模型，CUDA 是 native/fallback 混合实现：图输出、
持久状态、dense elementwise、norm、RoPE、SDPA 与 decode `lm_head` 在 CUDA
上执行，而 recurrent GDN / short-convolution 及部分专用模型系列可能使用 CPU
reference dispatcher。范围与证据见
[QSA 实现状态](docs/qsa-implementation-status.md) 与
[性能记录](docs/performance.md)。

## 本地 HTTP 服务

```bash
./build_i8mm/mollm_server \
    --package qwen35_4b_w4g128.mollm \
    --host 127.0.0.1 --port 8080 --threads 4
```

Server 提供 `GET /v1/models` 与 OpenAI 兼容的
`POST /v1/chat/completions`（含 SSE streaming），并在串行请求之间保留一个
精确 token-prefix KV cache。采样参数可按请求设置；默认仍为确定性
（`temperature=0`）。字段、示例与限制见 [SERVER.md](SERVER.md)。

## 文档

- [性能协议与完整结果](docs/performance.md)
- [SSD expert offload：cache sweep、I/O 行为与 tracing](docs/ssd-offload.md)
- [Qwen3.8 QSA 实现状态与限制](docs/qsa-implementation-status.md)
- [HTTP server API](SERVER.md)
- [文档索引](docs/README.md)

## 项目结构

```text
palm-infra/
├── core/       Tensor、量化布局与跨后端算子参数
├── runtime/    Backend 接口、线程池、主机计算接口与缓冲池
├── storage/    mmap、文件区间与 SSD expert 缓存
├── graph/      计算图格式、序列化与执行器
├── engine/     LLMEngine、模型加载、tokenizer 与生成流程
├── backends/   CPU、CUDA、Metal 的资源管理与算子调度
├── kernels/    按 cpu/、cuda/、metal/ 划分的数值实现
├── models/     Python 转换器与计算图构建器
├── examples/   CLI、服务端、benchmark 与 perplexity 程序
├── tools/      独立开发与诊断工具
├── tests/      单元、后端与端到端测试
└── cmake/      安装包与下游项目构建支持
```

## 路线图

- 优化 prefill 性能，特别是 W8/W4 稠密模型的 prompt 处理。
- 提升实验性 Metal 性能，重点是量化 prefill、MoE prefill 与 CPU/GPU 同步开销。
- 基于当前单用户 REPL cache，为 serving 工作负载实现完整的 prefix cache。
- 扩展 accelerator 覆盖范围，同时保持 CPU runtime 作为可移植基线。
- 增加更多模型系列，覆盖当前 Qwen、Youtu、Hy 与 DeepSeek 之外的模型。
- 将当前 Qwen3.5 单图路径扩展到多图、视频、serving 与 Metal vision encoder。
- 为 DeepSeek-V4 支持投机解码，先实现原生 MTP，并探索
  [DSpark](https://arxiv.org/abs/2607.05147) 风格的 confidence-scheduled
  semi-autoregressive drafting。
- 继续优化 SSD offload。

## 许可证

Copyright 2026 Tencent。根据 Apache License 2.0 发布；详见 [LICENSE](LICENSE)
与 [NOTICE](NOTICE)。捆绑依赖的声明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 致谢

- [Cider](https://github.com/Mininglamp-AI/cider)：其在 Apple Silicon 上使用
  Metal 4 INT8 TensorOps 实现 W8A8/W4A8 推理的工作，为 mollm 实验性的量化
  Metal 路径提供了启发。
- Fang 等人的 [Fate](https://arxiv.org/abs/2502.12224)：其跨层 gate 预测思路
  为 mollm 实验性的 expert 预取路径提供了启发。
