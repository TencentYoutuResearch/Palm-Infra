# 临时 handoff：Qwen3.8 Flash Next CUDA resident + offload-PLE

> 已归档的工程快照。下文的“尚未完成”和“QSA 尚未接入”等描述仅反映记录时状态。
> 当前 QSA 实现、支持范围和验证结果见 [QSA 状态文档](qsa-implementation-status.md)，
> 启动方式见 [README](../README_zh.md)。

记录日期：2026-10-13。此文用于交接当前工程快照，不是优化完成报告。

## 分支与范围

- 基线：`main` / 本地 `origin/main`，`a937e72`（drop the compat include shims）。
- 本次交接分支：`qwen38-cuda-resident-prefill`，从上述基线创建。
- 创建分支前，工作区有 61 个已跟踪文件修改、94 个未跟踪文件；新增内容包含实现、实验、测试、诊断工具及提示词 fixtures。全部尚未提交，不存在另一段 main..HEAD 的功能提交历史。
- 本次提交是工程快照，不意味着其中所有实验均可默认启用，也不意味着全量测试已通过。未推送远端。
- 快照提交 `573fe97` 保留在上述备份分支；后续依赖顺序拆分在
  `qwen38-cuda-resident-prefill-split`，共 13 个提交，拆分末端 `fd22682`
  的 Git tree 与快照完全一致。后续 review 记录是额外的文档提交。
- 主要详细证据：`docs/performance.md`、README 中的 CUDA 段落，以及本机 `build_cuda/` 日志、runner、CSV、Nsight 文件。build 目录被忽略，不会随提交保存，换机器时需另行保留必要证据。

## 用户目标（尚未完成）

继续优化 Qwen3.8 Flash Next 在 CUDA resident+offload-PLE 下的性能；减少实验开关和启动复杂度；不能仅在短输入、受限上下文或特殊测试配置下有效，必须验证真实多轮、长输入和 reset。

## 1. 加载方式与内存管理

- 增加 resident 模式的 `--offload-ple`：PLE 大表按需读取，其他工作集常驻。
- CUDA expert 使用长期租借、单份 arena，避免保留并锁定原始副本同时再持有整份重排副本。
- 修复 backend/provider 销毁顺序：设备工作与 expert 租借先结束，provider 后销毁。
- 添加 host headroom 预算、临时 prefill 内存回收、输出池容量控制和内存审计。
- 运行时 gate codes/scales K-major 重排，单份 scale payload 消除了约 4.69 GiB 额外副本。这类运行时优化不要求重转整个模型包。
- 修复 `tokenize-only` 意外加载整模型的问题：只读取必要 metadata、tokenizer 和模板，不能再次引入上百 GB 的权重加载。
- 原始模型目录：`/mnt/noindex/models/Qwen3.8-Flash-Next-NVFP4`（NFS）。本地推理包：仓库上一层 `qwen38_flash_next_nvfp4_calibrated.mollm`。不要覆盖原始模型或复制整个约 123GB 包。

## 2. CUDA decode 路径

补齐并优化 NVFP4 MoE、GPU routing、共享/routed expert 调度、dense W4/W8、LM head、GDN、ShortConv、gated residual、HC，以及 CUDA Graph/分段执行和缓存生命周期。

decode 主要仍使用 FP32 累加的 SIMT 路径，并未全部切换为 NVFP4 Tensor Core。

历史同一 24-token 提示词的 decode 中位数从 22.53 提高到 29.57 t/s；后来一些短上下文配置约 35–37 t/s。这些不是跨输入长度的统一性能保证，测试条件、冷/热运行不可混用。

## 3. NVFP4 Tensor Core prefill

- 在 GB10 SM121/121a 上证明可以执行原生 NVFP4 MMA，并推进到实验性模型 prefill 路径。
- 实现权重/activation 布局准备、激活校准表导出/加载、按 expert 分组、批量 gate/up/down。
- 实现 token K-major、共享量化、half 输入复用、BF16 输出边界、异步权重加载和路由规划等。
- 校准表是激活量化尺度，不是改写原始 expert 权重；带校准 metadata 的包用于 W4A4 实验，原有 W4/FP32 activation 路径保留。
- 独立探针、微测与模型收益必须分开解释，不能把 MMA 指令可执行当作模型性能完成。

## 4. 动态 prefill 与执行器

- 新增 `--cuda-prefill-reclaim` 和 `--cuda-prefill-logical-batch`。
- 协同修改 MoE/HC/attention 容量、last-use、view 链生命周期；减少中间物化、重复转换/RMSNorm 和输出池浪费。
- 增加 dense Lt GEMM、受限 FP16 权重缓存、HC 注入/归约等优化及实验。
- 部分 1743-token 受限测试达到约 1500 t/s，不能据此推断真实长上下文性能。
- `--n-ctx` 与 chunk 已解耦：logical chunk 支持 1..2048，0 默认 2048；短输入/尾块按实际长度执行，不补齐。2048 是单批上限，不是必须输入长度或上下文上限。
- 参数错误只显示诊断与 help 提示，不再打印整页 help。chat/bench 已重编译，参数、批量策略和 CLI diagnostics 三项针对性回归通过。

## 5. Dense attention 优化与测量

保留原生 QK/softmax，GEMM 计算 PV；添加 prefix 验证、零 PV 跳过、部分正零 mask 读取消除和 head 配对。合格 SM121 自动 PV 范围扩到实际 KV 32768；query slicing 和扩展双 head 尚未全部默认启用。

历史 28160-token 同配置对照：较窄 PV 约 389 t/s，长 KV 自动 PV 约 447 t/s，私有双 head 候选约 480 t/s。长输入 profile 中 attention 占窗口约 76.6%，其中实际 >16K 的 QK/softmax 占整个窗口约 52.7%。

这些仅是当前 dense bring-up 实现的优化与性能证据。kernel 时间覆盖不是硬件利用率；host synchronize 与 GPU kernel 重叠不能重复算 CPU 开销。

## 6. MTP（实验性，未获得收益）

添加 draft 图、独立 companion 流式权重导出/加载、共享目标 embedding/head、目标/草稿状态 tap、recurrent checkpoint 和 rollback/replay。

实际小规模 depth3 配置接受率低、验证/回滚成本高；一组 warm 测试约 10 t/s，同配置串行无 MTP 约 37 t/s。因此未默认启用，也未实现外部报告的 62.4 t/s。不能把合成状态测试当作真实模型跨模式质量验证。

## 7. 关键结构性缺口：QSA 尚未接入推理

当前 Qwen4Exp 导出图是短上下文 dense bring-up，没有 QSA indexer。扩大运行时 context 不等于补齐模型语义。此前 7K/14K/28K loss、token、回复一致性仅是本实现优化前后的等价，不是与原模型 QSA 的等价。README、launcher 和性能记录已补充警告；不能继续以 dense 自比较宣称原模型长上下文正确。

官方算法参考：
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py

重要实现语义：

- indexer Q/K 投影输出 640，4 个 Q head、1 个 KV head，head dim 128。
- 缓存 RAW K，跨 chunk 不可提前归一化、旋转或把不足 4 个 token 的组定稿。
- 按每个 query 的可见 token 顺序每 4 个成组，FP32 mean 后 cast 回 activation dtype，再 centered RMSNorm 和 RoPE。
- RoPE 使用组内第一个可见 token 的位置，复用主 attention 的位置嵌入宽度（不是 indexer_dim × rotary_fraction）。
- 分数为各 query head 的 dot 经 ReLU 后求和，再除 sqrt(128)；选 512 个完整组，所有未成组尾 token 必须保留。
- 默认预算 2048、ratio4；连续 causal 历史至 2051 仍全部可见，2052 首次可能丢弃一组。导出图保守预算限制仍为 2048。
- topk 平局顺序官方不保证，比较应避免 cutoff ties 或校验合法 tied set。

已落地：

- `models/qwen4_qsa_reference.py`：预计算分数的 token 选择 oracle，4 项测试。
- `models/qwen4_qsa_indexer_reference.py`：单序列 NumPy FP16/FP32 小规模计算 oracle，含投影、pooling/norm/RoPE、原始 K 缓存和 mask。
- `tests/test_qwen4_qsa_indexer_reference.py`：6 项测试，覆盖独立标量核对、offset 位置、mask 缺口、whole/chunk/token decode、reset、失败调用原子性。
- 为避免 BLAS GEMV/GEMM 批量差异，oracle 投影显式逐行 FP32 reduction；这是小规模参考，不是高性能 runtime 实现，也不模拟 BF16。
- 只读 safetensors header 确认 39 个 indexer tensor：12 套主模型、1 套 MTP，每套 BF16 640×2560 projection + 两个 BF16 128 norm。主模型 39,327,744 bytes（约 37.5MiB），含 MTP 共 42,605,056 bytes。

未完成：推理图算子、CUDA indexer/cache、selected-key attention、补充权重 artifact/加载路径、真实 QSA 模型质量与吞吐验证。没有为 QSA 重写模型包。

## 8. 易用性、安全和验证

- 独立 `chat-qwen38-cuda.sh` 不依赖 benchmark runner，默认 ctx16384/chunk2048/output512；用户参数可以覆盖。
- 部分已验证开关成为 logical 模式自动选择，脚本不再强制指定；仍有大量实验环境变量，易用性目标未完成。
- preset 启动要求 GPU 空闲、MemAvailable 至少 100GiB；chat/bench 每 250ms 检查 headroom，preset 阈值 24GiB。保护只终止模型自身，不重置 GPU或结束父 tmux；采样不能保证挡住瞬时内存峰值。
- GPU 模型/检查作业必须串行，先查真实进程/NVML，不凭旧日志猜测运行状态。不要 reset GPU、改驱动、drop caches 或启动并发整模型任务。
- 有算子、加载/失败恢复、执行生命周期、内存保护、loss、多轮/reset、profiling 测试；部分 sanitizer 先前超时，不算通过。没有最新覆盖整个工作区的全量绿色报告。
- 最近 QSA reference、QSA indexer reference、Qwen4Exp graph、CLI diagnostics 四个测试目标 4/4 通过；这是针对性测试，不是 CUDA QSA 或长上下文质量 gate。
- profiling 离线 overlap analyzer 从全量遍历改成 prefix sums/bisect，结果保持一致；这是工具加速，不是模型 TPS。

## 接手后的优先顺序

1. 接入真正 QSA：图/状态/CUDA indexer 与 selected-key attention，并设计小型补充权重加载，避免重写整个大包。
2. 独立核对原模型语义：跨预算、长输入、任意 chunk、decode、多轮/reset、mask/tail。不能以旧 dense logits 为长上下文正确性标准。
3. 在同输入、同 checkpoint、同冷热条件下测真实 QSA 性能，再按 profile 优化。
4. 收敛 preset/默认策略和实验开关；补实用配置覆盖及适当全量回归。

目标仍未完成。保留失败实验及原始证据，不把“能启动”“微测通过”“dense 自比较一致”重新定义为最终成功。

## 2026-10-14 推送前独立 review 与回归

用户要求 subagent 独立 review 最终状态后推送拆分分支（非 main）。
独立 reviewer 只读检查了 resident provider/arena 租借和销毁顺序、backend
析构同步、输出池 trim 与 capture graph 的失效、stateful last-use/segment
准入、图序列化 ID、companion canonical 路径/manifest/shape 和校准表校验。
在这些检查范围内未发现有充分证据的阻塞级新增问题。这不是对所有 kernel
数值、并发、全部实验配置或原模型长上下文质量的完整认证。

重新构建 chat/bench/PPL、执行器生命周期、MoE SSD、PLE、ShortConv、
checkpoint/watchdog、参数策略、小模型推理/MTP 以及 CUDA logical/segment
测试目标。24 项针对性回归通过（无跳过），包含 tiny Qwen3 CPU/CUDA
端到端推理、PLE offload、activation calibration 包校验、MTP 恢复、
tokenize-only、CLI、QSA oracle 和 profiling 工具测试。
另以串行方式补跑 7 项小型 CUDA 回归，全部通过（无跳过）：device
checkpoint、NVFP4 logical batch、分段执行及 GDN quad/validated SDPA/
one-graph 变体。两组共 31 项通过；未运行全量 295 项测试。

本轮没有加载完整 Qwen3.8 包，没有新吞吐结果，也没有验证每个中间提交。
此前完整模型 smoke/loss/性能证据仍是历史证据，不能冒充本轮重新验证。
已有的 QSA 未接入、长上下文 dense 近似和实验开关限制仍然有效。
`git diff --check main..HEAD` 的唯一提示是 extended prompt fixture 末尾
空行；为保持测试输入和原快照一致，未删除该空行。
