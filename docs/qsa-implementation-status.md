# QSA 接入状态

2026-10-14；当前是实现中状态，不是模型功能完成报告。

## 已实现

- Python token 选择及 indexer 投影/归一化/RoPE/raw-key-cache 小规模参考。
- GPU FP32 indexer update/pooling/scoring primitive：消费单独 dense projection
  的结果，使用原始 centered norm 权重和主 attention 的 inv_frequency；
  raw K 按 token 存储，只有新完成的组才进行 pooling/norm/组首 RoPE。
  已完成组缓存复用，未满组不写 pooled cache；query 分数仅读取其可见完整组。
  该 primitive 本身不进行 dense projection，也还没有接入 engine 状态。
- `kernels/cuda/qsa.cu` 原生 selected-key attention：只 gather 选中的 K/V，
  不分配全历史 H×Q×K scores，不展开整个 KV，也不靠 dense mask 假装稀疏。
  支持 FP16/FP32 KV、FP32 Q/output、GQA、显式 stride、causal/actual-length
  防越界、`-1` padding 和空选择行。每 CTA shared workspace 为
  `(slots+256)*4` bytes，默认 2051 slots 对应 9228 bytes。
- GPU causal block top-k：从调用方预先计算的 block scores 选择完整组，
  保留全部不完整尾组 token，输出按时间排序并以 `-1` 补齐；无需 host
  readback 或每次分配 scratch。CUB 排序 scratch 与 token staging 复用，
  静态 shared 容量编译期限制为不超过 48KiB。当前至多 8192 个组，即
  ratio4 的 32768 历史；更长历史需要扩展，不可静默切回错误 dense 模型。
- 当前 deterministic cutoff tie 规则为较小 block ID 优先。官方 topk
  不承诺相同平局顺序；质量对照应避免 cutoff ties 或接受合法 tied set。

## 本轮验证

`test_cuda_qsa_selected_attention` 在 GB10 完成0，覆盖：

- 11 个选择长度，含 0 完整组、2048..2054 的 dense/sparse 转折和32768。
- ascending/descending/tied block scores、未来 scores=NaN、全部尾 token、
  padding 与输出 guards、直接调用及更换 scores 后 CUDA Graph replay。
- attention 的 FP16/FP32 KV、padded/head-major/interleaved strides、GQA、
  独立 FP64 attention 参考、未选中 KV=NaN、全 mask 行及非法/future IDs、
  2051 槽、真实 D256/H24/K32768 decode 形状及更换 Q 后 capture replay。

未选中 KV=NaN 测试是读取语义证据，不是硬件带宽测量。算子通过不代表
indexer 生成的实际模型选择正确，更不代表整模型 TPS 达到目标。
memcheck 完成0，0 errors；完整日志在
`build_cuda/qsa-selected-attention-memcheck-1.log`。
synccheck、racecheck 同样完成0，各为0 errors；racecheck 为0 hazards。
日志为 `build_cuda/qsa-selected-attention-{synccheck,racecheck}-1.log`。
既有 QSA 两个 CPU reference、Qwen4Exp graph 与 CLI diagnostics 四项
针对性回归也全部通过。不是全量回归结果。

`test_cuda_qsa_indexer` 新增后完成0：73-token 小维度与129/2055-token
真实 indexer heads4/D128/RoPE64 形状，独立 FP64 scalar norm/RoPE/score
参考及合法 cutoff 检查。Whole versus irregular chunks/decode 的 normalized
Q、raw K、pooled K、visible scores、selected IDs 都按同一 GPU 算法精确
一致；第二组更换 projection 后从 past0 重建，并覆盖 capture、offset
positions、原始 K 精确保存、未完成/未来 pooled groups 不写和输出 guards。
两项 CUDA fixture（indexer + selected attention）同时回归通过。

另外验证真实32768历史的三 token continuation：独立准备的8191完整组
缓存保持逐字节不变，最后跨 chunk 组才更新；8192候选组的可见分数、
512选中组的独立 cutoff、尾 token 与 padding 都通过。此场景用
`--long-only` 单独运行，memcheck/synccheck/racecheck 各完成0，均0 errors，
racecheck 为0 hazards，日志 `build_cuda/qsa-indexer-long-{memcheck,synccheck,racecheck}-1.log`。
这是长历史 continuation 算子检查，不是整段32K模型 prefill 检查。

还添加 native dense projection → indexer update → top-k → selected attention
的组合 fixture，使用2560→640真实 projection 形状，FP16 weights/input、
FP32 accumulation/output。稀疏 dyadic 测试权重的 projection 与独立标量
结果精确一致；9 tokens/缩小预算4的选择与 attention 对照也通过。
该 fixture 故意使用合成、关联的主 attention Q/K/V，不是实际 checkpoint
或全模型图的验证，也没有吞吐测量。
Selected attention fixture 还补充了主图实际需要的 head-major Q/output
布局，与原有 token-major、KV interleaved/padded 布局一起回归通过。
扩充后的 selected fixture 重跑 memcheck/synccheck/racecheck 也各完成0、
0 errors/0 hazards，日志 `build_cuda/qsa-selected-attention-*-2.log`。

Indexer memcheck 目前只检查 `--small-only`（73/129 tokens）范围，完成0、
0 errors，日志 `build_cuda/qsa-indexer-memcheck-1.log`；不能据此声明完整
2055-token checker 范围已通过。当前 FP32 activation 数值参考不模拟原始
BF16 rounding；完整 checkpoint 对照还需要验证所选 precision 的质量影响。
最终 small-only fixture 又完成 memcheck/synccheck/racecheck 三项检查，均
完成0，0 errors/0 hazards，日志 `build_cuda/qsa-indexer-*-2.log`。

## 尚未实现/验证

1. 将 dense indexer projection 与现有 update/pooling/norm/RoPE primitive
   连接到真实推理图，并接入跨 chunk/raw-key 的 engine 状态。
2. 图算子/backend/engine 缓存 reset/resume 集成、带缺口 mask 的可见分组。
   当前 top-k primitive 只适用于明确的 contiguous causal history。
3. 小型 indexer 补充权重 artifact 与安全加载，避免重写约123GB原包。
4. 更长历史、真实模型 QSA 对照、完整模型多轮/reset 和吞吐测量。

当前 launcher 仍使用 dense bring-up，尚未调用新 primitives。没有新增
默认开关，没有改原始模型权重，没有完整模型 TPS 更新。QSA 能减少主
attention 的长历史计算，但 indexer 还有自身成本；prefill/decode 收益
必须在完整接入后测量，不能按复杂度比例直接宣布整模型加速。

算法参考（本轮按 hf-cli 技能只读核对，未下载模型）：
[Transformers Qwen4ExpTextQSAIndexer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)。
