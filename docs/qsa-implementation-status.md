# QSA 接入状态

2026-10-14；当前是实现中状态，不是模型功能完成报告。

## 已实现

- 独立图算子 `QSA_INDEXER`/`QSA_SDPA`、CUDA dispatch 与执行器生命周期
  契约；`build_graph(..., enable_qsa=True)` 连接 index projection、原始
  centered norm 权重、主 RoPE frequency 与 selected-key attention。
  Legacy graph 默认保持不变；新图不会静默以 dense SDPA 替代 QSA。
- Engine 使用既有 aux_state 分配/迁移/reset 路径共享 raw/pooled 状态；
  这两个状态容量按 runtime --n-ctx 调整，不受包内旧容量束缚。
  初始集成限制为 CUDA text/ctx<=32768、无 MTP/static padding。
- 原生 dispatch 在更新状态前验证连续因果 mask；有缺口、未来可见或
  无效未来 mask 明确报错。一次同步读取四字节标志，尚未缓存/优化；
  QSA_INDEXER backend dispatch 暂不支持外部 CUDA capture（primitive 仍支持）。
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

接入图执行后 `test_cuda_qsa_graph` 通过：FP16/FP32 KV、三种 liveness
策略、whole/chunk、reset 对照、独立 sparse attention 标量输出、raw K
保存以及不支持的 mask 在状态更新前拒绝。其 memcheck/synccheck/racecheck
日志在 `build_cuda/qsa-graph-*-1.log`；最终包含 NaN future mask 拒绝的
fixture 重跑三项 checker，`build_cuda/qsa-graph-*-2.log` 也各完成0、
0 errors/0 hazards。新 `test_qsa_engine_e2e` 使用序列化
的合成 Qwen4Exp 模型，包 ctx32、运行 ctx64；chunk3/17、越预算 prefill、
decode、多轮和 reset versus fresh hidden 精确对照通过。QSA enqueue
计数确认 indexer/attention 成对执行。该小模型允许其他算子 reference
fallback，不是全模型 require-native 或 checkpoint 质量认证。

接入阶段针对性11项回归通过（无跳过），并不代表全量测试已通过。

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

1. 完整 checkpoint 长上下文 companion 路径的正确性和性能验证。
2. 带缺口 mask 的可见分组；当前 fast path 仅接受连续因果历史。
3. 消除或降低接入版 mask 同步校验开销，完整模型 profiling 后优化。
4. 更长历史、真实模型 QSA 对照、完整模型多轮/reset 和吞吐测量。

当前完整模型 launcher 默认仍使用 dense bring-up，可显式增加
`--qsa-companion ../qwen38_qsa_bundle` 使用补充权重。没有改原始模型权重。
QSA 能减少主
attention 的长历史计算，但 indexer 还有自身成本；prefill/decode 收益
必须在完整接入后测量，不能按复杂度比例直接宣布整模型加速。

## Companion 接入（2026-10-14）

`--qsa-companion` 在加载 constants 前事务性重写 prefill/decode 图，插入
projection、centered norm、共享频率、raw/pooled auxiliary caches 和 indexer，
保留原始 QKV/HC/MoE 数学路径，仅将 SDPA 替换为 selected attention。
不写回 `.mollm`，不修改 checkpoint。CUDA text、context<=32768、无 MTP/padding。

加载检查 completion manifest、目录边界、完整层集合、shape/precision/header、
逐文件 SHA-256、finite 权重、main QKV/KV geometry 和 main RoPE。一旦不兼容
或损坏即报错，不静默回退 dense。哈希证明 bundle 自身完整，不能证明同
geometry 的大包来自同一 checkpoint；用户必须选择匹配来源的 companion。

扩展 `test_qsa_engine_e2e` 对照原生 QSA 包和 legacy dense 包+companion：
chunk3/17 的29-token sparse prefill、decode、多轮35-token 历史、reset hidden
逐值一致；错误 hash/path/main_heads/rope/layers 均拒绝。与 graph/export/CLI
四项回归一起通过。日志 `build_cuda/qsa-companion-e2e-1.log` 是首个对照运行，
随后 CTest 包含上述失败路径。

真实123745 MB checkpoint 首个短提示运行完成0，38-token 总上下文，
12层 QSA indexer/attention 各288次调用，decode32.8 t/s（24生成 token，
未预热）。日志 `build_cuda/qsa-companion-model-short-1.log` 和 memory log。
仅证明加载/短提示运行，不是原始 BF16 quality oracle，也不是长上下文加速证据。

真实7046-token 提示、ctx32768、chunk2048、warmup1/repeat1、32生成 token
运行完成0：prefill774.15 t/s（9101.57 ms），decode23.18 t/s。
日志 `build_cuda/qsa-companion-model-7k-2.log` 及 memory log。QSA indexer/
attention 各840次，未分配 dense attention score workspace。首版没有达到
性能增加：此前同提示 dense 历史样本约1250 t/s、30 t/s；不是本轮同步
A/B，也不是质量等价比较，不能据此宣布 QSA 本身没有价值。当前
selected-attention 是 correctness-first SIMT 实现，需要改善 GQA KV
复用和 PV 访存/并行度，随后再做同步 A/B。此任务尚未完成。

首个7K尝试 `qsa-companion-model-7k-1.log` 因 logical 模式禁止 profile
而在执行前失败，不能记为算子失败或成功性能样本。

算法参考（本轮按 hf-cli 技能只读核对，未下载模型）：
[Transformers Qwen4ExpTextQSAIndexer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)。

## 原始 checkpoint 小型补充权重导出

`tools/export_qwen4_qsa_weights.py` 只导出主模型 indexer 的三种权重。
Projection BF16→FP16，norm BF16→FP32 保留原始 centered 值，不加1；
manifest 最后写入，包含 geometry、每个文件 SHA256 和源 config SHA256。
只接受新/空目录，不覆盖已有文件。两项 fixture 测试确认转换、norm
中心语义、源文件未改、已有目录拒绝，以及无效 geometry 不写完成标记。

已从本机 NFS 原始 checkpoint 导出到工作区上层 `qwen38_qsa_bundle`：
36文件、39,337,056 bytes（含headers），主12 attention layers3/7/…/47。
不是独立模型；使用上面的 runtime companion loader 接入。未复制或改写原本
约123GB `.mollm`，没有下载原始模型。

## FP32 selected-attention 吞吐候选

新增两条 QSA 默认选路，不改选择集、不降低精度（环境变量可设0做A/B）：

- `MOLLM_CUDA_QSA_GQA_REUSE=1`：D256、GQA group2..16、queries>=32，
  每个 CTA 处理一个 query row/KV head，把16个选中 K/V 暂存一次给共享
  该 KV head 的 query heads 复用。FP32 online softmax，无 dense scratch。
- `MOLLM_CUDA_QSA_DECODE_SPLIT=1`：queries1、slots>=128，16个 partition
  分别做 selected attention，再在设备上按 max/sum 合并；复用 bounded
  workspace，真实H24/D256需要396288 bytes。prefill 不使用该拆分路径。

`bench_cuda_qsa_attention` 用 supplied selections 做0/1/2/2/1/0模式对照
（legacy/grouped/split），不是 indexer 或整模型 benchmark。
`build_cuda/qsa-gqa-split-bench-2.log`：Q2048/K32768 legacy约100ms、
grouped约58ms；Q1/K32768 legacy约0.29ms、split约0.064ms。
Q32的grouped仍略快，Q128/512收益更明确；grouped单token明显退化，
split大prefill无收益，因此 selector 按不同批量选路。

完整 primitive 数值、capture、strides、masked poison 对照通过。完整
shape memcheck/synccheck各完成0，0 errors，日志
`build_cuda/qsa-gqa-split-{memcheck,synccheck}-1.log`。完整 racecheck
`qsa-gqa-split-racecheck-1.log` 被主动终止（退出15），**没有通过**。
随后 `--optimized-small-only` 用131-slot/9-slot检查 grouped/split 的同步、
FP16/FP32、strides、poison和capture，racecheck完成0、0 hazards，日志
`build_cuda/qsa-gqa-split-racecheck-small-1.log`；不能扩张为完整racecheck证明。

`test_cuda_qsa_graph` 扩充 D256/H12、whole64 versus32/31/1 chunks，明确
执行 grouped prefill和split decode。输出对独立标量参考abs<3e-6，whole/
chunk选中ID精确一致；因online softmax不同FP32归约顺序，输出按相同
容差比较，不要求bit exact。日志 `build_cuda/qsa-gqa-split-graph-1.log`。

最新空tile跳过与64倍score-scale压力fixture重跑完整数值/memcheck/
synccheck均完成0、0 errors；`--optimized-small-only` racecheck完成0、
0 hazards。日志 `qsa-gqa-split-correctness-2.log`、
`qsa-gqa-split-{memcheck,synccheck}-2.log`、
`qsa-gqa-split-racecheck-small-2.log`。空tile跳过不依赖ID排序假设，仍
支持分散的invalid/masked IDs，并在共享状态复用前保持block barrier。

### 同构建整模型28K性能对照

原始模型包/提示不改，prompt28160 tokens、ctx32768、chunk2048、
warmup1/repeat1、greedy64生成 tokens、相同 accepted env、gate scale
cache0；串行、启动100GiB/停止24GiB内存保护。两个运行均完成0：

| 路径 | Prefill t/s | Decode t/s | Prefill ms | Decode ms |
| --- | ---: | ---: | ---: | ---: |
| QSA grouped+split | 1104.59 | 32.36 | 25493.57 | 1946.89 |
| Dense bring-up | 450.86 | 19.29 | 62458.75 | 3266.67 |

日志 `build_cuda/qsa-gqa-split-model-28k-1.log`、
`build_cuda/qsa-gqa-split-dense-control-28k-1.log` 及各 memory log。
约2.45倍prefill、1.68倍decode。最小可用内存分别35904004/27928968KiB。
QSA grouped336/split1512次，partial workspace396288 bytes、index score
workspace54525952 bytes、dense attention score workspace0。
仅QSA模式启用新默认选路，legacy dense行为不变。

这是吞吐证据，不是dense/QSA质量等价证明（越预算dense本身不是原始
模型算法）。本轮两边均单次timed样本；重复/反向验证与完整checkpoint
质量/多轮验证仍需继续，不据此宣布整个QSA任务完成。

默认选路（两个选择器均unset）、空tile跳过版继续完整模型
warmup1/repeat2：prefill1118.49/1114.57 t/s，decode32.52/32.47 t/s，
完成0；bench 内部验证两次greedy输出精确一致，也与显式开关候选的
64-token输出一致。日志 `build_cuda/qsa-gqa-split-default-model-28k-2.log`
和 memory log，最小35360240KiB。确认默认参数兑现性能提升及重复reset。

随后补正 split merge 的零权重行为：仍对有效partition的输出做IEEE FMA，
不因最终权重下溢为0就掩盖selected V的NaN（masked/unselected NaN仍不读）。
新增FP16/FP32、legacy/grouped/split、极端logit的对应fixture。对有限输入
的模型路径不改变算法；该非finite边界补正之后的最终检查以
`qsa-gqa-split-*-3.log` 和 `qsa-gqa-split-final-regression-2.log` 为准。
后者重新构建所有相关QSA fixture后10项通过、无skip；不是全量测试报告。
