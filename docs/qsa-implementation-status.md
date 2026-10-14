# QSA 接入状态

2026-10-15；以下各节按实现顺序保留证据，早期容量和接入限制不代表最新状态。

当前已接通真实 checkpoint 的 CUDA text QSA：小型 companion 补齐 indexer
权重，原始大包不重写；连续 causal history 的容量上限为262144，实际受
内存预算限制。28K ordered 路径约1091/24.6 prefill/decode t/s；与同构建
dense 对照相比约2.42/1.27倍。253376-token 单次真实运行为574.08/21.01
t/s。具体配置及限制见末尾各节，不能跨配置计算新的加速倍率。

独立算子参考、优化前后 token-loss 等价、真实模型状态验证是不同门禁；
目前不宣称与原始BF16完整模型的长上下文质量等价。MTP、static padding、
带缺口 mask 及外部 indexer graph capture 尚不支持，明确拒绝而非 dense
替代。最新最大边界/reset 验证结果见文末；本文不是全模型完成报告。

最新精度诊断还发现：只改变indexer投影后的BF16舍入，7K整模型CE就与
当前FP32路径明显不同（mean_abs1.4152），不满足等价门禁。此诊断默认
关闭，未改生产默认或吞吐结论；原始checkpoint整模型质量仍是未关闭
的门禁，不能以单层group overlap/选择一致替代。

**质量门禁更新：** c0ef942 的 online-softmax/partial-PV 优化未通过整模型
token-loss等价性验证，旧性能数字仅为历史运行，不能视为合格上线证据。
当前已删除该近似选路，改成保持原始FP32求和顺序的优化；新的质量与
状态验证通过后恢复ordered默认选路，吞吐需重新测量。细节见文末。

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

## 整模型质量门禁与 ordered 优化

`mollm_ppl` 新增 `--qsa-companion`；单独CLI原先没有传入该字段，不能
拿原本dense的PPL运行当成QSA验证。`tools/compare_token_loss.py` 检查
CSV的position/target对齐、finite CE及绝对差异；默认mean_abs<=1e-4、
max_abs<=1e-3。loss更低不是等价性证明，fixture对此明确拒绝。

同7046-token输入、ctx32768、chunk2048，原SIMT QSA与c0ef942近似优化：
7045 targets，仅1项CE精确一致，mean_abs1.59153675、max_abs21.1664531，
平均CE9.01722459→8.59651999。日志/CSV `qsa-quality-simt-7k-1` 与
`qsa-quality-optimized-7k-1`。该结果**不通过**门禁，不得因CE更低批准。
128-token actual-model shadow显示每个attention最大差异约1e-6..1e-5
(`qsa-quality-shadow-128-1.log`)；独立scalar容差测试不足以认证整模型。

移除online-softmax/partial-PV近似路径，新增ordered实现：
QK按GQA共享选中K或小批量分区并行，写入仅selected-slot的FP32 scores；
随后用与原SIMT相同的256-thread归约/normalize；PV按原slot顺序FMA，
GQA共享V或按feature分块。workspace为Q×H×slots，不是全历史dense矩阵。
真实Q2048/H24/slots2051为403243008 bytes；decode为196896 bytes。
不改模型权重，不降精度。低层API和backend均只提供ordered优化。

Primitive/capture fixture新增相同输入对legacy输出的逐字节比较；
覆盖D256/H24、early causal prefix、KV strides/poison、stress logits。
完整memcheck/synccheck完成0、0 errors，small-only racecheck完成0、
0 hazards，日志 `qsa-ordered-{memcheck,synccheck,racecheck-small}-1.log`。
数值日志 `qsa-ordered-correctness-2.log`。

同7046输入整模型ordered对照：7045项CE逐项一致，mean_abs/max_abs0。
带逐层shadow的 `qsa-quality-ordered-7k-1` 以及去掉shadow的
`qsa-quality-ordered-no-shadow-7k-1` 均完成0并与SIMT CSV相同。
说明匹配不依赖shadow同步。PPL输出本身不是与原始BF16 checkpoint
的绝对质量oracle，此处门禁只认证优化不改变当前QSA路径的模型结果。

`check_qsa_model_state` 是显式完整checkpoint诊断，不加入自动CTest。
检查4097-token prefix、7次teacher decode、余下prefill、8次生成decode、
follow-up prefill/8次decode，记录所有阶段最后一行hidden的完整float bits，
并在reset后重复全序列要求exact。不是单纯比较hash，也不是主观质量评分。
参考SIMT运行 `qsa-state-simt-full-model-1.log` 完成0：29个snapshot、
prompt7046、final_past7085。优化版对照仍需验证。

`qsa-ordered-regression-1.log` 失败：误与完整状态运行重叠启动，
小模型CUDA context初始化报告OOM；没有GPU reset，主状态运行完成0。
该回归不能算通过，后续只接受GPU空闲后的串行重跑。

GPU空闲后串行 `qsa-ordered-regression-2.log` 完成0，9项通过、无skip。
优化版完整状态工具 `qsa-state-ordered-full-model-1.log` 完成0，重复reset
精确一致；与SIMT参考运行全部58行state记录的完整hidden bits逐字节一致，
含29个阶段×2次运行、所有greedy token和past长度。grouped96/split576、
indexer/attention各696次。故不只验证单次prefill，也覆盖prefill→decode→
prefill、partial pooled group、第二轮追加、reset路径的真实checkpoint状态。

另以prefix2048、后续连续decode、总2112个token检查teacher-forced loss：
`qsa-quality-decode-{simt,ordered}-1.csv/.log` 均完成0，2111项CE逐项一致，
mean_abs/max_abs0（平均CE6.47324074）。据此恢复ordered两项默认选路，
环境变量设0仍可做参考A/B。旧近似kernel及其低层可选入口已删除，
不是把它留在一个隐藏开关后绕过门禁。

### 质量合格的ordered默认28K吞吐

`qsa-ordered-default-model-28k-1.log` 完成0，prompt28160、ctx32768、
chunk2048、warmup1/repeat2、64生成tokens；三个选择器均unset，确认
真实默认路径。两次greedy输出一致：

| 路径 | Prefill t/s | Decode t/s |
| --- | ---: | ---: |
| Ordered QSA repeat1 | 1092.48 | 24.60 |
| Ordered QSA repeat2 | 1088.89 | 24.52 |
| 同构建dense历史对照 | 450.86 | 19.29 |

约2.42倍prefill、1.27倍decode；不能继续宣称旧近似版1.68倍decode。
最小35010512KiB可用内存；grouped504/split2268，selected workspace
403243008 bytes、index score workspace54525952 bytes，未分配全历史
dense attention score workspace。性能与原始BF16模型质量是不同结论。

最终 `qsa-ordered-final-regression-1.log` 串行13项通过，无skip，含
execution lifecycle、三个CUDA QSA fixture、serialized engine、
CSV比较工具、PPL CLI/tokenize、graph/export和两个算法reference。
完整状态工具是额外真实模型证据，不替代独立数学参考，也非完整模型
原始BF16 oracle。当前32K支持边界、带缺口mask等限制尚需继续处理。

## 长历史选择扩展（2026-10-15）

本机原始config的max_position_embeddings=262144、ratio4。此前32K上限
源自top-k单CTA排序至多8192组，不是权重文件布局限制。
新streaming top-k按8192-group tile排序，保留每tile前budget/ratio组，
与已有候选再稳定合并，最后按token时间排序并追加所有尾组token。
prior tile的候选排在新tile前，稳定平局保持较小原始group ID优先。
所有排序shared memory仍<=48KiB；无CPU读回、无内部临时分配。
caller-owned score/ID scratch各Q×2×(budget/ratio)，默认Q2048共16MiB。
小历史默认仍用原selector，不增加额外scratch或排序开销。

原生indexer、mask validator、加载器、builder和CLI共同提高算法容量
上界到262144（`core/qsa_limits.h`）。这不等于GB10默认preset保证能以
默认chunk/cache运行262K：KV/raw/pooled/mask/score均占内存，继续保留
100GiB启动/24GiB停止保护和logical临时headroom检查，超预算明确失败。
没有重写`.mollm`、没有重新导出原始权重，没有GPU reset。

`--streaming-topk-only` fixture：32772/32775/65536/131071/262144长度，
四种merge widths、ratio1/4/16、asc/desc/ties、未来NaN、early prefix、
尾组、guard和更改score后的capture replay。CPU全量stable sort给出
相同ID序列。`qsa-streaming-topk-correctness-2.log` 完成0；完整该scope的
memcheck/synccheck/racecheck各完成0、0 errors/0 hazards，日志
`qsa-streaming-topk-{memcheck,synccheck,racecheck}-1.log`。

`--extended-history-only` 在64K/128K/262K历史末尾继续3个token：旧
pooled cache不变、新跨chunk group/独立scalar scores/cutoff通过。
日志 `qsa-extended-indexer-correctness-1.log`。这是preinitialized prefix
continuation的证据，不是完整262K prefill/真实checkpoint认证。

`MOLLM_TEST_CUDA_QSA_FORCE_STREAMING_TOPK=1` 强制现有短历史也走新
selector，用于质量A/B。实际7046-token完整checkpoint运行
`qsa-streaming-quality-7k-1.csv/.log` 完成0，与SIMT参考7045项CE
逐项一致、mean_abs/max_abs0。正常模式仅历史超过8192 groups使用新路。

已添加独立CTest entry `test_cuda_qsa_streaming_topk` 和
`test_cuda_qsa_extended_history`，RUN_SERIAL；builder容量boundary与
PPL tokenize-only的262K参数解析也有回归。

真实完整checkpoint首次跨32K运行 `qsa-extended-model-56k-1.log` 完成0：
prompt56312、ctx65536、chunk1024、warmup0/repeat2、32生成tokens，两个
greedy输出精确一致。prefill924.17/926.21 t/s，decode23.82/23.92 t/s。
这不是与28K/chunk2048的同步性能A/B，不据此计算新的加速倍率。
最小35073596KiB可用内存；streaming_topk1296次、workspace8MiB，
grouped1320/split744、indexer/attention各2064次。计数证实真实历史
超过8192 groups时使用新selector，而非仅提高CLI限制后回到dense。
数据文件 `qsa-64k-model-prompt.txt` 是旧28K原始测试提示重复两次，
原提示、原模型权重均未改。全部GPU工作串行并带内存保护。

完整262K模型prefill/多轮尚未执行；三token的算法测试不能代替这个门禁。
当前还保留连续causal/无MTP/padding等范围限制，目标仍未宣告全部完成。

扩展indexer的上述完整64K/128K/262K continuation scope也完成memcheck/
synccheck/racecheck三项、退出0、0 errors/0 hazards，日志
`qsa-extended-indexer-{memcheck,synccheck,racecheck}-1.log`。最终
`qsa-extended-final-regression-1.log` 串行15项通过、无skip，包含两个
新CTest entry；不是全仓库测试结果。

## 接近checkpoint最大上下文的真实运行（2026-10-15）

`qsa-extended-model-253k-1.log` 完成0：真实prompt253376 tokens、
ctx262144、chunk512、FP16 cache上限4096MiB、gate-scale cache0、
warmup0/repeat1、16生成tokens，仍为相同calibrated package+QSA bundle。
提示由旧28K测试文本重复9次构成，不改原文本或模型权重。
prefill574.08 t/s（441358.51 ms），decode21.01 t/s（714.03 ms）。
min MemAvailable31603400KiB，高于24GiB保护阈值，未触发guard。

streaming_topk5352、grouped5940/split180、indexer/attention各6120次。
selector workspace4MiB、selected attention workspace100810752 bytes、
index scores129499136 bytes；FP16 weight cache实际4292280320 bytes。
这是实际253K prefill+decode，不是preinitialized-prefix替代，也不是
完整262144-token输入或重复/reset验证；没有warmup/repeat2，不能据此
声称完整最大边界/原始BF16模型质量等价。新配置也不同于旧28K基准，
不计算跨配置加速倍率。全部GPU进程串行，没有GPU reset/driver修改。

启动preset新增保守profile：QSA且最后一个有效`--n-ctx`>65536时，
默认chunk512、FP16 cache4096MiB；较小上下文和dense模式保持旧默认。
显式`--prefill-chunk-size`仍在后面覆盖，显式cache环境变量也保留。
不解释任意shell表达式；十进制leading zero/plus正确归一，过大/非法
值交给C++验证。不是内存预留，也不取消100GiB/24GiB保护。
`test_qwen38_cuda_preset.py` 用fake chat/--help验证args/env，不初始化
GPU或读取真实模型。最终 `qsa-long-preset-regression-1.log` 串行16项
通过、无skip。当前仍需完成最大边界/更长多轮、mask范围等剩余门禁。

## 精确最大边界、多轮及 reset（2026-10-15）

`check_qsa_model_state` 新增工具专用的前置 `--fill-context` 参数：正常
混合序列完成后，以有效 token 补到 runtime n_ctx；在上限分别调用
decode_hidden、prefill_hidden 和 sampled prefill，要求三者拒绝且 past
不变。随后 reset 并完整重跑，对每个记录点的 past、完整最后位置 hidden
FP32 bits 和 greedy token 精确比较；所有记录的 hidden/logits 要求有限。
这个参数必须在其他 CLI 参数之前，不改变通用 CLI 或正常推理行为。

真实 checkpoint 的 `qsa-state-max-context-full-model-1.log` 完成0：相同
253376-token 提示，ctx262144、chunk512、FP16 cache4096MiB、gate-scale
cache0，repeat2。每轮先4097-token prefill、7个 teacher-forced decode、
剩余提示 prefill、生成8个 token，再追加真实 chat-template followup 并
decode8次，最后补齐262144。两轮均有
`qsa_boundary_pass=1 past=262144 rejected=3`；最终
`qsa_state_pass=1 snapshots=538 prompt_tokens=253376 final_past=262144`。
538个记录点全部 exact reset-versus-first-run 通过，不只是回复文本一致。

min MemAvailable31123800KiB（约29.68GiB），高于24GiB保护阈值，未触发
guard。streaming_topk11208、grouped12288/split576、indexer/attention
各12912次；selected workspace100810752 bytes、index scores134203392
bytes、streaming scratch4MiB。日志有大量 hidden_bits，查看状态应过滤
`state_repeat`/`qsa_boundary_pass`/`qsa_state_pass`，避免整段输出。
这是状态/边界门禁，不是吞吐基准，也不是原始BF16完整模型质量 oracle。

合成 serialized engine fixture 同时新增 runtime ctx64 精确填满、三种
超限拒绝、拒绝后 reset 对照；chunk3/17 都通过，包内ctx32不限制runtime
状态容量。所有真实模型/GPU回归严格串行；原始大包、NFS原始权重及驱动
未改。连续 causal CUDA text 的最大边界门禁已补齐，非canonical mask、
MTP/padding 等范围限制及原始BF16全模型质量认证仍未宣告完成。

最终 `qsa-max-context-final-regression-1.log` 串行16项通过、无skip，包含
更新后的 serialized engine fixture、CUDA QSA primitive/graph/indexer、
launcher、CLI/tokenize、导出及独立算法参考；不是全仓库回归结果。

## 真实 indexer 激活的独立 PyTorch 对照（2026-10-15）

新增默认关闭的 `MOLLM_TEST_CUDA_QSA_DUMP=<new-file>`。每个 backend
最多输出一次，选择首次 keys>budget+ratio 的 sparse dispatch，且限
keys<=8192、queries<=2048、payload<=64MiB。使用 exclusive create，
已有文件拒绝而非覆盖；失败明确报错。仅在 native kernel 之后同步读回，
不改输入/cache/选中ID，不用CPU结果替代CUDA结果；不能用于吞吐计时。
短提示若未达到门槛不会创建文件，不能把缺失快照视为测试通过。

QSAIDX1 小端快照保存当前 projected Q/K、centered norm 权重、原始
frequency、raw cache、normalized query/pooled keys、group scores 和
选中ID。`tools/check_qsa_indexer_snapshot.py` 在CPU PyTorch独立计算
centered RMSNorm、主attention partial RoPE、FP32 pooling、ReLU-head
sum以及stable top-k，不import仓库NumPy oracle或调用CUDA；检查当前
raw K精确保存、未来score=-inf、完整组/全部尾token、时间顺序和padding。
FP32数值比较用atol3e-4/rtol3e-5；cutoff在此界内的变化单独报告，不
冒充bit-exact。BF16舍入分支只作诊断，不以“数值更低/集合相似”判为等价。

官方方程来源仍为上面的 Transformers indexer/RMSNorm/RoPE 源码。
按hf-cli技能只读核对本地checkpoint和环境，未安装/升级包或下载模型；
checker使用既有 `/home/molly/miniconda3/envs/torch/bin/python`。

小图 `qsa-indexer-small-snapshot-1.log` 完成0、29query/29keys，独立
FP32 query/pool/scores最大误差2.38e-7/1.19e-7/9.54e-7，29行选择精确
一致。`test_cuda_qsa_snapshot` 自动验证同一backend continuation只写
一次、payload长度、已有文件拒绝且原hash不变，保留fixture独立attention
输出检查。`test_qsa_snapshot_format` 4项校验header/geometry/尺寸上限，
无torch依赖。另 `test_qsa_snapshot_oracle.py` 在torch环境的6项测试
覆盖值/未来score/ID/cutoff损坏、truncation/trailing bytes，全部通过，
日志 `qsa-snapshot-oracle-unit-1.log`；不是原始模型oracle。

真实完整checkpoint `qsa-indexer-real-model-7k-1.log` 完成0，正常7046-token
提示、ctx16384/chunk2048、warmup0/repeat1、8生成tokens，带原内存guard。
最小36396076KiB可用内存，未触发guard。快照取主模型首次稀疏chunk：
queries2048/past2048/keys4096/groups1024；原权重包、NFS文件均未修改。
checker命令：

```sh
/home/molly/miniconda3/envs/torch/bin/python \
  tools/check_qsa_indexer_snapshot.py build_cuda/qsa-indexer-real-4k-1.bin
```

`qsa-indexer-real-oracle-1.json/.log` 完成0：query/pool/scores最大误差
9.54e-7/7.15e-7/5.72e-6，2048行FP32选中集合全部精确一致，无near-cutoff
变化。BF16 post-projection分支最大score差0.0321779，921/2048行集合
完全一致，选中group总体重合率0.9987221；**不是BF16等价通过**。
快照始于投影之后，排除不了projection精度及上游hidden drift，也不证明
原始BF16完整模型loss/长上下文语义质量；不能把本节替代此剩余门禁。
诊断模型日志中的TPS包含同步dump开销，不更新合格吞吐基准。

最终 `qsa-snapshot-final-regression-1.log` 串行18项通过、无skip；原16项
加快照native lifecycle和portable格式校验。主模型诊断已终止、GPU空闲
后才运行CUDA回归，未重置GPU或改驱动。下一步需测BF16 indexer中间舍入
引起的选择差异对整模型token-loss的影响，不能仅凭group overlap宣告
完整质量门禁通过。

## BF16 中间舍入与整模型 loss 诊断（2026-10-15）

新增默认关闭的 `MOLLM_TEST_CUDA_QSA_INDEXER_BF16=1`：只有indexer
projection之后的激活舍入改变，raw K、query、pooling mean、centered
RMSNorm输出、RoPE cos/sin、逐次乘法和加法模拟BF16，再存入FP32容器；
score dot/head sum仍用FP32。模板specialization保留原FP32运算顺序。
不改变dense projection、上游GR hidden、主attention或权重布局，所以
不是原始BF16整模型reference。此flag不默认启用、不新增通用CLI选路。

诊断快照使用QSAIDX2，沿用QSAIDX1 header/eps后追加uint32 flags=1；
原FP32仍输出v1。checker自动识别precision，拒绝缺失/未知flags。
小图29query的query/pool/scores和selected IDs与CPU PyTorch BF16计算
完全一致。BF16 graph fixture保留whole/chunk/reset及独立主attention
输出检查，snapshot lifecycle继续保证no-overwrite/continuation。

真实完整checkpoint的同构建PPL A/B都完成0：原7046-token提示、ctx16384、
chunk2048、相同calibrated package/companion/preset，只有上述flag及诊断
dump不同，GPU严格串行、仍有100GiB启动/24GiB停止保护。日志/CSV：
`qsa-bf16-rounding-{control,candidate}-7k-1`。min MemAvailable分别
34753048/34633440KiB。两组各有48次indexer/attention；candidate明确
记录 `diagnostic_bf16_post_projection=48`，不是未执行的环境开关。

新control与历史ordered-no-shadow控制的7045项CE逐项完全相同，
`qsa-bf16-default-control-revalidation-1.json` equivalent=true、误差0。
这证明新诊断未改变当前默认FP32结果。

`qsa-bf16-rounding-model-comparison-1.json` equivalent=false：7045项、
exact2054、mean_abs1.4152091、max_abs20.2197151、p99_abs9.9219646；
control/candidate mean CE9.0172246/9.6245748、delta+0.6073502。
首次变化position2054，此前全部精确一致；2048..4095区间mean_abs
1.32794，4096..7044区间2.45864。说明稀疏选择的precision差异可能被
后续层放大，不允许据总体group overlap宣称无影响；平均CE更高/更低
也不是谁更忠于原始模型的证据。**不能将candidate作为合格默认路径，
也不能将当前FP32控制作为原始checkpoint质量认证。**

真实candidate首个稀疏chunk的strict CPU数值oracle未通过：query中
13/1048576元素不同、最大0.015625；pool2/131072元素不同，max0.00390625。
原始raw K的BF16舍入精确保存。日志
`qsa-indexer-real-bf16-oracle-1.log` 是失败证据，不计作通过。
尝试在诊断norm使用`__frsqrt_rn`后，CPUstrict gate仍失败，且相同快照
与PyTorch CUDA的选择一致从2048行降至2046行，故已撤回该尝试；
`qsa-indexer-bf16-rsqrt-{replay,oracle}-1.log` 不是合格优化证据。

`test_cuda_qsa_indexer --snapshot-replay input-v2 new-output` 新增限大小
重放：使用保存的raw K重算全部pooled history，prefix Q为dummy但只
保留实际current rows，不复用旧pooled结果。最多8192keys/2048 retained
queries/64MiB payload；已有输出、截断或超界header拒绝。小图source
与replay文件完全相同，并有自动拒绝/原hash保持测试。

checker新增显式 `--reference-device cuda --selection-only`，仅允许BF16
诊断：要求每行selected set与独立CUDA PyTorch方程精确相同，但分别
报告value gate，不豁免NaN/非法padding/缺组/漏tail。CUDA oracle启动前
检查GPU空闲；CPU为默认，不初始化GPU。不以selection-only代替strict
value或整模型质量门禁，也不能用于FP32绕过数值检查。

原candidate snapshot在该scope完成0：
`qsa-indexer-bf16-cuda-selection-oracle-1.json` selected_set_pass=true、
2048/2048 rows exact。query仅2个元素与GPU PyTorch不同（max0.0078125），
pool1个（max0.000244141），scores max0.00240660；因此
value_gate_pass=false、bf16_post_projection_oracle_pass=false、
whole_model_bf16_certified=false。CPU/GPU及不同求和树的舍入边界不能
混为一谈；这个scope只证明该chunk的选中集合，不证明其他layer/chunk
或原始整模型质量。

最终 `qsa-bf16-final-regression-1.log` 串行21项通过、无skip；torch环境
oracle rejection10项、portable格式5项通过。BF16 sparse小图快照与
continuation scope的memcheck/synccheck/racecheck都完成0，0 errors/
0 hazards，日志 `qsa-bf16-snapshot-{memcheck,synccheck,racecheck}-1.log`。
不是完整BF16原生kernel全形状sanitizer认证。原权重、NFS文件、驱动和
默认精度均未改；这一轮完成诊断，原始checkpoint整模型reference门禁
仍未关闭，目标保持进行中。
