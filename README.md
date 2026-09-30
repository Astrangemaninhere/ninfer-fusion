# NInfer Fusion

NInfer Fusion 是 [Neroued/ninfer](https://github.com/Neroued/ninfer) 的社区增强发行版：一个从零写的
C++/CUDA 单卡推理引擎，跑显式注册的 Qwen 系列 checkpoint，并在这条上游路径之上带一套自有的 KV 机制。

它面向单张 NVIDIA GeForce RTX 5090：一张卡、一个常驻模型、启动期定死的并发车道数；入口是本地 CLI
与 OpenAI / Anthropic 兼容的 HTTP API。本文说明这个发行版加了什么、怎么构建、怎么跑，以及哪些部分
还不完整。

本 README 不含未标定的性能读数：需要读数的地方给的是可复现的命令或文档入口。上游的完整 README 逐字
保留在 [docs/upstream-NInfer-README.md](docs/upstream-NInfer-README.md)。

## 本发行版的特色

相对上游，这个发行版把重量放在 KV 与长上下文这一侧，并把"谁能被引擎接受"写清楚：

- **块级 KV**：降档粒度不是"整层"，而是逐格——一个 token 块乘一个文本层，每趟对每格只降一档；预算
  可以按每元素位数给，也可以按字节给。见「块 KV 与逐块降档」。
- **K 与 V 各自成对**：两个平面各自选编解码器、按对定价，读侧标签与定价出自同一张表
  （`--kv-v-codec`、`--kv-tier-scores`、`--kv-k-tier-scores` / `--kv-v-tier-scores`）。
- **E8 格点与位宽轴**：E8 格点形式的窄档把位宽做成一根连续的轴（`rk4v4`、`rk3v4`、`rk2v4`），并带一组
  按对定价；窄档的解码与追加内核尚缺，解析到它的计划会在求解器处按名拒绝，而不是被宽档读法误读。
- **冷层、卸载与召回**：冷层可落主机与磁盘、按水位触发卸载、可召回并按窗口判据具名拒绝；权重量化
  卸载与按层预取在同一套水位里。见「冷层 / 卸载 / 水位 / 预取」。
- **运行时行尺度校准**：一次采集、写回、跨运行持久化，指纹失效时重采（`--kv-row-scale`、
  `--recalibrate`、`NINFER_KV_ROWSCALE`）。
- **子代理机制（设计，接线中）**：KV 里放纯索引，一个索引对应一条界限分明的并发，并发的工作直接注入
  前向而不是被当成文本读回来；指令与回复走非文字通道（`src/spec/inject_channel.h`）。见「服务与并发」。
- **服务面**：`ninfer-serve` 同时给 OpenAI Chat Completions、OpenAI Responses Core 与 Anthropic
  Messages 的路由与模型清单，都支持流式。
- **导入链与产物身份**：封闭的转换器契约、逐对象比对、GGUF 那一路的拆件、以及 `tools/archkit/` 的
  适配与几何/参数两道会红闸；制品自身内嵌 tokenizer、chat template 与媒体前端资源。见「导入与产物」。
- **平台与边界**：构建只接受 `sm_120a`；AMD/ROCm 的适配层与模拟器在树里，但未在真卡上跑过；`sm_70`
  是设计地板。见「平台与构建」与「已知限制」。

## 特性

以下按功能域分组，每组的旋钮都可以单独使用；新机制的旋钮不设时，跑的就是不设它的那条路径
（见「运行与配置」）。

### KV 量化与编解码

KV 在这里不是单个全局档位，而是几层词汇表叠起来的一族编解码器：CLI 的全局档位、逐层存储表，以及
热 / tail / 冷各档的格式表。带常驻编解码器的档位是 `bf16`、`fp8`（按行给尺度）、`int8`（按组给
尺度）、`nvfp4`（K 用 E2M1 码字、V 用 ISO 码字），以及引擎打印为 `iso4e` / `rk4v4` 的那一对；
`iso3` 与 `e8` 是它们的废弃别名，保留一个发布周期。K 与 V 两个平面按元素计位一起付账：一个 DType
同时驱动两个平面。`rk3v4` / `rk2v4` 是 E8 格点形式的窄档，词汇表、档梯成本与平面几何都在，但读窄
码板的解码与追加内核在这棵树里没有，所以解析成它们的计划会在求解器处按名拒绝，而不是被拿去用宽档读
法误读。位预算有合并与分离两种给法，编解码器偏好与质量滑块是求解器的另外两个轴，行尺度则是一条带
持久化的校准闭环。

- 全局档与逐层表：`--kv-dtype`、`--kv-layer-storage SPEC`、
  `--kv-tier-formats hot=...,tail=...,cold=...`、`--nvfp4-mode fusion|pure`、
  `--kv-v-codec iso4e|e2m1`。逐层表里没被写到的槽位保持 BF16，也就是继承全局档；`tail=` 只在重复
  `hot` 时被接受，因为引擎还没有近窗档。`pure` 档禁掉 `nvfp4` / `iso4e` / `rk4v4`。
  两处 `fp8` 拼写落到不同名的编解码器上（全局档是按行 E4M3，逐层存储是按组码板），这是源码自述的
  故意行为，不是抄写错误。`--kv-dtype nvfp4` 是本项目发布的权重档（制品自己记录的权重身份），它
  不能与 `--kv-bit-budget` / `--kv-bits` 同时给——后者的上界会替换掉它填的那张表。
- 位预算与求解：`--kv-bit-budget SPEC`、`--kv-bits`、`--kv-k-bits` / `--kv-v-bits`、
  `--kv-bits-mode joint|split|ceiling`、`--kv-codec-preference`、`--kv-quality-weight`、
  `--kv-tier-scores` / `--kv-k-tier-scores` / `--kv-v-tier-scores`、
  `--kv-score-table show|emit=PATH`。
  K 与 V 的需求凑不出单个档位的层按索引具名拒绝，同时打印按两者较小值算出的可部署方案。
- 成分开关与校准：`--kv-rotation on|off`、`--kv-row-scale auto|off|FILE`、`--recalibrate`、
  `NINFER_KV_ROWSCALE`、`NINFER_KV_DROP_LAYERS`。`auto` 用烘进制品的行尺度表，缺失或表是为别的
  模型、别的 KV 配置烤的就采集一次并写回；首次采集要求关掉图捕获。
- 残余平面：`--kv-residual-layers SPEC`（NVFP4 的二级残余平面，只对 `nvfp4` 有作用；带残余平面的层
  不能再进冷池）。
- 混合层的族：GDN（门控 delta 网络）式的线性注意力层与全注意力层在同一个族里交替，而 GDN 层不携带
  分页 KV——这正是位预算的层区间必须盖住每一层全注意力层的原因。GDN 侧没有专门的命令行旗标，它由
  装载的族决定。

### 块 KV 与逐块降档

块级 KV 是逐格的：一格由一个 token 块与一个文本层定出，每趟对每格只降一档。预算可以按速率给（这一
趟要达到的每元素位数），也可以按绝对口径给（字节或计费块），每趟的触发量由预算与实际总量之差反推；
跨趟保留的东西按页身份重新锚定，而不是靠位置。水位的 cap 与 pass 是分开的：水位之上也执行也打印
（`reason=above-watermark`）。第三根轴让一个平面按它自己那一类的页数定尺寸，于是窄类平面不再按宽类
的页数付账；K 与 V 是成对的，任意对的定价与读侧标签是同一张表的两个面。

- 预算与走道：`NINFER_KV_BLOCK_BUDGET_RATE_X10000`、`NINFER_KV_BLOCK_BUDGET_BYTES`、
  `NINFER_KV_BLOCK_BUDGET_BLOCKS`、`NINFER_KV_DESCENT_CHAIN`、`NINFER_KV_DESCENT_MAX_TIER`、
  `NINFER_KV_DESCENT_KEEP_RECENT_PAGES`（年龄闸：热窗保持 `int8`）。
- 求解与轴：`NINFER_KV_DESCENT_ALLOC`、`NINFER_KV_AXIS3_NARROW_PAGES`、`NINFER_KV_QUALITY_WEIGHT`、
  `--kv-unload-watermark-pages`；serve 侧另有 `--kv-auto-relayout SECS`，按观测量周期性地重新推导
  逐层表。
- 不设即不生效：走道在进入之前就按 `budget_bytes <= 0` 返回默认向量与原像计费；第三根轴的读者对
  未设、空或任何非数字字符都返回 `0`，而 `0` 就是原像形状（该平面用池自己的页数）。

### 冷层 / 卸载 / 水位 / 预取

KV 可以从设备窗口分层放到 host 与磁盘，分三层：决策层决定谁该冷（按观测窗口的 EWMA 与连续稳定性
判断驻留，未观测的层不会被降级），介质层决定冷层放哪（设备窗口、逐页一个钉住 extent 的 host 冷层、
或磁盘），衔接层把策略接到分页介质上，取回、准入与「窗口关不上缺口」各有具名拒绝。水位之上，引擎
会主动卸载语义目录判定不可加载的块，而不是等池溢出。权重卸载把权重放到 host 并按层预取回来，预取
层数有一个下限拒绝。召回回合允许重新预填充，但预填充量有上界，越界在计划期就按名拒绝，绝不截断。

- 冷池与介质：`--cold-policy none|off|window|host|disk|host-then-disk`（`host+disk` 是等价拼写）、
  `--cold-host-bytes`、`--cold-disk-path`、`--cold-disk-bytes`、`--cold-keep-tokens`、
  `--max-cold-pages`。冷档的槽位编码由层 dtype 推导，只有 `int8` 可达。
- 水位与页池：`--kv-unload-watermark-pages`（`0` 是关，未设时由预填充单元推导；环境拼写
  `NINFER_KV_UNLOAD_WATERMARK_PAGES`）、`NINFER_KV_PAGING_PREALLOC`。
- 权重卸载与预取：`--weight-host-bytes`、`--weight-device-arena-bytes`、`--weight-prefetch-layers`、
  `--weight-span-floor-bytes`、`NINFER_W13`、`NINFER_W13_STATS`。
- 召回与外接：`--recall-prefill-tokens` / `NINFER_RECALL_PREFILL_TOKENS`（越界时 stderr 报
  `refused-prefill-budget`，把想要的量与预算一起打出来）、`--ple-sidecar`、`NINFER_RECALL_TEXT`、
  `NINFER_TURN_RECALL`。
- 仪器：`NINFER_COLD_HOST_REFETCH_CENSUS`、`NINFER_COLD_FALLBACK_CENSUS_*`。

决策层只在 serve 侧可达：它由 `--kv-auto-relayout` 的周期驱动，CLI 前端没有决策周期。

### 投机解码与草稿

投机解码是草稿—验证式的一条链，后端是一个枚举（`--spec auto|none|off|mtp|dflash|dflash2|dspark`），
默认 `auto`，所以有一个关回去的拼写。`dspark` 不是第四个后端：它就是 DFlash 的运行时，跑不跑 Markov
头由制品自己的权重身份决定，没有那个身份就按普通 argmax 起草。草稿宽度可以钉住一个宽度、走自适应
阶梯（由生存/代价准则逐轮选档），或者验证一棵树——若干条 rank-path、若干步草稿，节点预算就是这一轮
草稿宽度，所以「树」与「草稿宽度」是同一个量的两种拼写。树路径自带一致性检查：退化成链的树必须逐
token 复现同样的链式草稿。草稿的驻留与视觉驻留一样在进程启动时定死（`--spec` 选投机解码的驻留），
而 35B-A3B 这个目标还支持纯文本的 DFlash。DFlash2 另有自己的一整套运行期：树/束走查与 selector、
特征链与分数表落盘、逐轮调试打印、配对尺度旋钮。

- 后端与宽度：`--spec`、`--draft-tokens`、`--draft-tree L,d`、`NINFER_MTP_ADAPTIVE`、
  `NINFER_MTP_WINDOW_CUT` 族。
- 草稿头与仪器：`--lm-head-draft` / `--no-lm-head-draft`（显式钉住优化头或完整词表头）、
  `NINFER_ACCEPTLOG`、`NINFER_ACCEPTLOG_STREAM_SYNC`、`NINFER_SVIP_THRESHOLD`。
- DFlash2 运行期：`NINFER_DF2FEAT` / `NINFER_DF2FEAT_DIR`、`NINFER_DF2SCORES` /
  `NINFER_DF2SCORES_DIR`、`NINFER_DF2DBG`、`NINFER_DF2SEL`、`NINFER_DF2_PAIR_SCALE`。

### 服务与并发

`ninfer-serve` 把引擎包成一个常驻服务，路由族是 OpenAI Chat Completions、OpenAI Responses Core
（含 input_tokens 与 compact）、Anthropic Messages（含 count_tokens）与模型清单；四者都支持流式、
工具调用回传、本地响应状态与用量记账，而解析出的工具调用是交回客户端的——引擎自己不执行工具。批的
性质在启动期定死：车道数与排队上限启动期给出，运行期不做请求抢占、不做优先级/QoS、不做活动请求
换出；一批解码走精确批次的图捕获，并配一条关掉图的回退。前缀复用的单位是一个检查点——KV 加上该
提示词前沿的完整续写状态；引擎在领先的系统提示处发布共享前缀候选，并在请求之间复用相容前缀，命中量
与走了哪条复用路径都被记账。压力之下，规划器按立即恢复的工作量与后续复用代价权衡设备驻留、钉住的
host 状态与换出，活动请求保留自己的完成预留。预填充单元的归属由带宽治理器决定：`dynamic` 让它在
给定上界内自己装一个单元，并在解码延迟高于它自己测得的噪声底时收缩它；`manual` 把它钉死在上界。

- 监听与身份：`--host`、`--port`、`--api-key`、`--model-id`、`--cors`。
- 车道与排队：`--max-concurrency`、`--max-pending-requests`、`--pending-timeout-ms`。
- 驻留与状态：`--device-state-slots`、`--host-state-slots`、`--host-kv-mib`、
  `--context-cost-presets`、`--max-request-mib`；观测面有 `NINFER_WS_DUMP`、
  `NINFER_WS_HEADROOM_PCT`、`NINFER_ARENA_TRACE`、`NINFER_KV_WINDOW_TOKENS`。
- 请求默认值：`--default-max-tokens`、`--default-thinking-budget`、`--preserve-thinking`。
- 预填充与图：`--prefill-chunk`、`--prefill-chunk-mode dynamic|manual`、`NINFER_FT_BW_GOV`、
  `NINFER_FT_BW_TRACE`、`--no-cuda-graph`、`--graph-capture-ceiling`。
- 前缀与重排：`--no-prefix-reuse`（关掉相容前缀缓存，默认开，且不能与上下文缓存的容量选项同时给）、
  `--no-auto-system-shared-prefix`、`--max-private-continuations`、`--max-shared-prefixes`、
  `--max-long-anchors-per-continuation`、`--kv-auto-relayout SECS`。
- 日志与存储：`--request-log-jsonl`、`--response-store-max-records`、`--response-store-max-mib`、
  `--log-stats-interval-ms`。
- 运行中输入与观测：`--append-context-text <text>`（按制品自己的 tokenizer 编码一段文本并追加到
  运行中的请求上）、`--print-prompt-ids`、`--print-token-ids`、`--ft-stats` / `NINFER_FT_STATS`
  （FreeToken 逐层注意力能量观测，默认关；它的消费者是 serve 侧的周期重排）。
- 分层流水线：`--stage-layers SPEC` 把文本层轴切成阶段，`--stage-handoff DIR` 给边界隐状态的交接
  目录；错形、不是覆盖、rank 轴推不出来的 SPEC 分别按名拒绝，虚拟设备那道守卫也仍然拒绝流水线并行。
  `--stage-handoff-cut` 是负对照，不是功能。

#### 子代理机制（设计，接线中）

这是一项正在接线的设计，今天还不是一个可跑的模式。设计本身是：KV 里只放纯粹的索引，每一条索引对应
一条独立且界限分明的并发；主 KV 只记载每一条并发所承担的部分；并发的工作直接注入主过程，而不是把
它当报告再读回来；指令与回复都不过文字。树里已在的部分有两面：

- **索引作为草稿来源**：`NINFER_INDEX_DRAFTS` 是进程级开关，未设即关，而关就是今天的引擎——那次
  咨询根本不会被调用，每个草稿槽保持引擎自己会写的值。打开后是每条车道每轮解码咨询一次，发生在
  host 上、该轮发射之前，并且在任何图捕获体之外；它返回索引从头填上的草稿槽数，返回 `0` 意味着
  这一轮的草稿逐字节仍是引擎自己的。树回合里它按名被记成 `refused_tree`：索引只能提供一条链，把
  索引 token 当树列去验证会报出一棵从未建过的树。
- **非文字通道**：`src/spec/inject_channel.h` 是声明面（host-only，可用普通 `g++` 单测），
  `src/targets/qwen3_6/impl/runtime/inject_ingress.h` 是引擎侧——一份声明在那里变成一次写进前向的
  写：每个预填充分块做一次连续拷贝，写进 input-embedding 矩阵的声明列（ingest）或从其中拷出
  （egress），并在消费任何东西之前按名拒绝与声明列冲突的分块。命令行面是 `--inject-spec`，环境拼写
  `NINFER_INJECT_SPEC`；未设时它第一行就返回，不写任何字节。

把索引草稿与其他全开项一起打开时，引擎会在短提示上按名失败（`phase=decode reason: index query
start`），关掉它则正常。接线在、模式未成：这是设计中的机制，不是加速手段。

### 多模态

结构化消息内容除了文本，还可以带 image / image_url 与 video / video_url 片段，媒体来源可以是本地
路径、HTTP(S) URL 或 base64 data URI。`--vision` 打开媒体输入并把固定的 Vision GPU 分配装上：视觉
相关的驻留是启动期定死的，后续请求不能临时开启启动时省掉的能力。serve 侧的媒体取回与前处理有自己
的缓存、实时预算与前处理线程数，取回路径另有一个编译期开关。

- 前端：`--vision`。
- serve 侧：`--media-cache-mib`、`--media-live-mib`、`--media-preprocess-threads`。
- 编译期：`NINFER_BUILD_MEDIA_ACQUIRE`（构建开关，不是运行期旋钮）。契约在
  [docs/sm120a-quantized-kv-vision.md](docs/sm120a-quantized-kv-vision.md)。

### 导入与产物

把外部 checkpoint 变成可跑的 `.ninfer` 制品的链路收在一支前门脚本里
（`tools/convert/import_model.py`）：注册的转换器是封闭、字节钉死的契约，导入之后由编排脚本自动跑
转换，验收闸是一个逐对象比较两个 `.ninfer` 的工具，按退出码判「这个 build 是否与参考一致」。GGUF
族的能力拆成几件：抽取、K-quant 块格式的尺寸与反量化、旋转契约（「谁施加这个旋转」只有一个决策
点），以及张量名到本树名字的映射；逐族各有一套转换器与自查脚本。模型适配还有一条工具链侧的路：从
spec 生成目标骨架、自动适配管线，以及几何覆盖与参数完备两道会红的门（是门，不是提示）。制品带自己
的身份：草稿头跑不跑 Markov 头由制品的权重身份决定，不由旗标决定；每个制品还内嵌它注册目标的
tokenizer、chat template 与媒体前端资源。

- 导入与验收：`tools/convert/import_model.py`、`convert_runner.py`、`artifact_diff.py`、
  `compress_probe.py`（测制品的无损压缩余量）。
- GGUF 族：`gguf_extract.py`、`gguf_kquant.py`、`gguf_hadamard.py`、`gguf_fold_back.py`、
  `gguf_fold_route.py`、`gguf_names.py`。
- 适配与离线镜像：`tools/archkit/`（`adapt.py`、`check_geometry.py`、`check_params.py`、
  `kv_budget_mirror.py`、`kv_tier_matrix.py`、`kv_auto_allocate.py`）。
- 接口与契约：`NINFER_EXPORT_HEAD_DIR`、`NINFER_KV_CALIB_DIR`、`--ple-sidecar`；
  [docs/maintainer/modelopt-nvfp4-import.md](docs/maintainer/modelopt-nvfp4-import.md)、
  [docs/maintainer/artifact-container.md](docs/maintainer/artifact-container.md)、
  [docs/maintainer/tensor-formats.md](docs/maintainer/tensor-formats.md)、
  [docs/features/importers.md](docs/features/importers.md)。

注册目标是一个封闭集合：没有运行期模型发现，也没有未注册 checkpoint 的兜底。

### 平台与构建

构建只接受 `sm_120a`，也就是单张 RTX 5090。能力面由构建期发布：`--capability-report` 不需要制品就
能回答「这个构建会拒绝什么、为什么」。树里另有 AMD / ROCm 一侧的适配层与模拟器，以及 `sm_70` 这个
设计地板；本项目的开发与实测路径是 WSL2（Linux），Docker 用法见「构建」。

- 构建与自报：`-DCMAKE_CUDA_ARCHITECTURES=120a`、`--capability-report`。
- 模拟（在没有对应硬件时跑适配层用）：`NINFER_SIM_ARCH`、`NINFER_SIM_VENDOR`、
  `NINFER_GFX906_COMPAT`。

平台的边界写在「已知限制」里。

## 模型与产物

制品身份决定确切的模型与权重档；每个制品内嵌它注册目标的 tokenizer、chat template 与媒体前端资源。
本仓库不附带权重，权重的许可见「许可」。

| 模型 | 权重 | 制品 | 下载与模型卡 |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

用 Hugging Face CLI 取下示例要用的那一份：

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

除上表已发布的制品外，`src/targets/` 下的族按构建归属与注册状态一起列在这里——**部分支持的
也在表内**，并注明它到哪一步：

| 族 | 构建归属 | registry | 备注 |
|---|---|---|---|
| `qwen3_6` | 编进 engine（核心族） | — | |
| `qwen3_6_27b` | 编进 engine | 认识 | |
| `qwen3_6_35b_a3b` | 编进 engine | 认识 | |
| `muse_glimmer_30b` | 编进 engine | 认识 | |
| `qwen3_5_9b` | 编进 engine | 认识 | |
| `spark_x2_5_4b` | 有条件加入（在条件分支里） | 认识 | |
| `qwen4_exp` | 编进 engine | — | 只有身份注册的骨架 |
| `gemma4_31b` | 编进 engine | — | 骨架 |
| `qwen3_vision` | 不默认构建（须点名才建） | — | 骨架 |
| `qwen3_8_flash_next` | 不默认构建（须点名才建） | — | 有完整实现，但未注册进 engine |

骨架与未注册的这几族**不能算作在引擎路径上跑过的目标**；发布出来、可下载的是上表那三族的五种制品。

## 构建

NInfer 需要 64 位 Linux、一张 NVIDIA GeForce RTX 5090、CUDA Toolkit 13.1 或更新、CMake 3.28 或
更新、支持 C++20 的宿主编译器、Ninja、`pkg-config`、FFmpeg 开发库（`libavformat`、`libavcodec`、
`libavutil`、`libswscale`）与 `libcurl`。构建会拒绝除 `sm_120a` 之外的 CUDA 架构。

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

测试、基准与维护者工具不进默认构建。把产品二进制装到一个前缀：

```bash
cmake --install build --prefix /path/to/prefix
```

`--prefix` 可选；不给就用配置期选定的 `CMAKE_INSTALL_PREFIX`。安装树是 `bin/`（三个可执行文件）加
`share/doc/ninfer/`（本 README、`LICENSE`、`CONTRIBUTING.md`，以及一份配置期生成的
`ninfer-install-manifest.txt`）。没有打包好的二进制发行版、没有安装器、没有发布归档：`cmake
--install` 就是部署路径；二进制也可以直接从源码构建树里跑。树里不构建共享库、也不导出头文件，所以
安装树里没有可供链接的 SDK。

本项目的开发与实测路径是 WSL2，常用的配置是显式给出 nvcc 与架构：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build -j
```

### Docker

在一台装了 NVIDIA Container Toolkit 的宿主上构建运行时镜像：

```bash
docker build --tag ninfer:local .
```

把下好的模型目录挂进去，跑与上面同一个服务配置：

```bash
docker run --rm \
  --gpus all \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context N \
  --kv-capacity N \
  --max-concurrency N \
  --kv-dtype fp8 \
  --device-state-slots N \
  --host-state-slots N \
  --host-kv-mib N \
  --spec mtp \
  --lm-head-draft \
  --preserve-thinking
```

## 快速开始

二进制默认从源码构建树里跑：`./build/apps/...`。示例里的 `N` 是留给你的容量与长度，其余参数可以直接
照抄。

一次性请求：

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context N \
  --max-new N \
  --kv-dtype fp8 \
  --spec mtp \
  --lm-head-draft
```

回答内容写到 stdout，加载进度、推理过程、时间、吞吐、内存与投机解码统计写到 stderr。结构化消息内容
用 `--messages FILE` 与 `--vision` 走图像 / 视频，示例请求在 [examples/cli/](examples/cli/)。

常驻服务：

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context N \
  --kv-capacity N \
  --max-concurrency N \
  --kv-dtype fp8 \
  --device-state-slots N \
  --host-state-slots N \
  --host-kv-mib N \
  --spec mtp \
  --lm-head-draft \
  --preserve-thinking
```

`--max-context` 是每条序列的逻辑上限；`--kv-capacity` 给共享的主文本 KV 池定尺寸，活跃请求与留存
前缀共用这一个池，`auto` 会在启动期按权重之后剩下的内存解出合法容量并留一份定尺寸余量，显式容量在
进程生命周期内固定。

发一个 OpenAI 风格的请求：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}]
  }'
```

## 运行与配置

两条通用规则：**旗标胜过同名环境变量**；**新机制的旋钮不设时等于不设它**——逐格降档的 OFF 路径是
`budget_bytes <= 0`，它在任何走道进入之前就返回默认向量与原像计费；第三根轴的读者对未设、空或任何
非数字字符都返回 `0`，而 `0` 就是原像形状。

下面是引擎自己的用法行（`ninfer` 前端，取自 `--help` 打印的那份文本）：

```text
usage: ninfer <model.ninfer> (--prompt <text>|--messages <messages.json>)
       [--max-context N] [--kv-capacity N|auto] [--max-concurrency N] [--prefill-chunk N] [--max-new N]
       [--prefill-chunk-mode dynamic|manual]
       [--device N]
       [--kv-dtype bf16|int8|fp8|nvfp4|iso4e|iso3|rk4v4|e8] [--kv-layer-storage SPEC] [--kv-bit-budget SPEC] [--spec auto|off|mtp|dflash|dflash2|dspark|none --draft-tokens N]
       [--stage-layers SPEC] [--stage-handoff DIR] [--stage-handoff-cut]
       [--kv-residual-layers SPEC]
       [--kv-bits B] [--kv-k-bits BK --kv-v-bits BV] [--kv-bits-mode joint|split|ceiling]
       [--kv-codec-preference CODEC[,CODEC...]]
       [--kv-quality-weight W] [--kv-tier-scores FILE|INLINE]
       [--capability-report]
       [--kv-score-table show|emit=PATH]
       [--kv-tier-formats hot=auto|bf16|int8,tail=...,cold=...] [--nvfp4-mode fusion|pure]
       [--kv-rotation on|off] [--kv-row-scale auto|off|FILE] [--recalibrate]
       [--kv-v-codec iso4e|e2m1]
       [--yarn]
       [--lm-head-draft]
       [--no-lm-head-draft]
       [--temperature F] [--top-p F] [--top-k N] [--min-p F]
       [--presence-penalty F] [--frequency-penalty F] [--seed N] [--greedy]
       [--stop-token-id N]... [--stop <text>]... [--reasoning-stop <text>]...
       [--print-prompt-ids] [--print-token-ids] [--no-thinking] [--thinking-budget N]
       [--reasoning-effort low|medium|xhigh] [--vision]
       [--cold-policy none|off|window|host|disk|host-then-disk|host+disk]
       [--cold-keep-tokens N]
       [--max-cold-pages N] [--kv-unload-watermark-pages N]
       [--recall-prefill-tokens N]
       [--append-context-text <text>]
       [--cold-host-bytes N[g|m|k]]
       [--cold-disk-path DIR] [--cold-disk-bytes N]
       [--ple-sidecar DIR]
       [--weight-host-bytes N] [--weight-device-arena-bytes N]
       [--weight-prefetch-layers N] [--weight-span-floor-bytes N]
       [--no-cuda-graph] [--graph-capture-ceiling N]
       [--ft-stats on|off] [--inject-spec PATH]

Streams answer content to stdout and reasoning plus diagnostics to stderr.
```

`--kv-capacity auto` 会留一份定尺寸余量；`--cold-policy` 的 `host+disk` 是 `host-then-disk` 的
等价拼写。两个不需要模型与提示词的入口也在上面：`--capability-report` 与 `--kv-score-table`。
采样默认值来自装载的模型与思考模式，旗标只覆盖被点名的那几个字段。

机制到旋钮的对应：

| 机制域 | 旋钮（旗标 / 环境） |
|---|---|
| KV 量化与编解码 | `--kv-dtype`、`--kv-layer-storage`、`--kv-tier-formats`、`--nvfp4-mode`、`--kv-v-codec`、`--kv-bit-budget`、`--kv-bits`、`--kv-k-bits`、`--kv-v-bits`、`--kv-bits-mode`、`--kv-codec-preference`、`--kv-quality-weight`、`--kv-tier-scores`、`--kv-rotation`、`--kv-row-scale`、`--recalibrate`、`--kv-residual-layers`、`NINFER_KV_ROWSCALE`、`NINFER_KV_DROP_LAYERS` |
| 块 KV 与逐块降档 | `NINFER_KV_BLOCK_BUDGET_RATE_X10000`、`NINFER_KV_BLOCK_BUDGET_BYTES`、`NINFER_KV_BLOCK_BUDGET_BLOCKS`、`NINFER_KV_DESCENT_CHAIN`、`NINFER_KV_DESCENT_MAX_TIER`、`NINFER_KV_DESCENT_ALLOC`、`NINFER_KV_DESCENT_KEEP_RECENT_PAGES`、`NINFER_KV_AXIS3_NARROW_PAGES`、`NINFER_KV_QUALITY_WEIGHT` |
| 冷层 / 卸载 / 水位 / 预取 | `--cold-policy`、`--cold-host-bytes`、`--cold-disk-path`、`--cold-disk-bytes`、`--cold-keep-tokens`、`--max-cold-pages`、`--kv-unload-watermark-pages`、`--weight-host-bytes`、`--weight-device-arena-bytes`、`--weight-prefetch-layers`、`--weight-span-floor-bytes`、`--recall-prefill-tokens`、`--ple-sidecar`、`NINFER_KV_UNLOAD_WATERMARK_PAGES`、`NINFER_KV_PAGING_PREALLOC`、`NINFER_W13` |
| 投机解码与草稿 | `--spec`、`--draft-tokens`、`--draft-tree`、`--lm-head-draft`、`--no-lm-head-draft`、`NINFER_MTP_ADAPTIVE`、`NINFER_MTP_WINDOW_CUT` 族、`NINFER_ACCEPTLOG`、`NINFER_DF2` 族 |
| 服务与并发 | `--host`、`--port`、`--api-key`、`--model-id`、`--cors`、`--max-concurrency`、`--max-pending-requests`、`--pending-timeout-ms`、`--prefill-chunk`、`--prefill-chunk-mode`、`--device-state-slots`、`--host-state-slots`、`--host-kv-mib`、`--context-cost-presets`、`--max-request-mib`、`--no-prefix-reuse`、`--no-auto-system-shared-prefix`、`--max-private-continuations`、`--max-shared-prefixes`、`--max-long-anchors-per-continuation`、`--request-log-jsonl`、`--response-store-max-records`、`--response-store-max-mib`、`--kv-auto-relayout`、`--ft-vram-axis`、`--stage-layers`、`--stage-handoff`、`--stage-handoff-cut`、`NINFER_FT_BW_GOV`、`NINFER_INDEX_DRAFTS`、`--inject-spec`、`NINFER_INJECT_SPEC` |
| 多模态 | `--vision`、`--media-cache-mib`、`--media-live-mib`、`--media-preprocess-threads`、`NINFER_BUILD_MEDIA_ACQUIRE`（编译期） |
| 导入与产物 | `tools/convert/` 一族、`NINFER_EXPORT_HEAD_DIR`、`NINFER_KV_CALIB_DIR`、`--ple-sidecar` |
| 平台与构建 | `-DCMAKE_CUDA_ARCHITECTURES=120a`、`--capability-report`、`NINFER_SIM_ARCH`、`NINFER_SIM_VENDOR`、`NINFER_GFX906_COMPAT` |
| 质量与仪器 | `ninfer-perplexity`、`--kv-score-table`、`--ft-stats`、`NINFER_FT_STATS`、`NINFER_ACCEPTLOG`、`--print-prompt-ids`、`--print-token-ids` |

serve 侧还有自己的一批旗标（监听、车道、驻留、前缀、日志、媒体与周期重排），见「服务与并发」；
serve 的 `--help` 用法行没有列全自己 parser 里注册的冷窗族与权重卸载族旗标，照 `--help` 读会以为
它们不可达，实测两族都能过 parser 并进入装载。分数表里那一列质量分是 PRIOR，不是测量——工具自己会
在输出里说明这一点。

## 已知限制

- **平台覆盖**：构建只接受 `sm_120a`；树里虽有 AMD / ROCm 的适配层与模拟器，但没有在真 AMD 卡上跑
  过，Vulkan 在本树没有实现，`sm_70` 只是设计地板且模拟覆盖不完整。
- **长上下文的量级**：把上下文推到族的原生窗口之外要同时解开三道并列的环——族的原生上下文门、权重
  之后剩下设备内存的缺口、以及把 KV 分层放出去之后冷层仍装不下的页数；这三道环今天都还没有走通。
  `--yarn` 本身可用，它把 rope 域扩到族的原生窗口之外并把 YaRN 的注意力缩放折进 sincos 表；它改的
  是被采集的 K 经过的 rope 域，所以它进入行尺度指纹——没点 `--yarn` 烤的行尺度表，不能被点了
  `--yarn` 的运行采用。
- **子代理机制**：索引 + 并发 + 非文字注入这套设计目前只落地了接线，作为一个可跑的模式还不存在；把
  索引草稿与其他全开项一起打开时，引擎会在短提示上按名失败。
- **窄档的读侧**：窄档对的定价按构造钉死（代码里由 `static_assert` 保证），但读侧在同一张表里标成
  `Reserved`，宽档的格点解码器没有调用者；`rk3v4` / `rk2v4` 可以选择，不可跑。
- **逐格走道与质量滑块的接口**：`NINFER_KV_QUALITY_WEIGHT` 只在按层天花板与分离求解器上生效，没有
  接到逐格走道；`NINFER_KV_DESCENT_ALLOC=solve` 选无状态逐格求解，但没有代价表生产者，所以它按构造
  塌回原像；预算标尺头文件（`src/product/kv_block_budget_stage.h`）里还有一个名字看起来像旋钮、其实
  是编译期 `#define` 的标尺，它设不了。
- **预填充比未改动的引擎慢**：这是唯一未过的验收项，判据是「快或等，绝不更慢」，成因尚未识别。
- **长上下文会侵蚀混档**：降档占比随上下文变长而下降，一次退役就能把某一臂的降档一次抹掉，而且走道
  欠报自己的降档。
- **解码列不作验收**：解码侧实测的仪器噪声比这套机制能产生的效应还大。
- **回读从未演练**：取回盘点器自己的头文件写明，一次什么都不回读的运行什么都不打印，所以「零回读」
  今天还只是一句注释，不是一个读数。
- **冷池与残余平面的互斥**：冷档的槽位编码由层 dtype 推导，只有 `int8` 可达；残余平面是 NVFP4 的
  非默认路径，带残余平面的层不能再进冷池。
- **MTP 层没有逐层表**：逐层存储表到不了 MTP 层。
- **草稿宽度的族上限**：每个族的草稿宽度上限定义在族里，CLI 侧今天仍按公共上界硬编码，所以族上限
  在命令行上到不了；把自适应草稿按环境变量强行打开的那条路，引擎自己记作临时逃生舱。
- **组合互斥**：`--vision` 与 DFlash 没有被共同验证，CLI 与 serve 都按名拒绝这个组合；DFlash 与
  DFlash2 的模型视图互斥；`--reasoning-effort` 与 `--thinking-budget` 都不能与 `--no-thinking`
  同时给。
- **多设备**：跨 rank 的分片算术与传输契约在树里，但没有调用路径（host-only）。
- **服务侧用法文本不完整**：见「运行与配置」的一节。
- **悬空的契约文档**：树里指向 TP2 / YaRN 长上下文的契约文档还不存在，那条接口目前只能读相关头文件。
- **未验证的目标族**：视觉族与 FlashNext 族在树里有实现或骨架，但都没有以引擎路径跑过（一个未注册进
  engine，一个是只有身份注册的骨架）。

## 工具

| 程序 | 用途 |
|---|---|
| `ninfer` | 本地 CLI：打开一个 `.ninfer` 制品并跑一次性请求（`--prompt` 或 `--messages`）；也承载两个不需要模型与提示词的入口——`--capability-report`（这个构建的能力面）与 `--kv-score-table show\|emit=PATH`（规划器会用的分数表，连每一列的真实出处一起打出来）。 |
| `ninfer-serve` | 常驻 HTTP 服务：OpenAI / Anthropic 兼容路由、批与并发、前缀复用、周期重排与媒体前处理。 |
| `ninfer-perplexity` | 离线困惑度工具：用同一份 Text 模型与可选 KV 档位跑语料，跑完写一份完整的 JSON 记录（`--corpus`、`--quick`、`--kv-dtype`）。 |
| `ninfer-cufree` | 在没有可用 CUDA 栈的机器上也能启动的前门：它不链接任何 ninfer 库、不查询设备，只报加载器与文件系统答得出的两件事——依赖的 soname 找不找得到、加速器内核接口在不在。 |

另有 `ninfer-hostpath`：同样不依赖 CUDA，用引擎自己的读取器打开一个真实制品并回答引擎自己的门，它
不产出 token。安装规则（`cmake --install`）里的只有 `ninfer`、`ninfer-serve`、`ninfer-perplexity`
三个；`ninfer-cufree` 与 `ninfer-hostpath` 要显式点名构建，不在安装集里，这是有意为之。

`tools/` 下还有一批维护者工具：`tools/convert/`（导入与转换）、`tools/bench/`（基准与 serve TTFT）、
`tools/perplexity/`（语料准备）、`tools/gui/`（serve、转换与 RAG 的极简控制台）、
`tools/archkit/`（目标骨架生成与几何、参数门）、`tools/test_kv/`。它们不属于下载即用的路径；入口见
[tools/README.md](tools/README.md)。

## 文档

- [README_FEATURES.md](README_FEATURES.md) —— 中文功能总览：逐个功能的机制与出处。
- [README.en.md](README.en.md) —— 同一份内容的英文版。
- [docs/upstream-NInfer-README.md](docs/upstream-NInfer-README.md) —— 上游 README 逐字保留（已标注
  对本树过时）。
- [docs/features/](docs/features/README.md) —— 按机制分册：CLI 旗标、serve 旗标、KV 压缩、KV 分层、
  投机解码、导入、内核、仪器、验证、未完成清单，以及
  [withdrawn-numbers-2026-09-30.md](docs/features/withdrawn-numbers-2026-09-30.md)。
- 引擎与运维：[docs/README.md](docs/README.md)、[docs/cli.md](docs/cli.md)、
  [docs/serving.md](docs/serving.md)、[docs/performance.md](docs/performance.md)、
  [docs/perplexity.md](docs/perplexity.md)、[docs/maintainer/](docs/maintainer/)（契约文档）。
- 其他：[examples/cli/](examples/cli/)（示例请求）、[CONTRIBUTING.md](CONTRIBUTING.md)，
  以及各前端自己的 `--help`。
- 仓库根的研究笔记：[RESEARCH-EXTERNAL.md](RESEARCH-EXTERNAL.md)、
  [RESEARCH-FLASHNEXT.md](RESEARCH-FLASHNEXT.md)、[RESEARCH-FREETOKEN.md](RESEARCH-FREETOKEN.md)、
  [PREFILL-OPT.md](PREFILL-OPT.md)。

## 许可

NInfer 以 [Apache License 2.0](LICENSE) 发布。

已发布的制品派生自 [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)、
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 与
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)。Qwen3.6-27B 的 NVFP4 制品还
用了 [rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm)
的固定打包权重，Qwen3.8-27B 的 NVFP4 制品还用了
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) 的混合 FP8/NVFP4
权重。这些上游仓库以 Apache-2.0 分发。随树附带的第三方依赖保留各自在 `third_party/` 下的许可文件。
