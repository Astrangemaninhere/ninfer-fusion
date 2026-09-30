# `ninfer` 命令行全量表

**权威来源**：`apps/cli/options.cpp` 的 `parse_options()`（`:311`）。用法文本由 `usage_text()` 打
（`:146-309`），口径是**以 parser 为准**：用法文本与 parser 不一致时，能跑的是 parser。

**计数（实测）**：`grep -n 'arg == "--' apps/cli/options.cpp` 得 **67 行**，其中 `:697` 是三元表达式
`arg == "--stop" ? ... : ...`（**假阳性，不是 case 分支**）⇒ **66 个 case 分支**；`--stop` /
`--reasoning-stop` 共用一个分支（`:690`）⇒ **67 个旗标名**；再加 `-h` / `--help`（`:313`，**只在
`argv[1]`**）⇒ **68 个入口**。

**2026-09-22 复核（`accept34` 线重测，同一棵树的当天读数）**：同一条 `grep -n` 今天得 **68 行**，
三元表达式在 `:714`（不是 `:697`）⇒ **67 个 case 分支**；共用分支在 `:707` ⇒ **68 个旗标名**；
加 `-h` / `--help` ⇒ **69 个入口**。`apps/cli/options.cpp` 今天 **1045** 行（原记 1016），
**本册其余行号以 2026-09-20 为基准，请按同一条 `grep -n` 重测后再引用**。

## 1. 位置参数与两个「不需要模型」的入口

| 项 | 语义 | 出处 |
|---|---|---|
| `<model.ninfer>` | 第一个非旗标参数是**位置**，不是旗标 | `apps/cli/options.cpp:331-334` |
| 模型缺失 | 用过旗标但从未点名模型时，**按名报错** | `:736-739` |
| `--help` / `-h` | 打印用法后返回；只在 `argv[1]` 生效 | `:313-316` |
| `--capability-report` | 打印**这个二进制被编译到**的 arch 清单与逐格式 tensor-core 底线，带 kernel 引证；**不探测设备** | `:479-484`；用法 `:207-215` |
| `--kv-score-table show\|emit=PATH` | 打印/导出 planner 会用的罚分表，含每列出处；「出厂质量列是先验不是测量」 | `:485-490`；用法 `:216-222` |

这两个入口可以**不给模型也不给提示词**（`--kv-score-table` 还能占 `argv[1]` 位置，`:323-324`），
判断在 `:976-979`。

## 2. 输入与上下文（8 条）

| 旗标 | 语义 | parser | 默认 / 规则 |
|---|---|---|---|
| `--prompt <text>` | 单轮提示 | `:358` | 与 `--messages` **恰一**（`:971-978`） |
| `--messages <json>` | 结构化消息文件；content 接受 text / image / video part，源可为本地路径、HTTP(S) URL、base64 data URI | `:360` | 同上；用法 `:281-282` |
| `--max-new N` | 生成 token 上限 | `:362` | 默认 **128**（`apps/cli/options.h:26`） |
| `--max-context N` | 上下文 token 上限 | `:364` | 默认 **2048**（`apps/cli/options.h:27`）；非 0 校验在 `src/targets/registry.cpp:40-42` |
| `--kv-capacity N\|auto` | KV 容量 | `:366` | 不给时由 `explicit_capacity(max_context)` 推出（`:839-841`）；`auto` 留 1024 MiB 余量（`include/ninfer/types.h:233`，用法文本 `:286-288`）；显式值必须非 0（`:983-986`） |
| `--prefill-chunk N` | 预填充块大小（也是上限） | `:369` | 默认 **3072**（`apps/cli/options.h:29`）；**必须是 128 的倍数**（`:980-982`） |
| `--prefill-chunk-mode dynamic\|manual` | 谁拥有预填充单元 | `:371` | 默认未设（`apps/cli/options.h:34` 是 `std::optional`）；先看 env `NINFER_FT_BW_GOV` 再落 `Dynamic`，解析一次在 `src/runtime/engine/engine.cpp:86`；**flag 胜过 env** |
| `--device N` | 设备序号 | `:376` | 默认 **0**（`apps/cli/options.h:35`） |

## 3. KV 档位与位预算（17 条）

| 旗标 | 语义 | parser | 默认 / 互斥（行号） |
|---|---|---|---|
| `--kv-dtype` | 给**整个** KV 栈一个存储档（`bf16/int8/fp8/nvfp4/iso4e/iso3/rk4v4/e8`） | `:378` | 默认 `BFloat16`（`apps/cli/options.h:37`）；拼写表 `:93-107`；废弃别名 `iso3`、`e8` 在 `:104-105`。**与位上限互斥**（`:872-879`） |
| `--kv-layer-storage SPEC` | 直接给逐层存储表，绕过求解器 | `:491` | 与 `--kv-bit-budget` 互斥（`:859-863`）、与 K/V 位宽互斥（`:880-884`）、与 `--kv-codec-preference` 互斥（`:936-941`） |
| `--kv-bit-budget SPEC` | 每个 KV 元素的位上限；或逐层区间 `"0-7:8,8-15:4.5"`（区间必须铺满全部 full-attention 层；本 variant 是 16 层，不是 64 层） | `:381` | 默认 `0.0` = 未命名（`apps/cli/options.h:55`）；区间形式在 `:384-395` |
| `--kv-bits B` | K/V 的 **JOINT** 上限（全栈一个） | `:403` | 默认 `0.0`（`apps/cli/options.h:67`）；与 `--kv-k-bits/--kv-v-bits` 互斥（`:914-919`） |
| `--kv-k-bits BK` / `--kv-v-bits BV` | **SPLIT** 形式：K、V 各自逐层分层 | `:408` / `:412` | 默认 `0.0`（`apps/cli/options.h:68-69`） |
| `--kv-bits-mode joint\|split\|ceiling` | SPLIT 请求的**读法** | `:416` | 默认 `Split`（`apps/cli/options.h:71`）；无 K/V 请求即拒绝（`:885-891`）；给了 `--kv-bits` 时只接受 `joint`（`:942-949`） |
| `--kv-quality-weight W` | 速度/质量滑杆（W=0 走快路径，W=1 取精度取向） | `:396` | 默认 `-1.0` = 未命名（`apps/cli/options.h:60`）；**无上限请求即拒绝**（`:892-900`） |
| `--kv-tier-scores FILE\|INLINE` | 换掉两列罚分表 | `:401` | **必须有 `--kv-quality-weight`**（`:903-913`） |
| `--kv-k-tier-scores` / `--kv-v-tier-scores` | 只给 K 平面 / 只给 V 平面 | `:473` / `:476` | 默认空（`apps/cli/options.h:74-75`）；JOINT 读法下拒绝（`:951-961`） |
| `--kv-codec-preference CODEC[,...]` | 同 bit 宽度下换一种量化：给候选一个偏好**顺序** | `:420` | 默认空 = 出厂 pack 顺序（`apps/cli/options.h:87`）；**必须配 `--kv-bits`**（`:927-935`）；与 `--kv-layer-storage` 互斥（`:936-941`）；候选表由 `product::kv_gear_candidate_list()` 现算并**拼进 help**（`:183-184`） |
| `--kv-residual-layers SPEC` | NVFP4 的第二级残差 plane：一层从 4 plane 变 8 | `:494` | 默认空（`apps/cli/options.h:52`）；语法同族 64 槽裸层号表 `"2-5"`/`"0,3,7"`；**只对 NVFP4 有效**，带残差的层不能进冷池（用法 `:160-166`） |
| `--kv-tier-formats SPEC` | 档位词汇表：`hot=` / `tail=` / `cold=` | `:502` | 默认空（`apps/cli/options.h:97`）；parse 期只查词汇表自身规则（`:998-1004`），逐层落地要层数、在 planner 里做 |
| `--nvfp4-mode fusion\|pure` | `pure` 禁止 nvfp4/iso4e/rk4v4 | `:509` | 默认 fusion（`apps/cli/options.h:99`） |
| `--kv-rotation on\|off` | K 写入与 Q 读取的 SO(4) 旋转开关 | `:519` | 默认 on/auto（`apps/cli/options.h:101`）；核侧读 `NINFER_KV_ROTATION`（`src/ops/kernel/gqa_isoquant_rot.cu:117`） |
| `--kv-v-codec iso4e\|e2m1` | NVFP4 档的 V 平面编码 | `:541` | 默认 `KvVCodec::Iso3`（`apps/cli/options.h:109`）；`iso3` 是**废弃拼写**并打告警（`:551-556`）；V 残差 plane 或冷池激活时拒 `e2m1`（用法 `:244-245`） |

## 4. 投机解码（5 条）

| 旗标 | 语义 | parser | 默认 / 规则 |
|---|---|---|---|
| `--spec auto\|none\|off\|mtp\|dflash\|dflash2\|dspark` | 投机后端 | `:565` | 默认 **`Auto`**（`apps/cli/options.h:42`）；名字表 `src/product/speculative_options.h:11-45` |
| `--draft-tokens N` | 草稿宽度 k；**`0` = 自适应** | `:567` | 接受 0（`:576` 传 `allow_zero=true`）；域校验在 `src/product/speculative_options.h:63-115`（mtp `[0,15]`、dflash/dflash2 `[1,15]`） |
| `--draft-tree L,d` | MTP **树**验证：L 条 rank-path × d 步，节点预算 `L*d` 就是草稿宽度 | `:578` | `L∈[1,16]`、`d∈[1,15]`（`:593-596`）；**必须配 `--spec mtp` 且不得同时给 `--draft-tokens`**（`:831-837`） |
| `--lm-head-draft` | 草稿头用优化头（shortlist） | `:598` | 默认由 profile 解析（`include/ninfer/types.h:251-260`） |
| `--no-lm-head-draft` | 回到全词表头 | `:678` | 同上；此二者与 `--spec` 的关系在 `src/product/speculative_options.h:66-69` |

## 5. 输出形态 / 思考 / 采样 / 停止（17 条）

| 旗标 | 语义 | parser | 默认 / 规则 |
|---|---|---|---|
| `--raw-output` | 原样输出 | `:600` | 默认 false（`apps/cli/options.h:149`） |
| `--print-token-ids` | 打印 token id | `:602` | 默认 false（`apps/cli/options.h:150`） |
| `--no-thinking` | 关思考 | `:604` | 默认开（`apps/cli/options.h:158`） |
| `--thinking-budget N` | **模型自产**思考 token 上限 | `:606` | 默认 nullopt（`apps/cli/options.h:159`）；与 `--no-thinking` 互斥（`:994-996`） |
| `--reasoning-effort low\|medium\|xhigh` | 推理档 | `:608` | 名字表 `:114-119`；与 `--no-thinking` 互斥（`:991-993`） |
| `--vision` | 打开图像/视频输入并装载固定的 Vision 显存分配 | `:610` | 默认 false（`apps/cli/options.h:43`）；**与 `--spec dflash` 互斥**（`:988-990`）；用法 `:283` |
| `--temperature F` | 采样温度 | `:699` | 域 `[0,2]`（`:700`）；省略时由模型给（`apps/cli/options.h:165-166`） |
| `--top-p F` | 核采样 | `:701` | 域 `[0,1]` |
| `--top-k N` | top-k | `:703` | 域 `[0,20]`，允许 0（`:704-706`） |
| `--min-p F` | min-p | `:707` | 域 `[0,1]` |
| `--presence-penalty F` | 存在惩罚 | `:709` | 域 `[-2,2]`（`:711`） |
| `--frequency-penalty F` | 频率惩罚 | `:712` | 域 `[-2,2]` |
| `--seed N` | 随机种子 | `:715` | 省略时由模型给 |
| `--stop-token-id N` | 停止 token，**可重复** | `:684` | 允许 0；超 token 域拒绝（`:686-688`） |
| `--stop <text>` | 停止串，绑 **Content** 通道 | `:690` | 空串拒绝（`:692-694`） |
| `--reasoning-stop <text>` | 停止串，绑 **Reasoning** 通道 | `:690`（同一分支） | 通道选择在 `:697` |
| `--greedy` | 贪心（等于把 temperature 置 0） | `:726` | 副作用在 `:997` |

## 6. 自省 / 仪器（3 条）

| 旗标 | 语义 | parser | 默认 / 规则 |
|---|---|---|---|
| `--ft-stats on\|off` | FreeToken 逐层注意力能量观测（每 decode 轮一次 device→host 拷贝） | `:672` | 默认关（`apps/cli/options.h:147`）；parse 期把它**写进** `NINFER_FT_STATS`（`:966-969`，实现 `:136-142`）；**flag 胜过 env**；消费者是服务端特性（用法 `:289-293`） |
| `--no-cuda-graph` | 关 CUDA graph | `:680` | 默认开（`apps/cli/options.h:44`）；行尺度**首次**校准要配它（用法 `:242`） |
| `--graph-capture-ceiling N` | 图捕获层数上限 | `:670` | 默认 **16**（`apps/cli/options.h:139`） |

## 7. 冷池 / 卸载（7 条）

| 旗标 | 语义 | parser | 默认 / 规则 |
|---|---|---|---|
| `--cold-policy none\|off\|window\|host\|disk\|host-then-disk\|host+disk` | 冷池策略 | `:612` | 默认 `None`（`apps/cli/options.h:111`）；`host+disk` 是等价拼写（`:621`）；给策略时只在未显式给 keep-tokens 时把 `cold_keep_tokens` 设 128（`:618-620`） |
| `--cold-keep-tokens N` | 冷池保留 token 数 | `:628` | 默认 **128**（`apps/cli/options.h:112`） |
| `--cold-host-bytes N[g\|m\|k]` | 钉住的 host 冷层预算 | `:631` | 默认 **7 GiB**（`apps/cli/options.h:116`，注释 `:114-115` 给出「4 GiB 会让 1M 带为空」的算例）；后缀解析器 `:40-64`（带溢出检查） |
| `--max-cold-pages N` | 冷池显式页上限；**0 = 由策略推** | `:633` | 默认 **0**（`apps/cli/options.h:121`）；允许 0（`parse_u32(..., true)`） |
| `--kv-unload-watermark-pages N` | 自由 text-KV 页降到该水位时，Engine **主动卸载**它判定不可加载的块 | `:636` | 默认 `kUnloadWatermarkDerive`（`apps/cli/options.h:128`）；**0 = 关**；env 同名，`CLI > env > default`，环境侧解析在 `:747-758`；不可解析值**具名拒绝** |
| `--cold-disk-path DIR` | 冷层落盘目录 | `:650` | 默认空（`apps/cli/options.h:131`） |
| `--cold-disk-bytes N` | 落盘预算 | `:652` | 默认 **32 GiB**（`apps/cli/options.h:130`）；**0 拒绝**（`:654-656`） |

## 8. 权重卸载 W13（4 条）

| 旗标 | 语义 | parser | 默认 / 规则 |
|---|---|---|---|
| `--weight-host-bytes N` | 权重卸载到 host 的字节数 | `:657` | 默认 **0**（`apps/cli/options.h:134`） |
| `--weight-device-arena-bytes N` | 设备侧权重 arena 大小 | `:659` | 默认 0（`apps/cli/options.h:135`） |
| `--weight-prefetch-layers N` | 预取层数；**<2 拒绝** | `:661` | 默认 **2**（`apps/cli/options.h:136`）；拒绝语点名理由（`:663-667`） |
| `--weight-span-floor-bytes N` | 权重跨度下限字节 | `:668` | 默认 0（`apps/cli/options.h:137`） |

## 9. 上下文追加（1 条）

| 旗标 | 语义 | parser | 默认 / 规则 |
|---|---|---|---|
| `--append-context-text <text>` | 运行中把一段已知文本用 artifact 自己的 tokenizer（**裸编、无 chat 模板、无隐式特殊 token**）编码后追加并预填充 | `:717` | 默认空 = 腿关闭（`apps/cli/options.h:156`）；**是输入不是输出**：不进生成 ids、不耗 `--max-new`（用法 `:300-306`）；空值拒绝（`:719-723`） |

## 10. 恢复出来的两个额外旗标（上一轮清单漏了）

上一轮的穷尽清单（记录目录 `dl/docinv/FEATURES.md`，不在本仓库内）的旗标表里**没有这两个**，但它们确实在 parser 与用法文本里：

| 旗标 | 语义 | parser | 默认 | 用法文本 |
|---|---|---|---|---|
| `--max-new N` | 生成 token 上限 | `:362` | 128（`apps/cli/options.h:26`） | `:149` |
| `--vision` | 图像/视频输入总开关 | `:610` | false（`apps/cli/options.h:43`） | `:266`、`:283` |

## 11. 名字存在但不是旗标（照名字写就会错）

| 名字 | 真实身份 | 出处 |
|---|---|---|
| `--some-flag` | `apps/cli/options.cpp:326` 的**注释**里的反例，讲的是「`argv[1]` 以 `--` 开头就不当模型路径」 | `:325-330`；全树只此一处 |
| `--kv-auto-relayout` | **serve 独有**。CLI 里只在用法文本被提到一次并明说它的消费者在服务端 | CLI：`apps/cli/options.cpp:292` 是**唯一**命中（`apps/cli/options.h` **0** 命中）；真正解析在 `src/serve/serve_options.cpp:535`，用法 `:121,177` |
| `--kv-tail-tokens` | **悬空的名字**：CLI 里 0 命中，env `NINFER_KV_TAIL_TOKENS` 0 命中，且没有东西可供它定尺寸 | 见 [`unfinished.md`](unfinished.md) 第 ?1 条 |

## 12. 互斥与拒绝规则（parse 期集中检查）

parser 尾部是一段集中的一致性检查（`:839-1012`），全是**具名拒绝**而不是「接受后忽略」：

- 位预算家族两两互斥：`:847-852`（`--kv-bit-budget` vs `--kv-bits` 家族）、
  `:859-863`（vs `--kv-layer-storage`）、`:872-879`（`--kv-dtype` vs 位上限，注释给出修复前的实测：
  同一命令不同 dtype 却产出同一 payload 278.00 MiB）、`:880-884`（K/V 位宽 vs 逐层表）、
  `:914-919`（`--kv-bits` vs `--kv-k-bits/--kv-v-bits`）。
- 依附关系：`:885-891`（`--kv-bits-mode` 需要 K/V 请求）、`:892-900`（滑杆需要上限）、
  `:903-913`（罚分表需要滑杆）、`:927-935`（codec 偏好需要 `--kv-bits`）。
- 树与链：`:831-837`（`--draft-tree` 需要 `--spec mtp` 且无 `--draft-tokens`）。
- 输入恰好一个：`:971-978`（`--prompt` 与 `--messages` 恰一，两个无模型入口除外）。
- 对齐：`:980-982`（`--prefill-chunk` 必须是 128 的倍数）。
- 组合禁止：`:988-990`（`--spec dflash` 不能配 `--vision`）、`:991-996`（`--no-thinking` 不能配
  `--reasoning-effort` / `--thinking-budget`）、`:1005-1012`（`--recalibrate` 不能配
  `--kv-row-scale` 的非 auto 态）。
- 投机域校验委托给 `src/product/speculative_options.h:63-115`（调用点 `:987`）。

## 13. 一处源码注释与代码不一致（照注释读会得出错的结论）

`apps/cli/options.cpp:770-772` 的注释说「`parse_u32()` 在 `apps/cli/options.cpp:237` 被**不带**
`allow_zero` 调用，所以 `--draft-tokens 0` 被拒、自适应入口不可达」。实测**该调用现在传了
`allow_zero=true`**，在 `:576`（`parse_u32(value(arg), "draft-tokens", /*allow_zero=*/true)`），
且 case 分支在 `:567`。⇒ 这段注释里的行号与结论都已过期（它的 (c) 条已经落地）。同一段注释
`:761-762` 自述是「TEMPORARY adaptive-MTP escape hatch ... DELETE THIS BLOCK when the real fix lands」，
所以它属于[部分实现](unfinished.md)一册的第 B9 条。
