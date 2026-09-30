# NInfer 新增功能总览（README_FEATURES）

本文件是 `README.md` 的**功能面**配套入口：把这一轮落进工作树的新增功能按族列清，每条给出可核对的出处。
细节分册在 [`docs/features/`](docs/features/README.md)；**未完成 / 部分实现 / 按名找不到**的逐条清单在
[`docs/features/unfinished.md`](docs/features/unfinished.md)。

**基准**：仓库 `HEAD = 3944a53eda1aac439a566a1cf46ea741f0415fdc`（实测 `git rev-parse HEAD`），
**工作区是脏的**：`git status --porcelain` 共 **794** 条，其中已 `A` **236** 条、未跟踪 `??` **168** 条
（实测命令 `git status --porcelain | wc -l`、`grep -c '^A'`、`grep -c '^??'`）。
**2026-09-22 复核**（`accept34` 线）：**817** 条，其中 `A` **236**（未变）、`??` **179**。
同一个基数在动 ⇒ 本行给的是**读数与读数的时刻**，不是一个固定值。
⇒ 本文档描述的是**这棵树**，不是某个已发布版本；新增功能里有相当一部分只在未提交的工作区里。
`CHANGELOG.md:20` 记录的基准修订与本文件一致。

---

## 1. 这是什么

NInfer 是一个从零写的 C++/CUDA 推理引擎，面向**显式注册**的 Qwen 系列 checkpoint，跑在单张
NVIDIA GeForce RTX 5090 上（`README.md:5-8`）。运行时是刻意特化的：一张 GPU、一个常驻模型、
启动期定死的一到八路并发请求（`README.md:7-8`）。入口有三条：本地 CLI、OpenAI 兼容 HTTP API、
Anthropic 兼容 HTTP API（`README.md:6-7`）。

三个可执行程序由 `apps/CMakeLists.txt:1,13,33` 定义：`ninfer`、`ninfer-serve`、`ninfer-perplexity`。
安装规则把这三个装到 `${CMAKE_INSTALL_BINDIR}`（`apps/CMakeLists.txt:60-62`），文档组件装到
`${CMAKE_INSTALL_DOCDIR}`（`CMakeLists.txt:515-520`），配置期还**生成**一个自描述的 `ninfer-install-manifest.txt` 并装进 docdir
（生成语句 `CMakeLists.txt:525` 的 `file(WRITE ...)`，安装 `CMakeLists.txt:567-569`）。
⚠ 它**不是源树里的文件**，是配置期的产物，所以 `ls` 源码树找不到它。

---

## 2. 支持哪些模型族

`src/targets/` 下**九个**目录（实测命令：`for d in src/targets/*/; do find "$d" -type f | wc -l; done` 与 `wc -l`）：

| 族 | 文件数 | 代码行数 | 构建归属 | 出处 |
|---|---|---|---|---|
| `qwen3_6` | 81 | 50289 | 编进 engine（核心族） | `src/CMakeLists.txt:604` |
| `qwen3_6_27b` | 8 | 3035 | 编进 engine | `src/CMakeLists.txt:605` |
| `qwen3_6_35b_a3b` | 8 | 1561 | 编进 engine | `src/CMakeLists.txt:606` |
| `muse_glimmer_30b` | 9 | 2385 | 编进 engine | `src/CMakeLists.txt:607` |
| `qwen3_5_9b` | 9 | 2147 | 编进 engine | `src/CMakeLists.txt:608` |
| `qwen4_exp` | 9 | 866 | 编进 engine | `src/CMakeLists.txt:612` |
| `gemma4_31b` | 4 | 522 | 编进 engine | `src/CMakeLists.txt:613` |
| `qwen3_vision` | 3 | 501 | `EXCLUDE_FROM_ALL`，须点名才建 | `src/CMakeLists.txt:635` |
| `qwen3_8_flash_next` | 86 | 21346 | `EXCLUDE_FROM_ALL`，须点名才建 | `src/CMakeLists.txt:636` |

- registry **认识的 package 有四个**：`qwen3_6_27b`、`qwen3_6_35b_a3b`、`muse_glimmer_30b`、`qwen3_5_9b`
  （`src/targets/registry.h:5-8`，别名声明在 `:19-22`）。
- ⚠ **`gemma4_31b` 是新增的第九族**（上一轮的穷尽清单里没有它）。它有 4 个文件，
  转换侧对应 `tools/convert/gemma4_31b/`：`convert.py` 438 行、`inventory.py` 311 行、`recipe.py` 260 行
  （实测 `wc -l`），且已进 `add_subdirectory`（`src/CMakeLists.txt:613`）。
- ⚠ `qwen4_exp` 是**骨架**：`src/targets/qwen4_exp/CMakeLists.txt:1-4` 自述
  「identity-only registration. No sources yet」，只登记了 `impl/ple_ngram.cpp` 一个源（`:14-15`）。
- ⚠ `qwen3_8_flash_next` 代码量很大（86 文件 / 21346 行，`src/targets/qwen3_8_flash_next/CMakeLists.txt:2-38`
  列出 36 个源）但**未注册进 engine**：`registry.h` 不 include 它。

---

## 3. 支持哪些 KV 格式

两条独立的词汇表，别混：

**① CLI 的全局档位拼写**（`apps/cli/options.cpp:93-107`，`parse_kv_cache`）：
`bf16`、`int8`、`fp8`、`nvfp4`、`iso4e`（`iso3` 是废弃别名）、`rk4v4`（`e8` 是废弃别名）。
用法表列在 `apps/cli/options.cpp:159`（`[--kv-dtype bf16|int8|fp8|nvfp4|iso4e|iso3|rk4v4|e8]`）。

**② 分层词汇表的枚举**（`src/kvcfg/kv_formats.h:27-66`）：
`Auto`、`Bf16`、`Fp16`、`Int8`、`Int4`、`Iso4`、`Iso4e`、`Rk4v4`、`Rk3v4`、`Rk2v4`；
配 `Nvfp4Mode{Fusion,Pure}`（`src/kvcfg/kv_formats.h:68`）与 `KvTierFormats`（`:70`）。
语法 `hot=bf16,cold=iso4e`（`src/kvcfg/kv_formats.h:5`；`tail=` 引擎按名拒绝：无尾层），入口是 `--kv-tier-formats`
（`apps/cli/options.cpp:502`），解析器 `src/kvcfg/kv_formats.h:329`。

`int8` 与 `fp8` 是**两个不同的 codec**，且 CLI 的 `fp8` 指 `Fp8E4M3Row256` 而不是 `Fp8Group16`
—— `apps/cli/options.cpp:100-103` 自述这两个表**故意**不同名，不是抄写。

---

## 4. 哪些加速手段

| 手段 | 入口 | 出处 |
|---|---|---|
| MTP / DFlash / DFlash2 投机解码 | `--spec auto\|none\|off\|mtp\|dflash\|dflash2\|dspark` | `apps/cli/options.cpp:565`；名字表 `src/product/speculative_options.h:11-45`；默认 `Auto` 在 `apps/cli/options.h:42` |
| 自适应草稿宽度（生存/代价准则逐轮选档） | `--spec mtp --draft-tokens 0` | `src/targets/qwen3_6/impl/runtime/mtp_window_cut.h`（370 行）；env 名表 `:15,71,95,146,267`；调用点 `src/targets/qwen3_6/impl/runtime/program_impl.h:11073-11128` |
| MTP **树**验证（L 条 rank-path × d 步） | `--draft-tree L,d` | `apps/cli/options.cpp:578`；结构常量 `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h:17,18,22,26,30` |
| DFlash2 树/束走查 | 由 `--spec dflash2` 路径进入，无独立旗标 | `include/ninfer/ops/dflash2_ddtree.h`、`dflash2_ddtree_beam.h`、`dflash2_tree_walk.h` |
| CUDA graph（可关、可限层数） | `--no-cuda-graph` / `--graph-capture-ceiling N` | `apps/cli/options.cpp:680`、`:670`；默认开 `apps/cli/options.h:44`，层数上限默认 16 在 `:139` |
| FreeToken 带宽治理的预填充单元 | `--prefill-chunk-mode dynamic\|manual` | `apps/cli/options.cpp:371`；机制 `src/runtime/engine/bandwidth_governor.h`；env `NINFER_FT_BW_GOV` 解析在 `src/runtime/engine/engine.cpp:86` |
| KV 位预算求解（逐层档位表） | `--kv-bit-budget` / `--kv-bits` / `--kv-k-bits`+`--kv-v-bits` | `apps/cli/options.cpp:381,403,408,412`；求解器 `src/product/kv_bit_budget.h`（2490 行）、`src/product/kv_kv_bits.h`（1341 行） |
| KV 温冷分层与冷池落盘 | `--cold-policy`、`--cold-host-bytes`、`--cold-disk-*` | `apps/cli/options.cpp:612,631,650,652`；host 层 `src/targets/qwen3_6/impl/runtime/cold_host_tier.h` |
| KV 页池预分配 | env `NINFER_KV_PAGING_PREALLOC` | 读取点 `src/targets/qwen3_6/impl/runtime/layouts_impl.h:2093`；头 `src/product/kv_paging_preallocation.h`（582 行） |
| 权重卸载（W13） | `--weight-host-bytes` 等四个 | `apps/cli/options.cpp:657,659,661,668`；`src/product/weight_offload_budget.h`、`src/product/weight_residency.h` |
| 行尺度校准闭环（KV 行尺度表落盘/复用） | `--kv-row-scale auto\|off\|FILE`、`--recalibrate` | `apps/cli/options.cpp:532,537`；闭环 `src/product/kv_rowscale_persist.h`（939 行） |
| YaRN factor-4 静态 rope（把上下文域扩到 4 倍） | `--yarn` | `apps/cli/options.cpp:682`；用法文本 `apps/cli/options.cpp:246-253`；上下文上限改动 `src/targets/qwen3_6/impl/runtime/layouts_impl.h:1166` |
| 平面丢弃（整层不建 KV plane） | env `NINFER_KV_DROP_LAYERS` | 解析 `src/targets/qwen3_6/impl/state/decoder_state.cpp:221`；丢层被具名拒绝在 `src/targets/qwen3_6/impl/runtime/layouts_impl.h:137` |
| 跨 rank / 多设备（host-only，无调用路径） | 无开关 | `src/core/shard_plan.h`、`src/core/tp_transport.h`；契约 `docs/maintainer/multi-device-and-shard-plan.md` |

---

## 5. 怎么读本文档里的出处

- **`路径:行号`** 一律指**上面那个基准修订的工作树**。行号是实测的，不是从别处转录的
  （上一轮的清单是**记录目录**里的产物，不在本仓库内；它的行号在这棵树里已经漂了：它记的
  `apps/cli/options.cpp` 是 1001 行，实测 1016 行；它记 `--kv-dtype` 在 `:363`，实测在 `:378`）。
- **实测数字**都给出复现命令，见 [`docs/features/verification.md`](docs/features/verification.md)。
- 凡**我没能证明**的条目，一条都不写进功能正文，而是按名留在
  [`docs/features/unfinished.md`](docs/features/unfinished.md) 里，并写清卡在哪。

---

## 6. 本轮聚焦（2026-09-28 → 09-30）：块级 KV 与它的当前状态

本节记录**正在处理什么**，每条都给出处；**凡未实测的不写**，凡预留/不可读的**按名写出**。

### 6.1 已落地并已实测

| 项 | 是什么 | 读数 |
|---|---|---|
| **逐格降档 + 速率标尺** | 一格 =（64 token 块 × 一个文本层）；`NINFER_KV_BLOCK_BUDGET_RATE_X10000`（读取点 `src/product/kv_block_budget_stage.h:362`） | 可动 **96.94%** 的 cell；计划币 **4.372266 b/el**（少 47% 字节） |
| **触发量由速率差反推** | `steps_per_call` 每趟按 `budget_bytes` 与 `total_bytes` 的差额算（`src/product/kv_block_descent.h`） | `TOTALS-DISAGREE` 由 26/27 趟 → **0/27** |
| **carry 按页身份重锚** | `carry_reanchor[kept=.. dropped=.. new=..]` | 窗口翻动时降档不再被整份丢弃 |
| **水位 cap/pass 分离** | `src/targets/qwen3_6/impl/runtime/program_impl.h:13411`（`pass_holds`） | 水位之上**也执行也打印**（`reason=above-watermark`） |
| **第三根轴**（平面按自己那一类定页数） | `KVPlaneGeometry::page_group_count`；旋钮 `NINFER_KV_AXIS3_NARROW_PAGES` | `kv cache payload` 4.38 → **3.56 GiB**、`gpu sequence used` 5.71 → **4.88 GiB**，计划币**逐字节一致**（884,736,000 B），prefill **+0.013%**（不可分辨） |
| **K/V 成对** | `CellPair` / `cell_pair_bytes` / `cell_pair_read_side`；任意对定价 `e8_kv_pair_bytes(K,V)`（`src/product/kv_e8_width.h`） | 计费与次序**按对为真**；每平面 B4=8704 / B3=6656 / B2=4608；⚠ **读侧仍是 `Reserved`** |
| **冷层（KVMem）** | `--cold-policy window\|host\|disk\|host-then-disk` | disk 落盘实测 **1,198,443,776 B**、`write_failures=0`、`drain_ms=933`；⚠ **回读（refetch）至今零行** |

### 6.2 正在处理

1. **窄档区填充**（线 `kvfill`）：上面那根轴今天是**预留**、不是已用；缺的一环是读侧（按类寻址 + int8 重编码驱动 + 块表 class 标签）。验收用引擎自己的设备读数在压力下证明。
2. **V 承载 E8**：今天部署的窄形态把 V 钉在 128 B 的 i4 平面，所以 `e8-2bit` 的对价是 **13,312 B/格**；V 能承载 E8 之后对称对是 **9,216 B/格（−30.8%）**。几何与任意对定价**已有**，缺的是**解码侧消费者**（`e8_kv_lattice_decode_group<2>` 无调用者）。
3. **KVMem × 文本 prefill 的协作**（线 `kvmpf`）：卸载腿在 prefill 的 chunk 循环里的行为，以及它按长度付出的代价。
4. **一个"未借鉴"的对照基线**（线 `oursbase`）：8k / 64k / 128k 上的质量、速度与两个设备币，先有我们自己的数，再谈任何外来机制。
5. **按"克服之后的收益"排序的候选**（见 `docs/features/`）：运行期 KLT 基 + 反水填、给 E8 平面选一个长度约束下的熵容器、KVarN 的通道轴。**收益诚实地为零的，就写零**，不写成"难"。

### 6.3 已知的、没有藏起来的问题

- **prefill 对未改动引擎偏低 4.11% / 5.31% / 5.54%**（三次独立读数、区间不交）。判据是"快或等、绝不更慢"，所以这是**唯一未过的验收项**，而它的机制**尚未识别**。
- **长上下文侵蚀混档**：降档占比 8k **9.32%** → 64k **2.68%** → 128k **0.60%**；128k 下**单条 `retired=1024`** 把 `rk4v4 672→16` 一次抹掉；且走道**欠报**自己的降档 2.05×–2.41×。
- **decode 列不作验收**：实测仪器噪声 **30.7%**。
- **1M 被三道环挡住**：原生上下文门 **262,144**（`--yarn` 放 4×）、设备**缺 10.43 GiB**、冷层**亏 8,457 页 = 8.86 GiB**；已实测跑到的最长上下文是 **260,096 token**。

### 6.4 新机制的旋钮表

| 旋钮 | 选什么 |
|---|---|
| `NINFER_KV_BLOCK_BUDGET_RATE_X10000` | **速率**预算（本趟要达成的 bits/element） |
| `NINFER_KV_BLOCK_BUDGET_BYTES` / `_BLOCKS` | 同一预算的绝对字节 / 计费块口径 |
| `NINFER_KV_BUDGET_RULER_F1231` | 绝对预算按哪把尺读 |
| `NINFER_KV_DESCENT_CHAIN` | 档位链（`lattice` = 4/3/2-bit 阶梯） |
| `NINFER_KV_DESCENT_MAX_TIER` | 任何格最深能到哪一档（0 = 不许动） |
| `NINFER_KV_DESCENT_ALLOC` | `solve` 选无状态逐格求解（⚠ **今天按构造塌回原像：没有代价表生产者**） |
| `NINFER_KV_DESCENT_KEEP_RECENT_PAGES` | 年龄闸的 K（**热窗保持 int8**） |
| `NINFER_KV_AXIS3_NARROW_PAGES` | 第三根轴：每层按窄类定尺寸的页数 |
| `NINFER_KV_UNLOAD_WATERMARK_PAGES` | 卸载的**通行**；`0` 是关，未设=由 `--prefill-chunk` 推导 |
| `NINFER_KV_QUALITY_WEIGHT` | **只在**天花板/分离求解器上生效的速度质量滑块——**没有接到逐格走道** |
