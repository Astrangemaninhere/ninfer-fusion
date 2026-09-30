<!-- 2026-09-30: this file is the CURRENT ENTRY POINT *and* a real readme again. Earlier today it
     was cut down to a 42-line pointer and the owner called that out ("你这个readme丢了不少东西啊");
     the operational half (artifact table, quick start, docker, limits, license) and the upstream
     performance tables are RESTORED below, each labelled with where it came from. Nothing was
     deleted: the full inherited upstream README also lives at docs/upstream-NInfer-README.md, the
     Chinese feature reference at README_FEATURES.md, and the English one at README.en.md. -->

# NInfer Fusion — 当前状态入口 / Current state entry point

**本文件 = 仓库首页。** 三份文档各司其职：

| 文档 | 内容 | 语言 |
|---|---|---|
| **README.md**（本文件） | 首页：是什么 / 模型与产物 / 快速开始 / 特性 / 块级 KV 现状 / 限制 / 许可 | 中文为主 |
| [README_FEATURES.md](README_FEATURES.md) | 功能总览的完整版（9 节：模型族、KV 格式、加速手段、块级 KV、按构造精确的代价、已知问题、旋钮、边界） | 中文 |
| [README.en.md](README.en.md) | the same pack written for an English reader | English |
| [docs/upstream-NInfer-README.md](docs/upstream-NInfer-README.md) | 继承自上游 `Neroued/ninfer` 的原始 README（**逐字保留**，已标注过时） | English |
| [docs/features/withdrawn-numbers-2026-09-30.md](docs/features/withdrawn-numbers-2026-09-30.md) | 2026-09-30 撤下的不实数字存档（**不是删除**：原文可读） | 中文 |

**证据规则（本树）**：只有本机本卡实测过的数字才写进结论；上游的实测数字照抄，但标明**非本树实测**。

---

# NInfer 新增功能总览（README_FEATURES）

本文件是 `README.md`（上游那份）与 `README.en.md`（本 fork 那份）的**功能面**配套入口：
把这一轮落进工作树的新增功能按族列清，**每条给出可核对的出处**。

细节分册在 [`docs/features/`](docs/features/README.md)；**未完成 / 部分实现 / 按名找不到**
的逐条清单在 [`docs/features/unfinished.md`](docs/features/unfinished.md)。

**本版的形态**：写**机制与状态**，不写数字。凡需要一个性能/内存读数的地方，写"**本机
实测；配对 sha 与完整 argv 记在 `dl/<line>/`**"，把数字本身留在证据路径里。唯一保留的
数字是**按构造精确**的字节对价（代码里被 `static_assert` 钉死，改一个 bit 就编译不过）
——见第 6 节。

**被撤下来的数字去哪了**：2026-09-30 从本文件与 `README.en.md` 撤走的每一行、每个数字，
都**逐字**保存在[撤档](docs/features/withdrawn-numbers-2026-09-30.md)里，逐条写明
**撤因**（四类）与**原本的证据在哪**（或"证据未找到"）。撤的是"读数"，不是数据。

**基准与出处读法**：行号一律指**本文件写入当天的这棵工作树**，实测
`git rev-parse HEAD = de80bd1f56b789ad5415c187721815ad1f6a963a`。

**行号会漂**，必须重测：上一版把各目标族写在 `src/CMakeLists.txt:604-613`，今天是
`:703-712`；上一版写 `apps/cli/options.cpp` 共 1016 行，今天是 **1551** 行。纪律见
[verification.md](docs/features/verification.md) 第 6.3 节。本节也不再写死"工作区有多少
条改动"这类计数：那是一个**会动的**基数，写下来即过期（撤下的两个读数与今天的实测都在
撤档 §2.1）。

---

## 1. 这是什么

NInfer 是一个从零写的 C++/CUDA 推理引擎，面向**显式注册**的 Qwen 系列 checkpoint，跑在
单张 NVIDIA GeForce RTX 5090 上（`README.md:5-8`）。运行时是刻意特化的：一张 GPU、一个
常驻模型、启动期定死的一到八路并发请求（`README.md:7-8`）。入口有三条：本地 CLI、
OpenAI 兼容 HTTP API、Anthropic 兼容 HTTP API（`README.md:6-7`）。

**可执行程序**由 `apps/CMakeLists.txt` 声明，共五个：`ninfer`（`:1`）、`ninfer-serve`
（`:13`）、`ninfer-perplexity`（`:24`）、`ninfer-cufree`（`:54`）、`ninfer-hostpath`
（`:118`）。其中**只有三个被安装**：规则是
`install(TARGETS ninfer ninfer-serve ninfer-perplexity`（`apps/CMakeLists.txt:75`）；
另两个**故意**不在安装规则里（`:49-53` 的自述）。

安装规则把文档组件装到 `${CMAKE_INSTALL_DOCDIR}`（`CMakeLists.txt:551`、`:600`）；
配置期还**生成**一个自描述的 `ninfer-install-manifest.txt`（生成语句
`CMakeLists.txt:557`）。⚠ 它**不是源树里的文件**，是配置期的产物，所以 `ls` 源码树
找不到它。

---

## 2. 支持哪些模型族

`src/targets/` 下的族目录，按 `src/CMakeLists.txt` 的构建归属分两档：

| 族 | 构建归属 | 出处 |
|---|---|---|
| `qwen3_6` | 编进 engine（核心族） | `:703` |
| `qwen3_6_27b` | 编进 engine | `:704` |
| `qwen3_6_35b_a3b` | 编进 engine | `:705` |
| `muse_glimmer_30b` | 编进 engine | `:706` |
| `qwen3_5_9b` | 编进 engine | `:707` |
| `qwen4_exp` | 编进 engine | `:711` |
| `gemma4_31b` | 编进 engine | `:712` |
| `qwen3_vision` | `EXCLUDE_FROM_ALL`，须点名才建 | `:762` |
| `qwen3_8_flash_next` | `EXCLUDE_FROM_ALL`，须点名才建 | `:763` |
| `spark_x2_5_4b` | **有条件**加入（在条件分支里） | `:730`、`:739` |

上表的"出处"一列省略 `src/CMakeLists.txt` 前缀。

- registry 认识的 package 在 `src/targets/registry.h:5-9`，逐条是 `qwen3_6_27b`、
  `qwen3_6_35b_a3b`、`muse_glimmer_30b`、`qwen3_5_9b`、`spark_x2_5_4b`；别名在 `:20-23`
  与 `:30`，可活跃目标的变体在 `:197`。
  ⚠ 上一版这里写"**四个**"，实测是**五个**（多了 `spark_x2_5_4b`）⇒ 已按实测改正。
- ⚠ `qwen4_exp` 是**骨架**：`src/targets/qwen4_exp/CMakeLists.txt:1-4` 自述
  「identity-only registration. No sources yet」，只登记了 `impl/ple_ngram.cpp`
  一个源（`:14-15`）。
- ⚠ `qwen3_8_flash_next` **未注册进 engine**：`registry.h` 不 include 它。
- 各族与转换工具的**文件数 / 行数**这一版**不写**：它们是单一时刻的树快照，且没有点名
  证据路径（撤因见撤档 §2.2、§2.3）。要复现就用
  [verification.md](docs/features/verification.md) 第 4 节的命令。

---

## 3. 支持哪些 KV 格式

两条**不同**的词汇表，别混：

**① CLI 的全局档位拼写**（`parse_kv_cache`，`apps/cli/options.cpp:186`）：
`bf16`、`int8`、`fp8`、`nvfp4`、`iso4e`（`iso3` 是废弃别名）、`rk4v4`（`e8` 是废弃别名）。
用法表列在 `apps/cli/options.cpp:252`。

**② 分层词汇表的枚举**（`src/kvcfg/kv_formats.h:27`）：
`Auto`、`Bf16`、`Fp16`、`Int8`、`Int4`、`Iso4`、`Iso4e`、`Rk4v4`、`Rk3v4`、`Rk2v4`；
配 `Nvfp4Mode{Fusion,Pure}`（`:68`）与 `KvTierFormats`（`:70`）。语法
`hot=bf16,cold=iso4e`（`:5`；`tail=` 引擎按名拒绝：无尾层），入口是 `--kv-tier-formats`
（`apps/cli/options.cpp:803`），解析器 `src/kvcfg/kv_formats.h:329`。

用法文本还自述一条限制：**冷档的槽位编码由层的 dtype 推导，只有 `int8` 可达**
（`apps/cli/options.cpp:355-358`）。

**分层存储的 SPEC 语法**（`--kv-layer-storage`，case 在 `apps/cli/options.cpp:765`）：
`all:<type>` / `<type>`（`all:` 的简写）/ `A:<type>` / `A-B:<type>`；`<type>` 是
`bf16, int8, fp8, nvfp4, iso4e, rk4v4, rk3v4, rk2v4`（`src/product/kv_options.h:9-22`）。
**没被写到的槽位保持 BF16 = 继承全局 `--kv-dtype`**（`:22-24`）；"这个槽写过没有"由掩码
携带（`:26-30`）。

- `int8` 与 `fp8` 是**两个不同的 codec**，而且两个解析器对 `fp8` 的答案**故意不同名**：
  CLI 的 `--kv-dtype fp8` 给 `Fp8E4M3Row256`（`apps/cli/options.cpp:189`），
  `parse_kv_storage("fp8")` 给 `Fp8Group16`（`src/product/kv_options.h:49`）。
  `apps/cli/options.cpp:193-196` 自述这两个表**故意**不同名，不是抄写。
- ⚠ `rk3v4` / `rk2v4` **可选但不可跑**：词汇表、档梯成本与平面几何都在，但**这棵树里
  没有任何 decode 或 append 内核读 3-bit / 2-bit 的 K 码板**，所以解析成它们的计划会在
  求解器处**按名被拒**，而不是被拿去用 4-bit 读法误读
  （`src/product/kv_options.h:52-58`）。

---

## 4. 哪些加速手段

每条给出入口与出处；读数按纪律不给，需要时沿 `dl/<line>/` 读。

- **MTP / DFlash / DFlash2 / DSpark 投机解码** ——
  `--spec auto|none|off|mtp|dflash|dflash2|dspark`（case `apps/cli/options.cpp:879`）。
  名字表 `src/product/speculative_options.h:11`；默认 `Auto` 在 `apps/cli/options.h:51`。
- **自适应草稿宽度**（生存/代价准则逐轮选档）—— `--spec mtp --draft-tokens 0`
  （case `:881`）。机制 `src/targets/qwen3_6/impl/runtime/mtp_window_cut.h`，
  其 env 名表在 `:15,71,95,146,267`。
- **MTP 树验证**（L 条 rank-path × d 步）—— `--draft-tree L,d`（case `:892`）。
  结构字段 `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h:99,100`
  （自述 `:95-98`、`:131-154`）。
- **DFlash2 树/束走查** —— 由 `--spec dflash2` 路径进入，无独立旗标。
  `include/ninfer/ops/dflash2_ddtree.h`、`dflash2_ddtree_beam.h`、`dflash2_tree_walk.h`。
- **CUDA graph**（可关、可限层数）—— `--no-cuda-graph`（`:1014`）与
  `--graph-capture-ceiling N`（`:1002`，用法 `:411`）。默认开
  `apps/cli/options.h:53`；层数上限默认 **16** 在 `:177`。
- **FreeToken 带宽治理的预填充单元** —— `--prefill-chunk-mode dynamic|manual`
  （case `:545`）。机制 `src/runtime/engine/bandwidth_governor.h`；env
  `NINFER_FT_BW_GOV` 在 `:250`。
- **KV 位预算求解**（逐层档位表）—— `--kv-bit-budget` / `--kv-bits` /
  `--kv-k-bits`+`--kv-v-bits`（case `:558,629,638,647`）。求解器
  `src/product/kv_bit_budget.h`、`src/product/kv_kv_bits.h`。
- **KV 温冷分层与冷池落盘** —— `--cold-policy`、`--cold-keep-tokens`、
  `--max-cold-pages`、`--cold-host-bytes`、`--cold-disk-*`（case `:928`、`:944` 等；
  用法 `:399-408`）。host 层 `src/targets/qwen3_6/impl/runtime/cold_host_tier.h`。
- **KV 页池预分配** —— env `NINFER_KV_PAGING_PREALLOC`；读取点
  `src/targets/qwen3_6/impl/runtime/layouts_impl.h:2589`。
- **权重卸载（W13）** —— `--weight-host-bytes` 等四个
  （case `:988,990,992,1000`）；`src/product/weight_offload_budget.h`。
- **行尺度校准闭环**（KV 行尺度表落盘/复用）—— `--kv-row-scale auto|off|FILE`、
  `--recalibrate`（case `:839,848`）；闭环 `src/product/kv_rowscale_persist.h`。
- **YaRN 静态 rope**（把上下文域扩到 **4 倍**）—— `--yarn`（`:1016`，用法 `:377`）。
  上限改动 `src/targets/qwen3_6/impl/runtime/layouts_impl.h:1333`；默认关
  `apps/cli/options.h:176`。
- **平面丢弃**（整层不建 KV plane）—— env `NINFER_KV_DROP_LAYERS`；解析
  `src/targets/qwen3_6/impl/state/decoder_state.cpp:1281`（自述 `:190`）。被丢的层被
  具名拒绝在 `src/targets/qwen3_6/impl/runtime/layouts_impl.h:137`。
- **自动前缀共享**（issue #142）—— serve 默认，可关；
  `src/serve/serve_options.h:109`、`src/serve/openai_common.cpp:147`（`:190`）。
- **KV 自动重排**（**serve 独有**）—— `--kv-auto-relayout SECS`；解析
  `src/serve/serve_options.cpp:582`，用法 `:130`、`:214`。
- **跨 rank / 多设备** —— 无开关。**host-only，无调用路径**：
  `src/core/shard_plan.h`、`src/core/tp_transport.h`；契约
  `docs/maintainer/multi-device-and-shard-plan.md`。

上面凡写作 `:NNN` 的出处，省略的都是 `apps/cli/options.cpp`。

---

## 5. 本轮聚焦（块级 KV）与它的当前状态

块级 KV 是**逐格**的：一格 =（64 token 块 × 一个文本层），一格一次只降一档。

- **速率标尺** —— 预算按 bits/element 给：
  `NINFER_KV_BLOCK_BUDGET_RATE_X10000`。键定义
  `src/product/kv_block_budget_stage.h:233`，读取点 `:362`；绝对口径
  `..._BYTES`（`:176`）与 `..._BLOCKS`（`:185`）。
- **触发量由速率差反推** —— `steps_per_call` 每趟按 `budget_bytes` 与 `total_bytes`
  的差额算（`src/product/kv_block_descent.h:764-788`；四列推导的自述在 `:766`）。
- **carry 按页身份重锚** —— `carry_reanchor[kept=.. dropped=.. new=..]`
  （`src/product/kv_block_descent.h:719`、`:937`）。
- **水位 cap/pass 分离** —— `pass_holds`：水位之上**也执行也打印**
  （`src/targets/qwen3_6/impl/runtime/program_impl.h:13627`；
  `reason=above-watermark` 在 `:14060`）。
- **第三根轴**（平面按自己那一类定页数）—— `KVPlaneGeometry::page_group_count`
  （`src/core/paged_kv_cache.h:157`），旋钮 `NINFER_KV_AXIS3_NARROW_PAGES`
  （`src/product/kv_block_descent.h:1458`）。**唯一读者** `:1463`；两个消费者
  `src/targets/qwen3_6/impl/runtime/program_impl.h:13812` 与
  `src/targets/qwen3_6/impl/state/decoder_state.cpp:1291`。
- **K/V 成对** —— `CellPair` / `cell_pair_bytes` / `cell_pair_read_side`；
  任意对定价 `e8_kv_pair_bytes(K,V)`。出处 `src/product/kv_cell_modes.h:517`、`:527`、
  `:667`；`src/product/kv_e8_width.h:284`。
- **冷层（KVMem）** —— `--cold-policy window|host|disk|host-then-disk`
  （`host+disk` 是等价拼写）；用法 `apps/cli/options.cpp:399-401`。
  host 层 `src/targets/qwen3_6/impl/runtime/cold_host_tier.h`。

**这一节的读数一律不写数字**（撤因见撤档 §1.6、§2.5）。需要读数就沿证据路径读：

- 第三根轴：`dl/kvaxisA/EVIDENCE.txt` §4 与 §6（**配对 sha 与完整 argv 都在那一节**）；
  同批还有 `dl/kvplanar/`、`dl/kvplanar2/`。
- 速率与触发：`dl/kvrate2/EVIDENCE.txt`、`dl/kvrate2/out/RATE2.tsv`、
  `dl/cmpfire3/out/CMP3.tsv`。
- 冷层落盘：`dl/kvmpf/out/`、`dl/kvmpf2/out/MPF2.tsv`、`dl/kv1m/EVIDENCE.txt`。

---

## 6. 按构造精确的对价（唯一保留的数）

下面这些是**代码里被 `static_assert` 钉死**的字节数。它们是定义的性质，不是某次运行的
性质——改一个 bit 的定义就编译不过。

- E8 的 K 平面，W4 / W3 / W2 —— 8704 / 6656 / 4608 B
  （`src/product/kv_e8_width.h:183,185,187`）。
- E8 的 V 平面（出厂 i4 板，三档共用）—— 8704 B
  （`src/product/kv_e8_width.h:148,157`）。
- E8 的层（K+V），W4 / W3 / W2 —— 17408 / 15360 / 13312 B
  （`src/product/kv_e8_width.h:189,190,191`）。
- `e8_kv_pair_bytes(K, V)` 的四个已断言对 —— `(B4,B4)=17408`、`(B4,B3)=15360`、
  `(B2,B4)=13312`、`(B2,B2)=9216`
  （`src/product/kv_e8_width.h:319,321,333,491,494`）。
- `(rk4v4, rk4v4)` = `(B4,B4)`，出厂对 —— 17,408 B
  （`src/product/kv_cell_modes.h:577`）。
- `(rk3v4, rk4v4)` = `(B3,B4)` —— 15,360 B（`src/product/kv_cell_modes.h:580`）。
- `(e8-2bit, rk4v4)` = **`rk2v4`** = `(B2,B4)`：K 2 bit、**V 仍钉在 i4 板** ——
  **13,312 B/格**（`src/product/kv_cell_modes.h:585`）。
- `(e8-2bit, e8-2bit)` = `(B2,B2)`，对称地板 —— **9,216 B/格**
  （`src/product/kv_cell_modes.h:589`）。
- 上面两个对价之差 = 一个平面的 `B4 − B2`；有断言
  （`src/product/kv_cell_modes.h:597`）。
- KVarN 记录（payload 与记录等宽）—— **26,880 B**
  （`include/ninfer/ops/kvarn.h:36,37`）。
- E8 的位/元素（×100），W4 / W3 / W2 —— 425 / 375 / 325
  （`src/product/kv_e8_width.h:193,194,195`）。

**`13,312` 与 `9,216` 是"定价"，不是"能读"。** 读侧在同文件的 `cell_pair_read_side`
里被标成 `Reserved`（`src/product/kv_cell_modes.h:666`、`:676`），而且
`e8_kv_lattice_decode_group<3>/<2>` **没有调用者**
（`src/product/kv_storage_dtype.h:90-91`、`:116`；
`src/product/kv_block_descent.h:1425`）。配合第 3 节"`rk3v4`/`rk2v4` 可选但不可跑"
一起读。

---

## 7. 已知的、没有藏起来的问题

每条写**是什么**与**证据在哪**，不写数字。

1. **prefill 对未改动引擎偏低** —— 这是**唯一未过的验收项**；判据是"快或等、绝不更慢"，
   而偏低背后的机制**尚未识别**。证据：`dl/kvspend/`、`dl/kvaxisA/EVIDENCE.txt` §6、
   `dl/cmpfire3/blob_F1250.md`。
2. **长上下文侵蚀混档** —— 降档占比随上下文变长而下降；一条 `retired` 就能把某一臂的
   降档一次抹掉；且走道**欠报**自己的降档。证据：`dl/kvplanar/EVIDENCE.txt`、
   `dl/kvplanar2/out/PLAN_EVIDENCE.txt`、`dl/kvrate2/`、`dl/kvaxis/`、`dl/cmpfire3/`。
3. **decode 列不作验收** —— 它的实测仪器噪声比这套机制能产生的效应还大。证据：
   `dl/cmpfire3/`（F-1250），引用在 `dl/kvaxisA/EVIDENCE.txt` §6。
4. **窄档区是"预留"，不是"已用"** —— 缺的一环是读侧：按类寻址、int8 重编码驱动、
   块表 class 标签。证据：`dl/kvfill/STATE.md`、`dl/kvfill2/`、`dl/kvplanar/`；
   树内是 `E8KvPlaneReadSide::Reserved`（`src/product/kv_cell_modes.h:666-685`）。
5. **回读（refetch）从未演练** —— 盘点器自己的头文件写明：一次什么都不回读的运行
   **什么都不打印**，所以"零回读"今天还只是一句注释，不是一个读数。证据：
   `src/targets/qwen3_6/impl/runtime/cold_refetch_census.h:16-66`；留档
   `dl/kvmemrun/`、`dl/kvmemfix/`。
6. **1M 被三道环挡住** —— 原生上下文门（每族自己的 `maximum_context`，
   `src/targets/*/impl/variant.h`；`--yarn` 放 4 倍，
   `src/targets/qwen3_6/impl/runtime/layouts_impl.h:1333`）、设备缺口、冷层缺口。
   证据：`dl/kv1m/EVIDENCE.txt`、`dl/long1m/logs/`、`dl/combo1m/REPORT.md`、
   `dl/1mmtp/`、`dl/ctxsweep/`。
7. **`NINFER_KV_QUALITY_WEIGHT` 没有接到逐格走道** —— 它只在**天花板/分离求解器**上
   生效：键定义 `src/product/kv_kv_bits.h:446`、读取 `:488`；而在逐格走道的三个文件
   （`src/product/kv_descent_control.h`、`src/product/kv_block_descent.h`、
   `src/product/kv_cell_alloc_solve.h`）里**一次都不出现**。第 8 节按名写了这条。
8. **两处机制今天按构造塌回原像** —— `NINFER_KV_DESCENT_ALLOC=solve` 选无状态逐格
   求解，但**没有代价表生产者**，所以它塌回
   （`src/product/kv_cell_alloc_solve.h:1102`）；`NINFER_KV_BUDGET_RULER_F1231` 是
   **编译期 `#define`**，不是旋钮（`src/product/kv_block_budget_stage.h:240`）。
   上一版把它列进旋钮表，是本版改正的一处**误述**（撤档 §4.2）。
9. **残余平面**（`--kv-residual-layers`，case `apps/cli/options.cpp:790`、用法 `:291`）
   —— 两个前端都解析，不在默认路径上；状态按名记在
   [unfinished.md](docs/features/unfinished.md)。本版**不再复述**"`block_tables` 损坏"
   那句——它在这棵树里没有坐标，已按名撤进撤档 §4.5。

---

## 8. 新机制的旋钮表

**未设 = 原像**，这是代码的性质，不是某次运行的性质：

- 降档的 OFF 路径是 `budget_bytes <= 0`，它在任何走道进入之前就返回默认向量与原像
  计费——"OFF is byte-for-byte the pre-image"
  （`src/product/kv_block_descent.h:478-480`）。
- 第三根轴的读者对**未设、空、或任何非数字字符**都返回 `0`，而 `0` 就是原像形状
  （`src/product/kv_block_descent.h:1460-1463`）；池自己的 `page_group_count = 0`
  意思是"这个平面用池的页数"，即每个既有运行量到的那块平面
  （`src/core/paged_kv_cache.h:149-150`、`:157-158`）。

⇒ **不设旋钮地跑这棵树，跑的就是原版路径**，再加上你确实设了的开关。

- `NINFER_KV_BLOCK_BUDGET_RATE_X10000` —— **速率**预算（本趟要达成的 bits/element）。
- `NINFER_KV_BLOCK_BUDGET_BYTES` / `..._BLOCKS` —— 同一预算的绝对字节 / 计费块口径。
- `NINFER_KV_DESCENT_CHAIN` —— 档位链（`lattice` = 4/3/2-bit 阶梯）。
- `NINFER_KV_DESCENT_MAX_TIER` —— 任何格最深能到哪一档（`0` = 不许动）。
- `NINFER_KV_DESCENT_ALLOC` —— `solve` 选无状态逐格求解
  （⚠ **今天按构造塌回原像：没有代价表生产者**）。
- `NINFER_KV_DESCENT_KEEP_RECENT_PAGES` —— 年龄闸的 K（**热窗保持 int8**）。
- `NINFER_KV_AXIS3_NARROW_PAGES` —— 第三根轴：每层按窄类定尺寸的页数。
- `NINFER_KV_UNLOAD_WATERMARK_PAGES` —— 卸载的**通行**；`0` 是关，未设 = 由
  `--prefill-chunk` 推导（`apps/cli/options.cpp:455-459`）。
- `NINFER_KV_QUALITY_WEIGHT` —— **只在**天花板/分离求解器上生效的速度质量滑块，
  **没有接到逐格走道**。
- `NINFER_KV_PAGING_PREALLOC` —— 页池预分配（opt-in）。
- `NINFER_KV_DROP_LAYERS` —— 整层丢弃 KV 平面（`"0,3,7"` 或 `"2-5"`）。
- `NINFER_FT_BW_GOV` —— 预填充单元的三态开关（`0` 关、`1` 开）。

**唯一不是运行期旋钮的名字**：`NINFER_KV_BUDGET_RULER_F1231` —— 它是
`src/product/kv_block_budget_stage.h:240` 的编译期 `#define`。上一版把它列进本表，
本版按实测改正（撤档 §4.2）。

---

## 9. 本文档的边界（明说）

- 本文件**不写性能/精度/内存读数**；需要时给证据路径。唯一例外是第 6 节的按构造对价。
- 本文件**每一条保留的断言都有出处**（`路径:行号` 或 `dl/<line>/`）。凡**我没能复核**
  的，不进正文：按名留在 [unfinished.md](docs/features/unfinished.md)，或者撤进
  [撤档](docs/features/withdrawn-numbers-2026-09-30.md)。
- 本文件**不承诺**任何性能，也不把"预留"写成"已用"、不把"定价"写成"能读"、不把"可选"
  写成"可跑"。
- 行号是**读数**：它们指 `de80bd1f56b789ad5415c187721815ad1f6a963a` 这棵树，会随别人的
  编辑漂。引用前请重测，并把当天读到的文件 sha256 一起记下
  （[verification.md](docs/features/verification.md) 第 6.3 节）。


---
# 本树的机制索引与平台现状

> 这一节是**对照你的记录逐项补齐**的：此前三份 README 里 AMD/ROCm、V100 地板、Vulkan、`--capability-report`、预取 全是 0 次提及。下面的名字全部取自源码（引擎自己的 usage 文本 `apps/cli/options.cpp` 与 `src/` 里的 `NINFER_*`），不是描述性声明。

## A. 机制 -> 旋钮（按子系统）

### KV 量化与编解码
- 命令行：`--kv-auto-relayout` `--kv-bit-budget` `--kv-bits` `--kv-bits-mode` `--kv-capacity` `--kv-codec-preference` `--kv-dtype` `--kv-k-bits` `--kv-k-tier-scores` `--kv-layer-storage` `--kv-quality-weight` `--kv-residual-layers` `--kv-rotation` `--kv-row-scale` `--kv-score-table` `--kv-tier-formats` `--kv-tier-scores` `--kv-unload-watermark-pages` `--kv-v-bits` `--kv-v-codec` `--kv-v-tier-scores` `--nvfp4-mode`

### 块 KV / 逐块降档 / 速率与水位
- 命令行：`--kv-auto-relayout` `--kv-bit-budget` `--kv-bits` `--kv-bits-mode` `--kv-capacity` `--kv-codec-preference` `--kv-dtype` `--kv-k-bits` `--kv-k-tier-scores` `--kv-layer-storage` `--kv-quality-weight` `--kv-residual-layers` `--kv-rotation` `--kv-row-scale` `--kv-score-table` `--kv-tier-formats` `--kv-tier-scores` `--kv-unload-watermark-pages` `--kv-v-bits` `--kv-v-codec` `--kv-v-tier-scores` `--max-cold-pages`

### 冷层 / 卸载 / 磁盘与主机层
- 命令行：`--cold-disk-bytes` `--cold-disk-path` `--cold-host-bytes` `--cold-keep-tokens` `--cold-policy` `--max-cold-pages` `--ple-sidecar` `--recall-prefill-tokens` `--weight-device-arena-bytes` `--weight-host-bytes` `--weight-prefetch-layers` `--weight-span-floor-bytes`

### 推测解码 / 草稿
- 命令行：`--draft-tokens` `--draft-tree` `--inject-spec` `--lm-head-draft` `--no-lm-head-draft` `--no-thinking` `--spec` `--thinking-budget`

### 服务 / HTTP / 批与并发
- 命令行：`--max-concurrency` `--prefill-chunk` `--prefill-chunk-mode` `--recall-prefill-tokens`

### 多模态 / 媒体
- 命令行：`--vision`

### 构建 / 工具链 / 平台
- 命令行：`--no-cuda-graph`

### 质量与仪器
- 命令行：`--ft-stats` `--kv-k-tier-scores` `--kv-quality-weight` `--kv-score-table` `--kv-tier-scores` `--kv-v-tier-scores`

### 运行期 / 图捕获 / 调度
- 命令行：`--graph-capture-ceiling` `--no-cuda-graph`

## B. 平台与路线（今天能核实的，与还没有实测的）

- **本卡**：构建只接受 `sm_120a`（RTX 5090）；本 README 里所有本树实测都在这一张卡上。
- **AMD / ROCm**：树里有 AMD 侧的适配层与模拟器（`src/compat/`、模拟用环境变量如 `NINFER_SIM_ARCH` / `NINFER_SIM_VENDOR` / `NINFER_GFX906_COMPAT`），**但从未在真 AMD 卡上跑过**。路线是走 ROCm（不是只走 Vulkan），Vulkan 在本树没有实现。— 这一条按现状写，**不含任何性能声明**。
- **V100 / `sm_70` 地板**：`sm_70` 是设计地板（低于它的架构不排期），但本树**没有在 V100 上实测过**；DOT/内存模型的模拟覆盖不完整（模拟器停在 `s_lshl_b32`）。
- **能力自报**：引擎自带 `--capability-report`，打印它自己认为可用的能力面；此前 README 未提。
- **未实测即不写数字**：上列三平台（AMD/V100/Vulkan）目前**没有本机实测**，所以这里只有范围与目标，没有吞吐、显存或质量数字。

## C. 命令行参考（引擎自己的 usage 文本，逐字）

```text
std::string usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.ninfer> (--prompt <text>|--messages <messages.json>)\n"
           "       [--max-context N] [--kv-capacity N|auto] [--max-concurrency N] [--prefill-chunk N] [--max-new N]\n"
           "       [--prefill-chunk-mode dynamic|manual]\n"
           "           (who owns the prefill unit. dynamic (default) lets the bandwidth governor\n"
           "            install a unit inside [128, --prefill-chunk] and shrink it while decode\n"
           "            latency sits above its measured noise floor; manual pins the unit to\n"
           "            --prefill-chunk itself for the whole run. NINFER_FT_BW_GOV=0 is the\n"
           "            environment spelling of manual, =1 of dynamic; the flag wins. Either way\n"
           "            --prefill-chunk is the ceiling, and NINFER_FT_BW_TRACE=1 prints the mode\n"
           "            and the unit the engine installed.)\n"
           "       [--device N]\n"
           "       [--kv-dtype bf16|int8|fp8|nvfp4|iso4e|iso3|rk4v4|e8] [--kv-layer-storage SPEC] [--kv-bit-budget SPEC] [--spec auto|off|mtp|dflash|dflash2|dspark|none --draft-tokens N]\n           (dspark is a spelling of dflash, not a fourth backend: the DSpark drafter IS\n            the DFlash (v1) runtime, and the artifact's own weights identity\n            (weights_id=nvfp4-dspark) is what decides whether its Markov head runs --\n            dflash/markov_w1+w2 are bound only for that identity, and without them the\n            same --spec dflash drafts by plain argmax.)\n           (--kv-dtype names ONE global KV tier: bf16, int8, fp8 (row-scaled E4M3 D256),\n            nvfp4, and the pair this engine prints as iso4e-g16 / rk4v4-g64 -- iso3 and e8\n            are their deprecated aliases, accepted for one release. nvfp4 is the WEIGHT tier\n            the shipped qwen3_8_27b_nvfp4_modelopt artifact records for itself (weights_id:\n            nvfp4-modelopt), i.e. the tier this project ships. It cannot be combined\n            with --kv-bit-budget/--kv-bits, whose ceiling would replace the table it fills.\n           (--kv-bit-budget takes a ceiling per KV element, or per layer range:\n            \"0-7:8,8-15:4.5\" -- the ranges must TILE every\n            FULL-ATTENTION layer (16 here, NOT the 64 the old example assumed: this\n            variant has 48 GDN layers, and they carry no paged KV); it never exceeds the\n            declared ceilings)\n"
           "       [--stage-layers SPEC] [--stage-handoff DIR] [--stage-handoff-cut]\n"
           "           (THE PIPELINE STAGE PARTITION of the text-layer axis, and the one\n"
           "            surface that makes a pp world askable from a command line instead of\n"
           "            only from a farm probe. SPEC is lo-hi layer ranges, one per STAGE:\n"
           "            0-17,18-35 is a pp world of 2 over 36 layers, and 0-35 is the\n"
           "            IDENTITY -- one stage, i.e. axis none, byte-for-byte a run with no\n"
           "            flag at all. lo-hi is inclusive hi, the same spelling --kv-layer-storage\n"
           "            documents as 0-7:bf16. The grammar, the cover and the axis check live\n"
           "            in ONE place, core/stage_plan.h, called by this front door AND by the\n"
           "            runtime -- a second spelling of the grammar is what this project keeps\n"
           "            paying for.)\n"
           "           (REFUSED BY NAME, never accepted and ignored: refused-stage-layers for a\n"
           "            mis-shaped SPEC; refused-stage-layers-partition for a spec that is not a\n"
           "            cover of [0, layers); refused-stage-layers-axis for a spec the rank axis\n"
           "            does not derive -- plan_shards() itself decides, so a layer range that\n"
           "            is not a shard of the world core/shard_plan.h hands out is refused\n"
           "            rather than run and reported as something it is not. NOTE what is NOT\n"
           "            touched: --stage-layers makes pp REACHABLE, it does not make\n"
           "            validate_virtual_request(pp).active read anything but 0. The virtual\n"
           "            -device guard still refuses pp, with its own reason, unchanged.)\n"
           "           (A PARTIAL range is additionally refused by name where the range cannot\n"
           "            be honoured: refused-stage-layers-spec with speculation on (the drafters\n"
           "            walk the layer axis themselves -- dflash_impl.h:143/:236,\n"
           "            dflash2_impl.h:190/:248, mtp_impl.h:264 -- and this range does not bound\n"
           "            them); refused-stage-layers-w13 with the W13 weight host-offload budget\n"
           "            set (product/weight_residency.h:374-383 asserts that every pass enters\n"
           "            every offloaded layer through note_layer(), and a partial pass does not);\n"
           "            refused-stage-layers-graph with CUDA-graph capture on (the seam is a\n"
           "            HOST-side write/read, which a captured graph would replay stale -- the\n"
           "            same reason W13's H2D is prefill-only).)\n"
           "           (--stage-handoff DIR names the directory the boundary hidden state\n"
           "            crosses through: stage k writes stage_k.bin, reads stage_{k-1}.bin, as\n"
           "            raw bytes plus a header carrying a magic, the producing layer, the\n"
           "            element count and an FNV-1a, so a payload left over from another prompt\n"
           "            is DETECTABLE rather than silently consumed. --stage-handoff-cut is a\n"
           "            NEGATIVE CONTROL, not a feature: it silences the producer so the ids\n"
           "            MOVE, which is how \"the handoff is load-bearing\" is falsified on the\n"
           "            shipped binary instead of asserted.)\n"
           "       [--kv-residual-layers SPEC]\n"
           "           (the per-layer NVFP4 SECOND-STAGE RESIDUAL planes, SPEC = a bare layer\n"
           "            list over the family-wide 64 slots, \"2-5\" or \"0,3,7\". This is the\n"
           "            only flag that expresses a real plane SUBSET: an NVFP4 layer goes from\n"
           "            4 planes to 8. It is NVFP4-only -- naming any other tier is accepted\n"
           "            and inert -- and a residual-bearing layer is NOT cold-capable, so a\n"
           "            stack that names this cannot also spill to the cold pool.)\n"
           "       [--kv-bits B] [--kv-k-bits BK --kv-v-bits BV] [--kv-bits-mode joint|split|ceiling]\n"
           "       [--kv-codec-preference CODEC[,CODEC...]]\n"
           "           (the K/V bit-width entries. --kv-bits is the JOINT form: ONE overall\n"
           "            ceiling for the whole KV stack, one tier per full-attention layer.\n"
           "            --kv-k-bits/--kv-v-bits are the SPLIT form: K and V each get their own\n"
           "            ceiling and each plane's PER-LAYER layering is solved on its own, then\n"
           "            reconciled per layer. A layer whose K and V requirements meet no single\n"
           "            tier is refused BY INDEX with the missing (K,V) cell named, and the run\n"
           "            also prints the deployable plan at min(k,v) so a working spec is one\n"
           "            flag away. --kv-bits-mode split is the default; joint reads the same two\n"
           "            ceilings as one; ceiling deploys min(k,v) on purpose and reports the\n"
           "            headroom that could not be spent. Both planes always cost the SAME bits\n"
           "            per element: one DType drives both planes in this engine.)\n"
           "           (--kv-codec-preference is the OTHER axis of the same fit: which codec\n"
           "            is chosen among candidates that cost the SAME bits -- \"同 bit 宽度下\n"
           "            换一种量化\", NOT fewer bits. The value is an ORDER over the candidate\n"
           "            grammar " + product::kv_gear_candidate_list() +
           " (most-wanted first; built from the ladder rows the\n"
           "            solver actually accepts, not written out by hand here), e.g.\n"
           "            --kv-codec-preference iso4e or --kv-codec-preference iso4e,nvfp4. It\n"
           "            makes the solver try the named codecs first, so it decides only a tie\n"
           "            the fit's own columns already call equal: it cannot lower or raise a\n"
           "            penalty and cannot change the achieved bits. It needs --kv-bits. An\n"
           "            unknown name, an empty element, a repeat, a missing --kv-bits, a\n"
           "            --kv-layer-storage request, and a preference the fit could NOT honour\n"
           "            are each refused by name with the accepted list attached -- never\n"
           "            accepted and ignored. Worked example at 16 layers:\n"
           "              --kv-bits 4.5 --kv-quality-weight 0 --kv-codec-preference iso4e\n"
           "            returns 0-15:iso4e (dtype iso4e-g16), where the same command without\n"
           "            the preference returns 0-15:nvfp4 (nvfp4-g16): SAME bit count,\n"
           "            different codec. Without --kv-quality-weight the shipped ladder's iso4e\n"
           "            pin (penalty 200 against nvfp4's 30) makes the preference lose, and\n"
           "            that loss is reported as a refusal rather than silently ignored.)\n"
           "       [--kv-quality-weight W] [--kv-tier-scores FILE|INLINE]\n"
           "           (speed/quality slider for the bit-budget fit: W=0 fastest KV path,\n"
           "            W=1 most accurate. Needs a ceiling to act on (--kv-bits /\n"
           "            --kv-bit-budget / --kv-k-bits/--kv-v-bits); without one it is refused\n"
           "            instead of being silently ignored. --kv-tier-scores replaces the score\n"
           "            table; --kv-k-tier-scores / --kv-v-tier-scores give the split entry one\n"
           "            table per plane.)\n"
           "       [--capability-report]\n"
           "           (the BUILD capability surface's own entry point: like --kv-score-table it\n"
           "            runs with NO model and NO prompt. It prints the arch list this binary was\n"
           "            COMPILED for -- or the literal <unreported> plus the reason, when the build\n"
           "            did not publish one -- then the arch ladder and the per-format tensor-core\n"
           "            floors THIS build ships, each with its kernel citation, so \"what would\n"
           "            this build refuse, and why\" is answerable without an artifact. It is NOT\n"
           "            a device probe: the card in this machine is not queried and no capability\n"
           "            is claimed for it. A model path given as well continues the run afterwards.)\n"
           "       [--kv-score-table show|emit=PATH]\n"
           "           (the penalty table's OWN entry point: it runs with NO model and NO prompt.\n"
           "            'show' prints the table the planner would use plus the real provenance of\n"
           "            each column; 'emit=PATH' writes it in the grammar --kv-tier-scores reads,\n"
           "            so the table can be inspected, edited and fed straight back in. The\n"
           "            shipped quality column is a PRIOR, not a measurement, and the output says\n"
           "            so.)\n"
           "           (--spec defaults to auto; none turns speculation off)\n           (--spec mtp --draft-tokens k > 0 pins one MTP draft width; --spec mtp --draft-tokens 0,\n            or with the width left unset, or NINFER_MTP_ADAPTIVE=1, uses the ADAPTIVE ladder,\n            where the mtp_window_cut criterion picks the rung per round)\n           (--spec mtp --draft-tree L,d verifies a TREE instead of one chain: L rank-paths\n            per depth, d draft steps, one verify column per node plus the anchor, so the node\n            budget L*d <= 15. That budget IS the round's draft width, so --draft-tree and\n            --draft-tokens are two spellings of one number and cannot both be given.\n            --draft-tree 1,d is the degenerate chain and must reproduce --draft-tokens d\n            token-for-token, which is the tree path's instrument check.)\n"
           "       [--kv-tier-formats hot=auto|bf16|int8,tail=...,cold=...] [--nvfp4-mode fusion|pure]\n"
           "           (hot = the resident format of every full-attention layer; only bf16 and\n"
           "            int8 have a resident codec. cold is the aged-out tier format; the cold\n"
           "            slot codec is derived from the layer dtype and only int8 is reachable\n"
           "            today. tail is accepted only when it repeats hot: the engine has no\n"
           "            recent-window tier yet. --nvfp4-mode pure forbids nvfp4/iso4e/rk4v4.)\n"
           "       [--kv-rotation on|off] [--kv-row-scale auto|off|FILE] [--recalibrate]\n"
           "       [--kv-v-codec iso4e|e2m1]\n"
           "           (component switches for the NVFP4/FP8/ISO4E KV tiers.\n"
           "            --kv-rotation off takes the identity SO(4) map on BOTH the K write\n"
           "            and the Q read, so QK^T stays exact and only the quantization domain\n"
           "            changes. --kv-row-scale off takes the identity row scale in the kernel\n"
           "            (no identity file needed); auto keeps the baked table; FILE loads an\n"
           "            NINFERKVRS1 sidecar (NINFER_KV_ROWSCALE is the env equivalent).\n"
           "            auto also runs the calibration loop: a table persisted next to the\n"
           "            artifact (MODEL.kvrowscale.bin) is loaded and the capture is skipped,\n"
           "            and when there is none -- or it was baked for another model or another\n"
           "            KV configuration -- THIS run captures once and writes one. The first\n"
           "            calibration run needs --no-cuda-graph. --recalibrate ignores the\n"
           "            persisted table, captures again and overwrites it.\n"
           "            --kv-v-codec e2m1 stores NVFP4-tier V as E2M1 instead of ISO4E and is\n"
           "            refused when a V residual plane or the cold pool is active.)\n"
           "       [--yarn]\n"
           "           (static YaRN factor-4 rope: the rope domain goes to 4x the variant's native\n"
           "            context and the yarn4 attention scaling 1.1386 is folded into the sincos\n"
           "            tables (include/ninfer/ops/rope.h). It moves the rope domain the captured K\n"
           "            is built through, so it enters the row-scale fingerprint as the\n"
           "            product/kv_rowscale_persist.h rope_regime knob: mixed in as ;rope=1 when\n"
           "            not default, the tag itself staying rs1.<12 hex> (NINFERKVRS1 is the\n"
           "            sidecar magic, not the tag). A table baked by a run that did not name\n"
           "            --yarn does not validate for a run that does. Off by default.)\n"
           "       [--lm-head-draft]\n"
           "       [--no-lm-head-draft]\n"
           "           (the drafter's proposal head, stated explicitly: --lm-head-draft selects the\n"
           "            optimized head, --no-lm-head-draft pins the full-vocabulary head\n"
           "            (ProposalHead::Full). Full is also what the profile-resolved Auto leaves in\n"
           "            place for an artifact that carries no text/draft_head, so this flag exists to\n"
           "            make a run's head independent of what the artifact resolves\n"
           "            (include/ninfer/types.h, ProposalHead).)\n"
           "       [--temperature F] [--top-p F] [--top-k N] [--min-p F]\n"
           "       [--presence-penalty F] [--frequency-penalty F] [--seed N] [--greedy]\n"
           "       [--stop-token-id N]... [--stop <text>]... [--reasoning-stop <text>]...\n"
           "       [--print-prompt-ids] [--print-token-ids] [--no-thinking] [--thinking-budget N]\n"
           "       [--reasoning-effort low|medium|xhigh] [--vision]\n"
           "       [--cold-policy none|off|window|host|disk|host-then-disk|host+disk] "
           "(host+disk is an accepted equivalent spelling of host-then-disk; "
           "docs/cli.md)\n"
           "[--cold-keep-tokens N]\n"
           "       [--max-cold-pages N] [--kv-unload-watermark-pages N]\n"
           "       [--recall-prefill-tokens N]\n"
           "       [--append-context-text <text>]\n"
           "       [--cold-host-bytes N[g|m|k]]\n"
           "       [--cold-disk-path DIR] [--cold-disk-bytes N]\n"
           "       [--ple-sidecar DIR]\n"
           "       [--weight-host-bytes N] [--weight-device-arena-bytes N]\n"
           "       [--weight-prefetch-layers N] [--weight-span-floor-bytes N]\n"
           "       [--no-cuda-graph] [--graph-capture-ceiling N]\n"
           "       [--ft-stats on|off] [--inject-spec PATH]\n"
           "\n"
           "Streams answer content to stdout and reasoning plus diagnostics to stderr.\n"
           "Structured message content accepts text, image/image_url, and video/video_url parts;\n"
           "media sources may be local paths, HTTP(S) URLs, or base64 data URIs.\n"
           "--vision enables image/video input and loads the fixed Vision GPU allocations.\n"
           "--thinking-budget caps model-origin thinking tokens; inserted control tokens count "
           "toward --max-new.\n"
           "--kv-capacity auto leaves " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB of sizing headroom.\n"
           "--inject-spec PATH declares a CHOSEN TENSOR and the position range it will "
           "occupy (src/spec/inject_channel.h): direction=ingest|egress, dtype=bf16|f16|f32, "
           "layout=token-major, rows, cols, position0, scale, path, digest. The tensor's "
           "alphabet is the input-embedding space, so it enters the model where a gathered "
           "token's embedding enters, and an ingress of the engine's own bytes is a no-op "
           "to the model (the etiquette requirement of src/spec/sum_dir.h:98, made into a "
           "measurement). The declaration is validated here, with the header's own parser, "
           "so a wrong field or a wrong dtype refuses by name before the artifact is loaded; "
           "the checks that need the model (rows, position range) are made at bind time. "
           "Every admitted ingest is reported on stderr with its shape, dtype, position "
           "range and ingested digest, and every mismatch with its refusal name. "
           "NINFER_INJECT_SPEC is the env spelling and the flag beats it.\n"
           "--print-prompt-ids prints, on stderr, the ids the PROMPT was tokenized to, in order. "
           "It is the input side of --print-token-ids, and it exists because a `sum_dir` row's "
           "identity is a digest over its block's token ids (src/spec/sum_dir.h:210-224), so a row "
           "cannot be NAMED -- and therefore cannot be bound to the inject channel "
           "(src/spec/sum_dir_inject.h) -- without them. Read-only: the engine already holds the "
           "sequence (include/ninfer/engine.h:30) and this flag is the surface it never had. "
           "--ft-stats on enables the FreeToken per-layer attention-energy observation "
           "(NINFER_FT_STATS=1 is the env spelling); it is off by default and the flag "
           "beats the env. Its consumer -- the periodic KV relayout -- is a serve-side "
           "feature (--kv-auto-relayout there), because this front end has no decision "
           "cycle.\n"
           "--recall-prefill-tokens N bounds the RE-PREFILL a recall round may do, in TOKENS "
           "(NINFER_RECALL_PREFILL_TOKENS is the env spelling and the flag beats it; no default, "
           "0 = off). It is a different dimension from the 256 MiB byte budget, which the tree "
           "itself calls a MEMORY-SAFETY limit rather than a speed limit. At the edge the round is "
           "REFUSED, by name: stderr carries `refused-prefill-budget` with the tokens wanted and "
           "the budget, at plan time and before a single token is re-prefilled. It is never "
           "truncated to fit -- a truncated run is a prefix of the answer's context, i.e. a "
           "partial or a confidently wrong answer. `--recall-prefill-tokens 0` is refused by "
           "name: 0 is the default, so it would be accepted and read by nothing.\n"
           "--kv-unload-watermark-pages N is the free text-KV pool pages at or below which "
           "the Engine proactively unloads the blocks its semantic directory judges "
           "unloadable, instead of waiting for the pool to overflow. 0 = off (the "
           "pre-watermark behaviour); the default derives the reserve from --prefill-chunk. "
           "NINFER_KV_UNLOAD_WATERMARK_PAGES is the env spelling and the flag beats it; an "
           "unparseable value from either is refused by name, never ignored.\n"
           "--append-context-text <text> encodes <text> with the artifact's own tokenizer (raw: "
           "no chat template, no implicit special token) and, mid-run, appends that run of tokens "
           "to the running request and prefills it, so the model can attend to it from the next "
           "round. It is input, not output: the appended tokens are never reported in the "
           "generated ids and never consume --max-new. It is armed before the request's first "
           "decode round, and [context-append] on stderr reports what was serviced. Off when the "
           "flag is absent.\n"
           "Sampling defaults come from the loaded model and thinking mode; flags override "
           "individual fields.\n";
```

---

# 功能与机制（按被反复提出的频次排序）

这一部分按**本项目记录里被反复提出的次数**降序排列：越靠前，是这轮工作里被追问得越多的机制。
例外只有一处：被当天两次点名的「子代理机制」放在首位。每节只写三件事：

- **是什么** —— 机制在引擎里干什么；
- **怎么用** —— 对应的旗标与环境旋钮，名字全部取自源码；
- **现状** —— 四档之一：**已落地可跑** / **在测** / **定价存在但读侧未接**（读侧标 `Reserved`）/ **未验证**。
  "预留"不写成"可用"，"定价"不写成"能读"，"能表达"不写成"能跑"。

**这一部分不给读数**：凡需要一个数的地方，本文件不写，读数留在各自的证据路径里（边界见
「本文档的边界（明说）」一节）。两条不在这里重复：**AMD / ROCm 与多平台**已经写在上面
「平台与路线（今天能核实的，与还没有实测的）」一节，这里只作交叉引用；**构建与流程**不是引擎机制。

---

## 子代理机制（索引与并发，主对话接力）

**是什么**：这是记录里点名的一项**设计**，并且要先与另一种模式分清楚：把引擎当编码代理后端、主代理靠
"读报告"交接的那条路，在记录里被明确记为**协作效率远低于**下面这个设计。设计本身是：

- **KV 里放纯粹的索引**，**每一条索引对应一条独立且界限分明的并发**；
- **主对话与并发接力混合**：主 KV **只记载每一条并发所承担的部分**；
- 并发的工作**直接注入主过程**（不是把它当报告再读回来），并且**指令与回复不经过文字**；
- 目的是把**并发带来的 decode 效率**做上去，而不是把并行摊成若干条分支。

**怎么用**：

- **索引作为草稿来源**：`NINFER_INDEX_DRAFTS` 是进程级开关（读一次，运行中不可改）；**未设 = 关，而
  "关"就是今天的引擎**——那次咨询根本不会被调用，每个草稿槽保持引擎自己会写的值。打开后是**每条车道、
  每轮解码一次咨询**，发生在 host 上、在该轮的入口循环里、在该轮发射之前，并且**在任何图捕获体之外**；
  它返回的是索引**从头填上的草稿槽数**，返回 `0` 就意味着这一轮的草稿逐字节仍是引擎自己的。这条链上的
  名字是 `index_query_term()`（查询构造）、`index_draft_fill()`（填槽）、`index_draft_census()`（账）、
  `index_drafts_armed()`（是否武装）；代码在 `src/targets/qwen3_6/impl/runtime/program_impl.h` 的
  `[IDXWIRE]` 段。它与链式草稿的关系是硬的：**索引只能提供一条链**——树回合里它会被按名记成 `refused_tree`，
  因为把索引 token 当树列去验证，会报出一棵从未建过的树。
- **非文字通道**：`src/spec/inject_channel.h` 是**声明面**（host-only、可用普通 `g++` 单测：每个字段、
  每个拒绝名、摘要与 dtype 算术都在那里）；`src/targets/qwen3_6/impl/runtime/inject_ingress.h` 是**引擎侧**，
  也就是"**一份声明在那里变成一次写进前向的写**"：每个预填充分块做一次连续的拷贝，写进 input-embedding
  矩阵的声明列（ingest）或从其中拷出（egress），并在消费任何东西之前**按名拒绝**与声明列冲突的分块，
  并在结尾报告声明范围是否真的被覆盖。命令行面是 `--inject-spec`，环境拼写 `NINFER_INJECT_SPEC`；**未设时
  它第一行就返回**，不写任何字节——所以这条通道对既有路径的代价是一个分支。
- **并发宽度**：`--max-concurrency`（启动期给定）；车道之间共享的资源与排队见「服务 / HTTP / API / 批与并发」一节。

**现状**：**部分接线已落地**，**模式未成**。已在树里的是"索引可以当这一轮的草稿来源"这一层（选路与它的账），
以及注入通道的声明面与引擎侧入口面；**"索引 = 一条独立并发 + 与主对话接力混合 + 并发结果直接注入主过程"
作为一个可跑的模式，今天并不存在**。另有一条按记录的实测：把索引草稿与其他全开项一起打开时，引擎在短提示上
**按名失败**（`phase=decode reason: index query start`），关掉它则正常。⇒ 读法是：接线在、模式未成、
全开时会红；这是设计中的机制，**不是加速手段**。

## KV 量化格式族（e8 / nvfp4 / int8 / rk 家族）

**是什么**：KV 在这棵树里不是一个全局档，而是几层词汇表叠起来的一族 codec：CLI 的全局档位、
逐层存储表、以及热 / tail / 冷各档的格式表。带常驻编解码器的档位是 `bf16`、`fp8`（按行给尺度）、
`int8`（按组给尺度）、`nvfp4`（K 用 E2M1 码字、V 用 ISO 码字），以及引擎打印为 `iso4e` / `rk4v4`
的那一对；`iso3` 与 `e8` 是它们的废弃别名，只保留一个发布周期。`rk3v4` / `rk2v4` 是 E8 格点形式的窄档。
引擎里 K 与 V 两个平面**按元素计位是一起付账的**（一个 DType 同时驱动两个平面）。另有几条独立的
成分开关：旋转、行尺度、以及整层平面丢弃。

**怎么用**：

- 全局档与逐层表：`--kv-dtype`、`--kv-layer-storage`、`--kv-tier-formats`、`--nvfp4-mode`、`--kv-v-codec`。
- 位预算与求解：`--kv-bit-budget`、`--kv-bits`、`--kv-k-bits` / `--kv-v-bits`、`--kv-bits-mode`、
  `--kv-codec-preference`、`--kv-quality-weight`、`--kv-tier-scores` / `--kv-k-tier-scores` / `--kv-v-tier-scores`、
  `--kv-score-table`。
- 成分开关与校准：`--kv-rotation`、`--kv-row-scale auto|off|FILE`、`--recalibrate`、`NINFER_KV_ROWSCALE`、
  `NINFER_KV_ROWSCALE_IDENTITY`、`NINFER_KV_CALIB_DIR`、`NINFER_KV_CALIB_MAX_TOKENS`。
- 逐层丢弃与页池：`NINFER_KV_DROP_LAYERS`、`NINFER_KV_PAGING_PREALLOC`。
- 残余平面：`--kv-residual-layers`（只对 `nvfp4` 有作用）。

**现状**：带常驻 codec 的那几个档位**已落地可跑**（写入与解码内核都在）；`rk3v4` / `rk2v4` 的
词汇表、档梯成本与平面几何都在，但**没有读窄码板的解码与追加内核**，所以解析成它们的计划会在
求解器处按名拒绝，而不是被拿去用宽档读法误读（与「支持哪些 KV 格式」一节合读）。**行尺度校准闭环
已落地**：`auto` 是唯一跑闭环的态，表存在就加载并跳过采集，为别的模型或别的 KV 配置烤的表不被采用；
首次校准要求关掉图捕获。**残余平面是部分实现**：非默认路径，CLI 与 serve 都解析它，`nvfp4` 之外按名
惰性，且带残余平面的层不能再进冷池。**`tail` 档没有尾层**：`--kv-tier-formats` 的 `tail=` 只在重复
`hot` 时被接受。还有一条读法上的坑：`--kv-dtype fp8` 与 `--kv-layer-storage fp8` 落到**不同名**的
codec 上，这是源码自述的故意行为，不是抄写错误。求解器自带的**质量列是 PRIOR，不是测量**——
`--kv-score-table show` 会在输出里自己说明，见「质量 / 困惑度 / 评测」一节。

## KVMem / 冷层 / 卸载 / 水位 / 预取

**是什么**：把 KV 从设备窗口分层放到 host 与磁盘的一整套机制，分三层。**决策层**：谁该冷——按观测窗口
的 EWMA 与连续稳定性判断 hot/warm/cold 的驻留，**未观测的层永不被降级**，置信不足的窗口也不能把冷层
提出池。**介质层**：冷层放哪——设备窗口、钉住的 host 冷层、或磁盘；host 冷层是逐页一个钉住的 extent，
取回免费判定、可取判定、准入判定与"窗口关不上缺口"的具名拒绝都在一处。**衔接层**：策略到分页介质怎么
接——这条映射的每一态都有自己的臂，另有分页 KV 池一族、host extent store 与逻辑 KV 视图。配套的还有
几件：**卸载水位**（自由页降到水位时主动卸载语义目录判定不可加载的块，而不是等池溢出）、**权重卸载 W13**
（把权重放到 host、按层预取回来）、以及**召回/复填**（召回回合允许重新预填充，但预填充量有上限，
越界在计划期就按名拒绝）。

**怎么用**：

- 冷池策略与介质：`--cold-policy none|off|window|host|disk|host-then-disk`（`host+disk` 是等价拼写）、
  `--cold-host-bytes`、`--cold-disk-path`、`--cold-disk-bytes`、`--cold-keep-tokens`、`--max-cold-pages`。
- 水位与预取：`--kv-unload-watermark-pages`（`0` 是关，未设时由预填充单元推导；环境拼写
  `NINFER_KV_UNLOAD_WATERMARK_PAGES`）、`--weight-host-bytes`、`--weight-device-arena-bytes`、
  `--weight-prefetch-layers`、`--weight-span-floor-bytes`、`NINFER_W13`、`NINFER_W13_STATS`。
- 召回与外接：`--recall-prefill-tokens`（`NINFER_RECALL_PREFILL_TOKENS`；越界时 stderr 报
  `refused-prefill-budget`，在计划期就把想要的量与预算一起打出来）、`NINFER_RECALL_TEXT`、`NINFER_TURN_RECALL`、
  `--ple-sidecar`、`NINFER_PLE_STATS`。
- 仪器：`NINFER_COLD_HOST_REFETCH_CENSUS`、`NINFER_COLD_FALLBACK_CENSUS_*`、`NINFER_KV_PAGING_PREALLOC`。

**现状**：介质层与水位**已落地可跑**（准入/取回/窗口带宽与具名拒绝都在；水位按 `CLI > env > 默认`
三档优先序，超域按名拒绝，不做静默忽略）。**决策层只在 serve 侧可达**：它由 `--kv-auto-relayout`
的周期驱动，CLI 前端没有决策周期，所以那条闭环在 CLI 上走不到。`NINFER_FT_COLD_MODE` 的 live 档在树里
自述为**为另一条池线保留**（预留）。落盘路径存在，但**回读（refetch）侧到今天还只是一句注释**——
状态与坐标见「已知的、没有藏起来的问题」。权重卸载的预取层数有一个**下限拒绝**（低于下限会让正在算的
层自己的槽被自己的预取覆盖），这条已经落地。

## 导入 / 转换 / 模型适配

**是什么**：把一个外部 checkpoint 变成可跑的 `.ninfer` 制品的整条链路。前门是一支脚本
（`tools/convert/import_model.py`），"本地模型 → `.ninfer`"收在这一处：**注册的转换器是封闭、字节钉死的
契约**，各转换器接受什么由前门统一收口。导入之后由编排脚本全自动跑转换；验收闸是一个**逐对象比较**
两个 `.ninfer` 的工具（按退出码判"这个 build 是否与参考一致"）；另有一支工具测制品的**无损**压缩余量。
GGUF 族的能力被拆成几件：抽取、K-quant 块格式的尺寸与反量化、旋转契约、"谁施加这个旋转"的唯一决策点，
以及张量名到本树名字的映射。逐族各有一套转换器与自查脚本。模型适配还有一条工具链侧的路：从 spec 生成
目标骨架 → 自动适配管线 → **会红的门**（几何覆盖、参数完备是门，不是提示）。KV 侧另有一组**离线镜像**：
位预算求解、档位×层的标定流水线、覆盖率扫描、以及由 `ft_stats` 观测生成逐层方案（它的消费者是 serve 侧的
周期重排）。制品还带**自己的身份**：草稿头跑不跑 Markov 头由制品的权重身份决定，而不是由旗标决定。

**怎么用**：

- 转换链路（都在 `tools/convert/`）：`import_model.py`、`convert_runner.py`、`artifact_diff.py`、
  `compress_probe.py`、`gguf_extract.py` / `gguf_kquant.py` / `gguf_hadamard.py` / `gguf_fold_back.py` /
  `gguf_fold_route.py` / `gguf_names.py`、逐族目录、`archkit/`（含 `adapt.py`、`adapt_all.py`、`check_geometry.py`、
  `check_params.py`、`kv_budget_mirror.py`、`kv_tier_matrix.py`、`kv_auto_allocate.py`）、`dequant/`、`lora/`、
  `ple_sidecar_build.py`、`kv_iso_ref.py`。
- 引擎侧与之对接的接口：`NINFER_EXPORT_HEAD_DIR`（导出头目录）、`NINFER_KV_CALIB_DIR`、`--ple-sidecar`。
- 契约文档：`docs/maintainer/modelopt-nvfp4-import.md`、`docs/maintainer/artifact-container.md`、
  `docs/maintainer/tensor-formats.md`、`docs/features/importers.md`。

**现状**：**已落地可跑**（前门、编排、逐对象验收闸与逐族转换器都在，且各带单测）。边界要写清：注册的
目标族是**封闭集合**——没有运行期模型发现，也没有未注册 checkpoint 的兜底；`qwen4_exp` 在这棵树里是
**骨架**（只有 identity 注册）；`qwen3_8_flash_next` 有完整实现但**未注册进 engine**，构建上须点名才建
（见「支持哪些模型族」一节）：⇒ 视觉族与 flash-next 族在本树的引擎路径上属于**未验证**一档，"有实现"
不等于"跑过"。

## 速度 / 吞吐 / 延迟（只讲机制，不给读数）

**是什么**：影响单位时间产出的一组彼此独立的机制。

- **预填充单元的归属**：一个带宽治理器在给定上界内自己装一个预填充单元，并在解码延迟高于它自己测得的
  噪声底时**收缩**它；`manual` 则把单元钉死在上界。
- **FreeToken 带宽治理**：一组 EMA / 信用 / 份额下限 / 容差 / 窗口旋钮，另有追踪开关把治理器模式与实际
  装的单元打出来。
- **图捕获**：解码默认走 CUDA 图，可整体关掉，也可给捕获的层数上限。
- **投机解码**：MTP / DFlash / DFlash2 / DSpark（见「MTP / 投机解码」一节）。
- **权重预取**：W13 按层把 host 上的权重取回（见「KVMem / 冷层 / 卸载 / 水位 / 预取」一节）。
- **流水线分段**：`--stage-layers` 把文本层轴切成阶段，`--stage-handoff` 给边界隐状态的交接目录。

**怎么用**：`--prefill-chunk`、`--prefill-chunk-mode dynamic|manual`、`NINFER_FT_BW_GOV`、`NINFER_FT_BW_TRACE`、
`NINFER_FT_BW_EMA_ALPHA`、`NINFER_FT_BW_BASE_ALPHA`、`NINFER_FT_BW_MAX_CREDIT`、`NINFER_FT_BW_MIN_SHARE`、
`NINFER_FT_BW_STREAK`、`NINFER_FT_BW_TOL_HI` / `NINFER_FT_BW_TOL_LO`、`NINFER_FT_BW_WINDOW_MS`、
`--no-cuda-graph`、`--graph-capture-ceiling`、`--stage-layers SPEC`、`--stage-handoff DIR`、
`--stage-handoff-cut`、`--ft-stats`、`NINFER_FT_STATS`、`NINFER_FT_PERIOD`。

**现状**：治理器与图捕获**已落地可跑**（`dynamic` 默认让治理器自己装单元，`manual` 钉死上界；
`NINFER_FT_BW_GOV=0/1` 是同一件事的环境拼写，旗标胜过环境）。**分段是部分可达**：`--stage-layers`
让流水线并行从命令行**问得到**，但虚拟设备那道守卫仍然拒绝它，所以"能表达"不等于"能跑"；
`--stage-handoff-cut` 是**负对照**，不是功能——它故意静音生产者，用 id 是否移动来证伪"交接是承重的"。

## 显存 / 内存 / 带宽（只讲机制，不给读数）

**是什么**：把有限内存分到"权重 / KV / 状态 / 媒体"几处，以及把压力从设备挪走的手段。**KV 容量**由
`--kv-capacity` 定尺寸：`auto` 在启动期按权重之后剩下的内存解出合法容量并留一份定尺寸余量，显式容量在
进程生命周期内固定；**页池是共享的**（活跃请求与留存前缀共用一个池）。**页池预分配**是一道 opt-in 闸，
做容量可行性检查。**状态槽**在 serve 侧给定（设备槽与 host 槽各一组），host KV 另有一份钉住的预算。
**权重驻留**（W13）把权重放到 host，并在设备侧留一个 arena 与一个跨度下限。**媒体**有自己的一份缓存与
实时预算。**观测面**：工作集转储、headroom 百分比、arena 追踪，以及"每类平面按自己的页数定尺寸"的第三根轴
（见「块 KV / 逐块降档 / 速率预算 / 第三轴」一节）。

**怎么用**：`--kv-capacity N|auto`、`--max-context`、`NINFER_KV_PAGING_PREALLOC`、`--device-state-slots`、
`--host-state-slots`、`--host-kv-mib`、`--weight-host-bytes`、`--weight-device-arena-bytes`、
`--weight-span-floor-bytes`、`--cold-host-bytes`、`--cold-disk-bytes`、`--max-cold-pages`、
`--kv-unload-watermark-pages`、`--media-cache-mib`、`--media-live-mib`、`--max-request-mib`、
`NINFER_WS_DUMP`、`NINFER_WS_HEADROOM_PCT`、`NINFER_ARENA_TRACE`、`NINFER_KV_WINDOW_TOKENS`。

**现状**：定尺寸、页池、状态槽与权重 arena **已落地可跑**；水位的主动卸载也已落地（见「KVMem / 冷层 /
卸载 / 水位 / 预取」一节）。**超长上下文还不是一个内存问题**：设备侧缺口与冷层缺口是并列的，
状态与坐标见「已知的、没有藏起来的问题」与「长上下文 / 超长上下文的目标」两节；本文件不给任何内存读数。

## 质量 / 困惑度 / 评测（仪器怎么跑，不报数）

**是什么**：一组与引擎**同一份 Text 模型、同一套 KV 档位**的离线仪器与观测面。

- **困惑度工具**：同一 artifact、可切 KV 存储、跑完写一份完整的 JSON 记录。
- **分数表入口**：`--kv-score-table show|emit=PATH` **不需要模型与提示词**就能跑；`show` 打印规划器会用的表，
  并打印每一列的**真实出处**。
- **FreeToken 逐层注意力能量观测**：`--ft-stats`，它的消费者是 serve 侧的周期重排。
- **接受率仪器**：每个验证轮打一个明细块（不设不打）。
- **输入侧 id 打印**：提示词被切成什么 id、生成了什么 id。
- **能力自报**：`--capability-report`，**不需要 artifact** 就能回答"这个构建会拒绝什么、为什么"。
- **取回核对与基准**：长上下文取回的工具与 oracle、TTFT 基准夹具、逐档位的标定脚本。

**怎么用**：`ninfer-perplexity`（`--corpus`、`--quick`、`--kv-dtype`）、`tools/perplexity/prepare_corpus.py`、
`--kv-score-table show|emit=PATH`、`--kv-tier-scores` / `--kv-k-tier-scores` / `--kv-v-tier-scores`、
`--ft-stats` / `NINFER_FT_STATS` / `NINFER_FT_PERIOD`、`NINFER_ACCEPTLOG` / `NINFER_ACCEPTLOG_STREAM_SYNC`、
`NINFER_SVIP_THRESHOLD`、`--print-prompt-ids`、`--print-token-ids`、`--capability-report`、
`tools/bench/`、`tools/archkit/longtest_57k.py`、`tools/test_kv/`、`docs/features/instrumentation.md`。

**现状**：仪器**已落地可跑**，但边界必须一起写：其一，**分数表里的质量列是 PRIOR，不是测量**——
工具自己会在输出里说明这一点；其二，**解码侧那一列的仪器噪声比这套机制能产生的效应还大**，所以它不作验收
（见「已知的、没有藏起来的问题」）。质量结论的读数不在本文件里：需要读数就沿证据路径读，本节的职责是讲清
"怎么跑、读数从哪来、哪一列不能拿来当验收"。

## MTP / 投机解码

**是什么**：草稿—验证式解码。后端是一个枚举：`none` / `off`、`mtp`、`dflash`、`dflash2`、`dspark`、`auto`
（默认 `auto`，所以必须有一个"关回去"的拼写）。**`dspark` 不是第四个后端**：它就是 DFlash 的运行时，
跑不跑 Markov 头由**制品自己的权重身份**决定（只有那个身份的制品才绑 Markov 权重，没有就按普通 argmax 起草）。
草稿宽度可以按几种形态给：**钉住一个宽度**、**自适应阶梯**（由生存/代价准则按轮选档）、以及**树**（若干
rank-path × 若干步，节点预算**就是**这一轮的草稿宽度，所以"树"与"草稿宽度"是同一个量的两种拼写，不能同时给）。
草稿头可以是完整词表头或优化头；`auto` 的解析在注册表里只有一处，因为它同时喂规划器、装载计划与运行期。
验证侧有逐轮明细块作为接受率仪器。

**怎么用**：`--spec`、`--draft-tokens`、`--draft-tree L,d`、`NINFER_MTP_ADAPTIVE`、
`NINFER_MTP_WINDOW_CUT` / `_RATIO` / `_UNOBSERVED` / `_DENOM` / `_TRACE` / `_SHRINK` / `_MINREACH`、
`--lm-head-draft` / `--no-lm-head-draft`、`NINFER_ACCEPTLOG`、`NINFER_ACCEPTLOG_STREAM_SYNC`、
`NINFER_SVIP_THRESHOLD`、`NINFER_DFLASH_SVIP_THRESHOLD`、`NINFER_ADAPTIVE_WINDOW`；
契约文档 `docs/maintainer/mtp-draft-tree.md`、`docs/features/mtp-and-speculation.md`。

**现状**：后端族与自适应阶梯**已落地可跑**；树验证也落地，并且它自带一致性检查——**退化成链的树必须逐 token
复现同样的链式草稿**，这就是树路径自己的仪器检查。边界：把自适应再按环境变量强行打开的那条路，引擎自己按
**临时逃生舱**记着（见 `docs/features/unfinished.md`）；每族自己的草稿宽度上限定义在族里，CLI 侧今天仍按公共
上界硬编码，所以族上限在 CLI 上到不了。`dspark` 这个拼写被接受**并不承诺**制品带 Markov 头——要在没有该头的
制品上拒绝它，需要制品自身的信息，纯枚举翻译做不到这一点。

## 块 KV / 逐块降档 / 速率预算 / 第三轴

**是什么**：把 KV 的档位下沉到**格子**这一级的机制。一格 = 一个 token 块 × 一个文本层，走道每趟对每格
**只降一档**。预算有两种给法：**速率标尺**（这一趟要达到的每元素位数）与**绝对口径**（字节 / 计费块）。
**触发量由速率差反推**：每趟的步数按预算与实际总量之差算。**carry 按页身份重锚**：跨趟保留的东西用页身份
重新锚定，而不是靠位置。**水位 cap/pass 分离**：水位之上也执行也打印（`reason=above-watermark` 会写出来）。
**第三根轴**：一个平面可以按**它自己那一类**的页数定尺寸，于是窄类平面不再按宽类的页数付账。
**K/V 成对**：任意对定价与读侧标签是同一张表的两个面。

**怎么用**：`NINFER_KV_BLOCK_BUDGET_RATE_X10000`、`NINFER_KV_BLOCK_BUDGET_BYTES`、`NINFER_KV_BLOCK_BUDGET_BLOCKS`、
`NINFER_KV_DESCENT_CHAIN`、`NINFER_KV_DESCENT_MAX_TIER`、`NINFER_KV_DESCENT_ALLOC`、
`NINFER_KV_DESCENT_KEEP_RECENT_PAGES`、`NINFER_KV_AXIS3_NARROW_PAGES`、`NINFER_KV_QUALITY_WEIGHT`、
`--kv-quality-weight`、`--kv-unload-watermark-pages`，serve 侧另有 `--kv-auto-relayout SECS`（周期性地从
观测量重新推导逐层表）。未设旋钮就是原像，这是代码的性质（见「新机制的旋钮表」一节）。

**现状**：走道本体**已落地可跑**（执行、打印、越界按名拒绝都在）。**定价存在但读侧未接**：任意对定价已经
按构造钉住，但读侧在 `cell_pair_read_side` 里被标成 `Reserved`，宽档的格点解码器**没有调用者**，所以"定价"
不等于"能读"（与「支持哪些 KV 格式」一节里 `rk3v4` / `rk2v4` 的可选但不可跑一起读）。还有几处**今天按构造
塌回原像**：无状态逐格求解缺一个代价表生产者；`NINFER_KV_QUALITY_WEIGHT` **没有接到逐格走道**，它只在
天花板/分离求解器上生效。第三根轴的旋钮有唯一读者，消费者在运行期（程序侧与解码状态侧）。

## 新后端 GDN / 线性注意力（以及 FlashNext 的运行期）

**是什么**：除去 softmax 注意力，这棵树里还有**第二条序列混合器**：门控 delta 网络（GDN）式的线性注意力。
它以**层**为单位与全注意力交替——一个族里哪些层是 GDN、哪些是全注意力，由族自己的配置函数决定；并且
**GDN 层不携带分页 KV**，这正是"位预算的区间必须盖住每一层全注意力层"这条约束的来源。实现分两条路：
**递推**与**分块**；状态另有一套（线性注意力状态），并带**重放记录**，卷积状态有快照与记录两种形态。
投影侧有独立的输入投影与门控投影算族。另有一族**完全独立的运行期**（FlashNext）：它有自己的注意力、MoE
路由与超连接内核，以及一大族运行期开关。

**怎么用**：GDN 侧**没有专门的命令行旗标**——它由装载的族决定；可用的旋钮是
`NINFER_GATED_DELTA_NET_PROPAGATE`、`NINFER_GDN_GATE_FUSED`、`NINFER_GDNDUMP_DIR` / `_LAYERS` / `_MAX_CALLS` /
`_MAX_ELEMS`。FlashNext 侧是一族 `NINFER_FLASH_NEXT_*`（MoE 分阶与共享 MMA、路由与超连接的 legacy 分支、
QSA 调度与前填充、阶段台账与追踪、草稿头行数）加 `NINFER_RANKING_PATH`。契约文档：
`docs/maintainer/replayssm-gdn.md`、`docs/maintainer/qwen3.8-flash-next-model.md`、
`docs/maintainer/qwen3.8-flash-next-artifact.md`、`docs/flash-next-mtp-speculation.md`。

**现状**：GDN 是**已落地可跑**的族实现（内核、状态、重放与单测都在）。它的 KV 含义是硬的：位预算的区间
**必须盖住每一层全注意力层**，盖不住就**按索引具名拒绝**并同时打印可部署的方案。FlashNext 是**未验证**的一档：
运行期与内核在树里存在，但这一族**未注册进 engine**，构建上须点名才建 ⇒ "实现存在"不等于"以引擎路径跑过"。

## 多模态（图 / 视频）

**是什么**：结构化消息内容除了文本，还可以带 image / image_url 与 video / video_url 片段；媒体来源可以是
本地路径、HTTP(S) URL 或 base64 data URI。`--vision` 打开媒体输入**并把固定的 Vision GPU 分配装上**——
也就是说视觉相关的驻留是**启动期定死**的，后续请求不能临时开启启动时省掉的能力。serve 侧的媒体取回与前处理
有自己的缓存、实时预算与前处理线程数，取回路径另有一个**编译期**开关。

**怎么用**：`--vision`；serve 侧 `--media-cache-mib`、`--media-live-mib`、`--media-preprocess-threads`；
编译期 `NINFER_BUILD_MEDIA_ACQUIRE`（构建开关，不是运行期旋钮）；契约 `docs/sm120a-quantized-kv-vision.md`。

**现状**：媒体获取、前处理与消息解析**已落地可跑**。边界写清：`--vision` **与 DFlash 后端没有被共同验证**，
CLI 与 serve 都按名拒绝这个组合（`not co-validated` 那句在两个前端里各有一处）；树里另有一个独立的视觉目标族，
它在本树是**骨架**（须点名才建），所以"有视觉族"不等于"这一族跑过"。

## 服务 / HTTP / API / 批与并发

> 并发与批量与这一节是同一批旋钮，因此合在这里写。

**是什么**：`ninfer-serve` 把引擎包成一个常驻服务。路由族里有 **OpenAI Chat Completions**、
**OpenAI Responses Core**（含 input_tokens 与 compact）、**Anthropic Messages**（含 count_tokens）与模型清单；
它们都支持流式、工具调用回传、本地响应状态与用量记账。**批**的性质是启动期定死的：车道数与排队上限在启动期
给出，运行期**不做请求抢占、不做优先级/QoS、不做活动请求换出**；一批解码走精确批次的图捕获，并配一条
无 CUDA 图的回退。鉴权、对外模型名、CORS、请求日志落 JSONL、响应存储上限都是启动期参数；**上下文代价预设**
把代价模型的选择也放到启动期。引擎把解析出的工具调用**交回客户端**，自己**不执行**工具。

**怎么用**：

- 监听与身份：`--host`、`--port`、`--api-key`、`--model-id`、`--cors`。
- 车道与排队：`--max-concurrency`、`--max-pending-requests`、`--pending-timeout-ms`。
- 批与预填充：`--prefill-chunk`、`--prefill-chunk-mode`、`--max-context`、`--kv-capacity`。
- 驻留与状态：`--device-state-slots`、`--host-state-slots`、`--host-kv-mib`、`--context-cost-presets`、`--max-request-mib`。
- 日志与存储：`--log-stats-interval-ms`、`--request-log-jsonl`、`--response-store-max-records`、`--response-store-max-mib`。
- 请求未给时的默认：`--default-max-tokens`、`--default-thinking-budget`。
- **serve 独有**：`--kv-auto-relayout SECS`、`--ft-vram-axis on|off`。

**现状**：**已落地可跑**。两处**要按现状写的边**：其一，两个前端的旗标集**不相等**——`--kv-auto-relayout`
只在 serve 侧（CLI 前端的用法文本会明说它的消费者是 serve 侧特性）；其二，serve 的 `--help` 用法行
**漏列了自己 parser 里注册的一批旗标**（冷窗族与权重卸载族），照 `--help` 写文档会以为它们不可达，实测两族
都能过 parser 并进入装载——这是**用法文本不完整**，不是旗标不存在（见 `docs/features/serve-flags.md`）。

## 前缀复用

**是什么**：一个可复用的前缀检查点 = **KV 加上该提示词前沿的完整续写状态**，所以可复用的单位不是几段文字，
而是一次检查点。开启后，引擎会在**领先的系统提示**处发布一个共享前缀候选，并在后续请求之间复用相容前缀；
命中的量与**走了哪条复用路径**都被记账（路径是一个枚举：根、私有端点、私有回合闭合、私有响应重放、
私有长锚点、共享稳定前缀）。续写与长锚点各有数量上限。压力之下，规划器按**立即恢复的工作量**与**后续复用
代价**权衡设备驻留、钉住的 host 状态与换出，活动请求保留自己的完成预留。

**怎么用**：serve 侧 `--no-prefix-reuse`（关掉相容前缀缓存，**默认开**）、`--no-auto-system-shared-prefix`
（关掉自动共享候选）、`--max-private-continuations`、`--max-shared-prefixes`、`--max-long-anchors-per-continuation`；
观测面是请求结果里的命中量与复用路径，以及 `--request-log-jsonl`。算法见
`docs/maintainer/resource-scheduling-and-context-cache.md`。

**现状**：**已落地可跑，但只在 serve 侧**——CLI 前端没有这条路径。互斥是按名写的：关掉前缀复用**不能**与
上下文缓存的容量选项同时给（`--no-prefix-reuse cannot be combined with context-cache capacity options`）。

## 长上下文 / 超长上下文的目标

**是什么**：把上下文推到一族的原生窗口之外，要同时解开几条**并列的约束**：(a) **原生上下文门**——每个族
自己在 variant 里声明窗口；(b) **设备缺口**——权重之后剩下的设备内存装不下目标长度的 KV；(c) **冷层缺口**——
把 KV 分层放出去之后，冷层能装下的页数仍然不够。手段里有一条 **`--yarn`**：静态 YaRN 把 rope 域扩到族的原生
窗口之外，并把 YaRN 的注意力缩放**折进 sincos 表**；它改的是"被采集的 K 经过的 rope 域"，所以它**进入行尺度
指纹**——没点 `--yarn` 烤的表，不能被点了 `--yarn` 的运行采用。尺寸侧的手段是 `--max-context` 与
`--kv-capacity`，复填侧是 `--recall-prefill-tokens`。

**怎么用**：`--yarn`、`--max-context`、`--kv-capacity N|auto`、`--recall-prefill-tokens N`、
`--cold-*` 与 `--weight-*` 两族（见「KVMem / 冷层 / 卸载 / 水位 / 预取」与「显存 / 内存 / 带宽」两节）、
`--kv-row-scale auto|off|FILE` 与 `--recalibrate`（与 `--yarn` 的指纹耦合）。

**现状**：`--yarn` **已落地可跑**（rope 域与 sincos 表的折入、以及指纹耦合都在）。**超长上下文本身尚未验证**：
原生门、设备缺口与冷层缺口是并列的几道环，读数与坐标见「已知的、没有藏起来的问题」一节——本文件不给读数。
另有一处**悬空的文档引用**：树里指向 TP2/YaRN 长上下文的契约文档**不存在**，要写那条接口只能读相关的几个头
（见 `docs/features/unfinished.md`）。

## 思考 / 推理

**是什么**：模型有思考与非思考两种提示模式。非思考用 `--no-thinking` 关掉思考段；思考模式下
`--thinking-budget` 给**模型自身产生的思考 token** 设上限，而**插入的控制 token 也算进 `--max-new`**；
`--reasoning-effort` 在低/中/高之间调推理强度；`--reasoning-stop` 给停止串。采样默认值来自**装载的模型与
思考模式**，旗标只覆盖被点名的那几个字段。serve 侧另有"保留思考段"与"请求未给时的默认思考预算"，
并有一条流式约束：推理通道与内容通道**各自守自己的前缀**（Anthropic 风格事件流在两条通道上都会校验已流出的
前缀与交出的结果一致）。

**怎么用**：`--no-thinking`、`--thinking-budget N`、`--reasoning-effort low|medium|xhigh`、
`--reasoning-stop <text>`、`--preserve-thinking`、`--default-thinking-budget`；
`docs/features/cli-flags.md` 与 `docs/features/serve-flags.md` 两册按前端分别列了这组旗标。

**现状**：**已落地可跑**。互斥按名写：`--reasoning-effort` 与 `--thinking-budget` 都**不能**与 `--no-thinking`
同时给（CLI 里按名拒绝，模板侧另有一句 `reasoning effort cannot be combined with disabled thinking`）；
助手预填充也**不能**与开启的思考同时给（serve 的翻译层按名拒绝）。

## DFlash / DFlash2

**是什么**：两条草稿路径。**DFlash**（含 `dspark` 拼写）是 v1 路径：它跑不跑 Markov 头由制品身份决定
（见「MTP / 投机解码」一节）。**DFlash2** 有自己的一整套运行期：树/束走查与 selector、特征链与分数表落盘、
逐轮调试打印、配对尺度旋钮；它还会让制品带上 `text/draft_head`，于是 profile 解析的 `auto` 选到优化草稿头。
转换侧另有 DFlash2 的补丁与校验脚本。

**怎么用**：`--spec dflash`、`--spec dflash2`、`--spec dspark`、`--draft-tokens`；运行期旋钮
`NINFER_DF2FEAT` / `NINFER_DF2FEAT_DIR`、`NINFER_DF2SCORES` / `NINFER_DF2SCORES_DIR`、`NINFER_DF2DBG`、
`NINFER_DF2SEL`、`NINFER_DF2_PAIR_SCALE`、`NINFER_DFLASH_SVIP_THRESHOLD`；转换侧
`tools/convert/qwen3_8_27b/patch_dflash2.py` 与 `verify_patch.py`；状态文档
`docs/maintainer/speculative-dflash2-status.md`。

**现状**：DFlash2 路径**已落地**（实现头、算子头与单测都在），但它的状态文档自己就是一份"进行中"的账，
值得照着读。按名拒绝的边：**DFlash 与 Vision 没有被共同验证**（前端按名拒绝这个组合）；
**DFlash 与 DFlash2 的模型视图互斥**（运行期按名抛出，而不是让两个后端叠着跑）。

## KV 方向的论文借鉴（指向文档，不在这里展开）

这条线不产生新旗标，它的产物是**契约文档与机制改动**：外部的注意力/压缩/检索类工作被折成本树里的几条机制——
FreeToken 风格的逐层注意力能量观测 → 周期重排闭环、E8 格点与 ISO 码字这一对 KV 编码、独立的 KVarn 注意力族、
以及热/温/冷的驻留闭环。读法的入口在仓库根的研究笔记（`RESEARCH-EXTERNAL.md`、`RESEARCH-FLASHNEXT.md`、
`RESEARCH-FREETOKEN.md`、`PORT-RK2V4E8.md`、`PREFILL-OPT.md`）与 `docs/maintainer/` 下的契约文档；
`docs/features/` 下的分册按机制把同一批坐标再列一次。这里不重复它们的读数。

---

# 附：恢复的运维与上游章节

以下是本仓库迁移前首页上原有的章节，**逐字**恢复。它们多数描述的是引擎的构建与运行方式（上游文本），
对继承自上游的本树仍然适用；数字类的表格一律标明来源。


<!-- restored verbatim from the pre-rewrite README (upstream text); see docs/upstream-NInfer-README.md -->
*构建与运行：如下为仓库原有的快速开始（上游文本，逐字）。*

## Quick start

NInfer requires 64-bit Linux, an NVIDIA GeForce RTX 5090, CUDA Toolkit 13.1 or newer, CMake 3.28 or
newer, a C++20 host compiler, Ninja, `pkg-config`, FFmpeg development libraries
(`libavformat >= 60`, `libavcodec >= 60`, `libavutil >= 58`, and `libswscale >= 7`), and
`libcurl >= 7.85`. The build rejects CUDA architectures other than `sm_120a`.

Build the product binaries:

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Tests, benchmarks, and maintainer tools are excluded from the default build. Install the product
binaries into a prefix with:

```bash
cmake --install build --prefix /path/to/prefix
```

`--prefix` is optional; without it the `CMAKE_INSTALL_PREFIX` chosen at configure time is used.
The install tree is `bin/` (the three executables) plus `share/doc/ninfer/` (this README, the
LICENSE, `CONTRIBUTING.md`, and a generated `ninfer-install-manifest.txt`). There is no packaged
binary distribution, no installer and no release archive: `cmake --install` is the deployment
path, and the binaries still run straight out of the source build tree if you prefer that.

Download the artifact used by this example with the Hugging Face CLI:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Start a long-running text/agent server with two active-request lanes and explicit Device/Host
checkpoint capacity:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

Each request has a 240,000-token logical ceiling. A shared 240,000-token Device KV pool serves
admitted requests; two requests run concurrently when their combined reservations fit. The cache
tiers provide two Device checkpoint slots, eight pinned Host State slots, and 8 GiB of pinned Host
KV beyond the two active StateImages.

Send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

Run a one-shot CLI request with a 32,768-token allocation:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Answer content is written to stdout. Loading progress, reasoning, timings, throughput, memory, and
speculative-decoding statistics are written to stderr. Use `--messages FILE` and `--vision` for
structured image/video input; see the [CLI guide](docs/cli.md) and [committed examples](examples/cli/).
<!-- restored verbatim from the pre-rewrite README (upstream text); see docs/upstream-NInfer-README.md -->
*长上下文复用（上游文本，逐字）。*

## Resource-aware long-context reuse

A reusable prefix checkpoint contains KV and the complete continuation state for its exact prompt
frontier. A Device-resident checkpoint resumes directly. Under pressure, the planner weighs Device
retention, pinned Host State/KV, and eviction by immediate restore work and later reuse cost. Active
requests retain their completion reservations.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.
<!-- restored verbatim from the pre-rewrite README (upstream text); see docs/upstream-NInfer-README.md -->
*⚠️ 以下性能表格是**上游的实测数字**，不是本树在 2026-09-30 的实测；本树自己的对比列只放本机本卡实测。*

## Performance

Published measurements use an RTX 5090. [Performance](docs/performance.md) records the exact
benchmark profiles and methodology.

### Concurrent MTP3 decode

Saturated decode used INT8 group-64 KV, CUDA Graphs, MTP3, and one 8,192-token generation per active
request. Values are aggregate committed decode throughput and MTP acceptance from complete
intervals whose actual decode batch equaled the configured concurrency.

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept | C8 / C1 |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% | 2.88× |
| Qwen3.6-27B `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% | 5.67× |
| Qwen3.6-35B-A3B `groupwise-int` | 593.0 / 67.2% | 877.7 / 68.2% | 1,166.0 / 69.8% | 1,313.8 / 67.3% | 2.22× |
| Qwen3.8-27B `nvfp4` | 143.8 / 48.9% | 267.6 / 48.1% | 461.1 / 45.8% | 766.6 / 46.0% | 5.33× |

### Single-request serving

The serial serving corpus used INT8 group-64 KV, CUDA Graphs, a 1,024-token prefill chunk, and five
fixed seeds after warm-up. The table keeps one short-prefill, one extreme-prefill, and one
structured-output MTP3 point for each published profile; the full context and scenario matrices are
in the performance document.

| Model profile | 7,680-token prefill | 260,096-token prefill | Structured MTP3 decode |
|---|---:|---:|---:|
| Qwen3.6-35B-A3B `groupwise-int` | 15,544.3 tok/s | 5,157.1 tok/s | 770.9 tok/s |
| Qwen3.6-27B `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| Qwen3.6-27B `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| Qwen3.8-27B `groupwise-int` | 3,274.7 tok/s | 1,609.7 tok/s | 224.4 tok/s |
| Qwen3.8-27B `nvfp4` | 8,340.4 tok/s | 2,203.1 tok/s | 219.8 tok/s |
<!-- restored verbatim from the pre-rewrite README (upstream text); see docs/upstream-NInfer-README.md -->
*⚠️ 同上：评测分数为上游数字。*

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens to
fit the RTX 5090 after weights. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

### Perplexity

Run the fixed four-domain quick corpus through the artifact's tokenizer and Text model:

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_nvfp4.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --quick --kv-dtype fp8
```

The evaluator reports token-weighted fixed-window causal perplexity and writes a complete JSON
record under `profiles/perplexity/`. See [Perplexity evaluation](docs/perplexity.md) for the metric,
corpus, custom-text mode, and comparison rules.
<!-- restored verbatim from the pre-rewrite README (upstream text); see docs/upstream-NInfer-README.md -->
*启动注意事项（上游文本，逐字）。*

## Startup notes

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` selects Vision residency. DFlash is available for text-only Qwen3.6-35B-A3B execution.
<!-- restored verbatim from the pre-rewrite README (upstream text); see docs/upstream-NInfer-README.md -->
*Docker 用法（上游文本，逐字）。*

## Docker

Build the runtime image on a host with the NVIDIA Container Toolkit:

```bash
docker build --tag ninfer:local .
```

Mount the downloaded model and run the same example server profile:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```
<!-- restored verbatim from the pre-rewrite README (upstream text); see docs/upstream-NInfer-README.md -->
*能力与限制（上游文本，逐字）。*

## Capabilities and limits

All registered model IDs support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16, INT8 group-64, and row-scaled FP8 E4M3 KV storage;
- offline causal-perplexity scoring with the same Text model and selectable KV storage;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

The 35B-A3B target additionally supports text-only DFlash with draft windows from one to fifteen.

The product boundary remains intentionally small:

- one RTX 5090 and one resident model per Engine;
- a startup-fixed capacity of one to eight active requests with bounded FIFO ingress;
- no request preemption, priority/QoS, active-request swapping, weight offload, multi-GPU, or
  distributed serving;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- no runtime model discovery or unregistered checkpoint fallback;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. `--kv-capacity` sizes the shared Main Text KV pool
used by active requests and retained prefixes; `auto` resolves the largest legal capacity at
startup from the memory remaining after weights while keeping 1 GiB of sizing headroom. Explicit
capacities remain fixed for the process lifetime.
<!-- restored verbatim from the pre-rewrite README (upstream text); see docs/upstream-NInfer-README.md -->
*文档索引（上游文本，逐字）。*

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.
<!-- restored verbatim from the pre-rewrite README (upstream text); see docs/upstream-NInfer-README.md -->
*许可（上游文本，逐字）。*

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.

---

## 出处与许可 / Provenance

本仓库是 `Neroued/ninfer` 的社区增强分支：完整的上游 README 逐字保留在
[docs/upstream-NInfer-README.md](docs/upstream-NInfer-README.md)，上游的模型卡与 HuggingFace
产物链接见其内。本树的改动、实测与撤稿记录在
[docs/features/withdrawn-numbers-2026-09-30.md](docs/features/withdrawn-numbers-2026-09-30.md) 与
[README_FEATURES.md](README_FEATURES.md)。

许可条款见下方恢复的 `License` 一节（与上游一致）。
