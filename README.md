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
