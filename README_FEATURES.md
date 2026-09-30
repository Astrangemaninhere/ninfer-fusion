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
