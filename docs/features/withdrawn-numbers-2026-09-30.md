# 撤下内容的存档：两份 README 撤掉的数字（2026-09-30）

> **这是一份存档，不是当前读数。** 本文件把 2026-09-30 从 `README.en.md` 与
> `README_FEATURES.md` 撤下来的每一行、每一个数字**逐字**保留在这里，并逐条写清
> **撤因**（下面四类里的哪一类）与**原本的证据在哪**（`dl/<line>/`、配对 sha、argv，
> 或"证据未找到"）。
>
> 引用本文件里的任何数字之前，必须回到它点名的那条证据路径复核一次。本文件本身
> **不是读数、不是结论、不是性能承诺**。两份 README 现在的形态是"机制 + 状态 +
> 证据指针"，不含这些数字。

**撤因的四类**（主人给的撤销口径）：

1. **非本机本卡实测**（别的卡 / 别的模型 / 别人论文里的数）；
2. **只在单一几何、单一语料、单一次运行下成立，却被写成通用结论**（例如把某一条
   argv 下的 ppl / 吞吐写成"本引擎的性能"）；
3. **从别的线转述、未经复核**（哪怕来源是自家线，只要没点名证据路径）；
4. **没有点名证据路径**（读者无法复核）。

⇒ 这四类从 README 撤走；**"按构造精确"的数保留并给出坐标**（见 §5）。

**本次撤销的基准**：仓库 `HEAD = de80bd1f56b789ad5415c187721815ad1f6a963a`
（`git -C /home/user/ninfer-fusion rev-parse HEAD` 实测）。`file:line` 一律指这棵树。

**前像**：两份 README 的改动前副本在 `dl/readme2/pre/`（逐字节副本，sha256 记在
`dl/readme2/out/SHAS.txt`）。逐字引用以下原行；**逐字优先于行宽**，所以本节个别行
超过 100 列。

---

## 1. 从 `README.en.md` 撤走的内容（逐字）

### 1.1 出处声明（原第 8-9 行）

```
> All performance/precision figures below are measured on our RTX 5090D +
> WSL2 (CUDA 13.3, sm_120a) setup.
```

**撤因**：④（对"下面每一个数"的**整体**出处声明，没有逐条点名证据路径；读者无法
逐条复核）。**证据在哪**：本声明覆盖的每个数各自的证据见 §3 各行；声明本身没有
可复跑的 argv 或配对 sha，所以它作为一句总括性断言不能留。

### 1.2 "Measured results" 全节（原第 11-26 行）

```
## Measured results (Qwen3.8-27B NVFP4, 4096 ctx, 13.3k zh corpus)

### KV schemes — ppl + prefill (ninfer-perplexity)

| config | ppl | NLL | prefill tok/s |
|---|---|---|---|
| **10L E8 + 6L NVFP4 (default)** | **1.0202** | 0.020 | **2027.7** |
| all-E8 (16 layers E8-lattice) | 1.1120 | 0.106 | 2082.7 |
| all-NVFP4 (E2M1 K + ISO3 V) | 1.7055 | 0.534 | 1949.1 |
| all-int8 (reference) | 1.5217 | 0.420 | 2017.2 |
| old default (10L NVFP4 + 6L I8) | 1.2962 | 0.259 | — |

The default 10L E8 + 6L NVFP4 mix wins on both precision and speed: the
E8-lattice K provides the lattice gain (H64 rotation aligned to the g64 scale
domain), while the NVFP4 layers' ISO3 V fits the value distribution better
than i4.
```

**撤因**：②（一条 argv、一个语料、一次运行下的五个读数，被写成"本引擎的"通用结论：
标题是 "Measured results"，正文写 "wins on both precision and speed"）＋ ④（没有点
名证据路径）。

**证据在哪**：**证据未找到**。对整个 `dl/` 逐字搜索 `1.0202` / `1.7055` / `1.5217` /
`1.2962`，命中的全是无关目录里的巧合子串（`dl/amdcpu/`、`dl/amdfull2/`、
`dl/adoptsuite/` 等），没有任何一处是这一节的产物。归档里**确实存在**的 ppl 家族是
另一批读数（`dl/cmpfire3/out/ppl/*/ppl/report.json`、`dl/kvspend/out/ppl/P_B_kvspend/`、
`dl/kvppl/out/arms/*/ppl/report.json`），它们的量级与上表**不同**（perplexity 量级在
十位数，例如 int8 臂 `12.168...`、nv4 臂 `12.221...`、rk4 臂 `17.651...`），
⇒ 上表既不是这批、也没有自己的一批，**无法复核**。
（相关但不是出处：`dl/cmpfire3/out/ppl/`、`dl/kvspend/out/ppl/`、`dl/kvppl/out/arms/`。）

### 1.3 "Capacity" 表（原第 28-34 行）

```
### Capacity (per head/token)

| scheme | bytes | bit/element |
|---|---|---|
| E8Kv (E8 K + i4 V, g64) | ≈260 B | ≈4.06 |
| NVFP4 (E2M1 K + ISO3 V, g16) | 288 B | 4.50 |
| int8 (reference) | 512 B | 8.00 |
```

**撤因**：④（三行都没有点名证据路径；其中 E8Kv 一行自述"≈"，是近似值被当成容量
结论写出来）。**证据在哪**：同一张表（含同样的三行，含 `≈260 B` 这一行）在本仓库
`VRAM.md:28-34`；但 `VRAM.md` 自己只写了"实测方法"（`VRAM.md:3-4`：`ninfer-serve`
加载各配置、`nvidia-smi` 读取），**没有配对 sha、没有完整 argv** ⇒ 按本次口径
仍算"证据不足"，故不留在 README，只在本档按名保留。**按构造的替代**见 §5：
家族里真正按构造精确的那几个对价（17408 / 13312 / 9216 B）已带坐标保留。

### 1.4 "VRAM reference" 表（原第 36-47 行）

```
### VRAM reference (RTX 5090D 32 GB, serve, 4096 ctx)

| config | VRAM |
|---|---|
| idle | 0.05 GB |
| base (int8 kv) | 20.5 GB |
| E8-mix / all-NVFP4 kv | 20.6 GB |
| + vision | 20.9 GB |
| + MTP3 | 21.4 GB |
| DFlash2 model | 20.6 GB |

Full table: [VRAM.md](VRAM.md).
```

**撤因**：②（单张卡、单一配置集、单次加载的读数；且以 `serve` 默认路径得到，被写成
一张通用"VRAM reference"表）＋ ④（README 里没有点名证据路径，只把读者送到
`VRAM.md`，而 `VRAM.md` 不带 sha / argv）。

**证据在哪**：本仓库 `VRAM.md:8-17`（同表）与 `VRAM.md:19-26`（余量表）。
`VRAM.md:3-4` 自述方法，**无配对 sha、无完整 argv** ⇒ 在 `dl/` 里**没有**这一批的
产物目录。**证据未找到（可复核性不足）**。

### 1.5 许可节里的悬空链接与无路径的转述（原第 92-96 行）

```
Apache-2.0. Third-party contributions and attribution: see
[THIRD-PARTY.md](THIRD-PARTY.md) — E8 codec lineage (PR #35 →
ninfer-4090 → ninfer-3090), IsoQuant tables (nvfp4rtx calibration).
```

**撤因**：④（`THIRD-PARTY.md` **在这棵树里不存在**，所以那个链接指向一个读者打不
开的目标；后半句的血缘转述也没有点名路径）。**证据在哪**：
`ls /home/user/ninfer-fusion/THIRD-PARTY.md` → 不存在。血缘本身是**有**坐标的（所以
撤的是"链接 + 无路径的写法"，不是血缘事实）：`src/ops/kernel/e8_root_codec.cuh:2`
与 `:274`、`src/ops/softmax_attention/dense/causal_cache/small_t_nvfp4.cuh:2`、
`src/ops/kernel/gqa_attention_kv_quant.cuh:20`（`PR #35 lineage`）、
`src/ops/kernel/gqa_isoquant_rot.cu:12` 与 `gqa_isoquant_rot.cuh:4`（`nvfp4rtx` 离线
标定）。

### 1.6 `README.en.md` 末尾一节里的数字

下面全部出自原第 100-146 行那一节（"What this pack is working on (2026-09-30)"）。

**(a) 逐格降档的"可动比例"与"计划币"（原第 110 行）**

```
| Per-cell demotion with a **rate** ruler | `NINFER_KV_BLOCK_BUDGET_RATE_X10000` (`src/product/kv_block_budget_stage.h:362` reader) | 96.94% of cells movable; plan coin 4.372266 b/el (47% fewer bytes) |
```

**撤因**：②（一条 argv 下的单次走道读数被写成"可动 96.94%"这样的通用结论）＋ ④。
**证据在哪**：`dl/kvrate2/`（`EVIDENCE.txt`、`blob_F1248.md`、`out/RATE2.tsv`）与
`dl/cmpfire3/`（`blob_F1250.md`、`EVIDENCE.txt`、`out/CMP3.tsv`）都承载 `96.94` 与
`4.372266` 这两个串；**这两个串落在哪些 argv / 配对 sha 之下，我没有逐条复核**
⇒ 按③处理。

**(b) 触发量反推（原第 111 行）**

```
| Trigger derived from the rate gap | `steps_per_call` computed per pass from `budget_bytes` vs `total_bytes` (`src/product/kv_block_descent.h`) | `TOTALS-DISAGREE` 26/27 passes -> 0/27 |
```

**撤因**：② ＋ ④。**证据在哪**：`dl/kvrate2/blob_F1248.md`、
`dl/kvrate2/EVIDENCE.txt`、`dl/kvaxis/raw/05_descent_face.txt`、
`dl/kvplanar/out/DIFF.txt` 都出现 `TOTALS-DISAGREE`；`26/27 -> 0/27` 这一对读数
**未找到**指名它的那一条 argv/sha ⇒ 证据不足。

**(c) 第三根轴的五条内存读数（原第 114 行）**

```
| **Third axis**: a plane may be sized for its own class' pages | `KVPlaneGeometry::page_group_count`; knob `NINFER_KV_AXIS3_NARROW_PAGES` | `kv cache payload` 4.38 -> 3.56 GiB, `gpu sequence used` 5.71 -> 4.88 GiB, plan coin byte-identical (884,736,000 B), prefill +0.013% (indistinguishable) |
```

**撤因**：②（六臂、单 argv 家族、单次电池的读数被写成结论）。**证据在哪**：
`dl/kvaxisA/EVIDENCE.txt` §4 与 §6 —— 这一条**证据最完整**：配对 sha
`f2d65460cdbb3868422bf163266a37bdae2c0e8b5e3e51e0af1a4fff019f878f`（`ninfer`）与
`6f66a541ab53b616ebf023b6ee8d3fef80a03546a23a150844038229cd526db6`（`ninfer-perplexity`），
完整 argv（`--kv-dtype int8 --messages … long_niah_64k.json --max-context 131072
--max-new 32 --kv-capacity auto --greedy --spec mtp --draft-tokens 9 --cold-policy
window --max-cold-pages 1024 --cold-keep-tokens 128`），六臂表与"A 自己的代价
+0.013%"的算法。⇒ 数字撤走，**指针留下**（新 README 写"本机实测；配对 sha 与完整
argv 记在 `dl/kvaxisA/`"）。同批另有 `dl/kvplanar/`、`dl/kvplanar2/`（`884736000`）。

**(d) 冷层落盘（原第 116 行）**

```
| Cold tiers (KVMem) | `--cold-policy window|host|disk|host-then-disk` | disk spill measured: 1,198,443,776 B, `write_failures=0`, `drain_ms=933`; **cold refetch is still never exercised** |
```

**撤因**：②（单次落盘电池的三个读数）＋ ④。**证据在哪**：`dl/kvmpf/out/`
（`n64k_onblk_r1.log`、`n64k_onax3_r1.log`）、`dl/kvmpf2/out/MPF2.tsv`、
`dl/kv1m/EVIDENCE.txt`、`dl/kv1m/out/HARVEST.tsv` 都承载 `1198443776` 与
`drain_ms` / `write_failures`。⇒ 指针留下，数字撤走。"回读从未演练"这句**不是数字**，
保留（坐标见 §5）。

**(e) "Being worked on now" 里的两个对价与百分比（原第 125-126 行）**

```
   `e8-2bit` pair costs **13,312 B/cell**; with V able to carry E8 the symmetric pair is
   **9,216 B/cell** (-30.8%). The geometry and the arbitrary-pair pricing already exist; the
```

**撤因**：`-30.8%` 属于 ②（由两个按构造对价推出来的百分比，写成了"收益"）⇒ 撤走；
`13,312` 与 `9,216` 本身**是按构造精确的，保留**（见 §5；撤档这里只记那个百分比的
下场）。**证据在哪**：`src/product/kv_e8_width.h:491,494`（`static_assert`）、
`src/product/kv_cell_modes.h:585,589,597`。

**(f) 三条"已知的、没有藏起来的问题"里的全部数字（原第 137-146 行）**

```
- Prefill against the unmodified engine is **lower** by 4.11% / 5.31% / 5.54% in three independent
  readings with disjoint intervals. The no-regression rule is "faster or equal, never slower", so
  this is the one open acceptance blocker, and its mechanism is **not yet identified**.
- Long context erodes the mixing: the demoted share is 9.32% at 8k, 2.68% at 64k and 0.60% at 128k;
  at 128k a single `retired=1024` wipes demotions (`rk4v4 672 -> 16`), and the walk under-reports its
  own demotions by 2.05x-2.41x.
- The **decode** column cannot be used as acceptance: its measured instrument noise is **30.7%**.
- 1M context is blocked by three separate rings: the native context gate (262,144, `--yarn` for 4x),
  a device shortfall of 10.43 GiB, and a cold-tier deficit of 8,457 pages (8.86 GiB) against a
  longest actually-run context of 260,096 tokens.
```

**撤因**：②＋③＋④。**证据在哪**：
- prefill 三个偏低值：`dl/kvspend/`（`EVIDENCE.txt`、`out/arms/B/stderr.txt`）、
  `dl/cmpfire3/blob_F1250.md`、`dl/kvaxisA/EVIDENCE.txt` §6 逐字点名（它自己说
  "-5.54 % on the means"、"`dl/kvspend` measured -4.11 %"、"F-1250 -4.11 %/-5.31 %"）
  ⇒ 这三个数**是转述**（③），原始三份电池我没有逐份打开。
- 侵蚀三档 `9.32% / 2.68% / 0.60%` 与 `rk4v4 672 -> 16`、`retired=1024`：
  `dl/kvplanar/EVIDENCE.txt`、`dl/kvplanar/out/GRADED.txt`、
  `dl/kvplanar2/out/PLAN_EVIDENCE.txt`、`dl/kvrate2/`、`dl/kvaxis/EVIDENCE.txt`。
- 走道欠报 `2.05x-2.41x`：`dl/cmpfire3/blob_F1250.md`、`dl/cmpfire3/STATE.md`。
- decode 仪器噪声 `30.7%`：`dl/cmpfire3/`（F-1250）；`dl/kvaxisA/EVIDENCE.txt` §6 逐字
  引用"`dl/cmpfire3`/F-1250 measured a 30.7 % span on six arms with the SAME argv"。
- 1M 三道环：`dl/kv1m/EVIDENCE.txt` 与 `dl/kv1m/out/HARVEST.tsv`（`10.43`、`8457`、
  `260096`）、`dl/long1m/logs/32_kvc_15_t7202_L8.probe.out`、
  `dl/kvmemfix/C1.err`、`dl/ctxsweep/R4_svc690k.err`、`dl/1mmtp/logs/`。

---

## 2. 从 `README_FEATURES.md` 撤走的内容（逐字）

### 2.1 基准与工作区规模（原第 7-11 行）

```
**基准**：仓库 `HEAD = 3944a53eda1aac439a566a1cf46ea741f0415fdc`（实测 `git rev-parse HEAD`），
**工作区是脏的**：`git status --porcelain` 共 **794** 条，其中已 `A` **236** 条、未跟踪 `??` **168** 条
（实测命令 `git status --porcelain | wc -l`、`grep -c '^A'`、`grep -c '^??'`）。
**2026-09-22 复核**（`accept34` 线）：**817** 条，其中 `A` **236**（未变）、`??` **179**。
同一个基数在动 ⇒ 本行给的是**读数与读数的时刻**，不是一个固定值。
```

**撤因**：②（一个**移动的**基数在三个时刻的读数，被写进文档头部当"基准"）＋ ④（不是
可从这棵树复核的性质，只能从历史记录目录复核）。**证据在哪**：`dl/accept34/`（2026-09-22
那次复核的账，见 `docs/features/verification.md` §6.5 自述的 `dl/accept34/pre/`、
`dl/accept34/logs/s70_docpatch.txt`）；`3944a53` 这个修订与 `794/236/168` 这一组是更早
一轮的记录目录读数。
**同一天稍晚的实测**（撤它的直接理由）：2026-09-30 14:09（本线开工时）
`HEAD = de80bd1f56b789ad5415c187721815ad1f6a963a`，
`git status --porcelain | wc -l` = **1**（就是 `src/product/kv_tier_formats.h` 那条别的
线的在飞改动）；而写这份保档的**同一个小时内**，工作区又多了一条未跟踪的
`?? dl/importopt` ⇒ 那个**1** 当场就过期了。
**这就是为什么文档头不该写死一个会动的基数。**

### 2.2 目标族表的"文件数 / 代码行数"两列（原第 34-46 行）

```
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
```

**撤因**：②（文件数与行数是"这一刻的树"的快照，且随别人的改动立刻漂）＋ ④（"九个"
与两列数字没有点名 `dl/` 证据路径，只有一条命令）。**证据在哪**：复现命令是
`docs/features/verification.md` §4 的那两条（`wc -l <file>`、`find <dir> -type f | wc -l`）；
**逐族的数字本身没有留档**。**同表的"出处"一列全部已漂**（见 §4.3）。

### 2.3 各族规模、工具行数、头文件行数（逐字）

```
- ⚠ **`gemma4_31b` 是新增的第九族**（上一轮的穷尽清单里没有它）。它有 4 个文件，
  转换侧对应 `tools/convert/gemma4_31b/`：`convert.py` 438 行、`inventory.py` 311 行、`recipe.py` 260 行
  （实测 `wc -l`），且已进 `add_subdirectory`（`src/CMakeLists.txt:613`）。
```

```
- ⚠ `qwen3_8_flash_next` 代码量很大（86 文件 / 21346 行，`src/targets/qwen3_8_flash_next/CMakeLists.txt:2-38`
  列出 36 个源）但**未注册进 engine**：`registry.h` 不 include 它。
```

```
| 自适应草稿宽度（生存/代价准则逐轮选档） | `--spec mtp --draft-tokens 0` | `src/targets/qwen3_6/impl/runtime/mtp_window_cut.h`（370 行）；env 名表 `:15,71,95,146,267`；调用点 `src/targets/qwen3_6/impl/runtime/program_impl.h:11073-11128` |
```

```
| KV 位预算求解（逐层档位表） | `--kv-bit-budget` / `--kv-bits` / `--kv-k-bits`+`--kv-v-bits` | `apps/cli/options.cpp:381,403,408,412`；求解器 `src/product/kv_bit_budget.h`（2490 行）、`src/product/kv_kv_bits.h`（1341 行） |
```

```
| KV 页池预分配 | env `NINFER_KV_PAGING_PREALLOC` | 读取点 `src/targets/qwen3_6/impl/runtime/layouts_impl.h:2093`；头 `src/product/kv_paging_preallocation.h`（582 行） |
```

```
| 行尺度校准闭环（KV 行尺度表落盘/复用） | `--kv-row-scale auto|off|FILE`、`--recalibrate` | `apps/cli/options.cpp:532,537`；闭环 `src/product/kv_rowscale_persist.h`（939 行） |
```

**撤因**：②＋④（"文件数 / 行数"都是单一时刻的树快照，没有点名证据路径）。
**证据在哪**：`tools/convert/gemma4_31b/*.py` 的 `wc -l`、`docs/features/importers.md`
（那一册记工具行数）、`docs/features/verification.md` §4。**行数本身未留档**。

### 2.4 第三节"怎么读出处"里的过期行号例（原第 102-104 行）

```
- **`路径:行号`** 一律指**上面那个基准修订的工作树**。行号是实测的，不是从别处转录的
  （上一轮的清单是**记录目录**里的产物，不在本仓库内；它的行号在这棵树里已经漂了：它记的
  `apps/cli/options.cpp` 是 1001 行，实测 1016 行；它记 `--kv-dtype` 在 `:363`，实测在 `:378`）。
```

**撤因**：②（"实测 1016 行 / `:378`"这两个读数当场是对的，但作为**正文数字**写下来就
立刻过期；本次实测是 **1551 行**，`--kv-dtype` 的 case 在 **`:552`**、用法文本在
**`:252`**）＋ ④。**证据在哪**：`apps/cli/options.cpp` 的 `wc -l` 与
`grep -n 'arg == "--kv-dtype"'`（命令在 `docs/features/verification.md` §4）。
⇒ 结论保留、数字撤走：**行号会漂，必须重测**。

### 2.5 第六节"本轮聚焦"里的读数（逐字）

**(a) 6.1 表两行**

```
| **逐格降档 + 速率标尺** | 一格 =（64 token 块 × 一个文本层）；`NINFER_KV_BLOCK_BUDGET_RATE_X10000`（读取点 `src/product/kv_block_budget_stage.h:362`） | 可动 **96.94%** 的 cell；计划币 **4.372266 b/el**（少 47% 字节） |
```

```
| **触发量由速率差反推** | `steps_per_call` 每趟按 `budget_bytes` 与 `total_bytes` 的差额算（`src/product/kv_block_descent.h`） | `TOTALS-DISAGREE` 由 26/27 趟 → **0/27** |
```

```
| **第三根轴**（平面按自己那一类定页数） | `KVPlaneGeometry::page_group_count`；旋钮 `NINFER_KV_AXIS3_NARROW_PAGES` | `kv cache payload` 4.38 → **3.56 GiB**、`gpu sequence used` 5.71 → **4.88 GiB**，计划币**逐字节一致**（884,736,000 B），prefill **+0.013%**（不可分辨） |
```

```
| **冷层（KVMem）** | `--cold-policy window|host|disk|host-then-disk` | disk 落盘实测 **1,198,443,776 B**、`write_failures=0`、`drain_ms=933`；⚠ **回读（refetch）至今零行** |
```

**撤因**：同 §1.6 的 (a)(b)(c)(d)。**证据在哪**：`dl/kvrate2/`、`dl/kvaxisA/EVIDENCE.txt`
（配对 sha + 完整 argv）、`dl/kvplanar/`、`dl/kvplanar2/`、`dl/kvmpf/`、`dl/kvmpf2/`、
`dl/kv1m/`。

**(b) 6.2 里 V 承载 E8 的两个对价（原第 130 行）**

```
2. **V 承载 E8**：今天部署的窄形态把 V 钉在 128 B 的 i4 平面，所以 `e8-2bit` 的对价是 **13,312 B/格**；V 能承载 E8 之后对称对是 **9,216 B/格（−30.8%）**。几何与任意对定价**已有**，缺的是**解码侧消费者**（`e8_kv_lattice_decode_group<2>` 无调用者）。
```

**撤因**：`−30.8%` 属于 ②，撤走；`13,312` 与 `9,216` 按构造精确，**保留**（见 §5）。
**证据在哪**：`src/product/kv_e8_width.h:491,494`、`src/product/kv_cell_modes.h:585,589,597`。

**(c) 6.3 全部四条（原第 137-140 行）**

```
- **prefill 对未改动引擎偏低 4.11% / 5.31% / 5.54%**（三次独立读数、区间不交）。判据是"快或等、绝不更慢"，所以这是**唯一未过的验收项**，而它的机制**尚未识别**。
- **长上下文侵蚀混档**：降档占比 8k **9.32%** → 64k **2.68%** → 128k **0.60%**；128k 下**单条 `retired=1024`** 把 `rk4v4 672→16` 一次抹掉；且走道**欠报**自己的降档 2.05×–2.41×。
- **decode 列不作验收**：实测仪器噪声 **30.7%**。
- **1M 被三道环挡住**：原生上下文门 **262,144**（`--yarn` 放 4×）、设备**缺 10.43 GiB**、冷层**亏 8,457 页 = 8.86 GiB**；已实测跑到的最长上下文是 **260,096 token**。
```

**撤因**：同 §1.6(f)。**证据在哪**：`dl/kvspend/`、`dl/cmpfire3/`、`dl/kvplanar/`、
`dl/kvplanar2/`、`dl/kvrate2/`、`dl/kvaxis/`、`dl/kv1m/`、`dl/long1m/`、`dl/combo1m*/`、
`dl/1mmtp/`、`dl/kvmemfix/`、`dl/ctxsweep/`。

---

## 3. 撤因与证据的逐条对照表

| # | 撤下的数字 | 撤因 | 原本的证据在哪 |
|---|---|---|---|
| 1 | `1.0202 / 0.020 / 2027.7`、`1.1120 / 0.106 / 2082.7`、`1.7055 / 0.534 / 1949.1`、`1.5217 / 0.420 / 2017.2`、`1.2962 / 0.259` | ② ④ | **证据未找到**（全 `dl/` 搜索只命中无关目录的巧合子串；归档里真实的 ppl 家族量级不同） |
| 2 | `13.3k zh corpus`、`4096 ctx` | ② ④ | **证据未找到**（该语料与那批 argv 没有留档名） |
| 3 | `≈260 B / ≈4.06`、`288 B / 4.50`、`512 B / 8.00` | ④ | 同表在 `VRAM.md:28-34`；无配对 sha、无 argv ⇒ 可复核性不足 |
| 4 | `0.05 / 20.5 / 20.6 / 20.9 / 21.4 / 20.6 GB` | ② ④ | 同表在 `VRAM.md:8-17`；无配对 sha、无 argv ⇒ 可复核性不足 |
| 5 | `96.94%`、`4.372266 b/el`、`47%` | ② ③ ④ | `dl/kvrate2/EVIDENCE.txt`、`dl/kvrate2/blob_F1248.md`、`dl/kvrate2/out/RATE2.tsv`、`dl/cmpfire3/out/CMP3.tsv`（未逐条复核 argv/sha） |
| 6 | `TOTALS-DISAGREE` `26/27 → 0/27` | ② ④ | `dl/kvrate2/`、`dl/kvaxis/raw/05_descent_face.txt`、`dl/kvplanar/out/DIFF.txt`（未点名 argv） |
| 7 | `4.38 → 3.56 GiB`、`5.71 → 4.88 GiB` | ② | `dl/kvaxisA/EVIDENCE.txt` §4（配对 sha `f2d65460…` / `6f66a541…`，完整 argv 在 §4） |
| 8 | `884,736,000 B` | ② | `dl/kvaxisA/EVIDENCE.txt` §4（同 sha/argv）；`dl/kvplanar/`、`dl/kvplanar2/` |
| 9 | `prefill +0.013%` | ② | `dl/kvaxisA/EVIDENCE.txt` §6（同 sha/argv，六臂表） |
| 10 | `1,198,443,776 B`、`write_failures=0`、`drain_ms=933` | ② ④ | `dl/kvmpf/out/n64k_onblk_r1.log`、`dl/kvmpf2/out/MPF2.tsv`、`dl/kv1m/EVIDENCE.txt` |
| 11 | `-30.8%`（`13,312`→`9,216` 的百分比） | ② | 两个对价本身按构造精确（见 §5）；这个百分比没有独立证据路径 |
| 12 | prefill `4.11% / 5.31% / 5.54%` | ② ③ ④ | `dl/kvspend/`、`dl/cmpfire3/blob_F1250.md`、`dl/kvaxisA/EVIDENCE.txt` §6（**转述**："F-1250 记 -4.11 %/-5.31 %"） |
| 13 | `9.32% / 2.68% / 0.60%`、`rk4v4 672 → 16`、`retired=1024` | ② ④ | `dl/kvplanar/EVIDENCE.txt`、`dl/kvplanar/out/GRADED.txt`、`dl/kvplanar2/out/PLAN_EVIDENCE.txt`、`dl/kvrate2/`、`dl/kvaxis/` |
| 14 | `2.05x - 2.41x`（走道欠报） | ② ④ | `dl/cmpfire3/blob_F1250.md`、`dl/cmpfire3/STATE.md` |
| 15 | `30.7%`（decode 仪器噪声） | ② ③ | `dl/cmpfire3/`（F-1250）；`dl/kvaxisA/EVIDENCE.txt` §6 是**转述**那条 |
| 16 | `262,144`、`10.43 GiB`、`8,457 页 / 8.86 GiB`、`260,096 token` | ② ③ ④ | `dl/kv1m/EVIDENCE.txt`、`dl/kv1m/out/HARVEST.tsv`、`dl/long1m/logs/`、`dl/1mmtp/logs/`、`dl/kvmemfix/`、`dl/ctxsweep/`（**多来源转述**，未逐条复核） |
| 17 | `794 / 236 / 168`、`817 / 236 / 179` | ② ④ | `dl/accept34/`（2026-09-22 复核）；今天实测 = `1`，见 §2.1 |
| 18 | 目标族两列（`81 / 50289` … `86 / 21346`） | ② ④ | `docs/features/verification.md` §4 的命令；数字未留档 |
| 19 | `convert.py 438`、`inventory.py 311`、`recipe.py 260` | ② ④ | `docs/features/importers.md`；数字未留档 |
| 20 | `mtp_window_cut.h（370 行）`、`kv_bit_budget.h（2490 行）`、`kv_kv_bits.h（1341 行）`、`kv_paging_preallocation.h（582 行）`、`kv_rowscale_persist.h（939 行）` | ② ④ | `wc -l`（`docs/features/verification.md` §4）；数字未留档 |
| 21 | `options.cpp 1001 → 1016 行`、`--kv-dtype :363 → :378` | ② ④ | 今天实测 `1551` 行、case `:552`、用法 `:252`；那两个旧读数已过期 |
| 22 | `NINFER_KV_BUDGET_RULER_F1231` 作为"旋钮" | 不属于四类：**误述** | 它是 `src/product/kv_block_budget_stage.h:240` 的 **编译期 `#define`**，不是环境变量、不是运行期旋钮 ⇒ 见 §4.2 |

---

## 4. 撤走的非数字内容（悬空引用 / 误述 / 过期坐标）

### 4.1 悬空引用：`THIRD-PARTY.md`

见 §1.5。**树里不存在**；新 README 不再链接它，血缘改写为带坐标的断言。

### 4.2 误述：`NINFER_KV_BUDGET_RULER_F1231` 不是旋钮

原 `README.en.md` 旋钮表与 `README_FEATURES.md` 6.4 都把它列成"绝对预算按哪把尺读"的
环境变量。实测：全树只有一处，`src/product/kv_block_budget_stage.h:240`：

```
#define NINFER_KV_BUDGET_RULER_F1231 1
```

它是**编译期宏**，不是 `getenv` 的键（同一文件里**真正**的 `getenv` 键是
`kKvBlockStageBudgetEnv`、`kKvBlockStageBudgetBlocksEnv`、`kKvBlockStageBudgetRateEnv`，
分别定义在 `:176`、`:185`、`:233`；速率臂的读取点在同文件 `:362`）。⇒ 该行从旋钮表
**撤走**，并在新 `README_FEATURES.md` 里按名写明它是构建期开关。

### 4.3 过期坐标（`file:line` 已漂，内容保留、坐标重测）

| 旧坐标 | 旧文写的 | 2026-09-30 实测 |
|---|---|---|
| `src/CMakeLists.txt:604-608,612,613` | 各目标族的 `add_subdirectory` | `:703-707`（六族）、`:711`（`qwen4_exp`）、`:712`（`gemma4_31b`） |
| `src/CMakeLists.txt:635,636` | 两个 `EXCLUDE_FROM_ALL` | `:762`（`qwen3_vision`）、`:763`（`qwen3_8_flash_next`） |
| `apps/CMakeLists.txt:33` | `ninfer-perplexity` | `:24`（另：`:1` `ninfer`、`:13` `ninfer-serve`） |
| `apps/CMakeLists.txt:60-62` | `install(TARGETS …)` | `:75` |
| `CMakeLists.txt:515-520 / 525 / 567-569` | 文档组件与安装清单 | `:551` 与 `:600`（`COMPONENT documentation`）、`:557`（`file(WRITE …)`） |
| `apps/cli/options.cpp:93-107` | `parse_kv_cache` | `:186`（别名差异的自述在 `:189` 与 `:193-196`） |
| `apps/cli/options.cpp:159` | `--kv-dtype` 用法 | `:252` |
| `apps/cli/options.cpp:502` | `--kv-tier-formats` | `:803` |
| `apps/cli/options.cpp:565/578/670/680/682` | `--spec` / `--draft-tree` / `--graph-capture-ceiling` / `--no-cuda-graph` / `--yarn` | `:879` / `:892` / `:1002` / `:1014` / `:1016` |
| `apps/cli/options.cpp:371/381/403/408/412` | `--prefill-chunk-mode` / 位预算族 | `:545` / `:558` / `:629` / `:638` / `:647` |
| `apps/cli/options.cpp:612,631,650,652` | 冷层族 | `:928`（`--cold-policy`）等，用法 `:399-408` |
| `apps/cli/options.cpp:532,537` | `--kv-row-scale` / `--recalibrate` | `:839` / `:848` |
| `apps/cli/options.cpp:657,659,661,668` | 权重卸载四旗标 | `:988` / `:990` / `:992` / `:1000` |
| `apps/cli/options.h:42,44` | `spec` / `use_cuda_graph` 默认 | `:51` / `:53` |
| `apps/cli/options.h:139` | `graph_capture_ceiling` 默认 | `:177` |
| `src/targets/qwen3_6/impl/runtime/program_impl.h:13411` | `pass_holds` | `:13627` |
| `src/targets/qwen3_6/impl/runtime/layouts_impl.h:2093` | `NINFER_KV_PAGING_PREALLOC` 读取点 | `:2589` |
| `src/targets/qwen3_6/impl/runtime/layouts_impl.h:1166` | `--yarn` 的上下文上限改动 | `:1333` |
| `src/targets/qwen3_6/impl/runtime/mtp_window_cut.h:15,71,95,146,267` | env 名表 | **仍然成立**（这一组未漂） |
| `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h:17,18,22,26,30` | 树/链结构常量 | `:99`（`draft_tree_paths`）、`:100`（`draft_tree_depth`）、`:95-98` 与 `:131-154` 是自述 |
| `src/kvcfg/kv_formats.h:5,27,68,70,329` | 语法行 / 枚举 / 模式 / 结构 / 解析器 | **仍然成立** |
| `src/product/kv_block_budget_stage.h:362` | 速率臂读取点 | **仍然成立** |
| `src/targets/registry.h:5-8,19-22` | registry 的四个 package | `:5-9` 是**五**个（多了 `spark_x2_5_4b`），别名 `:20-23` 与 `:30` |

### 4.4 registry 计数从"四个"改为"五个"

原 `README_FEATURES.md` 写"registry **认识的 package 有四个**"。实测
`src/targets/registry.h:5-9` 是五个 include（`qwen3_6_27b`、`qwen3_6_35b_a3b`、
`muse_glimmer_30b`、`qwen3_5_9b`、`spark_x2_5_4b`），`using` 别名在 `:20-23` 与 `:30`，
`ActiveTarget` 变体在 `:197`。⇒ 按实测改成五个（这是**计数**，不是性能数字，但它属于
"没点名证据路径的过期断言"，一并记在这里）。

### 4.5 已知限制里我未能复核的一条

原 `README.en.md` 的 Known limitations 写：

```
- Residual planes (`--kv-residual-layers`): block_tables corruption not yet
  located (experimental, off by default).
```

**撤因**：④（"`block_tables` 损坏"这个说法我在这棵树里**找不到记录坐标**）。
**已复核的部分**：旗标存在且两个前端都解析（`apps/cli/options.cpp:790` 的 case、
`:291` 的用法行；解析器 `product::parse_kv_residual_layers_spec`），非默认路径。
⇒ 新 README 只写已复核的部分，"损坏未定位"一句按名保留在本档。

---

## 5. 没有撤、且仍然保留的数（"按构造精确"，带坐标）

这些是**代码里 `constexpr` / `static_assert` 钉死的字节数**，不是运行读数：改一个
bit 的定义就会编译失败。它们在新 README 里保留，并给出坐标。

| 事实 | 值 | 坐标 |
|---|---|---|
| E8 的 K 平面（W4 / W3 / W2） | `8704` / `6656` / `4608` B | `src/product/kv_e8_width.h:183,185,187` |
| E8 的 V 平面（出厂 i4，三个宽度共用） | `8704` B | `src/product/kv_e8_width.h:148,157` |
| E8 的层（K+V）W4 / W3 / W2 | `17408` / `15360` / `13312` B | `src/product/kv_e8_width.h:189,190,191` |
| 任意对定价 `e8_kv_pair_bytes(K, V)` 的四个已断言对 | `(B4,B4)=17408`、`(B4,B3)=15360`、`(B2,B4)=13312`、`(B2,B2)=9216` | `src/product/kv_e8_width.h:319,321,333,491,494` |
| `(rk4v4, rk4v4)` = `(B4,B4)`，出厂对 | `17,408` B | `src/product/kv_cell_modes.h:577` |
| `(rk3v4, rk4v4)` = `(B3,B4)` | `15,360` B | `src/product/kv_cell_modes.h:580` |
| `(e8-2bit, rk4v4)` = **`rk2v4`** = `(B2,B4)` —— V 仍钉在 i4 平面 | `13,312` B/格 | `src/product/kv_cell_modes.h:585` |
| `(e8-2bit, e8-2bit)` = `(B2,B2)` —— 对称地板 | `9,216` B/格 | `src/product/kv_cell_modes.h:589` |
| 上面两个对价的差 = `B4 − B2` 平面 | 断言在 | `src/product/kv_cell_modes.h:597` |
| KVarN 记录（payload 与记录等宽） | `26880` B | `include/ninfer/ops/kvarn.h:36,37` |
| E8 位/元素（W4 / W3 / W2，×100） | `425` / `375` / `325` | `src/product/kv_e8_width.h:193,194,195` |
| 同上，同一组断言在测试里的复述 | `114/114` 等检查 | `src/ops/kv/e8_width_codec_test.cpp:127-132`、`:1070` |

**注意**：`13,312` 与 `9,216` 是"**定价**"，不是"**能读**"。同一棵树的
`src/product/kv_cell_modes.h:667`（`cell_pair_read_side`）把读侧标成 `Reserved`；
`e8_kv_lattice_decode_group<3>/<2>` 的全树状态见 §6 第 3 条。⇒ 引用这两个数时
必须同时写"读侧未接"。

---

## 6. 我无法复核、因而撤掉的（UNVERIFIED，逐条）

1. **`README.en.md` 的整张 ppl / prefill 表**（`1.0202`、`1.1120`、`1.7055`、`1.5217`、
   `1.2962` 与五个吞吐值，以及 `13.3k zh corpus` 这个语料名）。全 `dl/` 逐字搜索没有
   命中，`dl/` 里真实的 ppl 家族量级与它不同 ⇒ **证据未找到**。
2. **`VRAM.md` 那张表的六个内存值**（含 `0.05 GB` 空闲）。`VRAM.md` 自述方法但没有
   配对 sha / 完整 argv ⇒ 我不能复核到"同一次电池"。
3. **`≈260 B / 288 B / 512 B` 每 head/token 的容量行**。同一原因；且 `E8Kv` 一行
   自述为近似值。
4. **prefill 偏低的三个百分比**（`4.11 / 5.31 / 5.54`）。它们是**从别的线转述**进
   README 的（`dl/kvaxisA/EVIDENCE.txt` §6 也是转述 `dl/kvspend` 与 F-1250）⇒ ③。
5. **decode 列的 `30.7%` 仪器噪声**。同上，是转述；我没有打开 F-1250 的原始六臂产物。
6. **1M 的四组读数**（原生门 `262,144`、设备缺 `10.43 GiB`、冷层亏 `8,457 页 /
   8.86 GiB`、最长跑过 `260,096 token`）。来源散在 `dl/kv1m/`、`dl/long1m/`、
   `dl/1mmtp/`、`dl/kvmemfix/`、`dl/ctxsweep/` 五处，**未逐条复核到同一次 argv**。
7. **侵蚀三档 `9.32% / 2.68% / 0.60%` 与 `rk4v4 672 → 16`**。多目录承载，未复核到
   单次 argv。
8. **走道欠报 `2.05x–2.41x`**。只在 `dl/cmpfire3/` 的 `blob_F1250.md` 与 `STATE.md`
   里出现，我没有复核它的原始产物。
9. **工作区规模 `794 / 236 / 168` 与 `817 / 236 / 179`**。它们**当场就是对的**，但作为
   正文基准写下来即过期（今天实测 `1`）⇒ ②，撤。
10. **各族文件数与行数、五个头文件与三个转换脚本的行数**。只有命令、没有留档数字
    ⇒ ④，撤。
11. **`THIRD-PARTY.md` 这一份文件**。它不存在，所以链接与"see THIRD-PARTY.md"这句
    不可复核 ⇒ 撤（血缘断言的坐标已补，见 §1.5）。
12. **"`block_tables` 损坏未定位"**。找不到记录坐标 ⇒ 撤（见 §4.5）。

---

## 7. 本档的边界（明说）

- 本档**只**做两件事：**逐字保留**被撤下来的内容、**逐条**写清撤因与证据在哪。
- 本档**不**声称那里面的数字现在成立、也**不**声称它们不成立——它们只是"从 README
  撤下来的原话"。
- 本档**没有**跑引擎、没有 build、没有占卡；全部内容来自读文件、`grep` 与
  `git rev-parse`。
- 本档**没有**改任何源文件，也**没有**改 `README.md`（上游那份）。
