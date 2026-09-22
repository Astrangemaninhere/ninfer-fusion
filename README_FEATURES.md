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
语法 `hot=bf16,tail=fp16,cold=iso4e`（`src/kvcfg/kv_formats.h:5`），入口是 `--kv-tier-formats`
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
