# 未完成 / 部分实现 / 按名找不到（逐条）

本册是主人明确要的「不能漏」那一册。分三段：**A. 部分实现**（有实现但受限）、**B. 未做**
（只有名字或注释）、**C. 按名报：查到的是部分/未实现/找不到**（含我自己的负对照），
**D. 我在核验中发现的漂移与缺口**（上一轮清单的问题，逐条带实测）。

判定口径：**部分** = 有实现但能力受限、或只有骨架、或无调用路径、或已具名拒绝；
**未做** = 只有注释/名字，无实现。

---

## A. 部分实现（逐条）

### A17 —— `tail` 档只能重复 `hot`
- **是什么**：`--kv-tier-formats hot=...,tail=...,cold=...` 里的 `tail` 档（近期高精度窗）。
- **卡在哪**：引擎**还没有**近期窗档位，所以 `tail` 只在**与 `hot` 同值**时被接受。
- **出处**：用法文本逐字 `apps/cli/options.cpp:228-229`（「tail is accepted only when it repeats hot:
  the engine has no recent-window tier yet」）；词汇表头 `src/kvcfg/kv_formats.h:8`（中文原话「尾=近期高精度窗」）。
- **文档该怎么写**：可以写 `tail=` 这个键存在；**不能**写「有独立的近期窗档位」。

### B9 —— `NINFER_MTP_ADAPTIVE` 是自述的临时补丁
- **是什么**：`=1` 时把 MTP 草稿宽度强制成 0（走自适应阶梯）。
- **卡在哪**：源码自述是临时块。`apps/cli/options.cpp:761-762` 逐字
  「TEMPORARY adaptive-MTP escape hatch ... DELETE THIS BLOCK when the real fix lands」；
  真修法 (a)(b)(c) 列在 `:783-793`。
- **一处已过期的注释**：同一段 `:770-772` 说「`parse_u32()` 在 `:237` 被不带 `allow_zero` 调用，
  所以 `--draft-tokens 0` 被拒」，实测该调用现在**传了** `allow_zero=true`（`:576`），
  ⇒ 真修法 (c) 条（两个前端对 0 的接受度一致）在 CLI 侧已经落地，注释里的行号与结论都过期了。
- **出处**：`apps/cli/options.cpp:761-793`、`:576`、`:567`；env 读取点 `:796`；serve 侧 `src/serve/serve_options.cpp:697`。

### S4 —— per-variant 的 MTP 域在 CLI 侧到不了
- **是什么**：各族的草稿宽度上限不同（27b = 15，35b_a3b / muse_glimmer_30b / qwen3_5_9b = 5）。
- **卡在哪**：CLI 侧的域校验**硬编码 15**（`src/product/speculative_options.h:86`），
  per-variant 的值要到 `validate_target_options` 才可达。源码自己把这条列为待修 (b)
  （`apps/cli/options.cpp:787-790`）。
- **出处**：域值在 `src/targets/*/impl/variant.h:37/33/39/47`（`maximum_mtp_draft_tokens`）；
  统一入口 `src/targets/qwen3_6/impl/runtime/instance.h:62`。

### T5 —— `qwen4_exp` 只是 identity 注册
- **卡在哪**：`src/targets/qwen4_exp/CMakeLists.txt:1-4` 逐字「identity-only registration. No sources yet」；
  只登记了一个源 `impl/ple_ngram.cpp`（`:14-15`）；`impl/package.cpp` / `impl/variant.cpp` / `impl/load/bindings.cpp`
  属于 stage (b)，尚未落地（这三个文件名在各族里同名，故这里带上目录前缀）。
- **出处/规模**：该目录 9 文件 / 866 行（实测）；它在 `src/CMakeLists.txt:612` 被无条件加入构建。

### T6 —— `qwen3_vision` 是骨架
- **卡在哪**：只有一个源。`src/targets/qwen3_vision/CMakeLists.txt:2-3` 逐字
  `add_library(ninfer_qwen3_vision STATIC impl/encoder.cpp)`；该目录 3 文件 / 501 行（实测）。
- **另外**：它是 `EXCLUDE_FROM_ALL`（`src/CMakeLists.txt:635`），默认**不**构建。

### T19 —— 多设备分片是 host-only，无调用路径
- **卡在哪**：`CHANGELOG.md:33-35` 逐字「It is **host-only and has no call path**」，并列出**所有**缺失件：
  rank plumbing、collectives、peer access、逐 rank 权重加载器、召回记录里的 rank 字段。
- **已有的部分**：分片算术与逐 rank 命名在 `src/core/shard_plan.h`（564 行）、
  `src/core/shard_rank_axis.h`（650 行）；契约 `docs/maintainer/multi-device-and-shard-plan.md`（475 行）。
- **另一条同源证据**：`src/core/arch_caps.h:2065-2066` 自述 `allreduce` 在 `src/CMakeLists.txt` 里
  **0 命中**，而该文件的源列表是显式的（无 GLOB）⇒ 这套线是**惰性**的。

### P7 —— `_WIN32` 是逐点可移植，不是平台支持
- **实测**：`grep -rln '_WIN32' src/ apps/ tools/` 命中 **13** 个活文件 + 1 个前像
  （`src/product/kv_rowscale_persist.h.gb2-orig`）：`apps/cli/options.cpp`（`::_putenv_s` vs `::setenv`，
  见 `:136-142`）、`src/runtime/engine/context_cost.cpp`、`src/artifact/reader.cpp`、
  `src/serve/serve_options.cpp`、`src/serve/http_transport.cpp`、`src/serve/request_log.cpp`、
  `src/ops/ple/ple_table.h`、`src/ops/ple/ple_table.cu`、`src/core/uint128.h`、
  `src/product/kv_rowscale_persist.h`、`src/product/load_progress/load_progress.cpp`、
  `tools/reference/qwen3_8_flash_next/main.cpp`、`tools/kv_relayout_test.cpp`。
- ⚠ **上一轮清单列的 15 个文件里有三个现在不在命中里**（`src/runtime/engine/bandwidth_governor.h`、
  `src/targets/qwen3_6/impl/runtime/program_impl.h`、`src/spec/sum_dir_reach.h`）⇒ 以本册实测为准。
- **文档该怎么写**：写「有 N 处逐点 `_WIN32` 分支」，**不能**写「支持 Windows」。

### ?9 —— 三个「孤儿测试」是否注册：**已从「未测」变为「已测」**
- **上一轮的状态**：清单把 `test_kernel_route.cpp`、`test_shard_plan.cpp`、`test_multidev_wiring.cpp`
  是否注册进 CTest 列为「我没测」。
- **现在的实测（本册补上）**：三者**都已注册**：
  `tests/CMakeLists.txt:182-183`（`ninfer_add_test(ninfer_kernel_route_test SOURCES test_kernel_route.cpp)`）、
  `:249-250`（`test_shard_plan.cpp`）、`:252-253`（`test_multidev_wiring.cpp`）。
  文件规模：`tests/test_kernel_route.cpp` 47512 B、`tests/test_shard_plan.cpp` 27676 B、
  `tests/test_multidev_wiring.cpp` 28575 B。
- ⇒ 这一条**不再是缺口**；改成「上一轮漏测、本册已测」。

### ?10 —— `rowsplit_generic` 的入口是谁：部分回答
- **已测**：入口是 env `NINFER_GENERIC_ROWDEC`，读取点 `src/ops/generic/rowsplit_generic.cu:377`。
- **仍未做**：从该 env 到具体算子的**调用链**我没有逐跳追完，所以不写进功能正文。

### ?11 —— `docs/` 的跟踪比：已由我独立复核
- **实测**：`git ls-files docs | wc -l` = **30**；`find docs -type f | wc -l` = **55**。
  ⇒ `CMakeLists.txt:509-513` 的注释「55 个在盘文件里 30 个被 git 跟踪」**成立**（我原来是引用，现在是测量）。

---

## B. 未做（逐条按名）

### P6 —— Windows 原生构建：不存在
- **实测**：`CMakeLists.txt` 的 `WIN32|MSVC|windows` 命中 = **0**；`src/CMakeLists.txt` = **0**；
  `apps/CMakeLists.txt` = **0**；`tools/archkit/*.cmake` 只有 CUDA 12.8 的 sm70 与 legacy 两个工具链，
  没有 MSVC 工具链文件。
- **能写的替代品**：Windows 侧有一个 PowerShell + PyInstaller 打出的**单文件 GUI 壳**
  （`tools/package/build_exe.ps1`、spec `tools/package/ninfer-gui.spec`），且它**不含 CUDA 引擎**
  ⇒ 文档里「Windows 可双击运行」与「Windows 原生构建引擎」必须分开写，否则会误导。

### P10 —— AMD / ROCm：只有构建配置行，能力未验证
- **实测**：`src/core/arch_caps.h:1733` 自述「AMD rungs (ROCm / HIP offload-arch targets):
  A BUILD CONFIGURATION, NOT SUPPORT」；行表 `kAmdLadder` 在 `:2059`（大小常量 `:2111`），
  六行 gfx906/908/90a/942/1100/1201（`:2060,2077,2084,2091,2097,2103`）。
- **为什么是「未做」**：`:1739-1741` 逐字说写这张表的机器只有一张 NVIDIA 卡、完全没有 ROCm 工具链，
  所以「能不能在这些目标上跑」**在这里未被回答，也无法在这里回答**。
- **另一条**：`:1757-1762` 自述把 `gfx906` 硬映射成 `906` 会让 `--sm 906` 变成「模拟 gfx906」的合法入口，
  ⇒ 所以键用的是 ROCm offload-arch 目标串（`:2261`）。

### P12 —— Intel：被具名，并被具名拒绝
- **实测（严格词界 `intel|sycl|oneapi|xpu|i915|ponte`，限定源码树）**：命中集中在
  `src/core/vendor_sim.h`（枚举 `:247`、名字表 `:255,265`、拒绝值 `:291`、拒绝实现 `:469-483`），
  加一处注释 `src/ops/kernel/gqa_attention_simt_ffma.cuh:107` 与测试断言
  `tests/test_vendor_sim.cpp:224-226`。**`src/core/arch_caps.h` 命中 0**（该头自己也在 `:178` 说同一件事）。
- **结论**：`VendorClass::Intel` 与「未知厂商」是**两句不同的话**：`NoTableForVendor`
  （`src/core/vendor_sim.h:291` 逐字「a vendor this tree NAMES and has NO TABLE for (Intel -- measured, 0 rows)」）。
- **文档怎么写**：写「Intel 是被具名的拒绝」，**不能**写支持。

### P13 —— NPU / 国产卡：只有注释
- **实测（严格词界 `\bnpu\b|\bascend\b|cambricon|mlu|biren|hygon|metax|mthreads|MUSA`，限定源码树）**：
  命中全是注释或分类枚举值：`src/ops/kernel/e8_lattice_codec.cuh:101`、
  `src/ops/kernel/gqa_attention_simt_ffma.cuh:107`、`src/core/vendor_sim.h:169` 与 `:838`、
  `tools/archkit/qpn_port/placement_planner.py:28`（是 `kind:` 注释里的一个枚举值）。
  `cambricon` / `MLU` / `biren` / `hygon` / `metax` / `mthreads` / `MUSA` 命中 **0**。
- **我自己的负对照（必须记）**：`\bascend\b` 在源码树里有 **2 处假阳性**
  （`src/targets/qwen3_6/impl/runtime/program_impl.h:12587` 与 `src/product/kv_bit_budget.h:1630`，
  都是英文单词「ascend」= 上行/递增，不是 Ascend 加速器）⇒ **按词界也能出假阳性，结论只取分类枚举那三类**。
- **结论引用**：`src/core/vendor_sim.h:169` 逐字「NO FUNCTIONAL SUPPORT CLAIM for AMD, Intel,
  any domestic accelerator or any NPU」。

### ?1 —— `--kv-tail-tokens` / `NINFER_KV_TAIL_TOKENS`：**悬空的名字**
- **实测**：`--kv-tail-tokens` 在 `apps/cli/options.cpp` 命中 **0**；全树引用它的只有 **3 行**：
  `src/kvcfg/kv_formats.h:8`（语义行）、`src/kvcfg/kv_formats.h:13`（**该文件自己标注它是悬空名**）、
  `tools/gui/kv_tiers.py:14`。`NINFER_KV_TAIL_TOKENS` 引用文件数 = **1**（就是 `kv_formats.h` 的那两行）。
- **源码自己的说法**（`src/kvcfg/kv_formats.h:11-19` 的 `MEASURED 2026-09-20` 段）：该名字在 parser 里 0 命中、
  env 全树 0 命中，而且**没有东西可供它定尺寸** —— 尾层只在与热层同值时被接受。
- **文档怎么写**：**不要**写这个旗标存在。`tail` 档当前只能写 `tail=<hot 同值>`（见 A17）。

### ?3 —— `docs/maintainer/tp2-yarn-1m.md`：**悬空的文档引用**
- **实测**：该文件**不存在**（`ls` 失败）；引用它的有 3 个文件共 6 行：
  `src/core/virtual_device.h:78`、`:81`（该行自述 `MEASURED 2026-09-20: ... DOES NOT EXIST IN THIS TREE`）、
  `src/core/n_dim_slice.h:18`、`src/core/tp_transport.h:12`、`:13`、`:14`（`:13-14` 自述那一行里的
  `tp2-yarn-1m` 是**donor 分支名**，不是文档路径）。
- **文档怎么写**：要写 TP2/YaRN 的 1M 接口时**没有本地契约可依**，只能读这三个头。

### ?6 / ?7 / ?8 —— 与 B 段的 P6 / P13 / P12 是同一件事的两面
（上一轮清单在 §8 与 §10 两处各记了一次。）本册统一按上文的 P6 / P12 / P13 写，不重复计两遍。

---

## C. 按名报：其余几条（查到的是「不是旗标」或「脚本不存在」）

### ?2 —— `--kv-auto-relayout`：CLI 里找不到，serve 里有
- **实测**：`apps/cli/options.h` 命中 **0**；`apps/cli/options.cpp` 命中 **1**（`:292`，用法文本里提到它并明说
  「the consumer ... is a serve-side feature (`--kv-auto-relayout` there), because this front end has no
  decision cycle」）；真正解析在 `src/serve/serve_options.cpp:535`（用法 `:121,177`）。
- **结论**：这是 **serve 独有**的旗标。文档必须按 `ninfer` / `ninfer-serve` **两套**旗标分开写
  （本目录就是这么做的：`cli-flags.md` 与 `serve-flags.md`）。

### ?4 —— `--some-flag` 不是旗标
- **实测**：全树只出现在 `apps/cli/options.cpp:326` 的**注释**里，讲的是「`argv[1]` 以 `--` 开头就不当模型路径」
  （`:325-330`）。把它写进文档就是错的。

### ?5 —— `iso3` 一个字符串两个域
- **实测**：① `--kv-dtype iso3` = `KvCacheStorage::Iso3Group16` 的**废弃别名**
  （`apps/cli/options.cpp:104`，与规范拼写 `iso4e` 同一分支）；
  ② `--kv-v-codec iso3` = `KvVCodec::Iso3` 的**废弃拼写**，与 `iso4e` 同一状态并打告警
  （`:551-556` 打 `[kv-v-codec] --kv-v-codec iso3 is deprecated`，`:561` 的拒绝语里
  「expected iso4e|e2m1; iso3 is the deprecated ...」）。
- **第三个易错点**：`KvVCodec::Iso3` 这个**枚举名**对应的规范**拼写**是 `iso4e`
  （`apps/cli/options.cpp:545-550` 的注释自述「`iso3` 是同一个东西的废弃拼写」）。
  ⇒ 文档要分别写，并指出枚举名与拼写不同名。

### ?12 —— `s12_tools.sh` 这个文件名不存在
- **实测**：上一轮的记录目录 `dl/docinv/` 里没有 `s12_tools.sh` 这个文件名；那批记录里的 `s12_tools.txt` 是由 `s12_spec2.sh` 生成的。
  ⇒ 这是**上一轮记录**自身的问题，不是仓库里的事实；它说明「引用一个脚本名」也要按名核过。

---

## D. 我在核验中发现的漂移与缺口（上一轮清单的账）

| # | 上一轮清单说 | 实测 | 说明 |
|---|---|---|---|
| D1 | 基准 `HEAD = 38998db`，`git status` **788** 条（`A` 204 + `??` 166） | `HEAD = 3944a53eda1aac439a566a1cf46ea741f0415fdc`，`git status` **794** 条（`A` **236** + `??` **168**） | 树在这两次测量之间动了，**所有行号必须以本册的实测为准** |
| D2 | `apps/cli/options.cpp` **1001** 行；`--kv-dtype` 的 case 在 `:363` | **1016** 行；case 在 **`:378`** | 行号漂移的代表例 |
| D3 | `--yarn`、`--no-lm-head-draft` **usage 未列** | 现在**都列了**：`--yarn` 在用法 `:246-253`，`--no-lm-head-draft` 在 `:255` | 上一轮的 §2.10「对照差集」在这棵树里**已不成立** |
| D4 | 旗标表里没有 `--max-new` 与 `--vision` | 两者都在 parser（`:362`、`:610`）与用法文本（`:149`、`:266`）里 | 补进 `cli-flags.md` 第 10 节 |
| D5 | 目标族 **8** 个 | **9** 个：多了 `gemma4_31b`（`src/CMakeLists.txt:613`） | 补进 `README_FEATURES.md` 第 2 节 |
| D6 | `src/product/kv_*.h` **18** 个主题头（清单点名 8 个） | **19** 个：多出 `kv_summary_format.h`（165 行、mtime 2026-09-20 16:36）与 `kv_component_switch.h`（344 行） | 上一轮未列 |
| D7 | `per-variant` 的 MTP 域在 `src/targets/*/impl/config.h` | 实测在 **`src/targets/*/impl/variant.h`**（`:37/33/39/47`） | 字段名 `maximum_mtp_draft_tokens` |
| D8 | `tests/CMakeLists.txt` 的 `NEEDS_SOURCE_DIR` 出现 **8** 次 | **23** 次 | 以本册为准 |
| D9 | `tests/` 下源/测试文件 **254** 个 | **261** 个（`178 cpp + 44 py + 18 cu + 14 h + 2 pre_land + 2 jinja + 1 txt + 1 md + 1 json`） | 计法：`find tests -type f -not -path '*__pycache__*'` |
| D10 | 三个孤儿测试「是否注册进 CTest 我没测」 | **都已注册**（`tests/CMakeLists.txt:182,249,252`） | 见 A 段 ?9 |
| D11 | `--kv-tail-tokens` 的引用共 2 处 | 实测 **3 行**（`src/kvcfg/kv_formats.h:8,13`、`tools/gui/kv_tiers.py:14`） | 数量不同，结论一致（悬空名） |
| D12 | `_WIN32` 分支在 **15** 个文件 | 实测 **13** 个活文件 + 1 个前像（清单里的 3 个文件已不在命中里） | 见 A 段 P7 |
| D13 | `import_model.py` **1097** 行 | **1141** 行 | 其余逐个工具的实测行数见 `importers.md` |

**我自己也出了一次同型假阳性，记在这里**：用 `grep -c 'arg == "--' apps/cli/options.cpp` 得 **67**，
但其中 `:697` 是三元表达式 `arg == "--stop" ? ... : ...`，**不是 case 分支** ⇒ 真实分支数是 **66**
（与「68 个入口」的算法一致：66 分支 → 其中一条合并两名 = 67 名 → 加 `-h/--help` = 68）。
这与上一轮 `npu` ∈ `i-n-p-u-t`、`CANN` ∈ `SCANNER` 是同一个陷阱。

---

## E. 2026-09-22 复核发现（`accept34` 线，逐条带实测与复现）

本节只写**当天实测到、上一轮没记**的条目。凡涉及二进制处都给出 sha256 前 16。

### E1 —— `tests/CMakeLists.txt` 的 `sh/_lock.sh` 与 `sh4/` **在树里不存在**
- **是什么**：`:838`、`:840`（以及 `:863`）用这两条路径解释「这个可执行程序为什么不注册成 ctest 测试」：
  `:838` 写「every GPU run in this project goes through the fair lock in `sh/_lock.sh`」，
  `:840` 写「Its invocation path is therefore `sh4/_mg4_run*.sh`」。
- **实测**：`ls /home/user/ninfer-fusion/sh/_lock.sh` → `No such file or directory`；
  `ls /home/user/ninfer-fusion/sh4` → `No such file or directory`。**两处都不存在。**
- **后果（按名）**：`ninfer_tp2_sum_stress` 是 `add_executable`（`:842`）而**不是** ctest 测试，
  它被指名的唯一调用路径指向**不在树里**的脚本目录 ⇒ 这个可执行程序**今天没有可跑的调用路径**。
  注释描述的是**记录目录那一族**（`dl/_orch/`、`dl/mg4/`）里的东西，不是仓库路径。
- **本线没有改它**（既有问题，按纪律只按名记）。
- **复现**：`grep -n 'sh/_lock.sh\|sh4/' tests/CMakeLists.txt`
  （`tests/CMakeLists.txt` sha256 前 16 `29f911a7dff840bc`）。

### E2 —— `build/apps/ninfer-serve` **早于它自己的 parser 源码**（服务端结论会打到旧 parser 上）
- **实测**：`build/apps/ninfer-serve` mtime **2026-09-19 11:21**、sha256 前 16 **`91327a8ae55f70e0`**；
  `src/serve/serve_options.cpp` mtime **2026-09-20 14:25**、`src/product/kv_options.h` **2026-09-20 09:11**、
  `src/serve/generation_service.cpp` **2026-09-21 11:11** ⇒ **二进制比这些源都旧**。
- **一个被它坑到的实测例**：`--kv-residual-layers 2-5` 在**旧 serve 二进制**上被拒
  （`invalid kv-residual-layers: 2-5`），在**当天的 CLI 二进制**（`713cfcd6a1ad1150`）上被接受并继续跑。
  两者用的是同一个 `product::parse_kv_residual_layers_spec`（`src/product/kv_options.h:205`，
  按 `-` 解析区间）⇒ 这**不是**「两个前端不一致」，**是 serve 二进制陈旧**。
- **纪律含义**：服务端实测结论必须**当场报 bin sha16**，且**不得**用它推断当前源码的行为。
- **复现**：`stat -c '%y %s' build/apps/ninfer-serve`；`sha256sum build/apps/ninfer-serve`。

### E3 —— 有一个注册测试**没有任何命令**
- **名**：`ninfer_shard_contract_multidev_pending_test`（`ctest` 序号 #179）。
- **实测**：`ctest --show-only=json-v1` 里该条目**没有 `command` 字段**，只有 `WILL_FAIL: true`、
  一条 `PASS_REGULAR_EXPRESSION` 与 `WORKING_DIRECTORY` ⇒ ctest 无从执行它。
  同日 `ctest -N` 总数 **179**，其中 **178** 个能解析到可执行件、**1** 个解析不到。
- **顺带纠正一个数**：同一次 JSON 里带 `SKIP_RETURN_CODE` 属性的测试是 **85** 个
  （「带该属性」≠「会跳过」；是否跳过由运行期返回 77 决定）。
- **复现**：`cd build && ctest --show-only=json-v1`，取 `tests[].command` 为空的条目。

### E4 —— 服务端 `--help` 的用法行漏列 18 个已注册旗标
- **出处与后果**：见 `serve-flags.md` 第 4 节（本节不重复那张表）。一句话：照 `--help` 写文档的人
  会以为冷窗族与权重卸载族在服务端不可达，实测两族都能过 parser。
