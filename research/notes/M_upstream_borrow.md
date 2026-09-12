# ninfer 上游更新与可借鉴清单（2026-09-11 09:0X 刷新版）

> **本版相对上一版（2026-09-10 16:22）的变化**
> 1. 上游克隆已 `fetch` 到 `d492968`（09-11 01:08），**新增 2 条提交**；克隆已 `--unshallow` 到全史 1068 commits。
> 2. 新增 **PR 维度**（上一版只有 master 维度）：`refs/pull/*/head` 全部拉到 `refs/remotes/pr/*`，
>    发现 **18 个 PR 分支的内容不在 master 上**，其中 4 条是我们没数过的真修复。
> 3. **fork 点已用内容对账钉死 = 2026-08-31 / 09-01**（依据见 §0.2）。
> 4. 每项都补了 **判据**（修了什么 / 怎么验证生效 / 若无效如何确认不是它）。
> 5. 本轮产出 **9 个 `patch -p1 --dry-run --fuzz=0` rc=0 的补丁**，在 `_collab/UP1_*.diff`。
> 详细报告：`_collab/build/UP1_upstream_refresh.md`；dry-run 原文：`_collab/build/UP1_dryrun_evidence.txt`。

---

## 0. 怎么拿到上游（已验证可用的路径）与 baseline

### 0.1 取数路径（不变）
- 上游 = **`github.com/Neroued/ninfer`**，默认分支 **`master`**（`dev` 与 master **同点**，无独立分支）。
- **WSL 连不上代理**（代理只绑 Windows loopback）⇒ 一切 git 操作走 **Windows 侧**：
  `C:\Users\User\Documents\ziqinzhang\ninfer-upstream`（git 2.55.0.windows.2）。
- 刷新命令：`git fetch --all --prune --tags`；补全史：`git fetch --unshallow`。
- PR 离线取：`git fetch origin "+refs/pull/*/head:refs/remotes/pr/*"`（**匿名即可，无需 token**）。
- **WSL 侧只读可用**：`git --git-dir=/mnt/c/.../ninfer-upstream/.git <子命令>` 能直接读这个 Windows 仓库
  （本轮所有对账都这么做；权威树在 `/home/user/ninfer-fusion`，两边路径都读得到）。
- **上游没有任何 tag**（`git tag` 为空）⇒ 钉版本只能用 commit 号 / 日期。

### 0.2 fork 基线（本轮新钉，带依据）
方法：对上游 1411 个路径逐一比对"我们树字节 == 上游某次改动的 post 态"，取**最新**那一条。

| 类别 | 数量 |
|---|---|
| 与 `origin/master` **逐字节相同** | **744** |
| 两边都不同 | 427 |
| 上游有、我们没有 | 240 |
| 我们新增、上游没有 | 320 |

**最新"能对上"的日期 = 2026-09-01（2 个文件）/ 2026-08-31（26 个文件）；09-02 及之后 = 0 个。**

- `src/runtime/engine/materialization_planner.h` == `3d9fda22`(**2026-08-31**) 的 post 态
- `src/runtime/contract/types.h` == `3d9fda22`(**2026-08-31**) 的 post 态
- `src/targets/qwen3_6/impl/runtime/pressure_planner.h` == `da49c0d6`(**2026-09-01**) 的 post 态
- `tests/test_resource_manager.cpp` / `shared_capture_planner.h` / `api_impl.h` / `runtime.h` == 08-31

**⇒ fork 基线 ≈ 2026-08-31 / 09-01。** fork 点之后第一个"系统性没跟上"的上游提交 =
**`138d76ae`（2026-09-01 `refactor(runtime): clarify resource scheduling ownership`，21 文件处于我们的 pre 态）**。

**缺口规模**：2026-09-01..09-11 上游 **117 条**提交，其中 **37 条我们完全反映**、**80 条至少有一个文件没反映**。
（80 条里大头是 09-04..09-06 那波 **dflash2 op / KV cache 大建设**——我们是自己另起一套实现，**不是欠账**。）

---

## 1. 本轮新增提交（2 条，逐条判定）

### 1.1 `9f0575bb` (2026-09-10 21:51) `fix(build): include bf16 definitions in q4 topk kernel`
- 1 文件 +1：`src/ops/linear_topk/q4_m64.cu` 加 `#include <cuda_bf16.h>`。
- **判定：不适用。** 我们树**没有** `src/ops/linear_topk/` 整个 op 家族（逐个 `ls` 确认 ABSENT），
  所以这个"补 include"的构建修复**无处可落**。
- 同一条也在 **PR#219** 上（`fix(build): include cuda_bf16.h in linear_topk q4_m64`）。
- **判据**：若将来我们引入 fused linear top-k（`5499799d feat(ops): add fused linear top-k`），
  这一行必须一并带入，否则该 TU 编不过（bf16 类型缺定义）。

### 1.2 `d4929686` (2026-09-11 01:08) `perf(runtime): improve materialization search and adapt planning budgets`
- **18 文件 +1451 −634**。核心是 **planner/search 的 API 大重构**：
  - `materialization_planner.h` +435−294、`pressure_planner.h` +242−190、**新文件 `materialization_budget.h` +130**
  - `include/ninfer/types.h` +45−3：新公开枚举 `MaterializationSearchPhase`（none/setup/construction/
    assessment/expansion/refinement）+ `MaterializationDiagnostics` 新增 10 个字段；并把
    `MaterializationStopReason::ValueOfNextExpansion` **改名**成 `InsufficientExpectedGain` 且新增 `WorkBudget`
  - `src/targets/qwen3_6/export/.../runtime.h` +39−3：**`guided_closure_target()` → 可续跑游标 API**
    （`begin_construction` / `next_construction_option` / `choose_construction` / `construction_target`）
    + 新 RAII `PressureConstructionCursor`
  - `src/runtime/contract/types.h` +25：`PressureOwnerRecoveryGuidance` / `PressureConstructionOptionId` /
    `PressureConstructionStep`；`PressureTargetGuidance` 加 `source_mode`/`checkpoint_changes`/`recovery_estimates`
  - `src/serve/request_log.cpp` +13：把 10 个搜索诊断字段写进 JSON
  - 其余：`program.h` +50−6、`api_impl.h` +30−7、`engine_core.h` +16−4、`resource_manager.h` +5−4、
    `tests/*` +387 / 新 `tests/test_materialization_budget.cpp` +80、两份 docs
- **判定：不属于我们已落的任何一类（KV 覆盖 / SiLU / dflash2 K），也不建议落。**
- **与①（接受率）**：**弱相关**。它调 admission/materialization 搜索，不进 verify/draft 的数值路径；
  但新增的 `search_stop_phase` 等诊断对"为什么这一轮被降级/回退"有观测价值。
- **与②（去 Qwen 特化 / 描述驱动）**：**方向相关，实现不相关**。把"一次性闭包目标"换成"可续跑游标 + 预算"
  = 把调度决策从写死流程变成**可枚举/可描述的选择空间**；但它落在 `src/targets/qwen3_6/` 内部，
  **没有**引入任何"模型描述 → 引擎生成"的机制。
- **影响面 / 重编 / 风险**：18 文件、含 4 个我们改过的核心头（`engine_core.h`/`materialization_planner.h`/
  `pressure_planner.h`/`program.h`）⇒ **重编面很大**；数值风险**低**（纯调度），行为风险**中—高**
  （会改变前缀复用与降级决策 ⇒ 影响 TTFT/吞吐）。
- **可行性**：**0 个文件**处于我们的 pre 态（16 `OTHER` + 2 `ABSENT`），且前置 `138d76ae`（09-01，21 文件）
  也没落 ⇒ **不能干净落，cherry-pick 会连环冲突**。建议作为独立专题"planner 现代化"处理，
  先落 `138d76ae` 中间态，再与我们 09-10 的 planner 改动对账。
- **判据**：它改的是**调度**不是数学 ⇒ 落地前后**同 prompt 输出必须逐 token 相同**；
  若不同，说明改动越界了。诊断字段的效果只能靠 `--request-log-jsonl` 里新键的存在性验证。

---

## 2. 可借鉴清单（判据版）

图例：**已落** = 已在我们权威树；**待落** = 本轮已出 rc=0 补丁；**需实跑** = 必须 GPU/编译窗口；
**不建议** = 判断为不抄；**不可落** = 想抄但当前树落不了。

### 2.1 已落（复核结论：无需再动）

| 项 | 状态 | 判据 / 怎么确认它真在生效 | 若无效如何确认不是它 |
|---|---|---|---|
| **`03177b9` KV 覆盖下界语义**（上轮 P0-A） | **已落**（本轮实测） | `grep -n ensure_mapped_to_tokens src/targets/qwen3_6/impl/runtime/logical_kv_store.h` → **命中 :1497**；`grep -c materialize_to_tokens <同文件>` → **0**；`grep -c 'materialization exceeds active entitlement' <同文件>` → **0**；`grep -c ensure_sequence_kv_mapped src/targets/qwen3_6/impl/runtime/program.h` → **1**（分号结尾的函数体是 `...logical_kv_store.h:1497` 的 `void ensure_mapped_to_tokens(...)`）。判据：投机终止结算时**不再抛** `KV materialization exceeds active entitlement` | 若仍抛该串 ⇒ 树上还有老版本（我们的是**语义适配版**，文案已改成 `KV coverage exceeds active entitlement`）；注意我们 `M_landed_set.md` 记的 md5 `69f2281c30` 是**当时**快照，md5 变了**不代表**没落 |
| **PR#194 近似 SiLU 错零**（上轮 P0-B） | **已落**（本轮实测） | ⚠ **我们的落地位置与上游不同**：上游改的是 `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh` 里**文件内**的 `swiglu_silu`；我们树**没有** `__fdividef`（全树 0 处）、**没有** fast-math，所以**同名缺陷以另一种机制存在**（精确 `expf` 在 x ≤ −88.72284 溢出 → `x/inf` = −0）。我们在**共享定义**上修：`grep -n 'expf(-fabsf' src/ops/common/math.cuh` → **命中 :20（silu 与 :28 sigmoid）**，两处都写成"把指数折到不会溢出的那一侧"，除数恒在 (1,2]。判据：`x < −88.72284` 的输入不再返回 −0（该处真 SiLU = −2.6073e-37 = bf16 最小正规的 **22.18 倍**）。**改 1 行修 66 个调用点** | 若 `math.cuh:19` 仍是 `return x / (1.0f + expf(-x));` ⇒ 没落。⚠ `git cherry origin/master pr/194` 仍是 `+` ⇒ **上游自己也没并进 master**，所以"上游 master 有没有"**不能**当判据 |
| **`385b30c` dflash2 可配 K** | **部分已落（S52 切片）** | `--spec dflash2 --draft-tokens 3` 能起、日志 `speculative=dflash2`；K∈{1,3,7}（W 必须 2 的幂） | 若报 `object handle does not name a materialized tensor` ⇒ 选错后端（`dflash` vs `dflash2`）；若报 flag 校验 ⇒ 门没放行（`speculative_options.h:54` / `layouts_impl.h:869-870`）。上游剩余部分（47 文件全量重构）**我们不要** |

### 2.2 待落（本轮已出补丁，`patch -p1 --dry-run --fuzz=0` **rc=0**）

| # | 补丁 | 来源 | 规模 | 判据（怎么验证生效 / 若无效如何确认不是它） |
|---|---|---|---|---|
| 1 | `UP1_df2_lane_tests.diff` | `898603fd`+`5a6a54d9`+`d1d28c67`+`e81b0924`+`5c6dad35` | 5 测试文件 | **验证**：编 `tests/ops/` 后跑 `test_rope`/`test_argmax`/`test_position`/`test_prepare_masked_block`/`test_sigmoid_mul`，期望**全绿**。`test_rope` 新增 12 个 "27b dflash2 lanes" fixture（W∈{2,7,16}×B∈{1,8}×axes∈{1,3}，**每 lane 不同绝对起点** + T128 Graph 重放 + FP64 oracle + 非旋转维逐位保持）。**若无效**：先确认跑的是**新编出来的**二进制（时间戳晚于补丁）；若全绿但①仍在 ⇒ 这组用例**不能**证伪 "dflash2 漏 rope_delta" 假设（它们在 op 层，不覆盖 `dflash2_impl.h` 的编排），嫌疑回到 `dflash2_impl.h:410-411` |
| 2 | `UP1_ignore_eos.diff` | **PR#197**（master 没有） | 3 文件 +6 | **验证**：`POST /v1/chat/completions` 带 `{"ignore_eos": true, "max_tokens": 512}` ⇒ `finish_reason` 从 `stop` 变 `length`，`spec_accept_rate` 拿到完整 512-token 统计（对照不传时的 `gen=2`）。**若无效**：先看 `options.stop.include_model_defaults` 是否为 false；**caller 显式传的 `stop` 仍然生效**（设计如此，不是 bug）；若仍停止 ⇒ 停止来自 `max_new_tokens`/output limit，与本补丁无关 |
| 3 | `UP1_kv_page_fence_tests.diff` | **PR#211** | 3 文件 +162 | **验证**：`KVAddressSpace::activate()` 多一个 `cudaStream_t stream=nullptr` 形参并在 `bind_sequence_kv` 传 `device.stream`；`tests/test_kv_cache.cpp` +158 的 `release_page` fence/竞态用例绿。**若无效**：该补丁只改**流归属**与**测试**，不改 KV 语义 ⇒ 输出变了就不是它；测试红是**收益**（暴露我们 KV 实现的真竞态）不是回归 |
| 4 | `UP1_sparse_moe_t1_quad.diff` | **PR#199** | 1 文件 +59−27 | **验证**：T=1 decode 路径保持一个 quad 在飞。期望**输出逐位不变 + T=1 耗时下降**。**若无效**：确认走的是 T=1 分支；输出若变 ⇒ **立刻回退** |
| 5 | `UP1_moe_s2_perf.diff` | `487f8977`+`ce954918`+`7f14d963`(部分) | 7 文件 | **验证**：①现有 `test_sparse_moe.cpp` 全绿且同输入两次输出逐位相同；②`grep` 新重载调用点 = 0（我们**故意没打**调用方）。⚠ **落地后性能不会变**——它是"**使能态**"，真正提速的那半在 `UP1_moe_hint_callers.diff`（rc=1，6 个 CRLF 调用方）。**若有人报"没提速"**：那不是补丁无效，是这半没接 |
| 6 | `UP1_moe_s2_perf_v3.diff` | `487f8977`+`ce954918` | 3 文件 | 同上，更小切片（不含 `sparse_moe.h` API 新增），备用 |
| 7 | `UP1_nfc_ascii.diff` | `641ef3e7` | 1 文件 +12 | **验证**：`normalize_nfc()` 先扫纯 ASCII 直接返回原串 ⇒ 同输入**逐字节相同**，纯 ASCII 大 prompt 前端耗时下降。**若无效**：ASCII 判定是 `>=0x80`，**只会保守**（漏跳）不会误跳 ⇒ 输出变化必是别的改动 |
| 8 | `UP1_bpe_flat_table.diff` | `b158afe2` | 2 文件 | **验证**：BPE merge 规则改扁平开放寻址表 ⇒ 分词用例全绿 + `--print-token-ids` **逐字节不变** + 分词吞吐上升。**硬判据**：token id 变了就**回退** |
| 9 | `UP1_gdn_conv_column.diff` | **PR#220**（剔掉 CRLF 的 `tests/CMakeLists.txt`） | 2 文件 +354−1 | **验证**：融合 GDN epilogue 的**被代表卷积列**修复 + 新测试 `test_gdn_input_proj_conv_restart`（conv restart 语义）绿。⚠ 落地后需**手工**在 `tests/CMakeLists.txt` 登记新测试（该文件 CRLF 被剔）。**若无效**："测试编不出来" ⇒ 是**登记缺失**（`grep test_gdn_input_proj_conv_restart tests/CMakeLists.txt` 为空即证），不是修复失败 |
| 10 | `UP1_mtp_pack_contract.diff` | **PR#226**（剔掉 CRLF 的 `mtp_impl.h`） | 8 文件 | **验证**：MTP pack 语义契约 + kernel/launcher/wrapper 对齐 + 新 `tests/ops/test_mtp_pack.cpp`（+452）绿；现有 MTP 路径输出**逐位不变**。**已知缺口**："hold the weight shape"的运行期修复在 `mtp_impl.h`（CRLF，未含）⇒ 若 MTP 仍出形状类异常，**不是**本补丁失败 |

> **回退**（所有补丁都是纯文本、非 CRLF 文件 ⇒ `-R` 可靠）：
> `cd /home/user/ninfer-fusion && patch -p1 -R -i /mnt/c/.../_collab/UP1_<名>.diff`
> 补丁 9/10 含**新文件**，`-R` 会留成空文件 ⇒ 需手工 `rm tests/ops/test_gdn_input_proj_conv_restart.cpp` /
> `rm tests/ops/test_mtp_pack.cpp`。若 `-R` 报 FAILED，用打补丁前的 `tar` 备份覆盖。

### 2.3 需实跑 / 排后（本轮判定）

| 项 | 来源 | 为什么排后 / 需要什么 |
|---|---|---|
| `e51b585c fix(gdn): respect cooperative launch capacity` | master **+ PR** | 真修复（协作启动容量），20 文件。**实测 rc=1**：13 个目标文件是 **CRLF** + `layouts_impl.h` 真冲突 ⇒ 需字节级生成器（`E9_s52_mkpatch.py` 那种 `read_bytes`+`diff -u`）。判据：`src/core/device.cu` 的协作启动检查 + `tests/test_device.cpp` 绿 |
| **PR#222** `fix(ops): drop the invented token ceiling from the text rmsnorm_rope form` | **PR only** | 8 文件 +481。"凭空发明的 token 上限"正是我们 `prefill_i8` 一族踩过的坑。**实测 rc=1，4/4 hunk FAILED 且不是行尾原因** ⇒ 需人工适配 `rmsnorm_rope` op + `text_context_impl.h` |
| **PR#195** context-cost 预设回退 | **PR only** | 3 文件 +66，与我们 `Package::resolve_weights`/`resolved_auto_speculative` 同族。**实测 rc=1**：`include/ninfer/types.h` 是 **CRLF** |
| `21a0e85f feat(kv-cache): use fp16 V storage and PV compute` | master | 真 KV 改动（12 at_pre + 22 OTHER），**改数值** ⇒ 需要质量矩阵；排后 |
| `ee9d5192` nvfp4 W4A4 TMA 光栅化 | master（=PR#204，已在 master） | 2 个文件在我们树是 `OTHER` ⇒ 需适配，非干净可落 |
| `b88c0f6 fix(core): complete host uploads before returning` | master | `src/core/arena.h` 在 pre 态（可落），但 `arena.cu` 是 `OTHER` ⇒ **半干净**；小项 |
| `f0eb3ac7 fix(serve): upgrade cpp-httplib to 0.54.1` | master | `third_party/cpp-httplib/httplib.h` 在 pre 态（可落）；但第三方头升级会**触发大面积重编**，性价比低 |
| **PR#221 / PR#226 的运行期半** | **PR only** | MTP graph profile 拓扑分类 / weight shape 持有。MTP 是我们最常用后端 ⇒ 值得单独立项（目标文件 CRLF，需字节级补丁） |

### 2.4 不建议 / 不可落

| 项 | 结论 |
|---|---|
| `d492968` planner 大重构 | **不可落**（0 文件在 pre 态 + 前置 `138d76ae` 未落）。见 §1.2 |
| `385b30c` 的剩余部分（proposal/verify extent 分离、Vision/Host 复用、schema v14） | **不建议**（与我们的帧布局/recipe/envelope 架构不合） |
| `a2761ec1 refactor(kv-cache): centralize format contracts` | **不建议**。名字像 KV，**实际是 perplexity app 重构**（删 `apps/perplexity/digest.*`、移 corpus 预处理）；9 个 at_pre 文件全在 `apps/perplexity` 与语料清单 |
| 上游的转换器 / artifact 契约文档（`4df5e0b4`、`5bde5ed5`、`917d4c89`） | **不建议抄**（会退步）：我们 `tools/convert/`+`tools/archkit/` 已更强（tied-embedding 解析 + `check_source_map.py` 61 检查 0 失败 + 来源映射） |
| PR#224 tool_choice / schema 21 | 不是我们路径 |
| **PR#183 `--chat-template FILE`** | **不属于"不建议"，属于"值得单独立项但不要照抄"**：它是**唯一**方向正确指向②的去特化资产，但 13 文件 rc=1（5 CRLF + 3 真冲突）。建议自己重做一遍（模板家族常量 → 外部可装载描述 + `--chat-template` 开关） |

---

## 3. 与两个当前问题（① 接受率 / ② 去 Qwen 特化）的映射

### 3.1 ① 投机解码接受率崩（dflash2/dspark 位置 ≥1 恒 0）
本轮上游能给的**全部**是**测量工具与契约测试**，没有"上游已修好同一条 bug"这回事：

1. **`UP1_df2_lane_tests.diff`（#1）** —— 直接钉住 partial RoPE 逐 lane 起点 / per-lane position 偏移 /
   argmax 按列 masking / ragged masked block / sigmoid 128 列。是"dflash2 verify 漏 rope_delta"假设的**可证伪门**。
2. **`UP1_ignore_eos.diff`（#2）** —— 让"跑满固定 token 预算"成为可控口径，**排除 `stop_token` 干扰**的测量前提
   （今天 `gen=2` 就是被停止符截断）。
3. **PR#211（#3）** —— KV `release_page` fence/竞态用例 + `activate(..., stream)`：与 S50 同一区域
   （KV 覆盖/终止结算），补的是"页释放不归零、陈旧数据跨 release+重分配存活"这一结构前提。
4. **PR#220（#9）** —— 融合 GDN epilogue 卷积列修复；GDN 是 dspark/dflash2 的既成路径。
5. **PR#222（排后）** —— `rmsnorm_rope` 的"凭空 token 上限"。
6. **`e51b585c`（需实跑）** —— GDN 协作启动容量。

### 3.2 ② 引擎去 Qwen 特化 + 描述驱动生成
**上游没有这条线。** 三件最接近的东西：`d492968`（调度决策可枚举化，不可落）、
**PR#183 `--chat-template FILE`（模板外挂，方向对但 rc=1）**、
`4df5e0b4`+`5bde5ed5`（转换器+契约文档，**我们已更强**）。
⇒ **这一项要靠我们自己推进**；上游能提供的只有 PR#183 的设计参考。

---

## 4. "取不到"的部分

**本轮结论：没有取不到的东西。** 上游公开仓库的一切都能匿名拿到：

| 想要 | 能否拿到 | 需要什么 |
|---|---|---|
| master 全史（1068 commits） | ✅ 已拿到 | Windows 侧匿名 `git fetch`（**WSL 直连不行**） |
| 109 个 `refs/pull/*/head` | ✅ **已全部拉到** `refs/remotes/pr/*` | `git fetch origin "+refs/pull/*/head:refs/remotes/pr/*"`，**无需 token** |
| `refs/heads/dev` | ✅ 与 master 同点 | — |
| 上游 tag | ❌ **上游没有任何 tag** | 只能按 commit 号 / 日期钉版本 |
| **"DFlash2 后端支持"分支/PR** | ❌ **上游不存在** | 实测 `git ls-remote --heads` 只有 `dev`/`master` 同点；commit subject 里**没有** `backend` 关键字。上一轮猜测的这条**不存在**；DFlash2 的 op 建设（60+ 提交）**已全在 master 上** |
| 上游的 dflash2 草稿**权重 / 训练脚本** | ❌ 不在上游仓库 | 那是**我们自己的资产**（`data/dflash2_ckpts/`、`train_dflash2.py`） |
| 需要 token 的场景 | — | 只有当：仓库转私有 / 需要 push（我们不需要）/ **要读 PR 的讨论与 review 正文**（`refs/pull/*/head` 只给代码）。本轮结论**不需要**评论正文；若日后想查某 PR 的动机（例如 PR#194 为何至今没并入 master、PR#211 为何带 review 标记），需 `gh pr view <N> --repo Neroued/ninfer` + GitHub token |

---

## 5. 建议的执行顺序（本版）

1. **先落 #1 + #2**（都是 rc=0、都直接服务①）：`UP1_df2_lane_tests.diff` 建回归网，`UP1_ignore_eos.diff` 修测量口径。
   两者不冲突、不动同一文件，可在同一编译窗口一起落。
2. **#3 PR#211** 同窗（KV 页 fence 测试 + `activate(stream)`），与 S50 区域相邻，便于一起看。
3. **#7 + #8**（NFC / BPE，各 1-2 文件、零 API 耦合）随便哪个窗口顺手落。
4. **#9 / #10**（GDN 卷积列 / MTP 契约+测试）：落 #10 前确认要把 `mtp_impl.h` 的运行期半**单独立项**（CRLF）。
5. **#4 / #5**（sparse_moe）：等 FlashNext 权重可用时再评估（#5 是"使能态"，提速要另接调用方）。
6. **不建议做**：`d492968`、`385b30c` 剩余、转换器类、`a2761ec1`。
7. **需单独立项**：PR#183 模板外挂（②去特化）、planner 现代化（先 `138d76ae`）、
   `e51b585c`/PR#222/PR#195/PR#226-运行期半（都要**字节级（CRLF 安全）补丁生成器**）。
