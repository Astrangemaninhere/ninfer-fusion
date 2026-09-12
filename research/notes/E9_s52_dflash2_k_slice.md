# E9 / S52 · upstream `385b30c` 里可借的「dflash2 可变草稿宽度 K」最小切片

- 作者: E9（分析 agent）· 2026-09-10 · **纯 CPU**：只做文本读 + `git show` + `diff` + `patch --dry-run`，
  **未写任何 `src/` 文件**，未开 nvcc/ptxas，未用 GPU。
- 基准树（canonical）: `/home/user/ninfer-fusion`（**非 git 仓库**，故所有引用按行号）。
- 上游: `/mnt/c/Users/User/Documents/ziqinzhang/ninfer-upstream` @ `385b30ce1757bafe5a82680e9b5aeb940b14eec1`。
- 镜像 `ninfer-fusion-repo` **已漂移**（`layouts_impl.h` / `request_plan_impl.h` / `include/ninfer/types.h`
  md5 与基准树不同），所以本文件所有行号都取自基准树，不取镜像。
- 交付: 本 md + `_collab/E9_s52_dflash2_k_slice.diff`（6 文件 / +44 −32）+ 生成器 `_collab/E9_s52_mkpatch.py`。

## 0. 一句话结论（含一条必须先知道的更正）

上游 `385b30c` 对 dflash2 宽度做的是**大重构**（47 文件 / +1114 −982：proposal/verify extent 分离、
partial terminal commit、Vision/Host 复用、`DFlashConfig` 与 dflash2 权重合一、`--draft-tokens` 产品打通、
schema v14）；但**我们自己的树里，dflash2 本来就只剩 5 个常量 + 2 处 op 域门是固定宽度**——
帧布局（`columns = draft_window+1`）、workspace recipe（`drafts`/`verify`）、graph profile、envelope、
`extent` 计算都已经按运行期 K 泛化了。所以能吃下的不是上游的 diff，而是一个
**13 处 / 6 文件的收口切片**，见 §3 与 `E9_s52_dflash2_k_slice.diff`。

> **更正（重要）**：任务给的「dflash2 decode tok/s d1 65.4 / d3 80.5 / d7 70.0」**是 `--spec dflash`
> （DSpark v1 路）的数字，不是 dflash2**。见 `_collab/M_spec_sweep.md:9-11`：
> `dflash_d1 65.4 / dflash_d3 80.5 / dflash_d7 70.0`；而 `dflash2_d1`、`dflash2_d3` 两行是
> **`SERVE_FAILED`**，报错正是本次要拆的门 `--spec dflash2 uses th...`（`M_spec_sweep.md:12-13`，
> 文本来自 `src/product/speculative_options.h:55`）；唯一跑成的 `dflash2_d7` 是 `gen=2 / 31.7tok/s`
> （退化，不可用）。**"宽度对 dflash2 是真杠杆"目前是 待证实**——它是从 DSpark 路外推的，而
> upstream 自己写明非因果 attention 使 K 变化后的 hidden/candidates/q **不是**更宽 block 的前缀
> （`docs/maintainer/qwen3.8-27b-dflash2.md:386-390`），所以两个后端之间不可直接搬结论。

## 1. upstream `385b30c` 的事实（我读到的）

`git show --stat 385b30c`：47 文件、+1114/−982，标题
`feat(engine): integrate dflash2 with configurable draft counts`，日期 2026-09-06。
`src/product/speculative_options.h` 的那一处是**全 commit 里唯一的产品层改动**，而且只有 2 行：

```diff
     case SpeculativeBackend::DFlash2:
-        if (options.draft_tokens != 7) {
-            throw std::invalid_argument("--spec dflash2 requires --draft-tokens 7");
+        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
+            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
```

其余 1099 行分布在：`dflash_impl.h`(+308)、`layouts_impl.h`(+167)、`program_impl.h`(+140)、
`round_state.{h,cpp}`、`startup_features.h`、`speculative_target_impl.h`、`workspace_recipe.h`、
`model_view.h`、`27b/impl/config.h`、`bindings.{h,cpp}`、`variant.cpp`、bench/tests/docs。
注意 `385b30c` **没有碰 `src/ops/**` 与 `include/ninfer/ops/**`**（`git show --stat 385b30c -- src/ops include/ninfer/ops` 为空），
而上游 repo 里**根本不存在** `src/ops/wrapper/dflash2_selector.cpp`、`src/ops/launcher/dflash2_selector.cu`、
`src/ops/kernel/dflash2_grouped_conv.cuh`（`git ls-files | grep -i dflash2` 只列 docs/convert/tests + 两个 w8 内核）。
⇒ **那三个 op 域门是我们 fork 自己的代码**，上游不会替我们放宽；放宽的依据只能取上游的算法文档
（`qwen3.8-27b-dflash2.md:58-59, 81`：“`block_size=8` … 不是权重 shape、位置 embedding 容量或算法强制长度 …
检查不限制运行 K；改变 K 不需要改变或重新转换权重”，以及 `:177-178`“W 改变时，卷积的 request stride
和位置零边界必须同时改变”）。

## 2. 我们树里编码了固定宽度 K=7 的**全部**站点

### 2.1 硬点 —— 不改就跑不起 K<7（13 处代码 / 6 文件）

| # | file:line | 它假设了什么 | 编译期还是运行期 |
|---|---|---|---|
| 1 | `src/product/speculative_options.h:54` | CLI 校验：`draft_tokens != 0 && != 7` 直接 throw `--spec dflash2 uses the fixed 7-draft block`。从 `apps/cli/options.cpp:238` 与 `src/serve/serve_options.cpp:442` 两处调用 | **编译期字面量 7** |
| 2 | `src/targets/qwen3_6/impl/runtime/layouts_impl.h:869-870` | `validate_target_options`（同文件:805）里同款 `!= 0 && != 7` → throw `DFlash2 draft window must be 0 (fixed 7) or 7` | **编译期字面量 7** |
| 3 | `src/ops/launcher/dflash2_selector.cu:18` | `steps != 7` → throw（launcher 的 registered-domain 门） | **编译期字面量 7** |
| 4 | `src/ops/wrapper/dflash2_selector.cpp:34` | `steps != 7` → invalid（wrapper 的 registered-domain 门；与 3 是两道独立门） | **编译期字面量 7** |
| 5 | `src/ops/wrapper/dflash2_grouped_conv.cpp:22` | `block_size != 8` → throw。而 `dflash2_impl.h:214/269/286/300` 传的 `block_size` **就是 `k + 1`** ⇒ K≠7 必炸（**最容易漏的一处**） | **编译期字面量 8** |
| 6 | `src/targets/qwen3_6/impl/runtime/dflash2_impl.h:355` | 运行期门 `k == 0 \|\| k > 15 \|\| k != DFlash2Config::block_drafts` | **编译期常量** `DFlash2Config::block_drafts` |
| 7 | `src/targets/qwen3_6/impl/runtime/dflash2_impl.h:336` | selector scratch `candidates` 形状 `{B, Config::block_drafts, top_k}` | 编译期常量 |
| 8 | `src/targets/qwen3_6/impl/runtime/dflash2_impl.h:338` | 同上 `unary` | 编译期常量 |
| 9 | `src/targets/qwen3_6/impl/runtime/dflash2_impl.h:341` | 同上 `scores` `{B, block_drafts, top_k, top_k}` | 编译期常量 |
| 10 | `src/targets/qwen3_6/impl/runtime/dflash2_impl.h:345` | selector 的 `steps` 实参传 `Config::block_drafts`（不是 `k`） | 编译期常量 |
| 11 | `src/targets/qwen3_6/impl/runtime/layouts_impl.h:739` | 同一批 scratch 的 **arena recipe** 用 `batch * block_drafts * top_k`（注意：与 7-10 是**两个文件各写一遍**，必须一起动，否则 recipe 与实现不一致 → arena 越界/浪费） | 编译期常量 |
| 12 | `src/targets/qwen3_6/impl/runtime/layouts_impl.h:742` | 同上（FP32） | 编译期常量 |
| 13 | `src/targets/qwen3_6/impl/runtime/layouts_impl.h:745` | 同上（FP32，`top_k*top_k`） | 编译期常量 |

配套（切片顺手改，纯注释）：`dflash2_impl.h:313-314` “predicts the seven masked columns (1..7)”。

### 2.2 记录了宽度、但**不是**拦路（11 处，本切片不动）

| file:line | 内容 | 为什么不拦 |
|---|---|---|
| `layouts_impl.h:1089-1093` | `draft_window = (DFlash2 && draft_tokens==0) ? 7U : draft_tokens` | 只决定**默认值**；显式 `--draft-tokens 3` 会原样透传 |
| `src/targets/qwen3_6_27b/impl/package.cpp:127` | auto 解析时 `draft_tokens = 7` | 同上，只是 auto 默认 |
| `src/targets/qwen3_6_27b/impl/config.h:133` | `DFlash2Config::block_drafts = 7` | 切片后**无人引用**（成死常量）。**改它的值不会放宽任何门**（其余 12 处是字面量或独立常量） |
| `src/targets/qwen3_6_27b/impl/config.h:110` | 注释 “The block always drafts seven tokens after the bonus token” | 注释 |
| `src/targets/qwen3_6/impl/runtime/spec_decision.h:37` | 注释 “DFlash2 d7 acceptance 21-28%” | 注释 |
| `include/ninfer/ops/dflash2_selector.h:49` | 文档 “registered domain is … S=7 …” | 文档 |
| `include/ninfer/ops/dflash2_grouped_conv.h:30` | 文档 “domain is H=5120, T=1..64, … block_size=8 …” | 文档 |
| `src/targets/qwen3_6_35b_a3b/impl/config.h:123` | `DFlash2Config::block_drafts = 7` | 同文件 `:102` `supported = false` ⇒ 死配置 |
| `src/targets/muse_glimmer_30b/impl/config.h:182` | `block_drafts = 0` | `:161` `supported = false` ⇒ 死配置 |
| `src/targets/qwen3_6_27b/impl/config.h:145` | `kMaximumDFlashDraftTokens = 7` | **近义陷阱**：这是 DFlash v1 的帽（被 `layouts_impl.h:875` 的 DFlash 分支与 `variant.h:38` 用），**DFlash2 分支从不看它**。看到 "= 7" 会误以为是 dflash2 的上界 |
| （无） | `grep -rln dflash2 tests/` 为空 | **没有任何测试提到 dflash2** ⇒ 既没有测试挡路，也没有测试能兜住回归。这是 §5 必须用 CLI 观测而不指望单测的原因 |

### 2.3 本来就与 K 无关（切片**不碰**，这决定了切片能这么小）

- `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h:17` `kDFlashDecodeMaximumDrafts = 15`、`:18` `…Width = 16`。
- `src/targets/qwen3_6/impl/state/round_state.cpp:118-120` `columns = draft_window + 1`；`:205-210`
  `draft_candidate_ids/probs = {16, columns - 1, batch}`（与 upstream `385b30c` 加的 `{16, columns-1, batch}` 同形）。
- `layouts_impl.h:355-356` `drafts = plan.draft_window; verify = drafts + 1`；`:64-111` lambda
  `dflash2_proposal_capacity(verify, batch)` 全程用 `drafts` / `verify`（**只有 §2.1 的 11/12/13 三行例外**）。
- `dflash2_impl.h:172-173` `width = k + 1; columns = width * B`；`:214/269/286/300` `k + 1` 作 conv block_size；
  `:307/310-312/325/327/332` 全部按 `k` 缩放（`packed` / `flat_drafts` / `logits` / `projected`）；
  `:187-189` `attention_valid = width`。
- `program_impl.h:12312` `width = draft_window + 1`；`:12365/12373` `dflash2_envelopes(…, draft_window)`；
  `:12395` `extent = min({draft_window, max_by_budget, capacity - frontier - 1})`；`:12402` `target_valid_columns = extent + 1`。
- `src/targets/qwen3_6_27b/impl/variant.cpp:218-227` `Variant::dflash2_graph_profiles(capacity, draft_window, batch)`。
- op 域侧：`prepare_masked_block` 收 W=1..16（`src/ops/wrapper/prepare_masked_block.cpp:44-46`）；
  `prepare_ragged_prefix` 无宽度门；swa 的 token 域 `T=1..16`（`include/ninfer/ops/swa.h:43`）对 W=4 富余。
- `layouts_impl.h:34-44` 我们**本来就没有** `DFlash2PersistentLayout::full`（只有 local + rewrite_checkpoint_local），
  所以 upstream 为 DFlash v1 做的 `full → optional`（`dflash_context.h`、`layouts.h:26`）对我们 K 切片无关。

**计数**：硬点 **13 处代码 / 6 文件**；只记录宽度不拦路的 **11 处**（其中 2 处死配置、4 处纯注释/文档、2 处默认值、1 处近义陷阱、1 处会变死常量）。

## 3. 最小切片（＝交付的 `.diff`）

必须改：§2.1 的 1/2/3/4/5/6/10（7 处门与实参）+ 7/8/9（实现侧形状）+ 11/12/13（recipe 侧形状）。
可以留：§2.2 全部 11 处（含 `block_drafts = 7` 留作死常量、两处默认值 7 保持"不写 `--draft-tokens` 就是历史行为"）。
**真正与 `block_drafts` 耦合的只有 §2.1 的 6-10 与 11-13 两组**：前者是执行侧、后者是 arena 预算侧，
两边必须用同一个 K（否则 K<7 时实现要 `batch*7*16` 而 recipe 只给 `batch*3*16`）。

改动语义（每条都可逐字核对）：
1. `speculative_options.h` / `layouts_impl.h` 两个门：`!= 0 && != 7` → `> 15`（`qwen3_6::kDFlashDecodeMaximumDrafts`）。
   `0` 仍合法（＝默认 7），显式 1..15 放行，16+ 仍在**解析期**硬错。
2. `dflash2_impl.h`：新增 `const std::int32_t steps = static_cast<std::int32_t>(k);`，三个 scratch 形状与
   selector 的 `steps` 实参都改用它；删掉门里的 `k != DFlash2Config::block_drafts`。
3. `layouts_impl.h:739/742/745`：`block_drafts` → `drafts`（**K=7 时逐字节等价**，因为 `drafts == plan.draft_window == 7 == block_drafts`，
   所以这一步对现行唯一配置是零行为变化，也让 recipe↔实现的不变量成立）。
4. `dflash2_selector.{cpp,cu}` 两处 `steps != 7` → `steps < 1 || steps > 15`（域仍然被检查，只是放宽到 `W=K+1=2..16`）。
5. `dflash2_grouped_conv.cpp`：`block_size != 8` → `block_size < 2 || block_size > 16`，保留原有的
   `(block_size & (block_size - 1)) != 0` 幂次检查。

> **门槛偏差（请协调者裁决）**：任务给的"≤ 60 行且 ≤ 3 文件"里，本切片是 **+44/−32、6 文件**——
> 行数远低于 60，文件数超 1 倍。原因是 3 道 op 域门在 3 个互不相干的 op 文件里（selector 的
> wrapper + launcher、conv 的 wrapper），少改任何一道 K=3 都在运行期炸。若按 3 文件硬约束，
> 则**没有可交付的 diff**，§2.1 的 13 行表格就是等价规格，可按文件拆成 3 批（product+target /
> targets 内部 / ops 层）分别过窗。我选择交出完整可应用的 6 文件 diff，因为 dry-run 已证明它
> 干净可用（§6），而扣着它会让协调者多做一轮。

**K 的可达范围（切片之后）**：`K ∈ {1, 3, 7, 15}` 之中，
- `K = 3`（W=4）：本切片**即可**跑，是本次实验的目标；
- `K = 1`（W=2）：同理可达，是"最小开销探针"；
- `K = 7`：现状默认，零行为变化；
- `K = 15`（W=16）：**还差两处**——conv 的 `tokens ≤ 64`（`dflash2_grouped_conv.cpp:22`；K=15/B=8 → T=128）
  与 swa 优化域 `T=1..16`（`include/ninfer/ops/swa.h:43`，W=16 正好压线）。属 待证实，本切片不含。
- 其他 K（2/4/5/6/8..14）：见 §6 第 4 条，需要改内核取位置的方式。

## 4. upstream 还有、而本切片**没有**的东西（缺它会坏什么 / 基准实验要不要）

| upstream 件 | 缺了会坏什么 | benchmark-only 要吗 |
|---|---|---|
| proposal/verify extent 分离（`round_state.h:141 verify_positions`、`:79 proposal_valid_columns`） | 一个 lane 在 budget/tail 下 `extent < K` 时，草稿 SWA 仍按 W 列、verify 按 `extent+1` 列，"有效宽度"只能共用一个值——我们的共享点就是 `dflash2_impl.h:190 (void)valid_columns;`（proposal 侧不看 `target_valid_columns`） | **不要**：`max-context 8192` + 192 token 生成时 `extent ≡ draft_window`（`program_impl.h:12393-12396`），两条路径字节相同 |
| sparse rejection（`ops::speculative_accept_sparse_drafts` + `frame.proposal_q`） | 只能用 greedy accept；但我们的 `speculative_accept_greedy_drafts` 已经收 `draft_candidate_ids/probs`（`speculative_target_impl.h:27-33`；wrapper `:150` 要求 `[16,K,B]`），**分布校正拒绝已在**，且形状本来就随 K | **不要** |
| partial terminal commits（Frontend 选完最终前缀后才提交 GDN state / token counts） | 一轮在块中间结束（EOS/budget）时可能多提交或少提交状态。我们那两次 `dflash2 gen=2 stop_token`（board S47 段）正落在这类症状上 | **不要**，但**必须当观察项**：`gen=` 远小于 `max_tokens` 时这次数字作废 |
| Vision / Host 复用、`DFlashPersistentLayout::full → optional` | 与 K 无关；我们的 `DFlash2PersistentLayout`（`layouts.h:34-44`）本来就没有 full PagedKVCache | **不要** |
| `--draft-tokens` 产品打通 + schema v14 | 我们早已有 `--draft-tokens`（`apps/cli/options.cpp:149`、`src/serve/serve_options.cpp:350`）与 `SpeculativeOptions.draft_tokens`（`include/ninfer/types.h`），只是**数值域被夹住**；schema v14 是 bench 汇总输出格式，与引擎无关 | **不要** |
| `DFlashConfig`/`DFlash2Config` 合一 + `is_masked_draft_backend()` + `DFlash2BranchRoots` recipe 共享 | 见 §6 第 1/2 条；recipe 共享我们的 `dflash2_proposal_capacity` 已按 `drafts`/`verify` 泛化 | **不要** |

## 5. 最便宜的可证伪实验（切片落地、重编之后）

复用既有 harness `_spec_sweep.sh:21-51` 的调用形状（同一产物 / 同一计数 prompt / 同 `max_tokens=192` /
`--no-cuda-graph`），只留 3 和 7 两档：

```bash
# 0) 落地后重建（窗口！见 §6.6），然后确认门已拆——这一步只读日志，不占 GPU
BIN=/home/user/ninfer-fusion/build/apps/ninfer-serve
$BIN --spec dflash2 --draft-tokens 3 2>&1 | head -3
#    期望：不再出现 "--spec dflash2 uses the fixed 7-draft block"
#    （该校验在解析期跑：src/serve/serve_options.cpp:442）

# 1) 两跑（一次 GPU 窗口，约 2 × 40 s + 加载）
for n in 3 7; do
  pkill -f "$BIN"; sleep 3
  cd /home/user/ninfer-fusion/build
  setsid nohup $BIN /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
    --port 8387 --max-context 8192 --no-cuda-graph --no-thinking \
    --spec dflash2 --draft-tokens $n > /home/user/sw_df2k$n.log 2>&1 < /dev/null &
  for i in $(seq 1 120); do sleep 2
    curl -s -m 2 http://127.0.0.1:8387/health | grep -q ok && break
    pgrep -f "$BIN" >/dev/null || break
  done
  curl -s -m 300 -X POST http://127.0.0.1:8387/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,"}],"max_tokens":192,"temperature":0,"stream":false}' >/dev/null
  grep -E 'done .*speculative=' /home/user/sw_df2k$n.log | tail -1   # ← 唯一需要的观测
done
pkill -f "$BIN"
```

### 观测量与预先写死的判据

主观测（`_spec_sweep.sh:44-48` 已经在 grep 的同一行）：
`done round=… speculative=<backend> gen=<n> decode=<x>tok/s`。

1. **有效性闸（先看这条）**：两跑的 `speculative=` **都必须读 `dflash2`**。
   若读 `mtp`，说明验收地板把 lane 降级了——`dflash2_acceptance_too_low()`
   （`src/targets/qwen3_6/impl/runtime/spec_decision.h:99-107`，地板 `kDFlash2MinAcceptance = 0.05`、样本 `128`）
   在 `program_impl.h:12390-12396` 把 `extent` 置 0 ⇒ 这一跑测的不是 K，**作废**。
2. **有效性闸 2**：两跑 `gen=` 必须可比且接近 192。上一轮 dflash2 出过 `gen=2`
   （`M_spec_sweep.md:14`），那就是块中间终止的征状（§4 的 partial-commit 一类）⇒ 作废。
3. **判据（预先登记）**：以同机同 prompt 的 DSpark 参考（d3 80.5 > d7 70.0）为外推，
   预测 **K=3 的 decode tok/s > K=7**。
   - 成立 ⇒ 宽度这条杠杆**也**对 dflash2 成立，可以继续买 K∈{1,3,7,15} 的细扫；
   - **不成立（K=3 ≤ K=7）⇒ 证伪**，且结论是"dflash2 的瓶颈不在宽度，在草稿步的固定开销/输入链"，
     与 E5 的首选假设 H1（草稿输入链与 checkpoint 训练约定不一致）方向一致 ⇒ 杠杆要换个找法。
4. 次要观测（几乎免费，同一份日志）：`tok/round` 与 `accepted by pos` 直方图
   （serve `src/serve/request_log.cpp:438-456`；CLI `apps/cli/main.cpp:238-245`）。
   若 K=3 的 tok/round 更低而 tok/s 更高 ⇒ 赢在延迟；若 tok/round 也升高 ⇒ 接受率也变了
   （注意 upstream §7.4 明说不同 K 的 proposal 本来就不同，所以这**不是**异常）。

为何这是最便宜的：**不需要新产物、不需要重新转换权重**（`qwen3.8-27b-dflash2.md:81` 明写改 K 不动权重）、
不需要新 CLI 开关（`--draft-tokens` 已在）、不需要 CUDA Graph（`--no-cuda-graph` 走 eager，避开
`program_impl.h:11157-11206` 那套按 K 重新捕获 graph family 的面）、两跑即可给出可证伪结论。

## 6. 「现在先别做」清单（附理由）

1. **别移植 upstream 的 `DFlashConfig` 重构**（dflash/dflash2 权重合一、删 `DFlash2Weights`、加
   `backend`/`coherent_selector`/`layers`/`full_layers`；`model_view.h`、`bindings.{h,cpp}`、`variant.cpp`）。
   我们 27b 的 `DFlashConfig`（DSpark v1，`config.h:74-106`）与 `DFlash2Config`（`:111-139`）是**两套真并存**
   的设计，合一等于重写加载链；K 实验完全不需要。
2. **别把 `RoundStateSpec` 的两个 bool 换成 `SpeculativeBackend backend` + `is_masked_draft_backend()`**。
   我们有语义等价的 `StartupFeatures::dflash_like()`（`startup_features.h:26`）与
   `enable_mtp/enable_dflash`（`round_state.h:25-26`，构造点 `layouts_impl.h:319-320`）；换法是纯机械，
   但会扫到 `round_state.cpp` 全部 `if`、`decoder_state.cpp:190`，收益为零。
3. **别现在做 K>7（尤其 K=15）**。除本切片的门之外还差：conv 的 `tokens ≤ 64`
   （`dflash2_grouped_conv.cpp:22`，K=15/B=8 → T=128）与 swa 的 `T=1..16` 优化域
   （`include/ninfer/ops/swa.h:43`，W=16 压线）；且 W 变大后 recipe/graph profile 全部变宽，需要独立窗口与独立测量。
4. **别现在做非 2 的幂宽度（K=2/4/5/6/8..14）**。`src/ops/kernel/dflash2_grouped_conv.cuh:29` 用
   `position = t & (block_size - 1)` 求 block 内位置，只有 `W ∈ {2,4,8,16}` 时它才等于"请求内位置"
   （`qwen3.8-27b-dflash2.md:177-178` 要求卷积边界跟随请求边界）。要任意 K 得把该行改成 `t % block_size`，
   那是一次独立的内核改动 + 等价性复验。K=3 恰好落在幂上，所以第一枪选它最省。
5. **别改 `DFlash2Config::block_drafts` 的值**（`27b config.h:133`）。切片后它无人引用，改它对四个门
   毫无影响（门是字面量/独立常量），只会制造"已经放宽了"的假象。
6. **别在构建窗口里落盘**。本切片 6 文件与 window 正在编的 decode TU 不直接重合，但
   `layouts_impl.h` / `dflash2_impl.h` 是 TU 拆分（S23/S40）的入口头 → 要等窗口 D/E 结束，
   并按既有流程 touch 强制补编。
7. **别顺手把文档里的 "S=7" / "block_size=8" 也改了**（`include/ninfer/ops/dflash2_selector.h:49`、
   `dflash2_grouped_conv.h:30`）。它们只是注释，改了本切片就从 6 文件变成 8 文件，不值得在这个窗口做。
8. **别把"K 可调"读成"K 越小越快"**。upstream 自己写明：非因果 attention 使改变 W 后的
   hidden/candidates/q **通常不等于更宽 block 的前缀**，卷积也必须用匹配的 request 边界；
   算法合法性不要求不同 K 的 proposal 相同，**也不要求**新 K 的接受率或吞吐优于推荐的 K=7
   （`docs/maintainer/qwen3.8-27b-dflash2.md:386-396`）。所以 K=3 变慢是**合法结果**而非 bug。

## 7. 本轮证据与残留的 待证实

### 7.1 `patch -p1 --dry-run`（对当前基准树文本，原文粘贴）

```
===== md5 BEFORE =====
77a4bb066cd943e31e6a60cec838800c  src/product/speculative_options.h
7e7ec9f5ba802589412916254d561880  src/targets/qwen3_6/impl/runtime/layouts_impl.h
10f01befe2d276df5e967342a8fc003e  src/targets/qwen3_6/impl/runtime/dflash2_impl.h
3a66ad044f5c933a2a5c423e645522fc  src/ops/wrapper/dflash2_grouped_conv.cpp
43f804776237821ecd12636bfb57975e  src/ops/wrapper/dflash2_selector.cpp
4ceb362dce3974955d55877a6a34ad0b  src/ops/launcher/dflash2_selector.cu

===== patch -p1 --dry-run =====
checking file src/product/speculative_options.h
checking file src/targets/qwen3_6/impl/runtime/layouts_impl.h
checking file src/targets/qwen3_6/impl/runtime/dflash2_impl.h
checking file src/ops/wrapper/dflash2_grouped_conv.cpp
checking file src/ops/wrapper/dflash2_selector.cpp
checking file src/ops/launcher/dflash2_selector.cu
patch_dryrun_rc=0

===== md5 AFTER =====
77a4bb066cd943e31e6a60cec838800c  src/product/speculative_options.h
7e7ec9f5ba802589412916254d561880  src/targets/qwen3_6/impl/runtime/layouts_impl.h
10f01befe2d276df5e967342a8fc003e  src/targets/qwen3_6/impl/runtime/dflash2_impl.h
3a66ad044f5c933a2a5c423e645522fc  src/ops/wrapper/dflash2_grouped_conv.cpp
43f804776237821ecd12636bfb57975e  src/ops/wrapper/dflash2_selector.cpp
4ceb362dce3974955d55877a6a34ad0b  src/ops/launcher/dflash2_selector.cu

===== stray .orig/.rej =====
(nothing above = none)
```

复现命令（`dry-run` 只读，不写源）：

```
wsl.exe -e bash -c "cd /home/user/ninfer-fusion && patch -p1 --dry-run < /mnt/c/Users/User/Documents/ziqinzhang/_collab/E9_s52_dflash2_k_slice.diff"
```

生成器自带三条自检：每个替换**恰好命中 1 次**、生成前后源文件 md5 相同、替换后 `block_drafts` 计数为 0。

### 7.2 diff 的形状

`_collab/E9_s52_dflash2_k_slice.diff`：6 文件 / **+44 −32** / 10309 字节。
由 `_collab/E9_s52_mkpatch.py` 从**基准树字节**生成，因此**保留了 `dflash2_impl.h` 的 CRLF**
（`file` 探针：`src/targets/qwen3_6/impl/runtime/dflash2_impl.h: … with CRLF line terminators`，
其余 5 个是 LF）——手写 patch 会在这里失败，这也是用生成器而不是手写 diff 的原因。

### 7.3 待证实（本轮**没有**证明的）

1. **编译通过**：本轮按约束**未运行任何编译器**（连 `g++ -fsyntax-only` 都没跑）。唯一新引入的标识符是
   `qwen3_6::kDFlashDecodeMaximumDrafts`，其可见性靠读 include 链证明：`layouts_impl.h:2 → layouts.h:11 →
   round_state.h:17`（该文件已定义它），且 `layouts_impl.h` 内已有同形限定用法（`:74 qwen3_6::kKvInt8QuantGroup`、
   `:314 qwen3_6::begin_round_state_layout`）⇒ **结论：可见性零风险，但"能编过"仍是 待证实**。
2. **K=3 的数值/吞吐**：完全未测（无 GPU）。§5 的判据是预先登记的预测，不是结论。
3. **K=3 的正确性**：本切片只放宽"注册域"，没有改任何数学。但 upstream §7.4 明说不同 W 是**另一次合法
   forward**、不是更宽 block 的前缀 ⇒ K=3 的 proposal 与 K=7 的前 3 列**不应**逐字节相同；
   如果协调者想验证实现正确性，正确的期望值是"无 CUDA 错、无 NaT/NaN、生成不崩"，**不是**字节等价。
4. **`kMaximumDFlashDraftTokens = 7`（27b `config.h:145`）是否需要跟着改**：待证实。它现在只服务 DFlash v1；
   若将来想要 dflash2 走 variant 级上限（而不是产品层字面量 15），需要另开一条。本切片不碰。
5. **构建树里的陈旧残留**：`include/ninfer/types.h.orig` 存在（上一轮 patch 的 `.orig`），与本切片无关，
   但同类文件会让后续 `patch` 的排障变吵；建议顺手清（不属本切片）。

### 7.4 与任务书中数字的差异（复述 §0 的更正）

| 任务书说法 | 实测来源 | 判定 |
|---|---|---|
| “dflash2 decode tok/s d1 65.4 / d3 80.5 / d7 70.0” | `_collab/M_spec_sweep.md:9-11` 三行 backend 列写的是 **`dflash`**；dflash2 的 d1/d3 是 `SERVE_FAILED`（同一文件 `:12-13`），d7 是 `gen=2 / 31.7tok/s`（`:14`） | **前提需更正**：该 sweep 是 DSpark/DFlash v1 的宽度扫描，dflash2 的宽度杠杆 **待证实** |
| “width is a real lever” | DSpark 上成立（80.5 > 70.0）；dflash2 上无数据 | 待证实（K=3 实验即为此设计） |
