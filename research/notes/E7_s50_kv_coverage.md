# S50 / E7 — 移植上游 `03177b9`「kv coverage 下界化」到我们的树

产物：`_collab/E7_s50_kv_coverage.diff`（4 文件 / 26 hunk / +46 −38），本文件，`_collab/board.md` 的 S50 行。

> **取证口径 / 关于 build tree 的说明（必读）**
> 本会话跑在 Windows 侧，`/home/user/ninfer-fusion`（canonical build tree）**在本机不可达**
> （`ls /home/user` 下只有 `htslib-1.20`，无 `ninfer-fusion`；`/mnt` 不存在）。因此：
> - diff 的**行号与上下文文本取自镜像** `C:\Users\User\Documents\ziqinzhang\ninfer-fusion-repo`，
>   并已逐条核对 M 在 `_collab/M_upstream_borrow.md:28-29` 引用的两处现状文本确实在镜像里逐字存在
>   （`logical_kv_store.h:1494` 的签名、`:1498-1501` 的旧抛错与早返回；`program_impl.h` 9 处
>   `materialize_sequence_kv`）。
> - `patch -p1 --dry-run` 是在**镜像根目录**跑的，全绿（§6）。
> - **M 需要在真正的 build tree 上再跑一次**（命令见 §6 与 board S50 行）：若镜像与 build tree 在这 4 个
>   文件上已有分叉，hunk 会以 `Hunk #N FAILED` 的形式报出来，不会静默改错。
> - 下文所有 `file:line` 都指**镜像**的行号（= 打补丁**前**的行号）。

上游参考：`git show 03177b9`（commit `03177b910e70f783b00c4f980ce0d1896a6b8592`，
`fix(runtime): preserve kv coverage during speculative terminal settlement`，
9 文件 / +172 −38）+ `git show 03177b9 -- docs/maintainer/paged-kv-cache.md`
+ `git show 03177b9:tests/targets/qwen3_6/speculative_page_boundary.h`。

---

## 1. 这份 diff 到底改了什么

| # | 文件 | hunk | 内容 |
|---|---|---|---|
| 1 | `src/targets/qwen3_6/impl/runtime/logical_kv_store.h` | 2 | `#include <string>`（`:12` 后）；`:1494` 起改名 `materialize_to_tokens`→`ensure_mapped_to_tokens`、**删掉 `target < address.page_count` 这一半谓词**、`:1501` 改成 `if (target <= address.page_count) { return; }`、错误消息加富（tokens / required_pages / mapped_pages / reserved_pages / entitlement）+ 两行契约注释 |
| 2 | `src/targets/qwen3_6/impl/runtime/program.h` | 1 | `:1282` 声明 `materialize_sequence_kv`→`ensure_sequence_kv_mapped` |
| 3 | `src/targets/qwen3_6/impl/runtime/program_impl.h` | 12 | 8 个 `materialize_sequence_kv` 调用点 + 1 个定义（`:10320`）改名；5 个 `materialize_to_tokens` 直接调用点（`:1168`、`:10328`、`:10330`、`:10852`、`:11308`）改名。**参数逐字未动** |
| 4 | `tests/targets/qwen3_6/test_context_store.cpp` | 11 | 12 处 `materialize_to_tokens` 调用改名（纯机械；不改就编不过） |

**没有**包含在 diff 里的上游改动（刻意的，见 §5）：
`program_impl.h:11416`（`enqueue_dflash_context_append`）的**语义**改动
（上游把它换成「只保障 backend」，并把 Main KV 请求整条删掉）、
新增测试 `tests/targets/qwen3_6/speculative_page_boundary.h`、
`tests/README.md` 与两个 engine 测试的接线。

命名对齐说明：上游同时把 program 层的包装也改了名（`materialize_sequence_kv`→`ensure_sequence_kv_mapped`）。
本 diff 跟着改，好处是今后再借上游 hunk 时上下文能对上；代价是触及 14 个位置——但由 §7 的机械证明保证
**除标识符外零文本变化**。

---

## 2. 行为对照表：每个调用点改前 / 改后

`pages_for_tokens(t) = t==0 ? 0 : 1 + (t-1)/64`（`logical_kv_store.h:1820`，`kPagedKVPageSize=64`
见 `src/core/paged_kv_cache.h:17`）。
`page_count` = 当前已映射页数；`entitlement(addr) = page_count + reservation.pages()`（`:1889-1892`）。

设 `target = pages_for_tokens(请求 tokens)`：

| 改前 | 改后 |
|---|---|
| `target < page_count` → **throw `invalid_argument("KV materialization exceeds active entitlement")`** | `target <= page_count` → **直接 return，不裁剪、不动映射、不改 entitlement** |
| `target > entitlement` → throw 同上 | `target > entitlement` → throw（消息加富，见 §1） |
| `target == page_count` → return | 同左（被 `<=` 覆盖） |
| `target > page_count` → 映射新增页 | 同左，逐字不变 |

所以**唯一的语义差**是「严格收缩的请求」：老码抛错，新码是 no-op。
（`target == page_count` 这条老码本来就 return，所以「相等」这件事两边一致。）

### 2.1 全量调用点

`program.h:1282` 声明 `materialize_sequence_kv(seq, main_tokens, backend_tokens=0)`；
定义在 `program_impl.h:10320-10332`，内部对 **Main KV**（`:10328`）和 **backend KV**（`:10330`，
仅当 `backend_tokens != 0`）各发一次请求。
M 说的「9 处」= `program_impl.h` 里 **9 个文本命中 = 8 个调用点 + 1 个定义**（`:8149 :8711 :9699
:10320(定义) :11416 :11834 :11993 :12180 :12415`）。逐点如下。

| # | 站点（改前行号） | 请求的 (main, backend) tokens | 老码在该点会不会抛 | 该 throw 是否 **load-bearing** |
|---|---|---|---|---|
| T1 | `program_impl.h:1168` `causal_score` | `(predictor_count, —)` | **不会**。地址来自 `:1163 create_active(entitlement, 0)` → `create_active`=`create_inactive()`+`activate()`（`logical_kv_store.h:919-931`），`activate` 的 `required_pages = address.page_count = 0`（`:970-981`），所以 `page_count==0`，`target>=1>0` | 否 |
| T2 | `:10328` 包装体（Main） | `(main_tokens, —)` | 由调用方决定，见 W* | 见 W* |
| T3 | `:10330` 包装体（backend） | `(backend_tokens, —)` | 由调用方决定，见 W* | 见 W* |
| T4 | `:10852` `prepare_graphs` → `reserve_capture_rows` lambda | `(1, —)` | **不会**。`create_active(1, row)`（`:10849`）⇒ `page_count==0`，`target==1` | 否（一次性建表，之后同一地址不再 materialize） |
| T5 | `:11308` capture 行补图（`row = max_concurrency` 的专用行） | `(1, —)` | **不会**。同上 `create_active(1, max_concurrency)`（`:11304`） | 否 |
| W1 | `:8149` `publish_active_capture` | `(prefill.prompt_tokens, backend_materialized)` | **不会/相等**。`publish_shared` 分支刚做过 `commit_active_snapshot`（`:8127-8143`），其目标地址的 `page_count = pages_for_tokens(snapshot.frontier_)`（`logical_kv_store.h:1439`），而 `prefill.prompt_tokens == frontier_` 是同一前缀 ⇒ 多数情况下 `target == page_count`（老码的 return 分支） | 否 |
| W2 | `:8711` `append_forced_tokens` | `(end, end)`（`speculative_backend==None ? 0 : end`）；`end = base + row_stride`（`:8691`） | **不会**。进入前刚 `commit_sequence_kv(text_kv_valid, dflash_context_frontier)`（`:8705`），且 `:8700` 的 `enqueue_dflash_context_append` 只会在 DFlash 下走；Main 的 `page_count` 至多 `pages_for_tokens(text_kv_valid) <= pages_for_tokens(base) < pages_for_tokens(end)` ⇒ 单调增 | 否（**但它有一个别的问题，见 §5.2**） |
| W3 | `:9699` `start_sequence` | `(prompt_tokens, backend_materialized)` | **不会**（常规路径）。紧接着 `:9690` 的 `trim_sequence_kv(sequence, base, …)` 已把 `page_count` 打到 `pages_for_tokens(base)`（`destructive_truncate` 成功后 `page_count==target`，`logical_kv_store.h:1548-1580`），且 `base <= prompt_tokens` ⇒ `target >= page_count` | 否（`preserving_source` 分支无预裁剪，**理论上**可收缩；但老码抛错从未在任何日志里出现过，见 §4 末） |
| W4 | **`:11416` `enqueue_dflash_context_append`** | `(max(text_kv_valid, end), end)`，`end = start + counts[row]` | **会抛**。这是**唯一**可证可达的收缩点：调用方有两种，见下 | **是，但被它抓到的是上游自己的 bug**（不是我们树里的不变式） |
| W5 | `:11834` `decode_ordinary_batch` | `(frontier+1, 0)` | **不会**。轮首 `page_count == pages_for_tokens(text_kv_valid) == pages_for_tokens(frontier)`（上一轮 settle 结尾 `:9985` / `:8801` / `:7944` 都 trim 到已提交 frontier）；`target = pages_for_tokens(frontier+1) > page_count` | 否 |
| W6 | `:11993` `decode_mtp_batch` | `(frontier+extent+1, min(capacity, frontier+extent+draft_window))` | **不会**。轮首 Main `page_count==pages_for_tokens(frontier)`、backend `page_count==pages_for_tokens(mtp_kv_valid)==pages_for_tokens(frontier)`（`:9966-9967`/`:9984`），两个 target 都 ≥ 各自 page_count ⇒ 单调增 | 否 |
| W7 | `:12180` `decode_dflash_batch` | `(frontier+extent+1, frontier)` | **不会**。backend 侧 `page_count == pages_for_tokens(dflash_context_frontier)`，而 `dflash_context_frontier <= frontier` 恒成立（终止轮在 `:9979-9981` 被置为 `execution_frontier`；非终止轮被归位到 `base_E`，本轮 `frontier` 已推进到 `base_E+produced`，`:9848-9850` 强制非终止必须全接受）⇒ `target = pages_for_tokens(frontier) >= page_count`；Main 侧同理 ≥ | 否 |
| W8 | `:12415` `decode_dflash2_batch` | `(frontier+extent+1, 0)` | **不会**。DFlash2 无 backend（`:12407 dflash_kv_table_rows[row] = 0` 且注释「DFlash2 owns no backend KV」），main 轮首 `== pages_for_tokens(frontier)` ⇒ 单调增 | 否 |
| TF1 | `:187` `:200` `test_context_store.cpp` | `(65)`、`(129)` | **不会**（65 < 129，且起点是 `create_active(3,0)` ⇒ `page_count==0`） | 否（但注意：**上游给同一个地址加了第 3 次调用并把它当契约测**，见 §8） |
| TF2 | `:322/:347/:380/:402/:435/:449/:498/:512/:526/:533` | 各自常量 | **不会**。10 个站点都作用在 `create_active(...)` 出来的地址上，且首次调用即把覆盖推到 ≥ 后续请求值；`commit_frontier` 并不改 `page_count`（`:1528-1547`），故 `:200`(`129 == pages_for_tokens(129)=3` 覆盖 65 的 2 页) 那类「同址递增」是全部形态 | 否 |

**W4 展开（这是唯一真正可达的收缩点，也是本 diff 的实测价值所在）**

调用方 A —— `:9917`，`resolve_pending_raw` 的 DFlash 终止结算：
```
:9903  if (speculative_backend == SpeculativeBackend::DFlash) {
:9904-9912  for row: if (!cancelled[row] && terminal[row]) {
:9905      append_starts[i] = requests[lane].pending.base_E;
:9906      append_counts[i] = accepted_tokens[row];        // == committed
:9917      enqueue_dflash_context_append(...)
```
此时 Main 地址的 `page_count` 来自**本轮的 verify 映射**：
`decode_dflash_batch:12180` 请求过 `frontier + extent + 1`，即 `page_count = pages_for_tokens(base_E + extent + 1)`。
而 `end = base_E + accepted`，且 `:9961` 之前 `sequence.text_kv_valid` 仍等于 `base_E`
（`:9821-9827` 的轮首校验强制 `text_kv_valid == pending.base_E`，否则先抛
`"speculative pending row is not at its recorded base"`），所以 `max(text_kv_valid, end) == end`。

⇒ 当 `pages_for_tokens(base_E + accepted) < pages_for_tokens(base_E + extent + 1)`
（即**已接受前缀与 verify 窗口尾部落在不同页**，例如 `base_E=60, extent=8, accepted=2`：
`pages_for_tokens(62)=1 < pages_for_tokens(69)=2`），**老码必抛**
`"KV materialization exceeds active entitlement"`；**新码在第 4 行判断处直接 return**。

这正是上游 `speculative_page_boundary.h` 里那段注释
（`git show 03177b9:tests/targets/qwen3_6/speculative_page_boundary.h`）描述的场景：
「Begin samples one token at E=63. Verify then maps past 64, but this stop commits only one target
column, ending at E=64.」——跨页 verify + 只提交一列的终止。

调用方 B —— `:8700`（`append_forced_tokens` 内的 DFlash 上下文追平）：`end == base`，
`page_count == pages_for_tokens(text_kv_valid) == pages_for_tokens(base)` ⇒ **相等**，老码走 return。**不可达**。

### 2.2 结论：老 throw 是不是 load-bearing？

| 站点 | 判定 | 依据 |
|---|---|---|
| T1/T4/T5（3 处 `create_active` 后的首次物化） | **否**，结构性不可能收缩 | `logical_kv_store.h:919-931` + `:970-981`：新地址 `page_count==0` |
| W1 W5 W6 W7 W8（5 个轮首/绑定点） | **否**，轮首不变式 `page_count == pages_for_tokens(claim 的 frontier)` 由上一轮的 `trim_sequence_kv` 尾裁保证 | `:9985`、`:8801`、`:7944`、`:9962-9964`、`logical_kv_store.h:1548-1580` |
| W3（start_sequence） | **常规路径否**；`preserving_source` 分支理论上可收缩 | `:9690` 只在 `!preserving_source` 时裁剪（`:9303` 定义 `preserving_source`） |
| W2（append_forced_tokens） | **否** | 进入前 `:8705` 的 `commit_sequence_kv` + `end > base` |
| **W4（enqueue_dflash_context_append）** | **是**：老 throw 确实在拦一个真实事件；但被拦下的是**这一阶段对该 pool 的请求本身是多余的**（该阶段只写 DFlash backend KV），不是任何不变式破坏 | 上游把这条请求整条删掉：`git show 03177b9 -- src/targets/qwen3_6/impl/runtime/program_impl.h` 中 `enqueue_dflash_context_append` 的 hunk |
| TF1/TF2（12 个测试点） | **否** | 全部 `create_active` 起步、单调递增 |

**风险结论（M 最关心的那条）**：本 diff **不会**把任何「树里依赖它来抓 bug」的抛错变成静默 no-op。
唯一可达的收缩点 W4，其老 throw 抓的是上游同类阶段性 bug（阶段请求了不属于自己的 pool），
新语义把它变成「返回已有映射」，而我们在 §5.1 说明了上游随后又用更彻底的方式删掉了这条请求——
两条路都指向「请求本身多余」，不存在「本该报错却被吞掉」的情形。

---

## 3. 我们树今天在「投机终止结算」上做了什么

函数：`ProgramImplCore::resolve_pending_raw`（`program_impl.h:9769`），投机行分支从 `:9823` 开始。

**执行顺序（含行号）**

| 步骤 | 行号 | 内容 |
|---|---|---|
| ① 轮首不变式校验 | `:9821-9838` | 逐行要求 `execution_frontier==base_E`、`ledger/prefix_*` 尺寸、`text_kv_valid==base_E`、MTP 的 `mtp_kv_valid==base_E`、DFlash 的 `dflash_context_frontier==base_E`，否则抛 |
| ② recurrent(GDN) state 补齐 | `:9868-9869` | `replay_fold->execute(fold_rows, stream)`，`commit_columns = committed`（`:9856-9858`） |
| ③ hidden 补齐 | `:9871-9902` | 仅 `needs_hidden_correction`（= 存在 `partial_terminal`，即终止且 `committed < produced`，`:9858-9862`）时：`ops::speculative_select_accepted_hidden`（`:9897`）+ `ops::scatter(...)` 到 `state_images->continuation_hidden_store()`（`:9898-9899`）。MTP/DFlash/DFlash2 三个 frame 都覆盖（`:9881-9890`） |
| ④ draft context 补齐 | `:9903-9920` | **仅 DFlash**：对 `!cancelled && terminal` 的行 `enqueue_dflash_context_append(base_E, accepted_tokens)`（`:9917`） |
| ⑤ 等 GPU | `:9922-9926` | `device.synchronize()`，`work.reset()` |
| ⑥ fork 结算 / ledger 推进 | `:9950-9967` | `settle_state_fork`（`:9950`）、`ledger.insert`（`:9952`）、`prefix_identity/digests`（`:9953-9955`）、`advance_rebuild_work`（`:9961`）、`execution_frontier/ledger_frontier/text_kv_valid/tail_hidden_valid`（`:9962-9965`） |
| ⑦ backend frontier 归位 | `:9969-9982` | MTP：`mtp_kv_valid = execution_frontier`；else（DFlash/DFlash2）：`dflash_context_frontier = terminal ? execution_frontier : base_E` |
| ⑧ **发布 committed frontier** | `:9984` | `commit_sequence_kv(sequence, text_kv_valid, backend_kv_valid(sequence))` → `logical_kv_store.h:1528 commit_frontier`（对每个 changed page 做 `commit_coverage`，然后置 `committed_frontier`） |
| ⑨ **裁掉未提交尾页** | `:9985` | `trim_sequence_kv(...)` → `logical_kv_store.h:1548 destructive_truncate` |

**与上游 doc 契约的逐句比对**（`git show 03177b9 -- docs/maintainer/paged-kv-cache.md` 新增的两段）：

| 上游句子 | 我们树 | 结论 |
|---|---|---|
| 「投机映射可能已经超出本阶段所需；只有显式 truncate 才能释放，`commit_frontier` 才发布有效 token」 | 老码在 `logical_kv_store.h:1498` 违反了前半句（对更短的请求抛错）；`commit_frontier` 只改 `committed_frontier`/页内 `committed_columns`，不改 `page_count`（`:1528-1547`） | **本 diff 修的就是这句**；后半句已成立 |
| 「各阶段只保障自己会写入的 pool：target prefill/verify 负责 Main KV；draft context append 只保障 DFlash Full backend KV。**DFlash2 的 draft context 全写固定 cyclic state，不需要 paged KV 物化**」 | 我们的 `enqueue_dflash_context_append` 在 `:11416` **仍对 Main KV 发请求**；DFlash2 侧已经符合（`:12415` 只物化 Main、backend 传 0，且 `:12407` 注释明说 DFlash2 无 backend KV） | **一条未移植的偏离，见 §5.1**（不影响正确性，因为新语义把它变成 no-op） |
| 「投机终止先按最终提交数量完成 recurrent state、hidden 和 draft context 的补齐，等待 GPU 工作完成后发布 committed frontier，再裁掉未提交的尾页。不能为满足某个后续阶段更短的覆盖需求而提前裁剪 verify 的映射」 | **完全符合**：②③④（补齐）→ ⑤（`device.synchronize()`）→ ⑧（发布）→ ⑨（裁剪），顺序与句子逐项对齐；`:9903-9920` 的 append 发生在 ⑧⑨ **之前**，而它请求的更短覆盖在**老码**会抛、**新码**会 no-op——恰好是「不能提前裁 / 不能因此失败」的落点 | **结算顺序不需要改**（见下） |

### 3.1 coverage 改动单独是否足够？上游的排序修复我们需不需要？

**结论：对我们树当前的结算顺序，coverage 改动单独就足够；上游那份 commit 里也没有「排序」代码改动可移**——
`git show 03177b9 -- src/targets/qwen3_6/impl/runtime/program_impl.h` 的 6 个 hunk 全部位于
`:1061 / :8294 / :8916 / :9940(=我们的 `:9699` 一带) / :10792 / :10962 / :11370 / :11776 / :11935 / :12113`，
**没有任何一行落在 `resolve_pending_raw` 的结算段**。上游是把已有的顺序**写进 doc 当契约**，
再用 `enqueue_dflash_context_append` 的 pool 收窄把「更短的覆盖需求」从根上去掉。

**我们树的排序与上游契约无冲突的**具体证据**：**`resolve_pending_raw:9984-9985` 的 commit→trim 相邻两行、
`:9922-9926` 的同步屏障、`:9962-9965` 的 frontier 推进都发生在 trim 之前**。
我们**没有**找到任何「为了满足后续阶段更短的覆盖而提前裁 verify 映射」的行——即
**没有发现违反上游契约的具体行**。

（保留一点余地：这一条是「在 `resolve_pending_raw` 这条主链上没找到」。若要把它升格成全程断言，
需要在 `destructive_truncate` 上加一个「verify 未结算期间禁止收缩」的断言或计数器——
本会话是纯文本作业，**待证实**。）

---

## 4. 与上游的其余偏离（未移植，标明理由）

### 4.1 `enqueue_dflash_context_append:11416` 仍向 Main KV 发请求（上游已删）

- 我们：`materialize_sequence_kv(sequence, std::max(sequence.text_kv_valid, end), end);`（`:11416`）
- 上游改后：只在 `sequence.kv->backend` 存在时 `backend_kv_addresses->ensure_mapped_to_tokens(*sequence.kv->backend, end, device.stream);`，Main KV 请求整条消失，
  并留了注释「Context append writes only the draft caches. Target execution owns Main KV coverage,
  including any uncommitted suffix that survives until the round is settled.」

**为什么不放进本 diff**：这是**语义收窄**（同时也顺带把 `backend_kv_cache() ? end : 0U` 那套
backend 选择也改了），会改变「谁负责 Main KV」的责任划分；M 明确要求本 diff 只做机械改名 + 下界语义。
新语义下这条多余请求已经是 no-op，**正确性不受影响**；但如果要跟上游完全对齐，建议作为**下一个独立小补丁**
（1 hunk，且能把 §5.2 那个隐患一起收掉）。

### 4.2 `append_forced_tokens:8711` 的 `backend_tokens` 选择与上游不同（**新发现的隐患**）

- 我们（`:8711-8712`）：`materialize_sequence_kv(sequence, end, speculative_backend == SpeculativeBackend::None ? 0U : end);`
- 上游（改前）：`materialize_sequence_kv(sequence, end, backend_kv_cache() ? end : 0U);`（`git show 03177b9` 的 `:8916` hunk）

差别：我们按「**是不是投机后端**」判断要不要 backend KV，上游按「**这个后端到底有没有 backend cache**」判断。
而 `program_impl.h:10322-10324` 的包装体里有
`if (backend_tokens != 0 && !sequence.kv->backend) throw std::logic_error("backend KV materialization requested without an allocation");`。
⇒ **若 `append_forced_tokens` 在 DFlash2（无 backend KV，`:12407`、且 `:11381` 把 append 限定在 DFlash）
之下被调用，我们会在这一行抛 `logic_error`，上游不会。**
**可达性：待证实**（需要查 `append_forced_tokens` 的全部调用方是否允许 DFlash2；本会话未做，
但它与本 diff 无关——这条在打补丁前后完全一样）。

### 4.3 `start_sequence:9699` 与 `decode_dflash_batch:12180` 的 backend 参数措辞不同

- `:12180` 我们传 `frontier`，上游传 `backend_kv_cache() ? frontier : 0U`。DFlash 下两者等价
  （DFlash 必有 full cache），差异只在「无 cache」时的报错路径，同 §4.2 家族。
- 这些都不改变本 diff 的安全性（我们只改名）。

### 4.4 经验证据：老抛错在我们全部日志里**从未出现过**

对工作目录下所有 `*.log/*.txt/*.out/*.md` 做全文检索，
`"KV materialization exceeds active entitlement"` 只出现在 M 的简报 `_collab/M_upstream_borrow.md:17` 里
（即被引用，不是被触发）。这与 §2.2「唯一可达点是 W4 且条件很窄」一致：
page 边界 + 终止 + 只接受一列的组合在实测里尚未被触发。**因此本 diff 是预防性的**，
不是「重现已知崩点后再修」。

---

## 5. 证明：diff 能落

### 5.1 `patch -p1 --dry-run` 输出（在镜像根目录）

```
$ cd /c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
$ patch -p1 --dry-run < /c/Users/User/Documents/ziqinzhang/_collab/E7_s50_kv_coverage.diff
checking file src/targets/qwen3_6/impl/runtime/logical_kv_store.h
checking file src/targets/qwen3_6/impl/runtime/program.h
checking file src/targets/qwen3_6/impl/runtime/program_impl.h
checking file tests/targets/qwen3_6/test_context_store.cpp
RC=0

$ patch -p1 --dry-run --forward --fuzz=0 < .../E7_s50_kv_coverage.diff
checking file src/targets/…(同上 4 行)
RC=0
```

即 **rc=0，且 `--fuzz=0` 也 rc=0**（无模糊匹配、无 offset）。

**M 在 build tree 上要跑的命令**（board S50 行里也是这条）：

```bash
wsl.exe -e bash -c "cd /home/user/ninfer-fusion && patch -p1 --dry-run --forward --fuzz=0 < /mnt/c/Users/User/Documents/ziqinzhang/_collab/E7_s50_kv_coverage.diff"
# 期望 exit 0
```

hunk 头一览（打补丁**前**的行号）：

```
@@ -10,6 +10,7 @@        logical_kv_store.h   #include <string>
@@ -1491,14 +1492,21 @@  logical_kv_store.h   本体
@@ -1279,8 +1279,8 @@    program.h            声明
@@ -1165,7 +1165,7 @@     program_impl.h  T1
@@ -8146,7 +8146,7 @@     program_impl.h  W1
@@ -8708,7 +8708,7 @@     program_impl.h  W2
@@ -9696,7 +9696,7 @@     program_impl.h  W3
@@ -10317,18 +10317,18 @@ program_impl.h  定义 + T2 + T3
@@ -10849,7 +10849,7 @@   program_impl.h  T4
@@ -11305,7 +11305,7 @@   program_impl.h  T5
@@ -11413,7 +11413,7 @@   program_impl.h  W4
@@ -11831,7 +11831,7 @@   program_impl.h  W5
@@ -11990,7 +11990,7 @@   program_impl.h  W6
@@ -12177,7 +12177,7 @@   program_impl.h  W7
@@ -12412,7 +12412,7 @@   program_impl.h  W8
@@ -184,7 +184,7 @@       test_context_store.cpp
… (11 hunk，共 12 处改名)
```

### 5.2 非投机路径未被触碰 —— 机械证明

做法：把两侧文本里的两个标识符都替换成同一个占位符（`materialize_to_tokens`/`ensure_mapped_to_tokens`→`TOK`；
`materialize_sequence_kv`/`ensure_sequence_kv_mapped`→`SEQ`），然后用 `diff -u -w -B` 比较
（`-w` 忽略全部空白，`-B` 忽略空行）——**空白差异正是改名后的续行重排，本就不该算语义变化**。

```
=========== program.h ===========
(rc=0  除标识符外逐字相同)
=========== program_impl.h ===========
(rc=0  除标识符外逐字相同)
=========== tests/targets/qwen3_6/test_context_store.cpp ===========
(rc=0  除标识符外逐字相同)
=========== logical_kv_store.h ===========  ← 只有这一处有实质差异
@@ -1491,14 +1492,21 @@
-        if (target < address.page_count || target > entitlement(address)) {
-            throw std::invalid_argument("KV materialization exceeds active entitlement");
+        if (target > entitlement(address)) {
+            throw std::invalid_argument(
+                "KV coverage exceeds active entitlement: tokens=" + std::to_string(tokens) +
+                " required_pages=" + std::to_string(target) +
+                " mapped_pages=" + std::to_string(address.page_count) +
+                " reserved_pages=" + std::to_string(address.reservation.pages()) +
+                " entitlement=" + std::to_string(entitlement(address)));
         }
-        if (target == address.page_count) { return; }
+        if (target <= address.page_count) { return; }
```

再把每个文件的**调用点实参**抽出来对比（把标识符归一化后逐行 diff）：

```
--- program_impl.h ---
  [BEFORE]                                [AFTER]
  1 text_kv_addresses->TOK(*address, predictor_count, device.stream);
  2 SEQ(sequence, prefill.prompt_tokens, backend_materialized);
  3 SEQ(sequence, end,
  4 SEQ(sequence, prompt_tokens, backend_materialized);
  5 void ProgramImplCore::SEQ(SequenceState& sequence, std::uint32_t main_tokens,
  6 text_kv_addresses->TOK(sequence.kv->text, main_tokens, device.stream);
  7 backend_kv_addresses->TOK(*sequence.kv->backend, backend_tokens,
  8 addresses.TOK(*allocation, 1, device.stream);
  9 text_kv_addresses->TOK(*allocation, 1, device.stream);
 10 SEQ(sequence, std::max(sequence.text_kv_valid, end), end);
 11 SEQ(sequence, frontier + 1, 0);
 12 SEQ(sequence, frontier + extent + 1,
 13 SEQ(sequence, frontier + extent + 1U, frontier);
 14 SEQ(sequence, frontier + extent + 1U, 0);
  => ARGUMENTS IDENTICAL (only the identifier differs)
--- program.h ---      => ARGUMENTS IDENTICAL
--- test_context_store.cpp (12 行) => ARGUMENTS IDENTICAL
```

**「`target <= mapped` 只可能出现在投机/prefill 路径」的论证**：

1. `target <= page_count` 要成立，必须存在一次**更早的、更长的**覆盖请求且其后**没有被 trim**。
   而我们树里所有把 `page_count` 往下打的路径都只有 `destructive_truncate`
   （`logical_kv_store.h:1548`，以及 inactive 版 `:1610`），它在运行时只由
   `trim_sequence_kv`（`program_impl.h:10347`）调用，调用点是
   `:7944`（capture 释放）、`:8801`（append_forced 结尾）、`:9629/:9677/:9690`（start_sequence 重绑）、
   `:9985`（投机结算尾裁）、`:12556`（非投机 pending 结算）——**全部在「一轮结束」处**。
2. 于是「一轮中间的 `page_count`」只会被 `ensure_mapped_to_tokens` 单调抬高；
   轮首的 `page_count` 恰好等于**已提交 frontier** 的页数。
3. 因此 `target < page_count` 只可能来自「**同一轮内、在更长的请求之后、又发一个更短的请求**」。
   全树里唯一这种形态就是 W4（verify 先要 `base_E+extent+1`，同轮的终止 append 只要 `base_E+accepted`），
   而它**只在 DFlash 的投机终止结算里**出现（`:9903` + `:9917`）。
4. 非投机路径（`decode_ordinary_batch:11834`、`resolve_non_speculative_pending:12556`、
   prefill/`start_sequence:9699`）每轮只发**一次**覆盖请求，且请求值 ≥ 轮首页数 ⇒ 恒为增长或相等，
   **在改前与改后行为完全相同**。

---

## 6. 回归测试草图（已按我们的 tests 布局落地并 dry-run）

上游的 `tests/targets/qwen3_6/speculative_page_boundary.h`（64 行）是 **engine 级**用例，
依赖 `ninfer/engine.h`、`engine.tokenize_text`、`generated_token_ids`、
`stopped.speculative.rounds/drafted_tokens`，并且接在
`tests/targets/qwen3_6_27b/test_engine_dflash2_real.cpp` 与 `.../qwen3_6_35b_a3b/test_engine_dflash_real.cpp`
里（需要真 checkpoint）。**这两个文件我们树里都不存在**（我们有
`tests/targets/qwen3_6_35b_a3b/test_engine_dflash_real.cpp`，但没有 27b 的 dflash2 用例；
`grep -rl dflash2 tests/` 为空），所以**不照抄**。

我们树里能**逐字落地**的部分是上游同一个 commit 里那段 **store 级**用例：
`tests/targets/qwen3_6/test_context_store.cpp` 的 `test_kv_store`。
已核对：我们该函数体（`:158-581`）与上游改动前的版本**逐字节相同**（`diff` 只在我人为截取的边界上不同），
所以上游那段插入可以原样搬，插入点是 `:317`（`"KV release invalidates generations and closes physical ownership");`）
与 `:319`（`const auto snapshot_source`）之间的空行之后。

**建议作为独立补丁**（不在 `E7_s50_kv_coverage.diff` 里，避免与机械改名混在一起）：

```diff
--- a/tests/targets/qwen3_6/test_context_store.cpp
+++ b/tests/targets/qwen3_6/test_context_store.cpp
@@ -316,6 +316,42 @@
                physical_pages.allocated_pages() == 0 && physical_pages.reserved_pages() == 0,
            "KV release invalidates generations and closes physical ownership");
 
+    // Verify crosses a page boundary, but the terminal commit consumes only its first column.
+    const auto terminal = addresses.create_active(3, 0);
+    expect(terminal.has_value(), "terminal boundary KV address allocation");
+    addresses.ensure_mapped_to_tokens(*terminal, 63, device.stream);
+    addresses.commit_frontier(*terminal, 63);
+    addresses.ensure_mapped_to_tokens(*terminal, 71, device.stream);
+    device.synchronize();
+    const auto terminal_mapping   = read_block_table(physical_tables, 0, 2);
+    const auto unchanged_terminal = [&] {
+        return addresses.mapped_pages(*terminal) == 2 &&
+               addresses.committed_frontier(*terminal) == 63 &&
+               addresses.entitlement(*terminal) == 3 && physical_pages.allocated_pages() == 2 &&
+               physical_pages.reserved_pages() == 1 && physical_pages.available_pages() == 5 &&
+               read_block_table(physical_tables, 0, 2) == terminal_mapping;
+    };
+    addresses.ensure_mapped_to_tokens(*terminal, 71, device.stream);
+    addresses.ensure_mapped_to_tokens(*terminal, 64, device.stream);
+    device.synchronize();
+    expect(unchanged_terminal(), "covered KV requests preserve speculative mappings and ownership");
+    bool exceeded = false;
+    try {
+        addresses.ensure_mapped_to_tokens(*terminal, 193, device.stream);
+    } catch (const std::invalid_argument&) { exceeded = true; }
+    expect(exceeded && unchanged_terminal(),
+           "KV coverage beyond entitlement fails without mutation");
+    addresses.commit_frontier(*terminal, 64);
+    addresses.destructive_truncate(*terminal, 64);
+    expect(addresses.mapped_pages(*terminal) == 1 &&
+               addresses.committed_frontier(*terminal) == 64 &&
+               addresses.entitlement(*terminal) == 3 && physical_pages.allocated_pages() == 1 &&
+               physical_pages.reserved_pages() == 2 && physical_pages.available_pages() == 5,
+           "terminal trim returns the uncommitted page to the same active reservation");
+    addresses.deactivate(*terminal);
+    expect(addresses.release(*terminal) && physical_pages.allocated_pages() == 0 &&
+               physical_pages.reserved_pages() == 0 && physical_pages.available_pages() == 8,
+           "terminal settlement releases both mappings and unused growth");
     const auto snapshot_source      = addresses.create_active(3, 0);
     const auto snapshot_destination = addresses.create_inactive();
     expect(snapshot_source && snapshot_destination, "active KV snapshot endpoints allocate");
```

（上面这个 fenced 块是从 `_collab/E7_s50_regression_sketch.diff` **逐字节**贴出来的
（单 hunk `@@ -316,6 +316,42 @@`，+36 行），该 `.diff` 已单独
`patch -p1 --dry-run --forward --fuzz=0` 验证 **rc=0**，见下面的 dry-run。
⚠️ 落补丁请**直接用 `_collab/E7_s50_regression_sketch.diff` 文件**，不要从 md 里复制：
diff 的第 5 行是一条只有一个空格的上下文行（内容为空的源行），markdown 编辑器/格式化器
很可能把它吃成零宽空行，那样 `patch` 仍能接受但语义上不再是「上下文」；文件版本没有这个风险。）

这段用例**恰好覆盖 M 要的三件事**：

| 断言 | 覆盖的语义 |
|---|---|
| `ensure_mapped_to_tokens(*terminal, 64)`（`63 → 71 → 64`：先扩到 2 页，再回缩到 1 页） | **收缩不得抛**（老码：`pages_for_tokens(64)=1 < page_count=2` ⇒ `invalid_argument`） |
| `unchanged_terminal()` 里的 `mapped_pages()==2` + `committed_frontier()==63` + `entitlement()==3` + `allocated/reserved/available == 2/1/5` + `read_block_table(...)` 与 `terminal_mapping` 逐位相同 | **收缩不得 unmap**：映射、保留量、物理块表三者都要原封不动 |
| `193 > entitlement(3)` 仍须抛 `invalid_argument` 且 `unchanged_terminal()` | **上界语义不得被削弱**（加富消息之后仍必须是 `std::invalid_argument`） |
| `commit_frontier(64)` + `destructive_truncate(64)` 后 `1/3`、`reserved==2`、`available==5` → `deactivate`+`release` 后 `0/0/8` | **显式 truncate 才释放**，且尾页回归同一 active reservation |

**dry-run（同样在镜像上跑，rc=0、无 fuzz）**：

```
$ cd /c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
$ patch -p1 --dry-run --forward --fuzz=0 < /c/Users/User/Documents/ziqinzhang/_collab/E7_s50_regression_sketch.diff
checking file tests/targets/qwen3_6/test_context_store.cpp
RC=0
```

> 注意：这段用例**依赖 `E7_s50_kv_coverage.diff` 已经落**（它调用的是 `ensure_mapped_to_tokens`）。
> 两个补丁一起落、或者先落主补丁再落这个。

**engine 级那条（可选）**：若 M 想要上游 `speculative_page_boundary.h` 那种端到端门，
可行的最小适配是在我们的 `tests/targets/qwen3_6_35b_a3b/test_engine_dflash_real.cpp`
（存在）里加一个同构 fixture，触发条件照抄上游：
让 prompt 末端落在 token 63（`prompt.insert(begin, 63 - len, 198)`）、
用 `stop.token_ids` 停在跨页 verify 的第一列、断言
`reused_prompt_tokens == 64` 与后续生成可完成。
**本会话未落地**（我没有上游那种 27b dflash2 用例可挂，且 `dflash2` 在我们的 tests 里零出现）；
其触发条件与 §2.1 W4 的 `base_E=63, extent≈8, accepted=1` 是同一个算术，
**待证实**的是我们的 `--draft-tokens` 是否允许 extent 跨过 64。

---

## 7. 残留不确定 / 待证实清单

1. **镜像 ≠ build tree 的风险**：本会话无法读 `/home/user/ninfer-fusion`。若两个文件在该区域已分叉，
   dry-run 会在 build tree 上失败（M 的那条命令会报 `Hunk #N FAILED`），**不会静默错改**。
2. **`start_sequence` 的 `preserving_source` 分支**（`:9550`，`:9690` 被跳过）是否真的能让
   `page_count > pages_for_tokens(prompt_tokens)`：**待证实**（需要运行期探针；
   文本上是「不排除」，实测上「日志里从未出现该抛错」）。
3. **§4.2 的 `:8711` backend 参数**是否能在 DFlash2 下被触发：**待证实**（需要查
   `append_forced_tokens` 的调用方白名单）。
4. **§4.1 上游删掉 Main KV 请求**是否要在下一趟一并落：**建议是**，但它属于语义收窄，
   需要 M 决策（`_collab/M_upstream_borrow.md` §3 的编排里没有它）。
5. **本 diff 未编译**（硬约束：不跑 nvcc/ptxas/make）。语法层面做了人工核对：
   新增的 `std::to_string` 需要 `<string>`（已加，`:13`）；
   `address.reservation.pages()` 是 `const noexcept`（`src/core/paged_kv_cache.h:220`），
   在函数内以非 const `Address&` 访问，无 const 问题；`entitlement(const Address&)` 是同类的
   `private` 成员（`logical_kv_store.h:1889`），类内可见。
