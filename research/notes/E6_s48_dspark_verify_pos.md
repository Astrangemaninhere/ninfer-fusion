# S48 · DSpark(DFlash v1) verify 位置表少一列 — 补丁 + 报告

交付物：`_collab/E6_s48_dspark_verify_pos.diff`（1 hunk / +13 / −2，只动 `src/targets/qwen3_6/impl/runtime/dflash_impl.h`）
生成器 `_collab/E6_s48_mkpatch.py`，自检 `_collab/E6_s48_syntax_check.py`。
本回合 **纯 CPU**：只读源码 + python + `diff`/`patch --dry-run`；**未写任何 `src/` 文件**（build tree 与 mirror 的 `dflash_impl.h` 在回合前后 md5 都是 `60a51d1c69c73da6117ec55a858549a6`）；**未跑 nvcc/ptxas/make**。

---

## 0. 一句话结论

DSpark 的 verify 块把自己要用的位置表按 **`V=k`** 建（`dflash_impl.h:223-231`），而它自己的契约是
**`target_valid_columns = extent+1 = k+1`**（`program_impl.h:12170`）⇒ 第 k+1 列（列 k，携带 `drafts[k-1]`）
拿到的是 **`frontier+k-1`**，与列 k-1 同址。补丁把这张表的 `V` 改成 `width=k+1`（列 0..k-1 逐字节不变，
列 k 变成 `frontier+k`），随后把 `attention_valid` 复位成 `k`，**只**喂给草稿块自己的注意力 ⇒
草稿块输出逐位不变，verify 表与 MTP/DFlash2 同形。

---

## 1. 缺陷：before / after（k=7、frontier=F，即今天 27B DSpark 的配置）

`dflash_impl.h:204-231`（`propose_batch_impl`）里 `width = k+1`（`:205`），而给 `prepare_masked_block`
的 valid-columns 在 bf16(DSpark) 分支被写死成 `k`（`:226`）。该 op 的公式是
`positions[i] = lengths[b] + min(i, V[b]-1)`，`1 <= V <= W`（`include/ninfer/ops/prepare_masked_block.h:12-27`），
`W = frame.proposal_positions.ne[0] = draft_window+1 = k+1`（`round_state.cpp:118-120` + `:187-188`）。

| 物理列 j | 0 | 1 | … | k-1 | k |
|---|---|---|---|---|---|
| 输入 token（`speculative_prepare_verify_ids`，`speculative_round.h:18-42`） | anchor | drafts[0] | … | drafts[k-2] | drafts[k-1] |
| **现在** `V=k`：`positions[j]` | F | F+1 | … | F+k-1 | **F+k-1（重复）** |
| **修后** `V=width`：`positions[j]` | F | F+1 | … | F+k-1 | **F+k** |
| 参照 MTP `Pcur=k`（`program_impl.h:11979-11982`、`speculative_round.h:26`） | F | F+1 | … | F+k-1 | F+k |
| 参照 DFlash2 `V=width`（`dflash2_impl.h:172,186-193`） | F | F+1 | … | F+k-1 | F+k |

同一张表被 verify 直接当 `cache_positions` 与 `rope_positions` 用两次：`dflash_impl.h:522`
（`target_positions = frame.proposal_positions.slice(1,0,batch_size)`）→ `:556-557`
（`.cache_positions = target_positions, .rope_positions = target_positions`）。
三处消费者穷举（整树 grep）：`dflash_impl.h:215/522`、`dflash2_impl.h:179/377`、`round_state.cpp:187/351`
—— 没有别的读者，也没有 test/app 直接读它。

**补丁文本（`git apply`/`patch -p1` 均可用，CRLF 保真）**

```diff
-            ops::set_i32_scalar(attention_valid, static_cast<std::int32_t>(k),
-                                state.execution.device.stream);
+            // DSpark keeps exactly k live draft columns, but this table is also the
+            // target verify's cache/rope position table, and that contract is
+            // target_valid_columns = extent + 1 = width live columns. Build the table
+            // at width so its last column sits at frontier + k instead of repeating
+            // the position of column k-1 (which also aliases its KV cache slot).
+            ops::set_i32_scalar(attention_valid, width, state.execution.device.stream);
         }
 
         ops::prepare_masked_block(anchors, frontiers, attention_valid, Config::mask_token, ids,
                                   positions, state.execution.device.stream);
+        if constexpr (Config::bf16_weights) {
+            // Restore the draft block's own k live columns for the attention below;
+            // columns 0..k-1 of the table already hold frontier+0..frontier+k-1 and are
+            // not affected by the width above.
+            ops::set_i32_scalar(attention_valid, static_cast<std::int32_t>(k),
+                                state.execution.device.stream);
+        }
```

两处编辑**都在 `if constexpr (Config::bf16_weights)` 内**（`:224-228` 原有分支 + 新增分支），
所以 W8 legacy DFlash（`qwen3_6_35b_a3b/impl/config.h:73` `bf16_weights=false`）与
Muse（`muse_glimmer_30b/impl/config.h:135` `supported=false`）**一个字符都不变**。
27B 的 DSpark 是唯一 `bf16_weights=true` 的 DFlash draft（`qwen3_6_27b/impl/config.h:76`）。

---

## 2. 为什么"前 k 列逐字节不变"（补丁只加多出来的那一列）

1. **表值**：`V=k+1` ⇒ `positions[i]=F+min(i,k)=F+i`（i=0..k）；`V=k` ⇒ `F+i`（i<k）+ `F+k-1`（i=k）。
   两者在 **i=0..k-1 上完全相同**，与 `extent` 是否等于 k 无关（`width` 是编译期/图捕获期的常数，
   不是逐行的 `valid_columns`）—— 这正是任务书要的"draft 块与 verify 块在前 k 列上一致"。
2. **谁读它**：草稿块只在 `ops::rope(positions.view({columns}), ...)`（`:293`）里读表。
   列 k 那一行的 rope 角从 `F+k-1` 变成 `F+k`，但**列 k 在草稿块里是惰性尾列**：
   `attention_valid` 在层循环里仍是 `k`（补丁第二个 `set_i32_scalar` 复位），
   `bidirectional_gqa_attention` 的契约是"keys = context[0,L) + live query rows [0,V)，i>=V 为惰性尾、
   输出置零"（`include/ninfer/ops/bidirectional_gqa_attention.h:28-47`）⇒ 列 k 不是 key、输出为 0、
   `delta` 行 k 恒 0、`residual` 行 k 不变；而 `packed` 在 bf16 分支只拷 `residual` 的**前 k 列**
   （`source_column_offset=0`、`row_bytes = hidden*k*2`、`source_pitch = hidden*width*2`，`:428-444`）
   ⇒ 尾列永远到不了 proposal head。
3. **复位顺序安全**：两次 `set_i32_scalar` 与 `prepare_masked_block`、层循环全在
   `state.execution.device.stream` 上，按入队序执行 ⇒ 建表时读到 `width`，注意力读到 `k`。
   CUDA Graph 捕获同一路径（`:612-626` 共用 `dflash_decode_batch_body`），捕获的也是同一个 kernel 序列。
4. ⇒ 同一 (anchor, context) 下 **k 个 draft token 与 `target_tokens[0..k-1]` 逐位不变**（见 §6 的证伪线）。

---

## 3. 列 k 现在做什么 vs 修后做什么

要判定的语义（同一 kernel，`src/ops/kernel/speculative_round.cuh:131-152` 快速路径与 `:181-216` 采样路径
逐字一致）：贪心下 `a = 最长满足 row_targets[i]==row_drafts[i] 的前缀 (i<extent)`，`t_star = row_targets[a]`；
即 **`target_tokens[m]` 判定 `drafts[m]`**，而 `t_star` 落在列 `a`。

- **现在（列 k）**：查询 token = `drafts[k-1]`，但它的 rope 角、它的自身 KV 槽、它的因果上界
  用的都是 `F+k-1`（`small_t_bf16.cuh` 里 `qabs = pos[token]`、`key <= qabs`，`:362`、`:375-386`）；
  它的自身行在 key 下标 `F+k` 上，所以**看不见自己**（自注意力被砍掉）。真正的 `F+k` 那一格没人写。
  这是一对"序列里不存在的 (token, position)"。
- **修后（列 k）**：等同 MTP `Pcur=k` —— 查询 token `drafts[k-1]` 在 `F+k`，因果上界 `F+k`（含自身行），
  cache 槽 `F+k` 正好是"整块被接受时 `drafts[k-1]` 所在的位置"。
- **可见窗口**：核里 `window = pos[tokens-1]+1`（`small_t_bf16.cuh:119-126`）现在少一格（`F+k` vs `F+k+1`）。
  修后需要的上界 `F+extent+1` 已经在 ingress 侧备好：`target_envelope{1, maximum_target_tokens}`
  且 `maximum_target_tokens = max(..., frontier+extent+1)`（`program_impl.h:12130-12131`、`:12141`；
  图档分支 `:12149-12152` 用 `min(capacity, profile.max_execution_frontier+draft_window+1)`），
  KV 物化也是按 `frontier+extent+1` 铺的（`:12180`）⇒ **补丁不需要改任何 envelope/容量**，
  反而是这两处 ingress 声明本来就在暗示"最后一路应该在 `F+extent`"。
- **谁消费列 k 的 logits**：只有"该轮全部草稿都被接受（`a == extent == k`）"时的 `t_star`
  （bonus/correction token，`:135/:139`、采样路径 `:205-224`）；其余情况 `a ≤ extent`，列 k 的 logits
  不进任何判定。**列 k 的 KV 写则是每轮都发生**（见 §4）。

---

## 4. KV 双写后果（写两次的是哪个槽、写进去的是什么值）

`ops::gqa_attention`（A1）既把本块 K/V 写进分页 cache、又做因果注意力；契约明写
"valid 前缀内 positions 必须 sequential"、"overwrites every addressed cache row"、
"不修改 invalid 列"（`include/ninfer/ops/gqa_attention.h:62-89`，尤其 `:79-85`）。
落盘循环是**按列**的：`for (chunk...) { token = chunk/(D/8); p_tok = pos[token]; ... store_vec(&cache_k[cache_off], ...input.k[new_off]); }`
（`src/ops/softmax_attention/dense/causal_cache/small_t_bf16.cuh:150-168`）。

- **被写两次的槽 = `F+k-1`**：列 k-1 写入 `drafts[k-2]` 的 K/V，列 k 写入 `drafts[k-1]` 的 K/V。
  两列同属一个 owning split（同一 CTA），但 op 没有任何顺序保证（重复 position 本来就在它声明域外）
  ⇒ **存活者是调度顺序的副产物**，可能是 `drafts[k-2]` 也可能是 `drafts[k-1]`。
- **`F+k` 这一槽本轮无任何写入**，而只要 `a=k`，`drafts[k-1]` 就会被提交在 `F+k`。
- **本轮自身不受污染**：本块的 K/V 不是从 cache 读的，而是按 key 下标从在飞的 `input` 读
  （`from_new = key>=first_pos && new_token<valid_tokens` → `cp_async(&input.k[new_off])`，`:240-249`）
  ⇒ 列 0..k-1 的 logits 与 cache 里的撞写无关，**E5 §1.3 里"最后一路不可能被接受"的推断不成立**（见 §7）。
- **何时会真的留下伤痕**：`a ≥ k-1` 才可能（下一轮块从 `F' = F+a+1` 起，`a ≤ k-2` 时 `F' ≤ F+k-1`，
  撞写槽会在下一轮被正常重写）。`a = k-1` ⇒ 已提交槽 `F+k-1` 可能永久存着错的 K/V（下一轮从 `F+k` 起，
  不会再写它）；`a = k` ⇒ `F+k` 槽永远不会被后来的轮次写入（下一轮从 `F+k+1` 起）⇒ 永久陈旧 KV。
- **今天的出现概率**：需要一轮接受 ≥ k-1 = 6 个草稿；在 10.3% / EAL 1.38（`/home/user/s4w_dspark.log`）下
  基本不可能 ⇒ **这条不是 10.3% 的成因**，而是"H1 修好、接受率上来之后"的正确性前置（与 E5 的收尾判断同向，
  但机制与归属要按 §7 更正）。

---

## 5. 交付物与 dry-run（逐字）

`patch -p1 --dry-run`（cwd = `/home/user/ninfer-fusion`，输入 `_collab/E6_s48_dspark_verify_pos.diff`）：

```
checking file src/targets/qwen3_6/impl/runtime/dflash_impl.h
rc=0
```

（`rc` 为命令退出码；`patch` 无 `-N/-R`，`--dry-run` 不落盘：执行前后
`md5sum src/targets/qwen3_6/impl/runtime/dflash_impl.h` 都是 `60a51d1c69c73da6117ec55a858549a6`。）

补丁规模：`hunks=1 added=13 removed=2 bytes=1924`；正文行 CRLF（与源文件一致，`file` 报
"with CRLF line terminators"；diff 头三行 `---`/`+++`/`@@` 为 LF）。最强的一行 ≤ 88 列（`.clang-format: ColumnLimit 100`）。
另做了两项独立核对：
① 把补丁 apply 到 `/tmp/e6/apply` 的副本（不动 build tree）→ `apply_rc=0`，且结果与生成器写的
`/tmp/e6/b/...` **逐字节相同**（`diff` 空）；
② 补丁后文件括号平衡（`{}` 115/115、`()` 316/316、`[]` 14/14；补丁前 114/114、314/314、14/14），
变更行 15 行且全部落在 `propose_batch_impl` 内（`E6_s48_syntax_check.py` 输出）。

---

## 6. 可证伪预测

| 观测量 | 预测 | 依据 / 怎么测 |
|---|---|---|
| 每轮 **draft token**（`drafted tokens`、草稿本身） | **不动**（逐位相同） | §2：表前 k 列不变 + 尾列不进 `packed`；`apps/cli` 的 `dflash drafted tokens` |
| `target_tokens[0..k-1]`（即 `drafts[m]` 的判定列） | **不动** | 列 0..k-1 的 position/cache 槽/因果窗都不变（§2、§3） |
| `p_0`（位置 0 的条件接受率，`accepted by pos` 首项比值） | **不动** | 草稿块首 token 逐位不变；这是最容易被误当成"修好了"的一项，必须作为负对照 |
| 整场 token 流 + `accepted by pos` | **只要整场没有任何一轮 `a=k`，必须逐字节相同**（greedy ⇒ 确定性） | 若不同 ⇒ 补丁扰动了前 k 列 ⇒ 判定补丁错，回退 |
| **`a=k` 那一轮的 bonus token**（`t_star`） | **变**：从"`F+k-1` 角度下的 argmax"变成"`F+k` 角度下的 argmax" | 只在整轮全接受时出现；`--max-new` 较大时表现为该轮之后 token 流分叉 |
| KV 槽映射 | **变**：`F+k-1` 不再被写两次、`F+k` 被写 | §4；可用 `NINFER_KVDUMP_*`（`text_context_impl.h:973-993` 的 dump 机制）看 `pos` 表与槽位 |
| MTP / DFlash2 / W8-DFlash 的所有数字 | **不动** | 补丁在 `if constexpr (Config::bf16_weights)` 内，三者在 `dflash_impl.h` 走另一支 |
| **最敏感的探针**：`--spec dflash --draft-tokens 1` | 一旦那一路草稿被接受（`a=1=k`），**bonus 必然来自被污染的列 1**（现在位置 `F`，应为 `F+1`）⇒ 修前修后 token 不同 | `k=1` 时 `width=2`，列 1 = 唯一的草稿列；历史记录 `dflash k=1` 接受率 0%（`DSPARK-ADAPTATION.md:151-156`），所以先看 `accepted by pos` 有没有非零 |
| `accepted by pos` 的最后一位 | **不是**本 bug 的"结构指纹"（§7）；它是"两种列约定"的判别器：若修后最后一位显著跳升，说明真实约定其实是"列 j 判定 drafts[j-1]"（§8 case B） | `apps/cli/main.cpp:238-245` 打印，`program_impl.h:12237-12239`（`accepted_per_position[i]`，数组长度 = `draft_window`，`:11346-11347`） |

**验收建议（落批后）**：同 prompt 各跑 dspark `--draft-tokens 1` / `3` / `7` 三臂 greedy，先比对
`accepted by pos` 全 0 差分；若某臂出现 `a=k` 轮，再比对 token 流，应当**只**从该轮之后分叉。

---

## 7. 对 E5 §1.3 的两处更正（本回合读码得到，供 M 归档）

1. **"第 k 路草稿（drafts[k-1]）是在和上一列同一位置上被验证的 ⇒ 这一路不可能被接受"** —— 不成立。
   判定关系是 `target_tokens[m] ↔ drafts[m]`（`speculative_round.cuh:134` 的
   `while (a < extent && row_targets[a] == row_drafts[a])`，以及 `:207` 的 `selected == row_drafts[i]`），
   所以 `drafts[k-1]` 由**列 k-1** 判定，而列 k-1 的 position（`F+k-1`）本来就是对的。
   被写坏的是**列 k**，它只在 `a=k`（全接受）时提供 bonus（`t_star = row_targets[a]`）。
   `accepted by pos` 第 m 位就是 `drafts[m]`（`program_impl.h:12237-12239` + `:11346-11347`），
   因此"最后一位恒 0 ⇒ 位置表 bug 回归"这条断言**不能**当回归指纹用。
2. **"最后一路被重复位置砍掉"的机制**要改写成：列 k 的 (token, position) 组合不存在于序列中
   （角度少 1 + 自身行被因果窗排除）+ 它的 KV 与列 k-1 撞同一个 cache 槽（§4）。
   影响面因此小得多（只在 `a=k` 轮可见），这解释了为什么它在 10.3% 的盘面上看不出来。

---

## 8. 残余风险（如果我关于"契约"的假设错了）

我依赖三条互相独立的证据认定"列 j 判定 `drafts[j]`、最后一路在 `F+k`"：
(i) accept 核的索引对齐（`speculative_round.cuh:134,207`）；
(ii) 下一轮 anchor 的位置恒等式 —— `t_star` 来自列 `a`，下一轮 frontier 是 `F+a+1`，
     只有"列 a 的输入（`drafts[a-1]`）在 `F+a`、它的 logits 预测 `F+a+1`"才自洽（`program_impl.h:12164` 的
     `anchors = ledger.back()` 与 `:12248` 的 `produced = count_i = a+1`）；
(iii) MTP（42.6% 接受的健康对照）与 DFlash2 走同一个 op、同一套列约定，且它们的表都是 `F+0..F+k`
     （`program_impl.h:11979-11982`、`dflash2_impl.h:186-193`）。

若真实约定是 **case B**（"列 j 判定 `drafts[j-1]`"），那么：修好后受影响的就不是 bonus 而是
**最后一路草稿的接受**，`accepted by pos` 的最后一位会明显跳升 —— 这正好是可判别的（§6 末行）；
此时本补丁本身仍然无害（表变成严格递增 = MTP/DFlash2 的形状），但缺陷叙述与归因需要重写。
其他残余风险：
- DSpark 草稿 rope 用纯 `rope_theta` 而 checkpoint 声明 YaRN、verify 不加 `sequence.rope_delta`
  （E5 §1.3 脚注，MTP 在 `program_impl.h:11982` 加）—— 本补丁**不碰**这一层，长上下文/`--yarn` 下的角度误差仍在。
- 表尾 `V=width` 而非逐行 `valid_columns`：`extent<k` 的轮次里，**invalid 尾列**（j>extent）的位置是
  `F+j`（严格递增）而不是像 `prepare_masked_block` 默认那样 clamp 在 `F+extent`。它们既不写 KV
  也不参与数学（`gqa_attention.h:82` 明写不改 invalid 列），且 DFlash2 就是这个姿态（`V=width`）——
  如果 reviewer 更偏好"与 MTP 逐位相同"，把第二个 `set_i32_scalar` 的值换成逐行 `valid_columns`
  （`frame.target_valid_columns`）即可，代价是 `extent<k` 时前 k 列不再保证逐字节一致（那种写法下
  草稿块的 rope 会读到 clamp 过的尾行）。
- 我**没有**新增 workspace 分配（刻意避免动 `layouts_impl.h:622` 的 `dflash_round` 计划）；若改用
  "新分配一张 `width` 张量"的写法，需要复核 arena 余量（`attention_valid` 本身就不在计划里，
  说明有余量，但余量多少**未测**）。

---

## 9. 未能证实 / 本回合没做

- **没有编译、没有运行**（硬约束）：C++ 合法性只有 dry-run + 括号/列宽/锚点唯一性 + 副本 apply 复现
  这几项静态证据；`nvcc -fsyntax-only` 级别的确认 **待证实**。
- **`a ≥ k-1` 轮在今天的 DSpark 上是否真的出现过**：需要 `accepted by pos` 剖面（`--spec dflash --draft-tokens 7`），
  未跑 GPU ⇒ **待证实**；因此"这条不是 10.3% 的成因"是**推断**（依据 EAL 1.38 的乘积），不是实测。
- **撞写槽的存活者是谁**：代码只给出"同一 CTA、无顺序保证"，具体由硬件/调度决定 ⇒ **待证实**（可实测：
  dump cache 槽 `F+k-1` 的 K/V 与 `drafts[k-2]`/`drafts[k-1]` 的 K/V 比对）。
- **`sequence.rope_delta` 在未开 `--yarn` 时为 0**：沿用 E5 的"待证实"，本回合未重推（`program_impl.h:9701`）。
- **`target_tokens[k]` 的 argmax 在"对的位置"上是否会不同于"错的位置"**：由模型决定，未跑 GPU ⇒ 待实测
  （预测是"通常不同"，因为角度差一个位置、且自身行从因果窗里被排除）。
- 未纳入：SVIP 截断（`dspark_markov_argmax` 写回 `proposal_extents`）与 `valid_columns` 的口径差
  （E5 §2 已记录：DFlash 用 egress extent 计数、ingress `target_valid_columns` 不随之收缩）—— 本补丁不改这个口径差，
  只在表尾把"多出来的那一列"摆正。

---

## 10. 一条复现命令

```bash
wsl.exe -e bash -c "cd /home/user/ninfer-fusion && patch -p1 --dry-run < \
  /mnt/c/Users/User/Documents/ziqinzhang/_collab/E6_s48_dspark_verify_pos.diff; echo rc=$?"
# 期望：checking file src/targets/qwen3_6/impl/runtime/dflash_impl.h / rc=0
# 复核：grep -c '^@@' _collab/E6_s48_dspark_verify_pos.diff   # => 1
```
