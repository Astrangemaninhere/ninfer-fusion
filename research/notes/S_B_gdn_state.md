# S_B：GDN 状态记录 / 回滚 / 跨边界携带（只读源码分析）

作者：S_B 子代理（只读、未编译、未跑引擎/GPU、未改源码）
树：`/home/user/ninfer-fusion`（WSL），对应 Windows `C:\Users\User\Documents\ziqinzhang\ninfer-upstream` 之外的构建树
触发：`_collab/_HANDOFF.md` §3 S_B + 主代理追加的 `--prefill-chunk` 运行期实证（首 token 分叉）

---

## 0. 结论先行

1. **spec 轮不存在"回滚"**。verify 一次推进 W 列时，`RecordForReplay` 路径**完全不写任何状态**（key/value/gate/conv 只写记录平面），只有 committed 前缀被 `gdn_replay_fold` 从"旧状态槽"重放一次写进"新状态槽"。被拒列的去向是"从未被应用"，不是"被撤销"。
2. **fold 与"逐 token 推进 commit 次"在 GDN 递推上逐位等价**（同一段逐 token 代码、同一份记录、同一个 FP32 起点；记录平面里是 bf16/FP32 的**已发布值**，不是更宽累加器）。见 §3。
3. **conv 窗口的重建也是精确的**：记录平面存 `bf16(p)`（即"被推进窗口的那个已发布值"），fold 的 `publish_final_conv_history` 就是精确的 `tail_3` 切片。见 §3.2。
4. **因此任何 spec≠plain 的残余偏移都不在"状态携带/回滚"里**，而在"W 列这一批的**进料与出料**由另一形状的 kernel 算出"：见 §4 的 D1–D4。
5. **D1（与今天落地的 `gdn_conv.cuh:118` 同类、且尚未修）**：fused 路线里 conv 的第三个抽头用的是 **FP32 GEMM 累加器 `p`**，而同一个函数把 `bf16(p)` 发布进状态窗口/记录平面 ⇒ "算出来的输出"与"存下来的状态"不是同一个数。同一条 `:118` 补丁只修了**携带**、没修**进料**。★最高优先
6. **D3（新增，回答 Q3）**：chunked 路线把 `l2norm(q/k)` 发布成 **BF16** 再进递推（`gated_delta_net.cpp:186-191,256-260`），而 per-token/recurrent 路线在**寄存器里用 FP32 归一化值**进递推（`recurrent.cuh:615-631`）⇒ 同一个 token 因为"落在哪个 kernel 形状里"而拿到不同的 k。这正好解释"同 prompt 换 `--prefill-chunk` 首 token 就分叉"。
7. **可检验的三条最小实验**：① `batch=1` vs `batch=2` 同题对照（无需改码，见 P0）；② 强制 record plan 走 `MaterializedA16` 后跑 `_ga_check.sh`（P1）；③ 把 `gdn_conv.cuh:99` 的 `p` 先做 bf16 化（P2）。判据用**量级**：~1e-3 相对 = "更宽累加器 vs 已发布值"类；~1e-7 = 纯结合序/分块差异（不可消除）。
8. 引擎侧两个关键不变量是**已被校验**的（`accepted+1 == count`，`count ≤ extent+1 = valid_columns`），所以 fold 不会读到未写过的记录列；这点让"回滚正确性"的讨论可以收窄到精度，而不是索引。见 §3.3。

---

## 1. 我读过的权威位置（全部为当前树的 `file:line`）

| 主题 | 位置 |
| --- | --- |
| 记录平面布局/几何 | `src/core/gdn_replay_records.h:11-59`；`src/core/gdn_replay_records.cpp:96-140`（`record_capacity ≤ 8`；`layer()` 按 `layer*capacity` 切 `outer`） |
| fold op 契约 | `include/ninfer/ops/gdn_replay.h:19-53`（`GdnReplayFoldRow`、`commit_columns`、零=no-op 语义） |
| fold 校验 | `src/ops/linear_attention/gated_delta_net/replay.cpp:261-290`（行/槽校验、目的槽互斥且不得覆盖他人源槽）、`:315-320` |
| 记录写入（递推） | `recurrent.cuh:350-441`（`RecordAccess`，**无 `state_write`**）、`:675-684`（`recurrent_record_kernel`） |
| 记录写入（conv） | `src/ops/gdn_input_proj/gdn_conv.cuh:27-37`（`RecordColumnPublish`）、`:11-25`（`SnapshotHistoryPublish`）、`:47-121`（`GdnConvEpilogue`）。**注意**：`_HANDOFF`/主代理说的 `gdn_conv.cuh:116` 在当前树是注释行，实际语句在 **`:118`**（`s2 = __bfloat162float(__float2bfloat16_rn(p));`） |
| 状态动作开关 | `text_context.h:218-221,294,431`；`text_context_impl.h:371-378`；`speculative_target_impl.h:12-15` |
| verify 中的两条分支 | `text_context_impl.h:1094-1136`（conv：record vs snapshot）、`:1144-1177`（递推：record vs batch_update）；`gdn_mix` 里 `width != 1` 与 `UpdateInPlace` 互斥（`:1103-1105`） |
| fold 的调用点与提交量 | `program_impl.h:9820-9868`（`commit_columns = accepted_tokens`）、`:12491-12498`（DFlash2 元数据不变量）/`:12199-12226`（DFlash 同构） |
| 槽选择器 | `state_image_store.h:56-59`（`StateImageSelectors`）、`:353-371`（`begin_fork`：source=CheckpointImmutable / dest=ReservedDestination→ActiveMutable）、`:401-417`（`selectors()`）、`program_impl.h:10028-10035`、`:12412`（ingress 的 `state_source_slots = selectors.source`） |
| fold kernel | `recurrent.cuh:456-590`（`FoldAccess`，`:497-517` 源/目的槽基址）、`:549-589`（`publish_final_conv_history`）、`:611-635`（`run_recurrent_sequence`）、`:686-697`（`recurrent_fold_kernel`） |
| fold launcher/几何 | `recurrent.cu:104-128`（整面指针 + 几何分派）、`:190-208`（48×48 / 30×32） |
| 平路（plain）conv | `ops/kernel/causal_conv1d.cuh:234-259`（decode，`acc += w*x` 四段式）、`:96-100`、`:219-223`；launcher `ops/launcher/causal_conv1d.cu:43-140`；wrapper `ops/wrapper/causal_conv1d_silu.cpp:240-320` |
| 平路 qkv 是 BF16 | `workspace_recipe.h:117-119`（`gdn_prefill_conv` = `DType::BF16`）；`text_context_impl.h:1128-1135` |
| g/beta 是 FP32 | `workspace_recipe.h:90-97`；`recurrent.cuh:87-92` |
| fp8 路线分派 | `fp8_gdn_conv_plan.cpp:44-47`（`b1_a16_plan`：fused ⇔ W≤3 或 7≤W≤10）、`:99-115`（record plan）、`:180-201`（fused vs materialized）；`fp8_gdn_conv_fused.cu:60-70,127-137` |
| materialized record conv | `gdn_projected_conv.cu:53-58`（`p = bf16(projected[...])`、fmaf 链） |
| chunked GDN | `gated_delta_net.cpp:178-195,241-290`；`chunked/launch.h:26-35`（工作区精度）；`common.h:8`（`kChunkSize = 64`） |
| 现有单测 | `tests/ops/test_gdn_replay_fold.cpp:64-67`（容差判据）、`:258-295`（conv 期望）、`:335-375`（对 `gdn_ref` 的逐 token 预言机比对） |

---

## 2. 机制全貌：一轮 spec 的三段式

```
verify(target_verify_batch, phase=Verify)
├── gdn_state_action_ = RecordForReplay     (speculative_target_impl.h:15)
├── conv : Variant::gdn_input_projection_record   (text_context_impl.h:1118-1121)
│         → 读 source 槽的 3 抽头窗口，算 W 列的 q/k/v，写 conv 记录平面
│         → GdnConvEpilogue 不写任何 state（gdn_conv.cuh:113 只 publish）
└── rec  : ops::gated_delta_net_replay_record     (text_context_impl.h:1159-1162)
          → 读 source 槽的 FP32 递推态，逐 token 链式跑到第 W 列
          → 每列把 raw key(BF16) / raw value(BF16) / {g,beta}(FP32) 写进记录
          → RecordAccess 没有 state_write（recurrent.cuh:350-441）⇒ 状态不动
accept  : speculative_accept_greedy_drafts → accepted_drafts（≤ extent）
resolve : fold_rows[row] = {source=selectors.source, destination=selectors.destination,
                            commit_columns = accepted_tokens = accepted_i + 1}
          (program_impl.h:9847-9857)
          GdnReplayFoldPlan::execute                (recurrent.cu:104-128 → recurrent_fold_kernel)
          → 从 source 槽 FP32 态起，重放 [0, commit) 条记录，写 destination 槽 FP32 态
          → 同一 kernel 里把 conv 窗口写成 tail_3(源窗口 ‖ 记录列)
```

`commit_columns` 的取值：`accepted_i + 1`，第 0 列（anchor）**总是被提交**，之后 `accepted_i` 列提交，尾部 `W-1-accepted_i` 列被丢弃。引擎对 DFlash/DFlash2 都在边界上强校验（`program_impl.h:12219-12224` / `:12491-12498`）：`count_i>0`、`accepted_i+1 == count_i`、`accepted_i ≤ extent`、`extent ≤ width`。ingress 的 `valid_columns = extent+1`（`:12402`），记录 kernel 只写 `token < valid` 的列 ⇒ **`commit ≤ valid` 恒成立，fold 永不读未写过的列**（未写的列保持上一轮的陈旧值）。

### 2.1 被拒列的去向

- 递推态：被拒列**从未**乘进状态（记录 kernel 无写口），因此无需回滚。
- conv 窗口：source 槽的窗口**从不被 record 路径改写**（`GdnConvEpilogue` 只读 `state_read`）；fold 用"源窗口 + 记录列"重算 `tail_3`。
- 出料：`recurrent_record_kernel` 用 `zero_output_suffix`（`recurrent.cuh:637-646`）把 `token ≥ valid` 的 q/k/v 输出清零，所以即使有人误读越界列也是零。
- 因此"被拒列的副作用"只存在于 **KV cache**（由 `append_counts`/licensed counts 处理，不属 S_B 范围），GDN 侧无残留。

---

## 3. 严格等价性论证（回答 Q2 的核心）

### 3.1 递推状态：逐位等价

`run_recurrent_sequence<Normalize, Effects>`（`recurrent.cuh:611-635`）对**所有**路线只有一处实现，差异只在 `Effects`：`RecordEffects` 额外写记录、`FoldEffects` 什么都不做（`:211-225`）、`OutputEffects` 额外写输出。因此：

- 记录 pass 在寄存器里跑出的终态 `S_W` ＝ "从 source 槽 FP32 态出发、逐 token 推进 W 次"的结果（逐位）。
- fold 用**同一段代码**、**同一记录**、**同一起点**只跑 `commit` 次 ⇒ fold 结果 ＝ `S_commit`（逐位），即"逐 token 推进 commit 次"。
- 记录里的 key/value 是**原始 BF16 位**（`store_key` 存 `raw.bits`，在 `normalize_qk_lane` 之前，`:416-424`），gate 是 **FP32 位**（`:435-440`）。fold 用 `load_record_gate` 原样还原（`:94-97,531-535`）并做同样的归一化 ⇒ 无二次量化损失。
- 两条 kernel 都硬编码 `<true>`（normalize）：`recurrent_record_kernel` / `recurrent_fold_kernel`；平路 decode 也传 `normalize_qk=true`（`text_context_impl.h:1174,1166`）⇒ 归一化开关一致。
- `alpha = expf(g)`、warp 归约、`state[r][c] = alpha*state[r][c] + delta*key[c]` 都是同一表达式（`:99-117`）⇒ 逆序无关、无原子加。

**结论**：给定"每列的 k/v/g/beta 与 W 列 pass 相同"，fold 与逐 token 推进**逐位一致**；再对轮次做归纳（每轮 fold 的输出都是下一轮的 source），spec 的 GDN 状态 ＝ 同一条逐 token 链在相同进料下的状态。

### 3.2 conv 窗口：逐位等价

- 记录列 = `bf16(p)`，其中 `p` 是该列被推进窗口的那个**已发布值**（`gdn_conv.cuh:35`）。
- 窗口语义（`GdnConvEpilogue::store`，`:76-119`）：`s0,s1,s2 = state[0,+C,+2C]`，`conv = w0*s0+w1*s1+w2*s2+w3*p`，随后 `s0=s1; s1=s2; s2=bf16(p)`（`:118`，即今天落地的补丁）。
- fold 的重建（`:570-588`）：`commit==1 → [src1,src2,rec0]`；`commit==2 → [src2,rec0,rec1]`；`commit≥3 → [rec(commit-3), rec(commit-2), rec(commit-1)]`。
  ⇒ 与 `s0/s1/s2` 语义**逐下标对应**，且 `[oldest, mid, newest]` 顺序一致。
- 覆盖率自检：`tile_block = value_head*8 + state_tile`，只有 `< kConvChannels/128` 的参与（`:553`）；48 头 × 8 片 = 384 ≥ 80，每通道恰好写一次（tile_block 与 (head,tile) 双射）⇒ 不会漏写/重写。几何断言 `ConvChannels % 128 == 0` 在 `FoldGeometry`（`:443-454`）里 static_assert。

### 3.3 需要成立的前提（两条已被引擎校验、两条是精度问题）

| 前提 | 状态 |
| --- | --- |
| ① fold 的 `source` ＝ record pass 的 `initial_slots` | ✅ 同为 `selectors.source`（`program_impl.h:12412` vs `:9853`），且 resolve 时校验 `sequence` 仍在 `pending.base_*`（`:9835-9845`） |
| ② `commit ≤ valid_columns` | ✅ `accepted_i+1 == count_i ≤ extent+1`（`:12491-12498`） |
| ③ 记录平面索引对齐（layer/row/column） | ✅ record 用 patch 后的层切片（`outer = batch`），fold 用整面 + `layer*capacity + batch`，两者偏移恒等（`gdn_replay_records.cpp:133-139` vs `recurrent.cuh:497-535`） |
| ④ 被记录/被重放的值 ＝ 平路会算出的值 | ❌ **不成立**，这就是 §4 |

### 3.4 与"逐 token 推进 W 次"的等价（明确回答）

- **状态转移：等价（逐位）**。原因是设计上就没有 chunked/分块进递推：`RecordAccess`/`FoldAccess` 都是**逐 token 序贯**，没有跨列的状态量化（不像平路 prefill 走 chunked，见 §4.3/§5.2）。
- **不等价的只可能是"每列的进料"**：W 列一次算出的 q/k/v 与单列逐个算出的 q/k/v 不由同一条 kernel 产生（route 按 T 选择）：D1（fused 用 FP32 累加器）、D2（fmaf vs mul+add）、以及验证列可能走的其它 T 形状 kernel（attention/MLP/lm_head）。
- 所以 `_HANDOFF` §4② 的目标"spec 流与 plain 逐位一致"**在 GDN 之外还有多源**；但 GDN 内部存在**两处可以并且应该修掉的**不等价（D1、D3）。

---

## 4. spec≠plain 的 GDN 侧精度不一致（回答 Q1/Q3）

### D1 ★ 未修：fused 路线的 conv 用 FP32 累加器当第 4 个抽头

`gdn_conv.cuh:99-104`：
```cpp
const float p = projected[token];              // FP32 GEMM 累加器，未 bf16 化
float conv = fmaf(w0, s0, 0.0F);
conv = fmaf(w1, s1, conv); conv = fmaf(w2, s2, conv);
conv = fmaf(w3, p, conv);                      // ← 用 p_fp32
const __nv_bfloat16 output = __float2bfloat16_rn(silu(conv));
...
s2 = __bfloat162float(__float2bfloat16_rn(p));  // ← :118 已修：携带用 bf16(p)
```
即：**输出用 `p` 的 24 位，状态窗口存 `p` 的 9 位**。平路（`text_context_impl.h:1128-1135`）先把投影写进 **BF16** 的 qkv（`workspace_recipe.h:117-119`），再见 `causal_conv1d` 卷积 ⇒ 平路永远用 `bf16(p)`。量级：相对 2^-9 ≈ 2e-3，逐列、全通道系统性地改变 q/k/v ⇒ 足以偶发翻转 argmax（与"列 0 只有 85% 一致"相容）。

**触发面（可判定谓词）**：只有 `FusedA16` 计划走这段代码（`fp8_gdn_conv_fused.cu:26-70` 的 small_t / `:72-88` 的 fused decode，以及 nvfp4 `small_t`、q4_q5 `conv_snapshot`、w8 `splitk` 的 record/snapshot 变体）。fp8 的 record/snapshot 计划在 `batch==1` 且 `W ∈ {2,3} ∪ {7,8,9,10}`（AllowA8 时 W=10 走 A8；A16Only 时含 10）时为 `FusedA16`，**W ∈ {4,5,6,11..16} 时为 `MaterializedA16`**（`fp8_gdn_conv_plan.cpp:44-47,99-115`）。`batch>1` 时 A16 一律 Materialized（`:111-114`）。

> 这条与今天落地的 `:118` 是**同一个 bug 的两个面**：`:118` 修的是"携带更宽的累加器"，D1 是"用更宽的累加器算输出、却只把窄值写进状态"。自洽性论证（不依赖任何实验）：**被发布进状态窗口的值是 `bf16(p)`，那么同一次的卷积输出也应当只消费 `bf16(p)`；否则"验证列的输出"永远对应一个状态里不存在的数**。

### D2：fmaf 链 vs `acc += w*x`

- 平路 conv（`ops/kernel/causal_conv1d.cuh:248-257` decode、`:219-223` smallt、`:96-100` prefill，含 snapshot 变体）：`float acc = 0.0f; acc += w*s; …` ⇒ 4 次乘 + 4 次加（8 次舍入）。
- record/materialized conv（`gdn_conv.cuh:100-103`、`gdn_projected_conv.cu:54-57`）：`fmaf` 链（4 次舍入）。
⇒ 即使同为 `MaterializedA16`，两种路线的 conv 结果在 fp32 末位不同，`silu` + bf16 舍入后偶发 ±1 LSB。量级 ~1e-7，属"不可避免的结合序差异"，但会贡献偶发 token 分歧。

### D3 ★ 新增：chunked 路线把归一化 q/k 以 BF16 发布，recurrent 路线用寄存器 FP32

`gated_delta_net.cpp:186-191,255-261`：当 `T_full > 0` 且 `normalize_qk` 时，chunked 路线先把 `q/k` 做 `l2norm` 写进 **BF16** 工作区（`{kStateDim, qk_heads, tokens}`），并把 `recurrent_normalize` 置 false；chunked kernel 再从 BF16 读入（`chunked/state_passing.cuh:124-160` 用 `issue_load_k_bf16` → float）。
而 per-token/recurrent 路线（`recurrent.cuh:615-631`）在寄存器里对 `bf16→fp32` 的值做归一化并**保持 FP32** 进递推。
⇒ **同一个 token，落在"满 64 块"里 vs 落在 tail/decode 里，k 的有效精度不同（9 位 vs 24 位）**。这与 `:116` 补丁是同一类（"已发布值 vs 更宽累加器"），且它是**算法级**的（不是 tiling 级），量级同样 ~2e-3。

这正是主代理 `--prefill-chunk 128` vs `4096` 首 token 分叉的**同族**机制：`--prefill-chunk` 改变"每次调用处理多少 token"，从而改变 `T_full = floor(T/64)*64` 的切分、tail 归属、以及 `l2norm`(BF16) 与寄存器 FP32 归一化的混合比例。（prefill 侧还有更多 T 形状 kernel，见 §5.3。）

### D4：chunked 内部工作区是 BF16（64-token 粒度的已发布值）

`chunked/launch.h:26-35`：`W`、`U`、`v_new` 为 **BF16**，`h_chunk` 为 **BF16**（`{128,128,value_heads,chunks}`），只有 `g_cumsum` 与 `state_out` 是 FP32。
⇒ 每个 64-token 块的状态贡献被量化成 BF16 后再累加进 FP32 状态；块的**切分点/数量**变化会改变舍入模式。这不是"回滚"问题，但它使 prefill 的结果对 `--prefill-chunk`（以及任何改变 T 的东西）敏感。

---

## 5. 跨 chunk 边界携带了什么（回答 Q1）

### 5.1 引擎层（`TextContext::prefill_chunk` 之间）

| 携带物 | 位置 | 精度 |
| --- | --- | --- |
| conv 窗口 | `state_.conv_slot(gidx, linear_state_*_slot_)`，读写在 `text_context_impl.h:1131-1135`；fold 侧 `recurrent.cu:115`（`__nv_bfloat16* conv_layer0`，`3*ConvChannels`） | **BF16 已发布值**（`causal_conv1d.cuh:255-257` 直接写 `s1/s2/x0`，`x0` 来自 BF16 qkv；snapshot 变体 `gdn_conv.cuh:21-23` 写 `__float2bfloat16_rn(p)`） |
| 递推状态 | `state_.recurrent_slot(gidx, slot)`，`text_context_impl.h:1170-1176`；`gated_delta_net.cpp:166` 断言 `ssm_state_in` 为 FP32 | **精确 FP32**（无量化损失） |
| 门控 | g/beta 在 `workspace_recipe.h:94-95` 为 FP32（`gdn_norm_gating_proj` 产出） | FP32，`recurrent.cuh:87-92` 按 float 读 |
| KV/DFlash context | `ensure_sequence_kv_mapped` 等 | 不在 S_B 范围 |

**结论**：跨 chunk 边界携带的是**已发布值**（conv 的 BF16 窗口）＋ **精确 FP32 递推态**；**没有任何"更宽累加器被携带"**的点。所以 prefill-chunk 分叉**不是**"携带了更宽累加器"，而是"每一段内部的进料/中间量由不同形状 kernel 产生"（D2/D3/D4 + 非 GDN kernel）。

### 5.2 op 内（`kChunkSize = 64`，`common.h:8`）

`gated_delta_net` 的 in/out 版本把 `[0, T_full)` 交给 chunked（`launch_chunked` → `prepare_wy_wu` → `state_passing` → `output`），`tail = T - T_full` 交给 per-token recurrent（`gated_delta_net.cpp:262-290`）。chunked 内部跨块的运行量是 **BF16 的 `h_chunk`/`v_new`/`W`/`U`**（§D4），FP32 的 `state_out` 只在块边界被发布/累加。

### 5.3 为什么"首 token 就分叉"是预期的

`--prefill-chunk` 同时改变：GDN 的 `T_full`/tail 切分（D3/D4）、GDN 输入投影的 T 形状路线（`Variant::gdn_input_projection` 的 small_t/prefill/A8/A16 选择）、attention 与 MLP 的 T 形状 kernel、以及 lm_head 的 T 形状。任何一处末位差异都足以在 1200 token 的累积后翻转首 token。**因此这条实证不应被解读为"GDN 回滚有 bug"**，它是"同数学不同形状不逐位等价"的既知性质；S_B 的价值是把它**定位到可指认的 GDN 内部两处（D1/D3）**，并给出判据把"1e-3 级"与"1e-7 级"分开。

---

## 6. 可检验预测（全部可在现有脚本/单测上做）

**P0（零改码、最高性价比，先做）**：同一单请求 prompt，分别以 `batch=1` 与 `batch=2`（同一 prompt 复制两份）跑同一次 spec；比较 row0 的 token 流。
- 若两者**不同**且分叉模式与 K 扫描（62/62/29 那组）同族 ⇒ GDN 的 `FusedA16 ↔ MaterializedA16` 切换确实在动数（D1 成立），因为 `batch>1` 时 fp8 A16 record plan 一律 Materialized（`fp8_gdn_conv_plan.cpp:111-114`），而 `batch==1`、W∈{2,3,7..10} 是 Fused。
- 若两者**逐位相同** ⇒ D1 不在这条路径上（该 profile 可能取 A8 或另一 payload/route），应转向 D3（把 chunked 的 `l2norm` 输出改 FP32 后再比对 prefill-chunk 敏感性）。

**P1**：把 record plan 强制为 `MaterializedA16`（`fp8_gdn_conv_plan.cpp:44-47` 的 `fused` 恒 false，最小临时改法），跑 `_ga_check.sh` 与 `_ga_ksweep.sh`。
- 预测：**K=1（W=2）、K=7（W=8）的差异发生变化**；**K=3（W=4）不变**（W=4 本来就是 Materialized）。
- 若 K=1 也完全不变 ⇒ 全局偏移的主因不在 GDN 输入投影，转去 S_D（selector）与 attention/MLP 的 T 形状。

**P2**：按 `:118` 的范式把 `gdn_conv.cuh:99` 改为
`const float p = __bfloat162float(__float2bfloat16_rn(projected[token]));`
（并让 `publish` 收到同一个已 bf16 化的值，保持"输出/状态/记录"三者同源）。预测：fused 窗口（W=2,3,7..10）的 spec-vs-plain 偏移缩小；且 `mtp` 流**只有在该窗口内**才会变（见 §7）。

**P3**：`gated_delta_net.cpp:186-191` 把归一化缓冲改为 FP32（或让 chunked kernel 自己做寄存器 FP32 归一化，与 `recurrent.cuh:615-631` 对齐），再跑 `_prefill_chunk_equiv.sh`。
- 预测：`--prefill-chunk 128` vs `4096` 的首 token 差异**变小但不为 0**（残差来自 D2/D4 与非 GDN 的 T 形状 kernel）。
- 反证：若修完 D3/D4 后差异完全消失 ⇒ prefill chunk 分叉可单一归因于 chunked 路线的 bf16 发布；这会把问题范围收窄得非常好。

**P4（量级判据，用来给任何 A/B 定性）**：在同一位置对比两条路线的中间张量（建议：最后一个 prompt token 的 hidden / logits / 或 GDN 的 q,k,v 与递推终态）：
- 相对偏差 ~1e-3 ⇒ "更宽累加器 vs 已发布值"类（D1/D3/`:116` 同族，**是 bug，可修**）；
- 相对偏差 ~1e-7 ⇒ 结合序/tiling 类（D2/D4 与非 GDN kernel，**属既知形状依赖**，只能靠统一 kernel 形状消除）。
  这条判据不需要改任何代码，只要把张量 dump 出来比。

**P5（单测补强，防回归）**：`tests/ops/test_gdn_replay_fold.cpp:64-67` 用的是容差（`relative_l2=2.7e-3`）比对 `gdn_ref`。建议加一条**逐位**用例：用同一份记录平面，把 `commit∈{1..W}` 的 fold 结果与"逐 token 调用 `gated_delta_net`（`gated_delta_net.cpp:212-225` 的 in-place 平路）跑 commit 次"的结果做 bitwise 比较。
- 预测：**逐位相同**（这是 §3 的证明的可执行形式）。若不同，则 §3 的前提③/④在实现层面有偏差，优先查 `publish_final_conv_history` 与 conv 记录的列序。

**P6（不变量断言，防未来回归）**：在 `GdnReplayFoldPlan::execute`（`replay.cpp:315-320`）外侧或 `program_impl.h:9847-9857` 加一句开发期断言 `commit_columns == accepted+1 && commit_columns <= valid_columns[row]`。
- 现在靠 DFlash/DFlash2 各自的元数据校验（`:12219-12224`、`:12491-12498`）间接保证；MTP 路径的对应校验未在本次阅读中逐一核对（见 §8 边界），加断言可以覆盖它。

---

## 7. 与 S_A 的交叉校验：为什么 conv 补丁没改 mtp 的流

`:118` 所在的 `GdnConvEpilogue` **只在 `FusedA16` 计划里被执行**；`MaterializedA16/A8` 走 `gdn_projected_conv_kernel`（`gdn_projected_conv.cu`，另一个文件、没有这段 carry）。
⇒ 可判定的解释是：**MTP 那次的 verify width 落在 Materialized 窗口**（W ∈ {4,5,6,11..16}，或 batch>1，或 profile 走了 A8/非 fp8 payload），所以被改的那行从未执行。**建议 S_A 把"解析出的 `Fp8GdnConvScheduleId`"打出来**（`fp8_gdn_conv_plan.cpp:99-115`），一行日志即可定案；这比"生效的是另一条 GDN kernel"更精确。
同理，P2/P0 的对照只有在 Fused 窗口内才可能有反应——这是设计 A/B 时必须先确认的前置条件。

---

## 8. 我未验证的边界（诚实声明）

1. **一切均为静态阅读**：未编译、未运行、未 dump 任何张量；`5.3`/`P4` 的量级是量级估计，不是实测。
2. 我未确认该 27B DFlash2 artifact 实际使用的 `LinearPolicy`（A16Only / AllowA8）与 payload（Split / Fused），因此 D1 的**触发窗口是谓词而非断言**；这需要 S_A 的 route 观测（或一行日志）落实。
3. 我未逐行核对 MTP 路径（`mtp_impl.h`）与 DFlash(v1)（`dflash_impl.h`）是否与 DFlash2 有完全相同的元数据校验；P6 正是为此设计。
4. chunked 算法的内部细节（`prepare_wy_wu.cuh` / `state_passing.cuh` 共 56 KB）我只读了接口、精度与块划分，未逐行核对数学；因此 D4 的"块切分敏感"是结构判断，未定量。
5. 树内存在 `*.orig`（`program_impl.h.orig` 等）；我全部以非 `.orig` 文件为准，未核对两者差异。
6. `--prefill-chunk` 分叉我**没有**归因到某一个 kernel；§5.3 给出的是一组候选与区分方法，不是结论。
