# S_D：selector 保真度（BF16 进入点 / FP32 化改动面 / accept 侧消费） 2026-09-11

范围：只读定点分析（不编译、不跑引擎、未改任何源码）。WSL 树 `/home/user/ninfer-fusion`（权威构建树），
参照 `C:\Users\User\Documents\ziqinzhang\1Cat-vLLM`。行号均为读过当时的实际行号。

---

## 0. 结论速览（先看这 6 条）

1. **BF16 有且只有两处**进入 selector：① `logits`（head 输出，BF16 张量）——同时决定**候选集/名次**与 **unary 项的数值**；
   ② `projected`（hidden_projection 输出，BF16 张量）——决定 **edge(pair) 项**。codebook 权重本就是 BF16（参照也是）。
2. **参照的"FP32"只在 ① 一侧**：`_SM70_DFLASH2_VERIFIER_DEFAULTS`（`vllm/config/vllm.py:87-89`）把
   `VLLM_SM70_DFLASH2_FP32_LOGITS="1"` 设为 DFlash2 默认，配套 `sm70_fp32_lm_head.indexed_fp32_logits`
   （FP32 点积，"without an intermediate FP16 logit rounding"）。
   参照的 `hidden_projection` 与两个 codebook 仍是 `params_dtype`(bf16) ⇒ **`projected` 不需要改**。
   R9 §2.4 把它读成"selector 分数整体 FP32"是**过度概括**：`unary`/`scores` 两张 scratch 我们**本来就是 FP32**，
   唯一偏的是 head logits 的 dtype。
3. **greedy 下 accept 侧完全不消费 `draft_candidate_probs`**（两条 greedy 路径都在读到它之前 `return`）⇒
   softmax vs one-hot 在 greedy 下**没有**正确性风险，只有每步一次多余的 K=16 softmax 开销。
4. **FP32 化不可能修 spec-vs-plain 逐位偏差**：selector 只决定**草稿**，不影响 target verify 流；且 K=1 的偏差
   发生在 verify 列的 target logits（R12/R13）。它只是**接受率**项。
5. **比精度更大的分歧（新发现）**：`_verify_df2head.sh:41,44` 用 `--spec dflash2 --lm-head-draft`，
   即 `ProposalHead::Optimized`＝**131072 行 Q4G64 短列表 head**；参照的候选全集是 **248320 行全词表 head**
   （`compute_candidates` 对 `self.lm_head` 做 topk）。**候选宇宙不同**，单靠 FP32 化对不齐参照的候选集。
   要谈"候选顺序保真"，应先在 `ProposalHead::Full`（`TextConfig::output_rows=248320`、`domain=token_domain=248077`）这条路上谈。
6. **裸改 dtype 会静默产生垃圾**：`src/ops/linear/linear.cpp:52` 强制 `x/out` 都是 BF16，且各格式 dispatch 还把输出
   张量**重打类型**（`nvfp4_dispatch.cpp:58-59`、`fp8_dispatch.cpp:77-78`：`Tensor output_chunk(output, DType::BF16, …)`）。
   把 `dflash2_impl.h:342` 的 `DType::BF16` 改成 `FP32` 不会报错，会写错位。**见 §5 的正确改法**。

---

## 1. BF16 进入候选顺序/分数的确切位置

| 语义 | 位置 | 事实 |
|---|---|---|
| head logits 张量定型 | `src/targets/qwen3_6/impl/runtime/dflash2_impl.h:342-347` | `Tensor logits = work.alloc(DType::BF16, {head_rows, k*batch}); ops::linear(proposal_hidden, head, logits, …)`；`head_rows = optimized_head ? Config::draft_head_rows(131072) : TextConfig::output_rows(248320)`（:334-341） |
| **候选集与名次**由 BF16 值决定 | `src/ops/kernel/dflash2_selector.cuh:77,80-102` | `col_logits = logits + column*vocab`；`value = __bfloat162float(col_logits[v])`；快筛 `value < local_value[K-1]` 与插入比较 `value > local_value[k] \|\| (== && id < …)` 全在这 8-bit 尾数上 |
| 归并/落盘 | 同上 `:114-138` | `dflash2_selector_merge16` 同规则（值降序、global id 升序）；`unary[offset] = shared_value[0][k]` ⇒ **unary 的数值就是 BF16 logit 的 float 展开** |
| **edge 项**由 BF16 张量决定 | `src/ops/kernel/dflash2_selector.cuh:176-188` | `hidden = projected + R*flat`(BF16)、`predecessor_row`/`successor_row`(BF16 codebook)；`pair += float(hidden[r])*float(pred[r])*float(succ[r])`，FP32 累加；`scores = pair*pair_scale + shared_unary[c]` |
| projected 定型 | `dflash2_impl.h:350-353` | `work.alloc(DType::BF16, {selector_rank(256), k*batch})` + `ops::linear(proposal_hidden, selector_hidden_projection, …)` |
| walk 消费（已 FP32） | `dflash2_selector.cuh:206-252` | greedy：行最大值 `raw`、逐 lane 比较 `own_value == best`、`chosen = min(chosen, shfl…)`（最低 rank 破平） |
| `out_probs`（已 FP32） | `dflash2_selector.cuh:293-308` | `softmax(row_score(lane) - raw_max)`，行 = 本步 `pred = previous`；写出 `out_probs[lane + K*(s+steps*b)]` |
| scratch dtype | `dflash2_impl.h:357-362`；`layouts_impl.h:738-745` | `candidates` I32、`unary` **FP32**、`scores` **FP32**（已经是 FP32，无需改） |

**一个技术判断（值得实测确认）**：bf16 的有效位数是 8 位，三项乘积 ≤ 8+8+8=24 位 ⇒ 在 FP32 里**可精确表示**，
所以我们的 edge 项只有累加顺序误差；参照的 `_score_edges` 是 `unary + einsum("blpr,blcr->blpc", pred*hidden, succ)`，
其逐元素乘/中间结果会按 `params_dtype`(bf16) 舍入。**即我们的 edge 项大概率比参照更精确**。
⇒ "保真"在 edge 项上不要求我们更差；真正要改的只有 unary。

---

## 2. 参照实现的确切口径（引文）

**(a) 声明 FP32（唯一直白证据）** `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:904-907`
```python
def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
    # The selector walk and rejection sampler must consume identical scores.
    # BF16 rounding measurably changes candidate order, so keep this FP32.
    return torch.float32, -float("inf")
```

**(b) 该 FP32 是怎么来的** `vllm/config/vllm.py:87-93`（DFlash2 verifier 默认）
```python
_SM70_DFLASH2_VERIFIER_DEFAULTS = {
    # Preserve candidate and dense logits in FP32 through sampling.
    "VLLM_SM70_DFLASH2_FP32_LOGITS": "1",
    "VLLM_SM70_FP8_QPN8": "1",
    "VLLM_SM70_DFLASH2_QPN8_RERANK": "1",
    ...
```
`vllm/model_executor/layers/sm70_fp32_lm_head.py:1-4,37-42`
```python
"""Candidate LM-head dots without an intermediate FP16 logit rounding."""
    """Evaluate contiguous FP16 rows into an owned FP32 candidate buffer.
    ... avoiding both expanded cross-row products and FP16 storage of the logits used for top-k/top-p."""
```
`vllm/model_executor/layers/vocab_parallel_embedding.py:540-604`（两段式）：
QPN8(FP8 量化 head) 搜 top-64 support → **FP32 精确重算这 64 行**（`indexed_fp32_logits`），
`use_dense_order = fp32_logits or _sm70_dflash2_use_dense_order()`（默认 `QPN8_DENSE_ORDER=1`）。

**(c) 候选与 unary** `vllm/model_executor/models/qwen3_dflash2.py:456-509`
topk over **整个 lm_head** → `values = values.float() * self.output_multiplier` → （softcap 若有，tanh 在 FP32）；
`:261-282` `_score_edges`：`unary_logits[:, :, None] + einsum(...)`，`hidden = self.hidden_projection(hidden_states)`
（`CandidateSelector.__init__` 的 `params_dtype=vllm_config.model_config.dtype` ⇒ bf16），codebook 同为 bf16。

**(d) 参照自己也只用单一 temperature 口径**：walk 的 `SAMPLE_PROBABILISTIC=self.draft_logits is not None`
（`speculator.py:936,956`）+ `PROPOSAL_TEMPERATURE_SCALE/PROPOSAL_TOP_P`（`envs.py:269-270`，ninfer 无对应项——仅采样模式相关）。

---

## 3. 逐项判定

| 项 | 参照 | 我们 | 判定 |
|---|---|---|---|
| walk 语义（行=previous、列=C_s、previous=chosen、初值 0） | `speculator.py:53-127` | `dflash2_selector.cuh:206-252` | **一致**（F4 已修） |
| lattice 存储精度 | `_selector_scores`/`draft_logits` FP32 | `scores` scratch FP32 | **一致** |
| unary 数值精度 | **FP32 精确点积** | BF16 logit → float | **不一致（本报告主体）** |
| unary 决定的候选**名次/集合** | 对 248320 全词表按 FP32 值排序 | 对 `domain` 行按 **BF16** 值排序 | **不一致** |
| edge 项 | bf16 张量 einsum（含 bf16 舍入） | bf16 输入、FP32 累加（更精确） | 偏差但**方向不利于我们对齐**；建议先不动 |
| codebook / hidden_projection 精度 | params_dtype(bf16) | BF16 | **一致** |
| 候选宇宙 | 248320 全词表 | `--lm-head-draft`⇒131072 短列表；否则 248320 | **取决于 CLI**（见 §0.5） |
| 破平规则 | torch.topk sorted=True ⇒ 按**分片内下标**升序 | 按 **global id** 升序（`cuh:85-91`、`merge16:43`） | **不一致但仅在短列表路上可观测**：`validate_draft_ids`（`bindings.cpp:394-411`）只校验"域内+唯一"，**不校验单调**，故 131072→global id 非单调时两种破平会给出不同名次；Full 路上 index↔id 单调，等价 |

---

## 4. accept 侧：`draft_candidate_probs` 究竟有没有被消费

**链路（已核）**：`round_state.cpp:206-211` 建 `[16, columns-1, batch]`（I32 ids + **FP32 probs**）
→ `dflash2_impl.h:363-367` 交给 selector op（同一次调用同时产出草稿与 q）
→ `dflash2_impl.h:448-449` 放 `TargetVerifyFrameView`
→ `speculative_target_impl.h:27-33` 传给 `ops::speculative_accept_greedy_drafts(...)`
→ `src/ops/kernel/speculative_round.cuh:112-117`（入参 `draft_ids`/`draft_probs`）。

**结论：greedy（`cfg.temperature <= 0`）下永不读取。** 两条 greedy 出口都在读之前 `return`：

```c
// speculative_round.cuh:131-153  —— greedy 且无 penalty：只用预计算的 target argmax
if (!(cfg.temperature > 0.0f) && !penalties) {
    if (tid == 0) {
        int a = 0;
        while (a < extent && row_targets[a] == row_drafts[a]) { ++a; }
        ...
    }
    return;                                  // ← :152 返回，draft_probs/draft_ids 一次都没读
}
// speculative_round.cuh:181-237  —— greedy 带 penalty：重算 argmax 后同样 return
if (!(cfg.temperature > 0.0f)) { ... ; return; }
```
只有 `temperature > 0` 分支（`:239-310`）把它当 **q** 用：`qd = draft_probs[step_row_base + c]`、
`accept_prob = fminf(1, pd/qd)`、拒绝时 `sampling_pick_distributional_residual`（`:63-94`，`max(p-q,0)` 重归一）。
索引口径与 walk 的发布口径（`cuh:303-306` 的 `lane + K*(s + steps*b)`）**一致**（`step_row_base = 16*(i + k*blockIdx.x)`，`:262-263`）。

**所以：greedy 下 softmax vs one-hot 没有正确性风险。** 但三点附带事实：

- **每步仍在算 q**：`cuh:295` 的门只是"指针非空"，而 round_state 无条件分配这两张 buffer ⇒ greedy 下每步仍付
  一次 K=16 `__expf` + 两次 warp 归约（纯开销，可作可选优化）。
- **顺带好处**：既然 greedy 下 q 也总是被正确写出，**可以直接 dump 它做免改代码的诊断**（例如比较 q(argmax)
  与目标列 softmax 的差，量化 BF16 unary 对 q 的影响，先量再改）。
- **仅采样模式的两个次级注意点**（当前不触发，供留档）：① q 的支撑是 selector top-16，若它与 target 截断支撑不相交，
  `mass <= 0` 会**静默退化为纯 p**（`:80-84`）——统计上不致命但会偏离残差公式；② `sampling_uniform(cfg.seed, s+1, …, lane)`
  的噪声键**不含 batch 行 b**（`cuh:225`），同一 seed 下各 batch 行的 Gumbel 噪声相同（相关采样，非正确性问题）。
- 破平一致性（greedy 两条出口之间）已核：`argmax_better`（`argmax.cuh:22-25`）与 `sampling_better`
  （`sampling_device.cuh:76-78`）规则同为"值降序、id 升序"；且 `sampling_adjusted_logit`（`:137-149`）在
  `presence_penalty == 0 && frequency_penalty == 0` 时**原样返回**（`:141`）⇒ 无 penalty 的 fast path 与 penalty path 等价。

---

## 5. FP32 化改动面（给 diff 文本，**不落地**）

### 5.1 先排除错误做法：只翻 `logits` 的 dtype

```diff
--- a/src/targets/qwen3_6/impl/runtime/dflash2_impl.h
+++ b/src/targets/qwen3_6/impl/runtime/dflash2_impl.h
@@ -342,7 +342,7 @@
     Tensor logits = state.execution.work.alloc(
-        DType::BF16, {head_rows, static_cast<std::int32_t>(k) * batch_size});
+        DType::FP32, {head_rows, static_cast<std::int32_t>(k) * batch_size});
```
**这行会静默出错**，因为：
```c
// src/ops/linear/linear.cpp:52-54
if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
    throw std::invalid_argument("linear: x/out must be BF16");
}
// src/ops/linear/nvfp4/nvfp4_dispatch.cpp:58-59（fp8_dispatch.cpp:77-78 同形）
Tensor input_chunk(input, DType::BF16, {weight.k, active});
Tensor output_chunk(output, DType::BF16, {weight.n, active});   // ← 重打类型，把 BF16 写进 FP32 缓冲
```
即 `linear` 侧先抛异常；就算放宽检查，7 个格式 dispatch（q4/q5/q6/w8/bf16/nvfp4/fp8）的 epilogue 全都按 BF16 写。
⇒ 真正"全 FP32 head"= 给 head GEMM 加 FP32 输出变体，横跨 7 个 dispatch + `linear.cpp` 契约 + 每个调用点
（`logit_policy` 也只支持 BF16：`wrapper/logit_policy.cpp:34-35`；qwen3 下它是编译期 no-op，故非阻塞）
—— **改动面大、且动的是共享 op 的契约，回归风险高，不推荐与 GDN 修复合批。**

### 5.2 推荐：照参照的两段式，在**新 op** 里做精确重算（零共享-op 改动）

思路：**保 support（BF16 top-N）+ 用 FP32 精确重算这些行的 logit**，与参照
`QPN8 top-64 → indexed_fp32_logits → top-16` 同构。这样 `linear` 契约一个字都不动，风险被隔离在新 op 内。

```diff
--- a/src/ops/launcher/dflash2_selector.h
+++ b/src/ops/launcher/dflash2_selector.h
@@ -12,6 +12,9 @@
 inline constexpr int kDflash2SelectorRank = 256;
 inline constexpr int kDflash2SelectorTopK = 16;
+// Support width for the exact-FP32 unary rerank. The reference searches an
+// approximate (FP8 QPN8) top-64 support and then re-evaluates those rows with
+// exact FP32 dots before the final top-k; widen the BF16 support the same way.
+inline constexpr int kDflash2SelectorSupport = 64;
 inline constexpr int kDflash2GlobalVocab  = 248320;
```

```diff
--- a/src/ops/kernel/dflash2_selector.cuh
+++ b/src/ops/kernel/dflash2_selector.cuh
@@
+// Re-evaluates the BF16 support rows with an exact FP32 head dot and re-ranks
+// them, mirroring vLLM's indexed_fp32_logits ("without an intermediate FP16
+// logit rounding") + dense-order top-k. The head weight row is dequantized on
+// the fly; only `support` rows per column are touched (K*dot(5120) MACs/column).
+template <int Block, int Support, int K>
+__launch_bounds__(Block) __global__ void dflash2_selector_unary_fp32_kernel(
+    const std::int32_t* __restrict__ support_ids,  // [B,S,Support] global token ids
+    const std::int32_t* __restrict__ support_rows, // [B,S,Support] head row indices
+    const __nv_bfloat16* __restrict__ x,           // [hidden, S*B] proposal hidden (BF16)
+    const void* __restrict__ head_rows,            // head weight rows (format-specific)
+    std::int32_t* __restrict__ candidates, float* __restrict__ unary,
+    int hidden, int batch, int steps, int columns) { /* per-(column,row) FP32 dot + top-K */ }
```

调用点（`dflash2_impl.h:354-367` 附近，保持 scratch 语义但加 support 两行）：

```diff
--- a/src/targets/qwen3_6/impl/runtime/dflash2_impl.h
+++ b/src/targets/qwen3_6/impl/runtime/dflash2_impl.h
@@ -354,6 +354,12 @@
     // The selector runs one step per drafted token; its scratch must be sized by
     // the runtime width or it disagrees with the workspace recipe.
     const std::int32_t steps = static_cast<std::int32_t>(k);
+    Tensor support_ids = state.execution.work.alloc(
+        DType::I32, {batch_size, steps, Config::selector_support});
+    Tensor support_unary = state.execution.work.alloc(
+        DType::FP32, {batch_size, steps, Config::selector_support});
     Tensor candidates = state.execution.work.alloc(
         DType::I32, {batch_size, steps, Config::selector_top_k});
@@ -363,7 +369,9 @@
     ops::dflash2_selector(logits, projected, dflash2.selector_predecessor_codebook,
                           dflash2.selector_successor_codebook, anchors, candidates, unary, scores,
                           flat_drafts, frame.sampling, frame.draft_candidate_ids,
                           frame.draft_candidate_probs, steps, Config::selector_top_k,
-                          head_domain, head_token_ids, state.execution.device.stream);
+                          head_domain, head_token_ids, /*rerank=*/proposal_hidden_input, head_weight,
+                          support_ids, support_unary, state.execution.device.stream);
```
（`proposal_hidden_input` / `head_weight` 需在 `propose_batch_impl` 作用域可见——当前 `logits` 的 `ops::linear`
调用处就有 `proposal_hidden` 与 head `Weight`，取同一对即可。）

配方同步（`layouts_impl.h:731-745`，注意 `dflash2_proposal_capacity` 里也必须同步，否则 arena 尺寸不符）：

```diff
--- a/src/targets/qwen3_6/impl/runtime/layouts_impl.h
+++ b/src/targets/qwen3_6/impl/runtime/layouts_impl.h
@@ -736,6 +736,14 @@
                 // Scratch shapes must match the proposal step count the schedule
                 // actually runs (dflash2_impl.h), i.e. the runtime draft window.
+                matrix(layout, DType::I32, batch * drafts * DFlash2Config::selector_support,
+                       1);
+                matrix(layout, DType::FP32, batch * drafts * DFlash2Config::selector_support,
+                       1);
                 matrix(layout, DType::I32, batch * drafts * DFlash2Config::selector_top_k,
                        1);
```
`config.h:144-149` 加 `static constexpr int selector_support = 64;`（三处 config.h 都要，参照 F1 的
`dflash2::draft_head_rows` 改动模式）。`include/ninfer/ops/dflash2_selector.h` 的契约注释需补两段（support / 精确重算）。

### 5.3 最小变体（若只想先动 unary 数值、不碰 support 宽度）

同一新 kernel，把 `Support := K := 16` 即可：候选集/名次仍由 BF16 决定，只有 `unary[]`（以及随之的 scores）变成
精确 FP32 值。改动面与 5.2 相同（少两张 support scratch）。**注意这只能修正 edge/unary 的相对权重，
不能修正"BF16 把某行挤进/挤出 top-16"这类名次错误**——而那才是参照注释里"changes candidate order"所指。

### 5.4 落地顺序建议（与主代理的 ①合批）

- 与 GDN 修复合批是可行的：5.2/5.3 是**纯新增 op + 分配 + 配方**，不触碰任何既有 op 的 dtype 契约，
  不需要再 `touch` includer 之外的头文件依赖处理。
- 但**必须同时**选对路线：若仍在 `--lm-head-draft`（131072 短列表）上跑，先做 5.4 的第一步（下节）。
- 若先做 §6 的"先量后改"，则这一项可以推迟到 GDN 修复验收之后。

---

## 6. 可检验预测 / 验收判据（不跑引擎，只给判据）

1. **候选宇宙先定性**：同一 prompt 分别跑 `--lm-head-draft` 与不带它（`ProposalHead::Full`），
   dump `draft_candidate_ids`（`round_state` 里已有 buffer，greedy 下也被写出）。**预测**：两者的候选 id 集合
   在多数列上不同（131072 短列表 ⊂ 248320 但 top-16 未必相同）。若成立 ⇒ 参照保真必须在 Full 路上谈。
2. **BF16 vs FP32 的 unary 影响量级**：dump `unary`/`scores`（都已是 FP32 scratch）比对
   `unary[0] - unary[1]`（榜首margin）与 `|scores[p][c] - scores[p][c']|` 的量级。
   **预测**：若榜首 margin 常落在 bf16 ulp（≈|logit|·2⁻⁸）以内，则名次反转确实会发生，5.2 有收益；
   若 margin 普遍 ≫ ulp，则 5.2 只影响分数不影响名次，收益≈0。这一步能把"FP32 化值不值得"变成可测的。
3. **接受率**：5.2/5.3 落地后应满足（a）spec 流与 plain **逐位一致**（与改动无关，作为回归闸门），
   （b）`_df2_blame2.py` 的链式剖面 p1+ 非零、AL≥3 不退化，（c）接受率不下降。
   **预测**：接受率提升幅度上界 = 现在"因 BF16 名次/unary 而选错候选"的频率，通常是个位数百分点；
   不要指望它解释 R12 的 ckpt 口径上限。
4. **反证预测（重要）**：本项**不可能**让 K=1 的 spec-vs-plain 偏差消失（§0.4）。
   若落地后 K=1 仍偏在 62 位，那是预期内的——不要据此判定改动失败。

---

## 7. 留档：本次核过但**无需改动**的项

- `dflash2_selector.cuh:168-171`（`s==0 && p>0` 位置置 -inf）：walk 在 s=0 只读行 0，与参照等价。
- `NINFER_DF2_PAIR_SCALE`（`launcher/dflash2_selector.cu:27-30`）默认 1.0：参照 `_score_edges` 无任何缩放，
  **落地时必须确保该 env 不设**，否则引入非参照的比例因子。
- `argmax` 的 `valid_rows`/短列表 map 在 Full 路（`domain=token_domain=248077`）与参照的
  "exclude ids ≥ token_domain" 等价。
- 采样模式的 `PROPOSAL_TEMPERATURE_SCALE` / `PROPOSAL_TOP_P`（`envs.py:269-270`）在 ninfer 无对应实现——
  仅当启用 `temperature>0` 时才成为保真项，当前 greedy 口径下不触发。
