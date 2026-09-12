# R9：草稿→head→selector 全链路审查（对参照实现逐项）+ verify 侧定位（2026-09-11 14:3X）

## 一、审查用到的"权威基准"（全部为已知路径定点读取，未扫盘）
- 参照 dflash2 模型定义：`1Cat-vLLM/vllm/model_executor/models/qwen3_dflash2.py`
- 参照 dflash2 调度/walk：`1Cat-vLLM/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py`
- ckpt 自带超参：`data/draft_dflash2_ref/config.json`（`DFlash2DraftModel`）

## 二、逐项审查结果

### 2.1 声明型超参：**全中**（见 R8 表）
`mask_token_id 248070`、`selector_rank/top_k 256/16`、`conv 16/2`、`block_size 8`、
`target_layer_ids [5,19,33,47,61]`、`32/8` 头、`rope_theta 1e7`、`sliding_window 2048`、
`is_causal false` 全部与引擎 `DFlash2Config` 一致。

### 2.2 块与 positions：**一致**
参照 `speculator.py:859-872`：`num_tokens==8` / `num_draft_tokens==7` ⇒ W=8、K=7；
`_context_target_positions.copy_(input_batch.positions[:8])`，注释明写
"Raw positions equal the later masked positions for every accepted row. Rejected rows remain
scratch data and never reach the KV cache." ⇒ context 用 **target 块的 8 个绝对 position**，
与我们的 `append_context_impl` 用 sink 捕获的 feature positions 等价。

### 2.3 selector walk：**语义一致（F4 修复正是对齐参照）**
参照 `_selector_walk_kernel`（`speculator.py:53-127`）：
```python
previous = 0
for step in range(walk_steps):                 # walk_steps = draft_block = 7
    score_base = (flat*top_k + previous)*top_k  # 行 = previous（上一步的候选 rank）
    scores   = load(scores_ptr + score_base + c)     # 本步 E 行
    candidates = load(candidate_ptr + flat*top_k + c)
    _, index = gumbel_noised_argmax(scores, ..., temperature=0 for greedy)
    store tokens[flat] = candidates[index]
    previous = index
```
我们修好的实现：行 = `pred = previous`、列 = `C_s[c]`、`previous = chosen`、
`drafts[s] = C_s[chosen]`、初值 0 —— **逐条相同**。
（修前我们 `chosen` 恒 0 = 忽略边项，与参照语义不符 —— 这是 F4 修掉的实质缺陷。）

### 2.4 **抓到一处真实保真度分歧（新）**
参照 `speculator.py:904-907`：
```python
def draft_logits_spec(...):
    # The selector walk and rejection sampler must consume identical scores.
    # BF16 rounding measurably changes candidate order, so keep this FP32.
    return torch.float32, -float("inf")
```
⇒ 参照全程 **FP32** 的 selector 分数；而我们 `dflash2_impl.h` 的
`logits = work.alloc(DType::BF16, {head_rows, k*batch})` 是 **BF16**，top-K 与 unary 都吃这 8-bit 尾数。
参照自己标注"BF16 会 measurably 改变候选顺序"。**这是一处该对齐的参数（精度），改动小、风险低。**

## 三、verify 侧：今天实测出的真偏差（接受率上限）
同 prompt、同参数：
```
zh_plain vs zh_dflash2 : DIFFER at 29/96
num_plain vs num_dflash2: IDENTICAL
K 扫描: K=1 偏@62，K=3 偏@62（与 K=1 同位置同 token），K=7 偏@29
```
⇒ K=1 即偏、且与 K=3 同位置 ⇒ **不是注意力掩码的跨列污染**，而是**与草稿无关的每轮状态/参数偏差**
（paged KV 槽写入/回滚、GDN `RecordForReplay` 一致性、anchor/bonus 记账）。
按契约 §7.2 的 greedy 判据 `d_i == argmax(p_(i-1))`，差异只可能来自
**verify 列上的 target logits 与 plain 同上下文 logits 不等**，而接受逻辑本身无嫌疑
（licensed 的 token 必然与 target argmax 相等，这是结构保证）。

## 四、审计优先级（下一步，按收益排序）
1. **FP32 化 selector 分数**（2.4）：把 `logits` 与 selector 的 unary/scores 提到 FP32，与参照一致；
   可与其它改动合批，一次编译。
2. **verify 状态链**：审 `speculative_accept_greedy_drafts` 的列→licensed 映射、
   `speculative_select_accepted_hidden` + `scatter(state_destination_slots, continuation_hidden_store)`
   对**被拒列**的 KV 槽回滚、以及 GDN replay 的一致性；判据 = spec 流与 plain **逐位一致**。
   交叉验证：`--spec mtp --draft-tokens 3` 是否也在**同一 token 位置**偏（若同位置 ⇒ 共享路径）。
3. **参照草稿 A/B**（`data/draft_dflash2_ref/` 已在盘上，等你点头）：一次区分"引擎 vs 训练 ckpt"。
