# R6：dflash2 塌陷的根因定位（2026-09-11 13:1X）

## 一句话结论
草稿侧的 unary 候选一直是用 **target 的 248320 行 LM head** 算的；而这个 checkpoint 自带
**专用草稿头 `text/draft_head [131072,5120]`（Q4G64）+ `text/draft_head_token_ids [131072]`**，
契约 §6.1（`ninfer-upstream/docs/maintainer/qwen3.8-27b-dflash2.md:254`）明确要求
「必须先经 `text/draft_head_token_ids` 映射为 global token ids，再访问 selector codebook」。
上游 `propose_dflash2_batch` **两种头都支持**（`dflash_impl.h:313-318`），我们**只实现了 Full 分支**，
并在 CLI 层硬性拒绝另一分支（`src/product/speculative_options.h:59-61`；上游同文件 DFlash2 分支**无此限制**）。

## 实证链（全部本机可复现，脚本随附）

### 1. 目标侧是健康的（此前所有"target 侧"怀疑可以排除）
工具：`_df2_blame2.py`（87 轮探针 + 无投机贪心真值流）
- 对齐方法本身可自证：投机解码只会 licensed 与 target argmax 相等的 token ⇒ 引擎的 licensed 流
  必然是无投机贪心流的前缀，每轮偏移 = 累计 licensed 数；探针自带的 `count/pos` 与该对齐一致。
- **列 0：target argmax 命中真值 17/20 = 85%**
- **列 1（在引擎已接受的前缀上）：target argmax 命中 6/8 = 75%**（参考实现保持 ~82%）
⇒ `TargetVerifyFrameView` 的位置 / KV / 特征捕获 / 块注意力 / `valid_columns` 语义全部正常。
（此前列的差异 #3「attention 第 6 参语义」、#4「`proposal_valid_columns` vs `attention_valid`」不再是嫌疑。）

### 2. 草稿侧：per-column unary 可用，**链式耦合项失效**
- 中文散文档：草稿退化成「同一 token 重复 3 连」——`[104647,104647,104647,1710,1710,1710,3709]`；
  更关键的是 **s=3,4,5 三轮 anchor 不同（98325 / 96041 / 96672）、上下文不同，却给出逐位完全相同的草稿向量**
  ⇒ 输出既不依赖 anchor 也不依赖上下文差异，只随"缓慢变化的某个量"漂移。
- 可复制文档（数字模式）：草稿 **6/7 列正确**
  `verify[0..7]=[15,220,16,220,17,220,18,220]`，`draft=[220,16,220,17,220,18,220]`，
  `argmax=[220,16,220,17,248046,18,220]`；且整轮内 licensed 出 ~8 个 token（说明这一轮草稿真的被大量接受）。
⇒ 块构造（`prepare_masked_block` 第 0 列确为 anchor，已逐行核对内核）、RoPE 绝对位置、上下文 K/V、
   `feature_projection`/`context_norm` 都在工作；**只有"把相邻位置串起来"的边项没起作用**。
   这也解释了"改 target-feature-layers 无差别"：失配来自头部，不在特征层集合。

### 3. 仓库级证据
- 草稿头在 **dspark 分支**是实现了的：`dflash_impl.h:489-499`（`optimized_proposal` + `ops::proposal_remap_token_ids`）。
- **dflash2 分支**（`dflash2_impl.h:328`）硬编码 `state.execution.model.output_head`，没有分支。
- artifact 清单实测（`_art_df2.py`）：`text/draft_head [131072,5120] Q4G64_F16S`、
  `text/draft_head_token_ids [131072] I32` 确实存在（另有 `text/output_head [248320,5120]`）。

### 4. A/B 实测（`_lmhead_ab.sh` / `_lmhead_ab2.sh`）
- dspark K=7 + `--lm-head-draft`：`pos=[15,1,0,0,0,0,0]` vs 不加 `[17,1,...]` ⇒ **无实质变化**
  ⇒ dspark 的第 1 位崩塌另有来源（markov 项，见下"并行"）。
- dflash2 + `--lm-head-draft`：**CLI 直接拒绝** —— `error: --spec dflash2 requires the full proposal head`。
  即该路径从未被走过（也顺带说明 `--spec auto` 与 `--lm-head-draft` 不兼容是既有约定，非 bug）。

## 为什么"用错头"会必然导致"链式失效"（机理，可证伪）
`E_i[p,c] = u_i[c] + Σ_r W_pred[pred_token,r] · g_i[r] · W_succ[C_i[c],r]`
`u_i` 与边项的**相对标度**是训练出来的：`W_pred/W_succ`（[248320,256] 两本 codebook）与 `g_i`
（`selector_hidden_projection`）是按**草稿头 logits** 的标度标定的。换成 target 的 LM head ⇒
`u_i` 量级/分布不同 ⇒ 边项被压到无效 ⇒ walk 退化为"每步取 unary argmax" ⇒
相邻列输出同一 token（重复）⇒ **第 1 位起全灭、第 0 位仍 ~35%**。该解释同时覆盖：
复制文档全对（unary 足够）、改特征层无差别、以及 **vLLM 同样塌陷**（vLLM 也只喂了 target 的 head）。

## 修复方案（可立即实施）
1. 绑定 `text/draft_head`(Q4G64_F16S, [131072,5120]) 与 `text/draft_head_token_ids`([131072] I32)
   —— dspark 的 `optimized_proposal` 绑定段（`bindings.cpp:447` 附近）可照抄。
2. 删除 `speculative_options.h:59-61` 的硬性拒绝（与上游对齐）。
3. `dflash2_impl.h` 增加 `proposal_head` 分支：Optimized 时在 **131072 行草稿头**空间做 stable top-16，
   再用 `draft_head_token_ids` 映射成 global ids，把 `(ids, scores)` 作为**预计算候选**交给 selector。
   这要求把现在的 `dflash2_selector`（内部自算 top-K）拆成
   「头部 top-K + id 映射」与「边格 + walk」两段 —— 正是此前审计的差异 #5
   （上游 `candidate_selector_path(candidates, scores, projected, …)` 的候选由外部给出）。
4. 验收判据：同一 prompt 的 per-position 剖面从 `[8,0,0,0,0,0,0]` 变成链式（p1+ 非零），AL ≥ 3（参考 ~4.0）。

## 并行线
- (b) dspark 第 1 位崩（`[17,1,0,…]`）与 dflash2 的边项**同构**：查 `dspark_markov_argmax.cuh`
  的耦合项标度（同一"耦合项失效"家族，dspark 用的是 `markov_w1/w2`）。
- (c) vLLM 侧核对它是否也只用 target head：若是，则跨实现塌陷由同一条解释。
