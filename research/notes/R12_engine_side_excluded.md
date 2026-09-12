# R12：排除"引擎侧"可能性（2026-09-11 15:1X）

## 结论
**引擎侧被排除**：引擎已经把该 ckpt 的能力**榨干**了；限制器是 ckpt 的训练/推理输入口径错配
（A2 的 M1 shift 差一位 + M2 真 token vs mask）。以下 5 条都是实测/既有量化记录。

### E1 引擎实测 p0 ≈ 26–35% ≥ 该 ckpt 的离线上限 ≈ 25–31%
A2 §Q1 记录（同一 ckpt、679 anchors、两种输入口径离线跑）：
- 引擎真实输入口径（`[anchor, mask×7]`）：位置 0 命中 **0.0309 (21/679)** @step1200、**0.0427** @step1800；
- 训练脚本口径（真 token 块）：0.1708 / 0.1222；
- 连"按脚本自己的目标 `ids16[a+1,0]`"也只有 **0.2504 / 0.2047**。
A2 原话：**"位置 0 的上限远低于 41.8%；即便按对自己最有利的口径也只有 0.25"**。
今天引擎实测位置 0 接受率：修 walk 前 8/23=34.8%、修后 6/23≈26% ⇒ **已达/超过该 ckpt 的上限**
⇒ 引擎没有在丢好草稿。

### E2 A2 §Q2：引擎接受率 ≈ 草稿离线命中率（实测一致性）
A2 原话："**没有证据表明 verify 在丢好草稿**（引擎接受率 ≈ 草稿离线命中率）；41.8% 那一档
属于更容易的 prompt 组合，纯文本样本复现不出"。⇒ 引擎忠实实现了草稿质量。

### E3 可复制文本上，引擎的草稿在 mask 列 1..7 上是**对的**
今天实测（数字模式 prompt）：单轮 drafts = `[220,16,220,17,220,18,220]` vs target argmax
`[220,16,220,17,248046,18,220]` ⇒ **6/7 列（含全部 mask 列）正确**。
⇒ 引擎的草稿管线（context K/V、块构造、绝对 positions、dynamic conv、final_norm、
proposal head、selector）在该场景下**可证明地正常工作**——若有引擎级缺陷，这些列也会错。

### E4 共享的 verify/轮次机制能支撑多 token 链
今天实测同一二进制、同一 target：`mtp3` 位置剖面 **`[29,16,5]`**（p1=16/29=55%、p2=5/16=31%），
而 `dflash2` 是 `[6,1,0,…]`。⇒ verify/accept/轮次记账本身不是限制器（MTP 走同一条路）。

### E5 引擎侧参数/结构逐项已核
- 声明型超参 vs ckpt 自带 `config.json`：逐项一致（R8 表）。
- 块构造（`[anchor, mask×7]`、`positions=F+min(i,V-1)`）、context positions（target 块绝对位置）、
  selector walk 语义（行=previous、列=C_step、previous 链）：与契约/参照一致（R9）。
- 已排除：CUDA graph 回放（开/关逐位相同）、KV 精度（bf16 实测、int8 同形）、
  `rope_delta`（纯文本为 0）、`apply_final_logit_policy`（qwen3 编译期 no-op）、
  跨列掩码污染（K=1 就偏，K 无关）。

## 因此：
1. **限制器 = ckpt 的训练口径**：`train_dflash2.py` 喂真 token（mask id 在语料 0/4468）+ 默认
   `--target-shift 1`，而引擎推理喂 `[anchor, mask×7]` + shift 0 ⇒ 推理时第 1..K 位是分布外输入。
   A2 的 M2 表：mask 口径命中率比真 token 口径低 **5.5×**（0.031 vs 0.171 @step1200；step400 为 0/679）。
   ⇒ 修法就是已备好的 `_train_df2_shift0.bat`（mask 输入 + shift 0）；产出新 ckpt 后按
   `_collab/C_artifact_manifest.md` 的 patch→verify→round-trip 三闸门出 artifact，再用现成仪器复测。
2. **引擎侧只剩保真度项**（不影响接受率上限，但按契约该修）：
   a) selector 分数 FP32 化（参照自注 BF16 会改变候选顺序）；
   b) spec 流 vs plain 的偏移（G-A 瑕疵）：R10 §四(A) 的 logits/margin 探针定"数值 vs 状态"。
