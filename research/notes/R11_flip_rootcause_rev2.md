# R11：翻转成因的再定位（graph 排除 + MTP 链反证）（2026-09-11 15:0X）

## 一、本轮两个决定性实测

### 1. CUDA graph 捕获/回放被排除
同一 prompt，开/关 graph 逐位相同：
```
dflash2_graph   vs plain: DIFFER at 29   dflash2_nograph vs plain: DIFFER at 29（同值）
mtp3_graph      vs plain: DIFFER at 40   mtp3_nograph    vs plain: DIFFER at 40（同值）
```
⇒ spec≠plain 的偏移与 graph 无关（graph 路径无罪）。

### 2. **MTP 在同一 verify 路径上拿到活链** —— 这条推翻了 R10 的结论
```
mtp3   位置剖面 = [29, 16, 5]     条件接受率 p1=16/29=55%, p2=5/16=31%
dflash2 位置剖面 = [6, 1, 0, ...]
```
同一 target、同一 verify/轮次机制、同一二进制 ⇒ **verify 路径不是接受率的限制器**：
它能支撑多 token 链（MTP 就是证据）。spec≠plain 的偏移确实存在（是 G-A 层面的瑕疵，该修），
但它**不封顶**接受率。⇒ R10"缺口在 verify 块与顺序解码不等价"需要降级为"次要瑕疵"。

## 二、修正后的定位：限制器是 **dflash2 草稿块的第 1..K 位**
- dflash2 的第 1..K 位正是**吃 mask token** 的列；MTP 的草稿机制（markov/链）不依赖 mask 块。
- `_collab/A2_draft_ceiling.md`（既有记录）：该 ckpt 的训练**喂真 token**，
  mask id 在训练语料出现 **0/4468** ⇒ 推理喂 `[anchor, mask×K]` 时，第 1..K 位是**分布外输入**
  ⇒ 这些位置的 hidden 必然不可用 ⇒ 链在位置 1 断。
- 这与全部实测吻合：列 0（不依赖 mask）正常 85%；可复制文本上 unary 独自够用 ⇒ 6/7 列正确；
  边项扫描无益（边项吃的是第 1..K 位的 hidden）；K 无关（掩码/结构无病）。

### 顺带修正一条我自己的推论
"mask embedding 行未训练"**不成立**：草稿训练时该行是**冻结的固定向量**，行本身训不训练无所谓，
真正的错配是**输入模式**（真 token vs mask）。这也解释了为什么早前"换 mask token id"实验无效。

## 三、待你一句话就能定论的问题（决定"引擎 vs ckpt"）
你说干净环境那套 vLLM（支持 dflash2）跑 dflash2 时接受率正常（记得 50 几）。**它加载的是哪份草稿？**
- 若是 `data/draft_dflash2_ref/`（Sep 11 12:32 在盘上那份 HF `DFlash2DraftModel`）
  ⇒ **本诊断成立**：我们用 mask 块推理 step_006000（训练喂真 token）才是塌陷主因，
  引擎侧只需补 G-A 保真（FP32 分数）与 retrain 后复测。
- 若它加载的是我们自己的 ckpt（`data/dflash2_ckpts/step_*.pt` 或我打的 artifact）
  ⇒ **本诊断被推翻**，差异回到引擎某处我尚未定位的管线参数，我按 R10 的探针继续挖。

> 之所以必须问：`data/draft_model/`（vLLM 日志里出现的草稿）是 **dspark** 模型
> （`Qwen3DSparkModel`，mask 190221 / rope 1e6 / layers[1,14,29,44,57]），与 dflash2 不是一回事，
> 不能当 dflash2 对照；而 `data/draft_dflash2_ref/` 恰好是**训练过 mask 块**的参照 dflash2 草稿。

## 四、引擎侧待办（不依赖上面的答案）
1. selector 分数 FP32 化（对齐参照，消除 BF16 候选顺序抖动）——合批做。
2. spec==plain 的偏移（G-A 瑕疵）：按 R10 §四(A) 打 logits/margin 判别探针，
   定"数值路径 vs 状态回滚"，再决定改算术还是查 GDN replay。
3. retrain 就绪后（`_train_df2_shift0.bat`，mask 输入 + shift 0）→ 三闸门出新 artifact →
   用 `NINFER_DF2DBG` + `_df2_blame2.py` / `NINFER_DF2SEL` / `_verify_df2head.sh` 复测。
   验收判据不变：链式剖面（p1+ 非零）、AL ≥ 3、G-A IDENTICAL。
