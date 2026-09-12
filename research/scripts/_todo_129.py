#!/usr/bin/env python3
"""§129：dspark 低接受率的根因（block 列错位）—— 训练约定 vs 引擎实现。"""
import datetime
import pathlib

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
T.open("a", encoding="utf-8").write("""
### 129. **dspark 低接受率的根因找到并已修**：block 列错位一行（""" + stamp + """）
**根因（训练约定 vs 引擎实现直接冲突，实证）**：
- 训练脚本 `train_dspark.py`：
  - `:428` `block_pos[r] = torch.arange(a, a+B)` ⇒ **第 0 列 = anchor（位置 a 的已提交 token）**；
  - `:168-169` `logits = lm_head(out[:, 1:])` / `return logits, out[:, 1:]` ⇒ **输出只用 columns 1..k**；
  - `:430` `teacher_idx[r, p-1] = tok[a+p]` ⇒ **每一列预测"它自己那一列"的 token**（对齐/自预测）。
  ⇒ 推理时必须取 columns **1..k**（跳过 anchor 列）。
- 引擎 `dflash_impl.h:448-453`（修复前）：
  `std::size_t source_column_offset = 0; if constexpr (!Config::bf16_weights) { source_column_offset = 1; }`
  ⇒ **只有"非 bf16"才跳过 anchor 列**；而我们的 dspark 草稿正是 **bf16**（S47/A5 审计：55/55 张量与 checkpoint
  逐字节相同）⇒ 走 offset=0 ⇒ **把 anchor 列当第一个草稿槽** ⇒ 整块草稿错位一列 ⇒ p(pos0) 崩到 26.8%。
  （注：`if constexpr (!bf16_weights)` 那条注释写的是 "Legacy DFlash keeps the anchor column out of the
  proposal rows" —— 即"跳过 anchor"才是设计意图，bf16 分支反而没跳。）
**修法（已落地）**：`source_column_offset` 恒为 1（跳过 anchor 列），并把命名保留成常量以便日后配置化；
注释里写明训练约定的出处（`train_dspark.py:168-169,428-432`）。
**这条能解释**：dspark p(pos0)=26.8%（显著低于 dflash2 41.8%、MTP 53.5%）、维持 11.22% 的整体接受率、
以及历史上"dspark 在 k=1 时 0%"的记录（错位一列时 k=1 的那一个草稿完全错位）。
**验证（编译完成后立刻做）**：跑 `_dspark_posverify.sh`（CLI `--spec dflash --draft-tokens 7` + 位置剖面）。
判据：p(pos0) 从 26.8% 明显上升（接近 dflash2/MTP 一档）即确认；若不动则根因判断错误，回退并重查。
**顺带证伪**：A5 的首选候选 A（YaRN）**不成立**——`train_dspark.py:70-75` 用的是纯 rope，且它自己写出的 config
是 `'rope_type': 'default'`（`:548`）；artifact 旁那份 `yarn/factor 32` 是**基座 Qwen3.8-27B 的遗留字段**。
""" )
print("_TODO.md §129 已追加")
