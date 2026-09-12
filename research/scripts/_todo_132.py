#!/usr/bin/env python3
"""§132：A2 结论 + dflash2 两条根因修复就绪 + 三处 bug 总账。"""
import datetime
import pathlib

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
T.open("a", encoding="utf-8").write("""
### 132. A2 判决：草稿质量是主因；dflash2 两条根因均已修，重训前置条件就绪（""" + stamp + """）

**A2（草稿上限，CPU-only，679 anchors × 3 ckpt，`_collab/A2_draft_ceiling.md`）的三个答案**：
| 问题 | 数字 | 判读 |
|---|---|---|
| Q1 位置 0 的上限 | 引擎 mask 口径 **0.0309 / 0.0427 / 0.0000**（step1200/1800/400）；训练器真 token 口径 0.1708/0.1222；最有利口径也仅 0.2504 | 位置 0 上限**远低于** 41.8% |
| Q2 形态 | 引擎（41.8%→47.8%）与离线（0.171→0.474 / mask 0.031→0.524）**同形**；可比 prompt 上量级也接近（引擎 zh 4.51% ≈ 离线 mask 3.1–4.3%） | **无证据表明 verify 在丢好草稿**（与我的 token 级实验一致） |
| Q3 对齐 A/B | **shift=1 系统性更高**（0.2504 vs 0.1708）⇒ ckpt 学的是 shift=1，引擎要 shift=0 | **M1 确认** |
- **M2（新，A2 报出）**：引擎喂 `[anchor, mask×7]`，`train_dflash2.py` 喂**真 token**；语料里 mask id 出现 **0/4468** ⇒ mask embedding 从未被训练；mask 口径命中率低 **5.5×**（0.031 vs 0.171）。
- **额外的强证据**：仓库内**另外两个参考训练器都是「mask 块 + shift 0」**（`train_dspark.py:417-432`、`dl/aeon-train_head.py:25-60`）
  ⇒ **dflash2 的训练脚本是全仓唯一的异类**（真 token + shift 1）。
- **量化外推**：教师 top-16 熵地板 1.3654 nat，当前 loss = 地板+0.48；按 (loss,命中) 外推到地板，位置 0 也只有
  ≈0.10（引擎口径）⇒ **旧配方下 0.9 不可达**；新配方（mask + shift 0）必须重训后再测。

**已就绪的修复（`train_dflash2.py`，三处）**：
1. `--target-shift` 默认 **1 → 0**（M1；注释写明 ids16 口径实测：P(next)=0.2967 vs P(self)=0.0007）；
2. **输入模式与推理一致**：block 第 0 列 = anchor、1..B-1 列 = mask（新增 `--mask-block` 默认开，
   `--no-mask-block` 可复现旧配方做 A/B）；
3. `MASK_ID = 248077`，与引擎 `src/targets/qwen3_6_27b/impl/config.h:106` 的 `mask_token` **逐字一致**
   （A2 的离线实验用的是 248070，方向不变但数字偏悲观，已记录）。
**重训脚本**：`_train_df2_shift0.bat`（同 W9 配置 + `--target-shift 0`，从零开始；mask 输入走新默认）。
预计 ~2.1 小时到 1900 步；**等用户放行训练**。

**"取列/目标偏移"家族总账（用户问"别的有没有"）**：
| # | 位置 | 问题 | 状态 |
|---|---|---|---|
| 1 | `dflash_impl.h:448`（dspark 推理） | 取列含 anchor（应跳过）⇒ 整块错位一列 | **已修**，等二进制验证 |
| 2 | `train_dflash2.py`（dflash2 训练） | 目标行晚一行（shift 1）+ 输入模式用真 token | **已修**，待重训验证 |
| 3 | `mtp_round.cuh:47`（MTP） | `ar_valid_columns = s+1<next`（参照量存疑 + 边界差一） | **嫌疑待实验**（不凭读码就改） |
""" )
print("_TODO.md §132 已追加")
