#!/usr/bin/env python3
"""§133：四处偏移类 bug 全部修完 + MTP 那处的性质（零风险等价 + 边界恢复）。"""
import datetime
import pathlib

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
T.open("a", encoding="utf-8").write("""
### 133. 偏移/约定类 bug 总账：四处，全部已修（""" + stamp + """）
| # | 位置 | 问题（依据） | 影响面 | 状态 |
|---|---|---|---|---|
| 1 | `dflash_impl.h:448` dspark 推理 | 取列含 anchor 列（`if constexpr (!bf16_weights) offset=1`，而草稿是 bf16）⇒ 整块错位一列（训练约定见 `train_dspark.py:417,427-432`：block[0]=anchor、每列预测自己） | **每轮全部 7 个草稿** | **已修**（offset 恒 1） |
| 2 | `train_dflash2.py` M1 | 目标行晚一行：`ids16[a + target_shift + i]`，而 `ids16[t]` 是"位置 t 的 next-token 分布"（实测 P(next)=0.2904 vs P(self)=0.0008）⇒ 对齐值应为 0 | **每个草稿的目标行** | **已修**（默认 1→0） |
| 3 | `train_dflash2.py` M2 | 输入模式 skew：训练喂真 token、推理喂 `[anchor, mask×7]`；语料里 mask id 出现 0/4468（A2）⇒ mask embedding 从未训练 | **每个草稿的输入** | **已修**（mask 默认开，`--no-mask-block` 可 A/B；`MASK_ID=248077` 与 `config.h:106` 逐字一致） |
| 4 | `mtp_round.cuh:47` MTP | `ar_valid_columns = s + 1 < next`：`next` 是**下一轮草稿数**，第 s 步有效应满足 `s < next`；原式在 `next == steps`（预算刚好够）时把**最后一个草稿**判无效 | 仅"预算恰好受限"的轮次少一个草稿；`next > steps` 时两式**恒等** | **已修**（`s < next`；零风险等价改动 + 边界恢复） |

**修复的正确性依据（家族一致性的正面证据）**：
- 仓库内**三个**训练器里，`train_dspark.py:417-432` 与 `dl/aeon-train_head.py:25-60` 都是 **mask 块 + shift 0** ⇒
  我改后的 `train_dflash2.py` 与它们**一致**；改之前它是全仓唯一的异类（真 token + shift 1）。
- MTP 的修复在 `next > steps` 时与旧式**逐位等价**（可用旧二进制与新二进制对同一 prompt 比对 token id 验证）。

**待验证（编译完成后自动跑，`_offset_family_verify.sh`）**：
dspark 位置剖面（对照修复前 `accept=11.22% / p(pos0)=15/56=26.8% / 1.39 tok/round`）+ dflash2/mtp3 作对照。
**dflash2 的重训**（`_train_df2_shift0.bat`，含 mask 输入的新默认）等用户放行训练。
""" )
print("_TODO.md §133 已追加")
