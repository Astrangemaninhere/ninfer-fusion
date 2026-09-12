#!/usr/bin/env python3
"""§131：MTP 的 ar_valid_columns 嫌疑（待实验，不凭读码就改）+ dflash2 重训方案就绪。"""
import datetime
import pathlib

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
T.open("a", encoding="utf-8").write("""
### 131. MTP 的 `ar_valid_columns` 嫌疑（**待实验，不凭读码就改**）+ dflash2 重训方案（""" + stamp + """）

**MTP 嫌疑（代码级，语义未确证）**：`mtp_round.cuh:47`
```cpp
const int licensed       = licensed_counts[row];        // 本轮已发布 token 数（a+1）
const int remaining      = remaining_budgets[row] - licensed;
const int budget_extent  = remaining > 1 ? remaining - 1 : 0;
const int context_extent = max_context - updated_frontiers[row] - 1;
int next = min(budget_extent, context_extent);          // **下一轮**的 draft 上限
...
ar_valid_columns[offset] = s + 1 < next ? 1 : 0;        // 第 s 步（位置 frontier+s）
```
- 若该列语义是"位置 frontier+s 的 token 已存在"⇒ 判据应是 `s < licensed`（**本轮已发布数**），而不是 `next`
  （下一轮的 extent）；而且 `s + 1 < next` 比 `s < next` 还**少一位**（s = next-1 被误判无效）。
- ⚠️ 但 `ar_valid_columns` 也可能不是这个语义（可能只是给 attention 的可见列掩码）⇒ **不凭读码就改**：
  MTP 是当前最健康的档（p(pos0)=53.5%），凭空改可能更差。
- **实验设计（便宜、可判）**：把 `s + 1 < next ? 1 : 0` 改成 `s < next ? 1 : 0`（或 `s < licensed ? 1 : 0`，
  两版各跑一次），用 CLI 的 `accepted by pos` + `spec_accept_rate` 对比：
  若两者与现状**逐位相同** ⇒ 该量在当前配置下不生效（无害）；若上升 ⇒ 是 bug；若下降 ⇒ 回退。
  需一次编译（单行改动，但 `mtp_round.cuh` 会牵动 ops TU）+ 一次 GPU 窗口。

**dflash2 重训方案（已写脚本，等用户放行训练）**：
- 现有 checkpoint 是 `--target-shift 1`（晚一行）训出来的 ⇒ 要吃到修复的收益必须**重训**（不能靠微调）。
- 脚本 `_train_df2_shift0.bat`：与 W9 同配置（`--steps 6000 --batch-seqs 4 --anchors-per-seq 12 --max-ctx 128
  --lr 6e-4 --gpu-frac 0.98 --save-every 100`）**加 `--target-shift 0`**，**从零开始**（便于与 W9 的 step 对齐比较）。
- 判据：训到 ~1200-1900 步后，用 CLI 的 `accepted by pos` 对比 `p(pos0)` 与整体接受率（当前 dflash2：41.8% / 10.03%）；
  同时可用 `_df2_shift3_probe.py` 复核 top-1 与"自己那一行"的一致率是否随步数上升。
- 预计：0.25 steps/s ⇒ 到 1900 步约 2.1 小时。
""" )
print("_TODO.md §131 已追加")
