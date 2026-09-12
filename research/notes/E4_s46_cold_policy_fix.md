# S46 (agent E4): S34 冷策略 F3b 修复 —— 空闲窗不得移动 EWMA / 低置信窗不得计入 streak

日期 2026-09-10 · 被测物 `/home/user/ninfer-fusion/src/serve/kv_cold_policy.h`（建树副本，md5 **7662e4ad**，295 行，本轮**未改动**；修复以补丁交付）
产物：`E4_s46_cold_policy_fix.diff`（`patch -p1`，`a/src/serve/kv_cold_policy.h`，96 行/4 hunks）· `E4_s46_f3b.cpp`（我自己的最小复现驱动）· `E4_s46_regress.cpp` + `E4_s46_checker_fixed.h` + `E4_s46_mkchecker.py`（不变量回归，镜像与被测头同步）· `E4_s46_run.sh`（一键复跑）· `E4_s46_run_log.txt`（314 行原始日志）
约束：plain `g++ -std=c++20 -O1`（WSL `/tmp/e4/s46`，`-Wall -Wextra` 零告警）、**零 GPU**、**零 `src/` 写入**（补丁只做 `patch -p1 --dry-run` 对真建树 + 影子树实际应用；应用后建树 md5 仍为 7662e4ad）。宿主机内存 < 0.3 GB。

## 0. 结论（三句）

1. **F3b 已闭合**：空闲窗（`window_from_cumulative` 在 `count1<=count0` 时返回 `(mean 0.0, rounds 0)`，`:79-83`）现在既不动 EWMA、也不计入 streak；真能量恒定 0.5、切点 0.4 的"受害者层"从 **t=4 被降级** 变为 **永不降级**（我自己的驱动，before 3 FAIL / after 0 FAIL）。
2. **不变量无回归**：C 原测（未改）**TU_RESULT PASS**（demotions 22→23，同一条流）；D 对抗套件（未改）在**同步镜像**下 **0 字面违反 / 0 重推导分歧 / cut 差 5e-07**；随机 200×300 中"低置信窗计入 streak"这一类 spec-gap **清零**（原因普查 100% 只剩 F2 的"未观测窗"）。
3. **抖动结论不变**（item 4）：F1 锚定族 cap=4/5 仍 **318 flips/400c（最坏 5724 s/h）**、F2 单槽族 cap=1 仍 **263（4734 s/h）**，before/after **逐字节相同**（`diff` 空）⇒ "> 3600 s/h ⇒ 永久 reload 循环、必须上逐层翻转退避" 原样保留；本修复**未**触碰该机制。

## 1. 复现（before，我自己的最小驱动）

几何：16 层、deep_frac 0.2（deep_start 12）、cap=4、stable=2、dwell=3、floor=32、α=0.5。L0..L3=0.1/0.2/0.3/0.4（冷尾，切点=12 个非深层 EWMA 的第 4 小 = **0.4**）、L4=100、**L5=0.5（真能量恒定、按设计在切点之上）**、L6..L11=100、L12..15 深层。L5 在 t=1..3 空闲 —— 观测直接由被测头自己的 `window_from_cumulative(m*64,64,m*64,64,…)` 产生（= live tap 原路径），不是手搓的合成输入。

```
BEFORE (7662e4ad)  t=1 fed(L5)=(mean 0.000, rounds  0) ewma=0.2500 cut=0.300
                   t=3 fed(L5)=(mean 0.000, rounds  0) ewma=0.0625 cut=0.300 cold={0,1,2,}
                   t=4 fed(L5)=(mean 0.500, rounds 64) ewma=0.2812 cut=0.300 cold={0,1,2,5,}  <-- 降级 L5
  at that decision: cut=0.3000, L5 ewma[no-guard]=0.2812 (<= cut -> 降级被解释), ewma[guarded]=0.5000
A 控制（无空闲）PASS / B 单次空闲 FAIL first=4 / C 持续交错空闲 FAIL first=4 flips=2/400c / D 低置信窗载 streak FAIL first=2
AFTER  (2b7d33ec)  同一条流：t=1..6 cut 恒 0.400，cold={0,1,2,3}，L5 永不降级 ⇒ 4 项全 PASS
```
与 D §4.3 独立数字**逐条一致**（t=4、cut 0.300、ewma 0.28125、`{0,1,2,5}`、持续族 flips=2/400c）。
三个同时发生、彼此叠加的效应（都在同一根因上）：① 空闲窗的**伪造 0.0 均值**以全权重进 EWMA；② 中毒的 EWMA 同时**把切点从 0.4 拉到 0.3**，于是真冷层 L3(0.4) 被判为"高于切点"而**被提升出池**，空出的槽位让 L5 进来（`cold={0,1,2,}`→`{0,1,2,5,}`）；③ streak 用中毒值比切点，也在涨。

## 2. 修复：谁在驱动哪个效应（行号为原 295 行版）

| 输入字段 | 原代码中被谁消费 | 结论 |
|---|---|---|
| `window_mean_l`（mean） | `:121-124` EWMA 混合 → 经 `cold_cut :127` 与 `:133` 比较 | **mean 独家驱动 EWMA、切点、streak、候选值** |
| `window_rounds`（rounds） | `:147-149` 证据快照；`:153` 决策窗候选门；`:285-290` invariant 复核 | rounds **只在"本窗"被读**；EWMA/streak 的**历史**里从未被读 |

⇒ "空闲 = 能量 0 的真观测"这一个错误假设，被 EWMA 与 streak **两个消费者**同时吃掉；只堵一个都不够 —— D 的提案 A（只改 streak）在 D 自己的 RUN 3b 里只把降级从 t=4 推迟到 t=5（该数字出自 D 报告；本轮**未**重跑 D 的提案头，D 的提案补丁本轮未应用）。

补丁 4 处（新浪号见 diff）：

1. **EWMA 置信门**（新 `:142/:145`）：`obs.window_rounds == 0` 的窗口**不参与 EWMA 更新**（`continue`，不建立也不混合）。这是 tap 的"无新轮次"伪造值；**真零能量窗（rounds>0、delta sum=0）仍正常平滑** —— 该门只区分"没信息"与"测得 0"。
2. **streak 置信门**（新 `:162-171`）：`rounds < min_window_rounds` 的窗口把 streak **清零**（不是"跳过"）—— 规范要求"在**每一**个最近 stable_cycles 窗都 ≤ 切点"，一个无法作证的窗必须打断 streak。
3. **提升门补丁**（新 `:233/:236`）：`observed_now`（只看"是否出现"→ 改成 `confident_now`（是否**置信**出现）。**这是第 2 处改动的必要伴随**：2 之后"streak==0"也会由低置信窗产生，而提升条件正是 `streak <= 0` ⇒ 不加这一处，**层一空闲就会被踢出冷池**（把 F3b 从"误降级"换成"误提升"）。
4. 注释/文档：结构体字段注释改为 "consecutive CONFIDENT windows"（新 `:31`），头注补 S46 段（含"F2 仍开"的显式声明）。

**为什么 EWMA 门用 `rounds == 0` 而不是 D 提案 A+ 的 `rounds < floor`**：切点是对 EWMA 映射取分位（`:89-103` 遍历 `state.ewma`），而 `state.ewma` 的成员资格又由更新循环决定（`:118-126`）。把 `0<rounds<floor` 的观测也排除出 EWMA，会让**长期 thin 的层退出切点集合**（cut 随之改变），即一个**超出本次缺陷范围**的策略语义变更（F2 家族里 10 个 rounds=10 的层正是"塑形切点但从不候选"，结构上必然受影响）。本修复选择**最小爆炸半径**，并在 §5 把它列为需 C/规范所有者裁决的开放项；两种门对 F3b 本身的修复效果相同。

## 3. 回归（a）F3b（b）C 原测（c）独立检查器不变量

**(a) F3b**：`f3b_before` E4_S46_F3B_RESULT **FAIL (fails=3)** → `f3b_after` **PASS (fails=0)**（B/C/D 三项，见 §1）。

**(b) C 的 property test（`_collab/C_s34_cold_loop/C_s34_policy_test.cpp` 未改，md5 1c0c0f95）**：两侧 `TU_RESULT PASS`（15/15）。唯一差异：`property: 12x200-cycle randomized invariant … PASS decisions=2400 demotions=22`（before，正净增长口径）→ `demotions=23`（after）。C 自己的预期断言（工作例 {2,5} 第 4 周期降冷、L5 离切点即出池）两侧**均 PASS**（该流 rounds 全 64，置信门不触发）。

**(c) D 的对抗检查** —— 分两个镜像跑，因为 D 的检查器内建的是**修复前**语义：

| 运行 | 结果 | 判读 |
|---|---|---|
| `D_s39_adversarial.cpp`（未改）× before 头 | `D_S39_ADVERSARIAL_RESULT PASS (fails=0)`，随机 `dem=12837/prom=12421/flips=18243/gap=6110(136流)/cut_div=5e-07/分歧=0` | 与 D RUN 1 数字**逐条一致**（本轮重放成立） |
| 同上 × after 头（**镜像已过时**） | `FAIL (fails=4)`：`8b first=4→5`、`11 letter clean`、`11 cut_div=260.5`、`11 mirror streak 分歧=5601` | 4 条全是"检测器写死了未修语义/镜像随之失配"：`8b` 与 D 提案头同值（D RUN 6 亦 FAIL first=5）；后三条是**镜像的 EWMA/streak 不再等于被测头**的必然结果（D 自己就用 `diag_streak_disagreements=3125` 作为"提案改了 streak 语义"的佐证） |
| `E4_s46_regress.cpp` × after 头 + **同步镜像**（`E4_s46_checker_fixed.h`：对 D 的检查器**只**改 3 处——镜像 EWMA 跳 `rounds==0`、`streak_code` 加 `&&confident`、streak 块的 `ewma[]` 改成 `find()` 防 0.0 插入；由 `E4_s46_mkchecker.py` 逐处断言生成，可 diff 复核） | `E4_S46_REGRESS_RESULT PASS (fails=0)`：13 项不变量全 PASS；随机 200×300：**0 违反流**、`mirror streak 分歧=0`（9030 次降级逐次一致）、`cut_div=5e-07`（仅浮点） | **不变量无回归** |

**spec-gap 原因普查**（同一 200×300 流，检查器的 `spec_gap_detail` 采样上限 8 条/流，故采样数 < 总数）：before 合计 6110，采样 = `unobserved×535 + thin×526`（thin 含 `rounds 0`（空闲伪造值）×165、`rounds 3`×178、`rounds 20`×102、40..63 若干）；after 合计 **1389**，采样 = **`unobserved×832`，thin 类 0 条**。即"低置信/空闲窗载 streak"整类归零（按构造只剩 F2 一类：两个镜像的差别仅在缺席窗处理），剩下的**全部**是 F2 的"缺席窗冻结"（本轮显式不修）。

**D 的验收门 `D_s39_gate.cpp`（未改，额外情报）**：before `D39_GATE_FAIL (fails=3)`；after **`fails=2`** —— **G2（低置信窗不得承载降级）由 FAIL 转 PASS**，G4a/G4b/G4c 全 PASS（`at=3` / `promote5_at=5` / letter clean），余下 G1=F2（未修）与 G3=抖动（未修）。

## 4. item 4：本次修复**不改变**抖动（churn）结论

同几何、同驱动器两侧对拍（`diff` 为空 = 逐字节相同）：

| 族 | before | after | 最坏 s/h |
|---|---|---|---|
| F1 锚定 cap=4 / cap=5 | 318 flips/400c | **318** | 5724（不变，仍 > 3600） |
| F2 单槽 P=6 hi=12 m=5 cap=1 | 263 | **263** | 4734（不变） |
| F2 其余 cap=2/3、P=8、hi=6/40 | 1/100/264… | 同左 | 同左 |
| F4 热台地 cap=0..3 | 0/1/1/1 | 同左 | 同左 |

原因（结构性，不是巧合）：抖动族的观测**全是 rounds=64 的置信窗**，两处新门都不触发；且本修复**不触碰**提升侧的不对称滞后（`promote :196 streak<=0` vs 降级 stable+dwell）—— 那才是 F4/F5 的机制。**结论原样外包**：`> 3600 s/h ⇒ 系统追不上自己的重排`，live 接线前仍需逐层翻转退避（D §5/§6，非本任务）。

## 5. 明确边界（未修 / 代价 / 待裁决）

- **F2（缺席窗冻结 streak）仍开**：`:129-138` 的 streak 循环只遍历 `observations`，缺席层条目不动；本轮**未触碰**（不在任务给定范围内）。最小修法=改为遍历 `state.ewma`（D 提案 A 已含）；门 G1 因此仍红。
- **8b 的代价**：低置信窗现在会清 streak ⇒ 当**决策窗**恰好低置信时，合法降级可能晚 1 窗（60 s = `NINFER_FT_RELOAD_SECS`）。实测 `8b` 4→5（与 D 提案头同值），C 的工作例（全置信流）不受影响。
- **A+（`rounds<floor` 不进 EWMA）未采纳**，理由见 §2；如需采纳，属独立语义变更，应与 F2 一起决策。
- 未做且未声称：live 接线/GPU 验证；C 的 `net_growth` 计数口径（D F6）；`cold_cut` 内注释把非深层误写为 "deep layers"（`:95-96`，注释歧义，未改）。

## 6. 复跑

```
wsl.exe -- bash -lc "bash /mnt/c/Users/User/Documents/ziqinzhang/_collab/E4_s46_run.sh"   # 重建+重跑+重写日志, ~5 s
wsl.exe -- bash -lc "cd /home/user/ninfer-fusion && patch -p1 --dry-run -i /mnt/c/Users/User/Documents/ziqinzhang/_collab/E4_s46_cold_policy_fix.diff"   # 只读, 期望 checking file src/serve/kv_cold_policy.h rc=0
```
指纹：before `7662e4ad…`（建树，dry-run 后复验未变）→ after `2b7d33ec…`（影子树 `patch -p1` 应用后 `cmp` 与 `after/kv_cold_policy.h` 逐字节相同）；`D_s39_adversarial.cpp` a35d2991、`C_s34_policy_test.cpp` 1c0c0f95 均未改动。建树内另有 `_collab/E4_s46_f3b_driver.cpp`（15:34，上一轮中断残留、本轮**未使用未验证**），本报告证据一律来自 §0 列出的本轮产物。
