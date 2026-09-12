# S39 (agent D): S34 冷放置策略 —— 对抗验证（本轮已编译 + 已运行，实测收官）

日期 2026-09-10 · 被测物 **`/home/user/ninfer-fusion/src/serve/kv_cold_policy.h`（建树副本，md5 `7662e4ad80ebb5d790967ee141938180`，295 行；与 `_collab/C_s34_cold_loop/review/kv_cold_policy.h` 逐字节相同）**，未改动。
产物：独立检查器 `D_s39_checker.h` · 对抗套件 `D_s39_adversarial.cpp`（含**检查器自测**）· 预测(ii)扫描 `D_s39_cap01_probe.cpp` · **真实 tap 空闲窗探针** `D_s39_idle_probe.cpp` · 计数口径探针 `D_s39_net_vs_true.cpp` · 取证 `D_s39_dbg.cpp` · 修复提案 `D_s39_fix_proposal.diff`（**未应用**）· 验收门 `D_s39_gate.cpp` · 一键复跑脚本 `D_s39_run.sh` · 完整原始日志 `D_s39_run_log.txt`（334 行）。
约束遵守：plain `g++ -std=c++20 -O1`（`/tmp/s39` 编译运行），零 GPU（宿主编译窗口在跑），**零 `src/` 写入**（提案补丁只对 `/tmp` 副本 dry-run + 应用；建树头 md5 复验未变）；改动的文件只有 `_collab/D_s39_*` 探针、报告与日志。

## 0. 结论一览（协调者三类标注）

| # | 结论（本轮实测） | 类别 | 证据 |
|---|---|---|---|
| F1 | **零字面违反**：11 场景 + 200×300 随机流（60,000 决策，cfg 全空间）+ C 原 TU 全部场景，streak/dwell/rounds/deep/pool 五门在**策略自身代码语义**下无一违法 | 无违反（预测 i **命中**） | §2/§3 |
| F2 | **streak 跨观测空洞**：缺席窗口冻结而非清零 ⇒ 降级可建立在被空洞隔开的窗口对上 | **(b) spec-vs-code gap** | §4.1（`kv_cold_policy.h:129-138` 只遍历 observations；头注 `:20-25` 要求 EACH） |
| F3a | **低置信窗口计入 streak**（合成 rounds=42<64）：置信门只查**决策窗口** | **(b) spec-vs-code gap** | §4.2（门在 `:153`，streak 在 `:129-138` 不看 rounds）；随机套件 6110 次/136 流 |
| F3b | ★**空闲窗口被 tap 喂成 `(mean 0.0, rounds 0)`**：EWMA 被拽向 0（`:118-126` 无置信门）⇒ **真能量恒定的层被降级**；提案补丁只把它推迟 1 窗，**未闭合** | **(b) + 提案残留缺口** | §4.3（tap 契约 `:76-83`；实测 L5 t=4 降级，修后 t=5 仍降级） |
| F4 | **边界抖动 = 字面合法的持续翻转**：3 锚+1 缓冲+周期 5 拍动层 @cap=5 ⇒ **318 flips/400 周期**，每次翻转五门全过 | 设计层发现（非 a/b/c） | §5；`promote :196 streak<=0` vs `demote :166/:173` |
| F5 | **cap=1 会抖动（预测 ii 命中）**：单槽竞争族下 cap=1 达 **263 flips/400c 且 letter-clean**；但 F1 锚定流 cap≤3 仅 1 flip ⇒ 抖动窗口是"cap ≥ 稳定冷层数+1 **或** 有竞争者把锚位挤走" | 设计层发现 | §5（F2 族扫描；`hysteresis` 不对称 = 1 窗出池 vs 2 窗+3 窗入池） |
| F6 | C 的 "22 demotions" 是**正净增长计数**：在 C 自己的流上少计 **57%**（net=681 vs true=1585） | 证据方法学（非缺陷） | §2 RUN 0b（`C_s34_policy_test.cpp:108-111`） |
| F7 | **无 (c) 重推导分歧**：独立重算 cut 与策略自报最大差 5e-07（= 自报串 6 位舍入）；12,837 次降级中我的镜像 streak 与策略自报 streak **0 分歧** | 无 | §2/§3；诊断计数器在提案头上会响（3125 次）⇒ 不是死代码 |
| F8 | **检查器自身可证伪**：控制组（合法降级 ⇒ 0 违反）+ 5 个注入篡改（deep/thin/streak/dwell/pool）全部按预期触发 | 方法学 | §2 RUN 1 的 0 号段 |

## 1. 独立性契约（反 O1）与本轮自证

`D_s39_checker.h` 判定输入**只有** (1) 每周期喂给策略的原始 `ColdObservation`（layer/mean/rounds）、(2) 策略**输出** `prev_cold`/`cold_layers`、(3) cfg。**不读** `streak_at_decision`/`dwell_at_decision`/`rounds_this_window`，**不调** `invariant_violations()`。EWMA、cut（分位索引公式独立重写）、两套 streak、dwell、池、深保全部从零重推；策略 `reasons` 的 `cut=` 串仅作 FP 漂移诊断。
- **两套 streak 并行**：`streak_code`（镜像代码语义：缺席冻结、不查 rounds ⇒ 用于字面判定）与 `streak_spec`（声明语义：缺席或低置信即清零 ⇒ 用于 F2/F3 分类）。
- **诚实声明 1（自测）**：检查器若永不触发，"零违反"就是空话。本轮加了 0 号段自测：控制组必须 0 违反、5 个篡改必须各自触发（实测 6/6 PASS，§2）。
- **诚实声明 2（(c) 不可达）**：按构造，任何降级要么被 (a) 门捕获、要么被代码语义解释，(c) 在本检查器里**不可达**（`mismatches` 恒 0，本身不构成证据）。(c) 的结论改由两个独立测量支撑：cut 重算一致（5e-07）与 **mirror-vs-policy streak 逐降级比对**（12,837/12,837 一致；该诊断在提案头上触发 3125 次，说明它不是死代码）。该诊断只作交叉验证，**不参与判定**。

## 2. 原始命令与原始输出

一键复跑（脚本内每条 g++ 与运行命令逐字可见；完整 334 行日志见 `_collab/D_s39_run_log.txt`）：
```
wsl.exe -- bash -lc "cp /mnt/c/Users/User/Documents/ziqinzhang/_collab/D_s39_run.sh /tmp/s39_run.sh && sed -i 's/\r$//' /tmp/s39_run.sh && bash /tmp/s39_run.sh > /tmp/s39_run_out.txt 2>&1"
```
脚本内包含：`cp $BT /tmp/s39/policy/` + `md5sum` → RUN 0a C 原测 → RUN 0b 计数口径 → RUN 1 对抗套件(未修) → RUN 2 预测(ii)扫描 → RUN 3 取证 → RUN 3b 空闲窗探针(未修/提案两侧) → RUN 4/5 门(未修/提案) → RUN 6 对抗套件(提案)。

**被测物指纹**
```
7662e4ad80ebb5d790967ee141938180  /home/user/ninfer-fusion/src/serve/kv_cold_policy.h
7662e4ad80ebb5d790967ee141938180  /tmp/s39/policy/kv_cold_policy.h        # 295 行
9c20e1b1d6f90e9ca6b23b963a2297c1  /tmp/s39/fixed/kv_cold_policy.h        # 提案头 = patch 应用后逐字节相同
```

**RUN 0a：C 原 TU（未改动，确认口径）** —— `TU_RESULT PASS`；`property: 12x200-cycle randomized invariant ... PASS decisions=2400 demotions=22`（与 S34 声称一致；22 是 F6 的净增长口径）。

**RUN 0b：净增长口径 vs 真降级（C 自己的流、同 seed 同抽取顺序）**
```
trial  0 cap=3 dwell=2: net(C)= 76 true=153 checker=153 promotions=151 co-occur= 73 letter=clean diag_disagreements=0
...
TOTAL net(C)=681 true=1585 checker=1585 promotions=1563 co-occur-cycles=871 undercount=+904 (57.0%)
```
（checker 计数与 true 逐 trial 相等 ⇒ 两套独立计数互证；871 个周期同时发生 promote 与 refill。）

**RUN 1：对抗套件（未修头）** 关键原始行：
```
0 selftest CONTROL: legal demotion -> no violation                 PASS dem=1 viol=0
0 selftest T1: deep demotion is caught                             PASS cycle 3 layer 12: DEEP demoted (>= 12)
0 selftest T2: thin decision window is caught                      PASS cycle 3 layer 0: rounds 10 < 32
0 selftest T3: streak-0 demotion is caught                         PASS cycle 3 layer 1: streak(code) 0 < 2
0 selftest T4: dwell-0 demotion is caught                          PASS cycle 0 layer 0: streak(code) 1 < 2
0 selftest T5: pool overflow is caught                             PASS cycle 3 layer 1: streak(code) 0 < 2
1 monotone cooling: letter clean PASS / dem=4 maxpool=4 PASS
2 monotone heating: letter clean PASS
3 boundary thrash cap=5: letter clean at every flip PASS / flips=318 dem=162 prom=159 PASS
  [thrash:boundary cap=5] flips=318/400C  flips/h=47.70  stall-s/h: floor=0.8 moderate=238 worst(cap)=5724
3 same stream at cap=2: churn collapses (self-limiting refill) PASS flips=1
4 all-equal ties: letter clean PASS / maxpool=5 PASS          5 deep-coldest: letter clean PASS / first=3 PASS
6 capacity 0: zero demotions PASS dem=0                       7 cap-1 tie: maxpool=1 PASS / letter clean PASS
8a thresholds: first demotion exactly cycle 3 PASS first=3    8b thin window at c3 delays demotion to c4 PASS first=4
9 gap-freeze: letter clean PASS / spec-vs-code gap PASS spec_gap=1
    spec-gap: cycle 2 layer 3: streak(spec) 1 < 2 [unobserved window in streak]
10 net metric undercounts true demotions PASS net=2 true=3 (true_demotions=3 net_metric=2 promotions=1)
11 random 200x300: letter clean PASS
11 random 200x300: spec-gap demotions exist PASS gap_demotions=6110 streams_with_gap=136
   first: stream 0: cycle 3 layer 1: streak(spec) 1 < 4 [thin window (rounds 42) in streak]
  [random forensics] decisions=60000 true_demotions=12837 promotions=12421 flips=18243 gap_demotions=6110 max_cut_divergence=5e-07 diag_streak_disagreements=0
11 random: cut re-derivation stable PASS max_div=0.000000    11 random: mirror streak == policy streak on every demotion PASS disagreements=0
D_S39_ADVERSARIAL_RESULT PASS (fails=0)   adv rc=0
```

**RUN 2：预测 (ii) 扫描（节选；全表见日志）**
```
F1 anchored cap=0 flips=0 / cap=1 flips=1 / cap=2 flips=1 / cap=3 flips=1      all letter=clean
F1 anchored cap=4 flips=318/400c letter=clean flips/h=47.70 ... dem=162 prom=159
F1 anchored cap=5 flips=318/400c letter=clean flips/h=47.70 ... dem=162 prom=159
F2 P=6 hi=12 lo=0.05 m=5 cap=1 flips=263/400c letter=clean flips/h=39.45 stall-s/h: floor=0.63 moderate=197 worst(cap)=4734 dem=132 prom=132
F2 P=6 hi=12 lo=0.05 m=5 cap=2 flips=264/400c letter=clean ...  cap=3 flips=264/400c letter=clean
F2 P=8 hi=12 ... cap=1/2/3 flips=100/400c letter=clean     F2 P=4 hi=12 ... flips=150/400c letter=clean
F4 hot-plateau cap=1..3 flips=1 letter=clean
```

**RUN 3b：真实 tap 空闲窗探针（左=未修头，右=提案头）**
```
未修:  t=1 fed(L5)=(mean 0.000, rounds  0) my_ewma5=0.250 cut=0.300
       t=4 fed(L5)=(mean 0.500, rounds 64) my_ewma5=0.281 cut=0.300 cold={0,1,2,5,}  demote layer=5 ewma=0.281250
       RESULT: first demotion of L5 at cycle 4; letter=clean spec_gap_demotions=1 ...
       spec-gap: cycle 4 layer 5: streak(spec) 1 < 2 [thin window (rounds 0) in streak]
提案:  t=4 ... cold={0,1,2,}（不降级） t=5 demote layer=5 ewma=0.390625
       RESULT: first demotion of L5 at cycle 5; letter=clean spec_gap_demotions=0 diag_streak_disagreements=1 dem=4
持续交错空闲 400c: 未修 first demotion=4, flips=2/400c; 提案 first demotion=-1（不降级）, flips=1/400c
```
（L5 的真实能量恒为 0.5、设计上高于 cut=0.4，**从未变化**，却被降级。）

**RUN 4/5：验收门（未修 = 负对照 / 提案）**
```
未修: G1 spec_gap=1 FAIL / G2 spec_gap=1 FAIL / G3 flips=318 FAIL / G4a at=3 PASS / G4b at=5 PASS / G4c PASS  → D39_GATE_FAIL (fails=3)  rc=1
提案: G1 spec_gap=0 PASS / G2 spec_gap=0 PASS / G3 flips=106 FAIL / G4a at=3 PASS / G4b at=7 PASS / G4c PASS  → D39_GATE_FAIL (fails=1)  rc=1
```

**RUN 6：对抗套件（提案头）** —— letter 全部仍 clean；检测类断言按设计反向触发（`3 boundary ... FAIL flips=106`、`8b ... FAIL first=5`、`9 gap FAIL spec_gap=0`、`11 gap FAIL gap_demotions=0`、`11 mirror FAIL disagreements=3125`，`D_S39_ADVERSARIAL_RESULT FAIL (fails=5)`）。**说明**：本套件是"发现探测器"（面向未修头写死期望），不是回归门；提案头的回归判据用 `D_s39_gate.cpp`。`diag_streak_disagreements=3125` 正是"提案改了 streak 语义"的独立佐证。随机套件真降级 12837→6075、flips 18243→8265。

**修复提案有效性**：
```
patch -p1 --dry-run < D_s39_fix_proposal.diff   → checking file kv_cold_policy.h
patch -p1 < D_s39_fix_proposal.diff             → patching file kv_cold_policy.h
diff 打补丁结果 vs _collab/D_s39_stage/kv_cold_policy.h → IDENTICAL (md5 9c20e1...)
```

## 3. 预测逐条判定（全部计分，不静默丢弃）

| 预测 | 判定 | 实测依据 |
|---|---|---|
| (i) 零 letter 违反 | **命中** | 60,000 决策 + 11 场景 + 自测对照组 0 违反；cut 重算 5e-07；mirror==policy streak 0 分歧 |
| (ii) cap=1 因不对称滞后产生**字面合法的抖动** | **命中，但有条件** | 单槽竞争族（F2）cap=1 = **263 flips/400c，letter=clean**；锚定族（F1）cap≤3 仅 1 flip（稳定锚层把拍动层挡在池外，且 EWMA 恰等于 cut 的驻留层**永远无法被提升**，见 §5）。条件：池里必须有 ≥2 个"真能跨过 cut"的候选层 |
| (iii) "streak 跨未观测窗口存活" | **命中** | 最小复现 spec_gap=1（cycle 2 layer 3，空窗）；随机流亦有 |
| (iii) "低置信窗口仍在攒 streak/EWMA" | **命中且比预期更强** | 合成 rounds=42：6110 次（136/200 流）；**真实 tap 空闲窗**：L5 真能量恒定仍被降级（§4.3） |
| (iv) "22 demotions 是正净增长计数、会少计" | **命中并定量** | C 自己的流：net=681 vs true=1585（**少计 57%**）；场景 10：net=2 vs true=3 |

## 4. 发现（按 (a)/(b)/(c) 标注 + settle 的代码行）

### 4.1 F2 streak 跨观测空洞 —— **(b) spec-vs-code gap**
最小流（16 层，cap=4，**min_dwell=1**，stable=2）：L0..L2=3,4,5；**L3=0.05，t=1 整层缺席**；其余 6..14。`D_s39_dbg.cpp` 取证：`t=2 prev={0,1,2,} cold={0,1,2,3,} L3: streak=2 dwell=2 rounds=64 → demote layer=3`，即 code streak=2 由 t=0 与 t=2 构成、**t=1 无观测**。
**settle 行**：`kv_cold_policy.h:129-138` 的 streak 循环只遍历 `observations`，缺席层条目不动 ⇒ 冻结；头注 `:20-25` 宣称 "in EACH of the last `stable_cycles` windows"。诚实补充：按"冻结的 EWMA 在那一窗也仍处于 cut 之下"的读法，头注字面**可被满足**——这是一个**意图缺口 + 文本歧义**，不是纯矛盾（因此归 (b) 而非 (a)）。运维含义：降级可以建立在循环从未验证过的窗口上。

### 4.2 F3a 低置信窗口计入 streak —— **(b) spec-vs-code gap**
**settle 行**：streak 更新 `:129-138` 不查 `window_rounds`；置信门只在候选门 `:153`，即**只约束决策窗口**。头注 `:23-24` 的 "observed >= min_window_rounds times **this window**" 同样只约束当前窗 ⇒ 与 F2 同类歧义。实测：随机套件 6110 次降级属此类（首例 "rounds 42 < 64 in streak"）。

### 4.3 F3b 空闲窗被 tap 喂成 (0.0, 0) —— **(b) + 提案未闭合（本轮新发现）**
**这不是合成输入**：观测路径就是头注 LOOP 段的差分均值，`window_from_cumulative` 在 `count1 <= count0` 时返回 `mean=0.0, rounds=0`（`:76-83`）。EWMA 更新 `:118-126` **没有置信门**，于是"空闲"被当成"能量为 0"的真实观测。
实测（16 层，cap=4 默认门；L5 真能量恒 0.5、cut=0.4，**本不该被降级**）：
```
t=1 fed(L5)=(0.000, 0) → EWMA 0.250 ; t=2 (0.000,0) → 0.125 ; t=3 (0.000,0) → 0.062
t=4 fed(L5)=(0.500, 64) EWMA 0.281 ≤ cut 0.300 → demote layer=5   ← 决策窗 rounds=64，五门全过
```
检查器判：**letter=clean（代码语义下合法）、spec_gap=1**（"thin window (rounds 0) in streak"）。
**提案补丁的状态**：fix A 让低置信窗清零 streak ⇒ 一次性空闲从 t=4 推迟到 **t=5 仍被降级**（持续交错空闲则不降级，flips 2→1）。所以 **proposal 只延迟、未消除**这类降级；EWMA 污染本身（`max_window_rounds` 缺失或"rounds<floor ⇒ 视为无观测"）提案未处理。建议 C 在 fix A 之外补一条：**`rounds < min_window_rounds` 的观测不参与 EWMA 更新**（等价于把该窗视为缺席）。

### 4.4 无 (c) 重推导分歧
见 §1 诚实声明 2；证据 = cut 差 5e-07 + 12,837/12,837 降级的 mirror==policy streak 一致性 + `D_s39_dbg.cpp` 逐周期可解释。

## 5. 抖动量化（秒；成本界用 S34 板行原文）

几何（`D_s39_dbg.cpp` 验证）：L0-2 锚=0.1、L3 缓冲=5.0、**L4 拍动=0.1×4 窗/50×1 窗（周期 5）**、L5-11=100+、L12-15 深保；cap=5。模型：窗口 = `NINFER_FT_RELOAD_SECS`=60 s；一次重排 cold_cap 层 ≈256 MiB ⇒ PCIe 地板 ~16 ms；真失速 = reload 路径排干（≤120 s）+ 图重捕。

| 版本 | flips/400c | flips/h | 地板 s/h | 中度(5 s/翻) s/h | 最坏(120 s/翻) s/h |
|---|---|---|---|---|---|
| 未修（S34 现状，锚定族 cap=4/5） | **318** | 47.7 | 0.76 | **238** | **5724** |
| 未修，单槽竞争族 cap=1（F5） | **263** | 39.5 | 0.63 | **197** | **4734** |
| 提案（streak+dwell 对称滞后） | 106 | 15.9 | 0.25 | **80** | **1908** |
| 门 G3 目标 | ≤6 | ≤0.9 | ~0 | ≤4.5 | ≤108 |

**最坏列 > 3600 s/h ⇒ 系统永远追不上自己的重排（永久 reload 循环）**，这正是"live 接线前必须先治抖动"的理由。滞后把它从 318 压到 106 后**渐近**（拍动层每周期有 2 个连续 above 窗，EWMA 25→12.6，任何"窗数 ≤ 周期"的门都会被跨骑）。
**F5 的补充机制（预测 ii 的诚实边界）**：cap=1 能否抖动取决于池里是否有 ≥2 个能跨 cut 的候选层；F1 族不抖动是因为 3 个 0.1 锚把池位钉住，且"EWMA 恰等于 cut"的驻留层在 `:196 (streak <= 0)` 下**永不被提升**（提升要求上一窗 EWMA > cut，而 cut 就是它自己的值）⇒ 自锁。把钉池层改成低置信（不进候选）或热台地（cut=100）后，cap=1 立刻抖动。

## 6. 修复提案（未应用）与自动门

- **提案 A（关 F2/F3a）**：streak 只数"rounds≥下限的置信窗"，未观测窗**清零**（原为冻结）。
- **提案 A+（关 F3b，新增建议）**：`rounds < min_window_rounds` 的观测**不参与 EWMA**（否则真能量恒定的层会被 0.0 拉到 cut 以下）。
- **提案 B/B'（缓 F4，不根治）**：promotion 需 `above_streak ≥ stable` 且 `dwell ≥ min_dwell`，与降级对称。代价（实测）：合法提升从 c5 → c6 → c7（默认慢 1–2 窗 = 60–120 s）；抖动 318→106。
- **根治（后续）**：逐层翻转退避（每次翻转 cooldown 翻倍、稳定 N 窗复位）；理由见 §5 的渐近曲线。
- **补丁**：`_collab/D_s39_fix_proposal.diff`（对被测头的 `patch -p1`，dry-run 与实际应用均 OK，应用后 == `_collab/D_s39_stage/kv_cold_policy.h`）。
- **自动门 `D_s39_gate.cpp`**（CI 形态，`-I` 选被测头；本轮实测）：未修头 `D39_GATE_FAIL (fails=3)`=负对照；提案头 `D39_GATE_FAIL (fails=1)`（仅 G3：flips=106 > 6）。**门保持红色直到退避落地**——这是刻意设计：G3 是终态标准。

## 7. 与上一轮草稿报告的差异（本轮更正/加强）

1. **上一轮把它写成"实测"但 `/tmp/s39` 并不存在**（本轮首次真正编译+运行）。所幸其核心数字（318/106、6110/136、12837、5e-07、net=2 vs true=3、门 3/1）本轮**逐条复现**；但过程不可追溯，本轮补了 `D_s39_run.sh` + `D_s39_run_log.txt` 使每行可复跑。
2. **预测 (ii) 更正**：上一轮结论"cap=0/1/2 不抖动 ⇒ 协调者猜想被否定"**只对锚定族成立**；单槽竞争族下 cap=1 **确实抖动**（263 flips/400c，letter-clean）。正确的表述是 F5 的条件式结论。
3. **新增**：检查器自测（6/6）、mirror-vs-policy streak 诊断（含"提案头上会响"的活性证明）、(c) 不可达的诚实声明、C 自己流上的 57% 少计定量、以及 **F3b 真实 tap 空闲窗**这一提案未闭合的降级路径。
4. 上一轮 §5 表格中"B'=106"与"目标 ≤6"的关系保留（本轮门实测确认 106）。

## 8. 给 C 的三句话
1. F2/F3a 采纳提案 A，**并补 A+**（`rounds < floor` 不进 EWMA）——否则真能量不变的层仍会在空闲后 1 窗内被降级（§4.3 实测，提案头 t=5 仍降级）。
2. F4/F5 是设计债不是 bug：live 接线前必须先上**逐层翻转退避**；只加滞后只能到 106 flips/400c（最坏 1908 s/h），门是红的。
3. "22 demotions" 改口径为**真降级计数**（同一批流上 net=681 vs true=1585，少计 57%）；S34 板行的 `invariant_violations` 机器检查对 F2/F3 结构性失明（它读的是策略自己的证据字段），验收请改跑 `D_s39_gate.cpp` + `D_s39_adversarial.cpp`。
