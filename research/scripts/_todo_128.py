#!/usr/bin/env python3
"""§128：拆掉"固定第 36 步"假象 + 两条独立问题的定位 + A5 的 dspark 结论。"""
import datetime
import pathlib

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
T.open("a", encoding="utf-8").write("""
### 128. 拆掉"第 36 步机制"假象 + 低接受率的两条独立成因（""" + stamp + """）
**判别实验（`_collab/M_offset_probe.md` / `M_offset_probe2.md`）**：
| 档 | prompt tokens | 生成数 | 第一个偏离的生成序号 | 错值 |
|---|---|---|---|---|
| 中文短 | 36 | 48 | 36 | 98633→**133222** |
| 中文中 | 99 | 48 | **无偏离**（48 token 全同） | — |
| 中文长 | 222 | 48 | 36 | 115544→**133222** |
| 中文 max-new=192 | 36 | 131 | 36 | 98633→**133222** |
| **英文** | 45 | 48 | **37** | 2696→**369** |
**修正**：先前我把"三档都停在 36"读成"固定在生成第 36 步的机制"——**错了**。中文三档的 prompt 只是加了填充，
模型续写的前 36 个 token 本来就**逐字相同**（长档与短档同值 133222 即是证据），所以"同一位置"是同一条输出上的
同一个点；而**换内容的英文档偏离在 37、错值也变**（369）⇒ **偏离点随内容变化**，不是固定步数机制。
⇒ 真实机制：**verify 与 plain 之间存在数值差，在"argmax 近似并列"的位置把选择翻转**（翻转目标通常是次高 token）。
这与 A1 的 D1/D2（plain T=1 vs verify T=width 的 kernel 实例化差异 + KV 行标定量化的次 ULP 放大）一致；
也解释了为什么 T=2/4/6/8 会在同一点翻到同一个值（同一条输出上的同一个危险位置）。

**由此得到两条**互相独立**的问题，别再混为一谈**：
- **(A) verify ≠ plain 的数值差**：已复现、可判据（token id 对照），但**影响面小**——只在近似并列处翻转，
  解释不了"接受率从 0.9 掉到 0.42"。仍需修（它是正确性缺陷，且长上下文更容易踩），但不是低接受率的主因。
- **(B) 草稿本身在位置 0 只有 ~42% 命中率**：位置剖面 p(pos0)=dflash2 41.8% / MTP 53.5% / dspark 26.8%，
  与"草稿质量/训练对齐"一致 ⇒ **这才是 0.9 目标的主要缺口**。A2（草稿上限，正在跑）会用真实特征与 checkpoint
  给出"草稿在位置 0 的上限命中率"，若 ≈42% 则证实"verify 没吃草稿、是草稿不够好"。

**A5（dspark 深挖）已完成**：H1（tap 层号错）**在代码级被排除**（引擎 tap = 第 layer 层 post-MLP residual
= HF `hidden_states[layer+1]`，与参考实现一致；层号 `[4,16,28,40,52]` 与 config 一致），且**离线逐字节取证**
（`_collab/A5_weights_audit.py`）证明现役 artifact 的 **55/55 个 `dflash/*` 张量与 checkpoint 逐字节相同**
（含 markov 未互换、qkv/gate_up 的拼接方向、context_k/v 的来源）⇒ "权重绑定错"整族也排除。
剩下两条 **dspark 专属**候选：
- **候选 A（首选）草稿 rope 约定**：checkpoint 声明 `yarn/factor 32`（`tmp/dspark-config.json:54-68`），
  引擎草稿两条 rope 都走纯 `rope_theta`（`dflash_impl.h:177/304`），而 target 侧只有 factor-4 的
  `ops::rope_yarn4` 且需 `--yarn` ⇒ 若真，属可修的系统性错误；
- **候选 B（次选）block 行→verify 列约定**：引擎 bf16 取 rows 0..k-1（含 anchor 行），HF 参考取 `[:, 1:]`、
  引擎自有 W8 分支取 rows 1..k ⇒ 整路错位一行（也能解释历史的 `k=1 → 0%`）。
  但位置剖面判读**不支持** B 当主因（p0=26.8% 且后续不衰减，与"块内错位"应有的 p0≈0 矛盾）。
- 交付：`_collab/A5_dspark_rootcause.md` + 两份 dry-run rc=0 的 diff（诊断探针 / 候选 B 修法），未落地。

**下一步（全力修 bug 的顺序）**：
1. 等 A2 的草稿上限数字 ⇒ 判定 (B) 是否为 0.9 的主要缺口；
2. 若 (B) 成立：查 `--target-shift` 与训练目标对齐（A2 会给两种对齐的命中率对比）；
3. dspark 的候选 A（YaRN）按 A5 的判据验证并修；
4. (A) 的数值差：按 A1 的 E 序列继续（其中"verify 侧补 head probe"是唯一能看数值的手段，需改码+编译）。
""" )
print("_TODO.md §128 已追加")
