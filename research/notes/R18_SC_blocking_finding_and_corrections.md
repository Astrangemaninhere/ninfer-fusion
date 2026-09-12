# R18：S_C 的阻碍性发现 —— R12 的 E1/E2 论据作废，需先钉死一个混淆项（2026-09-11 17:4X）

## 一、必须纠正我自己的结论（诚实记账）
S_C 查明：**live artifact `/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer` 的 mtime 是 2026-08-26，
且全树唯一**（没有 `*_tuned` / `w9s1200` 之类变体）；唯一一次导出尝试
`dl/df2_w9_export.log`（09-10 15:45）报 `ModuleNotFoundError: No module named 'tools'`，
`dl/` 里**没有任何 `mapping ok`** ⇒ **任何训练 ckpt 从未进过引擎**。

⇒ 今天所有引擎接受率数字，描述的是**08-26 那份草稿**，而不是 step_006000 / step_1900。
⇒ **R12 的 E1 作废**：它把"引擎实测 p0 26–35%"与"*另一个* ckpt 的离线上限 25–31%"相比；
**E2 也作废**（A2 的离线命中率是训练 ckpt 的，不是 08-26 草稿的）。
⇒ **"引擎侧已排除"这条结论的定量桥被拆掉，问题重新开放**；但 E3（可复制文本上 mask 列 6/7 正确）、
E4（MTP 同路径拿到 `[29,16,5]`）、E5（参数/结构逐项一致）**不受影响**，仍支持"引擎机制健全"。

## 二、另一个必须先钉死的混淆项
`M_df2_serve_measure.md` 记的 **4.55 tok/round**（旧 binary、serve 路径）与今天 CLI 的
`[8,0,0,0,0,0,0]`（≈1.35 tok/round）**互相矛盾**。同一份 08-26 草稿 ⇒ 要么是 prompt/路径差异，
要么是我今天的改动（F4 walk 修复、GDN conv 补丁）把它改差了 ⇒ **retrain 前必须先钉死这一项**。

## 三、S_C 的两条便宜且决定性的判定（建议优先于 6000 步重训）
1. **已有 mask+shift0 的 ckpt**：`data/dflash2_ckpts/step_000100`、`step_000200`
   （09-10 23:32/23:36，**晚于** 19:34 的配方改动）。测"塌陷比" R（mask 口径命中 / 真 token 口径命中）：
   **旧 ckpt R≈0.18，预测新配方 R ≥ 0.6** ⇒ 一次离线评测即可判 M2 是否是修法。
2. **直接测 live artifact 那系草稿**（`data/draft_model/model.safetensors`，08-17）的 R：
   R≈0.18 ⇒ 口径论需重估；R≈1 ⇒ M2 就是修法。
两者都比重训便宜得多。

## 四、S_C 的其他确认与坑
- **M1 成立**（shift 必须 0）：`train_dflash2.py:227` 行 i ↔ 位置 a+1+i；`:457` 教师行 `ids16[a+shift+i]`；
  实测 `P(ids16[t,0]==tok[t+1])=0.2420` vs `==tok[t]=0.0023` ⇒ 教师是**下一个** token，shift=1 晚一行。
- **M2 成立**：legacy 配方有源码直证（`data/df2pilot/train_dflash2.py:420` 真 token 块、`:425` 硬编码 +1）；
  全量 **706,220** 位置里 mask id 出现 **0** 次 ⇒ mask embedding 梯度恒 0，而推理 7/8 列喂它。
- **M1 从未被实验测过**：`data/_ab_shift0|1` 是空目录、`dl/shift_ab.log` 无 step 行 ⇒ 现存结论全靠语义。
- **mask id 别动**：fork `config.h:106=248077` 与 trainer 一致；190221 实验已回退、无改善。
- **retrain 坑（必读）**：`data/dflash2_ckpts/` 混了两套配方，`--resume` 取 `sorted()[-1]` =
  `step_001900`（**legacy 错位权重**）⇒ 续训必须显式 `--out-dir`，否则接错配方。
- 可证伪阈值 P1–P5（见 `_collab/S_C_train_regime.md`）：如 P3 "引擎剖面不得再是 `[x,0,0,0,0,0,0]`，
  p1/p0≥0.5 且 AL≥1.8"。

## 五、下一步（顺序已定）
1. **钉死混淆项**：同 prompt 同 ckpt，比 serve 路径 vs CLI 路径的 tok/round（先用 08-26 草稿，
   再与我今天的改动做一次 A/B：F4 之前/之后、conv 补丁之前/之后）。
2. **走 S_C 的便宜判定**（step_000100/000200 的 R；以及 08-26 草稿的 R）⇒ 决定是否/如何 retrain。
3. **同时落地 GDN 形状修复 ①②**（与 ckpt 问题无关、独立有效）：删 `gated_delta_net.cpp:255-261`
   的 BF16 预归一化分支、`h_chunk` BF16→FP32；验收 = 非对齐 prompt 的 chunk A/B IDENTICAL
   + `_ga_check.sh` 首次偏离后移/消失 + 接受率不退化。
4. R12 的措辞已在 `_HANDOFF.md`/`_TODO.md` 里按本页纠正（E1/E2 作废）。
