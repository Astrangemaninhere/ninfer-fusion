
## 判别实验 v2：verify 与 plain 的 token 级等价性 (2026-09-10 19:06)

- plain_a vs plain_b: IDENTICAL (48 tokens)   ← 自洽性基线（必须 IDENTICAL）
- dflash2_auto_a vs dflash2_auto_b: IDENTICAL (48 tokens)   ← 投机路径自身确定性
- plain_a vs dflash2_auto_a: **DIFFER at token 36**  (A[36]=98633 B[36]=133222; lens 48/48)   ← **关键**
- plain_a vs mtp3: **DIFFER at token 36**  (A[36]=98633 B[36]=133222; lens 48/48)   ← 交叉验证（另一草稿后端）

判读：首轮 a=0 时发布的正是 verify 自己算出的**列 0 argmax** ⇒ 第 0 个 token 不同即为 verify 路径的等价性缺口。
已知**代码级**前提：`target_verify_batch_impl` 没有调用 `apply_final_logit_policy`（契约要求每个 lm_head 产生点都调），
但对 qwen 是编译期 no-op ⇒ 不是本模型的分歧来源（已排除）；**Muse 会中招**（softcap 20 / multiplier 0.196）。
另记一条独立线索：CLI 显式 `--spec dflash2` 报 `object handle does not name a materialized tensor`，而 `--spec auto` 正常 —— 待查。
