# U7: Muse e8 page-fill 内核修复 —— 补丁与协议 (2026-09-10 11:3X)

## 位置
`src/ops/kernel/gqa_attention_prefill_i8.cuh` 的 E8 分支（Muse 专用: 该 kernel 只在 `Geometry::KVHeads == 2`
时被选中，即 Muse；qwen 27b 的 KVHeads==4 走的是另外两处已修路径）。

## 三处缺陷 + 一处死代码（本次读代码实证，非推断）
1. **旋转缺失**: 第 260 行在**未旋转**域求 `k_abs`，而读侧对 Q 做了 Hadamard 旋转 ⇒ 域不匹配。
2. **除数错误**: `ksh_e/vsh_e = k_abs/127`，但 E8 码范围是 ±7 ⇒ 尺度小 ~18 倍 ⇒ 码全撞截断。
3. **格点投影缺失**: 该分支**完全没有** `e8_project_8d_warp`（已修站点有）⇒ 写的是普通取整坐标，
   而读侧按 E8 码解释。
4. 第 283-286 行两段空 `if` 死代码（调试残留，与已修站点第 192-196 行的残留同类）。

修复范式逐行对照已修站点（同文件 134-203 行的 fill 路径）与 decode 路径
（`gqa_attention_decode_i8.cuh`），补丁在 `_collab/M_muse_pagefill_patch.diff`（含邻居码同步处理：
`k0n_s/k1n_s` 先缩放再投影，再用投影后的值取整）。

## 协议（§116c 原文要求: 必须先用 Muse+e8 起一轮验证再改）
1. **先跑修复前基线**: `bash _e8_muse_check.sh`（已就绪, 判定式输出 `MUSE_E8_VERDICT=`）——
   它跑 Muse 的 `plain_nvfp4 / e8 0-7 / e8 8-15 / plain_bf16` 四档 32K 掉针。
   预期: e8 档**低于** plain ⇒ 判定 REVIEW，这正是 §116c 已知缺陷的特征（不是脚本 bug）。
   若 Muse 根本不支持 `--kv-layer-storage`（该模型 `supports_per_layer_kv_defaults=false`），
   脚本会打印 `SERVE_FAILED` + 日志里的 `invalid/unknown/not supported` 行 —— 那也是有效结论，先记录再决定。
2. 应用补丁 → 重编（窗口 H 与其他改动一起，一次编译）。
3. **复跑同一脚本**: `MUSE_E8_VERDICT=PASS`（即 e8 不再输给 plain，允许 ±1 针噪声）才算修复生效。

## 风险与不可验证的部分
- 该 kernel **无测试覆盖**（§116c）⇒ 补丁正确性只能靠上述端到端判定，不能靠单测。
- 修复范式在另外两处已被实测验证（§96: rel-err 0.45→0.094, clamp 3.2%→0.1%），
  因此这里的预期是"三档回到同一水平"，而不是"变得更好"。
