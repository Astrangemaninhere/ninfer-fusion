# R24：修④ 是假阳性（alpha 必需）——已回退；并给出整场"spec≠plain"战役的结论（2026-09-11 16:0X）

## 一、经过（P7 的"量化验证"救了一次回归）
1. S_E 指认 `nvfp4_linear_swiglu_w4a4_tma.cuh:242-245` 的 epilogue "少了 gate/up 的 bf16 回舍"。
2. 我逐字对照四条路径，发现真身不是缺回舍，而是**TMA 路径多乘 alpha**（gate 与 up 各乘一次 ⇒ 乘积 alpha²），
   且 `alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor)` 是**运行时反量化系数**；
   准入条件 `tokens>=256 && tokens%256==0` 正好解释"chunk 512 命中、chunk 128 不命中"。
   ⇒ 我判定"与三条兄弟路对齐 = 目标"，**落地去掉 alpha**。
3. **实测立刻否掉**：同一 prompt、同 token 集，`--prefill-chunk 128`（不命中 TMA）vs `512`（命中 TMA）的
   逐列 argmax 差异 **3.20% → 99.61%**（hidden 差异仍 1032/1032）。
   ⇒ 改动前两条路径**本就高度一致（仅 3.2% 舍入级差异）**，**alpha 是必需的**
   （TMA 的累加器是 raw 单位，兄弟路是 real 单位，各自内部自洽）。
4. **回退**：`cp /home/user/fix4_bak/…` 复原，md5 `698f409c71a4` 与备份一致，
   重编二进制大小与修复前同为 839637376、时间 16:01 ⇒ 树回到 ①+② 状态。
   ⇒ **S_E 的这条线索是假阳性**（P7 的"两路 + 量化"在此拦住了一次回归）。

## 二、由此得到的结论（本场战役的收敛点）
1. **chunk 依赖（3.2%）不是 bug，而是"同一数学、不同形状 kernel"的舍入级差异**：
   两条路径各自内部自洽（缩放各自处理正确），差异来自 silu/乘法的结合顺序与归约顺序。
   ⇒ **不能靠"删一个系数"消除**；只有统一算术（同 kernel 家族）才能消，代价见下。
2. **spec≠plain 的 3.8% 属同一类**（T=1 的 gemv 与 T∈[2,16] 的 small_t 各自自洽、舍入级差异）。
   ⇒ **G-A 的"逐位一致"在可接受的性能代价下不可达**（参照实现同样如此）；
   实测偏差量级：**1e-3 级 logits、3.8% 的 argmax 翻转**。
3. ⇒ 结论：**把 G-A 的判据定为"统计等价"**（同精度档、首次偏离后移/消失、接受率与性能不退化），
   而不是逐位 hash 相同；否则需立项"统一小 T 家族/统一 prefill 路径"，用实测 tok/s 换精确性
   （A 路 roofline 估 +0~3%/round，但需实测；这是**性能与逐位一致之间的取舍，属用户决策**）。
4. **接受率的杠杆仍不在这些舍入项**：按 R12/R18/S_C，真正杠杆是 **ckpt 口径（训练喂真 token + shift 1）
   与 live artifact 是 08-26 草稿**这两件事；最便宜的下一步是 S_C 的塌陷比 R 判定
   （用已有 `step_000100/000200` 与 08-26 草稿，离线跑，比重训便宜）。

## 三、当前树状态（可回退点）
- 已落且实测有效：**修①**（`gdn_conv.cuh:99` 的 `p` 先 bf16 化；`zh plain vs df2` 首次偏离 10→19，
  mtp3 逐位未变 ⇒ 触发谓词被交叉验证）✓
- 已落但**未测到效果**：**修②**（`nvfp4_gdn_snapshot_plan.cpp:43` 阈值 3→16）⇒ 待一行 schedule 日志定案。
- 已回退：**修④**（alpha）⇒ 备份 `/home/user/fix4_bak/` 保留，md5 校验一致。
- 其它备份：`/home/user/fix1_bak/`、`/home/user/fix2_bak/`、`/home/user/gdnconv_bak/`。

## 四、下一步（建议顺序）
1. **ckpt 线（收益最大）**：S_C 的 R 判定（离线，CPU）→ 据结果决定 retrain（`--out-dir` 必给，避免
   `--resume` 取到 legacy 的 `step_001900`）；顺带钉死 serve 4.55 vs CLI 1.35 tok/round 的混淆项。
2. 修② 定案：一行 schedule 日志。
3. 修⑤（可选、需用户决策）：统一小 T 家族以把 spec≠plain 从 3.8% 压到更低，代价是实测 tok/s。
