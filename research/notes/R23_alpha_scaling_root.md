# R23：修④ 的真身是 TMA epilogue 多乘 alpha（S_E 的"缺回舍"表述需修正）（2026-09-11 20:1X）

## 一、逐字对照（四条路径的 silu·up 算式）
| 文件:行 | 算式 | 有无 alpha |
|---|---|---|
| `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4.cu:47-48` | `__floats2bfloat162_rn(silu(gate_values.x) * up_values.x, …)` | **无** |
| `…/nvfp4_linear_swiglu_small_t.cu:70` | `__float2bfloat16_rn(silu(gate) * up)` | **无** |
| `…/nvfp4_linear_swiglu_decode.cu:61` | `__float2bfloat16_rn(silu(gate) * up)` | **无** |
| **`…/nvfp4_linear_swiglu_w4a4_tma.cuh:242-245`** | `__floats2bfloat162_rn(silu(gate[0] * alpha) * (up[0] * alpha), …)` ×4 | **有，且对 gate 与 up 各乘 ⇒ 乘积 ×alpha²** |

- `alpha` 是**运行时 float 参数**：`nvfp4_linear_swiglu_w4a4_tma.cu:60` 的 launcher 签名 → `:81` 传入内核；
  `…_tma_launch.h:14` 亦声明 ⇒ **不是编译期 1.0 的 no-op**。
- 因此这不是 S_E 说的"少了 bf16 回舍"（TMA 侧确实有 `__floats2bfloat162_rn` 回舍），
  而是**乘性系数不一致**：三条兄弟路无 alpha，唯 TMA 乘以 alpha²。
- 这正好解释实测：**大 T（≥TMA 阈值）走 TMA ⇒ 值级分叉从位置 0 起**，
  `--prefill-chunk 128 vs 512` 的 hidden 1032/1032 不同、argmax 翻转 3.20%（与 128 vs 4096 的 3.8% 同量级）。

## 二、必须实测定案（不可直接改）
两种可能，方向相反：
- **(a) TMA 的 alpha 是多余**（累加器已是真实单位；三条兄弟路正确）⇒ **删掉 alpha** 即对齐；
- **(b) alpha 是必需**（TMA 走另一套累加单位；兄弟路各自内部已处理）⇒ 删它会把长 prompt 的前 1024 token 算错。
判据（最便宜、一行）：**打印 alpha 的实际值**（gated 日志，加到 `nvfp4_linear_swiglu_w4a4_tma.cu:81` 附近）：
- `alpha == 1.0` ⇒ 该差异不存在，S_E 这条线索作废，chunk 分叉另有来源；
- `alpha != 1.0`（比如 ~0.02、~1.5 之类）⇒ 进入 (a)/(b) 判定：再看**兄弟路的 alpha 从哪来/是否已在别处乘过**
  （grep 它们的 launcher 参数与自己内部是否含 `* alpha` / 缩放），并做一次 A/B：删 alpha 后跑
  chunk 判别（128 vs 512 翻转率应→0 或显著下降）+ 同 prompt 的文本合理性（防止把 (b) 改坏）。

## 三、与主线的边界（不变）
- 修④ 属 **prefill 大 T**路径 ⇒ 修好它达成"chunk 不变量"，**不直接改变 spec≠plain**（短 prompt 下
  plain 与 spec 的 prefill 同为单块）；但**长 prompt 的 prefill 被算错**会同时污染两条路 ⇒ 对长上下文
  的接受率与正确性都有影响（此前未单独量化，值得在修④ 后补测长 prompt 的接受率）。
- spec≠plain 的 3.8% 仍归**小 T 家族**（T=1 gemv vs T∈[2,16] small_t；修② 改的 snapshot plan 阈值
  不覆盖该 dispatcher，故单变量测不到效果）。

## 四、顺手复核的结果（用户提醒的"对齐问题"= `_TODO.md` §132/§133 四处）
- ① `dflash_impl.h:448` 取列跳过 anchor（`source_column_offset = 1` 带完整注释）⇒ **已在树** ✓
- ④ `mtp_round.cuh:50` `ar_valid_columns = s < next ? 1 : 0` ⇒ **已在树** ✓
- ② ③ 在训练脚本 `train_dflash2.py`（shift / 输入模式）⇒ **只影响 retrain**，与引擎无关 ✓
⇒ 引擎侧两条"对齐"确认已落地；未在现二进制上做行为复验（本轮未发现它们复发的迹象：
dspark 的 p0 已由此前的 0% 恢复到 22% 一档，见 `_probe_a5.out` 的 A/B）。

## 五、下一步（按性价比）
1. **打印 alpha**（一行 gated 日志 + 一次编译）⇒ 定案 (a)/(b)。
2. 若 ≠1：按 (a) 删 alpha（或按 (b) 反向修正）→ 跑 chunk 判别 + 长 prompt 文本合理性 + 接受率。
3. 回到小 T 家族统一（修⑤），P7 双路出补丁。
