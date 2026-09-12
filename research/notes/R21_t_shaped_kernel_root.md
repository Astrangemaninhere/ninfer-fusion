# R21：根因定位 —— 按 T 选 kernel 的数值不等价是 spec≠plain 与 chunk 不变量的共同根（2026-09-11 19:1X）

## 一、实测（引擎自带 hidden dump，逐位置 + 逐层；同一 prompt 仅改 `--prefill-chunk`）
工具：`NINFER_HS_DUMP_DIR` + `NINFER_HS_DUMP_TOPK=1`（每个 prefill chunk 一个 `chunk_%06d.bin`：
tokens + ids[] + 5 层特征 25600×T + post-norm hidden 5120×T + 全词表逐列 argmax）。
prompt：912 token（非 128 对齐），chunk 128（9 块）vs 4096（2 块）。

| 项 | 不同位置数 | 首次位置 |
|---|---|---|
| post-norm hidden (bf16, 5120) | **912 / 912** | **0** |
| 5 层特征 (bf16, 25600) | **912 / 912** | **0**（layer 5/19/33/47/61 **全部**在 pos 0 即分叉） |
| 引擎逐列 argmax（全词表） | 35 / 912（**3.8%**） | 5 |

## 二、判读（修正此前所有猜测）
1. **差异从位置 0 就存在，且每一层都有** ⇒ 不是 S_A ① 的"尾块 <64 走 FP32 寄存器归一化"
   （那只影响最后一个 chunk）；**根因是首块 T 本身不同（128 vs 912）⇒ 按 T 选的 kernel 不同 ⇒ 从第 0 个位置起数值即不同**。
2. **P7 的"分析不同"由此裁决：B 路对，A 路错** —— A 说"③ 有机会 IDENTICAL"，B 说"不可能，
   残余来自 attention/MLP/lm_head 的 T 形状 kernel"⇒ 实测支持 B ✓。
3. **同一机制解释主线 spec≠plain**：verify 是 **T=W=8**、plain 是 **T=1** ⇒ 走不同 kernel
   ⇒ target 侧**约 3.8% 的 argmax 翻转** ⇒ G-A（spec 与 plain 一致）无法成立；
   这也解释了此前"首次偏离位置随内容漂移"的现象（4% 的翻转率 ⇒ 首次翻转点是随机命中 margin 极小处）。
4. 与"接受率上限"的关系：3.8% 的翻转率**不足以**解释 p1=0 的塌陷（那是 ckpt 口径，R12/R18）；
   但它**就是** G-A 的缺口来源，且会让 verify 的 logits 系统性偏离 plain ⇒ 对接受率有二次影响。

## 三、修复方向（性能优先，不变）
**统一小 T 家族**：让 T∈[1, W]（含 plain 的 T=1 与 verify 的 W=2/4/8/16）在**所有** T 形状分派点上
走**同一条** kernel 路线与同一精度档 —— 不限于 GDN 快照（修② 只覆盖了 GDN 那一处，故单变量测不到效果）。
需要覆盖：attention 输入/核心、MLP/linear、lm_head、GDN 输入（含 snapshot / w4a4 / independent 各档）。
- 已派 S_E 子代理产出**这些分派点的完整清单 + 分派谓词 + 统一方案 + 性能代价**（报告 `_collab/S_E_shape_kernels.md`）。
- 性能判据：小 T 家族统一后必须实测 tok/s（A 路 roofline 估计 +0~3%/round；>3% 则回退到更小的统一域）。
- 配套定案：给 ② 加一行 schedule 日志，确认 verify 的 GDN 输入实际走的路线（决定 ② 是否有效）。

## 四、附带确认（本轮工具链）
- dump 只在 `--spec dflash2`（启用 dflash2 特征捕获）时触发；不带 spec 会 0 文件。
- dump 的 bf16 精度足够做**定位**（见上），但不足以量化 1e-3 以下的差值 ⇒ 量化仍用 argmax 翻转率。
- 构建纪律再次生效：必须 `export PATH=/home/user/.local/bin:$PATH`（否则 ccache 找不到、127、删二进制）。
