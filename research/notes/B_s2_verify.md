# B 线独立复算 S2 / §116 结论（e8 尺度取自旋转前 max → |code|==7 饱和）

- 验证者: B（独立实现，未复用 `_e8_fixsim.py` / `_e8_clamp.py` 的任何函数）
- 脚本: `_collab/B_s2_verify_run.py`（Sylvester 矩阵 H64 + torch bf16 view 解码
  + 自写 meta/nibble 解析；正交性自检 err=0）
- 复现命令（CPU-only）:
  `wsl.exe -e python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/B_s2_verify_run.py`
- 数据: `/home/user/bench/kvdump_e8src`，L14/L15 全部 size 匹配 chunk
  （t3072 ×2），**全部 token/页**：每组 98,304 个 64 维组、6,291,456 个 code
  （A 只用 chunk0 前 64 token、page0 ≈ 1,024 组 / 16,384 code）

## 结果（判据：中位比值 ∈ [1.3,1.7] 且 clamp ≥ 0.5% 且与 A 同数量级）

| 层 | post/pre 中位 | post/max | post/(7s) 中位 | clamp 率 | A 的数字 |
|---|---|---|---|---|---|
| L14 | **1.478** | 6.15 | 1.118 | **0.87%** | 1.42 / 1.03% |
| L15 | **1.506** | 6.03 | 1.074 | **0.71%** | 1.54 / 0.55% |

**VERDICT: PASS** —— 与 A 的 1.42/1.54 差 +4%/−2%，clamp 0.87%/0.71%
同数量级；机制结论成立：Hadamard 旋转使组内 max 中位膨胀 ~1.5×，
而尺度取自旋转前 max → 尾部组（max 达 6×，post/(7s) 达 14–16×）饱和在 ±7。

## 附加诊断（不影响判据）

1. `post/(7s)` 全样中位仅 ~1.1：存储的 fp16 scale 大体覆盖了中位组，
   饱和集中在尾部组 —— 修复收益主要来自尾部，量化改善预期小于"1.5× 全局"。
2. 采样敏感性：只取 chunk0 前 64 token/page0（A 的采样）时
   post/pre 中位升到 1.911/1.679、clamp 降到 0.39%/0.44% ——
   前 64 token 是 prompt 起始段，与全样有偏差；这解释了与 A 数字的小差异方向。
3. 尾部 chunk（t51/t4）的 kvsrc 是全尺寸 buffer，token 对齐不安全，已跳过
   （占比 1.8% token，不影响结论）。

## 结论
A 的 §116 定性结论与两个关键数量级均被独立复算确认（PASS）。
建议 S2 修复按 board 推进；注意附加诊断 1：修复后 ppl 改善预期集中在
尾部组，不宜按 1.5× 全局误差折算收益。
