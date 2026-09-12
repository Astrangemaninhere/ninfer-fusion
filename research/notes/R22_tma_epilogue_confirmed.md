# R22：TMA epilogue 缺 bf16 回舍 —— 已定案（chunk 不变量的根）；并划清与 spec≠plain 的边界（2026-09-11 19:4X）

## 一、S_E 的单变量判别（我实测，零改码）
prompt ≈1032 token；`--prefill-chunk 128`（10 块，**永不 TMA**）vs `512`（4 块，**恒 TMA**）；
据 S_E：这一对只差 TMA 路径，gating proj 的 SplitK 两档相同（都 8）。

| 项 | 结果 |
|---|---|
| post-norm hidden 不同 | **1032 / 1032**（首次位置 = **0**） |
| 引擎逐列 argmax 不同 | **33 / 1032 = 3.20%**（首次 = 5） |

⇒ 与 128-vs-4096 的 3.8% **同量级** ⇒ **S_E 指认的缺陷就是活动机制**：
`src/ops/linear/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh:242-245` 的 epilogue **少了 gate/up 的 bf16 回舍**
（baseline 与 MMA-fused 两条路都有该回舍）；T 超阈值时按 head=1024 切块走 TMA ⇒
只改 `--prefill-chunk` 即命中 tokens 0..1023 ⇒ 从位置 0 起数值差 ~1e-3 ⇒ 3.2% argmax 翻转。

## 二、作用域划清（重要，避免误记收益）
- 这条缺陷只在 **prefill 的大 T 路径**生效 ⇒ 修它能让 **chunk 不变量**（同 prompt 不同 chunk 同输出）成立；
- **它不解释 spec≠plain**：短 prompt 下 plain 与 spec 的 prefill 都是单块、同路；
- spec≠plain 的 3.8% 来自**另一处**：**小 T 家族（T=1 vs T=2..16）的 kernel/路线不同**
  （A 路已指出 GDN 输入 T=1 走 gemv、T∈[2,16] 走 small_t；修② 改的是 snapshot plan 阈值，
  **不是这个 dispatcher**，故单变量测不到效果）。
- S_E 已排除（附代码证据）：nvfp4 MMA 六档 schedule、rope 各分档、causal_conv1d、KV fill、
  注意力路由、rmsnorm/l2norm、**lm_head（恒 T=1，不可能贡献第 0 token 差异）**。
  另注：MoE(35B) 有 `adaptive = tokens>=47 && <=51/52` 的专家核分档（家族级分叉），27B dense 不涉及。

## 三、修复清单（按 P7 双路取证，性能优先）
| # | 目标 | 改动 | 判据 | 性能 |
|---|---|---|---|---|
| 修④ | chunk 不变量 | 在 `nvfp4_linear_swiglu_w4a4_tma.cuh:242-245` 补 gate/up 的 bf16 回舍，与 baseline/MMA 一致 | 128 vs 512 与 128 vs 4096 的 argmax 翻转率 → 0（或显著下降） | 中性（多一次 cvt） |
| 修⑤ | spec==plain | 统一**小 T 家族**：T=1 的 gemv 路线与 T∈[2,16] 的 small_t 路线对齐（同算术/同累加顺序）；范围以 S_E 的清单为准 | `_ga_check.sh`（zh/num）首次偏离后移/消失 | 需实测 tok/s（小 T 家族代价应≈0） |
| 修③ | prefill 尾块 | `gated_delta_net.cpp:255` 的 BF16 预归一化双路（S_A ①） | 可与修④ 叠加验证 | 中性 |

## 四、工具与纪律（累积）
- dump：`NINFER_HS_DUMP_DIR=<dir> NINFER_HS_DUMP_TOPK=1`，**必须带 `--spec dflash2`** 才触发；
  每 chunk 一文件，含 5 层特征 + post-norm hidden + （tokens≤1024 时）全词表逐列 argmax；
  **无 GDN 内部量、无逐层 hidden**（KV 另有 `NINFER_KVDUMP_DIR`）。
- 构建：必须 `export PATH=/home/user/.local/bin:$PATH`（ccache 在里面，否则 127 且删二进制）。
- 判定 shape 分叉的最低成本手段：**同一 prompt 只改 `--prefill-chunk`，比 dump 的逐列 argmax 翻转率**。
