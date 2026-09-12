# FIX_PLAN_ABCD：性能优先的修复整合（2026-09-11 18:0X）

## 0. 硬约束（用户定调）
**最优性能，不惜代价** ⇒ 禁止任何"以并行度换精确性"的改法。
具体否决：**S_B §三(a)**（让 verify 的 GDN 逐 token 走 recurrent）——它换来 G-A 逐位一致，
但 verify 的 GDN 段失去 W 列并行 ⇒ **不做**。同理否决任何 FP32 化的*全量*缓冲升级
（`h_chunk` BF16→FP32 会 2× 带宽）。原则：**在保持/提升性能的前提下对齐数值**。

## 1. 四路结论 → 三个修复项 + 两项决策
| 来源 | 发现 | 本计划处置 |
|---|---|---|
| S_B D1 ★ | fused conv 第 4 抽头用 FP32 `p`、却把 `bf16(p)` 发布进状态（`:99-104` vs `:118`），量级 ~2e-3，触发 batch==1 且 W∈{2,3,7..10} | **修 ①（一行，性能中性）** |
| S_A ④ + ③ ★ | 小 T 家族按 T 选不同 kernel/精度档（`nvfp4_gdn_input_w4a4.cu:40-58` 激活量化 kernel 与 M-tile/TMA 表随 T 变；`nvfp4_gdn_snapshot_plan.cpp:42-44` 分档）⇒ **verify(T=W) 与 plain(T=1) 走不同路线** | **修 ②（统一小 T 家族，性能中性：这些形状本就极小）** |
| S_A ① | `gated_delta_net.cpp:254-261`：`T_full>0` 走 BF16 缓冲、`T_full==0` 走 FP32 寄存器 ⇒ prefill 切分敏感（已用 128 对齐实验证伪/证实） | **修 ③（仅影响 prefill 尾块；非 G-A 必需，列为次优先）** |
| S_A ② | `h_chunk` BF16（3 处） | **不做**（FP32 化 = 2× 带宽，违性能约束）；若 ③ 后仍差再评估 |
| S_B D2 | flat conv `acc+=w*x` vs record 侧 `fmaf` 链（~1e-7） | 不做（低于任何可观测阈值） |
| S_D | selector `logits` BF16 单侧 vs 参照 FP32；greedy 下 accept 不消费 `draft_candidate_probs`（无风险） | **不做**（S_D 已证：既不能修 spec≠plain，也不影响接受率上限；且裸改 dtype 会静默出错） |
| S_C | 训练口径（M1 shift / M2 真 token vs mask）+ live artifact 是 08-26 草稿 + `--resume` 取 legacy `step_001900` | **决策项**（见 §4） |

## 2. 修复项（按顺序，均性能中性或更优）
### 修 ①（S_B D1，一行）
`src/ops/gdn_input_proj/fp8/<fused conv kernel>`：把 `:99-104` 处的第 4 抽头累加从"直接用 FP32 `p`"
改为"先用 `bf16(p)` 再进状态"，与 `:118` 的 `s2 = __bfloat162float(__float2bfloat16_rn(p))` 同范式。
- 判据：`--prefill-chunk` 无关；更重要的是 `_ga_check.sh` 里 **K=1/K=7 的首次偏离后移或消失**、
  而 **K=3（W=4，本就 Materialized）不变**（S_B 给的可证伪预测）。
- 性能：单条 bf16 round，无结构变化。

### 修 ②（统一小 T 家族 —— 结构性修法）
目标：让 **T ∈ {1..16}**（decode 与 verify 的 W=2/4/8/16 全在内）走**同一** GDN 路线与精度档。
- `nvfp4_gdn_snapshot_plan.cpp:42-44`：把分档阈值放宽到覆盖 `tokens<=16`（S_A ③）。
- `nvfp4_gdn_input_w4a4.cu:40-58`：确认小 T 家族内激活量化 kernel 与 M-tile/TMA 选择**一致**；
  若 T=1 与 T=W 选了不同表，则把小 T 家族统一到同一条（这些形状极小，性能代价≈0）。
- 目的：**spec==plain 的结构性前提**（同形状家族同算术），且**不动大 T 的 prefill 快路**。
- 判据：`_ga_check.sh` 在 zh/num 上 IDENTICAL（或首次偏离显著后移）+ 接受率不退化 + 解码 tok/s 不退化。

### 修 ③（S_A ①，次优先）
`gated_delta_net.cpp:254-261`：让 `T_full==0`（尾块 <64）与 `T_full>0` 使用**同一发布精度口径**。
- 性能优先的选择：**把尾块对齐到已发布口径**（而不是把主路升到 FP32）。
- 判据：非 128 对齐 prompt 的 `--prefill-chunk 128 vs 4096` 变为 IDENTICAL。

## 3. 验收套件（每次改动都跑）
1. `_ga_check.sh`（spec vs plain 逐位，zh + num）
2. `_align128_ab.sh`（chunk 不变量；**必须用非 128 对齐 prompt**，否则假阴性）
3. `_verify_df2head.sh`（接受率 + G-A）
4. 解码 tok/s（性能不许退化）
5. （新增）`--prefill-chunk` 回归探针纳入常规验收

## 4. 两项决策项（用户/需 GPU 窗口）
- **S_C 的混淆项**：同 ckpt 下 serve 4.55 tok/round vs CLI 1.35 tok/round ⇒ 先钉死再谈 retrain。
- **retrain**：`_train_df2_shift0.bat`（mask + shift0），**必须显式 `--out-dir`**（否则 `--resume` 取
  legacy 的 `step_001900`）；先用已有 `step_000100/000200` 测塌陷比 R（旧≈0.18，预测新 ≥0.6）。
