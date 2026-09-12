# R26：P7 对比（UNIFY-A vs UNIFY-B）+ 合并方案 + 单变量测量计划（2026-09-11 16:2X）

## 一、两路对比
| 维度 | UNIFY-A | UNIFY-B |
|---|---|---|
| 产物 | `UNIFY_A_patch.diff`（**17 文件 / 28 处 / 373 行**）+ `UNIFY_A_report.md` | `UNIFY_B_patch.diff`（7 文件 / 8 hunk）+ `UNIFY_B_report.md` |
| **统一轴** | **分派点（dispatcher）**：把 T=1 / T∈[2,4] 的路线搬到 verify 已在用的路上（14 处） | **累加器分链方式**：T=1 的 gemv 用 **4 条链**、T∈[2,16] 的 small_t 用 **1 条**，其余（lane→K 映射、scale 分组、`warp_reduce_sum`、epilogue）逐字相同 ⇒ **改 2 行即逐位同算术** |
| 是否覆盖 live(FP8) | **覆盖**（`fp8_gdn_conv_fused.cu`、`fp8_linear_add_plan.cpp`、`fp8_linear_swiglu_plan.cpp`、attn 输入 FP8…） | **未覆盖**：走 NVFP4 表，并把"Qwen38Nvfp4DFlash2 profile 用 FP8"列为**必须先裁决**的风险 |
| 关键架构事实 | **plain 与 verify 是同一段代码同一 `Phase::Verify`（`text_context_impl.h:810/816` vs `:867/873`），只差 T；T=8 在 16 个分派点上 route 全部不变** ⇒ 统一只把 T=1/T2..4 搬上 verify 已用的路 ⇒ **verify tok/s 按构造不退化** | —— |
| 性能代价 | 访存量级不变；**T=1 估 +1.5~3.5%**（新增小 kernel 节点：swiglu 量化 ~56 层×2µs、linear_add ~64 层、attn ~16 层、gating 多一 split-K reduce ×48 层）；verify 0。超 3% 按粒度回退（先 gating 后 swiglu） | T=1 每 lane 串行 FMA 40→160，估 **0~2%（仅 T=1）**；T∈[4,16] A4→A16 激活 +0.24% 访存但**少 1 次 quantize launch**（净正）；唯一可感代价是 gating proj（+2~10 µs/层，独立 hunk 可丢） |
| dry-run | exit 0、全部 hunk 无 offset（4 个 CRLF 文件保留 CR） | exit 0、7 文件全绿 |
| 未统一项 | **核心注意力**（27B 24Q：T≤6 SmallT / T∈[7,16] Prompt；两侧互换都要牺牲性能）⇒ 只统一到"同精度档"；给出零改码判别实验（draft 宽度 4 vs 8 比翻转率） | 同（core attention） |

## 二、我的独立核查（补上两路的缺口）
- **live 路径 = FP8**：`variant.cpp` 里 `Qwen38Nvfp4DFlash2` profile 的 attn 输入投影 / attention 输出 / GDN 输入投影
  全是 `QType::FP8_E4M3FN_ROW_BF16S` + `kFp8TextPolicy` ⇒ **B 的 nvfp4 分派器改动打不到 live 路径** ✓（B 自己标了这条风险，我实测确认）。
- **B 的"累加链"洞察在 FP8 上同样成立**（我定位到具体行）：
  - T=1 GEMV：`fp8_config.h:254,259,…,301` = `Fp8GemvSchedule<8, 2, 8, **4**, …>`（**10 处，累加链 4**）
  - T∈[2,16] SmallT：`fp8_config.h:429,440` = `Fp8SmallTSchedule<8, 2, kValuesPerLane, kTokenTile, **1**, …>`（累加链 1）
  ⇒ **已落地 10 处 4→1**（备份 `/home/user/fp8chain_bak/`），正在后台**单变量**编译+测量。

## 三、合并方案（P7：求其同 + 析其异后落地）
1. **先量"累加链对齐"单独的效果**（进行中）：判据 = `_ga_check` 的 spec-vs-plain 首次偏离是否**消失/显著后移**；
   代价 = **T=1（plain）tok/s** 与 zh decode tok/s。这是"统一算术"的**最小解验证**。
2. **再叠加 A 路的 373 行分派统一**（dry-run 已绿；其"verify route 不变"的结构论证使 verify 性能按构造安全）：
   判据同上 + W=8/16 接受率不退化；代价看 T=1 tok/s，**超 3% 按 A 给的粒度回退（先 gating 后 swiglu）**。
3. 二者都过 ⇒ 把两处改动合并提交（P7 的"最后落地"），并把 A 的 373 行里**与 live FP8 无关**的 NVFP4/bf16 部分
   单独标注（它们对不同 profile/变体生效，属兼容性面，不阻塞本次验收）。
4. 保留项：核心注意力只到"同精度档"；A 给的零改码判别（draft 宽度 4 vs 8）作为后续量化手段。

## 四、验收口径（定稿，见 R25 §四）
强判据 = 同精度档 + 首次偏离消失/后移；效率判据 = **接受率与 tok/s 不退化**；一致性判据 = 非对齐 prompt 的
`--prefill-chunk` 不变量（回归探针常备）。
