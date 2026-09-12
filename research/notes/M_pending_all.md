# 待落地事项总清单 (M, 2026-09-10 13:5X) —— 用户要求先全列出来

状态口径: **已备**=补丁/工具就绪待编译或待 GPU; **在跑**=已派子代理; **待做**=未开工;
**待决**=需人拍板; **阻塞**=有明确前置; **已完成**=今日已落地。

## A. KV / 分配 (N 系列, 用户八问的实体)
| # | 事项 | 状态 | 位置 / 前置 |
|---|---|---|---|
| A1 | N1 `--kv-bit-budget` 引擎入口 | 已备 | `_collab/A_n1_patch.diff`; window H |
| A2 | N2 冷窗开关 `--max-cold-pages` | 已备 | `A_n1b_cold_pages.diff` (优先级由代码定) |
| A3 | N2 预算×冷容量耦合 (自动热/温/冷) | 已备 | `A_s30_budget_cold.diff`; 顺序 N1→n1b→s30 |
| A4 | **N2 遗留: `ColdPolicy::Host` 是死路径** | **待决** | `program_impl.h` 只对 Window/Disk 放行 ⇒ Host 恒 0 页; 要么接线要么删策略 |
| A5 | N3 校准闭环: 侧车+加载器+身份门 | 已备 | `C_s28_rowscale_sidecar` (host 12/12) |
| A6 | N3 采集侧接线 | 在跑 | S38 (C); 已发现同步 cudaMemcpy 竞态 + 写盘阻塞 + 批模式错位 |
| A7 | N4 热/温/冷闭环策略 (dry-only) | 已备 | `C_s34_cold_loop/patch/…` (2400 决策 0 违反) |
| A8 | N4 live 接线 (接到 S30 的池消费者) | 待做 | 依赖 A3 |
| A9 | N5=W13 权重 host 卸载 P0 | 已备 | `A_s32_w13_p0.diff` (host 10/10, -Wall 干净) |
| A10 | W13 P1 (GEMM 热路径钩子) | 待做 | 依赖 A9 |
| A11 | N6 PLE/ngram 接线接缝 | 已备 | `B_s33_ple_wiring.diff` (DRYRUN_OK) |
| A12 | N6 消费者 (qwen4_exp 运行时) | 在跑 | S37 (B); 见 C13 |
| A13 | N7 FlashNext 契约+写出器 | **已完成** | 契约 74,804/74,804 缺失 0; 写出往返 12/12 逐字节 |
| A14 | N7 全量转换 (135GB → artifact) | 阻塞 | 分片下载中 (~99GB/135GB); FP8 PLE 分片正在到 |
| A15 | §103 逐层窗口表 (E3) | 已备 | `A_s24_window_table.diff`; 验收需 Muse 掉针 |
| A16 | U7 Muse page-fill e8 三缺陷 | 已备 | `M_muse_pagefill_patch.diff`; **修复前基线待跑** |

## B. 引擎鲁棒性
| # | 事项 | 状态 | 位置 |
|---|---|---|---|
| B1 | W16 残留四修 (含 `/health` 免鉴权脱敏) | 已备 | `D_s31_503_p0.diff`; P0+P1 早已落地 |
| B2 | W16 §3.4 校验前移 | 已备 | `D_s35_frontload.diff` + 静态闸门 `D_s35_gate.py` (双向证明) |
| B3 | W16 四道验收门 (req-1 失败 req-2 正常等) | 待做 | 需重建后的 GPU 窗口 |

## C. 模型导入
| # | 事项 | 状态 | 备注 |
|---|---|---|---|
| C1 | FlashNext 真运行时 (registry+config+稠密层) | 在跑 | S37 阶段 (a): 已列出 seam 清单与几何交叉校验方案 |
| C2 | FlashNext MoE 512 专家路由算子 | 待做 | S37 阶段 (c); **最大缺口** |
| C3 | FlashNext GDN 36 层 + hyper-connection 块 | 待做 | GDN 路径已在, hc 算子缺 |
| C4 | FlashNext PLE (FP8 表) | 阻塞 | FP8 子布局待分片到齐后钉死 |
| C5 | FlashNext MTP 27 张量 | 待做 | 引擎已有 MTP 算子与 dflash2/dspark 先例 |
| C6 | LFM2-2.6B-Exp 适配 | 规格已定 | §125: 需 no-SiLU/宽3 卷积变体 + 两道门 + 新层型 |
| C7 | Falcon-H1R-7B (44 层 Mamba/SSD) | 待做 | new_op 大件, 暂不排期 |
| C8 | MiniCPM5-1B | 可导入 | 2 hook + 1 post, 无 new_op |

## D. 构建效率
| # | 事项 | 状态 | 备注 |
|---|---|---|---|
| D1 | L3 按 KV 档拆巨 TU | 已备 | `C_s23_tu_split` + 无人值守 apply.sh |
| D2 | L2 ccache 入口 | 待做 | 与 D1 同批 (`.o` 依赖 flags.make ⇒ 必触发全量重编) |
| D3 | L1 已落地: 26GB 上限 + split-compile + -j1 + 重试 | **已完成** | 实测峰值 20.2GB 瞬时 / available ≥8GB |

## E. dflash2 / 接受率 (用户主张: 非训练问题)
| # | 事项 | 状态 | 备注 |
|---|---|---|---|
| E1 | 接受率机制判定 (chain/beam/tree + 跨 ckpt 对比) | 待做 | 评估通路已去风险; **需 GPU 窗口** (可先用 step_001200) |
| E2 | 跨 checkpoint 趋势 (200 vs 1200 vs 6000) = "是否训练问题"的判据 | 待做 | 若 hit@1 几乎不随步数变 ⇒ 支持用户主张 |
| E3 | DDTree 链种子硬化 (EAL(tree)≥EAL(chain) 成定理) | **已完成** | 480 随机格 0 违反 |
| E4 | beam/tree 的真实收益 (CPU 只有 builder coverage) | 待做 | 同 E1 窗口 |
| E5 | §112④"树材料被单链浪费"落地 (训练侧 `ddtree_topk` 或推理侧树验证) | 待做 | 依赖 E1 结论 |

## F. 训练 (用户: 最后做)
| # | 事项 | 状态 | 备注 |
|---|---|---|---|
| F1 | W9 训练到 6000 步 | 在跑(将暂停) | 当前 step 1200+/6200, 0.40 steps/s |
| F2 | 6000 步后的质量矩阵 / 接受率终判 | 待做 | 排在 E 之后 |

## G. W 系列其余 + 今日新发现
| # | 事项 | 状态 | 备注 |
|---|---|---|---|
| G1 | **Muse + nvfp4 产出 NaN** (decode 遍 L00 爆 1e6 → L01+ nan) | 在跑 | S36 (A) 代码级审计; 待 bf16 vs nvfp4 对照 |
| G2 | Muse 0-15 层套用 qwen 标定表 | 待做 | N3 身份门已能拒绝; 引擎当前行为仍待修 |
| G3 | e8 三档真数据 (此前从未真正跑过 e8) | 待做 | 脚本 bug 已修; 同 GPU 窗口 |
| G4 | W1 Muse nvfp4 TT4 smem 修复 | 待做 | §65 |
| G5 | W3 1Cat 负载 MTP 调度化 / W4 QPN MT=2 host 接线 | 待做 | 施工卡 §2/§4 |
| G6 | W8 rmsnorm 权重折叠 PPL 回写 | 待做 | `_opt_plan/_progress_fold2` |
| G7 | W10 KV 组合矩阵全量重测 | 待做 | e8 修复后必须重测 |
| G8 | W11 Muse HF logits 对齐 | 阻塞 | 参考实现/QAT repo URL 失传 |
| G9 | W12 4090/SM-count 借鉴 / W15 Windows EXE 封装 | 待做 | W15 依赖 studio GUI 验收 |
| G10 | W17 各项 (多流宿主管线 / q16 验证图变体 等) | 待做 | 小项 |
| G11 | `--kv-layer-storage` 未列入 serve 用法文本 | 待做 | 小文档缺口 |
| G12 | 另外两个解析点未覆盖 `--kv-bit-budget` (`apps/cli`, `apps/perplexity`) | 待做 | 不阻塞 serve 路径 |
