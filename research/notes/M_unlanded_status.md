# 未落地项当前状态总表 (M 维护, 2026-09-10 12:3X)

状态口径: **已落地** = 代码/工具已在树上且有独立证据; **已备** = 补丁就绪待 window H 批处理编译;
**在跑** = 已派子代理本轮执行; **阻塞** = 有明确前置; **陈旧** = 清单条目与树中实际不符(已核实)。

## KV / 分配系列
| 项 | 状态 | 证据 / 位置 | 下一步 |
|---|---|---|---|
| N1 `--kv-bit-budget` 引擎入口 | **已备** | `_collab/A_n1_patch.diff`; 符号已核 (`kv_bit_budget.h:302`, `kv_options.h:43`, `TextConfig::full_attention_layers()` 在 `layouts_impl.h:138` 已在用) | window H 批处理 |
| N2 冷窗引擎开关 | **已备** | `A_n1b_cold_pages.diff`(显式上限+优先级) + `A_s30_budget_cold.diff`(预算×冷容量耦合, 顺序 N1→n1b→s30 已证明) | window H |
| N2 遗留: `ColdPolicy::Host` 死路径 | **未决** | 实证: `program_impl.h` 只对 Window/Disk 放行 ⇒ Host 恒 0 页 | 需决策: 接线 or 删策略 |
| N3 启动自测→固化→跳过→手动重校 | **在跑(S28)** | 旁车格式 + env 加载器 + 身份门 (拒绝外来表); host 12/12 PASS; 第一轮已交付 | 第二轮: 采集接线 (kv_calibration_dir 消费) |
| N4 热/温/冷自动动态分配 | **在跑(S34)** | 闭环设计 + dry-run 决策模式 | 验收依赖质量矩阵 (GPU) |
| N5 = W13 权重 host 卸载 | **在跑(S32)** | P0 范围已定 (§124): 驻留等级 + 取用钩子 + 预取策略 | 排在 L3/L2 之后实现 |
| N6 ngram 真表/前缀缓存 | **在跑(S33)** | 真表 gather + pinned LRU **已实现但是死代码**; 引擎已有序列级前缀复用 | 接线补丁已就绪 (`B_s33_ple_wiring.diff`); **但见下**: 消费者运行时是空桩 |
| N7 FlashNext bindings | **基本完成** | 契约 74,804/74,804, 缺失 0, 恒等式闭合 (M 独立复现 §append15); 写出器往返 12/12 逐字节 (S27) | 等分片到齐 → 全量转换 |
| §103 `sliding_window_tokens` | **已备** | `A_s24_window_table.diff` (与 `layer_residual` 同构; `sliding_window==0` 语义由代码回答) | window H + Muse 掉针验收 |

## W 系列
| 项 | 状态 | 说明 |
|---|---|---|
| W14 无损压缩 probe | **陈旧(已完成)** | §71: Muse 7.457% 可回收 (FP8 code 平面为主) |
| W16 EngineCore 503 | **主体已完成** | P0+P1 早在 `51a7a8f`/`487562a`/`a7ca4dd` 落地; D/S31 补 4 处残留 (含 `/health` 免鉴权线路脱敏); **S35 在做 §3.4 校验前移** |
| W17② ccache | **已备** | 本地已装 4.11.3; window H 与 L3 同批开启 (`.o` 依赖 flags.make ⇒ 必触发全量重编) |
| W2 PLE 真表 gather | **已完成(部分陈旧)** | §69 三方一致 PASS; 剩"接进引擎"= N6/S33 |
| W10 KV 组合矩阵全量重测 | **阻塞(GPU)** | e8 修复后需重测; 与接受率 eval 同窗口做 |
| W11 Muse HF logits 对齐 | **阻塞** | 需参考实现; §11 记 QAT repo URL 失传 |
| W1/W3/W4/W8/W13 | 未开工 | W13 已转 S32; W1(Muse nvfp4 TT4 smem)/W3(1Cat MTP 调度)/W4(QPN MT=2)/W8(rmsnorm 折叠) 待排 |
| W5/W6 FreeToken 步2/步3 | **在跑(S34 的邻域)** | 与 N4 同一闭环 |
| W15 Windows EXE 封装 | 未开工 | 需 studio GUI 验收 (用户侧) 后做 |
| W12/W17①③④ | 未开工 | 小项 |

## 模型导入
| 模型 | 状态 | 证据 |
|---|---|---|
| MiniCPM5-1B | **可导入** | 2 hook + 1 post, 无 new_op (§119) |
| LFM2-2.6B-Exp | **规格已定** | 需 no-SiLU/宽3 卷积变体 + 两道门 + 新层型 (§125); 中等工作量 |
| Falcon-H1R-7B | **new_op(大)** | 44 层 Mamba/SSD 未建模 |
| FlashNext (qwen4_exp) | **导入侧已备 / 运行时是空桩** | 契约全覆盖 + 写出器重构式 (往返 12/12 逐字节); PLE 表 FP8 待分片; **S33 实证: qwen4_exp 运行时只有 2 文件桩** ⇒ 还需 ① 真运行时 ② registry dispatch ③ `config.h` 的 `intermediate = None` 非编译 C++ 修复 ④ CLI PLE 字段 ⑤ EOS 载体 ⑥ FP8 PLE 表通路; MTP 27 张量待引擎设计 |

## 训练 / 接受率
| 项 | 状态 | 说明 |
|---|---|---|
| W9 dflash2 训练到 6000 | **在跑(window G 续训)** | 0.40 steps/s; 到达后 ETA ~4h |
| 接受率判定 | **已备** | `_df2_accept_eval.sh` (预注册闸门 hit@1≥0.35 且 headroom≥0.12) + CPU 路径冒烟中 |
| DDTree 链种子硬化 | **已落地** | `dflash2_tree.py` EAL(tree)≥EAL(chain) 成定理; 480 随机格 0 违反 |
