# 投机解码策略分析（2026-09-02 实测复核）

## 背景

用户复核速度曲线时提出四个问题：(1) 长文本混合 KV 表 prefill 为何变慢；
(2) "SVIP"（kSpecDemoteTokens=20480）为何没有约束住 DFlash2 的长文本降级；
(3) 报告的接受率是哪个位置的统计；(4) DFlash2 速度偏慢。

## 1. 混合表长文本变慢 —— 测量噪声，不成立

首轮单次测量显示 32K 时默认表（10L E8+6L NVFP4）1679 tok/s，明显低于全 E8
2031 / 全 NVFP4 1929 / 全 INT8 2048。逐点双跑复测后（相同配置、相同语料）：

| ctx=32K | run1 | run2 |
|---|---:|---:|
| 默认 10L E8+6L NVFP4 | 2319 | 2129 |
| 全 E8Kv | 2374 | 2477 |
| 全 NVFP4 | 2005 | 2056 |
| 全 INT8 | 2591 | 1812 |

- 同一配置两次测量波动 20-43%（INT8 的 2591→1812）；默认表与全 E8 的差异
  落在噪声带内。
- **结论：不存在"混合表长文本系统性变慢"。首轮曲线的 32K 凹陷是单次测量
  噪声（WSL2 调度/GPU 频率漂移）。**
- 修正：速度曲线改为逐点双跑并报告区间；图表已按 v3 数据重画。

## 2. "SVIP"（kSpecDemoteTokens=20480）与 DFlash2 降级

现状（spec_decision.h）：
- admission：projected context ≤ 20480 → DFlash2，否则 MTP；
- mid-flight：全部请求 frontier 越过阈值或 VRAM 压力 → 降级 MTP；
- promotion：仅引擎完全空闲时回 DFlash2。

实测（DFlash2 d7，中文 wiki prompt）：

| prompt | 接受率 | decode tok/s |
|---|---:|---:|
| 1.5K | 24.7% | 70.6 |
| 6K | 18.6% | 51.2 |
| 12K | 12.7% | 37.5 |

12K 时接受率已崩但仍在跑 DFlash2（12K < 20480）。同时 MTP3 在 12K 保持
45.6% / 83.2 tok/s —— **策略应基于接受率而非仅上下文长度**。

改进（待实施）：
1. **接受率监控降级**：每 ~256 decode 步统计 DFlash2 接受率，连续窗口低于
   阈值（默认 25%）→ 降级 MTP（复用现有 spec-degrade 机制）；
2. 阈值 kSpecDemoteTokens 保留为 admission 上限，新增
   kDFlash2MinAcceptance 运行时降级条件；
3. promotion 仍仅 idle 时（已有）。

## 3. 接受率的口径

当前报告的是**整次生成的累计接受率**：`accepted / drafted`（所有 round 的
总计数），不是"某个位置的接受率"。若要按 draft 位置分解（第 k 个 draft
token 的接受率），需要修改统计：按 draft 索引记录 accept/reject 计数。
该数据对调 draft 窗口长度有直接价值（若第 3 个 draft 接受率骤降，窗口
2-3 即最优）。列入 backlog。

## 4. DFlash2 为何慢

- 接受率低（12-24%）→ 每 round 实际收益 ~2.3 token，却要跑 8 位置验证；
- DFlash2 draft cache 为 BF16 全上下文（draft 序列越界时显存/带宽占用线性
  增长）；
- 中文 wiki 非 DFlash2 主场（其优势是代码/重复文本）。

改进（backlog，按收益排序）：
1. 接受率监控降级（见 #2）——直接止损；
2. Draft KV 降精度（FP8/int8）——减带宽/显存（外部调研已确认方向）；
3. draft 窗口自适应：接受率低时 draft 数从 7 降到 3-4；
4. 补充代码语料的 DFlash2 对照测量。
