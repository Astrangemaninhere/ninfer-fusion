#!/usr/bin/env python3
"""18:00 汇报正文 -> _report_1800.md"""
import pathlib

body = """# ninfer-fusion 进度汇报（2026-09-10 18:00）

> 本页由 18:00 调度自动落盘；数据全部来自只读采集（未启动构建/训练/杀进程）。
> 铁律：① 长任务不空等，等就并行推另一项；② 杀进程/端口先查 cmdline；③ 状态同步 `_TODO.md`；
> ④ 数据/数值/布局问题一律实证，不纸上推测；⑤ 不要等任何东西，等的时候就思考并行。

## 0. 一句话
**补丁 A 落地并实测生效**（DFlash2 的"gen=2 立即停止"消失），但**接受率仍远低于目标 0.9**；
位置剖面首次给出直接答案：**第一个位置的接受率就只有 27%~54%（MTP 也不到 0.9）**；
同时发现"投机输出 ≠ plain 输出"在 **MTP 与 DFlash2 上同时成立** ⇒ 问题在 verify 路径本身，不在草稿。
另外，今天花了很大力气先**把构建修绿**——不是编译慢，而是树里累了一批**从未被编译验证过的半落地补丁**。

## 1. 头条：DFlash2 根因（已修）与实测
### 1.1 根因（E1 定位、我复核、已落地）
`program_impl.h` 的 DFlash2 decode ingress 漏填 `state_source_slots`/`state_destination_slots`
（宿主 ingress 零初始化 ⇒ 恒为 0）。这两个是**设备侧 state-image slot**，verify 从 slot 0 读 GDN 循环状态、
continuation hidden 也写回 slot 0；兄弟后端 ordinary/MTP/DFlash-v1 都填了。
⇒ verify 第 0 列 logits 错 ⇒ 错的 argmax ⇒ a=0 时唯一发布的 token 就是它 ⇒ 立即 stop。
补丁在 `program_impl.h:12411-12413`，diff `_collab/M_patchA_dflash2_state_slots.diff`。

### 1.2 实测（新二进制，`--no-thinking`，同一 prompt/温度）
| 后端 | prompt | gen | 接受率 | rounds | fallback | decode |
|---|---|---|---|---|---|---|
| **dflash2** | zh | 83 | 6/133 = **4.51%** | 19 | 57 | 17.6 tok/s |
| **dflash2** | num | 76 | 54/154 = **35.06%** | 22 | 0 | 63.4 tok/s |
| mtp3 | zh | 96 | 48/137 = 35.04% | 46 | 1 | 49.3 tok/s |
| mtp3 | num | 88 | 55/96 = 57.29% | 32 | 0 | 78.3 tok/s |
| plain | zh / num | 96 / 88 | — | — | — | 30.4 / 32.7 tok/s |

**已解决**：修复前 dflash2 在同一 prompt 上 `gen=2` 立即停；现在 83/76（plain 对照 96/88）。
**未解决**：接受率距 0.9 很远；dflash2 在自然语言 prompt 上 19 轮里 57 个 fallback step。

### 1.3 位置剖面（本轮最有价值的数据，直接回答"第一个位置"）
CLI 三臂（`accepted by pos`，语义＝每轮对每个被接受位置 +1）：

| 后端 | rounds | drafted | 接受率 | p(pos0) | p(pos1\\|pos0) | p(pos2\\|pos1) | p(pos3\\|pos2) | 4+ |
|---|---|---|---|---|---|---|---|---|
| mtp3 | 43 | 129 | 33.33% | **23/43 = 53.5%** | 12/23 = 52.2% | 8/12 = 66.7% | — | — |
| dspark | 56 | 196 | 11.22% | **15/56 = 26.8%** | 5/15 = 33.3% | 2/5 = 40% | 0/2 | 0 |
| dflash2 | 55 | 379 | 10.03% | **23/55 = 41.8%** | 11/23 = 47.8% | 3/11 = 27.3% | 1/3 | **0** |

⇒ **第一个位置的接受率只有 26.8%~53.5%，不是 0.9**（MTP 也一样）。这条把"低接受率"从"dflash2 的问题"
升级为"**草稿预测质量/训练对齐**层面的问题"，与"机制有问题"的直觉部分吻合：机制至少有两层，
补丁 A 修掉的是第一层（状态错位），第二层是草稿本身。

### 1.4 同批新发现（重要）
1. **投机输出 ≠ plain 输出**：exactness 检验下 `dflash2 zh` 第 32 字符分歧、`mtp3 zh` 第 42 字符、
   `dflash2 num` 第 284 字符分歧，**只有 `mtp3 num` 逐字相同**。两个不同草稿后端在同一条 verify 路径上都偏离
   ⇒ 分歧在 `target_verify_batch` 与普通解码**不等价**（非草稿质量问题）。
2. 已排除"verify 漏 `apply_final_logit_policy`"：该 policy 对 qwen3 家族是**编译期 no-op**（softcap=0、multiplier=1）。
   **但 Muse 声明了 softcap=20 + multiplier=0.196 ⇒ Muse 的投机 verify 会漏 policy，是真 bug**（列下一趟）。
3. **方法论修正**：测量脚本最初没传 `--no-thinking` ⇒ 回答全进 `reasoning_content`、`content` 为空、
   两侧空文本会被 exactness 判成 **IDENTICAL**（用"空对空"证明"verifier 精确"）。已修，并加防呆：
   **任一侧为空 ⇒ INVALID，不算证据**。空结果永远不是一致性证据。

## 2. 本 build 的补丁清单（一次编译全部纳入）
| 补丁 | 内容 | 状态 |
|---|---|---|
| 补丁 A | DFlash2 ingress state slots | 已落地+实测生效 |
| E2/S44 | 接受率计数标签（`spec_drafted/accepted/accept_rate/rounds/fallback_steps`） | 已落地，本轮数据就靠它 |
| E4/S46 | 冷窗 F3b（空闲窗不进 EWMA + 置信度门） | 已落地 |
| S45d/E3 | prefill 128 几何逐臂守卫（256 下逐字节等价，0 删/+31 行） | 已落地 |
| S48/E6 | dspark verify 位置表 k→k+1 | 已落地 |
| S50/E7 | KV 覆盖改下界语义（借用上游 `03177b9`） | 已落地（修掉 dspark 终止结算跨页抛错） |
| S51/E8 | `ops::silu/sigmoid` 极端负值归零修复 | **已就绪未落**（改全局数值，避免污染补丁 A 归因） |
| S52/E9 | dflash2 可配置草稿宽度 K=1..15 最小切片 | **已就绪未落**（改 dflash2 行为） |

## 3. 构建：绿灯，但过程揭示一个系统性问题
- 树里有一批**从未被编译验证过的半落地补丁**：S35（`engine_core.h` 赋值给不可赋值类型）、
  S28（rowscale：缺头声明 + `.cu` 未登记 CMake ⇒ 符号从未编译/链接）、S24（`layouts_impl.h` 的
  `requires` 守卫在非模板上下文无效 + designator 顺序）、N3/S38（`text_context_impl.h` 把函数定义插进了函数体内）。
- S55 只修 3 处就修绿：补 `#include "product/kv_options.h"`、交换 designator 顺序、把 kvcalib 块搬出函数体。
  证据：`ninfer` 17:42:33 / `ninfer-serve` 17:44:00，三个 variant `.o` 均晚于 `program_impl.h` 16:31
  ⇒ **补丁 A 真进了对象**；两个 make 目标 rc=0 且重跑 no-op。
- 构建取证（长期有效）：本树**没有头文件依赖跟踪**（改头必须 touch 源）、`make ninfer` **不产 `ninfer-serve`**、
  ccache **从未真正缓存**（0 文件；6/6 未缓存原因是"编译失败"）。

## 4. 必须按更正版记录的两处
1. **我早前说"E3 的 S45c 守卫在 256 下静默跳过 ISO3/FP8"是错的**：括号深度追踪证明两条臂在守卫**内部**、
   256 下可达。S45c 真正缺口只是 128 下缺响亮抛错，S45d 已补上。
2. **E6 纠正 E5**：被污染的第 k 列只在"全部草稿被接受"（a == extent == k）时起作用
   ⇒ dspark 的位置 off-by-one **不是** 10.3% 的成因，而是高接受率恢复后的正确性前置条件。

## 5. 上游借鉴（今天新增）
- 上游 = `github.com/Neroued/ninfer`（默认分支 **master**）；我们 fork 76 commits，**上游近期修复一条都没有**。
  今天克隆到 `ninfer-upstream`（Windows 侧 + 代理；WSL 连不上代理），并拉了 PR #226/#225/#194/#195 的 ref。
- 清单与可行性矩阵：`_collab/M_upstream_borrow.md`。两条 P0：
  - `03177b9` KV 覆盖改**下界语义** + 终止结算顺序（我们树原本是旧严格版）→ **已借并落地（S50）**；
  - PR #194 nvfp4 W4A4 的**近似 SiLU 在 x < −88.72 被 `__fdividef` 归零**，而真值仍是 bf16 正规数（22× min-normal）
    → **已适配待落（S51）**，另附穷举证据（救回 1,145,241 / 回归 0 / 正半轴 1.11e9 点逐位相同）。
- 子代理：**E7(S50) / E8(S51) / E9(S52) 均已完成**；E9 顺手纠正我方两处前提
  （d1/d3/d7 那组速度数字其实是 dspark 的 sweep，dflash2 的宽度影响仍是**待证实**）。

## 6. 工作区善后（已执行）
- 13 个 `*.orig/*.bak/*.rej` 隔离到 `/home/user/ninfer-fusion/_orig_quarantine/`（含 MANIFEST，可 `mv` 回）。
- **628 个无引用的一次性脚本**移到 `_scratch/<日期>/`：根目录 `.sh` 414→73、`.py` 316→54；
  索引 `_scratch/README.md`（按主题分组）、白名单 `_scratch/WHITELIST.md`；
  活管线脚本（`_window_k4.sh`/`_post_build_measure.sh`/`_spec_4way.sh`/`_muse_serve_accept.sh`/
  `_train_df2_resume.bat` 等）逐条核对仍在原位。
- 内存纪律：曾跌到 MemAvailable=93MB（并发 nvcc + 导出 python），已串行化"同一时刻只有一个 nvcc 或一个模型"；
  post-build 看门狗在 <700MB 时自动杀 serve。

## 7. 导入器 / GUI / 下载（并行线）
- **Spark-X2.5（XHToken，Apache-2.0）导入**：元数据 → 规格 → 缺口报告跑通；导入器**两轮修正**（新增
  `cpp_ident` 命名空间归一化、几何注册表解析器、逐头门控与 gated-GELU 由 hook 改判 **new_op**、`qk_norm=absent` 探测器、
  spec 落盘）。当前诚实结论：**3 项 new_op 阻塞**（16Q/4KV@256 几何未注册、逐头输出门、gated GELU MLP），
  tied head 改判 `covered`（转换期物化 `lm_head=embed^T`，约 671 MB）。
- **S53（转换器侧 tied head + 嵌入别名）**：新增 `tools/convert/common/source_map.py` + 自检；
  **我复核：61 项检查 0 失败**；Spark 命中 `model.embedding.weight`、朝向 identity、全量物化成功；
  对照 Qwen3.8-Flash-Next 判定 not-tied 且 rebind 后 recipe 逐元素相等。
- **S54（Spark 接入计划）**：只读分析给出逐文件清单、7 个钩子的接入点与 14 项风险；
  我的三条裁决已写入其 §7：id 用 `spark_x2_5_4b`、首版权重档 **BF16**、SWA 首版**只支持 ≤512 上下文**。
- **GUI 接线 + 界面双语**：`serve_gui.py` 已消费引擎注册表（**58 旗标**渲染成 58 个控件，默认命令与改造前**逐字节相同**，
  含 20000 次随机滑块对拍）；五个模块（serve/convert/rag/import/tips）双语完成，
  全量检查器 **PASS**（164 源码键 / 323 表项 / zh-en 等量），`serve_gui_selftest.py` **29 项断言全过**。
  未验证项照实记录：真实浏览器点击、`/api/start` 实调（需引擎）。
- **下载**：FlashNext = **126 GB 全量完成**；Spark 权重 **7.1G/8.2G**（分片 4 下载中，速率波动 0.6~2.4 MB/s）。

## 8. 未落地清单（逐条：状态 / 下一步 / 依据）
| # | 项 | 状态 | 下一步 | 依据 |
|---|---|---|---|---|
| 1 | S51 极端负值归零修复 | 已就绪未落 | 下一趟编译；**必须同时补极端负值回归用例**（现有测试 gate 只到 ±12，证明不了） | `E8_s51_nvfp4_silu.md` |
| 2 | S52 dflash2 可配置 K | 已就绪未落 | 单独一轮（改 dflash2 行为） | `E9_s52_dflash2_k_slice.md` |
| 3 | S28 rowscale 补完 | 半落地，调用已摘 | 补头声明 + 登记 `.cu` 进 CMake + 首次编译 | `_TODO.md` §123 |
| 4 | S24 窗口接线重构 | 已回退 | 换掉非依赖上下文里的 `requires` 守卫（或按 target 显式声明） | S55 记录 |
| 5 | Muse verify 漏 logit policy | 新发现 | verify 侧补 policy（Muse 有 softcap 20 + multiplier 0.196） | `apply_final_logit_policy` 仅 3 处调用 |
| 6 | verify 与 plain 不等价 | 新发现、已排除 policy 嫌疑 | 判别实验：plain vs "零接受投机"，隔离 verify 路径本身 | §1.4 |
| 7 | dspark 位置 off-by-one 的验证 | S48 已落 | 跑 `--spec dflash --draft-tokens 1` 复测位置剖面 | `E6_s48_*.md` |
| 8 | 32-needle 长上下文 | 已跑但噪声大（基线自己 16K 13/32 vs 32K 16/32；dflash2 20/32 > 基线） | 换更可靠的质量度量（如 PPL/自洽性），或提高 needle 数并多次取样 | `dl/window_k4.log` |
| 9 | 草稿训练 `--target-shift` 假设 | 探针三臂都低（own 0.124 / next 0.162 / legacy 0.229） | 用位置剖面 + 训练侧对齐复核再定 | `dl/shift3_probe.log` |
| 10 | FlashNext FP8 PLE 子布局 | 权重已全量，子布局待钉 | 按 `_TODO.md` 的 PLE 条目继续 | 权重 126GB 已到位 |
| 11 | E3 的 i8 平面步长 + S36 恢复 | 未落（256 下逐字节等价） | 与 Muse 128 工作一起落 | `E3_s45_*.md` |
| 12 | E1 补丁 B（`attention_valid` 契约） | 未落（extent=7=k 不触发） | 边界场景一起落 | `E_s43_df2_verify.md` |
| 13 | 训练续训 | 未跑（队列最后） | window K4 第 5 步后自动起 | `_train_df2_resume.bat` |

## 9. 状态与下一步
- **正在运行**：window K4 第 5 步（Muse 验收）→ 之后自动续训；Spark 分片 4 下载中。
- **下一趟编译队列**：S51 + S52 + S28 补完 + Muse policy + E7 回归测试（`E7_s50_regression_sketch.diff`）。
- **最该先做的一件事**：把 §1.4 的"verify vs plain 不等价"做成判别实验——它同时决定"接受率能不能到 0.9"
  与"Muse 上投机是否可信"。
"""

p = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_report_1800.md")
p.write_text(body, encoding="utf-8")
print("wrote %s (%d chars, %d lines)" % (p.name, len(body), body.count("\n")))
