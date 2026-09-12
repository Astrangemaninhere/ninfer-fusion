# M · 未落地 / 待验证 总清单（2026-09-10 19:45 快照）

> 铁律：① 长任务不空等；② 杀进程先查 cmdline；③ 状态同步 `_TODO.md`；④ 数值/布局实证，
> 不纸上推测；⑤ 不等，等就并行。
> 本文件是"已发现但未落地 / 已落地未验证"的唯一索引；每项都留了依据文件与判据。

---

## A. 就绪待落（diff 已 `patch -p1 --dry-run` rc=0，可直接应用）

**两个批次已武装（无人值守，串行，互不争 GPU/内存）：**

- **批次 1** `_batch_build_next.sh`：等 `_post_fix_verify2.sh` 跑完 → 落 **A5b + S52**（都只改现有文件，
  **不重新配置** ⇒ 不触发全量重编）→ 重链 → 自动跑 A5b 自身对照 → `_collab/M_a5b_verify.md`。
  基线二进制 `/home/user/ninfer_before_a5b`（= 列偏移单独修复版）。
- **批次 2** `_batch_build_reconfig.sh`：等批次 1 完 → 落 **A3 三份 + E7 测试** → `cmake` 重新配置
  （A3 的 gelu_mul 带 6 个新文件，这一步躲不掉；既然要付全量重编的钱，就把所有需要新文件的改动一次塞满）
  → 重链 → `ctest -R 'gelu_mul|sigmoid_mul|prepare_masked_block|context_store'` 全 PASS 才算过
  → `_collab/M_sparkops_verify.md`。

> A3 实测拆分：`head_geometry`（8 个现有文件，0 新增）与 `headwise_gate`（6 个现有文件，0 新增）
> 本可进批次 1，但为保住 A5b 的干净归因（geometry 改动会碰到 gqa attention 的公共几何注册表，
> 可能污染 dspark 数值），它们与 `gelu_mul` 一起进批次 2 —— 反正重配置是全量重编，搭车零成本。

| # | 项 | 依据 | 规模 | 落地判据 |
|---|---|---|---|---|
| A1 | **S52**：dflash2 草稿宽度 K 可配（K ∈ {1,3,7}最小切片） | `_collab/E9_s52_dflash2_k_slice.{md,diff}` | 6 文件 +44/-32 | 批次 1；编译过 + `--draft-tokens 3` 能起 |
| A2 | **E7/S50 回归用例**：store 级页边界 | `_collab/E7_s50_regression_sketch.diff` | +36 行 | 批次 2；ctest PASS |
| A3 | **A3 Spark 三个 new_op**：head_geometry / headwise_gate / gelu_mul | `_collab/A3_*.diff` | 8+6+8 文件，+677 | 批次 2；ctest PASS |
| A4 | **S51 极端负值回归用例**（代码已落，测试**未写**；须覆盖 −88.7 / −100 / −1000，相对判据） | `_collab/E8_s51_nvfp4_silu.md` §6 | 小 | 未写 ⇒ 无判据 |
| A5 | **A5b（配对未落地）**：dflash_impl.h 的 bf16 分支不再把 `attention_valid` 降回 `k` | `_collab/A5b_attention_valid_width.diff` | −7 行 | 批次 1 |

> **A5 注**：A5 的 diff 是**一对**：(i) `source_column_offset = 1` 无条件（**我已于 09-10 落地**）；
> (ii) 删掉 bf16 的 `k` 恢复（**未落地**，实证：`grep -n set_i32_scalar dflash_impl.h` → 第 240 行仍在）。
> `_collab/A5_block_rows.diff` 现已**被取代**：其 hunk 1 = 本表的 A5，hunk 2 与我已落地的修复重复
> （dry-run：hunk 2 FAILED = 预期），**不要再打这个文件**。
>
> **A5b 的两条独立证据**（已备好补丁 `_collab/A5b_attention_valid_width.diff`，dry-run rc=0）：
> 1. `src/ops/kernel/prepare_masked_block.cuh` 的 kernel **只读** `valid_columns`
>    （`positions[offset] = lengths[b] + min(i, valid - 1)`），**从不写它** ⇒ 那张表纯粹是给
>    注意力用的。把它从 `width` 降回 `k`，只意味着**列 k 看不到自己的 KV 槽**。
> 2. 姊妹实现 `dflash2_impl.h:190-192` 正是「建在 width、**不降回 k**」，且显式 `(void)valid_columns;`；
>    同一个文件的取列处还内联写了偏移 1 并注明「DFlash2 predicts the seven masked columns (1..7);
>    the bonus token stays at column 0」（`dflash2_impl.h:311-317`）。我落地的 (i) 正是让
>    `dflash_impl.h` 的 bf16 分支对齐这个有注释、有断言的实现。
>
> **纪律：不叠加。** 三路对照（`_post_fix_verify2.sh`）测的是"(i) 单独"；A5b 已排进
> `_batch_build_next.sh`，以本轮构建产物为它自己的 A/B 基线（`/home/user/ninfer_before_a5b`），
> 对照结论写 `_collab/M_a5b_verify.md`。

---

## B. 已落地、待验证（等编译 / 等 GPU 窗口）

| # | 项 | 落地位置 | 判据（必须实测） |
|---|---|---|---|
| B1 | **四处偏移/约定修复的前后对照**（dspark 列偏移、dflash2 目标行 M1、dflash2 输入模式 M2、MTP AR 掩码 off-by-one） | `dflash_impl.h:448-455`、`mtp_round.cuh:47-50`、`train_dflash2.py` 三处 | `_post_fix_verify.sh` 三路：dspark accept/p0 **上升**；mtp token-id **IDENTICAL**；dflash2 **不变**。链已武装，等 `无 nvcc 且 rebuild done` |
| B2 | **Muse SWA wiring 回归**（我回退 layouts 时误删 52 层中 39 层的窗口 2048，已恢复） | `layouts_impl.h` | Muse bf16 `nan_lines=0` 且 L46+ 不再 NaN |
| B3 | **Muse 验收复测** | — | bf16 / i8 两档出文本；nvfp4 被守卫拒绝 = 设计（需确认是干净报错） |
| B4 | **S51 在真实 artifact 上是否触发** | `silu` 极端负值路径 | E8 的一行 `__any_sync` 计数实验：计数 > 0 才算真踩到 |
| B5 | **dflash2 重训**（M1+M2 的收益只能靠重训兑现） | `_train_df2_shift0.bat`（`--target-shift 0` + mask 走新默认） | 约 2.1 h 到 1900 步；**等用户放行**（"训练暂时放下"） |

---

## C. 未开工（能力缺口，按"最便宜先做"排序）

| # | 项 | 阻塞点 | 备注 |
|---|---|---|---|
| C1 | **MiniCPM5-1B 导入** | 2 hook + 1 post，**无 new_op** | 清单里最便宜的一项 |
| C2 | **Spark-X2.5 接入** | 3 个 new_op（=A3）+ target 骨架（config.h 已生成） | A3 落地后即可通 |
| C3 | **LFM2-2.6B-Exp** | no-SiLU/宽 3 变体 + 两道门 + 新层型 + 32q/8kv@64 几何 | 规格已定 |
| C4 | **Falcon-H1R-7B** | **同层并行** attn+SSM（44 层）+ 12q/2kv@128 | 导入器已给出缺口 |
| C5 | **FlashNext (qwen4_exp)** | 运行时 registry+config（stage a 补丁未应用）、MoE 512 专家路由算子、GDN/hyper-connection 层、MTP 27 张量、FP8 PLE 子布局 | 权重 126 GB 已全量在手——**资产最重、最贵** |
| C6 | **N2 冷槽代价模型**（9536 B/slot 并入同一 DP） | 未写 | 与 N3 配套 |
| C7 | **N3 运行时校准闭环** `--recalibrate` | 侧车+加载器已备待验 | 未接 CLI |
| C8 | **N4 抖动 >3600 s/h 的节奏/滞后** | F3b 已修（冷启动），滞后量未调 | |
| C9 | **N5/W13 P1 GEMM 卸载钩子** | P0 已应用待验 | P1 未写 |
| C10 | **N6 ngram 真表 gather + 前缀缓存**（含 GDN 循环性） | 未接线 | |
| C11 | **长上下文质量度量** | 32-needle 噪声太大（基线 13/32 vs 16/32 ⇒ 不能作判据） | 需换度量：PPL / 自洽性 |
| C12 | **Muse 判别实验**（同 prompt 重复 vs prompt±1） | 未跑 | 用于区分"分歧"来源 |
| C13 | **GUI 三合一 + LABD/ngram-SSD 控件 + 桌面启动器** | W 系列，未开工 | 导入器/GUI 翻译已 PASS |

---

## D. 待决 / 阻塞（设计未定，不能盲落）

| # | 项 | 现状 | 卡在哪 |
|---|---|---|---|
| D1 | **`ColdPolicy::Host` 死路径** | 实证恒 0 项命中 | 二选一：接线 还是 删除 |
| D2 | **verify ≠ plain 的数值差** | 已定位：只在近似并列处翻转，**与草稿无关**；宽度判别实验已证：d1/d3/d5 三者**同 token 36、同错值 133222** ⇒ 与 verify 宽度 T 无关的固定差异 | 修法未定：需要 verify head probe 才能定位数值来源 |
| D3 | **E3 i8 平面步长 + S36 恢复** | 256 下逐字节等价未验 | 与 Muse 128 工作同批 |
| D4 | **E1 补丁 B**：`dflash2_impl.h` 的 `attention_valid` 契约 | extent=7=k 时不触发 | 需先定契约语义 |

---

## E. 工程债（构建，已坑过多次）

| # | 项 | 实证 |
|---|---|---|
| E1 | **ccache 从未生效** | 0 个缓存文件；6/6 未缓存原因 = 编译失败，1 次 = 边改边编 |
| E2 | **头文件依赖未跟踪** | 只 touch 源文件；今天已坑两次（改头不改源 ⇒ 不重编） |
| E3 | **`make ninfer` 不产 `ninfer-serve`** | 必须单独 make，否则 serve 是旧的 |
| E4 | **构建前必须 `export PATH=/home/user/.local/bin:$PATH`** | 否则 `ccache: not found` → Error 127 |
| E5 | **`--kv-bit-budget` 未在两个 apps 解析**（G12）；`--kv-layer-storage` 未进 usage（G11） | 小修，可顺手做 |
| E6 | **112 个源文件是 CRLF**（`src/targets/.../dflash2_impl.h`、`mtp_impl.h`、`layouts_impl.h`、`engine.cpp`、`registry.*`…），其余是 LF。手写的 LF 补丁打不上这类文件（实测：给 S52 做了 `tr -d '\r'` 归一化 ⇒ 3/3 hunk FAILED "different line endings"）。**规矩：对 112 个 CRLF 文件用字节级生成的补丁（`E9_s52_mkpatch.py` 那种 `read_bytes`+`diff -u`），或先 dry-run 再决定要不要 strip。** | 实测 |
| E7 | **加新文件会触发 cmake 重新配置 ⇒ 全量重编（今天 2h 的由来）**。所以新测试文件（E7/A4）不进常规批次，单独排。 | 实测 |
