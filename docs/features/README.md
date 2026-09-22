# 功能分册索引（docs/features/）

本目录把 NInfer 这一轮的新增功能按族拆开写，每条功能声明都带 `路径:行号` 或实测数字。

**基准修订**：`3944a53eda1aac439a566a1cf46ea741f0415fdc`（`git rev-parse HEAD`）。
**工作区状态**：`git status --porcelain` 794 条（已 `A` 236、未跟踪 168）——这是 2026-09-20 的读数。
**2026-09-22 复核**（`accept34` 线，`git status --porcelain | wc -l`）：**817** 条，其中 `A` **236**（未变）、`??` **179**。
⇒ 本目录描述**这棵树**，
不是已发布版本；相当一部分新增功能只在未提交的工作区里。

## 分册

| 文件 | 内容 |
|---|---|
| [`cli-flags.md`](cli-flags.md) | `ninfer` 前端命令行全量表：66 个 case 分支 / 68 个入口，逐条给出 parser 行号、默认值出处、互斥规则行号 |
| [`serve-flags.md`](serve-flags.md) | `ninfer-serve` 前端旗标（与 CLI 分开写：两个前端的旗标集**不相等**） |
| [`kv-compression.md`](kv-compression.md) | KV 压缩与混合精度：档位、位预算求解器、逐层表、混合 store census、e8 宽度 |
| [`kv-tiering.md`](kv-tiering.md) | KV 温/冷分层、冷池 host/disk、卸载水位、行尺度校准闭环、页池与预分配 |
| [`mtp-and-speculation.md`](mtp-and-speculation.md) | MTP / DFlash / DFlash2 / DSpark 投机解码、自适应宽度、树验证 |
| [`importers.md`](importers.md) | 导入前门、编排器、artifact 逐对象 diff 闸、GGUF 族、ModelOpt、LoRA、各族转换器 |
| [`kernels.md`](kernels.md) | 内核族与后端：attention 各档位核、KV append/decode 核族、GEMM 路由、无 TC 回退、模拟器 |
| [`instrumentation.md`](instrumentation.md) | 仪器与诊断：运行期环境变量全表（含 `NINFER_FLASH_NEXT_*` 族） |
| [`verification.md`](verification.md) | 怎么验收与复现：测试面规模、复现命令、本文档的出处策略 |
| [`unfinished.md`](unfinished.md) | **未完成 / 部分实现 / 按名找不到** 逐条清单（这是本目录里必须读的一册） |

## 出处规则（写这些分册时遵守的）

1. 一条功能声明要么给 `路径:行号`，要么给实测数字并说明数字来自哪条命令。
2. 行号**必须在本基准修订上重新测**。上一轮的穷尽清单（记录目录 `dl/docinv/FEATURES.md`，**不是本仓库的一部分**）是原料，不是出处：
   它的行号已经漂了（例：它记 `apps/cli/options.cpp` 1001 行，实测 1016 行；它记 `--kv-dtype`
   在 `:363`，实测在 `:378`；它记 `maximum_mtp_draft_tokens` 在 `impl/config.h`，实测在 `impl/variant.h`）。
3. **没有出处的条目不许进功能正文**，按名留在 `unfinished.md`。
4. 不把「没验证的」写成「已验证」。带能力限制的条目（骨架、无调用路径、具名拒绝、临时补丁）
   一律按「部分」写，并写清限制是什么。

## 本目录与既有文档的关系

- 既有 `docs/cli.md`（519 行）、`docs/serving.md`（980 行）、`docs/maintainer/` 与 `docs/gfx906/`
  是**上一阶段**的文档，本目录**不改**它们，只在需要时引用。
- `CHANGELOG.md:11-19` 自述有一个「documentation completeness increment」，基准修订与本目录相同
  （`CHANGELOG.md:20`）。本目录是**功能面**的补充，两者可互相对照。
- 本目录**不覆盖任何既有文件**：它落在新建的 `docs/features/` 下，另加顶层 `README_FEATURES.md`。

## 关于 `dl/` 这一族路径

正文里少数地方会出现 `dl/ghdocs/*.txt` 或 `dl/docinv/FEATURES.md` 这类**记录目录**路径。它们**不是本仓库的一部分**，是测量批次留下的记录文件。本目录不把它们当出处：凡「实测数字」都在 [`verification.md`](verification.md) 第 4 节给出**可复跑的命令**，记录文件只是同一批命令的留存副本。
