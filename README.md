<!-- 2026-09-30: this file is the CURRENT ENTRY POINT (a pointer page, no figures). The previous
     content -- the inherited upstream NInfer README plus its outdated banner -- was MOVED VERBATIM to
     docs/upstream-NInfer-README.md. Nothing was deleted. -->

# NInfer Fusion — 当前状态入口 / Current state entry point

这是一棵**工作树**（单卡 RTX 5090 D / sm_120a 上的 C++20/CUDA 推理引擎，Qwen 系列显式注册模型）。
**本页只做指路，不复述任何数字**；需要数字请去证据。

This is a **working tree**. This page only points; it carries **no figures**. Every reading lives
with its evidence.

## 从这里开始 / Start here

- **中文** — [`README_FEATURES.md`](README_FEATURES.md)：功能面 + **逐条出处**（`路径:行号`）、默认值表、
  已知限制、以及**未解决问题**（只写"是什么 + 证据在哪"）。
- **English** — [`README.en.md`](README.en.md)：mechanisms, defaults, and evidence pointers.

两份都在 **2026-09-30 重写**，并把**没有出处的数字整批撤下**（撤稿存档：
[`docs/features/withdrawn-numbers-2026-09-30.md`](docs/features/withdrawn-numbers-2026-09-30.md)）。

## 上游原始 README（已过时）/ Upstream README (outdated)

[`docs/upstream-NInfer-README.md`](docs/upstream-NInfer-README.md) —— 它描述的是**上游 NInfer 引擎**，
不是这棵树。其中的实测表是**上游的**读数。

## 本树的证据约定 / The evidence rule for this tree

**每个读数都带证据路径**：`dl/<line>/` 下有该线的 `EVIDENCE.txt`（完整命令 + 引擎原文 + 二进制
sha256）、`pin/`（钉住的配对副本）、`out/`（每臂 argv/rc/stdout/stderr）。**不在本页复述数字**，
是因为本页无法携带证据。

**Every reading carries its evidence**: `dl/<line>/` holds that line's `EVIDENCE.txt` (full commands,
raw engine output, binary sha256), `pin/` (the pinned pair), and `out/` (per-arm argv/rc/stdout/stderr).
Figures are deliberately absent here because this page cannot carry their evidence.

## 分支 / Branch

当前工作推在分支 **`sync/2026-09-30`** 上（本仓库的惯例：每次会话状态推一条 `sync/<date>` 分支，
不动 `main`）。
The current work is pushed on the branch **`sync/2026-09-30`** (this repository's convention:
one `sync/<date>` branch per session state; `main` is not touched).
