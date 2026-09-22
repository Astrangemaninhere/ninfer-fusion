#pragma once

// 历史占位文件。原内容把 identity 内联在这里，并且带着生成器吐出的 `{{` 双花括号
// 缺陷（非法 C++；S33 的 6 项"不存在"之一、_TODO.md 72.3 记录）。
//
// S37 阶段 (a) 起，identity 与几何交叉核验搬到引擎 include 路径上的导出头
// <ninfer/targets/qwen4_exp/package.h>；本文件保留为薄别名，免得旧的引用路径断掉。
//
// 位置说明：目标根不是族约定位置 —— 兄弟目标是
// src/targets/<id>/export/ninfer/targets/<id>/package.h（见 muse_glimmer_30b）。
// 真正的 Package（LoadPlan/LoadedModel/Frontend/SequencePlan/Program/...）是阶段 (b)
// 起的产物，见 _collab/B_s37_flashnext_p1.md 的分阶段计划。

#include <ninfer/targets/qwen4_exp/package.h>
